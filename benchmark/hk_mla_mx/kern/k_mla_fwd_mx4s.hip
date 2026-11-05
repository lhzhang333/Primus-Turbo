/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// k_mla_fwd_mx (rung m4) -- m3's slot pipeline (WMMAs of QK(q_{i+1}) / PV(q_{i-1}) interleaved with softmax(q_i) by
// sched_barrier fences) with a VALU-light softmax. Operand recipe "m4" (k_mxq modes 16 + 32 for Q, 16 for K):
//  - Q8 is quantized from q * scale * log2(e): S' = Q8 . K8 is the centred logit in log2 units, no multiply;
//  - the nope WMMA's accumulator starts at Mt = -m (per q column): S'' = S' - m comes out of the WMMA, no fma;
//  - rope: a second MX WMMA over d 64..191 (a [nope 16 | rope 8] 24-VGPR fragment read as two overlapping
//    operands); scale layout R2 ([s0..s3, 0, 0, s4, s5]) puts E8M0 0 = 2^-127 on d 64..127, so they add nothing:
//    no zero half, 8 VGPRs less per K fragment;
//  - the row sum is one more PV WMMA against an all-ones V^T tile: the sum of the quantized P (the values PV takes);
//  - lazy rescale in a wave-uniform branch: when a lane's tile max lm (of S'') exceeds TAU2, or on the first tile,
//    every lane takes d = max(lm, 0) (first tile: lm): S'' -= d, O and the row sum *= 2^-d (first tile: 0), m += d,
//    Mt -= d. Without a rescale the softmax is the masked max (max3), the exp2 and one pk8 conversion per 8 values.
// Per 128-kv tile and wave: 4 x (16 QK + 8 PV + 1 sum) = 100 WMMAs. Q fragments double-buffered: slot i's QK reads
// qf[(i + 1) % 2] while the slot's first step loads q_{i+2} into qf[i % 2] (its last reader ran a slot earlier), so the
// load does not wait for in-flight WMMAs that read its registers.
#include "mx_common.h"

using namespace mx;

namespace {
constexpr int D_QK = 192, D_V = 128;
constexpr int BM = 256;
constexpr int QT = 4;
constexpr int NB = 128;
constexpr int NKV = NB / 16;
constexpr int NDT = D_V / 16;
constexpr int XK = 208, XV = 144;
constexpr int K_B = NB * XK;
constexpr int KS_B = NB * 8;
constexpr int V_B = D_V * XV;
constexpr int VS_B = D_V * 4;
constexpr int STG_B = K_B + KS_B + V_B + VS_B;
constexpr int NSTG = 4;
constexpr int Q_OFF = NSTG * STG_B;
constexpr int QS_OFF = Q_OFF + BM * XK;
constexpr int LDS_B = QS_OFF + BM * 8;
static_assert(LDS_B <= 327680, "LDS");
constexpr unsigned W0_KN = tdm_w0(0, 32, 20), W0_KR = tdm_w0(0, 16, 36), W0_V = tdm_w0(0, 32, 4), W0_S = tdm_w0(0, 0, 0);
#ifndef TAU2_X100
#define TAU2_X100 175
#endif
constexpr float TAU2 = TAU2_X100 / 100.0f;  // p <= 2^1.75 -> P = E4M3(p 2^7) <= 431 < 448
constexpr int PSCALE = 120 * 0x01010101;   // P = E4M3(p * 2^7)
constexpr int SC_ONE = 127 * 0x01010101;   // E8M0 2^0
constexpr int ONES = 0x38383838;           // E4M3 1.0 x 4

typedef int32_t i32x32 __attribute__((ext_vector_type(32)));

struct F24 {
    i32x16 n;
    i32x8 r;
};
struct KFrag {
    F24 x[NKV];  // [nope 16 | rope 8] of this lane's kv row half
    int sc[NKV];
};
struct VFrag {
    i32x16 t[NDT];
    int sc[NDT / 2];
};
struct QFrag {
    F24 x;
    int sc;
};
struct SmState {
    float pm[4];
};

MX_INLINE F24 frag24(u32x4 a0, u32x4 a1, u32x4 a2, u32x4 a3, u32x4 r0, u32x4 r1) {
    return {i32x16{(int)a0[0], (int)a0[1], (int)a0[2], (int)a0[3], (int)a1[0], (int)a1[1], (int)a1[2], (int)a1[3],
                   (int)a2[0], (int)a2[1], (int)a2[2], (int)a2[3], (int)a3[0], (int)a3[1], (int)a3[2], (int)a3[3]},
            i32x8{(int)r0[0], (int)r0[1], (int)r0[2], (int)r0[3], (int)r1[0], (int)r1[1], (int)r1[2], (int)r1[3]}};
}
MX_INLINE i32x16 lo16(const F24 &x) { return x.n; }
MX_INLINE i32x16 mid16(const F24 &x) {
    const i32x16 r = __builtin_shufflevector(x.r, x.r, 0, 1, 2, 3, 4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1);
    return __builtin_shufflevector(x.n, r, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23);
}
MX_INLINE void fence() { __builtin_amdgcn_sched_barrier(0); }

// Fragment loads take per-lane base addresses (VGPRs computed outside the WMMA stream, the lane's offset folded in) plus
// immediate offsets: no VALU address math between the WMMAs (in expert scheduling mode a ds_load whose address an
// in-flight VALU writes waits for every in-flight VALU, the WMMAs included).
struct KAddr {
    uint32_t n, r, s;  // nope (row lr, half lh), rope, scales of kv16 tile 0
};
struct VAddr {
    uint32_t t, s;
};
MX_INLINE uint32_t pin(uint32_t x) {  // keep x in a VGPR as computed here (no rematerialization at the use)
    asm volatile("" : "+v"(x));
    return x;
}
MX_INLINE KAddr kaddr(uint32_t stage_base, int lr, int lh) {
    return {pin(stage_base + (unsigned)(lr * XK + 64 * lh)), pin(stage_base + (unsigned)(lr * XK + 128 + 32 * lh)),
            pin(stage_base + (unsigned)(K_B + lr * 8 + 4 * lh))};
}
MX_INLINE VAddr vaddr(uint32_t stage_base, int lr, int lh, int lane) {
    return {pin(stage_base + (unsigned)(K_B + KS_B + lr * XV + 64 * lh)),
            pin(stage_base + (unsigned)(K_B + KS_B + V_B + lane * 4))};
}
template <int J>
MX_INLINE void load_k1(KFrag &k, const KAddr &a) {
    const lds_u8 *pn = lds_ptr(a.n), *pr = lds_ptr(a.r), *ps = lds_ptr(a.s);
    constexpr int o = 16 * J * XK;
    k.x[J] = frag24(lds_b128(pn, o), lds_b128(pn, o + 16), lds_b128(pn, o + 32), lds_b128(pn, o + 48),
                    lds_b128(pr, o), lds_b128(pr, o + 16));
    k.sc[J] = (int)lds_b32(ps, 16 * J * 8);
}
MX_INLINE void load_v(VFrag &v, const VAddr &a) {
    const lds_u8 *pt = lds_ptr(a.t), *ps = lds_ptr(a.s);
    static_for<NDT>([&](auto dd) {
        constexpr int o = 16 * (int)dd * XV;
        const u32x4 a0 = lds_b128(pt, o), a1 = lds_b128(pt, o + 16), a2 = lds_b128(pt, o + 32),
                    a3 = lds_b128(pt, o + 48);
        v.t[(int)dd] = i32x16{(int)a0[0], (int)a0[1], (int)a0[2], (int)a0[3], (int)a1[0], (int)a1[1], (int)a1[2],
                              (int)a1[3], (int)a2[0], (int)a2[1], (int)a2[2], (int)a2[3], (int)a3[0], (int)a3[1],
                              (int)a3[2], (int)a3[3]};
    });
    static_for<NDT / 2>([&](auto pp) { v.sc[(int)pp] = (int)lds_b32(ps, 32 * 4 * (int)pp); });
}
struct QAddr {
    uint32_t n, r, s;  // q tile 0's rows; q tile i is 64 rows further (immediate offsets)
};
template <int I>
MX_INLINE void load_q(QFrag &q, const QAddr &a) {
    const lds_u8 *pn = lds_ptr(a.n), *pr = lds_ptr(a.r);
    constexpr int o = 64 * I * XK;
    q.x = frag24(lds_b128(pn, o), lds_b128(pn, o + 16), lds_b128(pn, o + 32), lds_b128(pn, o + 48), lds_b128(pr, o),
                 lds_b128(pr, o + 16));
    q.sc = (int)lds_b32(lds_ptr(a.s), 64 * I * 8);
}

// QK WMMA w (0..15) of a q tile: w < 8 the nope WMMA of kv16 tile w (accumulator starts at Mt = -m), w >= 8 the rope
// WMMA (d 64..191, scale bytes R2) of kv16 tile w - 8
template <int W>
MX_INLINE void qk_w(f32x8 (&s)[NKV], const KFrag &k, const QFrag &q, const f32x8 &mt) {
    if constexpr (W < NKV) {
        s[W] = wmma_mx<0, 0>(lo16(k.x[W]), lo16(q.x), mt, k.sc[W], q.sc);
    } else {
        s[W - NKV] = wmma_mx<1, 1>(mid16(k.x[W - NKV]), mid16(q.x), s[W - NKV], k.sc[W - NKV], q.sc);
    }
}
// PV WMMA w (0..8) of a q tile: w < 8 V^T tile w, w == 8 the all-ones tile (the row sum)
template <int W>
MX_INLINE void pv_w(f32x8 (&o)[NDT], f32x8 &os, const VFrag &v, i32x16 pb, const i32x16 &ones) {
    if constexpr (W == NDT)
        os = wmma_mx<0, 0>(ones, pb, os, SC_ONE, PSCALE);
    else if constexpr (W % 2 == 0)
        o[W] = wmma_mx<0, 0>(v.t[W], pb, o[W], v.sc[W / 2], PSCALE);
    else
        o[W] = wmma_mx<1, 0>(v.t[W], pb, o[W], v.sc[W / 2], PSCALE);
}

// softmax chunk C (0..12) of one q tile: 0..3 the (masked) partial max of kv16 tiles 2C, 2C + 1; 4 the max, the
// rescale decision and (branch) the rescale; 5..12 P of kv16 tile C - 5 (8 exp2, one pk8 conversion)
template <int C, bool DIAG>
MX_INLINE void sm_chunk(SmState &st, f32x8 (&s)[NKV], float &m, f32x8 &mt, f32x8 (&o)[NDT], f32x8 &os, i32x16 &pb,
                        int kvrow0, int qrow, bool first) {
    if constexpr (C < 4) {
        float a = -__builtin_inff();
        static_for<2>([&](auto uu) {
            constexpr int j = 2 * C + uu;
            if constexpr (DIAG) {
                static_for<8>([&](auto rr) {
                    constexpr int r = rr;
                    s[j][r] = (kvrow0 + 16 * j + r > qrow) ? -__builtin_inff() : s[j][r];
                });
            }
            static_for<8>([&](auto rr) { a = __builtin_fmaxf(a, s[j][(int)rr]); });
        });
        st.pm[C] = a;
    } else if constexpr (C == 4) {
        const float mx = __builtin_fmaxf(__builtin_fmaxf(st.pm[0], st.pm[1]), __builtin_fmaxf(st.pm[2], st.pm[3]));
        const float lm = __builtin_fmaxf(mx, peer(mx));
        if (first || __builtin_amdgcn_ballot_w32(lm > TAU2) != 0u) {  // wave-uniform
            const float d = first ? lm : __builtin_fmaxf(lm, 0.f);
            const float corr = first ? 0.f : exp2_raw(-d);
            const f32x2 dv = {-d, -d}, cv = {corr, corr};
            static_for<NKV>([&](auto jj) {
                static_for<4>([&](auto hh) {
                    constexpr int j = jj, h2 = hh;
                    f32x2 x = {s[j][2 * h2], s[j][2 * h2 + 1]};
                    x = x + dv;
                    s[j][2 * h2] = x[0];
                    s[j][2 * h2 + 1] = x[1];
                });
            });
            static_for<NDT>([&](auto dd) {
                static_for<4>([&](auto hh) {
                    constexpr int dt = dd, h2 = hh;
                    f32x2 x = {o[dt][2 * h2], o[dt][2 * h2 + 1]};
                    x = x * cv;
                    o[dt][2 * h2] = x[0];
                    o[dt][2 * h2 + 1] = x[1];
                });
            });
            static_for<4>([&](auto hh) {
                constexpr int h2 = hh;
                f32x2 x = {os[2 * h2], os[2 * h2 + 1]}, y = {mt[2 * h2], mt[2 * h2 + 1]};
                x = x * cv;
                y = y + dv;
                os[2 * h2] = x[0];
                os[2 * h2 + 1] = x[1];
                mt[2 * h2] = y[0];
                mt[2 * h2 + 1] = y[1];
            });
            m += d;
        }
    } else {
        constexpr int j = C - 5;
        f32x8 p;
        static_for<8>([&](auto rr) { p[(int)rr] = exp2_raw(s[j][(int)rr]); });
        const u32x2 q = cvt8_e4m3(p, 1.0f / 128.0f);
        pb[2 * j] = (int)q[0];
        pb[2 * j + 1] = (int)q[1];
    }
}
constexpr int NCHUNK = 13;
}  // namespace

extern "C" __global__ __attribute__((amdgpu_flat_work_group_size(128, 128), amdgpu_waves_per_eu(1, 1)))
void k_mla_fwd_mx(const unsigned char *__restrict__ Q8, const unsigned char *__restrict__ QS,
                  const unsigned char *__restrict__ K8, const unsigned char *__restrict__ KS,
                  const unsigned char *__restrict__ V8T, const unsigned char *__restrict__ VS, bf16 *__restrict__ O,
                  float *__restrict__ LSE, float scale, int S, int H, int B, long long o_sb, long long o_ss) {
    __shared__ __attribute__((aligned(1024))) unsigned char smem[LDS_B];
    (void)scale;  // folded into Q8
    lock_simd();
    const uint32_t lds0 = lds_addr(smem);
    const int tid = (int)__builtin_amdgcn_workitem_id_x();
    const int wv = wave_id(tid);
    const int lane = tid & 31, lr = lane & 15, lh = lane >> 4;
    const int cnt0 = wv == 0 ? 1 : 0;
    const int nqb = S / BM;
    const int qb = nqb - 1 - (int)blockIdx.x;
    const int h = (int)blockIdx.y, b = (int)blockIdx.z;
    const int q0 = qb * BM;
    const int n = (q0 + BM) / NB;
    const unsigned long long bh = (unsigned long long)b * H + h;
    const unsigned long long gq = (unsigned long long)(uintptr_t)Q8 + (bh * S + q0) * D_QK;
    const unsigned long long gqs = (unsigned long long)(uintptr_t)QS + (bh * S + q0) * 8;
    const unsigned long long gk = (unsigned long long)(uintptr_t)K8 + bh * S * D_QK;
    const unsigned long long gks = (unsigned long long)(uintptr_t)KS + bh * S * 8;
    const unsigned long long gv = (unsigned long long)(uintptr_t)V8T + bh * D_V * S;
    const unsigned long long gvs = (unsigned long long)(uintptr_t)VS + bh * (unsigned long long)(S / NB) * VS_B;

    auto issue_slot = [&](int t) {
        const bool real = t < n;
        const int tt = real ? t : n - 1;
        const unsigned rk = real ? NB : 1, rv = real ? D_V : 1;
        const unsigned st = lds0 + (unsigned)((t & (NSTG - 1)) * STG_B);
        const unsigned long long k0 = (unsigned long long)tt * NB;
        tdm_load(tdm_g0(st, gk + k0 * D_QK, cnt0), tdm_g1(W0_KN, rk, 128, NB, D_QK));
        tdm_load(tdm_g0(st + 128, gk + k0 * D_QK + 128, cnt0), tdm_g1(W0_KR, rk, 64, NB, D_QK));
        tdm_load(tdm_g0(st + K_B, gks + k0 * 8, cnt0), tdm_g1(W0_S, rk, 8, NB, 8));
        tdm_load(tdm_g0(st + K_B + KS_B, gv + k0, cnt0), tdm_g1(W0_V, rv, 128, D_V, (unsigned long long)S));
        tdm_load(tdm_g0(st + K_B + KS_B + V_B, gvs + (unsigned long long)tt * VS_B, cnt0), tdm_g1(W0_S, rv, 4, D_V, 4));
    };
    auto stage = [&](int t) { return lds0 + (unsigned)((t & (NSTG - 1)) * STG_B); };

    tdm_load(tdm_g0(lds0 + Q_OFF, gq, wv == 1 ? 1 : 0), tdm_g1(W0_KN, BM, 128, BM, D_QK));
    tdm_load(tdm_g0(lds0 + Q_OFF + 128, gq + 128, wv == 2 ? 1 : 0), tdm_g1(W0_KR, BM, 64, BM, D_QK));
    tdm_load(tdm_g0(lds0 + QS_OFF, gqs, wv == 3 ? 1 : 0), tdm_g1(W0_S, BM, 8, BM, 8));
    issue_slot(0);
    issue_slot(1);
    issue_slot(2);
    tensor_wait<0>();
    bar_signal();
    bar_wait();  // B_0

    const unsigned qrow0 = (unsigned)(16 * wv + lr);
    const QAddr qa = {pin(lds0 + Q_OFF + qrow0 * XK + 64 * lh), pin(lds0 + Q_OFF + qrow0 * XK + 128 + 32 * lh),
                      pin(lds0 + QS_OFF + qrow0 * 8 + 4 * lh)};
    f32x8 o[QT][NDT], os[QT], mt[QT];
    float m[QT];
    static_for<QT>([&](auto ii) {
        constexpr int i = ii;
        m[i] = 0.f;
        os[i] = mt[i] = f32x8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        static_for<NDT>([&](auto dd) { o[i][(int)dd] = f32x8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}; });
    });
    const i32x16 ones = {ONES, ONES, ONES, ONES, ONES, ONES, ONES, ONES,
                         ONES, ONES, ONES, ONES, ONES, ONES, ONES, ONES};

    KFrag kf;
    VFrag vf;
    {
        const KAddr ka = kaddr(stage(0), lr, lh);
        static_for<NKV>([&](auto jj) { load_k1<(int)jj>(kf, ka); });
    }

    f32x8 sa[NKV], sb[NKV];
    i32x16 pa, pbb;
    QFrag qf[2];
    load_q<0>(qf[0], qa);
    load_q<1>(qf[1], qa);
    static_for<16>([&](auto ww) { qk_w<(int)ww>(sa, kf, qf[0], mt[0]); });
    fence();

    // tile t: on entry sa = S''(q0) of tile t, qf[1] = Q(q1), kf = K(t). Slot 3 computes QK(q0) of tile t + 1 (not
    // LAST).
    auto tile = [&](int t, auto diag_c, auto last_c) __attribute__((always_inline)) {
        constexpr bool DIAG = decltype(diag_c)::value, LAST = decltype(last_c)::value;
        const bool first = t == 0;
        const VAddr va = vaddr(stage(t), lr, lh, lane);
        const KAddr ka = kaddr(stage(t + 1), lr, lh);
        issue_slot(t + 3);
        load_v(vf, va);
        const int kvrow0 = t * NB + 8 * lh;
        fence();
        static_for<QT>([&](auto ii) {
            constexpr int i = ii;
            f32x8(&scur)[NKV] = (i % 2 == 0) ? sa : sb;
            f32x8(&snext)[NKV] = (i % 2 == 0) ? sb : sa;
            i32x16 &pcur = (i % 2 == 0) ? pa : pbb;
            i32x16 &pprev = (i % 2 == 0) ? pbb : pa;
            const int qr_i = q0 + 64 * i + 16 * wv + lr;
            SmState sm;
            if constexpr (i == 1) {
                tensor_wait<10>();
                bar_signal();  // B_{t+1}
                fence();
            }
            if constexpr (i == QT - 1) {
                bar_wait();
                if constexpr (!LAST) {
                    static_for<NKV>([&](auto jj) {
                        load_k1<(int)jj>(kf, ka);
                    });
                }
                fence();
            }
            constexpr int NPV = (i >= 1) ? NDT + 1 : 0;
            constexpr int NQK = (i + 1 < QT || !LAST) ? 16 : 0;
            constexpr int NW = NPV + NQK;
            constexpr int NSTEP = NW > 0 ? NW : 1;
            constexpr int qn_mt = (i + 1) % QT;  // the q tile the slot's QK is for
            static_for<NSTEP>([&](auto kk) {
                constexpr int k = kk;
                if constexpr (k == 0 && !(LAST && i >= 2)) {
                    constexpr int qn = (i + 2) % QT;  // the next slot's next q tile, into the buffer read a slot ago
                    load_q<qn>(qf[i % 2], qa);
                    fence();
                }
                if constexpr (k < NPV) {
                    pv_w<k>(o[i - 1 < 0 ? 0 : i - 1], os[i - 1 < 0 ? 0 : i - 1], vf, pprev, ones);
                } else if constexpr (k < NW) {
                    qk_w<k - NPV>(snext, kf, qf[(i + 1) % 2], mt[qn_mt]);
                }
                fence();
                // the chunks spread evenly over the slot's WMMAs: chunk c at step floor(c * NSTEP / NCHUNK)
                static_for<NCHUNK>([&](auto cc) {
                    constexpr int c = cc;
                    if constexpr (c * NSTEP / NCHUNK == k) {
                        sm_chunk<c, DIAG>(sm, scur, m[i], mt[i], o[i], os[i], pcur, kvrow0, qr_i, first);
                        fence();
                    }
                });
            });
            (void)pprev;
        });
        // tail: PV(q3)
        static_for<NDT + 1>([&](auto ww) { pv_w<(int)ww>(o[QT - 1], os[QT - 1], vf, pbb, ones); });
        fence();
    };
    for (int t = 0; t < n - 2; ++t) tile(t, std::false_type{}, std::false_type{});
    tile(n - 2, std::true_type{}, std::false_type{});
    tile(n - 1, std::true_type{}, std::true_type{});

    static_for<QT>([&](auto ii) {
        constexpr int i = ii;
        const int qr_i = q0 + 64 * i + 16 * wv + lr;
        const float l = os[i][0];
        const float inv = l > 0.f ? 1.0f / l : 0.f;
        bf16 *orow = O + (long long)b * o_sb + (long long)qr_i * o_ss + (long long)h * D_V;
        static_for<NDT>([&](auto dd) {
            constexpr int dt = dd;
            bf16x8 w;
            static_for<8>([&](auto rr) { w[(int)rr] = (bf16)(o[i][dt][(int)rr] * inv); });
            *(u32x4 *)(orow + 16 * dt + 8 * lh) = __builtin_bit_cast(u32x4, w);
        });
        if (lh == 0) LSE[(bh * S) + qr_i] = (m[i] + __builtin_log2f(l)) * LN2;
    });
    tensor_wait<0>();
}
