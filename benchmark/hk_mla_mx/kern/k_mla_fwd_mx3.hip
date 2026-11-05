/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// k_mla_fwd_mx (rung m3) -- m2 (Q read from LDS per tile, K prefetched one tile ahead, B_{t+1} after QK of q tile 0)
// with the q-tile pipeline interleaved instruction by instruction: in slot i of a 128-kv tile the WMMAs of QK(q_{i+1})
// and PV(q_{i-1}) alternate with the 22 chunks of softmax(q_i) (sched_barrier(0) fences keep the order), so the
// matrix unit runs under the softmax instead of after it:
//   head     V(t) reads, QK(q0)                                        16 WMMAs
//   slot 0   QK(q1)             | softmax(q0)                          16 WMMAs, 22 chunks
//   slot 1   QK(q2) + PV(q0)    | softmax(q1)                          24
//   slot 2   QK(q3) + PV(q1)    | softmax(q2)                          24
//   slot 3   B_{t+1} wait, K(t + 1) prefetch, PV(q2) | softmax(q3)       8
//   tail     PV(q3)                                                     8
// Softmax chunks: 0..3 the (masked) partial row max of kv16 tiles 2c, 2c + 1; 4 the max, rescale decision, corr; 5..12
// p of kv16 tile c - 5 (pk_fma, exp, pk_add, pk8 convert into the PV B operand); 13 the row sum; 14..21 the O rescale
// of d16 tile c - 14. Numerics are m0's (bitwise: same operations in the same order per element; the row-sum order
// differs: pairwise packed partial sums).
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
constexpr float TAU2 = 1.4426950408889634f;
constexpr int PSCALE = 120 * 0x01010101;  // P = E4M3(p * 2^7)

struct KFrag {
    i32x16 n[NKV], r[NKV];
    int sc[NKV];
};
struct VFrag {
    i32x16 t[NDT];
    int sc[NDT / 2];
};
struct QFrag {
    i32x16 n, r;
    int sc;
};
struct SmState {  // softmax(q_i) scratch between its chunks
    float pm[4];
    float mn, corr;
    f32x2 acc;
};

MX_INLINE i32x16 cat4(u32x4 a0, u32x4 a1, u32x4 a2, u32x4 a3) {
    return i32x16{(int)a0[0], (int)a0[1], (int)a0[2], (int)a0[3], (int)a1[0], (int)a1[1], (int)a1[2], (int)a1[3],
                  (int)a2[0], (int)a2[1], (int)a2[2], (int)a2[3], (int)a3[0], (int)a3[1], (int)a3[2], (int)a3[3]};
}
MX_INLINE i32x16 cat2z(u32x4 a0, u32x4 a1) {
    return i32x16{(int)a0[0], (int)a0[1], (int)a0[2], (int)a0[3], (int)a1[0], (int)a1[1], (int)a1[2], (int)a1[3],
                  0, 0, 0, 0, 0, 0, 0, 0};
}
MX_INLINE void fence() { __builtin_amdgcn_sched_barrier(0); }

template <int J>
MX_INLINE void load_k1(KFrag &k, const lds_u8 *kp, const lds_u8 *ksp, int lr, int lh) {
    const int o = (16 * J + lr) * XK;
    k.n[J] = cat4(lds_b128(kp, o + 64 * lh), lds_b128(kp, o + 64 * lh + 16), lds_b128(kp, o + 64 * lh + 32),
                  lds_b128(kp, o + 64 * lh + 48));
    k.r[J] = cat2z(lds_b128(kp, o + 128 + 32 * lh), lds_b128(kp, o + 128 + 32 * lh + 16));
    k.sc[J] = (int)lds_b32(ksp, (16 * J + lr) * 8 + 4 * lh);
}
MX_INLINE void load_v(VFrag &v, const lds_u8 *vp, const lds_u8 *vsp, int lr, int lh, int lane) {
    static_for<NDT>([&](auto dd) {
        constexpr int dt = dd;
        const int o = (16 * dt + lr) * XV + 64 * lh;
        v.t[dt] = cat4(lds_b128(vp, o), lds_b128(vp, o + 16), lds_b128(vp, o + 32), lds_b128(vp, o + 48));
    });
    static_for<NDT / 2>([&](auto pp) { v.sc[(int)pp] = (int)lds_b32(vsp, (32 * (int)pp + lane) * 4); });
}
MX_INLINE void load_q(QFrag &q, uint32_t qrow_lds, uint32_t qs_lds, int lh) {
    const lds_u8 *p = lds_ptr(qrow_lds);
    q.n = cat4(lds_b128(p, 64 * lh), lds_b128(p, 64 * lh + 16), lds_b128(p, 64 * lh + 32), lds_b128(p, 64 * lh + 48));
    q.r = cat2z(lds_b128(p, 128 + 32 * lh), lds_b128(p, 128 + 32 * lh + 16));
    q.sc = (int)lds_b32(lds_ptr(qs_lds), 4 * lh);
}

// QK WMMA w (0..15) of a q tile: w < 8 the nope WMMA of kv16 tile w, w >= 8 the rope WMMA of kv16 tile w - 8
template <int W>
MX_INLINE void qk_w(f32x8 (&s)[NKV], const KFrag &k, const QFrag &q) {
    if constexpr (W < NKV) {
        const f32x8 z = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        s[W] = wmma_mx<0, 0>(k.n[W], q.n, z, k.sc[W], q.sc);
    } else {
        s[W - NKV] = wmma_mx<1, 1>(k.r[W - NKV], q.r, s[W - NKV], k.sc[W - NKV], q.sc);
    }
}
template <int DT>
MX_INLINE void pv_w(f32x8 (&o)[NDT], const VFrag &v, i32x16 pb) {
    if constexpr (DT % 2 == 0)
        o[DT] = wmma_mx<0, 0>(v.t[DT], pb, o[DT], v.sc[DT / 2], PSCALE);
    else
        o[DT] = wmma_mx<1, 0>(v.t[DT], pb, o[DT], v.sc[DT / 2], PSCALE);
}

// softmax chunk C (0..21) of one q tile; see the header
template <int C, bool DIAG>
MX_INLINE void sm_chunk(SmState &st, f32x8 (&s)[NKV], float &m, float &l, f32x8 (&o)[NDT], i32x16 &pb, float c2,
                        int kvrow0, int qrow) {
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
        const float need = __builtin_fmaf(lm, c2, -m);
        const bool resc = __builtin_amdgcn_ballot_w32(need > TAU2) != 0u;
        st.mn = resc ? __builtin_fmaxf(m, lm * c2) : m;
        st.corr = exp2_raw(m - st.mn);
        st.acc = f32x2{0.f, 0.f};
    } else if constexpr (C < 13) {
        constexpr int j = C - 5;
        const f32x2 c2v = {c2, c2}, mnv = {-st.mn, -st.mn};
        f32x8 p;
        static_for<4>([&](auto hh) {
            constexpr int h2 = hh;
            f32x2 x = {s[j][2 * h2], s[j][2 * h2 + 1]};
            x = x * c2v + mnv;
            const f32x2 e = {exp2_raw(x[0]), exp2_raw(x[1])};
            st.acc = st.acc + e;
            p[2 * h2] = e[0];
            p[2 * h2 + 1] = e[1];
        });
        const u32x2 q = cvt8_e4m3(p, 1.0f / 128.0f);
        pb[2 * j] = (int)q[0];
        pb[2 * j + 1] = (int)q[1];
    } else if constexpr (C == 13) {
        const float ls = st.acc[0] + st.acc[1];
        l = __builtin_fmaf(st.corr, l, ls + peer(ls));
        m = st.mn;
    } else {
        constexpr int dt = C - 14;
        const f32x2 cv = {st.corr, st.corr};
        static_for<4>([&](auto hh) {
            constexpr int h2 = hh;
            f32x2 x = {o[dt][2 * h2], o[dt][2 * h2 + 1]};
            x = x * cv;
            o[dt][2 * h2] = x[0];
            o[dt][2 * h2 + 1] = x[1];
        });
    }
}
constexpr int NCHUNK = 22;
}  // namespace

extern "C" __global__ __attribute__((amdgpu_flat_work_group_size(128, 128), amdgpu_waves_per_eu(1, 1)))
void k_mla_fwd_mx(const unsigned char *__restrict__ Q8, const unsigned char *__restrict__ QS,
                  const unsigned char *__restrict__ K8, const unsigned char *__restrict__ KS,
                  const unsigned char *__restrict__ V8T, const unsigned char *__restrict__ VS, bf16 *__restrict__ O,
                  float *__restrict__ LSE, float scale, int S, int H, int B, long long o_sb, long long o_ss) {
    __shared__ __attribute__((aligned(1024))) unsigned char smem[LDS_B];
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

    // LDS addresses of this wave's Q rows (q tile i = block rows 64 i + 16 w + lr)
    uint32_t qrow[QT], qsrow[QT];
    static_for<QT>([&](auto ii) {
        constexpr int i = ii;
        qrow[i] = lds0 + Q_OFF + (unsigned)((64 * i + 16 * wv + lr) * XK);
        qsrow[i] = lds0 + QS_OFF + (unsigned)((64 * i + 16 * wv + lr) * 8);
    });
    f32x8 o[QT][NDT];
    float m[QT], l[QT];
    static_for<QT>([&](auto ii) {
        constexpr int i = ii;
        m[i] = -1.0e30f;
        l[i] = 0.f;
        static_for<NDT>([&](auto dd) { o[i][(int)dd] = f32x8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}; });
    });
    const float c2 = scale * LOG2E;

    KFrag kf;
    VFrag vf;
    static_for<NKV>([&](auto jj) { load_k1<(int)jj>(kf, lds_ptr(stage(0)), lds_ptr(stage(0) + K_B), lr, lh); });

    // S / P of the even / odd q tiles; Q fragments (one q tile at a time)
    f32x8 sa[NKV], sb[NKV];
    i32x16 pa, pbb;
    QFrag qf;
    // the first tile's head: QK(q0) on tile 0, then q tile 1's Q fragments
    load_q(qf, qrow[0], qsrow[0], lh);
    static_for<16>([&](auto ww) { qk_w<(int)ww>(sa, kf, qf); });
    load_q(qf, qrow[1], qsrow[1], lh);
    fence();

    // tile t: on entry sa = S(q0) of tile t, qf = Q(q1), kf = K(t). Slot 3 computes QK(q0) of tile t + 1 (not LAST).
    auto tile = [&](int t, auto diag_c, auto last_c) __attribute__((always_inline)) {
        constexpr bool DIAG = decltype(diag_c)::value, LAST = decltype(last_c)::value;
        issue_slot(t + 3);  // into the stage of tile t - 1, freed by B_t
        const unsigned st = stage(t);
        load_v(vf, lds_ptr(st + K_B + KS_B), lds_ptr(st + K_B + KS_B + V_B), lr, lh, lane);
        const int kvrow0 = t * NB + 8 * lh;
        fence();
        static_for<QT>([&](auto ii) {
            constexpr int i = ii;
            f32x8(&scur)[NKV] = (i % 2 == 0) ? sa : sb;
            f32x8(&snext)[NKV] = (i % 2 == 0) ? sb : sa;  // i == 3: sa = S(q0) of tile t + 1
            i32x16 &pcur = (i % 2 == 0) ? pa : pbb;
            i32x16 &pprev = (i % 2 == 0) ? pbb : pa;
            const int qr_i = q0 + 64 * i + 16 * wv + lr;
            SmState sm;
            if constexpr (i == 1) {
                tensor_wait<10>();  // this wave's slot t + 1 landed
                bar_signal();       // B_{t+1}: every read of stage t (K during tile t - 1, V at the top) retired
                fence();
            }
            if constexpr (i == QT - 1) {
                bar_wait();  // B_{t+1}
                if constexpr (!LAST) {
                    static_for<NKV>([&](auto jj) {
                        load_k1<(int)jj>(kf, lds_ptr(stage(t + 1)), lds_ptr(stage(t + 1) + K_B), lr, lh);
                    });
                }
                fence();
            }
            // the slot's WMMA list: PV(q_{i-1}) dt = 0..7 (i >= 1), then QK of the next q tile w = 0..15 (q_{i+1} on
            // this tile, or q0 of tile t + 1 in slot 3)
            constexpr int NPV = (i >= 1) ? 8 : 0;
            constexpr int NQK = (i + 1 < QT || !LAST) ? 16 : 0;
            constexpr int NW = NPV + NQK;
            constexpr int NSTEP = NW > NCHUNK ? NW : NCHUNK;
            static_for<NSTEP>([&](auto kk) {
                constexpr int k = kk;
                if constexpr (k < NPV) {
                    pv_w<k>(o[i - 1 < 0 ? 0 : i - 1], vf, pprev);
                } else if constexpr (k < NW) {
                    qk_w<k - NPV>(snext, kf, qf);
                    if constexpr (k == NW - 1) {
                        // the next slot's Q fragments: q_{i+2}; slot 2 loads q0 (for slot 3's QK of tile t + 1),
                        // slot 3 loads q1 (for slot 0 of tile t + 1)
                        constexpr int qn = (i + 2) % QT;
                        if constexpr (!(LAST && i >= 2)) {
                            fence();
                            load_q(qf, qrow[qn], qsrow[qn], lh);
                        }
                    }
                }
                fence();
                if constexpr (k < NCHUNK) {
                    sm_chunk<k, DIAG>(sm, scur, m[i], l[i], o[i], pcur, c2, kvrow0, qr_i);
                }
                fence();
            });
            (void)pprev;
        });
        // tail: PV(q3)
        static_for<NDT>([&](auto dd) { pv_w<(int)dd>(o[QT - 1], vf, pbb); });
        fence();
    };
    for (int t = 0; t < n - 2; ++t) tile(t, std::false_type{}, std::false_type{});
    tile(n - 2, std::true_type{}, std::false_type{});
    tile(n - 1, std::true_type{}, std::true_type{});

    static_for<QT>([&](auto ii) {
        constexpr int i = ii;
        const int qr_i = q0 + 64 * i + 16 * wv + lr;
        const float inv = l[i] > 0.f ? 1.0f / l[i] : 0.f;
        bf16 *orow = O + (long long)b * o_sb + (long long)qr_i * o_ss + (long long)h * D_V;
        static_for<NDT>([&](auto dd) {
            constexpr int dt = dd;
            bf16x8 w;
            static_for<8>([&](auto rr) { w[(int)rr] = (bf16)(o[i][dt][(int)rr] * inv); });
            *(u32x4 *)(orow + 16 * dt + 8 * lh) = __builtin_bit_cast(u32x4, w);
        });
        if (lh == 0) LSE[(bh * S) + qr_i] = (m[i] + __builtin_log2f(l[i])) * LN2;
    });
    tensor_wait<0>();
}
