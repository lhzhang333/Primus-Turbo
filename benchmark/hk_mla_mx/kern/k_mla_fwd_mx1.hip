/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// k_mla_fwd_mx (rung m1) -- the MXFP8 MLA forward of k_mla_fwd_mx.hip (same operands, numerics and output) with the
// work software-pipelined:
//   * the K fragments of tile t + 1 are read right after tile t's last QK (they live in the registers QK has freed),
//     its V fragments at the top of tile t + 1 (first used by PV of q tile 0, after QK and the softmax of q tile 0);
//   * B_{t+1} is signalled after QK of q tile 0 (every read of stage t has been issued long before: its dscnt wait
//     finds them retired) and waited after the last QK, right before the prefetch;
//   * QK of q tile i + 1 is issued before the softmax of q tile i (two S sets live);
//   * the tiles below the diagonal and the two diagonal tiles are separate code (no mask test in the steady loop);
//   * the O rescale is branch-free (corr = 1 exactly when the max did not move).
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

MX_INLINE i32x16 cat4(u32x4 a0, u32x4 a1, u32x4 a2, u32x4 a3) {
    return i32x16{(int)a0[0], (int)a0[1], (int)a0[2], (int)a0[3], (int)a1[0], (int)a1[1], (int)a1[2], (int)a1[3],
                  (int)a2[0], (int)a2[1], (int)a2[2], (int)a2[3], (int)a3[0], (int)a3[1], (int)a3[2], (int)a3[3]};
}
MX_INLINE i32x16 cat2z(u32x4 a0, u32x4 a1) {
    return i32x16{(int)a0[0], (int)a0[1], (int)a0[2], (int)a0[3], (int)a1[0], (int)a1[1], (int)a1[2], (int)a1[3],
                  0, 0, 0, 0, 0, 0, 0, 0};
}

MX_INLINE void load_k(KFrag &k, const lds_u8 *kp, const lds_u8 *ksp, int lr, int lh) {
    static_for<NKV>([&](auto jj) {
        constexpr int j = jj;
        const int o = (16 * j + lr) * XK;
        k.n[j] = cat4(lds_b128(kp, o + 64 * lh), lds_b128(kp, o + 64 * lh + 16), lds_b128(kp, o + 64 * lh + 32),
                      lds_b128(kp, o + 64 * lh + 48));
        k.r[j] = cat2z(lds_b128(kp, o + 128 + 32 * lh), lds_b128(kp, o + 128 + 32 * lh + 16));
        k.sc[j] = (int)lds_b32(ksp, (16 * j + lr) * 8 + 4 * lh);
    });
}
MX_INLINE void load_v(VFrag &v, const lds_u8 *vp, const lds_u8 *vsp, int lr, int lh, int lane) {
    static_for<NDT>([&](auto dd) {
        constexpr int dt = dd;
        const int o = (16 * dt + lr) * XV + 64 * lh;
        v.t[dt] = cat4(lds_b128(vp, o), lds_b128(vp, o + 16), lds_b128(vp, o + 32), lds_b128(vp, o + 48));
    });
    static_for<NDT / 2>([&](auto pp) { v.sc[(int)pp] = (int)lds_b32(vsp, (32 * (int)pp + lane) * 4); });
}

// S^T of one q16 tile: 8 kv16 tiles x (nope + rope) MX WMMAs
MX_INLINE void qk(f32x8 (&s)[NKV], const KFrag &k, i32x16 qn, i32x16 qr, int qsc) {
    static_for<NKV>([&](auto jj) {
        constexpr int j = jj;
        f32x8 z = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        z = wmma_mx<0, 0>(k.n[j], qn, z, k.sc[j], qsc);
        s[j] = wmma_mx<1, 1>(k.r[j], qr, z, k.sc[j], qsc);
    });
}

// online softmax of one q16 tile, P packed as the PV B operand, O rescaled; DIAG: the causal mask (kv > q -> -inf)
template <bool DIAG>
MX_INLINE i32x16 softmax(f32x8 (&s)[NKV], float &m, float &l, f32x8 (&o)[NDT], float c2, int kvrow0, int qrow) {
    if constexpr (DIAG) {
        static_for<NKV>([&](auto jj) {
            constexpr int j = jj;
            static_for<8>([&](auto rr) {
                constexpr int r = rr;
                s[j][r] = (kvrow0 + 16 * j + r > qrow) ? -__builtin_inff() : s[j][r];
            });
        });
    }
    float mx = -__builtin_inff();
    static_for<NKV>([&](auto jj) { static_for<8>([&](auto rr) { mx = __builtin_fmaxf(mx, s[(int)jj][(int)rr]); }); });
    const float lm = __builtin_fmaxf(mx, peer(mx));
    const float need = __builtin_fmaf(lm, c2, -m);
    const bool resc = __builtin_amdgcn_ballot_w32(need > TAU2) != 0u;
    const float mn = resc ? __builtin_fmaxf(m, lm * c2) : m;
    const float corr = exp2_raw(m - mn);
    const f32x2 c2v = {c2, c2}, mnv = {-mn, -mn};
    f32x2 acc = {0.f, 0.f};
    i32x16 pb;
    static_for<NKV>([&](auto jj) {
        constexpr int j = jj;
        f32x8 p;
        static_for<4>([&](auto hh) {
            constexpr int h2 = hh;
            f32x2 x = {s[j][2 * h2], s[j][2 * h2 + 1]};
            x = x * c2v + mnv;  // v_pk_fma_f32
            f32x2 e = {exp2_raw(x[0]), exp2_raw(x[1])};
            acc = acc + e;      // v_pk_add_f32
            p[2 * h2] = e[0];
            p[2 * h2 + 1] = e[1];
        });
        const u32x2 q = cvt8_e4m3(p, 1.0f / 128.0f);
        pb[2 * j] = (int)q[0];
        pb[2 * j + 1] = (int)q[1];
    });
    const float ls = acc[0] + acc[1];
    l = __builtin_fmaf(corr, l, ls + peer(ls));
    m = mn;
    const f32x2 cv = {corr, corr};
    static_for<NDT>([&](auto dd) {
        constexpr int dt = dd;
        static_for<4>([&](auto hh) {
            constexpr int h2 = hh;
            f32x2 x = {o[dt][2 * h2], o[dt][2 * h2 + 1]};
            x = x * cv;  // v_pk_mul_f32
            o[dt][2 * h2] = x[0];
            o[dt][2 * h2 + 1] = x[1];
        });
    });
    return pb;
}

MX_INLINE void pv(f32x8 (&o)[NDT], const VFrag &v, i32x16 pb) {
    static_for<NDT>([&](auto dd) {
        constexpr int dt = dd;
        if constexpr (dt % 2 == 0)
            o[dt] = wmma_mx<0, 0>(v.t[dt], pb, o[dt], v.sc[dt / 2], PSCALE);
        else
            o[dt] = wmma_mx<1, 0>(v.t[dt], pb, o[dt], v.sc[dt / 2], PSCALE);
    });
}
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
    const int n = (q0 + BM) / NB;  // tiles 0 .. n - 1; the last two (kv0 = q0, q0 + 128) are the diagonal ones
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

    i32x16 qn[QT], qr[QT];
    int qsc[QT];
    static_for<QT>([&](auto ii) {
        constexpr int i = ii;
        const lds_u8 *p = lds_ptr(lds0 + Q_OFF + (unsigned)((64 * i + 16 * wv + lr) * XK));
        qn[i] = cat4(lds_b128(p, 64 * lh), lds_b128(p, 64 * lh + 16), lds_b128(p, 64 * lh + 32),
                     lds_b128(p, 64 * lh + 48));
        qr[i] = cat2z(lds_b128(p, 128 + 32 * lh), lds_b128(p, 128 + 32 * lh + 16));
        qsc[i] = (int)lds_b32(lds_ptr(lds0 + QS_OFF + (unsigned)((64 * i + 16 * wv + lr) * 8)), 4 * lh);
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
    load_k(kf, lds_ptr(stage(0)), lds_ptr(stage(0) + K_B), lr, lh);  // tile 0's K fragments

    // one tile: kf holds its K fragments; on exit kf holds tile t + 1's (when there is one)
    auto tile = [&](int t, auto diag_c, auto last_c) __attribute__((always_inline)) {
        constexpr bool DIAG = decltype(diag_c)::value, LAST = decltype(last_c)::value;
        issue_slot(t + 3);  // into the stage of tile t - 1, freed by B_t
        const unsigned st = stage(t);
        load_v(vf, lds_ptr(st + K_B + KS_B), lds_ptr(st + K_B + KS_B + V_B), lr, lh, lane);
        const int kv0 = t * NB;
        const int kvrow0 = kv0 + 8 * lh;
        f32x8 s0[NKV], s1[NKV];
        qk(s0, kf, qn[0], qr[0], qsc[0]);
        tensor_wait<10>();  // this wave's slot t + 1 landed
        bar_signal();       // B_{t+1}: every read of stage t was issued above (K during tile t - 1, V just now)
        static_for<QT>([&](auto ii) {
            constexpr int i = ii;
            f32x8(&sc)[NKV] = (i % 2 == 0) ? s0 : s1;
            f32x8(&sn)[NKV] = (i % 2 == 0) ? s1 : s0;
            if constexpr (i + 1 < QT) {
                qk(sn, kf, qn[i + 1], qr[i + 1], qsc[i + 1]);
            } else {
                bar_wait();  // B_{t+1}: tile t + 1 visible, every wave done with stage t
                if constexpr (!LAST) load_k(kf, lds_ptr(stage(t + 1)), lds_ptr(stage(t + 1) + K_B), lr, lh);
            }
            const int qrow = q0 + 64 * i + 16 * wv + lr;
            const i32x16 pb = softmax<DIAG>(sc, m[i], l[i], o[i], c2, kvrow0, qrow);
            pv(o[i], vf, pb);
        });
    };
    for (int t = 0; t < n - 2; ++t) tile(t, std::false_type{}, std::false_type{});
    tile(n - 2, std::true_type{}, std::false_type{});
    tile(n - 1, std::true_type{}, std::true_type{});

    static_for<QT>([&](auto ii) {
        constexpr int i = ii;
        const int qrow = q0 + 64 * i + 16 * wv + lr;
        const float inv = l[i] > 0.f ? 1.0f / l[i] : 0.f;
        bf16 *orow = O + (long long)b * o_sb + (long long)qrow * o_ss + (long long)h * D_V;
        static_for<NDT>([&](auto dd) {
            constexpr int dt = dd;
            bf16x8 w;
            static_for<8>([&](auto rr) { w[(int)rr] = (bf16)(o[i][dt][(int)rr] * inv); });
            *(u32x4 *)(orow + 16 * dt + 8 * lh) = __builtin_bit_cast(u32x4, w);
        });
        if (lh == 0) LSE[(bh * S) + qrow] = (m[i] + __builtin_log2f(l[i])) * LN2;
    });
    tensor_wait<0>();
}
