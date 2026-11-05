/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// k_mla_bwd_mx.hip -- DeepSeek-V3 MLA attention backward on MXFP8 operands, gfx1250 (wave32). Reference kernels
// (rung b0): correctness first. Every GEMM is an MX 16x16x128 WMMA (E4M3, E8M0 scales); no dS scratch: dQ is
// recomputed in its own kernel, in the transposed orientation, so that both quantizations of dS are lane-local.
//
// Operands (k_mxq.hip; [b, h, ...] per head; *8 E4M3, *S E8M0; K centred by kbar as in the forward):
//   Q8 / QS, K8 / KS          rows (along d, 192 = nope 128 + rope 64)          [B, H, S, 192] / [B, H, S, 8]
//   V8 / VSR, DO8 / DOS       rows (along d_v)                                 [B, H, S, 128] / [B, H, S, 4]
//   DO8T / DOTS               cols (along s), d_v rows                         [B, H, 128, S] / [B, H, S / 128, 128, 4]
//   Q8T / QTS, K8T / KTS      cols (along s), d rows (K centred)               [B, H, 192, S] / [B, H, S / 128, 192, 4]
//   LSE [B, H, S] (the forward's lse of the centred logits, natural log), DELTA [B, H, S] = rowsum(dO * O) (fp32)
//   -> dQ, dK [B, S, H, 192], dV [B, S, H, 128] bf16 (row strides *_sb / *_ss elements)
// Math (P <= 1): P = exp2(S c2 - lse log2 e), dS = P (dP - delta), dV = P^T dO, dK = scale dS^T Q, dQ = scale dS K.
// Quantization inside: P -> E4M3(P 2^8) (fixed scale 2^-8); dS -> MX along the GEMM's reduction dim: along q in
// k_dkdv_mx (blocks of 32 q rows per kv column), along kv in k_dq_mx (blocks of 32 kv per q column).
//
// k_dkdv_mx: grid (S / 128, H, B), 4 waves; wave w owns kv rows kv0 + 32 w .. + 31 (two kv16 tiles j), whose K8 / V8
// rows are the B operands of S = Q K^T and dP = dO V^T (resident). It loops over 128-row q steps q0 = kv0 .. S - 128
// (the first one on the diagonal: masked) through a 3-stage TDM ring (wave 0 issues 11 ops per step). Per step: for
// each q16 tile qt: S, dP (C layout: lane = kv column, rows = q), P, dS; P packed into the dV B operand (bytes 8 qt + r
// = q 16 qt + 8 h + r), dS of q tiles 2m, 2m + 1 quantized per column (16 values per lane + its peer) into the dK B
// operand and its scale byte m; then dV^T += dO^T P (8 d16 tiles), dK^T += Q^T dS (12 d16 tiles).
#include "mx_common.h"

using namespace mx;

namespace {
constexpr int D_QK = 192, D_V = 128;
constexpr int QS_ = 128;         // q rows per step
constexpr int KB = 128;          // kv rows per workgroup (4 waves x 32)
constexpr int NQT = QS_ / 16;    // 8 q16 tiles per step
constexpr int NJ = 2;            // kv16 tiles per wave
constexpr int XK = 208, X128 = 144;
// ring stage layout
constexpr int O_Q = 0;                         // Q8 rows: 128 x 208
constexpr int O_QS = O_Q + QS_ * XK;           // 26624: QS 128 x 8
constexpr int O_DO = O_QS + QS_ * 8;           // 27648: DO8 rows 128 x 144
constexpr int O_DOS = O_DO + QS_ * X128;       // 46080: DOS 128 x 4
constexpr int O_DOT = O_DOS + QS_ * 4;         // 46592: DO8T 128 d rows x 144
constexpr int O_DOTS = O_DOT + D_V * X128;     // 65024: DOTS 128 x 4
constexpr int O_QT = O_DOTS + D_V * 4;         // 65536: Q8T 192 d rows x 144
constexpr int O_QTS = O_QT + D_QK * X128;      // 93184: QTS 192 x 4
constexpr int O_LSE = O_QTS + D_QK * 4;        // 93952: LSE 128 x 4
constexpr int O_DEL = O_LSE + QS_ * 4;         // 94464: DELTA 128 x 4
constexpr int STG_B = O_DEL + QS_ * 4;         // 94976
constexpr int NSTG = 3;
constexpr int LDS_B = NSTG * STG_B;            // 284928
static_assert(LDS_B <= 327680, "LDS");
constexpr unsigned W0_KN = tdm_w0(0, 32, 20), W0_KR = tdm_w0(0, 16, 36), W0_128 = tdm_w0(0, 32, 4),
                   W0_S = tdm_w0(0, 0, 0);
constexpr int P_E8M0 = 119;                    // P stored as E4M3(P 2^8)
constexpr int PSCALE = P_E8M0 * 0x01010101;

MX_INLINE i32x16 cat4(u32x4 a0, u32x4 a1, u32x4 a2, u32x4 a3) {
    return i32x16{(int)a0[0], (int)a0[1], (int)a0[2], (int)a0[3], (int)a1[0], (int)a1[1], (int)a1[2], (int)a1[3],
                  (int)a2[0], (int)a2[1], (int)a2[2], (int)a2[3], (int)a3[0], (int)a3[1], (int)a3[2], (int)a3[3]};
}
MX_INLINE i32x16 cat2z(u32x4 a0, u32x4 a1) {
    return i32x16{(int)a0[0], (int)a0[1], (int)a0[2], (int)a0[3], (int)a1[0], (int)a1[1], (int)a1[2], (int)a1[3],
                  0, 0, 0, 0, 0, 0, 0, 0};
}
MX_INLINE u32x4 gload16(const unsigned char *p) { return *(const u32x4 *)p; }
MX_INLINE i32x16 lds_frag64(const lds_u8 *p, int off) {  // 64 contiguous bytes: an x128 operand half-lane image
    return cat4(lds_b128(p, off), lds_b128(p, off + 16), lds_b128(p, off + 32), lds_b128(p, off + 48));
}
// the MX block exponent of a per-lane amax combined with the peer lane (same column, the other 8 rows)
MX_INLINE int col_exp(float amax_lane) {
    const float a = __builtin_fmaxf(amax_lane, peer(amax_lane));
    return mx_exp(a);
}
}  // namespace

extern "C" __global__ __attribute__((amdgpu_flat_work_group_size(128, 128), amdgpu_waves_per_eu(1, 1)))
void k_dkdv_mx(const unsigned char *__restrict__ Q8, const unsigned char *__restrict__ QS,
               const unsigned char *__restrict__ K8, const unsigned char *__restrict__ KS,
               const unsigned char *__restrict__ V8, const unsigned char *__restrict__ VSR,
               const unsigned char *__restrict__ DO8, const unsigned char *__restrict__ DOS,
               const unsigned char *__restrict__ DO8T, const unsigned char *__restrict__ DOTS,
               const unsigned char *__restrict__ Q8T, const unsigned char *__restrict__ QTS,
               const float *__restrict__ LSE, const float *__restrict__ DELTA, bf16 *__restrict__ DK,
               bf16 *__restrict__ DV, float scale, int S, int H, int B, long long dk_sb, long long dk_ss,
               long long dv_sb, long long dv_ss) {
    __shared__ __attribute__((aligned(1024))) unsigned char smem[LDS_B];
    lock_simd();
    const uint32_t lds0 = lds_addr(smem);
    const int tid = (int)__builtin_amdgcn_workitem_id_x();
    const int wv = wave_id(tid);
    const int lane = tid & 31, lr = lane & 15, lh = lane >> 4;
    const int cnt0 = wv == 0 ? 1 : 0;
    const int kb = (int)blockIdx.x, h = (int)blockIdx.y, b = (int)blockIdx.z;
    const int kv0 = kb * KB;
    const int nst = (S - kv0) / QS_;  // q steps kv0 .. S - 128
    const unsigned long long bh = (unsigned long long)b * H + h;
    const float c2 = scale * LOG2E;

    // ---- the wave's K / V rows as B operands (global -> registers once)
    i32x16 kn[NJ], kr[NJ], vb[NJ];
    int ksc[NJ], vsc[NJ];
    static_for<NJ>([&](auto jj) {
        constexpr int j = jj;
        const unsigned long long row = bh * S + kv0 + 32 * wv + 16 * j + lr;
        const unsigned char *kp = K8 + row * D_QK;
        kn[j] = cat4(gload16(kp + 64 * lh), gload16(kp + 64 * lh + 16), gload16(kp + 64 * lh + 32),
                     gload16(kp + 64 * lh + 48));
        kr[j] = cat2z(gload16(kp + 128 + 32 * lh), gload16(kp + 128 + 32 * lh + 16));
        ksc[j] = *(const int *)(KS + row * 8 + 4 * lh);
        const unsigned char *vp = V8 + row * D_V;
        vb[j] = cat4(gload16(vp + 64 * lh), gload16(vp + 64 * lh + 16), gload16(vp + 64 * lh + 32),
                     gload16(vp + 64 * lh + 48));
        vsc[j] = *(const int *)(VSR + row * 4);
    });

    // ---- ring: step s (q rows q0 = kv0 + 128 s) into stage s % 3, 11 TDM ops by wave 0 (1-row copies past the last)
    auto issue_step = [&](int s) {
        const bool real = s < nst;
        const int ss = real ? s : nst - 1;
        const unsigned rq = real ? QS_ : 1, rd = real ? D_V : 1, rdq = real ? D_QK : 1;
        const unsigned long long q0 = (unsigned long long)(kv0 + ss * QS_);
        const unsigned st = lds0 + (unsigned)((s % NSTG) * STG_B);
        const unsigned long long qrow = bh * S + q0;
        const unsigned long long qt = q0 / 128;
        tdm_load(tdm_g0(st + O_Q, (unsigned long long)(uintptr_t)Q8 + qrow * D_QK, cnt0), tdm_g1(W0_KN, rq, 128, QS_, D_QK));
        tdm_load(tdm_g0(st + O_Q + 128, (unsigned long long)(uintptr_t)Q8 + qrow * D_QK + 128, cnt0),
                 tdm_g1(W0_KR, rq, 64, QS_, D_QK));
        tdm_load(tdm_g0(st + O_QS, (unsigned long long)(uintptr_t)QS + qrow * 8, cnt0), tdm_g1(W0_S, rq, 8, QS_, 8));
        tdm_load(tdm_g0(st + O_DO, (unsigned long long)(uintptr_t)DO8 + qrow * D_V, cnt0),
                 tdm_g1(W0_128, rq, 128, QS_, D_V));
        tdm_load(tdm_g0(st + O_DOS, (unsigned long long)(uintptr_t)DOS + qrow * 4, cnt0), tdm_g1(W0_S, rq, 4, QS_, 4));
        tdm_load(tdm_g0(st + O_DOT, (unsigned long long)(uintptr_t)DO8T + bh * D_V * S + q0, cnt0),
                 tdm_g1(W0_128, rd, 128, D_V, (unsigned long long)S));
        tdm_load(tdm_g0(st + O_DOTS, (unsigned long long)(uintptr_t)DOTS + ((bh * (S / 128) + qt) * D_V) * 4, cnt0),
                 tdm_g1(W0_S, rd, 4, D_V, 4));
        tdm_load(tdm_g0(st + O_QT, (unsigned long long)(uintptr_t)Q8T + bh * D_QK * S + q0, cnt0),
                 tdm_g1(W0_128, rdq, 128, D_QK, (unsigned long long)S));
        tdm_load(tdm_g0(st + O_QTS, (unsigned long long)(uintptr_t)QTS + ((bh * (S / 128) + qt) * D_QK) * 4, cnt0),
                 tdm_g1(W0_S, rdq, 4, D_QK, 4));
        tdm_load(tdm_g0(st + O_LSE, (unsigned long long)(uintptr_t)(LSE + qrow), cnt0), tdm_g1(W0_S, 1, 512, 1, 512));
        tdm_load(tdm_g0(st + O_DEL, (unsigned long long)(uintptr_t)(DELTA + qrow), cnt0), tdm_g1(W0_S, 1, 512, 1, 512));
    };
    issue_step(0);
    issue_step(1);

    f32x8 dv[8][NJ], dk[12][NJ];
    static_for<8>([&](auto d) { static_for<NJ>([&](auto j) { dv[(int)d][(int)j] = f32x8{0, 0, 0, 0, 0, 0, 0, 0}; }); });
    static_for<12>([&](auto d) { static_for<NJ>([&](auto j) { dk[(int)d][(int)j] = f32x8{0, 0, 0, 0, 0, 0, 0, 0}; }); });

    for (int s = 0; s < nst; ++s) {
        tensor_wait<11>();   // this wave's step s landed (step s + 1 in flight)
        bar_signal();
        bar_wait();          // step s visible; every wave is done with step s - 1
        issue_step(s + 2);   // into the stage of step s - 1
        const unsigned st = lds0 + (unsigned)((s % NSTG) * STG_B);
        const int q0 = kv0 + s * QS_;
        i32x16 pB[NJ], dsB[NJ];
        int dss[NJ] = {0, 0};
        f32x8 dsprev[NJ];
        static_for<NQT>([&](auto qq) {
            constexpr int qt = qq;
            const lds_u8 *qp = lds_ptr(st + O_Q + (unsigned)((16 * qt + lr) * XK));
            const i32x16 qn = lds_frag64(qp, 64 * lh);
            const i32x16 qr = cat2z(lds_b128(qp, 128 + 32 * lh), lds_b128(qp, 128 + 32 * lh + 16));
            const int qsc = (int)lds_b32(lds_ptr(st + O_QS), (16 * qt + lr) * 8 + 4 * lh);
            const i32x16 dop = lds_frag64(lds_ptr(st + O_DO + (unsigned)((16 * qt + lr) * X128)), 64 * lh);
            const int dsc = (int)lds_b32(lds_ptr(st + O_DOS), (16 * qt + lr) * 4);
            // lse / delta of this lane's rows q = 16 qt + 8 lh + r
            const u32x4 l0 = lds_b128(lds_ptr(st + O_LSE), (16 * qt + 8 * lh) * 4),
                        l1 = lds_b128(lds_ptr(st + O_LSE), (16 * qt + 8 * lh) * 4 + 16);
            const u32x4 d0 = lds_b128(lds_ptr(st + O_DEL), (16 * qt + 8 * lh) * 4),
                        d1 = lds_b128(lds_ptr(st + O_DEL), (16 * qt + 8 * lh) * 4 + 16);
            float nl[8], del[8];
            static_for<4>([&](auto e) {
                nl[(int)e] = -__builtin_bit_cast(float, l0[(int)e]) * LOG2E;
                nl[(int)e + 4] = -__builtin_bit_cast(float, l1[(int)e]) * LOG2E;
                del[(int)e] = __builtin_bit_cast(float, d0[(int)e]);
                del[(int)e + 4] = __builtin_bit_cast(float, d1[(int)e]);
            });
            static_for<NJ>([&](auto jj) {
                constexpr int j = jj;
                f32x8 z = {0, 0, 0, 0, 0, 0, 0, 0};
                z = wmma_mx<0, 0>(qn, kn[j], z, qsc, ksc[j]);
                const f32x8 sv = wmma_mx<1, 1>(qr, kr[j], z, qsc, ksc[j]);
                const f32x8 zz = {0, 0, 0, 0, 0, 0, 0, 0};
                const f32x8 dp = wmma_mx<0, 0>(dop, vb[j], zz, dsc, vsc[j]);
                f32x8 p, ds;
                const int kvc = kv0 + 32 * wv + 16 * j + lr;  // this lane's kv column
                static_for<8>([&](auto rr) {
                    constexpr int r = rr;
                    const int q = q0 + 16 * qt + 8 * lh + r;
                    float pv = exp2_raw(__builtin_fmaf(sv[r], c2, nl[r]));
                    pv = (q < kvc) ? 0.f : pv;
                    p[r] = pv;
                    ds[r] = pv * (dp[r] - del[r]);
                });
                const u32x2 pq = cvt8_e4m3(p, 1.0f / 256.0f);
                pB[j][2 * qt] = (int)pq[0];
                pB[j][2 * qt + 1] = (int)pq[1];
                if constexpr (qt % 2 == 0) {
                    dsprev[j] = ds;
                } else {
                    float a = 0.f;
                    static_for<8>([&](auto rr) {
                        a = __builtin_fmaxf(a, __builtin_fabsf(ds[(int)rr]));
                        a = __builtin_fmaxf(a, __builtin_fabsf(dsprev[j][(int)rr]));
                    });
                    const int e = col_exp(a);
                    const float sc = exp2i(e);
                    const u32x2 q0v = cvt8_e4m3(dsprev[j], sc), q1v = cvt8_e4m3(ds, sc);
                    dsB[j][2 * (qt - 1)] = (int)q0v[0];
                    dsB[j][2 * (qt - 1) + 1] = (int)q0v[1];
                    dsB[j][2 * qt] = (int)q1v[0];
                    dsB[j][2 * qt + 1] = (int)q1v[1];
                    dss[j] |= (e + 127) << (8 * (qt / 2));
                }
            });
        });
        // dV^T += dO^T P, dK^T += Q^T dS
        static_for<8>([&](auto dd) {
            constexpr int dt = dd;
            const i32x16 a = lds_frag64(lds_ptr(st + O_DOT + (unsigned)((16 * dt + lr) * X128)), 64 * lh);
            const int as = (int)lds_b32(lds_ptr(st + O_DOTS), (32 * (dt / 2) + lane) * 4);
            static_for<NJ>([&](auto jj) {
                constexpr int j = jj;
                if constexpr (dt % 2 == 0)
                    dv[dt][j] = wmma_mx<0, 0>(a, pB[j], dv[dt][j], as, PSCALE);
                else
                    dv[dt][j] = wmma_mx<1, 0>(a, pB[j], dv[dt][j], as, PSCALE);
            });
        });
        static_for<12>([&](auto dd) {
            constexpr int dt = dd;
            const i32x16 a = lds_frag64(lds_ptr(st + O_QT + (unsigned)((16 * dt + lr) * X128)), 64 * lh);
            const int as = (int)lds_b32(lds_ptr(st + O_QTS), (32 * (dt / 2) + lane) * 4);
            static_for<NJ>([&](auto jj) {
                constexpr int j = jj;
                if constexpr (dt % 2 == 0)
                    dk[dt][j] = wmma_mx<0, 0>(a, dsB[j], dk[dt][j], as, dss[j]);
                else
                    dk[dt][j] = wmma_mx<1, 0>(a, dsB[j], dk[dt][j], as, dss[j]);
            });
        });
    }
    // ---- epilogue: lane = kv column, rows d = 8 lh + r of each d16 tile
    static_for<NJ>([&](auto jj) {
        constexpr int j = jj;
        const int kv = kv0 + 32 * wv + 16 * j + lr;
        bf16 *dkr = DK + (long long)b * dk_sb + (long long)kv * dk_ss + (long long)h * D_QK;
        bf16 *dvr = DV + (long long)b * dv_sb + (long long)kv * dv_ss + (long long)h * D_V;
        static_for<12>([&](auto dd) {
            constexpr int dt = dd;
            bf16x8 w;
            static_for<8>([&](auto rr) { w[(int)rr] = (bf16)(dk[dt][j][(int)rr] * scale); });
            *(u32x4 *)(dkr + 16 * dt + 8 * lh) = __builtin_bit_cast(u32x4, w);
        });
        static_for<8>([&](auto dd) {
            constexpr int dt = dd;
            bf16x8 w;
            static_for<8>([&](auto rr) { w[(int)rr] = (bf16)dv[dt][j][(int)rr]; });
            *(u32x4 *)(dvr + 16 * dt + 8 * lh) = __builtin_bit_cast(u32x4, w);
        });
    });
    tensor_wait<0>();
}

// k_dq_mx: grid (S / 256, H, B), longest q block first; 4 waves, wave w owns q16 tiles i at block rows 64 i + 16 w
// (as the forward). Over the 128-kv tiles up to the diagonal (3-stage ring: K8 rows + KS, V8 rows + VSR, K8T + KTS;
// 6 TDM ops), in the forward's orientation: S^T = K Q^T, dP^T = V dO^T (C layout: lane = q column, rows = kv), P^T,
// dS^T, dS^T of kv16 tiles 2m, 2m + 1 quantized per q column (16 values + peer) into the dQ B operand (bytes 8 j + r = kv
// 16 j + 8 h + r) and its scale byte m; then dQ^T += K^T dS^T (12 d16 tiles, A = K8T rows).
namespace {
constexpr int BMQ = 256;
constexpr int QTW = 4;   // q16 tiles per wave
constexpr int NBK = 128; // kv rows per tile
constexpr int NKV = NBK / 16;
constexpr int Q_O_K = 0;                       // K8 rows 128 x 208
constexpr int Q_O_KS = Q_O_K + NBK * XK;       // 26624
constexpr int Q_O_V = Q_O_KS + NBK * 8;        // 27648: V8 rows 128 x 144
constexpr int Q_O_VS = Q_O_V + NBK * X128;     // 46080: VSR 128 x 4
constexpr int Q_O_KT = Q_O_VS + NBK * 4;       // 46592: K8T 192 x 144
constexpr int Q_O_KTS = Q_O_KT + D_QK * X128;  // 74240: KTS 192 x 4
constexpr int Q_STG = Q_O_KTS + D_QK * 4;      // 75008
constexpr int Q_NSTG = 3;
constexpr int Q_OFF_Q = Q_NSTG * Q_STG;        // 225024: Q8 rows 256 x 208
constexpr int Q_OFF_QS = Q_OFF_Q + BMQ * XK;   // 278272: QS 256 x 8
constexpr int Q_OFF_DO = Q_OFF_QS + BMQ * 8;   // 280320: DO8 rows 256 x 144
constexpr int Q_OFF_DOS = Q_OFF_DO + BMQ * X128;  // 317184: DOS 256 x 4
constexpr int Q_OFF_LSE = Q_OFF_DOS + BMQ * 4;    // 318208: LSE 256 x 4
constexpr int Q_OFF_DEL = Q_OFF_LSE + BMQ * 4;    // 319232: DELTA 256 x 4
constexpr int Q_LDS_B = Q_OFF_DEL + BMQ * 4;      // 320256
static_assert(Q_LDS_B <= 327680, "LDS");
}  // namespace

extern "C" __global__ __attribute__((amdgpu_flat_work_group_size(128, 128), amdgpu_waves_per_eu(1, 1)))
void k_dq_mx(const unsigned char *__restrict__ Q8, const unsigned char *__restrict__ QS,
             const unsigned char *__restrict__ K8, const unsigned char *__restrict__ KS,
             const unsigned char *__restrict__ V8, const unsigned char *__restrict__ VSR,
             const unsigned char *__restrict__ DO8, const unsigned char *__restrict__ DOS,
             const unsigned char *__restrict__ K8T, const unsigned char *__restrict__ KTS,
             const float *__restrict__ LSE, const float *__restrict__ DELTA, bf16 *__restrict__ DQ, float scale, int S,
             int H, int B, long long dq_sb, long long dq_ss) {
    __shared__ __attribute__((aligned(1024))) unsigned char smem[Q_LDS_B];
    lock_simd();
    const uint32_t lds0 = lds_addr(smem);
    const int tid = (int)__builtin_amdgcn_workitem_id_x();
    const int wv = wave_id(tid);
    const int lane = tid & 31, lr = lane & 15, lh = lane >> 4;
    const int cnt0 = wv == 0 ? 1 : 0;
    const int nqb = S / BMQ;
    const int qb = nqb - 1 - (int)blockIdx.x;
    const int h = (int)blockIdx.y, b = (int)blockIdx.z;
    const int q0 = qb * BMQ;
    const int n = (q0 + BMQ) / NBK;
    const unsigned long long bh = (unsigned long long)b * H + h;
    const float c2 = scale * LOG2E;

    auto issue_tile = [&](int t) {
        const bool real = t < n;
        const int tt = real ? t : n - 1;
        const unsigned rk = real ? NBK : 1, rdq = real ? D_QK : 1;
        const unsigned st = lds0 + (unsigned)((t % Q_NSTG) * Q_STG);
        const unsigned long long krow = bh * S + (unsigned long long)tt * NBK;
        tdm_load(tdm_g0(st + Q_O_K, (unsigned long long)(uintptr_t)K8 + krow * D_QK, cnt0), tdm_g1(W0_KN, rk, 128, NBK, D_QK));
        tdm_load(tdm_g0(st + Q_O_K + 128, (unsigned long long)(uintptr_t)K8 + krow * D_QK + 128, cnt0),
                 tdm_g1(W0_KR, rk, 64, NBK, D_QK));
        tdm_load(tdm_g0(st + Q_O_KS, (unsigned long long)(uintptr_t)KS + krow * 8, cnt0), tdm_g1(W0_S, rk, 8, NBK, 8));
        tdm_load(tdm_g0(st + Q_O_V, (unsigned long long)(uintptr_t)V8 + krow * D_V, cnt0),
                 tdm_g1(W0_128, rk, 128, NBK, D_V));
        tdm_load(tdm_g0(st + Q_O_VS, (unsigned long long)(uintptr_t)VSR + krow * 4, cnt0), tdm_g1(W0_S, rk, 4, NBK, 4));
        tdm_load(tdm_g0(st + Q_O_KT, (unsigned long long)(uintptr_t)K8T + bh * D_QK * S + (unsigned long long)tt * NBK, cnt0),
                 tdm_g1(W0_128, rdq, 128, D_QK, (unsigned long long)S));
        tdm_load(tdm_g0(st + Q_O_KTS, (unsigned long long)(uintptr_t)KTS + ((bh * (S / 128) + tt) * D_QK) * 4, cnt0),
                 tdm_g1(W0_S, rdq, 4, D_QK, 4));
    };
    // prologue: the block's Q8 / QS / DO8 / DOS / LSE / DELTA (waves 1..3 and 0 share the 6 ops), tiles 0, 1
    {
        const unsigned long long qrow = bh * S + q0;
        tdm_load(tdm_g0(lds0 + Q_OFF_Q, (unsigned long long)(uintptr_t)Q8 + qrow * D_QK, wv == 1 ? 1 : 0),
                 tdm_g1(W0_KN, BMQ, 128, BMQ, D_QK));
        tdm_load(tdm_g0(lds0 + Q_OFF_Q + 128, (unsigned long long)(uintptr_t)Q8 + qrow * D_QK + 128, wv == 2 ? 1 : 0),
                 tdm_g1(W0_KR, BMQ, 64, BMQ, D_QK));
        tdm_load(tdm_g0(lds0 + Q_OFF_QS, (unsigned long long)(uintptr_t)QS + qrow * 8, wv == 3 ? 1 : 0),
                 tdm_g1(W0_S, BMQ, 8, BMQ, 8));
        tdm_load(tdm_g0(lds0 + Q_OFF_DO, (unsigned long long)(uintptr_t)DO8 + qrow * D_V, wv == 1 ? 1 : 0),
                 tdm_g1(W0_128, BMQ, 128, BMQ, D_V));
        tdm_load(tdm_g0(lds0 + Q_OFF_DOS, (unsigned long long)(uintptr_t)DOS + qrow * 4, wv == 2 ? 1 : 0),
                 tdm_g1(W0_S, BMQ, 4, BMQ, 4));
        tdm_load(tdm_g0(lds0 + Q_OFF_LSE, (unsigned long long)(uintptr_t)(LSE + qrow), wv == 3 ? 1 : 0),
                 tdm_g1(W0_S, 1, 1024, 1, 1024));
        tdm_load(tdm_g0(lds0 + Q_OFF_DEL, (unsigned long long)(uintptr_t)(DELTA + qrow), wv == 3 ? 1 : 0),
                 tdm_g1(W0_S, 1, 1024, 1, 1024));
    }
    issue_tile(0);
    issue_tile(1);
    tensor_wait<0>();
    bar_signal();
    bar_wait();

    f32x8 dq[12][QTW];
    static_for<12>([&](auto d) { static_for<QTW>([&](auto i) { dq[(int)d][(int)i] = f32x8{0, 0, 0, 0, 0, 0, 0, 0}; }); });
    float nlq[QTW], delq[QTW];
    i32x16 qnb[QTW], qrb[QTW], dob[QTW];  // B operands of the wave's q tiles (the block's Q / dO, resident)
    int qscb[QTW], dscb[QTW];
    static_for<QTW>([&](auto ii) {
        constexpr int i = ii;
        const int row = 64 * i + 16 * wv + lr;
        nlq[i] = -__builtin_bit_cast(float, lds_b32(lds_ptr(lds0 + Q_OFF_LSE), row * 4)) * LOG2E;
        delq[i] = __builtin_bit_cast(float, lds_b32(lds_ptr(lds0 + Q_OFF_DEL), row * 4));
        const lds_u8 *qp = lds_ptr(lds0 + Q_OFF_Q + (unsigned)(row * XK));
        qnb[i] = lds_frag64(qp, 64 * lh);
        qrb[i] = cat2z(lds_b128(qp, 128 + 32 * lh), lds_b128(qp, 128 + 32 * lh + 16));
        qscb[i] = (int)lds_b32(lds_ptr(lds0 + Q_OFF_QS), row * 8 + 4 * lh);
        dob[i] = lds_frag64(lds_ptr(lds0 + Q_OFF_DO + (unsigned)(row * X128)), 64 * lh);
        dscb[i] = (int)lds_b32(lds_ptr(lds0 + Q_OFF_DOS), row * 4);
    });

    for (int t = 0; t < n; ++t) {
        if (t > 0) {
            tensor_wait<7>();  // tile t landed (t + 1 in flight)
            bar_signal();
            bar_wait();        // tile t visible; every wave is done with tile t - 1
        }
        issue_tile(t + 2);     // into the stage of tile t - 1
        const unsigned st = lds0 + (unsigned)((t % Q_NSTG) * Q_STG);
        const int kv0 = t * NBK;
        i32x16 dsB[QTW];
        int dss[QTW] = {0, 0, 0, 0};
        // kv16 pairs (2m, 2m + 1) = MX block m of the dQ reduction: K / V fragments of the pair, then every q tile
        static_for<NKV / 2>([&](auto mm) {
            constexpr int m = mm;
            i32x16 ka[2], kr[2], va[2];
            int ksc[2], vsc[2];
            static_for<2>([&](auto uu) {
                constexpr int u = uu;
                const int krow = 16 * (2 * m + u) + lr;
                const lds_u8 *kp = lds_ptr(st + Q_O_K + (unsigned)(krow * XK));
                ka[u] = lds_frag64(kp, 64 * lh);
                kr[u] = cat2z(lds_b128(kp, 128 + 32 * lh), lds_b128(kp, 128 + 32 * lh + 16));
                ksc[u] = (int)lds_b32(lds_ptr(st + Q_O_KS), krow * 8 + 4 * lh);
                va[u] = lds_frag64(lds_ptr(st + Q_O_V + (unsigned)(krow * X128)), 64 * lh);
                vsc[u] = (int)lds_b32(lds_ptr(st + Q_O_VS), krow * 4);
            });
            static_for<QTW>([&](auto ii) {
                constexpr int i = ii;
                const int qcol = q0 + 64 * i + 16 * wv + lr;
                f32x8 ds[2];
                static_for<2>([&](auto uu) {
                    constexpr int u = uu;
                    constexpr int j = 2 * m + u;
                    f32x8 z = {0, 0, 0, 0, 0, 0, 0, 0};
                    z = wmma_mx<0, 0>(ka[u], qnb[i], z, ksc[u], qscb[i]);
                    const f32x8 sv = wmma_mx<1, 1>(kr[u], qrb[i], z, ksc[u], qscb[i]);
                    const f32x8 zz = {0, 0, 0, 0, 0, 0, 0, 0};
                    const f32x8 dp = wmma_mx<0, 0>(va[u], dob[i], zz, vsc[u], dscb[i]);
                    static_for<8>([&](auto rr) {
                        constexpr int r = rr;
                        const int kv = kv0 + 16 * j + 8 * lh + r;
                        float pv = exp2_raw(__builtin_fmaf(sv[r], c2, nlq[i]));
                        pv = (kv > qcol) ? 0.f : pv;
                        ds[u][r] = pv * (dp[r] - delq[i]);
                    });
                });
                float a = 0.f;
                static_for<8>([&](auto rr) {
                    a = __builtin_fmaxf(a, __builtin_fabsf(ds[0][(int)rr]));
                    a = __builtin_fmaxf(a, __builtin_fabsf(ds[1][(int)rr]));
                });
                const int e = col_exp(a);
                const float sc = exp2i(e);
                const u32x2 q0v = cvt8_e4m3(ds[0], sc), q1v = cvt8_e4m3(ds[1], sc);
                dsB[i][4 * m] = (int)q0v[0];
                dsB[i][4 * m + 1] = (int)q0v[1];
                dsB[i][4 * m + 2] = (int)q1v[0];
                dsB[i][4 * m + 3] = (int)q1v[1];
                dss[i] |= (e + 127) << (8 * m);
            });
        });
        // dQ^T += K^T dS^T: A = K8T rows d, K = this tile's kv
        static_for<12>([&](auto dd) {
            constexpr int dt = dd;
            const i32x16 a = lds_frag64(lds_ptr(st + Q_O_KT + (unsigned)((16 * dt + lr) * X128)), 64 * lh);
            const int as = (int)lds_b32(lds_ptr(st + Q_O_KTS), (32 * (dt / 2) + lane) * 4);
            static_for<QTW>([&](auto ii) {
                constexpr int i = ii;
                if constexpr (dt % 2 == 0)
                    dq[dt][i] = wmma_mx<0, 0>(a, dsB[i], dq[dt][i], as, dss[i]);
                else
                    dq[dt][i] = wmma_mx<1, 0>(a, dsB[i], dq[dt][i], as, dss[i]);
            });
        });
    }
    static_for<QTW>([&](auto ii) {
        constexpr int i = ii;
        const int q = q0 + 64 * i + 16 * wv + lr;
        bf16 *dqr = DQ + (long long)b * dq_sb + (long long)q * dq_ss + (long long)h * D_QK;
        static_for<12>([&](auto dd) {
            constexpr int dt = dd;
            bf16x8 w;
            static_for<8>([&](auto rr) { w[(int)rr] = (bf16)(dq[dt][i][(int)rr] * scale); });
            *(u32x4 *)(dqr + 16 * dt + 8 * lh) = __builtin_bit_cast(u32x4, w);
        });
    });
    tensor_wait<0>();
}
