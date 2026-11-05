/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// k_mla_fwd_mx -- DeepSeek-V3 MLA attention forward on MXFP8 operands, gfx1250 (wave32). Reference kernel (rung m0):
// correctness first, the structure the optimized rungs keep.
//
//   Q8 / K8 [B, H, S, 192] E4M3 (MX-interleaved nope 128 + rope 64), QS / KS [B, H, S, 8] E8M0 (k_mxq_qk; K centred)
//   V8T [B, H, 128, S] E4M3 (kv MX-interleaved per 128), VS [B, H, S / 128, 128, 4] E8M0 (k_mxq_vt)
//   -> O [B, S, H, 128] bf16 (row strides o_sb / o_ss elements), LSE [B, H, S] fp32 (natural log, of the centred
//   logits: the host adds scale * q . kbar for the true lse)
// Causal (Sq == Skv = S, S % 256 == 0). S^T = K Q^T per kv16 x q16 tile: an MX 16x16x128 WMMA on the nope bytes and
// one on the rope bytes (upper half zero); online softmax in log2 units with the lazy rescale (O rescaled only when
// the row max of some column of the q16 tile grew by more than one nat), P = E4M3(p / 2^-7) (p <= e), O^T += V^T P^T
// as one MX WMMA per d16 tile and 128-kv tile.
//
// Work: grid (S / 256, H, B), longest q block first; 4 waves, wave w owns the q16 tiles i = 0..3 at rows
// 64 i + 16 w of the block. LDS: a 4-stage ring of 128-kv tiles (K 128 x 208 B, K scales 1 KB, V^T 128 x 144 B,
// V scales 512 B) and the Q block (256 x 208 B + 2 KB scales). Wave 0 issues the ring's TDM ops (5 per tile); the
// other waves issue the same ops with NULL descriptors, so every wave waits on the same static tensor counts.
#include "mx_common.h"

using namespace mx;

#ifndef ABL
#define ABL 0  // timing ablations (wrong outputs): 1 no exp, 2 no softmax, 3 no QK WMMA, 4 no PV WMMA, 5 no K/V reads
#endif

namespace {
constexpr int D_QK = 192, D_V = 128;
constexpr int BM = 256;  // q rows per block
constexpr int QT = 4;    // q16 tiles per wave
constexpr int NB = 128;  // kv rows per tile
constexpr int NKV = NB / 16;    // kv16 tiles per tile
constexpr int NDT = D_V / 16;   // d16 tiles of O
constexpr int XK = 208, XV = 144;  // LDS row pitches of K / Q and V^T (bytes)
constexpr int K_B = NB * XK;       // 26624
constexpr int KS_B = NB * 8;       // 1024
constexpr int V_B = D_V * XV;      // 18432
constexpr int VS_B = D_V * 4;      // 512
constexpr int STG_B = K_B + KS_B + V_B + VS_B;  // 46592
constexpr int NSTG = 4;
constexpr int Q_OFF = NSTG * STG_B;   // 186368
constexpr int QS_OFF = Q_OFF + BM * XK;  // 239616
constexpr int LDS_B = QS_OFF + BM * 8;   // 241664
static_assert(LDS_B <= 327680, "LDS");
// TDM word 0 (1-byte elements): K nope 128 B rows -> 208-B LDS rows (pad 80 B), K rope 64 B -> 208 (pad 144),
// V^T 128 B -> 144 (pad 16), scales unpadded
constexpr unsigned W0_KN = tdm_w0(0, 32, 20), W0_KR = tdm_w0(0, 16, 36), W0_V = tdm_w0(0, 32, 4), W0_S = tdm_w0(0, 0, 0);
constexpr float TAU2 = 1.4426950408889634f;  // one nat, in log2 units
constexpr int P_E8M0 = 120;                   // P stored as E4M3(p * 2^7): scale 2^-7
constexpr int PSCALE = P_E8M0 * 0x01010101;
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
    const int qb = nqb - 1 - (int)blockIdx.x;  // longest first
    const int h = (int)blockIdx.y, b = (int)blockIdx.z;
    const int q0 = qb * BM;
    const int n = (q0 + BM) / NB;  // kv tiles 0 .. n - 1
    const unsigned long long bh = (unsigned long long)b * H + h;
    const unsigned long long gq = (unsigned long long)(uintptr_t)Q8 + (bh * S + q0) * D_QK;
    const unsigned long long gqs = (unsigned long long)(uintptr_t)QS + (bh * S + q0) * 8;
    const unsigned long long gk = (unsigned long long)(uintptr_t)K8 + bh * S * D_QK;
    const unsigned long long gks = (unsigned long long)(uintptr_t)KS + bh * S * 8;
    const unsigned long long gv = (unsigned long long)(uintptr_t)V8T + bh * D_V * S;
    const unsigned long long gvs = (unsigned long long)(uintptr_t)VS + bh * (unsigned long long)(S / NB) * VS_B;

    // ring slot t: tile t into stage t % 4, 5 TDM ops by wave 0 (NULL descriptors on waves 1..3). Past the last tile
    // (t >= n) wave 0 issues 1-row copies of tile n - 1 into the (free) stage instead, so it has 5 real ops in every
    // slot and its tensor waits stay static.
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
    // prologue: Q (waves 1..3 issue one op each: nope, rope, scales), slots 0..2 (wave 0)
    tdm_load(tdm_g0(lds0 + Q_OFF, gq, wv == 1 ? 1 : 0), tdm_g1(W0_KN, BM, 128, BM, D_QK));
    tdm_load(tdm_g0(lds0 + Q_OFF + 128, gq + 128, wv == 2 ? 1 : 0), tdm_g1(W0_KR, BM, 64, BM, D_QK));
    tdm_load(tdm_g0(lds0 + QS_OFF, gqs, wv == 3 ? 1 : 0), tdm_g1(W0_S, BM, 8, BM, 8));
    issue_slot(0);
    issue_slot(1);
    issue_slot(2);
    tensor_wait<0>();
    bar_signal();
    bar_wait();

    // Q B operands resident: q tile i = block rows 64 i + 16 w + lr; nope 64 B at 64 lh, rope 32 B at 128 + 32 lh
    i32x16 qn[QT], qr[QT];
    int qsc[QT];
    static_for<QT>([&](auto ii) {
        constexpr int i = ii;
        const int row = 64 * i + 16 * wv + lr;
        const lds_u8 *p = lds_ptr(lds0 + Q_OFF + (unsigned)(row * XK));
        u32x4 a0 = lds_b128(p, 64 * lh), a1 = lds_b128(p, 64 * lh + 16), a2 = lds_b128(p, 64 * lh + 32),
              a3 = lds_b128(p, 64 * lh + 48);
        u32x4 r0 = lds_b128(p, 128 + 32 * lh), r1 = lds_b128(p, 128 + 32 * lh + 16);
        qn[i] = i32x16{(int)a0[0], (int)a0[1], (int)a0[2], (int)a0[3], (int)a1[0], (int)a1[1], (int)a1[2], (int)a1[3],
                       (int)a2[0], (int)a2[1], (int)a2[2], (int)a2[3], (int)a3[0], (int)a3[1], (int)a3[2], (int)a3[3]};
        qr[i] = i32x16{(int)r0[0], (int)r0[1], (int)r0[2], (int)r0[3], (int)r1[0], (int)r1[1], (int)r1[2], (int)r1[3],
                       0, 0, 0, 0, 0, 0, 0, 0};
        qsc[i] = (int)lds_b32(lds_ptr(lds0 + QS_OFF + (unsigned)(row * 8)), 4 * lh);
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

    for (int t = 0; t < n; ++t) {
        // slot t + 3 into the stage of tile t - 1, freed by B_t (every wave read tile t - 1 before arriving there)
        issue_slot(t + 3);
        const unsigned st = lds0 + (unsigned)((t & (NSTG - 1)) * STG_B);
        const lds_u8 *kp = lds_ptr(st);
        const lds_u8 *ksp = lds_ptr(st + K_B);
        const lds_u8 *vp = lds_ptr(st + K_B + KS_B);
        const lds_u8 *vsp = lds_ptr(st + K_B + KS_B + V_B);
        const int kv0 = t * NB;
        // this tile's operands: K (A, 8 kv16 tiles), V^T (A, 8 d16 tiles), their scales
        i32x16 kn[NKV], kr[NKV];
        int ksc[NKV];
        static_for<NKV>([&](auto jj) {
            constexpr int j = jj;
            const int row = 16 * j + lr;
#if ABL == 5
            u32x4 a0 = {(unsigned)row, 0u, 0u, 0u}, a1 = a0, a2 = a0, a3 = a0, r0 = a0, r1 = a0;
#else
            u32x4 a0 = lds_b128(kp, row * XK + 64 * lh), a1 = lds_b128(kp, row * XK + 64 * lh + 16),
                  a2 = lds_b128(kp, row * XK + 64 * lh + 32), a3 = lds_b128(kp, row * XK + 64 * lh + 48);
            u32x4 r0 = lds_b128(kp, row * XK + 128 + 32 * lh), r1 = lds_b128(kp, row * XK + 128 + 32 * lh + 16);
#endif
            kn[j] = i32x16{(int)a0[0], (int)a0[1], (int)a0[2], (int)a0[3], (int)a1[0], (int)a1[1], (int)a1[2],
                           (int)a1[3], (int)a2[0], (int)a2[1], (int)a2[2], (int)a2[3], (int)a3[0], (int)a3[1],
                           (int)a3[2], (int)a3[3]};
            kr[j] = i32x16{(int)r0[0], (int)r0[1], (int)r0[2], (int)r0[3], (int)r1[0], (int)r1[1], (int)r1[2],
                           (int)r1[3], 0, 0, 0, 0, 0, 0, 0, 0};
            ksc[j] = (int)lds_b32(ksp, row * 8 + 4 * lh);
        });
        i32x16 vt[NDT];
        int vsc[NDT / 2];
        static_for<NDT>([&](auto dd) {
            constexpr int dt = dd;
            const int row = 16 * dt + lr;
#if ABL == 5
            u32x4 a0 = {(unsigned)row, 1u, 0u, 0u}, a1 = a0, a2 = a0, a3 = a0;
#else
            u32x4 a0 = lds_b128(vp, row * XV + 64 * lh), a1 = lds_b128(vp, row * XV + 64 * lh + 16),
                  a2 = lds_b128(vp, row * XV + 64 * lh + 32), a3 = lds_b128(vp, row * XV + 64 * lh + 48);
#endif
            vt[dt] = i32x16{(int)a0[0], (int)a0[1], (int)a0[2], (int)a0[3], (int)a1[0], (int)a1[1], (int)a1[2],
                            (int)a1[3], (int)a2[0], (int)a2[1], (int)a2[2], (int)a2[3], (int)a3[0], (int)a3[1],
                            (int)a3[2], (int)a3[3]};
        });
        static_for<NDT / 2>([&](auto pp) { vsc[(int)pp] = (int)lds_b32(vsp, (32 * (int)pp + lane) * 4); });
        // every LDS read of tile t is done: wait until this wave's slot t + 1 landed (slots t + 2, t + 3 = 10 ops may
        // be in flight), arrive at B_{t+1}
        tensor_wait<10>();
        bar_signal();

        static_for<QT>([&](auto ii) {
            constexpr int i = ii;
            f32x8 s[NKV];
            static_for<NKV>([&](auto jj) {
                constexpr int j = jj;
                f32x8 z = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#if ABL == 3
                z[0] = (float)kn[j][0] * (float)qn[i][1];
                s[j] = z;
#else
                z = wmma_mx<0, 0>(kn[j], qn[i], z, ksc[j], qsc[i]);
                s[j] = wmma_mx<1, 1>(kr[j], qr[i], z, ksc[j], qsc[i]);
#endif
            });
            // causal mask: kv = kv0 + 16 j + 8 lh + r, q = q0 + 64 i + 16 w + lr; masked when kv > q
            const int qrow = q0 + 64 * i + 16 * wv + lr;
            if (kv0 + NB - 1 > q0 + 64 * i + 16 * wv) {  // the tile reaches past this q16 tile's first row
                static_for<NKV>([&](auto jj) {
                    constexpr int j = jj;
                    static_for<8>([&](auto rr) {
                        constexpr int r = rr;
                        const int kv = kv0 + 16 * j + 8 * lh + r;
                        s[j][r] = kv > qrow ? -__builtin_inff() : s[j][r];
                    });
                });
            }
            float mx = -__builtin_inff();
            static_for<NKV>([&](auto jj) { static_for<8>([&](auto rr) { mx = __builtin_fmaxf(mx, s[(int)jj][(int)rr]); }); });
            const float lm = __builtin_fmaxf(mx, peer(mx));
            const float need = __builtin_fmaf(lm, c2, -m[i]);
            const bool resc = __builtin_amdgcn_ballot_w32(need > TAU2) != 0u;
            const float mfull = __builtin_fmaxf(m[i], lm * c2);
            const float mn = resc ? mfull : m[i];
            const float corr = exp2_raw(m[i] - mn);
            float ls = 0.f;
            i32x16 pb;
            static_for<NKV>([&](auto jj) {
                constexpr int j = jj;
                f32x8 p;
                static_for<8>([&](auto rr) {
                    constexpr int r = rr;
#if ABL == 1
                    p[r] = __builtin_fmaf(s[j][r], c2, -mn);
#elif ABL == 2
                    p[r] = s[j][r];
#else
                    p[r] = exp2_raw(__builtin_fmaf(s[j][r], c2, -mn));
#endif
                    ls += p[r];
                });
                const u32x2 q = cvt8_e4m3(p, 1.0f / 128.0f);
                pb[2 * j] = (int)q[0];
                pb[2 * j + 1] = (int)q[1];
            });
            const float tot = ls + peer(ls);
            l[i] = __builtin_fmaf(corr, l[i], tot);
            m[i] = mn;
            if (resc) {
                static_for<NDT>([&](auto dd) { o[i][(int)dd] = o[i][(int)dd] * corr; });
            }
            static_for<NDT>([&](auto dd) {
                constexpr int dt = dd;
#if ABL == 4
                o[i][dt][0] += (float)pb[dt] + (float)vt[dt][0];
#else
                if constexpr (dt % 2 == 0)
                    o[i][dt] = wmma_mx<0, 0>(vt[dt], pb, o[i][dt], vsc[dt / 2], PSCALE);
                else
                    o[i][dt] = wmma_mx<1, 0>(vt[dt], pb, o[i][dt], vsc[dt / 2], PSCALE);
#endif
            });
        });
        bar_wait();  // B_{t+1}: every wave is done reading tile t, and wave 0's slot t + 1 landed
    }

    // epilogue: out = O / l (bf16), lse = (m + log2 l) ln 2
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
