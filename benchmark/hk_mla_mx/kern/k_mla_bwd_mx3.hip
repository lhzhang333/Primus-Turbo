/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// k_mla_bwd_mx3.hip -- rung b3 = b1 + XCD-local workgroup order (the blocks of one (b, h) on one XCD: dispatch sends
// linear id L to XCD L % 8, so its L2 serves their shared operand stream) + the TDM ops of a step / tile spread over
// the four waves (LLVM waits for tensorcnt <= 10 before every TDM op; b1 put all 11 / 7 on wave 0, waves 1..3 NULL).
// b1's notes follow.
// k_mla_bwd_mx1.hip -- rung b1 of the MXFP8 MLA backward (k_mla_bwd_mx.hip = b0: the operand layouts, the math and the
// quantization of P / dS are documented there). Changes:
//  - operand recipe m4 (the forward rung m4's): Q8 rows quantized from q * scale * log2(e), so S' = Q8 . K8 is in log2
//    units; Q / K row scales in layout R2, and the rope WMMA runs over d 64..191 on 24-VGPR [nope | rope] fragments
//    read as two overlapping operands (E8M0 0 silences d 64..127): no zero halves;
//  - LSE2 = the forward's lse of the centred logits in log2 units (lse * log2(e)); the S WMMA's accumulator starts at
//    LSE2 and the dP WMMA's at DELTA, both negated by the C modifier: P = exp2(S' - lse2) and dP - delta come out of the
//    WMMAs with no VALU;
//  - k_dkdv_mx: 48 kv rows per wave (3 kv16 tiles: every Q / dO / Q^T / dO^T fragment read from LDS feeds 3 WMMAs, b0:
//    2), 192 per workgroup (grid ceil(S / 192): rows >= S are computed on clamped loads and not stored); the causal
//    mask only on the steps that cross the diagonal;
//  - dS exponents from the float bits (e8m0_byte), LDS addresses precomputed per step (no VALU address math between
//    the WMMAs), expert-mode friendly.
#include "mx_common.h"

using namespace mx;

// software prefetch of the next fragment set (registers permitting): q16 tiles (dkdv), kv16 pairs (dq), dV / dK / dQ
// A fragments
#ifndef PF_Q
#define PF_Q 1
#endif
#ifndef PF_KV
#define PF_KV 1
#endif
#ifndef DQ_PRO_PARTIAL
#define DQ_PRO_PARTIAL 1
#endif
#ifndef DQ_SGB
#define DQ_SGB 0  // 1: sched_group_barrier windows in the pair phase (measured slower: 0.703 vs 0.631 ms)
#endif
#ifndef OSTAGE
#define OSTAGE 1  // epilogues through LDS and TDM stores
#endif
#ifndef DQ_QB
#define DQ_QB 1  // dq: mx_exp / exp2i dS quantization (the e8m0-bits form spills 468 B in this kernel)
#endif
#ifndef PF_A
#define PF_A 2
#endif
#ifndef QPIPE
#define QPIPE 0  // dkdv: software-pipelined q16 tiles (S / dP WMMAs of tile qt + 1 beside the VALU of tile qt)
#endif
#ifndef KEEP_WAR
// keep a fragment the previous WMMAs read alive past the next loads (no register reuse while they may be in flight):
// bit 0 the dV / dK A fragments (dkdv 0.979 -> 0.950 ms), bit 1 QPIPE's q16 tiles (58 more VGPRs: spills)
#define KEEP_WAR 1
#endif
#ifndef PF_AD
#define PF_AD 2  // PF_A 2: dV / dK A fragments loaded PF_AD groups ahead (PF_AD + 2 buffers)
#endif
#ifndef QFENCE2
#define QFENCE2 1  // 0: no fence after a q16 tile's math (its P / dS conversion may sink under the next tile's work)
#endif
#ifndef QSGB
#define QSGB 0  // QPIPE + sched_group_barrier windows per iteration: [W, QSGB_L x L, QSGB_T x T, QSGB_V x V] x 6
#endif
#ifndef QSGB_L
#define QSGB_L 3
#endif
#ifndef QSGB_T
#define QSGB_T 3
#endif
#ifndef QSGB_V
#define QSGB_V 6
#endif

namespace {
constexpr int D_QK = 192, D_V = 128;
constexpr int XK = 208, X128 = 144;
constexpr unsigned W0_KN = tdm_w0(0, 32, 20), W0_KR = tdm_w0(0, 16, 36), W0_128 = tdm_w0(0, 32, 4),
                   W0_S = tdm_w0(0, 0, 0);
constexpr int PSCALE = 119 * 0x01010101;  // P stored as E4M3(P 2^8)
constexpr int NEG_C = 1;

typedef int32_t i32x32 __attribute__((ext_vector_type(32)));

MX_INLINE i32x16 cat4(u32x4 a0, u32x4 a1, u32x4 a2, u32x4 a3) {
    return i32x16{(int)a0[0], (int)a0[1], (int)a0[2], (int)a0[3], (int)a1[0], (int)a1[1], (int)a1[2], (int)a1[3],
                  (int)a2[0], (int)a2[1], (int)a2[2], (int)a2[3], (int)a3[0], (int)a3[1], (int)a3[2], (int)a3[3]};
}
MX_INLINE i32x32 frag24(u32x4 a0, u32x4 a1, u32x4 a2, u32x4 a3, u32x4 r0, u32x4 r1) {
    const i32x16 n = cat4(a0, a1, a2, a3);
    const i32x8 r8 = {(int)r0[0], (int)r0[1], (int)r0[2], (int)r0[3], (int)r1[0], (int)r1[1], (int)r1[2], (int)r1[3]};
    const i32x16 r = __builtin_shufflevector(r8, r8, 0, 1, 2, 3, 4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1);
    return __builtin_shufflevector(n, r, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                   22, 23, -1, -1, -1, -1, -1, -1, -1, -1);
}
MX_INLINE i32x16 lo16(const i32x32 &x) {
    return __builtin_shufflevector(x, x, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
}
MX_INLINE i32x16 mid16(const i32x32 &x) {
    return __builtin_shufflevector(x, x, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23);
}
MX_INLINE u32x4 gload16(const unsigned char *p) { return *(const u32x4 *)p; }
MX_INLINE i32x16 lds_frag64(const lds_u8 *p, int off) {
    return cat4(lds_b128(p, off), lds_b128(p, off + 16), lds_b128(p, off + 32), lds_b128(p, off + 48));
}
MX_INLINE uint32_t pin(uint32_t x) {
    asm volatile("" : "+v"(x));
    return x;
}
// this lane's id, read again at each call (volatile: not hoisted), so per-lane LDS offsets are recomputed where the
// addresses are formed instead of living in v0..v255 across the loop (where the DS address and WMMA scale operands
// must be) and being spilled
MX_INLINE int lane_now() {
    int l;
    asm volatile("v_mbcnt_lo_u32_b32 %0, -1, 0" : "=v"(l));
    return l;
}
MX_INLINE f32x8 f8(u32x4 a, u32x4 b) {
    return f32x8{as_f(a[0]), as_f(a[1]), as_f(a[2]), as_f(a[3]), as_f(b[0]), as_f(b[1]), as_f(b[2]), as_f(b[3])};
}
// MX-quantize two 8-value column halves x0, x1 (16 values of this lane + 16 of its peer: one block of 32): the E8M0
// byte and the two E4M3 pairs
struct DsQ {
    int byte;
    u32x2 a, b;
};
MX_INLINE DsQ quant_pair(const f32x8 &x0, const f32x8 &x1) {
    float a = 0.f;
    static_for<8>([&](auto rr) {
        a = __builtin_fmaxf(a, __builtin_fmaxf(__builtin_fabsf(x0[(int)rr]), __builtin_fabsf(x1[(int)rr])));
    });
    a = __builtin_fmaxf(a, peer(a));
    const int byte = e8m0_byte(a);
    const float sc = e8m0_scale(byte);
    return {byte, cvt8_e4m3(x0, sc), cvt8_e4m3(x1, sc)};
}
// XCD-local order: the k-th workgroup of XCD c (k = L >> 3, L the linear id) takes block slot k % nblk of
// bh = c + 8 (k / nblk); B H % 8 != 0: the plain order
struct BlockOf {
    int slot, h, b;
};
MX_INLINE BlockOf xcd_block(int nblk, int H, int B) {
    if ((B * H) % 8 == 0) {
        const int L = (int)blockIdx.x + nblk * ((int)blockIdx.y + H * (int)blockIdx.z);
        const int k = L >> 3;
        const int bhx = (L & 7) + 8 * (k / nblk);
        return {k % nblk, bhx % H, bhx / H};
    }
    return {(int)blockIdx.x, (int)blockIdx.y, (int)blockIdx.z};
}
// one TDM op of a ring: LDS offset in the stage, global base and bytes per row (step) / tile, group 1
struct TdmOp {
    unsigned lds;
    unsigned long long base, mult;
    i32x8 g1;
    int on;
};
}  // namespace

// =====================================================================================================================
// k_dkdv_mx: grid (ceil(S / 192), H, B), 4 waves; wave w owns kv rows kv0 + 48 w .. + 47 (kv16 tiles j = 0..2), whose
// K8 / V8 rows are the resident B operands of S = Q K^T and dP = dO V^T. Steps of 128 q rows from the 128-row block
// holding kv0 to S - 128 through a 3-stage TDM ring (wave 0 issues 11 ops per step).
// =====================================================================================================================
namespace {
constexpr int QS_ = 128;
constexpr int NQT = QS_ / 16;
#ifndef NJ_DKDV
#define NJ_DKDV 2
#endif
constexpr int NJ = NJ_DKDV;  // kv16 tiles per wave
constexpr int KBW = 16 * NJ;  // kv rows per wave
constexpr int KB = 4 * KBW;   // per workgroup
constexpr int O_Q = 0;
constexpr int O_QS = O_Q + QS_ * XK;
constexpr int O_DO = O_QS + QS_ * 8;
constexpr int O_DOS = O_DO + QS_ * X128;
constexpr int O_DOT = O_DOS + QS_ * 4;
constexpr int O_DOTS = O_DOT + D_V * X128;
constexpr int O_QT = O_DOTS + D_V * 4;
constexpr int O_QTS = O_QT + D_QK * X128;
constexpr int O_LSE = O_QTS + D_QK * 4;
constexpr int O_DEL = O_LSE + QS_ * 4;
constexpr int STG_B = O_DEL + QS_ * 4;
constexpr int NSTG = 3;
constexpr int LDS_B = NSTG * STG_B;
static_assert(LDS_B <= 327680, "LDS");

// per-step LDS addresses (stage base + this lane's offset); fragments add immediate offsets
struct StepAddr {
    uint32_t qn, qr, qs, dO, dos, lse, del, dot, dots, qt, qts;
};
// one q16 tile's A operands and C tuples
struct QTile {
    i32x32 qx;
    i32x16 dox;
    f32x8 lt, dt;
    int qsc, dsc;
};
template <int QT>
MX_INLINE QTile load_qtile(const StepAddr &a) {
    const lds_u8 *pqn = lds_ptr(a.qn), *pqr = lds_ptr(a.qr), *pdo = lds_ptr(a.dO);
    constexpr int oq = QT * 16 * XK, od = QT * 16 * X128;
    QTile t;
    t.qx = frag24(lds_b128(pqn, oq), lds_b128(pqn, oq + 16), lds_b128(pqn, oq + 32), lds_b128(pqn, oq + 48),
                  lds_b128(pqr, oq), lds_b128(pqr, oq + 16));
    t.dox = cat4(lds_b128(pdo, od), lds_b128(pdo, od + 16), lds_b128(pdo, od + 32), lds_b128(pdo, od + 48));
#if PROBE_SC  // timing probe only (wrong numerics): tile 0's scales for every tile
    if constexpr (QT > 0) {
        t.qsc = 0x7f7f7f7f;
        t.dsc = 0x7f7f7f7f;
    } else
#endif
    {
        t.qsc = (int)lds_b32(lds_ptr(a.qs), QT * 16 * 8);
        t.dsc = (int)lds_b32(lds_ptr(a.dos), QT * 16 * 4);
    }
#if PROBE_LTDT  // timing probe only (wrong numerics): lse / delta tuples of tile 0 only
    if constexpr (QT > 0) {
        t.lt = f32x8{1, 1, 1, 1, 1, 1, 1, 1};
        t.dt = f32x8{1, 1, 1, 1, 1, 1, 1, 1};
    } else
#endif
    {
        t.lt = f8(lds_b128(lds_ptr(a.lse), QT * 64), lds_b128(lds_ptr(a.lse), QT * 64 + 16));
        t.dt = f8(lds_b128(lds_ptr(a.del), QT * 64), lds_b128(lds_ptr(a.del), QT * 64 + 16));
    }
    return t;
}
struct AFrag {
    i32x16 x;
    int s;
};
template <int D, int OFF>
MX_INLINE AFrag load_afrag(uint32_t base, uint32_t sbase) {
    const lds_u8 *p = lds_ptr(base);
    constexpr int o = D * 16 * X128;
    return {cat4(lds_b128(p, o), lds_b128(p, o + 16), lds_b128(p, o + 32), lds_b128(p, o + 48)),
            (int)lds_b32(lds_ptr(sbase), (D / 2) * 128)};
}
MX_INLINE void fence() { __builtin_amdgcn_sched_barrier(0); }
// an empty use: x stays in its registers up to here, so loads issued before it cannot be given those registers while
// WMMAs that read x may still be in flight (the register allocator does not model the ~37-cycle write-after-read wait)
template <typename T>
MX_INLINE void keep(const T &x) {
    asm volatile("" ::"v"(x));
}
MX_INLINE void keep_qtile(const QTile &t) {
    keep(t.qx);
    keep(t.dox);
    keep(t.lt);
    keep(t.dt);
    keep(t.qsc);
    keep(t.dsc);
}
}  // namespace

extern "C" __global__ __attribute__((amdgpu_flat_work_group_size(128, 128), amdgpu_waves_per_eu(1, 1)))
void k_dkdv_mx(const unsigned char *__restrict__ Q8, const unsigned char *__restrict__ QS,
               const unsigned char *__restrict__ K8, const unsigned char *__restrict__ KS,
               const unsigned char *__restrict__ V8, const unsigned char *__restrict__ VSR,
               const unsigned char *__restrict__ DO8, const unsigned char *__restrict__ DOS,
               const unsigned char *__restrict__ DO8T, const unsigned char *__restrict__ DOTS,
               const unsigned char *__restrict__ Q8T, const unsigned char *__restrict__ QTS,
               const float *__restrict__ LSE2, const float *__restrict__ DELTA, bf16 *__restrict__ DK,
               bf16 *__restrict__ DV, float scale, int S, int H, int B, long long dk_sb, long long dk_ss,
               long long dv_sb, long long dv_ss) {
    __shared__ __attribute__((aligned(1024))) unsigned char smem[LDS_B];
    lock_simd();
    const uint32_t lds0 = lds_addr(smem);
    const int tid = (int)__builtin_amdgcn_workitem_id_x();
    const int wv = wave_id(tid);
    const int lane = tid & 31, lr = lane & 15, lh = lane >> 4;
    const int cnt0 = wv == 0 ? 1 : 0;
    const BlockOf blk = xcd_block((S + KB - 1) / KB, H, B);
    const int kb = blk.slot, h = blk.h, b = blk.b;
    const int kv0 = kb * KB;
    const int qs0 = (kv0 / QS_) * QS_;  // the first step: the 128-row q block holding kv0
    const int nst = (S - qs0) / QS_;
    const unsigned long long bh = (unsigned long long)b * H + h;
    const int kvw = kv0 + KBW * wv;  // this wave's first kv row

    // ---- resident B operands: the wave's K / V rows (rows >= S clamped: computed, never stored)
    i32x32 kx[NJ];
    i32x16 vb[NJ];
    int ksc[NJ], vsc[NJ];
    static_for<NJ>([&](auto jj) {
        constexpr int j = jj;
        const int kvr = kvw + 16 * j + lr;
        const unsigned long long row = bh * S + (unsigned long long)(kvr < S ? kvr : S - 1);
        const unsigned char *kp = K8 + row * D_QK;
        kx[j] = frag24(gload16(kp + 64 * lh), gload16(kp + 64 * lh + 16), gload16(kp + 64 * lh + 32),
                       gload16(kp + 64 * lh + 48), gload16(kp + 128 + 32 * lh), gload16(kp + 128 + 32 * lh + 16));
        ksc[j] = *(const int *)(KS + row * 8 + 4 * lh);
        const unsigned char *vp = V8 + row * D_V;
        vb[j] = cat4(gload16(vp + 64 * lh), gload16(vp + 64 * lh + 16), gload16(vp + 64 * lh + 32),
                     gload16(vp + 64 * lh + 48));
        vsc[j] = *(const int *)(VSR + row * 4);
    });

    // the step's 11 TDM ops, 3 per wave: [Q^T, QS, LSE], [Q nope, dO scales, delta], [dO, Q rope, -], [dO^T, dO^T
    // scales, Q^T scales] (global address = base + q0 * mult)
    auto dk_op = [&](int k) -> TdmOp {
        const unsigned long long bS = bh * (unsigned long long)S, bS128 = bh * (unsigned long long)(S / 128);
        switch (k) {
        case 0: return {O_Q, (unsigned long long)(uintptr_t)Q8 + bS * D_QK, D_QK, tdm_g1(W0_KN, QS_, 128, QS_, D_QK), 1};
        case 1: return {O_Q + 128, (unsigned long long)(uintptr_t)Q8 + bS * D_QK + 128, D_QK, tdm_g1(W0_KR, QS_, 64, QS_, D_QK), 1};
        case 2: return {O_QS, (unsigned long long)(uintptr_t)QS + bS * 8, 8, tdm_g1(W0_S, QS_, 8, QS_, 8), 1};
        case 3: return {O_DO, (unsigned long long)(uintptr_t)DO8 + bS * D_V, D_V, tdm_g1(W0_128, QS_, 128, QS_, D_V), 1};
        case 4: return {O_DOS, (unsigned long long)(uintptr_t)DOS + bS * 4, 4, tdm_g1(W0_S, QS_, 4, QS_, 4), 1};
        case 5: return {O_DOT, (unsigned long long)(uintptr_t)DO8T + bh * D_V * S, 1, tdm_g1(W0_128, D_V, 128, D_V, (unsigned long long)S), 1};
        case 6: return {O_DOTS, (unsigned long long)(uintptr_t)DOTS + bS128 * D_V * 4, 4, tdm_g1(W0_S, D_V, 4, D_V, 4), 1};
        case 7: return {O_QT, (unsigned long long)(uintptr_t)Q8T + bh * D_QK * S, 1, tdm_g1(W0_128, D_QK, 128, D_QK, (unsigned long long)S), 1};
        case 8: return {O_QTS, (unsigned long long)(uintptr_t)QTS + bS128 * D_QK * 4, 6, tdm_g1(W0_S, D_QK, 4, D_QK, 4), 1};
        case 9: return {O_LSE, (unsigned long long)(uintptr_t)(LSE2 + bS), 4, tdm_g1(W0_S, 1, 512, 1, 512), 1};
        case 10: return {O_DEL, (unsigned long long)(uintptr_t)(DELTA + bS), 4, tdm_g1(W0_S, 1, 512, 1, 512), 1};
        default: return {0, 0, 0, tdm_g1(W0_S, 1, 4, 1, 4), 0};
        }
    };
    constexpr int TDM_PER_STEP = 3;
    const TdmOp op0 = dk_op(wv == 0 ? 7 : wv == 1 ? 0 : wv == 2 ? 3 : 5);
    const TdmOp op1 = dk_op(wv == 0 ? 2 : wv == 1 ? 4 : wv == 2 ? 1 : 6);
    const TdmOp op2 = dk_op(wv == 0 ? 9 : wv == 1 ? 10 : wv == 2 ? 11 : 8);
    auto issue_step = [&](int s) {
        const bool real = s < nst;
        const unsigned long long q0 = (unsigned long long)(qs0 + (real ? s : nst - 1) * QS_);
        const unsigned st = lds0 + (unsigned)((s % NSTG) * STG_B);
        tdm_load(tdm_g0(st + op0.lds, op0.base + q0 * op0.mult, real ? op0.on : 0), op0.g1);
        tdm_load(tdm_g0(st + op1.lds, op1.base + q0 * op1.mult, real ? op1.on : 0), op1.g1);
        tdm_load(tdm_g0(st + op2.lds, op2.base + q0 * op2.mult, real ? op2.on : 0), op2.g1);
    };
    issue_step(0);
    issue_step(1);

    f32x8 dv[8][NJ], dk[12][NJ];
    static_for<8>([&](auto d) { static_for<NJ>([&](auto j) { dv[(int)d][(int)j] = f32x8{0, 0, 0, 0, 0, 0, 0, 0}; }); });
    static_for<12>([&](auto d) { static_for<NJ>([&](auto j) { dk[(int)d][(int)j] = f32x8{0, 0, 0, 0, 0, 0, 0, 0}; }); });


    // one step: the 8 q16 tiles (fragments of tile qt + 1 loaded while tile qt computes), then dV / dK (A fragment of
    // d16 tile d + 1 loaded while d computes)
    auto step = [&](const StepAddr &a, int q0, auto diag_c) __attribute__((always_inline)) {
        constexpr bool DIAG = decltype(diag_c)::value;
        // the causal mask as constant-vs-threshold compares (no per-element index registers): q < kv iff
        // 16 qt + r < thr[j] = kv - q0 - 8 lh
        int thr[NJ];
        static_for<NJ>([&](auto jj) { thr[(int)jj] = kvw + 16 * (int)jj + lr - q0 - 8 * lh; });
        i32x16 pB[NJ], dsB[NJ];
        int dss[NJ] = {};
        f32x8 dsprev[NJ];
#if QPIPE
        // iteration qt: the S / dP WMMAs of tile qt + 1 (fragments loaded an iteration earlier), the fragment loads of
        // tile qt + 2 (into the buffer tile qt's WMMAs read, issued an iteration ago) and the softmax / dS VALU of tile
        // qt (accumulators of the previous iteration): three independent streams for the scheduler to interleave (b3:
        // each tile's exps waited for its own WMMAs, the loads went out in bursts)
        constexpr int NFB = QPIPE == 2 ? 3 : 2;  // fragment buffers: QPIPE 2 loads two tiles ahead
        QTile fb[NFB];
        f32x8 sacc[2][NJ], pacc[2][NJ];
        auto sdp = [&](const QTile &t, f32x8 (&sv)[NJ], f32x8 (&dp)[NJ]) __attribute__((always_inline)) {
            static_for<NJ>([&](auto jj) {
                constexpr int j = jj;
                const f32x8 s0 = wmma_mxc<0, 0, NEG_C>(lo16(t.qx), lo16(kx[j]), t.lt, t.qsc, ksc[j]);
                sv[j] = wmma_mx<1, 1>(mid16(t.qx), mid16(kx[j]), s0, t.qsc, ksc[j]);
                dp[j] = wmma_mxc<0, 0, NEG_C>(t.dox, vb[j], t.dt, t.dsc, vsc[j]);
            });
        };
        fb[0] = load_qtile<0>(a);
        fb[1] = load_qtile<1>(a);
        if constexpr (NFB == 3) fb[2 % NFB] = load_qtile<2>(a);
        sdp(fb[0], sacc[0], pacc[0]);
        fence();
        static_for<NQT>([&](auto qq) {
            constexpr int qt = qq;
            if constexpr (qt + 1 < NQT) sdp(fb[(qt + 1) % NFB], sacc[(qt + 1) % 2], pacc[(qt + 1) % 2]);
            if constexpr (qt + NFB < NQT) {
#if KEEP_WAR & 2
                const QTile old = fb[qt % NFB];  // tile qt: read by the previous iteration's WMMAs
                fb[qt % NFB] = load_qtile<(qt + NFB < NQT ? qt + NFB : 0)>(a);
                keep_qtile(old);
#else
                fb[qt % NFB] = load_qtile<(qt + NFB < NQT ? qt + NFB : 0)>(a);
#endif
            }
            static_for<NJ>([&](auto jj) {
                constexpr int j = jj;
                const f32x8 &sv = sacc[qt % 2][j], &dp = pacc[qt % 2][j];
                f32x8 p, ds;
                static_for<8>([&](auto rr) {
                    constexpr int r = rr;
                    float pv = exp2_raw(sv[r]);
                    if constexpr (DIAG) pv = (16 * qt + r < thr[j]) ? 0.f : pv;  // q < kv
                    p[r] = pv;
                    ds[r] = pv * dp[r];
                });
                const u32x2 pq = cvt8_e4m3(p, 1.0f / 256.0f);
                pB[j][2 * qt] = (int)pq[0];
                pB[j][2 * qt + 1] = (int)pq[1];
                if constexpr (qt % 2 == 0) {
                    dsprev[j] = ds;
                } else {
                    const DsQ z = quant_pair(dsprev[j], ds);
                    dsB[j][2 * (qt - 1)] = (int)z.a[0];
                    dsB[j][2 * (qt - 1) + 1] = (int)z.a[1];
                    dsB[j][2 * qt] = (int)z.b[0];
                    dsB[j][2 * qt + 1] = (int)z.b[1];
                    dss[j] |= z.byte << (8 * (qt / 2));
                }
            });
#if QSGB
            static_for<6>([&](auto) {
                __builtin_amdgcn_sched_group_barrier(0x8, 1, 0);
                __builtin_amdgcn_sched_group_barrier(0x100, QSGB_L, 0);
                __builtin_amdgcn_sched_group_barrier(0x400, QSGB_T, 0);
                __builtin_amdgcn_sched_group_barrier(0x2, QSGB_V, 0);
            });
#endif
            fence();
        });
#else
#if PF_Q
        QTile cur = load_qtile<0>(a);
#endif
        static_for<NQT>([&](auto qq) {
            constexpr int qt = qq;
#if PF_Q
            QTile nxt;
            if constexpr (qt + 1 < NQT) nxt = load_qtile<qt + 1>(a);
            fence();  // the next tile's loads ahead of this tile's math: distinct registers, no write-after-read wait
#else
            const QTile cur = load_qtile<qt>(a);
#endif
            const i32x32 &qx = cur.qx;
            const i32x16 &dox = cur.dox;
            const int qsc = cur.qsc, dsc = cur.dsc;
            static_for<NJ>([&](auto jj) {
                constexpr int j = jj;
                f32x8 sv = wmma_mxc<0, 0, NEG_C>(lo16(qx), lo16(kx[j]), cur.lt, qsc, ksc[j]);
                sv = wmma_mx<1, 1>(mid16(qx), mid16(kx[j]), sv, qsc, ksc[j]);
                const f32x8 dp = wmma_mxc<0, 0, NEG_C>(dox, vb[j], cur.dt, dsc, vsc[j]);
                f32x8 p, ds;
                static_for<8>([&](auto rr) {
                    constexpr int r = rr;
                    float pv = exp2_raw(sv[r]);
                    if constexpr (DIAG) pv = (16 * qt + r < thr[j]) ? 0.f : pv;  // q < kv
                    p[r] = pv;
                    ds[r] = pv * dp[r];
                });
                const u32x2 pq = cvt8_e4m3(p, 1.0f / 256.0f);
                pB[j][2 * qt] = (int)pq[0];
                pB[j][2 * qt + 1] = (int)pq[1];
                if constexpr (qt % 2 == 0) {
                    dsprev[j] = ds;
                } else {
                    const DsQ z = quant_pair(dsprev[j], ds);
                    dsB[j][2 * (qt - 1)] = (int)z.a[0];
                    dsB[j][2 * (qt - 1) + 1] = (int)z.a[1];
                    dsB[j][2 * qt] = (int)z.b[0];
                    dsB[j][2 * qt + 1] = (int)z.b[1];
                    dss[j] |= z.byte << (8 * (qt / 2));
                }
            });
#if QFENCE2
            fence();
#endif
#if PF_Q
            if constexpr (qt + 1 < NQT) cur = nxt;
#endif
        });
#endif  // QPIPE
        // dV^T += dO^T P (d16 tiles 0..7 of dO^T), dK^T += Q^T dS (12 of Q^T): 20 A fragments in one prefetched sequence
#if PF_A == 2
        // 4 buffers, loads two groups ahead and fenced before the group's WMMAs: the buffer a load overwrites was read
        // two groups earlier (PF_A 1 let the scheduler sink the loads under the WMMAs into the registers they still read:
        // ~36-cycle write-after-read waits, 20 per step)
        auto ld = [&](auto dc) __attribute__((always_inline)) {
            constexpr int d = decltype(dc)::value;
            if constexpr (d < 8)
                return load_afrag<d, 0>(a.dot, a.dots);
            else
                return load_afrag<d - 8, 1>(a.qt, a.qts);
        };
        constexpr int NAB = PF_AD + 2;
        AFrag ab[NAB];
        static_for<PF_AD>([&](auto i) { ab[(int)i] = ld(std::integral_constant<int, (int)i>{}); });
        static_for<20>([&](auto dd) {
            constexpr int d = dd;
            if constexpr (d + PF_AD < 20)
                ab[(d + PF_AD) % NAB] = ld(std::integral_constant<int, (d + PF_AD < 20 ? d + PF_AD : 0)>{});
#if KEEP_WAR & 1
            if constexpr (d >= 1 && d + PF_AD < 20) {
                keep(ab[(d + NAB - 1) % NAB].x);  // fragment d - 1: read by the previous group's WMMAs
                keep(ab[(d + NAB - 1) % NAB].s);
            }
#endif
            fence();
            const AFrag &af = ab[d % NAB];
            static_for<NJ>([&](auto jj) {
                constexpr int j = jj;
                if constexpr (d < 8) {
                    if constexpr (d % 2 == 0)
                        dv[d][j] = wmma_mx<0, 0>(af.x, pB[j], dv[d][j], af.s, PSCALE);
                    else
                        dv[d][j] = wmma_mx<1, 0>(af.x, pB[j], dv[d][j], af.s, PSCALE);
                } else {
                    constexpr int e = d - 8;
                    if constexpr (e % 2 == 0)
                        dk[e][j] = wmma_mx<0, 0>(af.x, dsB[j], dk[e][j], af.s, dss[j]);
                    else
                        dk[e][j] = wmma_mx<1, 0>(af.x, dsB[j], dk[e][j], af.s, dss[j]);
                }
            });
            fence();
        });
#else
#if PF_A
        AFrag af = load_afrag<0, 0>(a.dot, a.dots);
#endif
        static_for<20>([&](auto dd) {
            constexpr int d = dd;
#if PF_A
            AFrag an;
            if constexpr (d + 1 < 8)
                an = load_afrag<d + 1, 0>(a.dot, a.dots);
            else if constexpr (d + 1 < 20)
                an = load_afrag<d + 1 - 8, 1>(a.qt, a.qts);
#else
            const AFrag af = d < 8 ? load_afrag<(d < 8 ? d : 0), 0>(a.dot, a.dots)
                                   : load_afrag<(d < 8 ? 0 : d - 8), 1>(a.qt, a.qts);
#endif
            static_for<NJ>([&](auto jj) {
                constexpr int j = jj;
                if constexpr (d < 8) {
                    if constexpr (d % 2 == 0)
                        dv[d][j] = wmma_mx<0, 0>(af.x, pB[j], dv[d][j], af.s, PSCALE);
                    else
                        dv[d][j] = wmma_mx<1, 0>(af.x, pB[j], dv[d][j], af.s, PSCALE);
                } else {
                    constexpr int e = d - 8;
                    if constexpr (e % 2 == 0)
                        dk[e][j] = wmma_mx<0, 0>(af.x, dsB[j], dk[e][j], af.s, dss[j]);
                    else
                        dk[e][j] = wmma_mx<1, 0>(af.x, dsB[j], dk[e][j], af.s, dss[j]);
                }
            });
            fence();
#if PF_A
            if constexpr (d + 1 < 20) af = an;
#endif
        });
#endif
    };

    // the steps that cross the diagonal (q0 < kv0 + 192: at most 3) run masked, in their own loop (no branch between
    // two instantiations inside one loop: it costs the register allocator ~100 VGPRs)
    const int ndiag = min((kv0 + KB - qs0 + QS_ - 1) / QS_, nst);
    auto run = [&](int s, auto diag_c) __attribute__((always_inline)) {
        tensor_wait<TDM_PER_STEP>();  // this wave's step s landed (step s + 1 in flight)
        bar_signal();
        bar_wait();         // step s visible; every wave is done with step s - 1
        issue_step(s + 2);  // into the stage of step s - 1
        const unsigned st = lds0 + (unsigned)((s % NSTG) * STG_B);
        const int ln = lane_now(), r16 = ln & 15, h = ln >> 4;
        const unsigned rq = st + (unsigned)(r16 * XK), r1 = st + (unsigned)(r16 * X128);
        const StepAddr a = {pin(rq + O_Q + 64 * h),
                            pin(rq + O_Q + 128 + 32 * h),
                            pin(st + O_QS + r16 * 8 + 4 * h),
                            pin(r1 + O_DO + 64 * h),
                            pin(st + O_DOS + r16 * 4),
                            pin(st + O_LSE + 32 * h),
                            pin(st + O_DEL + 32 * h),
                            pin(r1 + O_DOT + 64 * h),
                            pin(st + O_DOTS + ln * 4),
                            pin(r1 + O_QT + 64 * h),
                            pin(st + O_QTS + ln * 4)};
        step(a, qs0 + s * QS_, diag_c);
    };
    for (int s = 0; s < ndiag; ++s) run(s, std::true_type{});
    for (int s = ndiag; s < nst; ++s) run(s, std::false_type{});
#if OSTAGE
    // ---- epilogue: dK / dV rows (lane = kv column, rows d = 8 lh + r of each d16 tile) as bf16 into LDS (the ring,
    // idle now), then one TDM store each (wave 0: dK, 384 B rows; wave 1: dV, 256 B rows), rows >= S clipped by the
    // descriptor's rows in bounds (no exec change; the scattered SBHD stores held their VGPRs for hundreds of cycles)
    tensor_wait<0>();
    bar_signal();
    bar_wait();  // every wave is done with the ring
    constexpr int XDK = 2 * D_QK, XDV = 2 * D_V, O_SDK = 0, O_SDV = KB * XDK;
    static_assert(O_SDV + KB * XDV <= LDS_B, "staging");
    {
        const int ln = lane_now();
        static_for<NJ>([&](auto jj) {
            constexpr int j = jj;
            const unsigned r = (unsigned)(KBW * wv + 16 * j + (ln & 15)), hh = (unsigned)(ln >> 4);
            const unsigned dkr = lds0 + O_SDK + r * XDK + 16 * hh, dvr = lds0 + O_SDV + r * XDV + 16 * hh;
            static_for<12>([&](auto dd) {
                constexpr int d = dd;
                bf16x8 w;
                static_for<8>([&](auto rr) { w[(int)rr] = (bf16)(dk[d][j][(int)rr] * scale); });
                *(__attribute__((address_space(3))) u32x4 *)(uintptr_t)(dkr + 32 * d) = __builtin_bit_cast(u32x4, w);
            });
            static_for<8>([&](auto dd) {
                constexpr int d = dd;
                bf16x8 w;
                static_for<8>([&](auto rr) { w[(int)rr] = (bf16)dv[d][j][(int)rr]; });
                *(__attribute__((address_space(3))) u32x4 *)(uintptr_t)(dvr + 32 * d) = __builtin_bit_cast(u32x4, w);
            });
        });
    }
    bar_signal();
    bar_wait();
    {
        const unsigned rows = (unsigned)min(KB, S - kv0);
        if (wv == 0) {
            const unsigned long long g = (unsigned long long)(uintptr_t)(DK + (long long)b * dk_sb + (long long)kv0 * dk_ss +
                                                                         (long long)h * D_QK);
            tdm_store(tdm_g0(lds0 + O_SDK, g), tdm_g1(tdm_w0(0, 0, 0), rows, XDK, KB, 2ull * (unsigned long long)dk_ss));
        } else if (wv == 1) {
            const unsigned long long g = (unsigned long long)(uintptr_t)(DV + (long long)b * dv_sb + (long long)kv0 * dv_ss +
                                                                         (long long)h * D_V);
            tdm_store(tdm_g0(lds0 + O_SDV, g), tdm_g1(tdm_w0(0, 0, 0), rows, XDV, KB, 2ull * (unsigned long long)dv_ss));
        }
    }
#else
    // ---- epilogue: lane = kv column, rows d = 8 lh + r of each d16 tile
    static_for<NJ>([&](auto jj) {
        constexpr int j = jj;
        const int kv = kvw + 16 * j + lr;
        if (kv < S) {
            bf16 *dkr = DK + (long long)b * dk_sb + (long long)kv * dk_ss + (long long)h * D_QK;
            bf16 *dvr = DV + (long long)b * dv_sb + (long long)kv * dv_ss + (long long)h * D_V;
            static_for<12>([&](auto dd) {
                constexpr int d = dd;
                bf16x8 w;
                static_for<8>([&](auto rr) { w[(int)rr] = (bf16)(dk[d][j][(int)rr] * scale); });
                *(u32x4 *)(dkr + 16 * d + 8 * lh) = __builtin_bit_cast(u32x4, w);
            });
            static_for<8>([&](auto dd) {
                constexpr int d = dd;
                bf16x8 w;
                static_for<8>([&](auto rr) { w[(int)rr] = (bf16)dv[d][j][(int)rr]; });
                *(u32x4 *)(dvr + 16 * d + 8 * lh) = __builtin_bit_cast(u32x4, w);
            });
        }
    });
#endif
    tensor_wait<0>();
}

// =====================================================================================================================
// k_dq_mx (b1): b0's kernel with the m4 recipe (overlapped [nope | rope] fragments, LSE2 / DELTA as negated
// accumulator tuples: the lane's lse2 / delta x 8, as the lane is a q column here), the causal mask only on the two
// diagonal kv tiles (peeled after the loop, as constant-vs-threshold compares), dS exponents from the float bits.
// (Variant history: a version with a K / V pair struct, prefetch and pinned addresses spilled 260 B; this one does
// not: 911 VGPRs.)
// =====================================================================================================================
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
             const float *__restrict__ LSE2, const float *__restrict__ DELTA, bf16 *__restrict__ DQ, float scale, int S,
             int H, int B, long long dq_sb, long long dq_ss) {
    __shared__ __attribute__((aligned(1024))) unsigned char smem[Q_LDS_B];
    lock_simd();
    const uint32_t lds0 = lds_addr(smem);
    const int tid = (int)__builtin_amdgcn_workitem_id_x();
    const int wv = wave_id(tid);
    const int lane = tid & 31, lr = lane & 15, lh = lane >> 4;
    const int cnt0 = wv == 0 ? 1 : 0;
    const int nqb = S / BMQ;
    const BlockOf blk = xcd_block(nqb, H, B);
    const int qb = nqb - 1 - blk.slot;
    const int h = blk.h, b = blk.b;
    const int q0 = qb * BMQ;
    const int n = (q0 + BMQ) / NBK;
    const unsigned long long bh = (unsigned long long)b * H + h;
    const float c2 = scale * LOG2E;

    // the tile's 7 TDM ops, 2 per wave: [K^T, KS], [K nope, VS], [V, K^T scales], [K rope, -] (base + tile * mult)
    auto dq_op = [&](int k) -> TdmOp {
        const unsigned long long bS = bh * (unsigned long long)S, bS128 = bh * (unsigned long long)(S / 128);
        switch (k) {
        case 0: return {Q_O_K, (unsigned long long)(uintptr_t)K8 + bS * D_QK, (unsigned long long)NBK * D_QK, tdm_g1(W0_KN, NBK, 128, NBK, D_QK), 1};
        case 1: return {Q_O_K + 128, (unsigned long long)(uintptr_t)K8 + bS * D_QK + 128, (unsigned long long)NBK * D_QK, tdm_g1(W0_KR, NBK, 64, NBK, D_QK), 1};
        case 2: return {Q_O_KS, (unsigned long long)(uintptr_t)KS + bS * 8, (unsigned long long)NBK * 8, tdm_g1(W0_S, NBK, 8, NBK, 8), 1};
        case 3: return {Q_O_V, (unsigned long long)(uintptr_t)V8 + bS * D_V, (unsigned long long)NBK * D_V, tdm_g1(W0_128, NBK, 128, NBK, D_V), 1};
        case 4: return {Q_O_VS, (unsigned long long)(uintptr_t)VSR + bS * 4, (unsigned long long)NBK * 4, tdm_g1(W0_S, NBK, 4, NBK, 4), 1};
        case 5: return {Q_O_KT, (unsigned long long)(uintptr_t)K8T + bh * D_QK * S, NBK, tdm_g1(W0_128, D_QK, 128, D_QK, (unsigned long long)S), 1};
        case 6: return {Q_O_KTS, (unsigned long long)(uintptr_t)KTS + bS128 * D_QK * 4, (unsigned long long)D_QK * 4, tdm_g1(W0_S, D_QK, 4, D_QK, 4), 1};
        default: return {0, 0, 0, tdm_g1(W0_S, 1, 4, 1, 4), 0};
        }
    };
    constexpr int TDM_PER_TILE = 2;
    const TdmOp top0 = dq_op(wv == 0 ? 5 : wv == 1 ? 0 : wv == 2 ? 3 : 1);
    const TdmOp top1 = dq_op(wv == 0 ? 2 : wv == 1 ? 4 : wv == 2 ? 6 : 7);
    auto issue_tile = [&](int t) {
        const bool real = t < n;
        const unsigned long long tt = (unsigned long long)(real ? t : n - 1);
        const unsigned st = lds0 + (unsigned)((t % Q_NSTG) * Q_STG);
        tdm_load(tdm_g0(st + top0.lds, top0.base + tt * top0.mult, real ? top0.on : 0), top0.g1);
        tdm_load(tdm_g0(st + top1.lds, top1.base + tt * top1.mult, real ? top1.on : 0), top1.g1);
    };
    // prologue: the block's Q8 / QS / DO8 / DOS / LSE / DELTA, two ops per wave: [Q nope, LSE], [Q rope, DELTA],
    // [QS, DO8], [DOS, -]
    {
        const unsigned long long qrow = bh * S + q0;
        const unsigned lds_a = wv == 0 ? Q_OFF_Q : wv == 1 ? Q_OFF_Q + 128 : wv == 2 ? Q_OFF_QS : Q_OFF_DOS;
        const unsigned long long ga_a = wv == 0 ? (unsigned long long)(uintptr_t)Q8 + qrow * D_QK
                                        : wv == 1 ? (unsigned long long)(uintptr_t)Q8 + qrow * D_QK + 128
                                        : wv == 2 ? (unsigned long long)(uintptr_t)QS + qrow * 8
                                                  : (unsigned long long)(uintptr_t)DOS + qrow * 4;
        const i32x8 g1a = wv == 0 ? tdm_g1(W0_KN, BMQ, 128, BMQ, D_QK)
                          : wv == 1 ? tdm_g1(W0_KR, BMQ, 64, BMQ, D_QK)
                          : wv == 2 ? tdm_g1(W0_S, BMQ, 8, BMQ, 8)
                                    : tdm_g1(W0_S, BMQ, 4, BMQ, 4);
        tdm_load(tdm_g0(lds0 + lds_a, ga_a), g1a);
        const unsigned lds_b = wv == 0 ? Q_OFF_LSE : wv == 1 ? Q_OFF_DEL : Q_OFF_DO;
        const unsigned long long ga_b = wv == 0 ? (unsigned long long)(uintptr_t)(LSE2 + qrow)
                                        : wv == 1 ? (unsigned long long)(uintptr_t)(DELTA + qrow)
                                                  : (unsigned long long)(uintptr_t)DO8 + qrow * D_V;
        const i32x8 g1b = wv <= 1 ? tdm_g1(W0_S, 1, 1024, 1, 1024) : tdm_g1(W0_128, BMQ, 128, BMQ, D_V);
        tdm_load(tdm_g0(lds0 + lds_b, ga_b, wv <= 2 ? 1 : 0), g1b);
    }
    issue_tile(0);
    issue_tile(1);
#if DQ_PRO_PARTIAL
    tensor_wait<TDM_PER_TILE>();  // the block's Q / dO region and tile 0 (in order); tile 1's ops may stay in flight
#else
    tensor_wait<0>();
#endif
    bar_signal();
    bar_wait();

    f32x8 dq[12][QTW];
    static_for<12>([&](auto d) { static_for<QTW>([&](auto i) { dq[(int)d][(int)i] = f32x8{0, 0, 0, 0, 0, 0, 0, 0}; }); });
    f32x8 ltq[QTW], dtq[QTW];
    i32x32 qx[QTW];
    i32x16 dob[QTW];
    int qscb[QTW], dscb[QTW];
    static_for<QTW>([&](auto ii) {
        constexpr int i = ii;
        const int row = 64 * i + 16 * wv + lr;
        const float l2 = __builtin_bit_cast(float, lds_b32(lds_ptr(lds0 + Q_OFF_LSE), row * 4));
        const float de = __builtin_bit_cast(float, lds_b32(lds_ptr(lds0 + Q_OFF_DEL), row * 4));
        ltq[i] = f32x8{l2, l2, l2, l2, l2, l2, l2, l2};
        dtq[i] = f32x8{de, de, de, de, de, de, de, de};
        const lds_u8 *qp = lds_ptr(lds0 + Q_OFF_Q + (unsigned)(row * XK));
        qx[i] = frag24(lds_b128(qp, 64 * lh), lds_b128(qp, 64 * lh + 16), lds_b128(qp, 64 * lh + 32), lds_b128(qp, 64 * lh + 48), lds_b128(qp, 128 + 32 * lh), lds_b128(qp, 128 + 32 * lh + 16));
        qscb[i] = (int)lds_b32(lds_ptr(lds0 + Q_OFF_QS), row * 8 + 4 * lh);
        dob[i] = lds_frag64(lds_ptr(lds0 + Q_OFF_DO + (unsigned)(row * X128)), 64 * lh);
        dscb[i] = (int)lds_b32(lds_ptr(lds0 + Q_OFF_DOS), row * 4);
    });

    auto tile = [&](int t, auto diag_c) __attribute__((always_inline)) {
        constexpr bool DIAG = decltype(diag_c)::value;
        if (t > 0) {
            tensor_wait<TDM_PER_TILE>();  // tile t landed (t + 1 in flight)
            bar_signal();
            bar_wait();        // tile t visible; every wave is done with tile t - 1
        }
        issue_tile(t + 2);     // into the stage of tile t - 1
        const unsigned st = lds0 + (unsigned)((t % Q_NSTG) * Q_STG);
        const int kv0 = t * NBK;
        i32x16 dsB[QTW];
        int dss[QTW] = {0, 0, 0, 0};
#if PF_KV
        // kv16 pairs (2m, 2m + 1) = MX block m of the dQ reduction. Pair m + 1's K / V fragments load (fenced) ahead of
        // pair m's math (two buffers); the dQ phase's K^T fragments load two groups ahead into four buffers, the first
        // two during pair 3 (loads issued just before use waited ~100+ cycles each)
        struct KVPair {
            i32x32 kx[2];
            i32x16 va[2];
            int ksc[2], vsc[2];
        };
        // per-lane fragment bases of this tile, formed once (pinned): every load below is base + immediate offset, no
        // VALU address math between the WMMAs (a per-load v_add3 landed in a register the previous WMMAs still read:
        // ~31-cycle write-after-read stall + 4 v_nop per dQ group)
        const int ln = lane_now(), r16 = ln & 15, hh = ln >> 4;
        const uint32_t kb_n = pin(st + Q_O_K + (unsigned)(r16 * XK + 64 * hh)),
                       kb_r = pin(st + Q_O_K + (unsigned)(r16 * XK + 128 + 32 * hh)),
                       kb_s = pin(st + Q_O_KS + (unsigned)(r16 * 8 + 4 * hh)),
                       vb_t = pin(st + Q_O_V + (unsigned)(r16 * X128 + 64 * hh)),
                       vb_s = pin(st + Q_O_VS + (unsigned)(r16 * 4)),
                       kt_t = pin(st + Q_O_KT + (unsigned)(r16 * X128 + 64 * hh)),
                       kt_s = pin(st + Q_O_KTS + (unsigned)(ln * 4));
        auto ldkv = [&](auto mc, KVPair &kv) __attribute__((always_inline)) {
            constexpr int m = decltype(mc)::value;
            static_for<2>([&](auto uu) {
                constexpr int u = uu;
                constexpr int kr = 16 * (2 * m + u);
                const lds_u8 *kn = lds_ptr(kb_n), *krp = lds_ptr(kb_r);
                kv.kx[u] = frag24(lds_b128(kn, kr * XK), lds_b128(kn, kr * XK + 16), lds_b128(kn, kr * XK + 32),
                                  lds_b128(kn, kr * XK + 48), lds_b128(krp, kr * XK), lds_b128(krp, kr * XK + 16));
                kv.ksc[u] = (int)lds_b32(lds_ptr(kb_s), kr * 8);
                const lds_u8 *vp = lds_ptr(vb_t);
                kv.va[u] = cat4(lds_b128(vp, kr * X128), lds_b128(vp, kr * X128 + 16), lds_b128(vp, kr * X128 + 32),
                                lds_b128(vp, kr * X128 + 48));
                kv.vsc[u] = (int)lds_b32(lds_ptr(vb_s), kr * 4);
            });
        };
        struct AF {
            i32x16 x;
            int s;
        };
        auto lda = [&](auto dc) __attribute__((always_inline)) {
            constexpr int dt = decltype(dc)::value;
            const lds_u8 *tp = lds_ptr(kt_t);
            constexpr int o = 16 * dt * X128;
            return AF{cat4(lds_b128(tp, o), lds_b128(tp, o + 16), lds_b128(tp, o + 32), lds_b128(tp, o + 48)),
                      (int)lds_b32(lds_ptr(kt_s), 32 * (dt / 2) * 4)};
        };
        AF ab[4];
        KVPair kvb[2];
        ldkv(std::integral_constant<int, 0>{}, kvb[0]);
        static_for<NKV / 2>([&](auto mm) {
            constexpr int m = mm;
            if constexpr (m + 1 < NKV / 2) {
                ldkv(std::integral_constant<int, m + 1>{}, kvb[(m + 1) % 2]);
            } else {
                ab[0] = lda(std::integral_constant<int, 0>{});
                ab[1] = lda(std::integral_constant<int, 1>{});
            }
            fence();
            const KVPair &kv = kvb[m % 2];
            static_for<QTW>([&](auto ii) {
                constexpr int i = ii;
                const int thr = q0 + 64 * i + 16 * wv + lr - kv0 - 8 * lh;  // masked iff 16 j + r > thr
                f32x8 ds[2];
                static_for<2>([&](auto uu) {
                    constexpr int u = uu;
                    constexpr int j = 2 * m + u;
                    f32x8 z = wmma_mxc<0, 0, 1>(lo16(kv.kx[u]), lo16(qx[i]), ltq[i], kv.ksc[u], qscb[i]);
                    const f32x8 sv = wmma_mx<1, 1>(mid16(kv.kx[u]), mid16(qx[i]), z, kv.ksc[u], qscb[i]);
                    const f32x8 dp = wmma_mxc<0, 0, 1>(kv.va[u], dob[i], dtq[i], kv.vsc[u], dscb[i]);
                    static_for<8>([&](auto rr) {
                        constexpr int r = rr;
                        float pv = exp2_raw(sv[r]);
                        if constexpr (DIAG) pv = (16 * j + r > thr) ? 0.f : pv;
                        ds[u][r] = pv * dp[r];
                    });
                });
                float am = 0.f;
                static_for<8>([&](auto rr) {
                    am = __builtin_fmaxf(am, __builtin_fabsf(ds[0][(int)rr]));
                    am = __builtin_fmaxf(am, __builtin_fabsf(ds[1][(int)rr]));
                });
                const int e = mx_exp(__builtin_fmaxf(am, peer(am)));
                const float sc = exp2i(e);
                const u32x2 q0v = cvt8_e4m3(ds[0], sc), q1v = cvt8_e4m3(ds[1], sc);
                dsB[i][4 * m] = (int)q0v[0];
                dsB[i][4 * m + 1] = (int)q0v[1];
                dsB[i][4 * m + 2] = (int)q1v[0];
                dsB[i][4 * m + 3] = (int)q1v[1];
                dss[i] |= (e + 127) << (8 * m);
            });
#if DQ_SGB
            // issue windows for the pair: 24 WMMAs (S nope + rope, dP per kv16 x 4 q tiles), 64 exps, ~100 VALU: each
            // q tile's exps behind other q tiles' WMMAs instead of right after their own (v_nop hazard fills)
            static_for<24>([&](auto kk) {
                __builtin_amdgcn_sched_group_barrier(0x8, 1, 0);    // WMMA
                __builtin_amdgcn_sched_group_barrier(0x400, 2, 0);  // TRANS
                __builtin_amdgcn_sched_group_barrier(0x2, 3, 0);    // VALU
                if constexpr ((int)kk % 3 == 0) __builtin_amdgcn_sched_group_barrier(0x400, 1, 0);
            });
#endif
            fence();
        });
        // dQ^T += K^T dS^T: A = K8T rows d, K = this tile's kv
        static_for<12>([&](auto dd) {
            constexpr int dt = dd;
            if constexpr (dt + 2 < 12) ab[(dt + 2) % 4] = lda(std::integral_constant<int, dt + 2>{});
            fence();
            const AF &af = ab[dt % 4];
            static_for<QTW>([&](auto ii) {
                constexpr int i = ii;
                if constexpr (dt % 2 == 0)
                    dq[dt][i] = wmma_mx<0, 0>(af.x, dsB[i], dq[dt][i], af.s, dss[i]);
                else
                    dq[dt][i] = wmma_mx<1, 0>(af.x, dsB[i], dq[dt][i], af.s, dss[i]);
            });
            fence();
        });
#else
        // kv16 pairs (2m, 2m + 1) = MX block m of the dQ reduction: K / V fragments of the pair, then every q tile
        static_for<NKV / 2>([&](auto mm) {
            constexpr int m = mm;
            i32x32 kx[2];
            i32x16 va[2];
            int ksc[2], vsc[2];
            static_for<2>([&](auto uu) {
                constexpr int u = uu;
                const int krow = 16 * (2 * m + u) + lr;
                const lds_u8 *kp = lds_ptr(st + Q_O_K + (unsigned)(krow * XK));
                kx[u] = frag24(lds_b128(kp, 64 * lh), lds_b128(kp, 64 * lh + 16), lds_b128(kp, 64 * lh + 32), lds_b128(kp, 64 * lh + 48), lds_b128(kp, 128 + 32 * lh), lds_b128(kp, 128 + 32 * lh + 16));
                ksc[u] = (int)lds_b32(lds_ptr(st + Q_O_KS), krow * 8 + 4 * lh);
                va[u] = lds_frag64(lds_ptr(st + Q_O_V + (unsigned)(krow * X128)), 64 * lh);
                vsc[u] = (int)lds_b32(lds_ptr(st + Q_O_VS), krow * 4);
            });
            static_for<QTW>([&](auto ii) {
                constexpr int i = ii;
                const int thr = q0 + 64 * i + 16 * wv + lr - kv0 - 8 * lh;  // masked iff 16 j + r > thr
                f32x8 ds[2];
                static_for<2>([&](auto uu) {
                    constexpr int u = uu;
                    constexpr int j = 2 * m + u;
                    f32x8 z = wmma_mxc<0, 0, 1>(lo16(kx[u]), lo16(qx[i]), ltq[i], ksc[u], qscb[i]);
                    const f32x8 sv = wmma_mx<1, 1>(mid16(kx[u]), mid16(qx[i]), z, ksc[u], qscb[i]);
                    const f32x8 dp = wmma_mxc<0, 0, 1>(va[u], dob[i], dtq[i], vsc[u], dscb[i]);
                    static_for<8>([&](auto rr) {
                        constexpr int r = rr;
                        float pv = exp2_raw(sv[r]);
                        if constexpr (DIAG) pv = (16 * j + r > thr) ? 0.f : pv;
                        ds[u][r] = pv * dp[r];
                    });
                });
#if DQ_QB
                float am = 0.f;
                static_for<8>([&](auto rr) {
                    am = __builtin_fmaxf(am, __builtin_fabsf(ds[0][(int)rr]));
                    am = __builtin_fmaxf(am, __builtin_fabsf(ds[1][(int)rr]));
                });
                const int e = mx_exp(__builtin_fmaxf(am, peer(am)));
                const float sc = exp2i(e);
                const u32x2 q0v = cvt8_e4m3(ds[0], sc), q1v = cvt8_e4m3(ds[1], sc);
                dsB[i][4 * m] = (int)q0v[0];
                dsB[i][4 * m + 1] = (int)q0v[1];
                dsB[i][4 * m + 2] = (int)q1v[0];
                dsB[i][4 * m + 3] = (int)q1v[1];
                dss[i] |= (e + 127) << (8 * m);
#else
                const DsQ z = quant_pair(ds[0], ds[1]);
                dsB[i][4 * m] = (int)z.a[0];
                dsB[i][4 * m + 1] = (int)z.a[1];
                dsB[i][4 * m + 2] = (int)z.b[0];
                dsB[i][4 * m + 3] = (int)z.b[1];
                dss[i] |= z.byte << (8 * m);
#endif
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
#endif
    };
    for (int t = 0; t < n - 2; ++t) tile(t, std::false_type{});
    tile(n - 2, std::true_type{});
    tile(n - 1, std::true_type{});
#if OSTAGE
    // epilogue: dQ rows as bf16 into LDS (the ring, idle now), one TDM store of 256 x 384 B by wave 0
    tensor_wait<0>();
    bar_signal();
    bar_wait();
    constexpr int XDQ = 2 * D_QK;
    static_assert(BMQ * XDQ <= Q_OFF_Q, "staging inside the ring");
    {
        const int ln = lane_now();
        static_for<QTW>([&](auto ii) {
            constexpr int i = ii;
            const unsigned rowb = lds0 + (unsigned)((64 * i + 16 * wv + (ln & 15)) * XDQ + 16 * (ln >> 4));
            static_for<12>([&](auto dd) {
                constexpr int dt = dd;
                bf16x8 w;
                static_for<8>([&](auto rr) { w[(int)rr] = (bf16)(dq[dt][i][(int)rr] * scale); });
                *(__attribute__((address_space(3))) u32x4 *)(uintptr_t)(rowb + 32 * dt) = __builtin_bit_cast(u32x4, w);
            });
        });
    }
    bar_signal();
    bar_wait();
    if (wv == 0) {
        const unsigned long long g = (unsigned long long)(uintptr_t)(DQ + (long long)b * dq_sb + (long long)q0 * dq_ss +
                                                                     (long long)h * D_QK);
        tdm_store(tdm_g0(lds0, g), tdm_g1(tdm_w0(0, 0, 0), BMQ, XDQ, BMQ, 2ull * (unsigned long long)dq_ss));
    }
#else
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
#endif
    tensor_wait<0>();
}
