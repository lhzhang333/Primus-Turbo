/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// k_mla_bwd_mx5.hip -- rung b5 = b3 with the dK / dV kernel software-pipelined across steps (see k_dkdv_mx below;
// k_dq_mx as b3). b3's notes follow.
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

// dq (b3's kernel): software prefetch of the kv16 pairs, partial prologue wait, epilogue through LDS
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
#ifndef DQ_KEEP
// keep the fragments the previous WMMAs read alive past the next loads (the register allocator does not model the
// write-after-read wait of a load into registers an in-flight WMMA reads, ~37-47 cycles): bit 0 the dQ phase's K^T
// fragments (dq 0.624 -> 0.598 ms), bit 1 the K / V pairs (112 spills)
#define DQ_KEEP 1
#endif
#ifndef DQ_KVMID
#define DQ_KVMID 0  // pair m + 1's K / V loads after q tile DQ_KVMID - 1 of pair m behind a fence (slower: 0.611..0.649)
#endif
// dkdv (b5)
#ifndef KEEP_WAR
#define KEEP_WAR 1  // bit 0: keep the previous slot's dV / dK A fragments alive past the next loads (as DQ_KEEP)
#endif
#ifndef PRO_TDM_FIRST
#define PRO_TDM_FIRST 0  // 1: dkdv TDM rounds 0, 1 before the resident K / V loads (neutral: 0.908 vs 0.909 ms)
#endif
#ifndef EPI_WAVE
#define EPI_WAVE 0  // 1: dkdv waves TDM-store their own staged rows, no barrier (neutral: 0.910 vs 0.909 ms)
#endif
#ifndef TDM_INC
#define TDM_INC 1  // dkdv: each wave's 3 TDM ops keep their next step, global address and LDS stage (advanced per round)
#endif
#ifndef DKDV_ORDER
#define DKDV_ORDER 2  // dkdv workgroup order: 0 XCD-local (b, h) at a time, kv block 0 first; 2 long halves then short halves
#endif
#ifndef DQ_ORDER
#define DQ_ORDER 2  // dq workgroup order: 0 longest q block first per (b, h); 1 snake; 2 long halves then short halves
#endif
#ifndef DQ_WAIT0
#define DQ_WAIT0 0  // dq: the per-tile TDM wait + barrier also before tile 0 (the tile loop becomes one basic block)
#endif
#ifndef DQ_DSTAT
#define DQ_DSTAT 1  // dq: static mask classes on the two diagonal kv tiles (fully masked q tiles skipped, kept ones unmasked)
#endif
#ifndef NO_FINAL_WAIT
#define NO_FINAL_WAIT 0  // timing probe only: dkdv ends without waiting for its TDM stores (LDS may be reused early)
#endif
#ifndef EPI_PAD
#define EPI_PAD 0  // dkdv epilogue: padded staging rows, per-wave dense re-pack + TDM stores (no bank conflicts, no barrier)
#endif
#ifndef EPI_DIRECT
#define EPI_DIRECT 0  // dkdv: dK / dV rows by global stores from the accumulators (no LDS staging / TDM store / waits)
#endif
#ifndef PIN_Q
#define PIN_Q 0  // dkdv: bit 0 P(qt) converted in slot qt, bit 1 dS(qt - 1, qt) quantized in slot qt (empty asm defs)
#endif
#ifndef UNROLL2
#define UNROLL2 0  // dkdv: two steps per loop trip, ping-pong P / dS buffers (no loop-carried copies)
#endif
#ifndef MIDSYNC
// the per-step TDM wait + barrier + issue of round s + 2 between slots MIDSYNC - 1 and MIDSYNC of step s (0: at the step
// start); slot 7 then prefetches step s + 1's tile 0 and its slot-0 A fragments, so a step starts with no barrier
#define MIDSYNC 0  // measured: 4 + MS_PA 1 spills (1.45 ms), MS_PA 0 0.914 vs 0.912 (step 2315 vs 2361 cycles): off
#endif
#ifndef MS_PA
#define MS_PA 1  // MIDSYNC: slot 7 also prefetches the next step's slot-0 A fragments (0: loaded at the step start)
#endif
#ifndef LTDT_G
#define LTDT_G 0  // 1: lse / delta C tuples by buffer loads LTDT_D tiles ahead (slower: 1.000 / 0.978 / 0.962 ms, D 2 / 3 / 4)
#endif
#ifndef LTDT_D
#define LTDT_D 2
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
// k_dkdv_mx (b5): grid (S / 128, H, B), 4 waves; wave w owns kv rows kv0 + 32 w .. + 31 (kv16 tiles j = 0, 1), whose
// K8 / V8 rows are the resident B operands of S = Q K^T and dP = dO V^T. Steps of 128 q rows from kv0's block to S - 128.
// Cross-step pipeline: the eight q16-tile slots of step s also issue the dV / dK WMMAs of step s - 1 (its P / dS kept in
// registers): b3 ran each step's 20 A-fragment groups (dO^T, Q^T: 100 LDS loads per wave) after its q16 tiles, all four
// waves at once, LDS-bandwidth bound (~40-cycle waits per group). The LDS ring is split into a q part (Q / dO rows,
// scales, lse, delta: read in step s) and an A part (dO^T, Q^T, scales: read in step s + 1), 3 stages each; TDM round r =
// the q part of step r + the A part of step r - 1, issued at the start of step r - 2 (round 0, 1 before the loop).
// =====================================================================================================================
namespace {
constexpr int QS_ = 128;
constexpr int NQT = QS_ / 16;
constexpr int NJ = 2;         // kv16 tiles per wave
constexpr int KBW = 16 * NJ;  // kv rows per wave
constexpr int KB = 4 * KBW;   // per workgroup
static_assert(KB == QS_, "b5: step 0 is the only step crossing the diagonal");
// q part of a stage
constexpr int O_Q = 0;
constexpr int O_QS = O_Q + QS_ * XK;      // 26624
constexpr int O_DO = O_QS + QS_ * 8;      // 27648
constexpr int O_DOS = O_DO + QS_ * X128;  // 46080
constexpr int O_LSE = O_DOS + QS_ * 4;    // 46592
constexpr int O_DEL = O_LSE + QS_ * 4;    // 47104
constexpr int QSTG = O_DEL + QS_ * 4;     // 47616
// A part
constexpr int O_DOT = 0;
constexpr int O_DOTS = O_DOT + D_V * X128;  // 18432
constexpr int O_QT = O_DOTS + D_V * 4;      // 18944
constexpr int O_QTS = O_QT + D_QK * X128;   // 46592
constexpr int ASTG = O_QTS + D_QK * 4;      // 47360
constexpr int NSTG = 3;
constexpr int ARING = NSTG * QSTG;          // 142848
constexpr int LDS_B = ARING + NSTG * ASTG;  // 284928
static_assert(LDS_B <= 327680, "LDS");
static_assert(QSTG % 128 == 0 && ASTG % 128 == 0, "TDM alignment");

// per-step LDS addresses (q fields: the q stage of step s; A fields: the A stage of step s - 1)
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
    t.qsc = (int)lds_b32(lds_ptr(a.qs), QT * 16 * 8);
    t.dsc = (int)lds_b32(lds_ptr(a.dos), QT * 16 * 4);
#if !LTDT_G
    t.lt = f8(lds_b128(lds_ptr(a.lse), QT * 64), lds_b128(lds_ptr(a.lse), QT * 64 + 16));
    t.dt = f8(lds_b128(lds_ptr(a.del), QT * 64), lds_b128(lds_ptr(a.del), QT * 64 + 16));
#endif
    return t;
}
struct AFrag {
    i32x16 x;
    int s;
};
template <int D>
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
// dV / dK groups (A fragment g: dO^T d16 tile g < 8, Q^T d16 tile g - 8) of slot k: gs(k) .. gs(k + 1) - 1
__host__ __device__ constexpr int gs(int k) { return k <= 0 ? 0 : k <= 4 ? 3 * k : 12 + 2 * (k - 4); }
static_assert(gs(NQT) == 20, "groups");
}  // namespace

// the kernel body; SPLIT: dK's nope columns (d < 128) to DK (its rows d 0..127 only) and the rope columns to DKP (the
// fused-RoPE path: DK = the kv gradient [.., 0:128], DKP a per-head [b, s, h, 64] buffer), else 192-column dK rows
template <bool SPLIT>
__device__ __forceinline__ void dkdv_body(
    const unsigned char *__restrict__ Q8, const unsigned char *__restrict__ QS, const unsigned char *__restrict__ K8,
    const unsigned char *__restrict__ KS, const unsigned char *__restrict__ V8, const unsigned char *__restrict__ VSR,
    const unsigned char *__restrict__ DO8, const unsigned char *__restrict__ DOS, const unsigned char *__restrict__ DO8T,
    const unsigned char *__restrict__ DOTS, const unsigned char *__restrict__ Q8T, const unsigned char *__restrict__ QTS,
    const float *__restrict__ LSE2, const float *__restrict__ DELTA, bf16 *__restrict__ DK, bf16 *__restrict__ DV,
    float scale, int S, int H, int B, long long dk_sb, long long dk_ss, long long dv_sb, long long dv_ss,
    bf16 *__restrict__ DKP, long long dkp_sb, long long dkp_ss, unsigned char *smem) {
    lock_simd();
    const uint32_t lds0 = lds_addr(smem);
    const int tid = (int)__builtin_amdgcn_workitem_id_x();
    const int wv = wave_id(tid);
    const int lane = tid & 31, lr = lane & 15, lh = lane >> 4;
    const BlockOf blk = xcd_block((S + KB - 1) / KB, H, B);
#if DKDV_ORDER == 2
    // two passes per XCD: every (b, h)'s long half of kv blocks (0 .. nkb / 2 - 1: the most steps) first, then the
    // short halves (the dispatch tail made of short workgroups)
    int kb, h, b;
    {
        const int nkb = (S + KB - 1) / KB;
        if ((B * H) % 8 == 0 && nkb % 2 == 0) {
            const int L = (int)(blockIdx.x + nkb * (blockIdx.y + H * blockIdx.z)), k = L >> 3, xc = L & 7;
            const int nbx = B * H / 8, half = nkb / 2, k1 = k % (nbx * half);
            const int bhx = xc + 8 * (k1 / half), sl = k1 % half;
            kb = k < nbx * half ? sl : half + sl;
            h = bhx % H;
            b = bhx / H;
        } else {
            kb = blk.slot;
            h = blk.h;
            b = blk.b;
        }
    }
#else
    const int kb = blk.slot, h = blk.h, b = blk.b;
#endif
    const int kv0 = kb * KB;
    const int qs0 = kv0;  // KB == QS_
    const int nst = (S - qs0) / QS_;
    const unsigned long long bh = (unsigned long long)b * H + h;
    const int kvw = kv0 + KBW * wv;  // this wave's first kv row

#if !PRO_TDM_FIRST
    // ---- resident B operands: the wave's K / V rows
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

#endif
    // the 11 TDM ops of a round, 3 per wave: [Q^T, QS, LSE], [Q nope, dO scales, delta], [dO, Q rope, -], [dO^T, dO^T
    // scales, Q^T scales] (global address = base + q0 * mult; ops 5..8 are the A part)
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
    constexpr int TDM_PER_ROUND = 3;
    const int k0 = wv == 0 ? 7 : wv == 1 ? 0 : wv == 2 ? 3 : 5;
    const int k1 = wv == 0 ? 2 : wv == 1 ? 4 : wv == 2 ? 1 : 6;
    const int k2 = LTDT_G ? (wv == 3 ? 8 : 11) : (wv == 0 ? 9 : wv == 1 ? 10 : wv == 2 ? 11 : 8);
    const TdmOp op0 = dk_op(k0), op1 = dk_op(k1), op2 = dk_op(k2);
#if TDM_INC
    // the op's next step si (A ops start at -1: round r loads q part r, A part r - 1), its global address (step
    // max(si, 0), held at the last step past the end) and its LDS stage address; rounds are issued in order 0, 1, 2..
    struct OpRun {
        unsigned long long ga, inc;
        unsigned lds, lbase, lstep;
        int si;
    };
    auto mk_run = [&](const TdmOp &op, int k) __attribute__((always_inline)) -> OpRun {
        const bool ap = k >= 5 && k <= 8;
        OpRun o;
        o.si = ap ? -1 : 0;
        o.inc = (unsigned long long)QS_ * op.mult;
        o.ga = op.base + (unsigned long long)qs0 * op.mult;
        o.lbase = (ap ? lds0 + ARING : lds0) + op.lds;
        o.lstep = ap ? (unsigned)ASTG : (unsigned)QSTG;
        o.lds = ap ? o.lbase + 2 * o.lstep : o.lbase;  // stage (si mod 3)
        return o;
    };
    OpRun run0 = mk_run(op0, k0), run1 = mk_run(op1, k1), run2 = mk_run(op2, k2);
    auto issue_one = [&](const TdmOp &op, OpRun &o) __attribute__((always_inline)) {
        const bool real = o.si >= 0 && o.si < nst;
        tdm_load(tdm_g0(o.lds, o.ga, real ? op.on : 0), op.g1);
        o.ga = (o.si >= 0 && o.si + 1 < nst) ? o.ga + o.inc : o.ga;
        o.lds = o.lds == o.lbase + 2 * o.lstep ? o.lbase : o.lds + o.lstep;
        o.si += 1;
    };
    auto issue_round = [&](int) __attribute__((always_inline)) {
        issue_one(op0, run0);
        issue_one(op1, run1);
        issue_one(op2, run2);
    };
#else
    auto issue_one = [&](const TdmOp &op, int k, int r) __attribute__((always_inline)) {
        const bool ap = k >= 5 && k <= 8;
        const int si = ap ? r - 1 : r;  // the step this op loads
        const bool real = si >= 0 && si < nst;
        const int sc = si < 0 ? 0 : (si < nst ? si : nst - 1);
        const unsigned long long q0 = (unsigned long long)(qs0 + sc * QS_);
        const unsigned st = ap ? lds0 + ARING + (unsigned)(((si + NSTG) % NSTG) * ASTG) : lds0 + (unsigned)((si % NSTG) * QSTG);
        tdm_load(tdm_g0(st + op.lds, op.base + q0 * op.mult, real ? op.on : 0), op.g1);
    };
    auto issue_round = [&](int r) __attribute__((always_inline)) {
        issue_one(op0, k0, r);
        issue_one(op1, k1, r);
        issue_one(op2, k2, r);
    };
#endif
    issue_round(0);
    issue_round(1);
#if PRO_TDM_FIRST
    // ---- resident B operands: the wave's K / V rows
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

#endif

    f32x8 dv[8][NJ], dk[12][NJ];
    static_for<8>([&](auto d) { static_for<NJ>([&](auto j) { dv[(int)d][(int)j] = f32x8{0, 0, 0, 0, 0, 0, 0, 0}; }); });
    static_for<12>([&](auto d) { static_for<NJ>([&](auto j) { dk[(int)d][(int)j] = f32x8{0, 0, 0, 0, 0, 0, 0, 0}; }); });

    auto addr = [&](int s) __attribute__((always_inline)) -> StepAddr {
        const unsigned sq = lds0 + (unsigned)((s % NSTG) * QSTG);
        const unsigned sa = lds0 + ARING + (unsigned)(((s + NSTG - 1) % NSTG) * ASTG);  // the A part of step s - 1
        const int ln = lane_now(), r16 = ln & 15, hh = ln >> 4;
        const unsigned rq = sq + (unsigned)(r16 * XK), r1 = sq + (unsigned)(r16 * X128), ra = sa + (unsigned)(r16 * X128);
        return {pin(rq + O_Q + 64 * hh),         pin(rq + O_Q + 128 + 32 * hh), pin(sq + O_QS + r16 * 8 + 4 * hh),
                pin(r1 + O_DO + 64 * hh),        pin(sq + O_DOS + r16 * 4),     pin(sq + O_LSE + 32 * hh),
                pin(sq + O_DEL + 32 * hh),       pin(ra + O_DOT + 64 * hh),     pin(sa + O_DOTS + ln * 4),
                pin(ra + O_QT + 64 * hh),        pin(sa + O_QTS + ln * 4)};
    };
    auto ldA = [&](const StepAddr &a, auto gc) __attribute__((always_inline)) {
        constexpr int g = decltype(gc)::value;
        if constexpr (g < 8)
            return load_afrag<g>(a.dot, a.dots);
        else
            return load_afrag<g - 8>(a.qt, a.qts);
    };
    // dV^T += dO^T P (g < 8), dK^T += Q^T dS (g >= 8) for the previous step's P / dS
    auto group = [&](auto gc, const AFrag &af, const i32x16 (&pBo)[NJ], const i32x16 (&dsBo)[NJ],
                     const int (&dsso)[NJ]) __attribute__((always_inline)) {
        constexpr int g = decltype(gc)::value;
        static_for<NJ>([&](auto jj) {
            constexpr int j = jj;
            if constexpr (g < 8) {
                dv[g][j] = wmma_mx<g % 2, 0>(af.x, pBo[j], dv[g][j], af.s, PSCALE);
            } else {
                constexpr int e = g - 8;
                dk[e][j] = wmma_mx<e % 2, 0>(af.x, dsBo[j], dk[e][j], af.s, dsso[j]);
            }
        });
    };

#if LTDT_G
    // the lse / delta C tuples of q16 tile t (rows 16 t + 8 lh .. + 7 of the step at q0) straight from global memory
    // (L2 / WGP cache: every wave reads the same 1 KB per step), LTDT_D tiles ahead through a ring carried across steps
    struct LtDt {
        f32x8 lt, dt;
    };
    // buffer loads: the (b, h) row base in the descriptor, the row offset in an SGPR, the lane's 32 B in a fixed VGPR
    // (no per-load address math; rows past S read as 0 by the range check)
    const __amdgpu_buffer_rsrc_t rl = __builtin_amdgcn_make_buffer_rsrc(
        (void *)(LSE2 + bh * (unsigned long long)S), (short)0, (long)S * 4, 159744);
    const __amdgpu_buffer_rsrc_t rd = __builtin_amdgcn_make_buffer_rsrc(
        (void *)(DELTA + bh * (unsigned long long)S), (short)0, (long)S * 4, 159744);
    const int voff = 32 * lh;
    auto ld_ltdt = [&](int qrow) __attribute__((always_inline)) -> LtDt {
        const int so = qrow * 4;
        return {f8(__builtin_bit_cast(u32x4, __builtin_amdgcn_raw_buffer_load_b128(rl, voff, so, 0)),
                   __builtin_bit_cast(u32x4, __builtin_amdgcn_raw_buffer_load_b128(rl, voff + 16, so, 0))),
                f8(__builtin_bit_cast(u32x4, __builtin_amdgcn_raw_buffer_load_b128(rd, voff, so, 0)),
                   __builtin_bit_cast(u32x4, __builtin_amdgcn_raw_buffer_load_b128(rd, voff + 16, so, 0)))};
    };
    LtDt ring[LTDT_D];
    static_for<LTDT_D>([&](auto i) { ring[(int)i] = ld_ltdt(qs0 + 16 * (int)i); });
#endif
    // one step: q16 tile qt in slot qt (fragments of tile qt + 1 loaded at the slot's start), with the dV / dK groups
    // gs(qt) .. gs(qt + 1) - 1 of the previous step (A fragments of slot qt + 1 loaded at the slot's start)
    auto step = [&](int s, const StepAddr &a, int q0, i32x16 (&pB)[NJ], i32x16 (&dsB)[NJ], int (&dss)[NJ],
                    const i32x16 (&pBo)[NJ], const i32x16 (&dsBo)[NJ], const int (&dsso)[NJ], QTile &pre_t,
                    AFrag (&pre_a)[3], auto diag_c, auto prev_c) __attribute__((always_inline)) {
        constexpr bool DIAG = decltype(diag_c)::value, PREV = decltype(prev_c)::value;
        // the causal mask as constant-vs-threshold compares: q < kv iff 16 qt + r < thr[j] = kv - q0 - 8 lh
        int thr[NJ];
        static_for<NJ>([&](auto jj) { thr[(int)jj] = kvw + 16 * (int)jj + lr - q0 - 8 * lh; });
        f32x8 dsprev[NJ];
        AFrag ab[20];
#if MIDSYNC
        StepAddr an;  // step s + 1 (after the mid-step barrier)
        if constexpr (PREV) {
            if constexpr (MS_PA)
                static_for<gs(1)>([&](auto i) { ab[(int)i] = pre_a[(int)i]; });
            else
                static_for<gs(1)>([&](auto i) { ab[(int)i] = ldA(a, std::integral_constant<int, (int)i>{}); });
        }
        QTile cur = pre_t;
#else
        if constexpr (PREV)
            static_for<gs(1)>([&](auto i) { ab[(int)i] = ldA(a, std::integral_constant<int, (int)i>{}); });
        QTile cur = load_qtile<0>(a);
#endif
        static_for<NQT>([&](auto qq) {
            constexpr int qt = qq;
#if MIDSYNC
            if constexpr (qt == MIDSYNC) {
                tensor_wait<0>();  // round s + 1 landed: the q part of step s + 1, the A part of step s
                bar_signal();
                bar_wait();          // visible; every wave is done with step s - 1 (q part) and the dV / dK of s - 2
                issue_round(s + 2);  // q part of step s + 2 into step s - 1's q stage, A part of s + 1 into s - 2's
                an = addr(s + 1);
            }
#endif
            QTile nxt;
            if constexpr (qt + 1 < NQT) nxt = load_qtile<(qt + 1 < NQT ? qt + 1 : 0)>(a);
            if constexpr (PREV && qt + 1 < NQT)
                static_for<gs(qt + 2) - gs(qt + 1)>([&](auto i) {
                    constexpr int g = gs(qt + 1) + (int)i;
                    ab[g] = ldA(a, std::integral_constant<int, g>{});
                });
#if MIDSYNC
            if constexpr (qt + 1 == NQT) {  // step s + 1's tile 0 and slot-0 A fragments (A part of step s); past the
                                            // last step: stale LDS, never used
                pre_t = load_qtile<0>(an);
                if constexpr (MS_PA)
                    static_for<gs(1)>([&](auto i) { pre_a[(int)i] = ldA(an, std::integral_constant<int, (int)i>{}); });
            }
#endif
#if KEEP_WAR & 1
            if constexpr (PREV && qt >= 1)
                static_for<gs(qt) - gs(qt - 1)>([&](auto i) {  // slot qt - 1's fragments (its WMMAs may be in flight)
                    constexpr int g = gs(qt - 1) + (int)i;
                    keep(ab[g].x);
                    keep(ab[g].s);
                });
#endif
#if LTDT_G
            // tile qt's tuples from the ring; refill with tile qt + LTDT_D (of the next step past the end: clamped rows
            // beyond the last step are read and never used)
            const LtDt cl = ring[qt % LTDT_D];
            ring[qt % LTDT_D] = ld_ltdt(q0 + 16 * (qt + LTDT_D));
            const f32x8 &clt = cl.lt, &cdt = cl.dt;
#else
            const f32x8 &clt = cur.lt, &cdt = cur.dt;
#endif
            fence();
            const i32x32 &qx = cur.qx;
            const i32x16 &dox = cur.dox;
            const int qsc = cur.qsc, dsc = cur.dsc;
            static_for<NJ>([&](auto jj) {
                constexpr int j = jj;
                f32x8 sv = wmma_mxc<0, 0, NEG_C>(lo16(qx), lo16(kx[j]), clt, qsc, ksc[j]);
                sv = wmma_mx<1, 1>(mid16(qx), mid16(kx[j]), sv, qsc, ksc[j]);
                const f32x8 dp = wmma_mxc<0, 0, NEG_C>(dox, vb[j], cdt, dsc, vsc[j]);
                f32x8 p, ds;
                static_for<8>([&](auto rr) {
                    constexpr int r = rr;
                    float pv = exp2_raw(sv[r]);
                    if constexpr (DIAG) pv = (16 * qt + r < thr[j]) ? 0.f : pv;  // q < kv
                    p[r] = pv;
                    ds[r] = pv * dp[r];
                });
#if PIN_Q & 1
                u32x2 pq = cvt8_e4m3(p, 1.0f / 256.0f);
                asm volatile("" : "+v"(pq));  // P(qt) converted in its slot (LLVM sank it to the step end, next to its use)
#else
                const u32x2 pq = cvt8_e4m3(p, 1.0f / 256.0f);
#endif
                pB[j][2 * qt] = (int)pq[0];
                pB[j][2 * qt + 1] = (int)pq[1];
                if constexpr (qt % 2 == 0) {
                    dsprev[j] = ds;
                } else {
#if PIN_Q & 2
                    DsQ z = quant_pair(dsprev[j], ds);
                    asm volatile("" : "+v"(z.a), "+v"(z.b), "+v"(z.byte));  // dS(qt - 1, qt) quantized in its slot
#else
                    const DsQ z = quant_pair(dsprev[j], ds);
#endif
                    dsB[j][2 * (qt - 1)] = (int)z.a[0];
                    dsB[j][2 * (qt - 1) + 1] = (int)z.a[1];
                    dsB[j][2 * qt] = (int)z.b[0];
                    dsB[j][2 * qt + 1] = (int)z.b[1];
                    dss[j] |= z.byte << (8 * (qt / 2));
                }
            });
            if constexpr (PREV)
                static_for<gs(qt + 1) - gs(qt)>([&](auto i) {
                    constexpr int g = gs(qt) + (int)i;
                    group(std::integral_constant<int, g>{}, ab[g], pBo, dsBo, dsso);
                });
            fence();
            if constexpr (qt + 1 < NQT) cur = nxt;
        });
    };

    i32x16 pBo[NJ], dsBo[NJ];
    int dsso[NJ];
    QTile pre_t;
    AFrag pre_a[gs(1)];
#if MIDSYNC
    tensor_wait<TDM_PER_ROUND>();  // round 0 (the q part of step 0) landed; round 1 may be in flight
    bar_signal();
    bar_wait();
    pre_t = load_qtile<0>(addr(0));
#endif
    auto run = [&](int s, auto diag_c, auto prev_c) __attribute__((always_inline)) {
#if !MIDSYNC
        tensor_wait<TDM_PER_ROUND>();  // round s landed: the q part of step s, the A part of step s - 1
        bar_signal();
        bar_wait();          // visible to every wave; every wave is done with step s - 1 (and the dV / dK of s - 2)
        issue_round(s + 2);  // q part of step s + 2 into step s - 1's q stage, A part of s + 1 into step s - 2's A stage
#endif
        const StepAddr a = addr(s);
        i32x16 pB[NJ], dsB[NJ];
        int dss[NJ] = {};
        step(s, a, qs0 + s * QS_, pB, dsB, dss, pBo, dsBo, dsso, pre_t, pre_a, diag_c, prev_c);
        static_for<NJ>([&](auto jj) {
            pBo[(int)jj] = pB[(int)jj];
            dsBo[(int)jj] = dsB[(int)jj];
            dsso[(int)jj] = dss[(int)jj];
        });
    };
#if UNROLL2
    // two steps per loop trip with ping-pong P / dS buffers (X: even steps, Y: odd steps): no per-step copies of the
    // loop-carried P / dS of the previous step (LLVM moved ~66 VGPRs per step through v_mov at the step end)
    i32x16 pY[NJ], dsY[NJ];
    int dssY[NJ];
    auto runpp = [&](int s, auto diag_c, auto prev_c, i32x16 (&pB)[NJ], i32x16 (&dsB)[NJ], int (&dss)[NJ],
                    const i32x16 (&pBq)[NJ], const i32x16 (&dsBq)[NJ], const int (&dssq)[NJ])
        __attribute__((always_inline)) {
        tensor_wait<TDM_PER_ROUND>();
        bar_signal();
        bar_wait();
        issue_round(s + 2);
        const StepAddr a = addr(s);
        static_for<NJ>([&](auto jj) { dss[(int)jj] = 0; });
        step(s, a, qs0 + s * QS_, pB, dsB, dss, pBq, dsBq, dssq, pre_t, pre_a, diag_c, prev_c);
    };
    runpp(0, std::true_type{}, std::false_type{}, pBo, dsBo, dsso, pY, dsY, dssY);  // step 0 -> X
    {
        int s = 1;
        bool lastY = false;
        if (s < nst) {
            for (;;) {
                runpp(s, std::false_type{}, std::true_type{}, pY, dsY, dssY, pBo, dsBo, dsso);  // X -> Y
                if (s + 1 >= nst) {
                    lastY = true;
                    break;
                }
                runpp(s + 1, std::false_type{}, std::true_type{}, pBo, dsBo, dsso, pY, dsY, dssY);  // Y -> X
                s += 2;
                if (s >= nst) break;
            }
        }
        if (lastY)
            static_for<NJ>([&](auto jj) {
                pBo[(int)jj] = pY[(int)jj];
                dsBo[(int)jj] = dsY[(int)jj];
                dsso[(int)jj] = dssY[(int)jj];
            });
    }
#else
    run(0, std::true_type{}, std::false_type{});
    for (int s = 1; s < nst; ++s) run(s, std::false_type{}, std::true_type{});
#endif

    // ---- the dV / dK of the last step (its A part: round nst), A fragments two groups ahead
    tensor_wait<0>();
    bar_signal();
    bar_wait();  // the last A part visible; every wave is done with the q ring
    {
        const StepAddr a = addr(nst);
        constexpr int NAB = 4;
        AFrag ab[NAB];
        ab[0] = ldA(a, std::integral_constant<int, 0>{});
        ab[1] = ldA(a, std::integral_constant<int, 1>{});
        static_for<20>([&](auto dd) {
            constexpr int d = dd;
            if constexpr (d + 2 < 20) ab[(d + 2) % NAB] = ldA(a, std::integral_constant<int, (d + 2 < 20 ? d + 2 : 0)>{});
#if KEEP_WAR & 1
            if constexpr (d >= 1 && d + 2 < 20) {
                keep(ab[(d + NAB - 1) % NAB].x);
                keep(ab[(d + NAB - 1) % NAB].s);
            }
#endif
            fence();
            group(std::integral_constant<int, d>{}, ab[d % NAB], pBo, dsBo, dsso);
            fence();
        });
    }

#if EPI_DIRECT
    // ---- epilogue: dK / dV rows straight from the accumulators (lane = kv row lr of kv16 tile j, 8 bf16 = 16 B of
    // d16 tile d at column 16 d + 8 lh): global stores, no LDS staging (16-way bank conflicts: the TDM store forbids
    // padding), no barrier, no TDM-store wait (the stores complete after the wave ends)
    {
        const int ln = lane_now(), l16 = ln & 15, hh = ln >> 4;
        static_for<NJ>([&](auto jj) {
            constexpr int j = jj;
            const int kvr = kvw + 16 * j + l16;
            if (kvr < S) {
                bf16 *pk = DK + (long long)b * dk_sb + (long long)kvr * dk_ss + (long long)h * (SPLIT ? 256 : D_QK) + 8 * hh;
                bf16 *pv = DV + (long long)b * dv_sb + (long long)kvr * dv_ss + (long long)h * (SPLIT ? 256 : D_V) + 8 * hh;
                bf16 *pp = SPLIT ? DKP + (long long)b * dkp_sb + (long long)kvr * dkp_ss + (long long)h * 64 + 8 * hh : pk;
                static_for<8>([&](auto dd) {
                    constexpr int d = dd;
                    bf16x8 w;
                    static_for<8>([&](auto rr) { w[(int)rr] = (bf16)dv[d][j][(int)rr]; });
                    *(u32x4 *)(pv + 16 * d) = __builtin_bit_cast(u32x4, w);
                });
                static_for<12>([&](auto dd) {
                    constexpr int d = dd;
                    bf16x8 w;
                    static_for<8>([&](auto rr) { w[(int)rr] = (bf16)(dk[d][j][(int)rr] * scale); });
                    bf16 *dst = (SPLIT && d >= 8) ? pp + 16 * (d - 8) : pk + 16 * d;
                    *(u32x4 *)dst = __builtin_bit_cast(u32x4, w);
                });
            }
        });
    }
    return;
#endif
#if EPI_PAD
    // ---- epilogue without bank conflicts: each wave stages its 32 dK / dV rows into padded rows (pitch + 16 B: the
    // 16 rows of a ds_store hit different banks; dense rows 256 / 384 B apart put all 16 on the same banks), re-packs
    // them densely (32 consecutive 16-byte chunks per copy: conflict-free) and TDM-stores its own rows; no barrier.
    // Padded staging in the q ring (idle); dense dK (+ rope) after it in the q ring, dense dV in the A stage no TDM op
    // and no final-phase read uses (step nst's: NULL past the end).
    {
        constexpr int XDK = 2 * D_QK, XDV = 2 * D_V;
        constexpr int XDN = SPLIT ? 2 * 128 : XDK;
        constexpr int PDN = XDN + 16, PDV = XDV + 16, PDP = 128 + 16;
        constexpr int P_DK = 0, P_DV = P_DK + KB * PDN, P_DP = P_DV + KB * PDV;
        constexpr int P_END = P_DP + (SPLIT ? KB * PDP : 0);
        constexpr int D_DK = (P_END + 127) / 128 * 128, D_DP = D_DK + KB * XDN;
        static_assert(D_DP + (SPLIT ? KB * 128 : 0) <= ARING, "padded staging + dense dK inside the q ring");
        static_assert(KB * XDV <= ASTG, "dense dV inside one A stage");
        const unsigned d_dv = (unsigned)(ARING + (nst % NSTG) * ASTG);
        const int ln = lane_now();
        static_for<NJ>([&](auto jj) {
            constexpr int j = jj;
            const unsigned r = (unsigned)(KBW * wv + 16 * j + (ln & 15)), hh = (unsigned)(ln >> 4);
            const unsigned dkr = lds0 + P_DK + r * PDN + 16 * hh, dvr = lds0 + P_DV + r * PDV + 16 * hh;
            const unsigned dpr = lds0 + P_DP + r * PDP + 16 * hh;
            static_for<12>([&](auto dd) {
                constexpr int d = dd;
                bf16x8 w;
                static_for<8>([&](auto rr) { w[(int)rr] = (bf16)(dk[d][j][(int)rr] * scale); });
                const unsigned a = (SPLIT && d >= 8) ? dpr + 32 * (d - 8) : dkr + 32 * d;
                *(__attribute__((address_space(3))) u32x4 *)(uintptr_t)a = __builtin_bit_cast(u32x4, w);
            });
            static_for<8>([&](auto dd) {
                constexpr int d = dd;
                bf16x8 w;
                static_for<8>([&](auto rr) { w[(int)rr] = (bf16)dv[d][j][(int)rr]; });
                *(__attribute__((address_space(3))) u32x4 *)(uintptr_t)(dvr + 32 * d) = __builtin_bit_cast(u32x4, w);
            });
        });
        // re-pack this wave's rows (KBW from row KBW wv): chunk c = row * (X / 16) + col, 32 per copy (a wave's own
        // LDS operations complete in order: no wait between its padded stores and these loads)
        auto repack = [&](auto xc, auto pc, unsigned src, unsigned dst) __attribute__((always_inline)) {
            constexpr int X = decltype(xc)::value, PP = decltype(pc)::value, CPR = X / 16, NCH = KBW * CPR;
            static_assert(NCH % 32 == 0, "whole copies");
            const unsigned r0 = (unsigned)(KBW * wv);
            static_for<NCH / 32>([&](auto kk) {
                const int c = 32 * (int)kk + ln;
                const unsigned row = r0 + (unsigned)(c / CPR), col = (unsigned)(c % CPR);
                const u32x4 v = *(const __attribute__((address_space(3))) u32x4 *)(uintptr_t)(src + row * PP + 16 * col);
                *(__attribute__((address_space(3))) u32x4 *)(uintptr_t)(dst + row * X + 16 * col) = v;
            });
        };
        repack(std::integral_constant<int, XDN>{}, std::integral_constant<int, PDN>{}, lds0 + P_DK, lds0 + D_DK);
        repack(std::integral_constant<int, XDV>{}, std::integral_constant<int, PDV>{}, lds0 + P_DV, lds0 + d_dv);
        if constexpr (SPLIT)
            repack(std::integral_constant<int, 128>{}, std::integral_constant<int, PDP>{}, lds0 + P_DP, lds0 + D_DP);
        asm volatile("s_wait_dscnt 0x0" ::: "memory");
        const int r0 = KBW * wv, rw = min(KBW, S - kv0 - r0);
        if (rw > 0) {
            const unsigned rows = (unsigned)rw;
            const long long kr = (long long)(kv0 + r0);
            const unsigned long long gk = (unsigned long long)(uintptr_t)(DK + (long long)b * dk_sb + kr * dk_ss +
                                                                          (long long)h * (SPLIT ? 256 : D_QK));
            tdm_store(tdm_g0(lds0 + D_DK + (unsigned)(r0 * XDN), gk),
                      tdm_g1(tdm_w0(0, 0, 0), rows, XDN, KBW, 2ull * (unsigned long long)dk_ss));
            const unsigned long long gv = (unsigned long long)(uintptr_t)(DV + (long long)b * dv_sb + kr * dv_ss +
                                                                          (long long)h * (SPLIT ? 256 : D_V));
            tdm_store(tdm_g0(lds0 + d_dv + (unsigned)(r0 * XDV), gv),
                      tdm_g1(tdm_w0(0, 0, 0), rows, XDV, KBW, 2ull * (unsigned long long)dv_ss));
            if constexpr (SPLIT) {
                const unsigned long long gp = (unsigned long long)(uintptr_t)(DKP + (long long)b * dkp_sb +
                                                                              kr * dkp_ss + (long long)h * 64);
                tdm_store(tdm_g0(lds0 + D_DP + (unsigned)(r0 * 128), gp),
                          tdm_g1(tdm_w0(0, 0, 0), rows, 128, KBW, 2ull * (unsigned long long)dkp_ss));
            }
        }
        tensor_wait<0>();
        return;
    }
#endif
    // ---- epilogue: dK / dV rows (lane = kv column, rows d = 8 lh + r of each d16 tile) as bf16 into LDS (the q ring,
    // idle since the barrier above), then one TDM store each (wave 0: dK, 384 B rows; wave 1: dV, 256 B rows)
    constexpr int XDK = 2 * D_QK, XDV = 2 * D_V, O_SDK = 0, O_SDV = KB * XDK;
    static_assert(O_SDV + KB * XDV <= ARING, "staging inside the q ring");
    // SPLIT: the nope part as 256-byte rows at O_SDK, the rope part as 128-byte rows at O_SDP (same total bytes)
    constexpr int XDN = SPLIT ? 2 * 128 : XDK, O_SDP = O_SDK + KB * 2 * 128;
    {
        const int ln = lane_now();
        static_for<NJ>([&](auto jj) {
            constexpr int j = jj;
            const unsigned r = (unsigned)(KBW * wv + 16 * j + (ln & 15)), hh = (unsigned)(ln >> 4);
            const unsigned dkr = lds0 + O_SDK + r * XDN + 16 * hh, dvr = lds0 + O_SDV + r * XDV + 16 * hh;
            const unsigned dpr = lds0 + O_SDP + r * 128 + 16 * hh;
            static_for<12>([&](auto dd) {
                constexpr int d = dd;
                bf16x8 w;
                static_for<8>([&](auto rr) { w[(int)rr] = (bf16)(dk[d][j][(int)rr] * scale); });
                const unsigned a = (SPLIT && d >= 8) ? dpr + 32 * (d - 8) : dkr + 32 * d;
                *(__attribute__((address_space(3))) u32x4 *)(uintptr_t)a = __builtin_bit_cast(u32x4, w);
            });
            static_for<8>([&](auto dd) {
                constexpr int d = dd;
                bf16x8 w;
                static_for<8>([&](auto rr) { w[(int)rr] = (bf16)dv[d][j][(int)rr]; });
                *(__attribute__((address_space(3))) u32x4 *)(uintptr_t)(dvr + 32 * d) = __builtin_bit_cast(u32x4, w);
            });
        });
    }
#if EPI_WAVE
    {
        // the wave's own staged rows (KBW from row KBW wv): its ds_stores done, then its TDM stores; no barrier
        asm volatile("s_wait_dscnt 0x0" ::: "memory");
        const int r0 = KBW * wv, rw = min(KBW, S - kv0 - r0);
        if (rw > 0) {
            const unsigned rows = (unsigned)rw;
            const long long kr = (long long)(kv0 + r0);
            const unsigned long long gk = (unsigned long long)(uintptr_t)(DK + (long long)b * dk_sb + kr * dk_ss +
                                                                          (long long)h * (SPLIT ? 256 : D_QK));
            tdm_store(tdm_g0(lds0 + O_SDK + (unsigned)(r0 * XDN), gk),
                      tdm_g1(tdm_w0(0, 0, 0), rows, XDN, KBW, 2ull * (unsigned long long)dk_ss));
            const unsigned long long gv = (unsigned long long)(uintptr_t)(DV + (long long)b * dv_sb + kr * dv_ss +
                                                                          (long long)h * (SPLIT ? 256 : D_V));
            tdm_store(tdm_g0(lds0 + O_SDV + (unsigned)(r0 * XDV), gv),
                      tdm_g1(tdm_w0(0, 0, 0), rows, XDV, KBW, 2ull * (unsigned long long)dv_ss));
            if constexpr (SPLIT) {
                const unsigned long long gp = (unsigned long long)(uintptr_t)(DKP + (long long)b * dkp_sb +
                                                                              kr * dkp_ss + (long long)h * 64);
                tdm_store(tdm_g0(lds0 + O_SDP + (unsigned)(r0 * 128), gp),
                          tdm_g1(tdm_w0(0, 0, 0), rows, 128, KBW, 2ull * (unsigned long long)dkp_ss));
            }
        }
    }
#else
    bar_signal();
    bar_wait();
    {
        const unsigned rows = (unsigned)min(KB, S - kv0);
        if (wv == 0) {
            if constexpr (SPLIT) {
                // DK: the kv gradient's rows (head stride as dV's view: 256 elements), DKP: [b, s, h, 64]
                const unsigned long long g = (unsigned long long)(uintptr_t)(DK + (long long)b * dk_sb +
                                                                             (long long)kv0 * dk_ss + (long long)h * 256);
                tdm_store(tdm_g0(lds0 + O_SDK, g), tdm_g1(tdm_w0(0, 0, 0), rows, XDN, KB, 2ull * (unsigned long long)dk_ss));
            } else {
                const unsigned long long g = (unsigned long long)(uintptr_t)(DK + (long long)b * dk_sb +
                                                                             (long long)kv0 * dk_ss + (long long)h * D_QK);
                tdm_store(tdm_g0(lds0 + O_SDK, g), tdm_g1(tdm_w0(0, 0, 0), rows, XDK, KB, 2ull * (unsigned long long)dk_ss));
            }
        } else if (wv == 1) {
            const unsigned long long g = (unsigned long long)(uintptr_t)(DV + (long long)b * dv_sb + (long long)kv0 * dv_ss +
                                                                         (long long)h * (SPLIT ? 256 : D_V));
            tdm_store(tdm_g0(lds0 + O_SDV, g), tdm_g1(tdm_w0(0, 0, 0), rows, XDV, KB, 2ull * (unsigned long long)dv_ss));
        } else if (SPLIT && wv == 2) {
            const unsigned long long g = (unsigned long long)(uintptr_t)(DKP + (long long)b * dkp_sb +
                                                                         (long long)kv0 * dkp_ss + (long long)h * 64);
            tdm_store(tdm_g0(lds0 + O_SDP, g), tdm_g1(tdm_w0(0, 0, 0), rows, 128, KB, 2ull * (unsigned long long)dkp_ss));
        }
    }
#endif
#if !NO_FINAL_WAIT
    tensor_wait<0>();
#endif
}

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
    dkdv_body<false>(Q8, QS, K8, KS, V8, VSR, DO8, DOS, DO8T, DOTS, Q8T, QTS, LSE2, DELTA, DK, DV, scale, S, H, B, dk_sb,
                     dk_ss, dv_sb, dv_ss, nullptr, 0, 0, smem);
}

// the fused-RoPE path's variant: DK / DV views of the kv gradient [b, s, h, 256] (DK at d 0, DV at d 128; the head
// stride 256 is fixed), dK's rope columns to DKP [b, s, h, 64] (dkp_sb / dkp_ss its b / s strides)
extern "C" __global__ __attribute__((amdgpu_flat_work_group_size(128, 128), amdgpu_waves_per_eu(1, 1)))
void k_dkdv_mxr(const unsigned char *__restrict__ Q8, const unsigned char *__restrict__ QS,
                const unsigned char *__restrict__ K8, const unsigned char *__restrict__ KS,
                const unsigned char *__restrict__ V8, const unsigned char *__restrict__ VSR,
                const unsigned char *__restrict__ DO8, const unsigned char *__restrict__ DOS,
                const unsigned char *__restrict__ DO8T, const unsigned char *__restrict__ DOTS,
                const unsigned char *__restrict__ Q8T, const unsigned char *__restrict__ QTS,
                const float *__restrict__ LSE2, const float *__restrict__ DELTA, bf16 *__restrict__ DK,
                bf16 *__restrict__ DV, float scale, int S, int H, int B, long long dk_sb, long long dk_ss,
                long long dv_sb, long long dv_ss, bf16 *__restrict__ DKP, long long dkp_sb, long long dkp_ss) {
    __shared__ __attribute__((aligned(1024))) unsigned char smem[LDS_B];
    dkdv_body<true>(Q8, QS, K8, KS, V8, VSR, DO8, DOS, DO8T, DOTS, Q8T, QTS, LSE2, DELTA, DK, DV, scale, S, H, B, dk_sb,
                    dk_ss, dv_sb, dv_ss, DKP, dkp_sb, dkp_ss, smem);
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
template <typename T>
MX_INLINE void dq_keep(const T &x) {
    asm volatile("" ::"v"(x));
}
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
#if DQ_ORDER == 1
    // snake: (b, h) groups alternate longest-first / shortest-first (the k-th workgroup of an XCD: group k / nqb)
    const int grp = (int)((blockIdx.x + nqb * (blockIdx.y + H * blockIdx.z)) >> 3) / nqb;
    const int qb = (grp & 1) ? blk.slot : nqb - 1 - blk.slot;
#elif DQ_ORDER == 2
    // two passes per XCD: every (b, h)'s long half (q blocks nqb - 1 .. nqb / 2) first, then the short halves (the
    // dispatch tail made of short workgroups; each (b, h)'s K / V read in both passes)
    int qb, h, b;
    if ((B * H) % 8 == 0 && nqb % 2 == 0) {
        const int L = (int)(blockIdx.x + nqb * (blockIdx.y + H * blockIdx.z)), k = L >> 3, xc = L & 7;
        const int nbx = B * H / 8, half = nqb / 2, k1 = k % (nbx * half);
        const int bhx = xc + 8 * (k1 / half), sl = k1 % half;
        qb = k < nbx * half ? nqb - 1 - sl : half - 1 - sl;
        h = bhx % H;
        b = bhx / H;
    } else {
        qb = nqb - 1 - blk.slot;
        h = blk.h;
        b = blk.b;
    }
#else
    const int qb = nqb - 1 - blk.slot;
#endif
#if DQ_ORDER != 2
    const int h = blk.h, b = blk.b;
#endif
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
        // DC 0: below the diagonal; 1: kv tile n - 2 (q tiles i < 2 masked, i >= 2 fully kept); 2: kv tile n - 1 (q tiles
        // i < 2 fully masked: skipped with DQ_DSTAT, i >= 2 masked). Without DQ_DSTAT every q tile of DC 1 / 2 is masked
        constexpr int DC = (int)decltype(diag_c)::value;
        constexpr bool DIAG = DC != 0;
        if (DQ_WAIT0 || t > 0) {  // DQ_WAIT0: also at t = 0 (tile 0 landed already: one extra barrier; one-block loop)
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
            if constexpr (m + 1 < NKV / 2 && DQ_KVMID > 0) {
                // loaded inside the q-tile loop below
            } else if constexpr (m + 1 < NKV / 2) {
#if DQ_KEEP & 2
                if constexpr (m >= 1) {
                    const KVPair old = kvb[(m + 1) % 2];  // pair m - 1: read by the previous region's WMMAs
                    ldkv(std::integral_constant<int, m + 1>{}, kvb[(m + 1) % 2]);
                    static_for<2>([&](auto uu) {
                        dq_keep(old.kx[(int)uu]);
                        dq_keep(old.va[(int)uu]);
                        dq_keep(old.ksc[(int)uu]);
                        dq_keep(old.vsc[(int)uu]);
                    });
                } else {
                    ldkv(std::integral_constant<int, m + 1>{}, kvb[(m + 1) % 2]);
                }
#else
                ldkv(std::integral_constant<int, m + 1>{}, kvb[(m + 1) % 2]);
#endif
            } else {
                ab[0] = lda(std::integral_constant<int, 0>{});
                ab[1] = lda(std::integral_constant<int, 1>{});
            }
            fence();
            const KVPair &kv = kvb[m % 2];
            static_for<QTW>([&](auto ii) {
                constexpr int i = ii;
                if constexpr (DQ_KVMID > 0 && i == DQ_KVMID && m + 1 < NKV / 2) {
                    fence();  // pair m - 1's WMMAs are long done: the loads may take its registers
                    ldkv(std::integral_constant<int, (m + 1 < NKV / 2 ? m + 1 : 0)>{}, kvb[(m + 1) % 2]);
                }
                constexpr bool SKIPQ = DQ_DSTAT && DC == 2 && i < 2;
                constexpr bool MASKQ = DQ_DSTAT ? ((DC == 1 && i < 2) || (DC == 2 && i >= 2)) : DIAG;
                if constexpr (!SKIPQ) {
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
                        if constexpr (MASKQ) pv = (16 * j + r > thr) ? 0.f : pv;
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
                }
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
#if DQ_KEEP & 1
            if constexpr (dt >= 1 && dt + 2 < 12) {
                dq_keep(ab[(dt + 3) % 4].x);  // fragment dt - 1: read by the previous group's WMMAs
                dq_keep(ab[(dt + 3) % 4].s);
            }
#endif
            fence();
            const AF &af = ab[dt % 4];
            static_for<QTW>([&](auto ii) {
                constexpr int i = ii;
                if constexpr (DQ_DSTAT && DC == 2 && i < 2) {
                } else if constexpr (dt % 2 == 0)
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
    for (int t = 0; t < n - 2; ++t) tile(t, std::integral_constant<int, 0>{});
    tile(n - 2, std::integral_constant<int, 1>{});
    tile(n - 1, std::integral_constant<int, 2>{});
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
