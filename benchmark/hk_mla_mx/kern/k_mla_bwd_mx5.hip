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
#ifndef DQ_SKEW
#define DQ_SKEW 1  // dq epilogue staging as 16 dense regions (rows q % 16 = j) whose bases step by rows * 384 + 16 B: a
                   // ds_store's 16 rows hit distinct bank groups (2-way: the minimum) instead of two (8-way); one 2-D TDM
                   // store per region (global row stride 16 rows), spread over the waves
#endif
#ifndef PF_CODE
#define PF_CODE 1  // prefetch the kernel's code from the entry at the start (the cold first workgroup per CU): bit 0 the dq
                   // kernels (k_dq_mxr -2.7% in the package's fwd + bwd sequence), bit 1 dk / dv (no gain); the package
                   // checks that the prefetched range ends inside the code object (else it builds with PF_CODE 0)
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
#define DQ_ORDER 4  // dq workgroup order: 0 longest q block first per (b, h); 1 snake; n >= 2: n passes, longest parts first
#endif
#ifndef DQ_WAIT0
#define DQ_WAIT0 1  // k_dq_mx (not k_dq_mxr): the per-tile TDM wait + barrier also before tile 0 (the tile loop becomes one
                    // basic block; 11-06 with the skewed staging: k_dq_mx 0.8570 vs 0.8653 ms, k_dq_mxr 0.8770 vs 0.8717)
#endif
#ifndef DQ_DSTAT
#define DQ_DSTAT 1  // dq: static mask classes on the two diagonal kv tiles (fully masked q tiles skipped, kept ones unmasked)
#endif
#ifndef NO_FINAL_WAIT
#define NO_FINAL_WAIT 0  // timing probe only: dkdv ends without waiting for its TDM stores (LDS may be reused early)
#endif
#ifndef EPI_SKEW
#define EPI_SKEW 1  // dkdv epilogue staging as 16 dense regions per output (rows r % 16 = j: KB / 16 rows each) whose bases
                    // step by those rows + 16 B: 2-way ds_stores instead of 8 / 16-way; a 2-D TDM store per region and
                    // output (global row stride 16 rows), spread over the waves
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
#ifndef LTDT_DPP
// dkdv: a q16 tile's lse / delta C tuples (rows 8 lh + r, r < 8) from one ds_load_b32 each (lane l of a 16-lane row holds
// row 8 lh + l % 8) and 8 DPP row_share broadcasts, instead of 2 + 2 ds_load_b128 (the kernel is LDS-bandwidth bound:
// four waves read every q-side byte)
#define LTDT_DPP 0
#endif
#ifndef DSO_LATE
// b7 (k_dkdv_mxd): where slot qt's two dS stores issue: 0 right after the conversion (waits for the WMMA -> exp -> mul ->
// cvt chain: ~23 cycles per store pair, ATT; 1.543 ms), 1 after the slot's dV / dK WMMAs (1.542), 2 after the slot's
// closing fence (1.531), 3 the step's 16 stores together at its end (1.486; ATT: interleaved with the LDS stream the
// stores trigger sporadic 400-1100-cycle ds_load stalls), 4 slots 0..3 after slot 3 and 4..7 at the end (1.644, spills)
#define DSO_LATE 3
#endif
#ifndef DKDV_SGB
// dkdv: sched_group_barrier issue pattern per slot (0 off): 1 [W, 2 T, 4 V, 2 DS] x 12; 2 [W, T, 2 V, DS] x 16 then the
// rest; 3 the S / dP WMMAs first ([W, T, T] x 6) then [W, 3 V, 2 DS] x 6
#define DKDV_SGB 0
#endif
#ifndef DKDV_SCPROBE
#define DKDV_SCPROBE 0  // timing probe only (wrong dK / dV): 1 the q tiles' Q / dO scale words and the A fragments' scale
                        // words loaded for tile / group 0 only (the cost of ~26 b32 LDS loads per step)
#endif
#ifndef DSO_PROBE
#define DSO_PROBE 0  // timing probes only: 1 every step stores to the block's diagonal tile (L2-resident), 2 no stores
#endif
#ifndef DK_BF16
#define DK_BF16 0  // b7 (DSO): dK's dS operand MX-quantized (along q) from the bf16 dS that is stored (as dQ's), not f32
#endif
#ifndef DSO_NT
#define DSO_NT 0  // b7: the dS stores non-temporal (th:TH_STORE_NT: streamed, not kept in the caches)
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
template <int... I>
MX_INLINE f32x8 bcast8_impl(uint32_t x, std::integer_sequence<int, I...>) {
    return f32x8{as_f((uint32_t)__builtin_amdgcn_mov_dpp((int)x, 0x150 + I, 0xf, 0xf, false))...};
}
// lane l gets lane 16 (l / 16) + r's x as element r (DPP row_share r)
MX_INLINE f32x8 bcast8(uint32_t x) { return bcast8_impl(x, std::make_integer_sequence<int, 8>{}); }
MX_INLINE f32x8 f8(u32x4 a, u32x4 b) {
    return f32x8{as_f(a[0]), as_f(a[1]), as_f(a[2]), as_f(a[3]), as_f(b[0]), as_f(b[1]), as_f(b[2]), as_f(b[3])};
}
typedef short s16x2 __attribute__((ext_vector_type(2)));
#ifndef DQG_AMAX
#define DQG_AMAX 1  // |x| max of 16 bf16: 0 AND then max on the magnitudes; 1 signed max (largest positive) and unsigned max
                    // (the negative of largest magnitude) trees, one AND at the end
#endif
#ifndef DQG_ORDER
#define DQG_ORDER 1  // DQG_V 1: 0 the 12 K8T fragments loaded first; 1 the first two q tiles' transposed loads first, the
                     // K8T fragments next to the first quantization
#endif
typedef unsigned short u16x2 __attribute__((ext_vector_type(2)));
// |x| max of 16 bf16 values (two u32x4) on the magnitude bits (their integer order is the float order), as a float
MX_INLINE float amax16_bf16(const u32x4 &x0, const u32x4 &x1) {
    if constexpr (DQG_AMAX == 1) {
        // as int16 a positive bf16 orders by magnitude and every negative is below it; as uint16 every negative
        // (0x8000 | mag) is above every positive and orders by magnitude
        auto si = [](unsigned x) { return __builtin_bit_cast(s16x2, x); };
        auto su = [](unsigned x) { return __builtin_bit_cast(u16x2, x); };
        auto mx3 = [](auto a, auto b, auto c) { return __builtin_elementwise_max(__builtin_elementwise_max(a, b), c); };
        const s16x2 p = mx3(mx3(si(x0[0]), si(x0[1]), si(x0[2])), mx3(si(x0[3]), si(x1[0]), si(x1[1])),
                            __builtin_elementwise_max(si(x1[2]), si(x1[3])));
        const u16x2 n = mx3(mx3(su(x0[0]), su(x0[1]), su(x0[2])), mx3(su(x0[3]), su(x1[0]), su(x1[1])),
                            __builtin_elementwise_max(su(x1[2]), su(x1[3])));
        // p: the largest positive (int16 < 0 when there is none: clamped to 0); n & 0x7fff: the largest magnitude among
        // the negatives when there is one, else the largest positive again
        const u16x2 pm = __builtin_bit_cast(u16x2, __builtin_elementwise_max(p, s16x2{0, 0}));
        const u16x2 nm = __builtin_bit_cast(u16x2, __builtin_bit_cast(unsigned, n) & 0x7fff7fffu);
        const u16x2 m2 = __builtin_elementwise_max(pm, nm);
        // the two halves' max by 32-bit ops (a rotate + one packed max: no 16-bit register halves, which asm_resched's
        // parser does not read), the result in the high half = the f32 bits of the bf16 max
        const unsigned m2u = __builtin_bit_cast(unsigned, m2);
        const u16x2 mm = __builtin_elementwise_max(m2, __builtin_bit_cast(u16x2, __builtin_amdgcn_alignbit(m2u, m2u, 16)));
        return __builtin_bit_cast(float, __builtin_bit_cast(unsigned, mm) & 0xffff0000u);
    }
    auto ab = [](unsigned x) { return __builtin_bit_cast(s16x2, x & 0x7fff7fffu); };
    const s16x2 m0 = __builtin_elementwise_max(__builtin_elementwise_max(ab(x0[0]), ab(x0[1])),
                                               __builtin_elementwise_max(ab(x0[2]), ab(x0[3])));
    const s16x2 m1 = __builtin_elementwise_max(__builtin_elementwise_max(ab(x1[0]), ab(x1[1])),
                                               __builtin_elementwise_max(ab(x1[2]), ab(x1[3])));
    const s16x2 mm = __builtin_elementwise_max(m0, m1);
    const int mb = mm[0] > mm[1] ? (int)mm[0] : (int)mm[1];
    return __builtin_bit_cast(float, (unsigned)mb << 16);
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
// rung b7's materialized dS (bf16): per (b, h) the causal 128 x 128 tiles (kv block kb <= q block qb) at index
// qb (qb + 1) / 2 + kb (a q block's tiles consecutive), DS_TILE bytes each. Inside a tile, chunk c = 16 w + 2 qt + j
// (512 B) holds kv16 tile 2 w + j x q16 tile qt as k_dkdv_mx's wave w computes it in slot qt: lane L = kv % 16 + 16 (q %
// 16 / 8) at 16 L, its 8 q values (q % 8 = r) at 16 L + 2 r
constexpr int DS_TILE = 128 * 128 * 2;
__host__ __device__ constexpr int ds_tile_index(int kb, int qb) { return qb * (qb + 1) / 2 + kb; }
typedef __attribute__((address_space(1))) unsigned char g_u8;
typedef __attribute__((address_space(1))) u32x4 g_u32x4;
struct DsOut {
    g_u8 *tile;     // uniform: this wave's part of the step's dS tile
    unsigned voff;  // the lane's 16-byte slot of a chunk
};
MX_INLINE void ds_store16(g_u8 *p, u32x4 v) {
    if constexpr (DSO_PROBE == 2) {
        asm volatile("" ::"v"(v));
        return;
    }
    if constexpr (DSO_NT)
        __builtin_nontemporal_store(v, (g_u32x4 *)p);
    else
        *(g_u32x4 *)p = v;
}
__host__ __device__ constexpr unsigned long long ds_tiles_per_bh(int S) {
    return (unsigned long long)(S / 128) * (unsigned long long)(S / 128 + 1) / 2;
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
    t.qsc = (int)lds_b32(lds_ptr(a.qs), DKDV_SCPROBE ? 0 : QT * 16 * 8);
    t.dsc = (int)lds_b32(lds_ptr(a.dos), DKDV_SCPROBE ? 0 : QT * 16 * 4);
#if LTDT_DPP
    t.lt = bcast8(lds_b32(lds_ptr(a.lse), QT * 64));
    t.dt = bcast8(lds_b32(lds_ptr(a.del), QT * 64));
#elif !LTDT_G
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
            (int)lds_b32(lds_ptr(sbase), DKDV_SCPROBE ? 0 : (D / 2) * 128)};
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
// fused-RoPE path: DK = the kv gradient [.., 0:128], DKP a per-head [b, s, h, 64] buffer), else 192-column dK rows.
// DSO (rung b7): also store dS = P (dP - delta) as bf16 to DSB in the tile layout k_dqg_mx reads (ds_tile_index)
template <bool SPLIT, bool DSO = false>
__device__ __forceinline__ void dkdv_body(
    const unsigned char *__restrict__ Q8, const unsigned char *__restrict__ QS, const unsigned char *__restrict__ K8,
    const unsigned char *__restrict__ KS, const unsigned char *__restrict__ V8, const unsigned char *__restrict__ VSR,
    const unsigned char *__restrict__ DO8, const unsigned char *__restrict__ DOS, const unsigned char *__restrict__ DO8T,
    const unsigned char *__restrict__ DOTS, const unsigned char *__restrict__ Q8T, const unsigned char *__restrict__ QTS,
    const float *__restrict__ LSE2, const float *__restrict__ DELTA, bf16 *__restrict__ DK, bf16 *__restrict__ DV,
    float scale, int S, int H, int B, long long dk_sb, long long dk_ss, long long dv_sb, long long dv_ss,
    bf16 *__restrict__ DKP, long long dkp_sb, long long dkp_ss, unsigned char *__restrict__ DSB, unsigned char *smem) {
    lock_simd();
    if constexpr (PF_CODE & 2) prefetch_code<6>();  // 24 KB: k_dkdv_mx(r) 29.4 KB of code
    const uint32_t lds0 = lds_addr(smem);
    const int tid = (int)__builtin_amdgcn_workitem_id_x();
    const int wv = wave_id(tid);
    const int lane = tid & 31, lr = lane & 15, lh = lane >> 4;
    const BlockOf blk = xcd_block((S + KB - 1) / KB, H, B);
#if DKDV_ORDER >= 2
    // DKDV_ORDER passes per XCD: every (b, h)'s first part of kv blocks (the most steps) first, then the next parts
    // (the dispatch tail made of short workgroups)
    int kb, h, b;
    {
        const int nkb = (S + KB - 1) / KB;
        if ((B * H) % 8 == 0 && nkb % DKDV_ORDER == 0) {
            const int L = (int)(blockIdx.x + nkb * (blockIdx.y + H * blockIdx.z)), k = L >> 3, xc = L & 7;
            const int nbx = B * H / 8, part = nkb / DKDV_ORDER, pass = k / (nbx * part), k1 = k % (nbx * part);
            const int bhx = xc + 8 * (k1 / part), sl = k1 % part;
            kb = pass * part + sl;
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
        const unsigned lo = LTDT_DPP ? (unsigned)(4 * (8 * hh + (ln & 7))) : (unsigned)(32 * hh);
        return {pin(rq + O_Q + 64 * hh),         pin(rq + O_Q + 128 + 32 * hh), pin(sq + O_QS + r16 * 8 + 4 * hh),
                pin(r1 + O_DO + 64 * hh),        pin(sq + O_DOS + r16 * 4),     pin(sq + O_LSE + lo),
                pin(sq + O_DEL + lo),            pin(ra + O_DOT + 64 * hh),     pin(sa + O_DOTS + ln * 4),
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
    auto step = [&](int s, const DsOut &dsp, const StepAddr &a, int q0, i32x16 (&pB)[NJ], i32x16 (&dsB)[NJ], int (&dss)[NJ],
                    const i32x16 (&pBo)[NJ], const i32x16 (&dsBo)[NJ], const int (&dsso)[NJ], QTile &pre_t,
                    AFrag (&pre_a)[3], auto diag_c, auto prev_c) __attribute__((always_inline)) {
        constexpr bool DIAG = decltype(diag_c)::value, PREV = decltype(prev_c)::value;
        // the causal mask as constant-vs-threshold compares: q < kv iff 16 qt + r < thr[j] = kv - q0 - 8 lh
        int thr[NJ];
        static_for<NJ>([&](auto jj) { thr[(int)jj] = kvw + 16 * (int)jj + lr - q0 - 8 * lh; });
        f32x8 dsprev[NJ];
        [[maybe_unused]] u32x4 dswprev[NJ];  // DK_BF16: the even slot's bf16 dS
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
        // DSO_LATE 3: the step's dS stored at its end; 4: slots 0..3 after slot 3, slots 4..7 at the end
        [[maybe_unused]] u32x4 dso_all[DSO_LATE >= 3 ? NQT : 1][NJ];
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
            [[maybe_unused]] u32x4 dso_w[NJ];
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
                [[maybe_unused]] u32x4 wq = {0u, 0u, 0u, 0u};
                if constexpr (DSO) {
                    // the chunk (wv, qt, j): this lane's 8 q values (rows 8 lh + r) of kv column lr, as bf16 (one vector
                    // conversion: scalar ones let the SLP vectorizer re-schedule the block, moving slots' dV / dK
                    // WMMAs to the step end, 96 VGPRs spilled)
                    wq = __builtin_bit_cast(u32x4, __builtin_convertvector(ds, bf16x8));
                    if constexpr (DSO_LATE == 0)
                        ds_store16(dsp.tile + dsp.voff + (2 * qt + j) * 512, wq);
                    else
                        dso_w[j] = wq;
                }
#if PIN_Q & 1
                u32x2 pq = cvt8_e4m3(p, 1.0f / 256.0f);
                asm volatile("" : "+v"(pq));  // P(qt) converted in its slot (LLVM sank it to the step end, next to its use)
#else
                const u32x2 pq = cvt8_e4m3(p, 1.0f / 256.0f);
#endif
                pB[j][2 * qt] = (int)pq[0];
                pB[j][2 * qt + 1] = (int)pq[1];
                if constexpr (DSO && DK_BF16) {
                    // dK's dS block (q 32 (qt / 2) .. + 31 of kv column lr): the bf16 values as stored (dQ's recipe)
                    if constexpr (qt % 2 == 0) {
                        dswprev[j] = wq;
                    } else {
                        const float am = amax16_bf16(dswprev[j], wq);
                        const int byte = e8m0_byte(__builtin_fmaxf(am, peer(am)));
                        const float sc = e8m0_scale(byte);
                        const u32x2 c0 = __builtin_amdgcn_cvt_scalef32_pk8_fp8_bf16(__builtin_bit_cast(bf16x8, dswprev[j]), sc);
                        const u32x2 c1 = __builtin_amdgcn_cvt_scalef32_pk8_fp8_bf16(__builtin_bit_cast(bf16x8, wq), sc);
                        dsB[j][2 * (qt - 1)] = (int)c0[0];
                        dsB[j][2 * (qt - 1) + 1] = (int)c0[1];
                        dsB[j][2 * qt] = (int)c1[0];
                        dsB[j][2 * qt + 1] = (int)c1[1];
                        dss[j] |= byte << (8 * (qt / 2));
                    }
                } else if constexpr (qt % 2 == 0) {
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
            auto dso_store = [&]() __attribute__((always_inline)) {
                if constexpr (DSO && DSO_LATE != 0)
                    static_for<NJ>([&](auto jj) {
                        ds_store16(dsp.tile + dsp.voff + (2 * qt + (int)jj) * 512, dso_w[(int)jj]);
                    });
            };
            if constexpr (DSO_LATE == 1) dso_store();
            if constexpr (DKDV_SGB == 1) {
                static_for<12>([&](auto) {
                    __builtin_amdgcn_sched_group_barrier(0x8, 1, 0);
                    __builtin_amdgcn_sched_group_barrier(0x400, 2, 0);
                    __builtin_amdgcn_sched_group_barrier(0x2, 4, 0);
                    __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
                });
            } else if constexpr (DKDV_SGB == 2) {
                static_for<16>([&](auto) {
                    __builtin_amdgcn_sched_group_barrier(0x8, 1, 0);
                    __builtin_amdgcn_sched_group_barrier(0x400, 1, 0);
                    __builtin_amdgcn_sched_group_barrier(0x2, 2, 0);
                    __builtin_amdgcn_sched_group_barrier(0x100, 1, 0);
                });
            } else if constexpr (DKDV_SGB == 3) {
                static_for<6>([&](auto) {
                    __builtin_amdgcn_sched_group_barrier(0x8, 1, 0);
                    __builtin_amdgcn_sched_group_barrier(0x400, 2, 0);
                });
                static_for<6>([&](auto) {
                    __builtin_amdgcn_sched_group_barrier(0x8, 1, 0);
                    __builtin_amdgcn_sched_group_barrier(0x2, 3, 0);
                    __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
                });
            }
            fence();
            if constexpr (DSO_LATE == 2) dso_store();
            if constexpr (DSO && DSO_LATE >= 3) static_for<NJ>([&](auto jj) { dso_all[DSO_LATE >= 3 ? qt : 0][(int)jj] = dso_w[(int)jj]; });
            if constexpr (DSO && DSO_LATE == 4 && qt == 3)
                static_for<4>([&](auto qq) {
                    static_for<NJ>([&](auto jj) {
                        ds_store16(dsp.tile + dsp.voff + (2 * (int)qq + (int)jj) * 512, dso_all[(int)qq][(int)jj]);
                    });
                });
            if constexpr (qt + 1 < NQT) cur = nxt;
        });
        if constexpr (DSO && DSO_LATE >= 3)
            static_for<NQT>([&](auto qq) {
                if constexpr (DSO_LATE == 3 || (int)qq >= 4)
                    static_for<NJ>([&](auto jj) {
                        ds_store16(dsp.tile + dsp.voff + (2 * (int)qq + (int)jj) * 512, dso_all[(int)qq][(int)jj]);
                    });
            });
    };

    i32x16 pBo[NJ], dsBo[NJ];
    int dsso[NJ];
    QTile pre_t;
    AFrag pre_a[gs(1)];
    // DSO: step s's dS tile (kv block kb, q block kb + s), this wave's 8 KB, this lane's 16 B of each 512-byte chunk
    const unsigned long long ds_bh = DSO ? (unsigned long long)(uintptr_t)DSB + bh * ds_tiles_per_bh(S) * DS_TILE : 0ull;
    // (the wave's 8 KB of the tile as a uniform global base + the lane's 32-bit offset: global_store saddr)
    auto ds_ptr = [&](int s) __attribute__((always_inline)) -> DsOut {
        if constexpr (!DSO) return DsOut{nullptr, 0u};
        const int qbk = DSO_PROBE == 1 ? kb : kb + s;
        return DsOut{(g_u8 *)(uintptr_t)(ds_bh + (unsigned long long)ds_tile_index(kb, qbk) * DS_TILE +
                                         (unsigned long long)(wv * 8192)),
                     16u * (unsigned)lane_now()};
    };
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
        step(s, ds_ptr(s), a, qs0 + s * QS_, pB, dsB, dss, pBo, dsBo, dsso, pre_t, pre_a, diag_c, prev_c);
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
        step(s, ds_ptr(s), a, qs0 + s * QS_, pB, dsB, dss, pBq, dsBq, dssq, pre_t, pre_a, diag_c, prev_c);
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
    // EPI_SKEW: region strides and bases (dK nope / full rows, dV, dK rope)
    constexpr int RRW = KB / 16, RK = RRW * XDN + 16, RV = RRW * XDV + 16, RP = RRW * 128 + 16;
    constexpr int S_DK = 0, S_DV = S_DK + 16 * RK, S_DP = S_DV + 16 * RV, S_END = S_DP + (SPLIT ? 16 * RP : 0);
    static_assert(!EPI_SKEW || S_END <= ARING, "skewed staging inside the q ring");
    {
        const int ln = lane_now();
        static_for<NJ>([&](auto jj) {
            constexpr int j = jj;
            const unsigned r = (unsigned)(KBW * wv + 16 * j + (ln & 15)), hh = (unsigned)(ln >> 4);
#if EPI_SKEW
            // row r: region r % 16 = lr, its row r / 16
            const unsigned rg = (unsigned)(ln & 15), rm = r >> 4;
            const unsigned dkr = lds0 + S_DK + rg * RK + rm * XDN + 16 * hh, dvr = lds0 + S_DV + rg * RV + rm * XDV + 16 * hh;
            const unsigned dpr = lds0 + S_DP + rg * RP + rm * 128 + 16 * hh;
#else
            const unsigned dkr = lds0 + O_SDK + r * XDN + 16 * hh, dvr = lds0 + O_SDV + r * XDV + 16 * hh;
            const unsigned dpr = lds0 + O_SDP + r * 128 + 16 * hh;
#endif
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
#elif EPI_SKEW
    bar_signal();
    bar_wait();
    {
        // wave w: regions j = 4 w .. 4 w + 3 of each output: rows kv0 + j + 16 m (m < KB / 16; those past S dropped)
        const int rows = min(KB, S - kv0);
        static_for<4>([&](auto jj) {
            const int j = 4 * wv + (int)jj;
            const unsigned nr = rows > j ? (unsigned)min(RRW, (rows - j + 15) / 16) : 0u;
            const long long kr = (long long)(kv0 + j);
            const unsigned long long gk = (unsigned long long)(uintptr_t)(DK + (long long)b * dk_sb + kr * dk_ss +
                                                                          (long long)h * (SPLIT ? 256 : D_QK));
            tdm_store(tdm_g0(lds0 + S_DK + (unsigned)(j * RK), gk),
                      tdm_g1(tdm_w0(0, 0, 0), nr, XDN, RRW, 32ull * (unsigned long long)dk_ss));
            const unsigned long long gv = (unsigned long long)(uintptr_t)(DV + (long long)b * dv_sb + kr * dv_ss +
                                                                          (long long)h * (SPLIT ? 256 : D_V));
            tdm_store(tdm_g0(lds0 + S_DV + (unsigned)(j * RV), gv),
                      tdm_g1(tdm_w0(0, 0, 0), nr, XDV, RRW, 32ull * (unsigned long long)dv_ss));
            if constexpr (SPLIT) {
                const unsigned long long gp = (unsigned long long)(uintptr_t)(DKP + (long long)b * dkp_sb +
                                                                              kr * dkp_ss + (long long)h * 64);
                tdm_store(tdm_g0(lds0 + S_DP + (unsigned)(j * RP), gp),
                          tdm_g1(tdm_w0(0, 0, 0), nr, 128, RRW, 32ull * (unsigned long long)dkp_ss));
            }
        });
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
                     dk_ss, dv_sb, dv_ss, nullptr, 0, 0, nullptr, smem);
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
                    dk_ss, dv_sb, dv_ss, DKP, dkp_sb, dkp_ss, nullptr, smem);
}

// rung b7: k_dkdv_mx / k_dkdv_mxr that also store dS (bf16) to DSB for k_dqg_mx / k_dqg_mxr
extern "C" __global__ __attribute__((amdgpu_flat_work_group_size(128, 128), amdgpu_waves_per_eu(1, 1)))
void k_dkdv_mxd(const unsigned char *__restrict__ Q8, const unsigned char *__restrict__ QS,
                const unsigned char *__restrict__ K8, const unsigned char *__restrict__ KS,
                const unsigned char *__restrict__ V8, const unsigned char *__restrict__ VSR,
                const unsigned char *__restrict__ DO8, const unsigned char *__restrict__ DOS,
                const unsigned char *__restrict__ DO8T, const unsigned char *__restrict__ DOTS,
                const unsigned char *__restrict__ Q8T, const unsigned char *__restrict__ QTS,
                const float *__restrict__ LSE2, const float *__restrict__ DELTA, bf16 *__restrict__ DK,
                bf16 *__restrict__ DV, float scale, int S, int H, int B, long long dk_sb, long long dk_ss,
                long long dv_sb, long long dv_ss, unsigned char *__restrict__ DSB) {
    __shared__ __attribute__((aligned(1024))) unsigned char smem[LDS_B];
    dkdv_body<false, true>(Q8, QS, K8, KS, V8, VSR, DO8, DOS, DO8T, DOTS, Q8T, QTS, LSE2, DELTA, DK, DV, scale, S, H, B,
                           dk_sb, dk_ss, dv_sb, dv_ss, nullptr, 0, 0, DSB, smem);
}
extern "C" __global__ __attribute__((amdgpu_flat_work_group_size(128, 128), amdgpu_waves_per_eu(1, 1)))
void k_dkdv_mxrd(const unsigned char *__restrict__ Q8, const unsigned char *__restrict__ QS,
                 const unsigned char *__restrict__ K8, const unsigned char *__restrict__ KS,
                 const unsigned char *__restrict__ V8, const unsigned char *__restrict__ VSR,
                 const unsigned char *__restrict__ DO8, const unsigned char *__restrict__ DOS,
                 const unsigned char *__restrict__ DO8T, const unsigned char *__restrict__ DOTS,
                 const unsigned char *__restrict__ Q8T, const unsigned char *__restrict__ QTS,
                 const float *__restrict__ LSE2, const float *__restrict__ DELTA, bf16 *__restrict__ DK,
                 bf16 *__restrict__ DV, float scale, int S, int H, int B, long long dk_sb, long long dk_ss,
                 long long dv_sb, long long dv_ss, bf16 *__restrict__ DKP, long long dkp_sb, long long dkp_ss,
                 unsigned char *__restrict__ DSB) {
    __shared__ __attribute__((aligned(1024))) unsigned char smem[LDS_B];
    dkdv_body<true, true>(Q8, QS, K8, KS, V8, VSR, DO8, DOS, DO8T, DOTS, Q8T, QTS, LSE2, DELTA, DK, DV, scale, S, H, B,
                          dk_sb, dk_ss, dv_sb, dv_ss, DKP, dkp_sb, dkp_ss, DSB, smem);
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
// k_dq_mxr (the fused-RoPE path): the same with dQ's rope columns through the inverse RoPE in the epilogue (the block's
// cos / sin rows TDM-loaded into the idle stage of tile n during tile n - 2): replaces k_dq_mx + k_rope_bwd_q, bit-exact.
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

// the kernel body; ROPE: dQ's rope columns (d 128..191, the gradient of the roped q values) through the inverse
// rotation (k_rope_bwd_q's math on the bf16-rounded values: bit-exact with that pass) with the COS / SIN rows [S, 64]
template <bool ROPE>
__device__ __forceinline__ void dq_body(
    const unsigned char *__restrict__ Q8, const unsigned char *__restrict__ QS, const unsigned char *__restrict__ K8,
    const unsigned char *__restrict__ KS, const unsigned char *__restrict__ V8, const unsigned char *__restrict__ VSR,
    const unsigned char *__restrict__ DO8, const unsigned char *__restrict__ DOS, const unsigned char *__restrict__ K8T,
    const unsigned char *__restrict__ KTS, const float *__restrict__ LSE2, const float *__restrict__ DELTA,
    bf16 *__restrict__ DQ, float scale, int S, int H, int B, long long dq_sb, long long dq_ss,
    const bf16 *__restrict__ COS, const bf16 *__restrict__ SIN, unsigned char *smem) {
    lock_simd();
    if constexpr (PF_CODE & 1) prefetch_code<10>();  // 40 KB: k_dq_mx 43.9 KB, k_dq_mxr 47.8 KB of code
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
#elif DQ_ORDER >= 2
    // DQ_ORDER passes per XCD: every (b, h)'s longest part of q blocks first, then the next parts (the dispatch tail
    // made of short workgroups; each (b, h)'s K / V read in every pass)
    int qb, h, b;
    if ((B * H) % 8 == 0 && nqb % DQ_ORDER == 0) {
        const int L = (int)(blockIdx.x + nqb * (blockIdx.y + H * blockIdx.z)), k = L >> 3, xc = L & 7;
        const int nbx = B * H / 8, part = nqb / DQ_ORDER, pass = k / (nbx * part), k1 = k % (nbx * part);
        const int bhx = xc + 8 * (k1 / part), sl = k1 % part;
        qb = nqb - 1 - (pass * part + sl);
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
#if DQ_ORDER < 2
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
    // ROPE: the block's cos / sin rows (256 x 128 B each, rows padded to 144 B) into the stage of tile n (idle from tile
    // n - 2 on), with tile n's ops (issued in tile n - 2): waves 0 / 1 in place of their first (NULL) op; the dQ staging
    // then in the other two stages (128 rows each)
    constexpr unsigned RS_B = BMQ * X128;  // 36864: cos (sin at + RS_B)
    static_assert(2 * RS_B <= Q_STG && 128 * 2 * D_QK <= Q_STG, "cos / sin in one stage, half the dQ staging in one");
    [[maybe_unused]] auto issue_rope_tile = [&](int t) {  // t = n
        const unsigned st = lds0 + (unsigned)((t % Q_NSTG) * Q_STG);
        const bf16 *rs = wv == 0 ? COS : SIN;
        const i32x4 g0 = wv < 2 ? tdm_g0(st + (unsigned)wv * RS_B, (unsigned long long)(uintptr_t)(rs + (long long)q0 * 64))
                                : tdm_g0(st + top0.lds, top0.base, 0);
        tdm_load(g0, wv < 2 ? tdm_g1(W0_128, BMQ, 128, BMQ, 128) : top0.g1);
        tdm_load(tdm_g0(st + top1.lds, top1.base, 0), top1.g1);
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
        if ((DQ_WAIT0 && !ROPE) || t > 0) {  // DQ_WAIT0: also at t = 0 (tile 0 landed already: one extra barrier; one-block loop)
            tensor_wait<TDM_PER_TILE>();  // tile t landed (t + 1 in flight)
            bar_signal();
            bar_wait();        // tile t visible; every wave is done with tile t - 1
        }
        if constexpr (ROPE && DC == 1)
            issue_rope_tile(t + 2);  // t + 2 = n: cos / sin
        else
            issue_tile(t + 2);  // into the stage of tile t - 1
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
    // epilogue: dQ rows as bf16 into LDS (the ring, idle now), one TDM store of 256 x 384 B by wave 0 (ROPE: the halves
    // of 128 rows in the stages of tiles n + 1, n + 2, stored by waves 0 / 1)
    tensor_wait<0>();
    bar_signal();
    bar_wait();
    constexpr int XDQ = 2 * D_QK;
    static_assert(BMQ * XDQ <= Q_OFF_Q, "staging inside the ring");
    {
        const int ln = lane_now();
        [[maybe_unused]] const unsigned csb = lds0 + (unsigned)((n % Q_NSTG) * Q_STG);
        [[maybe_unused]] const unsigned sth0 = lds0 + (unsigned)(((n + 1) % Q_NSTG) * Q_STG);
        [[maybe_unused]] const unsigned sth1 = lds0 + (unsigned)(((n + 2) % Q_NSTG) * Q_STG);
        // DQ_SKEW: region stride of the 16 row classes (q % 16): 16 rows (ROPE: 8 rows of a 128-row half) + 16 B
        constexpr unsigned RG = (ROPE ? 8 : 16) * XDQ + 16;
        static_assert(!DQ_SKEW || (ROPE ? 16 * RG <= Q_STG : 16 * RG <= Q_OFF_Q), "skewed staging");
        auto row_b = [&](int i) -> unsigned {
            if constexpr (DQ_SKEW) {
                // row q = 64 i + 16 wv + lr (ROPE: q' = 64 (i & 1) + 16 wv + lr of half i / 2): region lr, its row q / 16
                if constexpr (ROPE)
                    return (i < 2 ? sth0 : sth1) + (unsigned)((ln & 15) * RG + (4 * (i & 1) + wv) * XDQ + 16 * (ln >> 4));
                else
                    return lds0 + (unsigned)((ln & 15) * RG + (4 * i + wv) * XDQ + 16 * (ln >> 4));
            } else if constexpr (ROPE) {
                return (i < 2 ? sth0 : sth1) + (unsigned)((64 * (i & 1) + 16 * wv + (ln & 15)) * XDQ + 16 * (ln >> 4));
            } else {
                return lds0 + (unsigned)((64 * i + 16 * wv + (ln & 15)) * XDQ + 16 * (ln >> 4));
            }
        };
        // ROPE: the cos / sin values of the lane's rows (position q0 + 64 i + 16 wv + lr, values 8 lh + 16 j, j < 4)
        [[maybe_unused]] u32x4 rcs[ROPE ? QTW : 1][4], rsn[ROPE ? QTW : 1][4];
        if constexpr (ROPE) {
            static_for<QTW>([&](auto ii) {
                const unsigned rb = csb + (unsigned)((64 * (int)ii + 16 * wv + (ln & 15)) * X128 + 16 * (ln >> 4));
                static_for<4>([&](auto j) {
                    rcs[(int)ii][(int)j] = *(__attribute__((address_space(3))) u32x4 *)(uintptr_t)(rb + 32 * (int)j);
                    rsn[(int)ii][(int)j] = *(__attribute__((address_space(3))) u32x4 *)(uintptr_t)(rb + RS_B + 32 * (int)j);
                });
            });
        }
        static_for<QTW>([&](auto ii) {
            constexpr int i = ii;
            const unsigned rowb = row_b(i);
            static_for<ROPE ? 8 : 12>([&](auto dd) {
                constexpr int dt = dd;
                bf16x8 w;
                static_for<8>([&](auto rr) { w[(int)rr] = (bf16)(dq[dt][i][(int)rr] * scale); });
                *(__attribute__((address_space(3))) u32x4 *)(uintptr_t)(rowb + 32 * dt) = __builtin_bit_cast(u32x4, w);
            });
        });
        // ROPE (after every row's nope columns): d 128 + 8 m + r (m = 2 p + lh): gl = d16 tile 8 + p, gr = tile 10 + p
        // (the value 32 further); the rotated pair (o0, o1) to the row's bytes 256 + 32 m (+ 16), as k_rope_bwd_q: bit-
        // exact with k_dq_mx + k_rope_bwd_q (the rotation of bf16 values is exact per op: bf16 x bf16 products). The
        // values rotated are bf16 of the f32 products as the unfused path stores them: the products pass an empty asm
        // (else LLVM folds fpext(fptrunc(x * scale)) into v_fma_mixlo_bf16: one rounding of the exact product, which
        // moves 2.5e-6 of the inputs by an ulp, and the rotated values by more where the two terms cancel)
        if constexpr (ROPE) {
            static_for<QTW>([&](auto ii) {
                constexpr int i = ii;
                const unsigned rowb = row_b(i);
                f32x8 pr[4];
                static_for<4>([&](auto k) {
                    static_for<8>([&](auto rr) { pr[(int)k][(int)rr] = dq[8 + (int)k][i][(int)rr] * scale; });
                });
                asm("" : "+v"(pr[0]), "+v"(pr[1]), "+v"(pr[2]), "+v"(pr[3]));
                static_for<2>([&](auto pp) {
                    constexpr int p = pp;
                    f32x8 gl, gr;
                    static_for<8>([&](auto rr) {
                        gl[(int)rr] = (float)(bf16)pr[p][(int)rr];
                        gr[(int)rr] = (float)(bf16)pr[2 + p][(int)rr];
                    });
                    u32x4 o0, o1;
                    irope8(gl, gr, rcs[i][p], rsn[i][p], rcs[i][2 + p], rsn[i][2 + p], o0, o1);
                    const unsigned a = rowb + 256 + 64 * p + 16 * (ln >> 4);
                    *(__attribute__((address_space(3))) u32x4 *)(uintptr_t)a = o0;
                    *(__attribute__((address_space(3))) u32x4 *)(uintptr_t)(a + 16) = o1;
                });
            });
        }
        bar_signal();
        bar_wait();
        const unsigned long long g = (unsigned long long)(uintptr_t)(DQ + (long long)b * dq_sb + (long long)q0 * dq_ss +
                                                                     (long long)h * D_QK);
        if constexpr (DQ_SKEW) {
            // wave w: regions j = 4 w .. 4 w + 3 (of each half with ROPE): rows q0 (+ 128 half) + j + 16 m
            const unsigned long long rs = 2ull * (unsigned long long)dq_ss;  // one row (bytes)
            static_for<ROPE ? 2 : 1>([&](auto hh) {
                const unsigned base = ROPE ? ((int)hh == 0 ? sth0 : sth1) : lds0;
                static_for<4>([&](auto jj) {
                    const int j = 4 * wv + (int)jj;
                    tdm_store(tdm_g0(base + (unsigned)j * RG, g + ((unsigned long long)(128 * (int)hh) + (unsigned long long)j) * rs),
                              tdm_g1(tdm_w0(0, 0, 0), ROPE ? 8 : 16, XDQ, ROPE ? 8 : 16, 16ull * rs));
                });
            });
        } else if constexpr (ROPE) {
            if (wv < 2)
                tdm_store(tdm_g0(wv == 0 ? sth0 : sth1, g + (unsigned long long)wv * 128ull * 2ull * (unsigned long long)dq_ss),
                          tdm_g1(tdm_w0(0, 0, 0), 128, XDQ, 128, 2ull * (unsigned long long)dq_ss));
        } else if (wv == 0) {
            tdm_store(tdm_g0(lds0, g), tdm_g1(tdm_w0(0, 0, 0), BMQ, XDQ, BMQ, 2ull * (unsigned long long)dq_ss));
        }
    }
#else
    static_assert(!ROPE, "the fused rotation is in the staged epilogue");
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

extern "C" __global__ __attribute__((amdgpu_flat_work_group_size(128, 128), amdgpu_waves_per_eu(1, 1)))
void k_dq_mx(const unsigned char *__restrict__ Q8, const unsigned char *__restrict__ QS,
             const unsigned char *__restrict__ K8, const unsigned char *__restrict__ KS,
             const unsigned char *__restrict__ V8, const unsigned char *__restrict__ VSR,
             const unsigned char *__restrict__ DO8, const unsigned char *__restrict__ DOS,
             const unsigned char *__restrict__ K8T, const unsigned char *__restrict__ KTS,
             const float *__restrict__ LSE2, const float *__restrict__ DELTA, bf16 *__restrict__ DQ, float scale, int S,
             int H, int B, long long dq_sb, long long dq_ss) {
    __shared__ __attribute__((aligned(1024))) unsigned char smem[Q_LDS_B];
    dq_body<false>(Q8, QS, K8, KS, V8, VSR, DO8, DOS, K8T, KTS, LSE2, DELTA, DQ, scale, S, H, B, dq_sb, dq_ss, nullptr,
                   nullptr, smem);
}

// the fused-RoPE path's variant: dQ's rope columns as the gradient of q before RoPE (COS / SIN [S, 64] bf16)
extern "C" __global__ __attribute__((amdgpu_flat_work_group_size(128, 128), amdgpu_waves_per_eu(1, 1)))
void k_dq_mxr(const unsigned char *__restrict__ Q8, const unsigned char *__restrict__ QS,
              const unsigned char *__restrict__ K8, const unsigned char *__restrict__ KS,
              const unsigned char *__restrict__ V8, const unsigned char *__restrict__ VSR,
              const unsigned char *__restrict__ DO8, const unsigned char *__restrict__ DOS,
              const unsigned char *__restrict__ K8T, const unsigned char *__restrict__ KTS,
              const float *__restrict__ LSE2, const float *__restrict__ DELTA, bf16 *__restrict__ DQ, float scale,
              int S, int H, int B, long long dq_sb, long long dq_ss, const bf16 *__restrict__ COS,
              const bf16 *__restrict__ SIN) {
    __shared__ __attribute__((aligned(1024))) unsigned char smem[Q_LDS_B];
    dq_body<true>(Q8, QS, K8, KS, V8, VSR, DO8, DOS, K8T, KTS, LSE2, DELTA, DQ, scale, S, H, B, dq_sb, dq_ss, COS, SIN,
                  smem);
}

// =====================================================================================================================
// k_dqg_mx (rung b7): dQ = scale dS K from the dS that k_dkdv_mxd / k_dkdv_mxrd materialize (bf16 of the f32 P (dP -
// delta), ds_tile_index layout): a GEMM with no S / P / dP recompute (k_dq_mx: 96 of its 144 WMMAs per tile and every
// exp were the recompute). grid (S / 256, H, B) and block order as k_dq_mx; wave w owns q16 tiles i at block rows
// 64 i + 16 w. Per 128-kv tile t, a 3-stage ring of [dS tile (t, 2 qb), dS tile (t, 2 qb + 1), K8T 192 x 144, KTS] (one
// TDM op per wave). The dQ B operand of q tile i (dS^T: lane = q column, bytes 8 J + r = kv 16 J + 8 h + r) comes from
// transposed LDS loads (ds_load_tr16_b128: a group of 8 lanes reads 8 rows of 16 B, lane c gets column c), MX-quantized
// along kv per block m = kv16 tiles 2 m, 2 m + 1 with the peer lane as k_dq_mx (from the bf16 values), then dQ^T += K^T
// dS^T (12 d16 tiles, A = K8T rows). The last kv tile has no dS for q block 2 qb (its q tiles i < 2 are skipped).
// k_dqg_mxr: k_dq_mxr's fused inverse-RoPE epilogue.
// =====================================================================================================================
#ifndef DQG_V
// k_dqg tile body: 0 every q tile's B operand first, then 12 dQ groups (K8T fragments two ahead); 1 the 12 K8T fragments
// resident, the q tiles pipelined (tile i + 2's transposed loads and tile i + 1's quantization next to tile i's WMMAs)
#define DQG_V 1
#endif
#ifndef DQG_PROBE
#define DQG_PROBE 0  // timing probes only (wrong dQ): 1 no K8T fragment loads (one fragment reused), 2 no dS quantization
#endif
#ifndef DQG_TR
#define DQG_TR 0  // the transposed-load lane map (0: lane 16 h + 8 qh + c reads chunk row 16 qh + 8 h + c), see tr_off
#endif
namespace {
constexpr int G_DS1 = DS_TILE;                // the dS tile of q block 2 qb + 1 (2 qb at 0)
constexpr int G_KT = 2 * DS_TILE;             // 65536: K8T 192 x 144
constexpr int G_KTS = G_KT + D_QK * X128;     // 93184: KTS 192 x 4
constexpr int G_STG = G_KTS + D_QK * 4;       // 93952
constexpr int G_NSTG = 3;
constexpr int G_LDS_B = G_NSTG * G_STG;       // 281856
static_assert(G_LDS_B <= 327680, "LDS");
static_assert(G_STG % 128 == 0, "TDM alignment");
typedef short s16x8 __attribute__((ext_vector_type(8)));
typedef __attribute__((address_space(3))) s16x8 lds_s16x8;
// this lane's byte offset in a 512-byte chunk for the transposed load: lane l = 16 h + 8 qh + c reads chunk row
// L = 16 qh + 8 h + c (the 16 bytes of kv column 8 h + c, q rows 8 qh .. 8 qh + 7) and receives column c of its group's
// 8 rows = dS[q16 = 8 qh + c][kv16 = 8 h + r], r = 0..7: the B operand's bytes 8 J .. 8 J + 7 of lane (q16, h)
MX_INLINE unsigned tr_off(int l) {
    if constexpr (DQG_TR == 0) return (unsigned)(256 * ((l >> 3) & 1) + 128 * (l >> 4) + 16 * (l & 7));
    else return (unsigned)(16 * l);
}
MX_INLINE u32x4 lds_tr16(uint32_t base, int off) {
    return __builtin_bit_cast(u32x4, __builtin_amdgcn_ds_load_tr16_b128_v8i16((lds_s16x8 *)(uintptr_t)(base + off)));
}
}  // namespace

template <bool ROPE>
__device__ __forceinline__ void dqg_body(const unsigned char *__restrict__ DSB, const unsigned char *__restrict__ K8T,
                                         const unsigned char *__restrict__ KTS, bf16 *__restrict__ DQ, float scale,
                                         int S, int H, int B, long long dq_sb, long long dq_ss,
                                         const bf16 *__restrict__ COS, const bf16 *__restrict__ SIN,
                                         unsigned char *smem) {
    lock_simd();
    const uint32_t lds0 = lds_addr(smem);
    const int tid = (int)__builtin_amdgcn_workitem_id_x();
    const int wv = wave_id(tid);
    const int nqb = S / BMQ;
    const BlockOf blk = xcd_block(nqb, H, B);
#if DQ_ORDER >= 2
    int qb, h, b;
    if ((B * H) % 8 == 0 && nqb % DQ_ORDER == 0) {
        const int L = (int)(blockIdx.x + nqb * (blockIdx.y + H * blockIdx.z)), k = L >> 3, xc = L & 7;
        const int nbx = B * H / 8, part = nqb / DQ_ORDER, pass = k / (nbx * part), k1 = k % (nbx * part);
        const int bhx = xc + 8 * (k1 / part), sl = k1 % part;
        qb = nqb - 1 - (pass * part + sl);
        h = bhx % H;
        b = bhx / H;
    } else {
        qb = nqb - 1 - blk.slot;
        h = blk.h;
        b = blk.b;
    }
#else
    const int qb = nqb - 1 - blk.slot, h = blk.h, b = blk.b;
#endif
    const int q0 = qb * BMQ;
    const int n = (q0 + BMQ) / NBK;  // kv tiles 0 .. n - 1; tile n - 1 holds q block 2 qb + 1 only
    const int qa = 2 * qb;           // q blocks of 128: qa, qa + 1
    const unsigned long long bh = (unsigned long long)b * H + h;
    const unsigned long long dsbh = (unsigned long long)(uintptr_t)DSB + bh * ds_tiles_per_bh(S) * DS_TILE;

    // the tile's TDM op of this wave: 0 dS (t, qa) (t <= qa), 1 dS (t, qa + 1), 2 K8T rows, 3 KTS
    TdmOp op;
    {
        const unsigned long long bS128 = bh * (unsigned long long)(S / 128);
        if (wv == 0)
            op = {0u, dsbh + (unsigned long long)ds_tile_index(0, qa) * DS_TILE, (unsigned long long)DS_TILE,
                  tdm_g1(W0_S, 256, 128, 256, 128), 1};
        else if (wv == 1)
            op = {(unsigned)G_DS1, dsbh + (unsigned long long)ds_tile_index(0, qa + 1) * DS_TILE,
                  (unsigned long long)DS_TILE, tdm_g1(W0_S, 256, 128, 256, 128), 1};
        else if (wv == 2)
            op = {(unsigned)G_KT, (unsigned long long)(uintptr_t)K8T + bh * D_QK * (unsigned long long)S,
                  (unsigned long long)NBK, tdm_g1(W0_128, D_QK, 128, D_QK, (unsigned long long)S), 1};
        else
            op = {(unsigned)G_KTS, (unsigned long long)(uintptr_t)KTS + bS128 * D_QK * 4, (unsigned long long)D_QK * 4,
                  tdm_g1(W0_S, D_QK, 4, D_QK, 4), 1};
    }
    auto issue_tile = [&](int t) __attribute__((always_inline)) {
        const bool real = t < n && (wv != 0 || t <= qa);
        const unsigned long long tt = (unsigned long long)(t < n ? t : n - 1);
        const unsigned st = lds0 + (unsigned)((t % G_NSTG) * G_STG);
        tdm_load(tdm_g0(st + op.lds, op.base + tt * op.mult, real ? 1 : 0), op.g1);
    };
    // ROPE: the block's cos / sin rows (256 x 128 B, rows padded to 144 B) into the stage of tile n (idle from tile n - 2
    // on), issued in tile n - 2 by waves 0 / 1 in place of their (NULL) op of tile n
    constexpr unsigned RS_B = BMQ * X128;
    static_assert(2 * RS_B <= G_STG, "cos / sin in one stage");
    [[maybe_unused]] auto issue_rope_tile = [&](int t) __attribute__((always_inline)) {  // t = n
        const unsigned st = lds0 + (unsigned)((t % G_NSTG) * G_STG);
        const bf16 *rs = wv == 0 ? COS : SIN;
        const i32x4 g0 = wv < 2 ? tdm_g0(st + (unsigned)wv * RS_B, (unsigned long long)(uintptr_t)(rs + (long long)q0 * 64))
                                : tdm_g0(st + op.lds, op.base, 0);
        tdm_load(g0, wv < 2 ? tdm_g1(W0_128, BMQ, 128, BMQ, 128) : op.g1);
    };
    issue_tile(0);
    issue_tile(1);
    tensor_wait<1>();  // tile 0 (tile 1 may stay in flight)
    bar_signal();
    bar_wait();

    f32x8 dq[12][QTW];
    static_for<12>([&](auto d) { static_for<QTW>([&](auto i) { dq[(int)d][(int)i] = f32x8{0, 0, 0, 0, 0, 0, 0, 0}; }); });

    struct AF {
        i32x16 x;
        int s;
    };
    // CL 0: interior tile; 1: tile n - 2 (ROPE: issues the cos / sin rows); 2: tile n - 1 (no dS of q block qa)
    auto tile = [&](int t, auto cl_c) __attribute__((always_inline)) {
        constexpr int CL = decltype(cl_c)::value;
        if (t > 0) {
            tensor_wait<1>();  // tile t landed (t + 1 in flight)
            bar_signal();
            bar_wait();  // tile t visible; every wave is done with tile t - 1
        }
        if constexpr (ROPE && CL == 1)
            issue_rope_tile(t + 2);
        else
            issue_tile(t + 2);  // into the stage of tile t - 1
        const unsigned st = lds0 + (unsigned)((t % G_NSTG) * G_STG);
        const int ln = lane_now(), r16 = ln & 15, hh = ln >> 4;
        // per-lane bases (pinned): every load below is base + immediate offset
        const uint32_t trb = pin(st + tr_off(ln) + (unsigned)(wv * 1024)),
                       kt_t = pin(st + G_KT + (unsigned)(r16 * X128 + 64 * hh)),
                       kt_s = pin(st + G_KTS + (unsigned)(ln * 4));
        auto lda = [&](auto dc) __attribute__((always_inline)) {
            constexpr int dt = decltype(dc)::value;
            const lds_u8 *tp = lds_ptr(kt_t);
            constexpr int o = 16 * dt * X128;
            return AF{cat4(lds_b128(tp, o), lds_b128(tp, o + 16), lds_b128(tp, o + 32), lds_b128(tp, o + 48)),
                      (int)lds_b32(lds_ptr(kt_s), 32 * (dt / 2) * 4)};
        };
#if DQG_V == 0
        AF ab[4];
        ab[0] = lda(std::integral_constant<int, 0>{});
        ab[1] = lda(std::integral_constant<int, 1>{});
        // the dQ B operands: q tile i (q block qa + i / 2, its q16 tile qt = 4 (i % 2) + wv), kv16 tile J = 2 m + u in
        // chunk 16 (J / 2) + 2 qt + J % 2
        i32x16 dsB[QTW];
        int dss[QTW] = {0, 0, 0, 0};
        static_for<QTW>([&](auto ii) {
            constexpr int i = ii;
            if constexpr (!(CL == 2 && i < 2)) {
                constexpr int oi = (i / 2) * DS_TILE + 4 * (i % 2) * 1024;
                static_for<4>([&](auto mm) {
                    constexpr int m = mm;
                    const u32x4 x0 = lds_tr16(trb, oi + (16 * m) * 512);
                    const u32x4 x1 = lds_tr16(trb, oi + (16 * m + 1) * 512);
                    const float am = amax16_bf16(x0, x1);
                    const int e = mx_exp(__builtin_fmaxf(am, peer(am)));
                    const float sc = exp2i(e);
                    const u32x2 c0 = __builtin_amdgcn_cvt_scalef32_pk8_fp8_bf16(__builtin_bit_cast(bf16x8, x0), sc);
                    const u32x2 c1 = __builtin_amdgcn_cvt_scalef32_pk8_fp8_bf16(__builtin_bit_cast(bf16x8, x1), sc);
                    dsB[i][4 * m] = (int)c0[0];
                    dsB[i][4 * m + 1] = (int)c0[1];
                    dsB[i][4 * m + 2] = (int)c1[0];
                    dsB[i][4 * m + 3] = (int)c1[1];
                    dss[i] |= (e + 127) << (8 * m);
                });
            }
        });
        fence();
        // dQ^T += K^T dS^T: A = K8T rows d (fragments two groups ahead, four buffers)
        static_for<12>([&](auto dd) {
            constexpr int dt = dd;
            if constexpr (dt + 2 < 12) ab[(dt + 2) % 4] = lda(std::integral_constant<int, (dt + 2 < 12 ? dt + 2 : 0)>{});
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
                if constexpr (CL == 2 && i < 2) {
                } else if constexpr (dt % 2 == 0)
                    dq[dt][i] = wmma_mx<0, 0>(af.x, dsB[i], dq[dt][i], af.s, dss[i]);
                else
                    dq[dt][i] = wmma_mx<1, 0>(af.x, dsB[i], dq[dt][i], af.s, dss[i]);
            });
            fence();
        });
#else
        // the B operand of q tile i (q block qa + i / 2, its q16 tile qt = 4 (i % 2) + wv): kv16 tile J = 2 m + u in chunk
        // 16 (J / 2) + 2 qt + J % 2, MX blocks m along kv (with the peer lane), the E8M0 rule on the bits (= mx_exp)
        AF kf[12];
        if constexpr (DQG_ORDER == 0) static_for<12>([&](auto dd) { kf[(int)dd] = lda(dd); });
        constexpr int I0 = CL == 2 ? 2 : 0;  // tile n - 1: no dS of q block qa
        u32x4 xr[2][8];
        i32x16 db[2];
        int dsc[2];
        auto ldx = [&](auto ic) __attribute__((always_inline)) {
            constexpr int i = ic;
            constexpr int oi = (i / 2) * DS_TILE + 4 * (i % 2) * 1024;
            static_for<8>([&](auto jj) {
                constexpr int J = jj;
                xr[i % 2][J] = lds_tr16(trb, oi + (16 * (J / 2) + J % 2) * 512);
            });
        };
        auto quant = [&](auto ic) __attribute__((always_inline)) {
            constexpr int i = ic;
            int sc_word = 0;
            static_for<4>([&](auto mm) {
                constexpr int m = mm;
                const u32x4 &x0 = xr[i % 2][2 * m], &x1 = xr[i % 2][2 * m + 1];
                const float am = amax16_bf16(x0, x1);
                const int byte = e8m0_byte(__builtin_fmaxf(am, peer(am)));
                const float sc = e8m0_scale(byte);
                const u32x2 c0 = __builtin_amdgcn_cvt_scalef32_pk8_fp8_bf16(__builtin_bit_cast(bf16x8, x0), sc);
                const u32x2 c1 = __builtin_amdgcn_cvt_scalef32_pk8_fp8_bf16(__builtin_bit_cast(bf16x8, x1), sc);
                db[i % 2][4 * m] = (int)c0[0];
                db[i % 2][4 * m + 1] = (int)c0[1];
                db[i % 2][4 * m + 2] = (int)c1[0];
                db[i % 2][4 * m + 3] = (int)c1[1];
                sc_word |= byte << (8 * m);
            });
            dsc[i % 2] = sc_word;
        };
        ldx(std::integral_constant<int, I0>{});
        ldx(std::integral_constant<int, I0 + 1>{});
        fence();
        if constexpr (DQG_ORDER == 1) static_for<12>([&](auto dd) { kf[(int)dd] = lda(dd); });
        quant(std::integral_constant<int, I0>{});
        fence();
        static_for<QTW>([&](auto ii) {
            constexpr int i = ii;
            if constexpr (i >= I0) {
                if constexpr (i + 2 < QTW) ldx(std::integral_constant<int, (i + 2 < QTW ? i + 2 : 0)>{});
                if constexpr (i + 1 < QTW) quant(std::integral_constant<int, (i + 1 < QTW ? i + 1 : 0)>{});
                static_for<12>([&](auto dd) {
                    constexpr int dt = dd;
                    if constexpr (dt % 2 == 0)
                        dq[dt][i] = wmma_mx<0, 0>(kf[dt].x, db[i % 2], dq[dt][i], kf[dt].s, dsc[i % 2]);
                    else
                        dq[dt][i] = wmma_mx<1, 0>(kf[dt].x, db[i % 2], dq[dt][i], kf[dt].s, dsc[i % 2]);
                });
                fence();
            }
        });
#endif
    };
    for (int t = 0; t < n - 2; ++t) tile(t, std::integral_constant<int, 0>{});
    tile(n - 2, std::integral_constant<int, 1>{});
    tile(n - 1, std::integral_constant<int, 2>{});

    // epilogue (k_dq_mx's): dQ rows as bf16 into LDS (the ring, idle now), TDM stores (skewed regions)
    tensor_wait<0>();
    bar_signal();
    bar_wait();
    constexpr int XDQ = 2 * D_QK;
    {
        const int ln = lane_now();
        [[maybe_unused]] const unsigned csb = lds0 + (unsigned)((n % G_NSTG) * G_STG);
        [[maybe_unused]] const unsigned sth0 = lds0 + (unsigned)(((n + 1) % G_NSTG) * G_STG);
        [[maybe_unused]] const unsigned sth1 = lds0 + (unsigned)(((n + 2) % G_NSTG) * G_STG);
        constexpr unsigned RG = (ROPE ? 8 : 16) * XDQ + 16;
        static_assert(ROPE ? 16 * RG <= G_STG : 16 * RG <= G_LDS_B, "skewed staging");
        auto row_b = [&](int i) -> unsigned {
            if constexpr (ROPE)
                return (i < 2 ? sth0 : sth1) + (unsigned)((ln & 15) * RG + (4 * (i & 1) + wv) * XDQ + 16 * (ln >> 4));
            else
                return lds0 + (unsigned)((ln & 15) * RG + (4 * i + wv) * XDQ + 16 * (ln >> 4));
        };
        [[maybe_unused]] u32x4 rcs[ROPE ? QTW : 1][4], rsn[ROPE ? QTW : 1][4];
        if constexpr (ROPE) {
            static_for<QTW>([&](auto ii) {
                const unsigned rb = csb + (unsigned)((64 * (int)ii + 16 * wv + (ln & 15)) * X128 + 16 * (ln >> 4));
                static_for<4>([&](auto j) {
                    rcs[(int)ii][(int)j] = *(__attribute__((address_space(3))) u32x4 *)(uintptr_t)(rb + 32 * (int)j);
                    rsn[(int)ii][(int)j] = *(__attribute__((address_space(3))) u32x4 *)(uintptr_t)(rb + RS_B + 32 * (int)j);
                });
            });
        }
        static_for<QTW>([&](auto ii) {
            constexpr int i = ii;
            const unsigned rowb = row_b(i);
            static_for<ROPE ? 8 : 12>([&](auto dd) {
                constexpr int dt = dd;
                bf16x8 w;
                static_for<8>([&](auto rr) { w[(int)rr] = (bf16)(dq[dt][i][(int)rr] * scale); });
                *(__attribute__((address_space(3))) u32x4 *)(uintptr_t)(rowb + 32 * dt) = __builtin_bit_cast(u32x4, w);
            });
        });
        if constexpr (ROPE) {
            static_for<QTW>([&](auto ii) {
                constexpr int i = ii;
                const unsigned rowb = row_b(i);
                f32x8 pr[4];
                static_for<4>([&](auto k) {
                    static_for<8>([&](auto rr) { pr[(int)k][(int)rr] = dq[8 + (int)k][i][(int)rr] * scale; });
                });
                asm("" : "+v"(pr[0]), "+v"(pr[1]), "+v"(pr[2]), "+v"(pr[3]));
                static_for<2>([&](auto pp) {
                    constexpr int p = pp;
                    f32x8 gl, gr;
                    static_for<8>([&](auto rr) {
                        gl[(int)rr] = (float)(bf16)pr[p][(int)rr];
                        gr[(int)rr] = (float)(bf16)pr[2 + p][(int)rr];
                    });
                    u32x4 o0, o1;
                    irope8(gl, gr, rcs[i][p], rsn[i][p], rcs[i][2 + p], rsn[i][2 + p], o0, o1);
                    const unsigned a = rowb + 256 + 64 * p + 16 * (ln >> 4);
                    *(__attribute__((address_space(3))) u32x4 *)(uintptr_t)a = o0;
                    *(__attribute__((address_space(3))) u32x4 *)(uintptr_t)(a + 16) = o1;
                });
            });
        }
        bar_signal();
        bar_wait();
        const unsigned long long g = (unsigned long long)(uintptr_t)(DQ + (long long)b * dq_sb + (long long)q0 * dq_ss +
                                                                     (long long)h * D_QK);
        const unsigned long long rs = 2ull * (unsigned long long)dq_ss;  // one row (bytes)
        static_for<ROPE ? 2 : 1>([&](auto hc) {
            const unsigned base = ROPE ? ((int)hc == 0 ? sth0 : sth1) : lds0;
            static_for<4>([&](auto jj) {
                const int j = 4 * wv + (int)jj;
                tdm_store(tdm_g0(base + (unsigned)j * RG, g + ((unsigned long long)(128 * (int)hc) + (unsigned long long)j) * rs),
                          tdm_g1(tdm_w0(0, 0, 0), ROPE ? 8 : 16, XDQ, ROPE ? 8 : 16, 16ull * rs));
            });
        });
    }
    tensor_wait<0>();
}


#ifndef DQG_PAIR
#define DQG_PAIR 1  // k_dqg: q blocks nqb - 1 - p and p in one workgroup (every workgroup 2 nqb + 2 tiles; the first block's
                    // epilogue overlaps the second's first tiles); grid (nqb / 2, H, B) when nqb % 2 == 0 and B H % 8 == 0
#endif
namespace {
__host__ __device__ constexpr bool dqg_paired(int nqb, int B, int H) {
    return DQG_PAIR != 0 && nqb % 2 == 0 && (B * H) % 8 == 0;
}
// paired k_dqg: a stage also holds a block's whole dQ staging (16 regions of 16 rows x 384 B + 16 B: 98560 B; ROPE: two
// halves of 8 rows per region: 98816 B) so the first block's epilogue needs one stage only
constexpr int G2_STG = 98816;
constexpr int G2_LDS_B = G_NSTG * G2_STG;  // 296448
static_assert(G2_LDS_B <= 327680 && G2_STG % 128 == 0 && G_KTS + D_QK * 4 <= G2_STG, "LDS");
}  // namespace

template <bool ROPE>
__device__ __forceinline__ void dqg2_body(const unsigned char *__restrict__ DSB, const unsigned char *__restrict__ K8T,
                                          const unsigned char *__restrict__ KTS, bf16 *__restrict__ DQ, float scale,
                                          int S, int H, int B, long long dq_sb, long long dq_ss,
                                          const bf16 *__restrict__ COS, const bf16 *__restrict__ SIN,
                                          unsigned char *smem) {
    lock_simd();
    const uint32_t lds0 = lds_addr(smem);
    const int tid = (int)__builtin_amdgcn_workitem_id_x();
    const int wv = wave_id(tid);
    const int nqb = S / BMQ;
    int qbs[2], h, b, nblk;
    if (dqg_paired(nqb, B, H) && 2 * (int)gridDim.x == nqb) {  // the host launched the paired grid
        // XCD-local: the k-th workgroup of XCD x takes pair k % (nqb / 2) of (b, h) = x + 8 (k / (nqb / 2))
        const int np = nqb / 2;
        const int L = (int)(blockIdx.x + np * (blockIdx.y + H * blockIdx.z)), k = L >> 3, xc = L & 7;
        const int bhx = xc + 8 * (k / np), pp = k % np;
        qbs[0] = nqb - 1 - pp;
        qbs[1] = pp;
        h = bhx % H;
        b = bhx / H;
        nblk = 2;
    } else {
        const BlockOf blk = xcd_block(nqb, H, B);
        qbs[0] = qbs[1] = nqb - 1 - blk.slot;
        h = blk.h;
        b = blk.b;
        nblk = 1;
    }
    const int n0 = 2 * qbs[0] + 2, n1 = 2 * qbs[1] + 2;  // kv tiles of each block
    const int NT = nblk == 2 ? n0 + n1 : n0;
    const unsigned long long bh = (unsigned long long)b * H + h;
    const unsigned long long dsbh = (unsigned long long)(uintptr_t)DSB + bh * ds_tiles_per_bh(S) * DS_TILE;
    const unsigned long long kt_base = (unsigned long long)(uintptr_t)K8T + bh * D_QK * (unsigned long long)S;
    const unsigned long long kts_base = (unsigned long long)(uintptr_t)KTS + bh * (unsigned long long)(S / 128) * D_QK * 4;
    // this wave's op of a tile: 0 dS (t, 2 qb) (t <= 2 qb), 1 dS (t, 2 qb + 1), 2 K8T rows, 3 KTS
    const unsigned op_lds = wv == 0 ? 0u : wv == 1 ? (unsigned)G_DS1 : wv == 2 ? (unsigned)G_KT : (unsigned)G_KTS;
    const i32x8 op_g1 = wv <= 1 ? tdm_g1(W0_S, 256, 128, 256, 128)
                        : wv == 2 ? tdm_g1(W0_128, D_QK, 128, D_QK, (unsigned long long)S)
                                  : tdm_g1(W0_S, D_QK, 4, D_QK, 4);
    auto stage = [&](int T) __attribute__((always_inline)) { return lds0 + (unsigned)((T % G_NSTG) * G2_STG); };
    // per wave and block: tile 0's global address, the bytes per tile, the last real tile (dS of q block 2 qb: 2 qb)
    const unsigned long long op_mult = wv <= 1 ? (unsigned long long)DS_TILE : wv == 2 ? (unsigned long long)NBK
                                                                                      : (unsigned long long)(D_QK * 4);
    auto op_base = [&](int qb) __attribute__((always_inline)) {
        return wv <= 1 ? dsbh + (unsigned long long)ds_tile_index(0, 2 * qb + wv) * DS_TILE : wv == 2 ? kt_base : kts_base;
    };
    const unsigned long long base0 = op_base(qbs[0]), base1 = op_base(qbs[1]);
    const int lim0 = wv == 0 ? 2 * qbs[0] : 1 << 20, lim1 = wv == 0 ? 2 * qbs[1] : 1 << 20;
    auto issue = [&](int T) __attribute__((always_inline)) {  // selects only (the tile loop stays one basic block)
        const bool first = T < n0;
        const int t = first ? T : T - n0, n = first ? n0 : n1, lim = first ? lim0 : lim1;
        const bool real = T < NT && t <= lim;
        const unsigned long long tt = (unsigned long long)(t < n ? t : n - 1);
        const unsigned long long ga = (first ? base0 : base1) + tt * op_mult;
        tdm_load(tdm_g0(stage(T) + op_lds, ga, real ? 1 : 0), op_g1);
    };
    // ROPE: a block's cos / sin rows (256 x 128 B, rows padded to 144 B) into the stage of tile T (waves 0 / 1; 2 / 3 NULL)
    constexpr unsigned RS_B = BMQ * X128;
    static_assert(2 * RS_B <= G2_STG, "cos / sin in one stage");
    auto issue_extra = [&](int T, int q0) __attribute__((always_inline)) {
        if constexpr (ROPE) {
            const bf16 *rs = wv == 0 ? COS : SIN;
            const i32x4 g0 = wv < 2 ? tdm_g0(stage(T) + (unsigned)wv * RS_B,
                                             (unsigned long long)(uintptr_t)(rs + (long long)q0 * 64))
                                    : tdm_g0(stage(T) + op_lds, kt_base, 0);
            tdm_load(g0, wv < 2 ? tdm_g1(W0_128, BMQ, 128, BMQ, 128) : op_g1);
        } else {
            tdm_load(tdm_g0(stage(T) + op_lds, kt_base, 0), op_g1);  // NULL (keeps every wave's count in step)
        }
    };
    issue(0);
    issue(1);  // tile 0's wait + barrier: in tile(0) as in every tile (the tile loop stays one basic block)

    f32x8 dq[12][QTW];
    static_for<12>([&](auto d) { static_for<QTW>([&](auto i) { dq[(int)d][(int)i] = f32x8{0, 0, 0, 0, 0, 0, 0, 0}; }); });
    struct AF {
        i32x16 x;
        int s;
    };
    // tile T (of the block starting at q0): CL 0 interior, 1 the block's tile n - 2, 2 its tile n - 1 (no dS of q block
    // 2 qb; issues the block's cos / sin (ROPE) into tile T + 2's stage instead of the next block's tile 1)
    auto tile = [&](int T, int q0, auto cl_c) __attribute__((always_inline)) {
        constexpr int CL = decltype(cl_c)::value;
        tensor_wait<1>();  // tile T landed (T + 1 in flight)
        bar_signal();
        bar_wait();  // tile T visible; every wave is done with tile T - 1
        if constexpr (CL == 2)
            issue_extra(T + 2, q0);
        else
            issue(T + 2);  // into the stage of tile T - 1
        const unsigned st = stage(T);
        const int ln = lane_now(), r16 = ln & 15, hh = ln >> 4;
        const uint32_t trb = pin(st + tr_off(ln) + (unsigned)(wv * 1024)),
                       kt_t = pin(st + G_KT + (unsigned)(r16 * X128 + 64 * hh)),
                       kt_s = pin(st + G_KTS + (unsigned)(ln * 4));
        auto lda = [&](auto dc) __attribute__((always_inline)) {
            constexpr int dt = decltype(dc)::value;
            const lds_u8 *tp = lds_ptr(kt_t);
            constexpr int o = 16 * dt * X128;
            return AF{cat4(lds_b128(tp, o), lds_b128(tp, o + 16), lds_b128(tp, o + 32), lds_b128(tp, o + 48)),
                      (int)lds_b32(lds_ptr(kt_s), 32 * (dt / 2) * 4)};
        };
        // the B operand of q tile i (q block qa + i / 2, its q16 tile qt = 4 (i % 2) + wv): kv16 tile J = 2 m + u in chunk
        // 16 (J / 2) + 2 qt + J % 2, MX blocks m along kv (with the peer lane), the E8M0 rule on the bits (= mx_exp)
        AF kf[12];
        if constexpr (DQG_ORDER == 0) static_for<12>([&](auto dd) { kf[(int)dd] = lda(dd); });
        constexpr int I0 = CL == 2 ? 2 : 0;  // tile n - 1: no dS of q block qa
        u32x4 xr[2][8];
        i32x16 db[2];
        int dsc[2];
        auto ldx = [&](auto ic) __attribute__((always_inline)) {
            constexpr int i = ic;
            constexpr int oi = (i / 2) * DS_TILE + 4 * (i % 2) * 1024;
            static_for<8>([&](auto jj) {
                constexpr int J = jj;
                xr[i % 2][J] = lds_tr16(trb, oi + (16 * (J / 2) + J % 2) * 512);
            });
        };
        auto quant = [&](auto ic) __attribute__((always_inline)) {
            constexpr int i = ic;
            int sc_word = 0;
            if constexpr (DQG_PROBE == 2) {
                static_for<16>([&](auto e) { db[i % 2][(int)e] = (int)xr[i % 2][(int)e / 4][(int)e % 4]; });
                dsc[i % 2] = 0x7f7f7f7f;
                return;
            }
            static_for<4>([&](auto mm) {
                constexpr int m = mm;
                const u32x4 &x0 = xr[i % 2][2 * m], &x1 = xr[i % 2][2 * m + 1];
                const float am = amax16_bf16(x0, x1);
                const int byte = e8m0_byte(__builtin_fmaxf(am, peer(am)));
                const float sc = e8m0_scale(byte);
                const u32x2 c0 = __builtin_amdgcn_cvt_scalef32_pk8_fp8_bf16(__builtin_bit_cast(bf16x8, x0), sc);
                const u32x2 c1 = __builtin_amdgcn_cvt_scalef32_pk8_fp8_bf16(__builtin_bit_cast(bf16x8, x1), sc);
                db[i % 2][4 * m] = (int)c0[0];
                db[i % 2][4 * m + 1] = (int)c0[1];
                db[i % 2][4 * m + 2] = (int)c1[0];
                db[i % 2][4 * m + 3] = (int)c1[1];
                sc_word |= byte << (8 * m);
            });
            dsc[i % 2] = sc_word;
        };
        ldx(std::integral_constant<int, I0>{});
        ldx(std::integral_constant<int, I0 + 1>{});
        fence();
        if constexpr (DQG_PROBE == 1) {
            kf[0] = lda(std::integral_constant<int, 0>{});
            static_for<11>([&](auto dd) { kf[(int)dd + 1] = kf[0]; });
        } else if constexpr (DQG_ORDER == 1) static_for<12>([&](auto dd) { kf[(int)dd] = lda(dd); });
        quant(std::integral_constant<int, I0>{});
        fence();
        static_for<QTW>([&](auto ii) {
            constexpr int i = ii;
            if constexpr (i >= I0) {
                if constexpr (i + 2 < QTW) ldx(std::integral_constant<int, (i + 2 < QTW ? i + 2 : 0)>{});
                if constexpr (i + 1 < QTW) quant(std::integral_constant<int, (i + 1 < QTW ? i + 1 : 0)>{});
                static_for<12>([&](auto dd) {
                    constexpr int dt = dd;
                    if constexpr (dt % 2 == 0)
                        dq[dt][i] = wmma_mx<0, 0>(kf[dt].x, db[i % 2], dq[dt][i], kf[dt].s, dsc[i % 2]);
                    else
                        dq[dt][i] = wmma_mx<1, 0>(kf[dt].x, db[i % 2], dq[dt][i], kf[dt].s, dsc[i % 2]);
                });
                fence();
            }
        });
    };
    // a block's epilogue: dQ rows as bf16 into the stage of its last tile (ROPE: the cos / sin rows in tile T + 2's
    // stage), skewed regions, TDM stores; then (first of two blocks) the dQ accumulators cleared and the second block's
    // tile 1 issued into the cos / sin stage (read by every wave before the barrier)
    constexpr int XDQ = 2 * D_QK;
    constexpr unsigned RG = (ROPE ? 8 : 16) * XDQ + 16;
    static_assert((ROPE ? 32 : 16) * RG <= G2_STG, "a block's staging in one stage");
    auto epilogue = [&](int Tl, int q0) __attribute__((always_inline)) {
        tensor_wait<0>();
        bar_signal();
        bar_wait();
        const int ln = lane_now();
        const unsigned stg = stage(Tl);
        [[maybe_unused]] const unsigned csb = stage(Tl + 2);
        const unsigned sth0 = stg, sth1 = stg + 16 * RG;
        auto row_b = [&](int i) -> unsigned {
            if constexpr (ROPE)
                return (i < 2 ? sth0 : sth1) + (unsigned)((ln & 15) * RG + (4 * (i & 1) + wv) * XDQ + 16 * (ln >> 4));
            else
                return stg + (unsigned)((ln & 15) * RG + (4 * i + wv) * XDQ + 16 * (ln >> 4));
        };
        [[maybe_unused]] u32x4 rcs[ROPE ? QTW : 1][4], rsn[ROPE ? QTW : 1][4];
        if constexpr (ROPE) {
            static_for<QTW>([&](auto ii) {
                const unsigned rb = csb + (unsigned)((64 * (int)ii + 16 * wv + (ln & 15)) * X128 + 16 * (ln >> 4));
                static_for<4>([&](auto j) {
                    rcs[(int)ii][(int)j] = *(__attribute__((address_space(3))) u32x4 *)(uintptr_t)(rb + 32 * (int)j);
                    rsn[(int)ii][(int)j] = *(__attribute__((address_space(3))) u32x4 *)(uintptr_t)(rb + RS_B + 32 * (int)j);
                });
            });
        }
        static_for<QTW>([&](auto ii) {
            constexpr int i = ii;
            const unsigned rowb = row_b(i);
            static_for<ROPE ? 8 : 12>([&](auto dd) {
                constexpr int dt = dd;
                bf16x8 w;
                static_for<8>([&](auto rr) { w[(int)rr] = (bf16)(dq[dt][i][(int)rr] * scale); });
                *(__attribute__((address_space(3))) u32x4 *)(uintptr_t)(rowb + 32 * dt) = __builtin_bit_cast(u32x4, w);
            });
        });
        if constexpr (ROPE) {
            static_for<QTW>([&](auto ii) {
                constexpr int i = ii;
                const unsigned rowb = row_b(i);
                f32x8 pr[4];
                static_for<4>([&](auto k) {
                    static_for<8>([&](auto rr) { pr[(int)k][(int)rr] = dq[8 + (int)k][i][(int)rr] * scale; });
                });
                asm("" : "+v"(pr[0]), "+v"(pr[1]), "+v"(pr[2]), "+v"(pr[3]));
                static_for<2>([&](auto pc) {
                    constexpr int p = pc;
                    f32x8 gl, gr;
                    static_for<8>([&](auto rr) {
                        gl[(int)rr] = (float)(bf16)pr[p][(int)rr];
                        gr[(int)rr] = (float)(bf16)pr[2 + p][(int)rr];
                    });
                    u32x4 o0, o1;
                    irope8(gl, gr, rcs[i][p], rsn[i][p], rcs[i][2 + p], rsn[i][2 + p], o0, o1);
                    const unsigned a = rowb + 256 + 64 * p + 16 * (ln >> 4);
                    *(__attribute__((address_space(3))) u32x4 *)(uintptr_t)a = o0;
                    *(__attribute__((address_space(3))) u32x4 *)(uintptr_t)(a + 16) = o1;
                });
            });
        }
        bar_signal();
        bar_wait();
        const unsigned long long g = (unsigned long long)(uintptr_t)(DQ + (long long)b * dq_sb + (long long)q0 * dq_ss +
                                                                     (long long)h * D_QK);
        const unsigned long long rs = 2ull * (unsigned long long)dq_ss;  // one row (bytes)
        static_for<ROPE ? 2 : 1>([&](auto hc) {
            const unsigned base = (int)hc == 0 ? sth0 : sth1;
            static_for<4>([&](auto jj) {
                const int j = 4 * wv + (int)jj;
                tdm_store(tdm_g0(base + (unsigned)j * RG, g + ((unsigned long long)(128 * (int)hc) + (unsigned long long)j) * rs),
                          tdm_g1(tdm_w0(0, 0, 0), ROPE ? 8 : 16, XDQ, ROPE ? 8 : 16, 16ull * rs));
            });
        });
    };
    int T0 = 0;
    for (int k = 0; k < nblk; ++k) {
        const int n = k == 0 ? n0 : n1, q0 = (k == 0 ? qbs[0] : qbs[1]) * BMQ;
        for (int t = 0; t < n - 2; ++t) tile(T0 + t, q0, std::integral_constant<int, 0>{});
        tile(T0 + n - 2, q0, std::integral_constant<int, 1>{});
        tile(T0 + n - 1, q0, std::integral_constant<int, 2>{});
        epilogue(T0 + n - 1, q0);
        T0 += n;
        if (k + 1 < nblk) {
            static_for<12>([&](auto d) {
                static_for<QTW>([&](auto i) { dq[(int)d][(int)i] = f32x8{0, 0, 0, 0, 0, 0, 0, 0}; });
            });
            issue(T0 + 1);  // the next block's tile 1 (its tile 0 was issued in this block's tile n - 2)
        }
    }
    tensor_wait<0>();
}

extern "C" __global__ __attribute__((amdgpu_flat_work_group_size(128, 128), amdgpu_waves_per_eu(1, 1)))
void k_dqg_mx(const unsigned char *__restrict__ DSB, const unsigned char *__restrict__ K8T,
              const unsigned char *__restrict__ KTS, bf16 *__restrict__ DQ, float scale, int S, int H, int B,
              long long dq_sb, long long dq_ss) {
    __shared__ __attribute__((aligned(1024))) unsigned char smem[DQG_PAIR ? G2_LDS_B : G_LDS_B];
    if constexpr (DQG_PAIR)
        dqg2_body<false>(DSB, K8T, KTS, DQ, scale, S, H, B, dq_sb, dq_ss, nullptr, nullptr, smem);
    else
        dqg_body<false>(DSB, K8T, KTS, DQ, scale, S, H, B, dq_sb, dq_ss, nullptr, nullptr, smem);
}

// the fused-RoPE path's variant: dQ's rope columns as the gradient of q before RoPE (COS / SIN [S, 64] bf16)
extern "C" __global__ __attribute__((amdgpu_flat_work_group_size(128, 128), amdgpu_waves_per_eu(1, 1)))
void k_dqg_mxr(const unsigned char *__restrict__ DSB, const unsigned char *__restrict__ K8T,
               const unsigned char *__restrict__ KTS, bf16 *__restrict__ DQ, float scale, int S, int H, int B,
               long long dq_sb, long long dq_ss, const bf16 *__restrict__ COS, const bf16 *__restrict__ SIN) {
    __shared__ __attribute__((aligned(1024))) unsigned char smem[DQG_PAIR ? G2_LDS_B : G_LDS_B];
    if constexpr (DQG_PAIR)
        dqg2_body<true>(DSB, K8T, KTS, DQ, scale, S, H, B, dq_sb, dq_ss, COS, SIN, smem);
    else
        dqg_body<true>(DSB, K8T, KTS, DQ, scale, S, H, B, dq_sb, dq_ss, COS, SIN, smem);
}
