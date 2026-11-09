/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// Index formulas of the backward kernels (k_attn_bwd.hip): head dim 128, bf16, Hq = 4 Hkv,
// bottom-right causal (kv <= q + Skv - Sq), lse fp32 [B, Hq, Sq] natural log.
//
//   S = Q K^T (unscaled), P = exp2(S scale log2e - LSE log2e), dP = dO V^T, delta = rowsum(dO o O),
//   dS = P o (dP - delta), dV = P^T dO, dK = scale dS^T Q, dQ = scale dS K
//
// Main kernel geometry: a workgroup = 4 waves (one per SIMD) = 128 kv rows [kvw0, kvw0 + 128) of
// one (b, hkv); wave w owns kv rows kvw0 + 32 w + [0, 32) (two 16-row kv tiles kh). A step is 32
// queries (one q pair qt, two 16-row q tiles) of one q head gh of the GQA group; the steps run
// heads inner, q pairs descending from the last pair to the diagonal. Accumulators: S = Q K^T and
// dP = dO V^T (lane = kv, rows = q), dV^T += dO^T P and dK^T += Q^T dS (lane = kv, rows = d; P / dS
// packed from the S / dP accumulators), dQ = dS K (lane = d, rows = q; A = dS read back from the
// dS^T exchange tile). dQ of step s is computed in step s + 1 and added to an fp32 workspace
// (dQ_acc) with device-scope atomics.
//
// Every BSHD-family tensor (prep: dO, O; main: Q, K, V, dO, dK, dV; convert: dQ) is addressed with
// its own int64 (batch, seq) element strides: element (b, s, h, d) at b sb + s ss + h D + d. The
// host admits contiguous BSHD ((S H D, H D)) and a BSHD view of SBHD storage ((H D, B H D)); both
// cover exactly [0, B S H D), so whole-tensor resources keep their extent.
#pragma once

#define BWD_HD __host__ __device__ __attribute__((always_inline)) constexpr

#include <stdint.h>

namespace bwd {

constexpr int D     = 128; // q / k / v head dim
constexpr int WAVE  = 32;
constexpr int NWAVE = 4; // one wave per SIMD, one workgroup per CU
constexpr int BLOCK = NWAVE * WAVE;
constexpr int KVW   = 32;          // kv rows per wave (2 kv tiles kh)
constexpr int BKV   = NWAVE * KVW; // 128 kv rows per workgroup
constexpr int QS    = 32;          // queries per step (one q pair = 2 q tiles)
constexpr int NDO   = D / 16;      // 8: 16-wide d tiles of dV^T / dK^T
constexpr int ROW_B = D * 2;       // 256: one bf16 head row
constexpr int G4    = 4;           // the GQA ratio Hq / Hkv the main kernel is compiled for

// LDS pitches (bytes): XP for the Q / dO ring tiles and the K / V images (read by the rows and the
// tr loader), SP for the dS^T exchange tile [128 kv][32 q] (tr16 reads conflict-free, 2-way
// stores).
constexpr int XP = 272;
constexpr int SP = 96;

// Ring / exchange depths the LDS map reserves (RING / NDS below are the ones used).
constexpr int RING_MAX = 5, NDS_MAX = 3;

constexpr int QT_B = QS * XP;  // 8704: one Q or dO tile [32 q][XP]
constexpr int LD_B = QS * 8;   // 256: the {NL, delta} records of one step (32 q x 2 fp32)
constexpr int DS_B = BKV * SP; // 12288: one dS^T exchange tile [128 kv][SP]
constexpr int KV_B = BKV * XP; // 34816: one K or V image [128 kv][XP]

// LDS map (byte offsets): dO ring + LD ring in segment 0, Q ring in segment 1, dS^T tiles in
// segment 2, K / V images in segments 3-4.
constexpr int OFF_DO    = 0;
constexpr int OFF_LD    = OFF_DO + RING_MAX * QT_B; // 43520
constexpr int OFF_Q     = 65536;
constexpr int OFF_DS    = 131072;
constexpr int OFF_K     = 196608;
constexpr int OFF_V     = OFF_K + KV_B; // 231424
constexpr int LDS_BYTES = OFF_V + KV_B; // 266240 (<= 327680)
static_assert(OFF_LD + RING_MAX * LD_B <= OFF_Q, "dO + LD rings fit segment 0");
static_assert(OFF_Q + RING_MAX * QT_B <= OFF_DS, "Q ring fits segment 1");
static_assert(OFF_DS + NDS_MAX * DS_B <= OFF_K, "dS^T tiles fit segment 2");
static_assert(LDS_BYTES <= 327680, "320 KB LDS");

// TDM roles (wave-uniform): wave 0 issues the dO tiles, wave 1 the Q tiles, wave 2 the LD records
// (one ring op per step each); wave 3 issues the K and V images in the prologue and then a 16-B
// dummy op per step (the same tensorcnt waits for every wave, no role branch).
constexpr int ISS_DO = 0, ISS_Q = 1, ISS_LD = 2, ISS_KV = 3;

// Wave 3's dummy op moves the first 16 B of K row kvw0 into its own LDS cell, which nothing reads
// (segment 0, after the reserved LD ring slots).
constexpr int DUMMY_OFF  = OFF_LD + RING_MAX * LD_B; // 44800
constexpr int DUMMY_COLS = 8;                        // bf16
constexpr int DUMMY_B    = 2 * DUMMY_COLS;           // 16
static_assert(DUMMY_OFF % 16 == 0 && DUMMY_OFF + DUMMY_B <= OFF_Q,
              "dummy cell: segment 0, below the Q ring");

constexpr float LOG2E = 1.4426950408889634f;

// ---- grid decode: grid (Hkv, Skv / 128, B), blockIdx = (hkv, kv block, b) -----------------------
// x fastest = the kv heads of one block (one per XCD), y ascending = longest first under the
// bottom-right mask.
struct Work {
    int b, hkv, blk;
};
BWD_HD Work wg_work(int bx, int by, int bz) {
    return Work{bz, bx, by};
}
BWD_HD int kvw0_of(int blk) {
    return blk * BKV;
}

// ---- causal split (workgroup-uniform) -----------------------------------------------------------
BWD_HD int nqp_of(int Sq) {
    return Sq / QS;
}
// first q pair with any visible kv of the block: 32 qt + 31 + cshift >= kvw0
BWD_HD int qp_start(int kvw0, int cshift, int causal) {
    const int c = kvw0 - cshift;
    return causal ? (c < 0 ? 0 : c) / QS : 0;
}
// first q pair that sees all 128 kv rows of the block: kvw0 + 127 <= 32 qt + cshift
BWD_HD int qp_full(int kvw0, int cshift, int causal, int nqp) {
    if (!causal)
        return 0;
    const int u = kvw0 + BKV - 1 - cshift;
    const int q = u <= 0 ? 0 : (u + QS - 1) / QS;
    return q < nqp ? q : nqp;
}
BWD_HD int n_pairs(int kvw0, int cshift, int causal, int nqp) {
    return nqp - qp_start(kvw0, cshift, causal);
}
BWD_HD int n_maskp(int kvw0, int cshift, int causal, int nqp) {
    int       m  = qp_full(kvw0, cshift, causal, nqp) - qp_start(kvw0, cshift, causal);
    const int np = n_pairs(kvw0, cshift, causal, nqp);
    m            = m < 0 ? 0 : m;
    return m < np ? m : np;
}
BWD_HD int n_all(int kvw0, int cshift, int causal, int nqp, int G) {
    return G * n_pairs(kvw0, cshift, causal, nqp);
}
BWD_HD int n_mask(int kvw0, int cshift, int causal, int nqp, int G) {
    return G * n_maskp(kvw0, cshift, causal, nqp);
}
BWD_HD int n_full(int kvw0, int cshift, int causal, int nqp, int G) {
    return n_all(kvw0, cshift, causal, nqp, G) - n_mask(kvw0, cshift, causal, nqp, G);
}

// ---- step -> tile -------------------------------------------------------------------------------
// Steps [0, n_full) are unmasked for the whole block, [n_full, n) are the diagonal pairs. s is
// clamped into [0, n - 1], so the prefetch of steps >= n reloads tile n - 1 and every issuer issues
// exactly one TDM op per step.
BWD_HD int step_clamp(int s, int n) {
    return s < 0 ? 0 : (s >= n ? n - 1 : s);
}
BWD_HD int tile_qt(int s, int n, int G, int nqp) {
    return nqp - 1 - step_clamp(s, n) / G;
}
// head rotation: kv block blk starts its head cycle at gh = blk % G, so the workgroups of one XCD
// add into different dQ_acc blocks at any time
BWD_HD int rot_of(int blk, int G) {
    return blk % G;
}
BWD_HD int tile_gh(int s, int n, int G, int rot) {
    return (step_clamp(s, n) % G + rot) % G;
}
BWD_HD int qh_of(int hkv, int G, int gh) {
    return hkv * G + gh;
}
// the dQ atomics issued in step s belong to tile s - 1 (s = n: the drain; s = 0 adds dQ(-1) = 0
// into tile 0)
BWD_HD int atom_qt(int s, int n, int G, int nqp) {
    return tile_qt(s - 1, n, G, nqp);
}
BWD_HD int atom_gh(int s, int n, int G, int rot) {
    return tile_gh(s - 1, n, G, rot);
}

// Wave kind of a step (wave-uniform): FULL (no mask), DIAG (masked), DEAD (every pair masked: no
// S / dP / softmax / dV / dK work, a zero dS^T, every barrier, atomic and dQ WMMA kept).
enum { K_FULL = 0, K_DIAG = 1, K_DEAD = 2 };
BWD_HD int wave_kind(int kvw0, int w, int qt, int cshift, int causal) {
    if (!causal)
        return K_FULL;
    const int kv_lo = kvw0 + KVW * w, kv_hi = kv_lo + KVW - 1, q_lo = QS * qt, q_hi = q_lo + QS - 1;
    if (kv_lo > q_hi + cshift)
        return K_DEAD;
    if (kv_hi <= q_lo + cshift)
        return K_FULL;
    return K_DIAG;
}
// mask offset of S tile (hh, kh) of wave w (frag.h mask_col_gt_row: masked iff n > m + off)
BWD_HD int mask_off(int qt, int hh, int kvw0, int w, int kh, int cshift) {
    return (QS * qt + 16 * hh) + cshift - (kvw0 + KVW * w + 16 * kh);
}

// ---- ring slots and exchange buffers ------------------------------------------------------------
BWD_HD int lds_do(int slot) {
    return OFF_DO + slot * QT_B;
}
BWD_HD int lds_q(int slot) {
    return OFF_Q + slot * QT_B;
}
BWD_HD int lds_ld(int slot) {
    return OFF_LD + slot * LD_B;
}
BWD_HD int lds_ds(int buf) {
    return OFF_DS + buf * DS_B;
}

// ---- global extents and offsets -----------------------------------------------------------------
BWD_HD int64_t qdo_bytes(int B, int Sq, int Hq) {
    return (int64_t) B * Sq * Hq * ROW_B;
}
// LD = the prep output [B, Hq, Sq / 4] of 32-B records {NL[4 q], delta[4 q]}: one step = 256
// contiguous bytes
BWD_HD int64_t ld_rec_byte(int b, int Hq, int qh, int Sq, int q) {
    return (((int64_t) b * Hq + qh) * Sq + q) * 8;
} // q % 4 == 0
BWD_HD int64_t ld_bytes(int B, int Hq, int Sq) {
    return (int64_t) B * Hq * Sq * 8;
}
// byte offset of row (b, s, h) (element d = 0) of a strided tensor
BWD_HD int64_t row_byte_s(int64_t sb, int64_t ss, int b, int s, int h) {
    return ((int64_t) b * sb + (int64_t) s * ss + (int64_t) h * D) * 2;
}
// dO / Q tile: 32 rows q0.. of head qh, row stride ss
BWD_HD int64_t qdo_tile_byte_s(int64_t sb, int64_t ss, int b, int qh, int q0) {
    return row_byte_s(sb, ss, b, q0, qh);
}
// K / V image (128 rows kvw0.. of kv head hkv) and the dK / dV slab base
BWD_HD int64_t kv_tile_byte_s(int64_t sb, int64_t ss, int b, int hkv, int kvw0) {
    return row_byte_s(sb, ss, b, kvw0, hkv);
}

// ---- LDS operand addressing (lane (r, h) = (l % 16, l / 16)) ------------------------------------
BWD_HD int ct_imm(int pitch, int n0, int m0) {
    return n0 * pitch + 2 * m0;
} // + rows_lane: row n0 + r, columns m0 + 8h
// LD read-back: lane (r, h) of q tile hh reads 4 b128 at lds_ld(slot) + ld_lane + ld_imm(hh, k):
// k 0 = NL[16 hh + 8h + 0..3], 1 = delta[0..3], 2 = NL[4..7], 3 = delta[4..7]
BWD_HD int ld_lane(int lane) {
    return (lane / 16) * 64;
}
BWD_HD int ld_imm(int hh, int k) {
    return hh * 128 + k * 16;
}

// ---- dQ_acc: fp32 in WMMA C-fragment order, one 16 KB block per (b, qh, q pair) -----------------
// Block (b Hq + qh) nqp + qt; inside it slot (2 qh16 + j) 8 + i of 512 B, + 128 B per wave, + 4 B
// per lane, so one atomic instruction covers one aligned 128-B line. Each workgroup's resource
// spans its GQA group; the per-step part lives in voffset.
BWD_HD int64_t dqa_grp_byte(int b, int Hq, int hkv, int G, int Sq) {
    return (((int64_t) b * Hq + (int64_t) hkv * G) * nqp_of(Sq)) * 16384;
}
BWD_HD int64_t dqa_grp_bytes(int G, int Sq) {
    return (int64_t) G * nqp_of(Sq) * 16384;
}
BWD_HD int64_t dqa_bytes(int B, int Hq, int Sq) {
    return (int64_t) B * Hq * nqp_of(Sq) * 16384;
}
BWD_HD int dqa_voff(int lane, int w, int gh, int qt, int nqp) {
    return ((gh * nqp + qt) << 14) | (w << 7) | (lane << 2);
}
// dQ split over the 4 waves by (16-query half, 64-column half): wave w computes q rows 16 (w >> 1)
// and d columns 64 (w & 1) + 16 jj + l % 16 (jj = 0..3) of the step; the dQ_acc layout is unchanged
constexpr int DQ_NQH = 1, DQ_NJ = 4;
BWD_HD int    dqs_voff(int lane, int w) {
    return ((w >> 1) * 2 * 8 * 512) | ((w & 1) * 2 * 128) | (lane << 2);
}
BWD_HD int dqs_imm(int jj, int i) {
    return ((jj % 2) * 8 + i) * 512 + (jj / 2) * 128;
}

// ---- dQ convert (grid (Sq / 32, Hq, B), 256 threads): fp32 block -> scale -> bf16 dQ rows -------
constexpr int  CVT_PITCH = 272; // LDS [32 q][128 d] bf16 image
BWD_HD int64_t cvt_src_byte(int b, int Hq, int qh, int nqp, int qt) {
    return (((int64_t) b * Hq + qh) * nqp + qt) * 16384;
}
// float p in [0, 4096) of a block -> (q, d) inside the step (the inverse of the dQ_acc layout)
BWD_HD int cvt_q(int p) {
    const int k = p / 128, ln = p % 32;
    return 16 * (k / 16) + 8 * (ln / 16) + k % 8;
}
BWD_HD int cvt_d(int p) {
    const int k = p / 128, e = p % 128, ln = e % 32;
    return 32 * (e / 32) + 16 * ((k / 8) % 2) + ln % 16;
}
BWD_HD int cvt_lds(int q, int d) {
    return q * CVT_PITCH + 2 * d;
}
BWD_HD int64_t cvt_dst_byte_s(int64_t sb, int64_t ss, int b, int qh, int qt, int row, int chunk) {
    return row_byte_s(sb, ss, b, QS * qt + row, qh) + 16 * chunk;
}

// ---- prep (grid (Sq / 64, Hq, B), 256 threads): 64 rows (b, q0 + [0, 64), h) --------------------
BWD_HD int64_t lse_byte(int b, int Hq, int h, int Sq, int q) {
    return (((int64_t) b * Hq + h) * Sq + q) * 4;
}
// thread t, pass p (0..3): row q0 + 16 p + t / 16, bytes 16 (t % 16) of it; thread t < 16 writes
// the record of rows q0 + 4 t .. + 3
BWD_HD int prep_row(int t, int p) {
    return 16 * p + t / 16;
}
BWD_HD int64_t prep_row_byte_s(int64_t sb, int64_t ss, int b, int h, int q) {
    return row_byte_s(sb, ss, b, q, h);
}

// ---- TDM ring ops of waves 0..2: global byte offset of the tile of step (qt, gh) ----------------
// Affine in (qt, gh): the kernel builds it as base + qt qstep + gh hstep, the steps derived here.
BWD_HD int64_t role_step_byte(int64_t base, int64_t qstep, int64_t hstep, int qt, int gh) {
    return base + (int64_t) qt * qstep + (int64_t) gh * hstep;
}
BWD_HD int64_t role_tile_byte_s(int role, int64_t sb, int64_t ss, int b, int Sq, int Hq, int hkv,
                                int G, int qt, int gh) {
    return role == ISS_LD ? ld_rec_byte(b, Hq, qh_of(hkv, G, gh), Sq, QS * qt)
                          : qdo_tile_byte_s(sb, ss, b, qh_of(hkv, G, gh), QS * qt);
}
BWD_HD int64_t role_qstep_s(int role, int64_t ss, int Sq, int Hq) {
    return role_tile_byte_s(role, 0, ss, 0, Sq, Hq, 0, 1, 1, 0) -
           role_tile_byte_s(role, 0, ss, 0, Sq, Hq, 0, 1, 0, 0);
}
BWD_HD int64_t role_hstep_s(int role, int64_t ss, int Sq, int Hq) {
    return role_tile_byte_s(role, 0, ss, 0, Sq, Hq, 0, 1, 0, 1) -
           role_tile_byte_s(role, 0, ss, 0, Sq, Hq, 0, 1, 0, 0);
}
BWD_HD int64_t role_stride_el_s(int role, int64_t ss) {
    return role == ISS_LD ? 64 : ss;
}
// LDS slot of the role's ring (affine in the slot: base and stride selected once)
BWD_HD int role_lds(int role, int slot) {
    return role == ISS_DO ? lds_do(slot) : role == ISS_Q ? lds_q(slot) : lds_ld(slot);
}
BWD_HD int role_lds0(int role) {
    return role_lds(role, 0);
}
BWD_HD int role_lds_step(int role) {
    return role_lds(role, 1) - role_lds(role, 0);
}
BWD_HD int role_lds_at(int lds0, int lstep, int slot) {
    return lds0 + slot * lstep;
}
BWD_HD int role_rows(int role) {
    return role == ISS_LD ? 1 : QS;
}
// wave 3's dummy source: the first 16 B of K row kvw0
BWD_HD int64_t dummy_src_byte_s(int64_t k_sb, int64_t k_ss, int b, int hkv, int kvw0) {
    return kv_tile_byte_s(k_sb, k_ss, b, hkv, kvw0);
}

// per-wave parts of the LDS operand bases (the runtime w folded into the lane base, so every
// fragment coordinate stays a compile-time immediate)
BWD_HD int kimg_rows_wave(int w) {
    return KVW * w * XP;
}
BWD_HD int ds_rows_wave(int w) {
    return KVW * w * SP;
}

// ---- epilogue -----------------------------------------------------------------------------------
// The dV^T / dK^T accumulator (dt, kh) of lane (r, h) goes to bf16 row kv = kvw0 + 32 w + 16 kh +
// r, d = 16 dt + 8h + [0, 8), straight from registers. Each workgroup's resource spans its slab of
// rows kvw0 .. kvw0 + 127 of head hkv (127 seq strides + one row); the lane voffsets use the int32
// byte row stride ss_b = 2 ss.
BWD_HD int64_t epi_bytes_s(int64_t ss) {
    return (int64_t) (BKV - 1) * ss * 2 + ROW_B;
}
BWD_HD int epi_voff_s(int lane, int w, int kh, int dt, int ss_b) {
    return (KVW * w + 16 * kh + lane % 16) * ss_b + (16 * dt + 8 * (lane / 16)) * 2;
}
BWD_HD int epi_voff_kh_s(int lane, int w, int kh, int ss_b) {
    return epi_voff_s(lane, w, kh, 0, ss_b);
}
BWD_HD int epi_imm(int dt) {
    return 32 * dt;
}

// prep (grid (Sq / 64, Hq, B)) and convert (grid (Sq / 32, Hq, B)) thread maps
constexpr int PREP_THREADS = 256, PREP_ROWS = 64, PREP_LDS_B = PREP_ROWS * 4;
constexpr int CVT_THREADS = 256, CVT_LDS_B = QS * CVT_PITCH; // 8704
BWD_HD int    prep_chunk(int t) {
    return t % 16;
}
BWD_HD int prep_zero_off(int t, int i) {
    return 16 * (t + PREP_THREADS * i);
} // i = 0..7: 32 KB per workgroup
BWD_HD int cvt_src_off(int t, int i) {
    return 16 * (t + CVT_THREADS * i);
} // i = 0..3: the 16 KB block
BWD_HD int cvt_row(int t, int i) {
    return (t + CVT_THREADS * i) / 16;
} // i = 0..1: 32 rows x 16 chunks
BWD_HD int cvt_chunk(int t, int i) {
    return (t + CVT_THREADS * i) % 16;
}

// ---- main-kernel step schedule ------------------------------------------------------------------
// TDM: op(k) is issued at the top of step k - 3, retired at the signal of step k - 2 (tensorcnt 1)
// and read from step k - 1 on; tile s - 1's ring slot is free at the top of step s.
constexpr int RING = 4, NDS = 2, LEAD = 3, TCNT = 1;
static_assert(RING <= RING_MAX && NDS <= NDS_MAX, "LDS map reserves the depths");

// E = dV^T (16 WMMA) then dK^T (16 WMMA); E1 = the first NE1 before the signal, E2 = the last NWIN.
constexpr int NWIN = 16;
constexpr int NE = 2 * NDO * 2, NE1 = NE - NWIN;

// VALU-class ops the causal mask adds to one softmax phase (2 blocks x (8 v_cmp + 8 v_cndmask +
// threshold)); sizes the scheduling groups of a diagonal step.
constexpr int MASK_VALU = 34;

// The read-back of step s + 1's operands as 24 items in consumption order: items 0..15 = the S / dP
// A operands (dt-major: qA hh 0, qA hh 1, dA hh 0, dA hh 1; 2 ds_load_b128 each), items 16..23 =
// the LD records ld[hh][k] (1 ds_load_b128 each).
constexpr int RB_ITEMS = 24;
BWD_HD int    rb_loads(int items) {
    return 2 * (items < 16 ? items : 16) + (items > 16 ? items - 16 : 0);
}
// items 0..RB1-1 are issued before the signal (in E1, not covered by the barrier), the rest in E2
constexpr int RB1  = 8;
constexpr int NUNC = rb_loads(RB1); // LDS loads in flight past a full step's signal
static_assert(NUNC >= 0 && NUNC < 64, "dscnt is a 6-bit field");
// item decode: kind 0 qA, 1 dA (hh, dt), 2 ld (hh, k)
BWD_HD int rb_kind(int it) {
    return it < 16 ? (it % 4) / 2 : 2;
}
BWD_HD int rb_hh(int it) {
    return it < 16 ? it % 2 : (it - 16) / 4;
}
BWD_HD int rb_dt(int it) {
    return it < 16 ? it / 4 : -1;
}
BWD_HD int rb_k(int it) {
    return it < 16 ? -1 : (it - 16) % 4;
}

// dQ atomics: 32 per wave and step, ATOM_E1 of them in E1 and the rest in E2
constexpr int NATOM   = 32;
constexpr int ATOM_E1 = (NATOM * NE1) / NE;

// even split of x items over n groups: the items of group g
BWD_HD int share(int x, int n, int g) {
    return (x * (g + 1)) / n - (x * g) / n;
}

} // namespace bwd
