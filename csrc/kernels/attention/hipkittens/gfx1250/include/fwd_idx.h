/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// Index formulas of the forward kernel (k_attn_fwd.hip): head dim 128, bf16, GQA, bottom-right
// causal (query p sees kv <= p + Skv - Sq).
//
//   O[b, p, hq, :] = softmax_kv(scale * Q[b, p, hq, :] K[b, kv, hkv, :]^T) V[b, kv, hkv, :]
//   LSE[b, hq, p]  = natural log of the softmax denominator
//
// Work item: 64 query positions of one head quad (4 q heads of one kv head), 4 waves of 16
// positions. K / V reach LDS through a depth-4 TDM ring of 64-row stages (one 32-row op per wave
// per stage) with one split-barrier phase per stage. A persistent grid of W workgroups runs the
// items:
//   MODE_ITEMS  item v = j W + g in longest-first order (W = #items: one item per workgroup);
//   MODE_PAIRS  units = zig-zag pairs (q block nqb - 1 - pr, then pr) of one head, plus one middle
//               block per head when nqb is odd; unit v = s W + rank(g) with the rank XCD-major, so
//               the workgroups of one XCD work on one head at a time. Every pair has the same stage
//               count.
// The K / V op stream, ring slots and barrier phases are workgroup-lifetime counters over the
// global stage T = T0_j + t (T0_j = the stage count of items 0 .. j - 1). Every stage waits for
// phase T + 1 and issues op(T + 3) at its even barrier point, and retires its TDM ops and signals
// phase T + 2 at its odd one, so the next item's first stages and its Q tile load under the current
// item's last stages. Ops past the stream are 16-B dummies (constant tensorcnt waits). Q and the O
// image have separate per-wave LDS regions. Q, K, V and O are each addressed by their own (batch,
// seq) element strides (Lay), so every tensor may be contiguous BSHD or a BSHD view of SBHD
// storage.
#pragma once

#define FWD_HD __host__ __device__ __attribute__((always_inline)) constexpr

#include <stdint.h>

namespace fwd {

constexpr int D     = 128;
constexpr int WAVE  = 32;
constexpr int NWAVE = 4; // one wave per SIMD, one workgroup per CU
constexpr int BLOCK = NWAVE * WAVE;
constexpr int QB    = 64; // query positions per item
constexpr int QW    = 16; // query positions per wave
constexpr int NH    = 4;  // q heads per item = q tiles per wave
constexpr int ST    = 64; // kv rows per ring stage
constexpr int CH    = 32; // kv rows per chunk
constexpr int OPR   = 32; // kv rows per TDM op (one op per wave per stage)
constexpr int NDK   = D / 32;
constexpr int NDT   = D / 16;
constexpr int NXCD  = 8; // XCC = workgroup id % 8

// LDS pitches (bytes) and ring geometry
constexpr int K_PITCH  = 272;
constexpr int V_PITCH  = 288;
constexpr int Q_PITCH  = 1040;
constexpr int KD       = 4;
constexpr int K_SLOT_B = ST * K_PITCH; // 17408
constexpr int V_SLOT_B = ST * V_PITCH; // 18432
constexpr int QO_W_B   = QW * Q_PITCH; // 16640: one wave's Q staging / O image

// LDS map (byte offsets)
constexpr int K_OFF     = 0;                          // K ring [0, 69632)
constexpr int O_OFF_LO  = K_OFF + KD * K_SLOT_B;      // O images of waves 0..2: [69632, 119552)
constexpr int DUMMY_OFF = 130048;                     // 4 K/V dummy cells + 4 Q dummy cells (128 B)
constexpr int V_OFF     = 131072;                     // V ring [131072, 204800)
constexpr int O_OFF_HI  = V_OFF + KD * V_SLOT_B;      // O image of wave 3: [204800, 221440)
constexpr int LDS_BYTES = 327680;                     // the whole 320 KB: one workgroup per CU
constexpr int Q_OFF     = LDS_BYTES - NWAVE * QO_W_B; // 4 per-wave Q regions [261120, 327680)

FWD_HD int op_role_v(int w) {
    return w >> 1;
} // waves 0, 1: K; waves 2, 3: V
FWD_HD int op_half(int w) {
    return w & 1;
}
constexpr int TCNT_STEADY = 1; // odd barrier point of every stage but an item's last
constexpr int TCNT_LAST   = 2; // odd barrier point of an item's last stage (Q(j+1) in flight too)
constexpr int TCNT_ITEM   = 1; // item prologue: own Q(j) landed (op(T0+2) may fly)

FWD_HD uint32_t tdm_w0(int row_b, int pitch_b) {
    int lg = 0;
    for (int x = row_b / 4; x > 1; x /= 2)
        ++lg;
    return (1u << 16) | (1u << 20) | ((uint32_t) (lg - 1) << 22) |
           ((uint32_t) ((pitch_b - row_b) / 4 - 1) << 25);
}
constexpr uint32_t K_W0 = tdm_w0(2 * D, K_PITCH);      // 0x07510000
constexpr uint32_t V_W0 = tdm_w0(2 * D, V_PITCH);      // 0x0f510000
constexpr uint32_t Q_W0 = tdm_w0(2 * NH * D, Q_PITCH); // 0x07d10000

constexpr float LOG2E = 1.4426950408889634f;
constexpr float LN2   = 0.6931471805599453f;
constexpr float TH2   = 8.0f * LOG2E; // lazy-rescale threshold: 8 nat in the log2 domain

FWD_HD int imin(int a, int b) {
    return a < b ? a : b;
}
FWD_HD int imax(int a, int b) {
    return a > b ? a : b;
}

// ---- work items ---------------------------------------------------------------------------------
struct Work {
    int b, hq4, qb;
};
// kv head of head quad hq4 = (NH hq4) / G with G a power of two (4, 8 or 16): a shift
FWD_HD int hkv_of(int hq4, int G) {
    return (NH * hq4) >> __builtin_ctz((unsigned) G);
}
FWD_HD int wave_pos0(int qb, int w) {
    return QB * qb + QW * w;
}
FWD_HD int n_stages(int qb, int cst, int nkv_st, int causal) {
    return causal != 0 ? qb + 1 + cst : nkv_st;
}

// Division by a runtime constant with a host-computed magic number: for n < 2^31,
// n / d == (n m) >> sh with sh = 31 + ceil(log2 d) and m = ceil(2^sh / d) < 2^32 (s_mul_u64 +
// s_lshr_b64). The host passes (m, sh) of every divisor as kernel arguments.
struct Fdiv {
    uint32_t m;
    int      sh;
};
FWD_HD uint32_t fdiv(uint32_t n, const Fdiv &f) {
    return (uint32_t) (((uint64_t) n * f.m) >> f.sh);
}

// ---- persistent schedule ------------------------------------------------------------------------
enum : int { MODE_ITEMS = 0, MODE_PAIRS = 1 };
// dW, dnp, dh4, dnh: the Fdiv of W, max(nqb / 2, 1), nhq4 and B nhq4
struct Sched {
    int  mode, W, B, nhq4, nqb;
    Fdiv dW, dnp, dh4, dnh;
};
FWD_HD int n_items(int B, int nhq4, int nqb) {
    return B * nhq4 * nqb;
}
FWD_HD int n_pair_units(int B, int nhq4, int nqb) {
    return B * nhq4 * (nqb / 2);
}
FWD_HD int n_units(int mode, int B, int nhq4, int nqb) {
    return mode == MODE_PAIRS ? B * nhq4 * ((nqb + 1) / 2) : n_items(B, nhq4, nqb);
}
FWD_HD int n_units(const Sched &s) {
    return n_units(s.mode, s.B, s.nhq4, s.nqb);
}
// unit order position of workgroup g: XCD-major in pairs mode (a permutation of [0, W) when
// W % 8 == 0)
FWD_HD int wg_rank(const Sched &s, int g) {
    return s.mode == MODE_PAIRS && s.W % NXCD == 0 ? (g % NXCD) * (s.W / NXCD) + g / NXCD : g;
}
// number of s >= 0 with s W + r < lim (every operand non-negative and < 2^31)
FWD_HD int count_below(int lim, int r, const Fdiv &dW) {
    return r < lim ? (int) fdiv((uint32_t) (lim - 1 - r), dW) + 1 : 0;
}
FWD_HD int wg_items(const Sched &s, int g) {
    const int r = wg_rank(s, g);
    if (s.mode != MODE_PAIRS)
        return count_below(n_items(s.B, s.nhq4, s.nqb), r, s.dW);
    const int np = count_below(n_pair_units(s.B, s.nhq4, s.nqb), r, s.dW);
    const int nu = count_below(n_units(s), r, s.dW);
    return 2 * np + (nu - np);
}
// item j (0 <= j < wg_items) of workgroup g; every quotient by a runtime value is an fdiv
FWD_HD Work wg_item(const Sched &s, int g, int j) {
    const int r = wg_rank(s, g);
    if (s.mode != MODE_PAIRS) { // longest first: the heaviest q block of every head first
        const uint32_t i = (uint32_t) (j * s.W + r), nh = (uint32_t) (s.B * s.nhq4);
        const uint32_t rank = fdiv(i, s.dnh), rem = i - rank * nh, b = fdiv(rem, s.dh4);
        return Work{(int) b, (int) (rem - b * (uint32_t) s.nhq4), s.nqb - 1 - (int) rank};
    }
    const uint32_t npair = (uint32_t) (s.nqb / 2), NP = (uint32_t) n_pair_units(s.B, s.nhq4, s.nqb);
    const int      np   = count_below((int) NP, r, s.dW);
    const bool     pair = j < 2 * np;
    // pair unit j / 2 (heavy block first) or single middle block (odd nqb) number j - np
    const uint32_t v  = (uint32_t) ((pair ? j >> 1 : j - np) * s.W + r);
    const uint32_t hp = fdiv(v, s.dnp), pr = v - hp * npair;
    const uint32_t hb = pair ? hp : v - NP;
    const int      qb = pair ? ((j & 1) == 0 ? s.nqb - 1 - (int) pr : (int) pr) : s.nqb / 2;
    const uint32_t b  = fdiv(hb, s.dh4);
    return Work{(int) b, (int) (hb - b * (uint32_t) s.nhq4), qb};
}

// ---- causal mask (frag.h mask_row_gt_col; only the last stage of an item is masked) -------------
constexpr int MASK_NONE = 64;
FWD_HD bool   last_stage_chunk(int c, int n) {
    return c >= 2 * n - 2;
}
FWD_HD int mask_off(int w, int c, int n, int kt, int causal) {
    return causal != 0 && last_stage_chunk(c, n) ? QW * (w - (2 * (c - (2 * n - 2)) + kt))
                                                 : MASK_NONE;
}

// ---- ring (global stage T) ----------------------------------------------------------------------
FWD_HD int slot(int T) {
    return T & (KD - 1);
}
static_assert((KD & (KD - 1)) == 0, "slot = T & (KD - 1)");
FWD_HD int k_slot_off(int T) {
    return K_OFF + slot(T) * K_SLOT_B;
}
FWD_HD int v_slot_off(int T) {
    return V_OFF + slot(T) * V_SLOT_B;
}
FWD_HD int dummy_off(int w) {
    return DUMMY_OFF + 16 * w;
} // K/V dummy cell of wave w
FWD_HD int q_dummy_off(int w) {
    return DUMMY_OFF + 64 + 16 * w;
} // Q dummy cell of wave w
FWD_HD int o_lds(int w) {
    return w < 3 ? O_OFF_LO + w * QO_W_B : O_OFF_HI;
}
FWD_HD int q_op_lds(int w) {
    return Q_OFF + w * QO_W_B;
}

FWD_HD int64_t lse_el(int b, int Hq, int hq, int Sq, int p) {
    return ((int64_t) b * Hq + hq) * Sq + p;
}

// ---- TDM descriptors (tensor_dim1 = the rows below the origin, clamped to the tile) -------------
struct G1 {
    uint32_t w[8];
};
FWD_HD G1 g1_2d(uint32_t w0, uint32_t tdim0, uint32_t tdim1, uint32_t tile0, uint32_t tile1,
                uint64_t stride0) {
    return G1{{w0, (tdim0 & 0xFFFFu) << 16, (tdim0 >> 16) | ((tdim1 & 0xFFFFu) << 16),
               (tdim1 >> 16) | (tile0 << 16), tile1, (uint32_t) stride0,
               (uint32_t) (stride0 >> 32) & 0xFFFFu, 0u}};
}
// Q op of wave w for item (b, hq4, qb): [16 positions][4 heads x 128]
FWD_HD int q_op_dim1(int Sq, int qb, int w) {
    return imin(imax(Sq - wave_pos0(qb, w), 0), QW);
}
// K / V op of wave w for local stage s of an item: rows [64 s + 32 half, +32)
FWD_HD int kv_op_row0(int w, int s) {
    return ST * s + OPR * op_half(w);
}
// clamp(Skv - row0, 0, 32) in 32-bit unsigned arithmetic (SALU only)
FWD_HD int kv_op_dim1(int Skv, int w, int s) {
    const uint32_t a = (uint32_t) Skv, r0 = (uint32_t) kv_op_row0(w, s);
    const uint32_t d = a > r0 ? a - r0 : 0u;
    return (int) (d < (uint32_t) OPR ? d : (uint32_t) OPR);
}
// 16-B dummy op: an 8 x 1 element tile at row 0 of a tensor that exists
FWD_HD G1 dummy_g1(int role_v) {
    return g1_2d(role_v ? V_W0 : K_W0, 8, 1, 8, 1, 0);
}
// LDS address of a real K / V op of wave w at global stage T
FWD_HD int kv_op_lds(int w, int T) {
    return op_role_v(w) ? v_slot_off(T) + OPR * op_half(w) * V_PITCH
                        : k_slot_off(T) + OPR * op_half(w) * K_PITCH;
}

// ---- the K / V op stream ------------------------------------------------------------------------
// Stage k (0, 1, 2) of the stream after the first j1 items of the workgroup: the owning item and
// its local stage, or none (a dummy).
struct StRef {
    int real, b, hkv, s;
};
struct Shape {
    int B, Sq, Skv, Hq, Hkv, G, nqb, nhq4, nkv_st, cst, causal;
};

// Stages 0, 1, 2 of the stream: items j1, j1 + 1, j1 + 2 decoded once each (an index past the list
// is clamped to the last item and gets 0 stages), then selects only.
FWD_HD void stream3(const Shape &sh, const Sched &sc, int g, int J, int j1, StRef (&out)[3]) {
    const Work w0 = wg_item(sc, g, imin(j1, J - 1)), w1 = wg_item(sc, g, imin(j1 + 1, J - 1)),
               w2 = wg_item(sc, g, imin(j1 + 2, J - 1));
    const int n0  = j1 < J ? n_stages(w0.qb, sh.cst, sh.nkv_st, sh.causal) : 0;
    const int n1  = j1 + 1 < J ? n_stages(w1.qb, sh.cst, sh.nkv_st, sh.causal) : 0;
    const int n2  = j1 + 2 < J ? n_stages(w2.qb, sh.cst, sh.nkv_st, sh.causal) : 0;
    const int h0 = hkv_of(w0.hq4, sh.G), h1 = hkv_of(w1.hq4, sh.G), h2 = hkv_of(w2.hq4, sh.G);
    for (int k = 0; k < 3; ++k) { // selects only (a runtime-indexed local array would be scratch)
        const int  r1 = k - n0, r2 = k - n0 - n1;
        const bool i0 = k < n0, i1 = !i0 && r1 < n1, re = i0 || i1 || r2 < n2;
        const int  s = i0 ? k : i1 ? r1 : r2;
        out[k]       = StRef{re ? 1 : 0,
                       i0   ? w0.b
                             : i1 ? w1.b
                                  : w2.b,
                       i0   ? h0
                       : i1 ? h1
                            : h2,
                       re ? s : 0};
    }
}

// Compact TDM op (2 SGPRs): d1 > 0 = a real op of d1 rows at element el of the tensor, d1 == 0 = a
// 16-B dummy. Element offsets fit 31 bits (the host keeps every Q / K / V extent < 2^31 B).
struct OpC {
    uint32_t el;
    int      d1;
};
// the item's own K / V op at local stage s: el0 + s * sst (sst = elements per stage)
FWD_HD OpC own_c(int Skv, uint32_t el0, uint32_t sst, int w, int s) {
    return OpC{el0 + (uint32_t) s * sst, kv_op_dim1(Skv, w, s)};
}

// ---- LDS read addresses (per-lane base + compile-time DS immediate) -----------------------------
FWD_HD int k_lane(int l) {
    return (l % 16) * K_PITCH + (l / 16) * 16;
}
FWD_HD int k_imm(int hc, int kt, int k, int half) {
    return (CH * hc + 16 * kt) * K_PITCH + 64 * k + 32 * half;
}
FWD_HD int v_lane(int l) {
    return (8 * (l / 16) + l % 8) * V_PITCH + ((l / 8) % 2) * 16;
}
FWD_HD int v_imm(int hc, int half, int dt) {
    return (CH * hc + 16 * half) * V_PITCH + 32 * dt;
}
FWD_HD int q_lane(int w, int l) {
    return q_op_lds(w) + (l % 16) * Q_PITCH + (l / 16) * 16;
}
FWD_HD int q_imm(int h, int k, int half) {
    return 2 * D * h + 64 * k + 32 * half;
}
// O image (the wave's O region in Q's layout): write = the O^T C fragment of (h, dt)
FWD_HD int ow_lane(int w, int l) {
    return o_lds(w) + (l % 16) * Q_PITCH + (l / 16) * 16;
}
FWD_HD int ow_imm(int h, int dt) {
    return 2 * D * h + 32 * dt;
}
// read-back of store s (0..31): position s / 2, bytes [512 (s % 2), +512) of the position's 1 KB
FWD_HD int or_lane(int w, int l) {
    return o_lds(w) + 16 * l;
}
FWD_HD int or_imm(int s) {
    return (s / 2) * Q_PITCH + 512 * (s % 2);
}

// LSE store byte offset (the host keeps every extent < 2^31 B)
FWD_HD int64_t lse_store_b(int b, int Hq, int Sq, int hq4, int qb, int w, int k, int l) {
    return 4 * lse_el(b, Hq, NH * hq4 + 2 * (l / 16) + k, Sq, wave_pos0(qb, w) + l % 16);
}

// ---- strided tensors ----------------------------------------------------------------------------
// Q, K, V and O are each addressed through two int64 element strides, Lay{sb (batch), ss (seq)}:
// element (b, s, h, d) at b sb + s ss + h D + d. The host admits two layouts per [B, S, H, 128]
// tensor, both bijections onto [0, B S H D):
//   contiguous BSHD            sb = S H D, ss = H D
//   SBHD storage, BSHD view    sb = H D,   ss = B H D
// The TDM row stride (48 bits) is ss; the host keeps every stride in [D, 2^31).
struct Lay {
    int64_t sb, ss;
};
FWD_HD int64_t el_s(const Lay &L, int b, int s, int h) {
    return (int64_t) b * L.sb + (int64_t) s * L.ss + (int64_t) h * D;
}
// Q op: [16 positions][4 heads x 128] at row stride ss (the 4 heads of a quad are contiguous)
FWD_HD G1 q_g1_s(int64_t ss, int dim1) {
    return g1_2d(Q_W0, NH * D, (uint32_t) dim1, NH * D, QW, (uint64_t) ss);
}
FWD_HD int64_t q_op_el_s(const Lay &L, int b, int hq4, int qb, int w) {
    return el_s(L, b, wave_pos0(qb, w), NH * hq4);
}
// The K / V seq stride is held in one SGPR across the stage loop (exact: ss < 2^31, so the
// descriptor's high stride word is 0).
FWD_HD uint32_t kv_ss32(int64_t ss) {
    return (uint32_t) ss;
}
FWD_HD G1 kv_g1_s(int role_v, uint32_t ss32, int dim1) {
    return g1_2d(role_v ? V_W0 : K_W0, D, (uint32_t) dim1, D, OPR, (uint64_t) ss32);
}
FWD_HD int64_t kv_op_el_s(const Lay &L, int b, int hkv, int w, int s) {
    return el_s(L, b, kv_op_row0(w, s), hkv);
}
FWD_HD OpC op_c_s(const Shape &sh, const Lay &L, const StRef &r, int w) {
    const uint32_t el = (uint32_t) kv_op_el_s(L, r.b, r.hkv, w, r.s);
    const int      d1 = kv_op_dim1(sh.Skv, w, r.s);
    return OpC{r.real ? el : 0u, r.real ? d1 : 0};
}
FWD_HD uint32_t kv_el0_s(const Lay &L, int b, int hkv, int w) {
    return (uint32_t) kv_op_el_s(L, b, hkv, w, 0);
}
FWD_HD uint32_t kv_sst_s(uint32_t ss32) {
    return (uint32_t) ST * ss32;
} // elements per 64-row stage (own_c's step)
FWD_HD OpC q_c_s(int real, const Work &x, int Sq, const Lay &L, int w) {
    const uint32_t el = (uint32_t) q_op_el_s(L, x.b, x.hq4, x.qb, w);
    const int      d1 = q_op_dim1(Sq, x.qb, w);
    return OpC{real ? el : 0u, real ? d1 : 0};
}
// O store s (0..31) of lane l: position s / 2, bytes [512 (s % 2) + 16 l, +16) of the position's
// 4-head row
FWD_HD int64_t o_store_b_s(const Lay &L, int b, int hq4, int qb, int w, int s, int l) {
    return 2 * el_s(L, b, wave_pos0(qb, w) + s / 2, NH * hq4) + 512 * (s % 2) + 16 * l;
}

// Stage n - 2 of an n-stage item runs outside the steady loop when n >= 2.
FWD_HD bool has_pen_stage(int n) {
    return n >= 2;
}

} // namespace fwd
