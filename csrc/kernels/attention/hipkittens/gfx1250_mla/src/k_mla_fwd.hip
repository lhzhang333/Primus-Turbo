/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// k_mla_fwd -- the DeepSeek-V3 MLA attention forward on gfx1250: qk head dim 192, v 128, bf16, MHA, bottom-right
// causal. out / lse are bitwise those of the FlyDSL forward (primus_turbo/flydsl/attention/gfx1250_mla): the same
// per-row operations in the same order.
//   Q [B, Sq, H, 192], K [B, Skv, H, 192], V [B, Skv, H, 128] -> O [B, Sq, H, 128] bf16, LSE [B, H, Sq] fp32 (natural
//   log); Sq % 64 == 0, Sq <= Skv; query i sees keys j <= i + Skv - Sq.
//
// Structure (FlyDSL's eight-wave design): a workgroup of 8 wave32 waves owns 256-row q blocks of one (batch, head) and
// streams 64-row K / V tiles through two LDS slots (TDM copies), one split barrier per tile. Wave w owns rows
// 32 w .. 32 w + 31 of the block (two 16-row q tiles). Waves w and w + 4 share a SIMD; the high one (w >= 4) loses
// issue arbitration to the low one, so it is the critical path. Per tile and wave: QK (48 WMMAs) of both q tiles, then
// q tile 0's softmax, then q tile 0's PV interleaved with q tile 1's softmax, then q tile 1's PV. A high wave arrives
// at the tile's barrier after q tile 0's softmax (its K / V reads are done), a low wave after both softmaxes, where it
// also waits for its TDM copy of rows 16 w .. 16 w + 15 of the next tile. Only a wave's own diagonal tiles (and the kv
// tail) run masked; tiles past a wave's rows only keep the barrier and the copies.
// Persistent grid (npw > 0, one workgroup per CU): a static schedule (see first_blk). The low waves copy the next
// block's Q (theirs and their partner's) and first tile in their last, sync-only tile, so a block starts without a
// barrier, and the next block's Q is read before the previous block's output is written. npw == 0: one workgroup per
// block, longest q blocks first.
//
// Numerics (FlyDSL's): m in log2 units of the scaled score; the running max is updated only when some row's max grew by
// more than log2(e) (wave-uniform per q tile); p = exp2(s * scale * log2(e) - m) as v_exp_f32; row sums by 3-ary trees;
// out = bf16(O * (d > 0 ? 1 / d : 0)), lse = (m + log2 d) ln 2. Floating-point contraction is off.
#include "fwd_prims.h"

#pragma clang fp contract(off)

using namespace f1;

namespace {
constexpr int D_QK = 192, D_V = 128;
constexpr int NW = 8;           // waves
constexpr int QT = 2;           // q tiles per wave
constexpr int BM = 16 * QT * NW;  // 256 query rows per block
constexpr int NB = 64;          // kv rows per tile
constexpr int NKS = D_QK / 32;  // QK k-steps
constexpr int NKV = NB / 16;    // 16-row kv sub-tiles
constexpr int NDT = D_V / 16;   // 16-column O tiles
constexpr int XK = 400;         // Q / K LDS row bytes (192 bf16 + 16 B pad)
constexpr int XV = 288;         // V LDS row bytes
constexpr int XO = 272;         // O staging row bytes
// LDS: 64 KB segments behind two read ports that read in the same cycle only from different segments. Each K / V
// slot straddles a segment boundary at its middle row; Q and O have per-wave regions outside the slots.
constexpr int SEG = 64 * 1024;
constexpr int SLOT = 2 * SEG;
constexpr int K_LDS = NB * XK;
constexpr int V_LDS = NB * XV;
constexpr int K_SLOT_OFF = SEG - K_LDS / 2;          // K slot p centred on (2p + 1) * 64 KB
constexpr int V_SLOT_OFF = 2 * SEG - V_LDS / 2;      // V slot p centred on (2p + 2) * 64 KB
constexpr int QRB = 16 * QT * XK;                    // a wave's Q region
constexpr int Q_HI_OFF = SLOT + V_SLOT_OFF + V_LDS;  // waves 4..7 (waves 0..3 at 0)
constexpr int OWB = 16 * QT * XO;                    // a wave's O region
constexpr int O_LO_OFF = K_SLOT_OFF + K_LDS;         // waves 0..3: between K and V of slot 0
constexpr int O_HI_OFF = V_SLOT_OFF + V_LDS + 32;    // waves 4..7: between V of slot 0 and K of slot 1
constexpr int LDS_B = 320 * 1024;
static_assert(4 * QRB <= K_SLOT_OFF, "low Q regions below K slot 0");
static_assert(O_LO_OFF + 4 * OWB <= V_SLOT_OFF, "low O regions between K0 and V0");
static_assert(O_HI_OFF + 4 * OWB <= SLOT + K_SLOT_OFF, "high O regions between V0 and K1");
static_assert(Q_HI_OFF + 4 * QRB <= LDS_B, "high Q regions above V slot 1");
static_assert(SLOT == 1 << 17, "slot stride");
constexpr int CROWS = 16;               // rows of a tile per copying wave
constexpr unsigned W0_A = 0x47510000u;  // TDM: 128 columns, pad 72 elements -> 400-B rows
constexpr unsigned W0_B = 0x87110000u;  // TDM: 64 columns, pad 136 elements -> 400-B rows
constexpr unsigned W0_V = 0x0F510000u;  // TDM: 128 columns, pad 16 elements -> 288-B rows
constexpr float LOG2E = 1.4426950408889634f;
constexpr float LN2 = 0.69314718055994531f;
constexpr float BIG_NEG = -1.0e30f;  // finite running-max seed: a fully masked row has no -inf - -inf
// a q tile's 16 PV WMMAs: kv-step major (an accumulator's previous WMMA 8 WMMAs back) or d-tile major
constexpr bool PV_KV_MAJOR = true;

F1_INLINE i32x8 tdm_g1(unsigned w0, unsigned tdim1, unsigned tile0, unsigned tile1, unsigned long long stride_el) {
  return i32x8{(int)w0, (int)0xFFFF0000u, (int)(((tdim1 & 0xFFFFu) << 16) | 0x7FFFu),
               (int)(((tdim1 >> 16) & 0xFFFFu) | (tile0 << 16)), (int)tile1, (int)(unsigned)stride_el,
               (int)((unsigned)(stride_el >> 32) & 0xFFFFu), 0};
}
F1_INLINE i32x4 tdm_g0(unsigned lds, unsigned long long ga) {
  return i32x4{1, (int)lds, (int)(unsigned)ga, (int)((unsigned)(ga >> 32) | 0x80000000u)};
}

template <typename T, int N>
F1_INLINE T tree_sum(const T (&x)[N]) {
  if constexpr (N == 1) {
    return x[0];
  } else {
    constexpr int M = N / 3 + ((N % 3) ? 1 : 0);
    T y[M];
    static_for<M>([&](auto j) {
      constexpr int i = 3 * j;
      if constexpr (N - i >= 3) {
        y[j] = (x[i] + x[i + 1]) + x[i + 2];
      } else if constexpr (N - i == 2) {
        y[j] = x[i] + x[i + 1];
      } else {
        y[j] = x[i];
      }
    });
    return tree_sum<T, M>(y);
  }
}
template <int N>
F1_INLINE float tree_max(const float (&x)[N]) {
  if constexpr (N == 1) {
    return x[0];
  } else {
    constexpr int M = N / 3 + ((N % 3) ? 1 : 0);
    float y[M];
    static_for<M>([&](auto j) {
      constexpr int i = 3 * j;
      if constexpr (N - i >= 3) {
        y[j] = __builtin_fmaxf(__builtin_fmaxf(x[i], x[i + 1]), x[i + 2]);
      } else if constexpr (N - i == 2) {
        y[j] = __builtin_fmaxf(x[i], x[i + 1]);
      } else {
        y[j] = x[i];
      }
    });
    return tree_max<M>(y);
  }
}

// the kernel arguments as the kernarg segment holds them
struct KArgs {
  const bf16* Q;
  const bf16* K;
  const bf16* V;
  bf16* O;
  float* LSE;
  float scale;
  int Sq, Skv, H, B, nqb, npw, pstep;
};
static_assert(sizeof(KArgs) == 72, "kernarg layout");

// the arguments re-read from the kernarg segment at offset 48 z (z an opaque 0): LLVM hoists nothing argument-derived
// out of the block loop (it would stay live across the tile loops)
F1_INLINE KArgs kargs(int z) {
  return *(const __attribute__((address_space(4))) KArgs*)(
      (const __attribute__((address_space(4))) unsigned char*)__builtin_amdgcn_kernarg_segment_ptr() + 48 * z);
}

// a block: q block, head, batch, and hb = batch * H + head
struct Blk {
  int qb, hh, bb, hb;
};

// npw == 0: the block of this workgroup of the (nqb, H, B) grid, longest q blocks first (FlyDSL's remap: with
// nqb and H B multiples of 16, windows of 16 q blocks go out 16 heads at a time)
F1_INLINE void remap_block(int gx, int gy, int gz, int& bx, int& by, int& bz) {
  const int lin = (int)__builtin_amdgcn_workgroup_id_x() +
                  gx * ((int)__builtin_amdgcn_workgroup_id_y() + gy * (int)__builtin_amdgcn_workgroup_id_z());
  const int gyz = gy * gz;
  int rank = lin / gyz;
  int rem = lin - rank * gyz;
  const int r = (rank & 15) * gyz + rem;
  const int c = r & 255;
  const bool grouped = ((gx & 15) | (gyz & 15)) == 0;
  const int lin2 = grouped ? ((rank & -16) + (c >> 4)) * gyz + ((r >> 8) << 4) + (c & 15) : lin;
  rank = lin2 / gyz;
  rem = lin2 - rank * gyz;
  bx = gx - 1 - rank;
  by = rem % gy;
  bz = rem / gy;
}

// The workgroup's first block; false for an idle one. Persistent (npw > 0): workgroup w, x = w % 8 (its XCD),
// w / 8 = r nqb + i; step s runs head-batch 8 (R s + r) + x (R = npw / 8 / nqb, so pstep = 8 R) and q block i on even
// steps, nqb - 1 - i on odd ones. The host enables it only when that covers the grid exactly with an even number of
// steps per workgroup (attention_hipkittens_gfx1250_mla_impl._mla_fwd_grid).
F1_INLINE bool first_blk(const KArgs& a, Blk& c) {
  if (a.npw > 0) {
    const int wg = (int)__builtin_amdgcn_workgroup_id_x();
    const int g = wg >> 3;
    const int pr = g / a.nqb;
    c.qb = g - pr * a.nqb;
    c.hb = 8 * pr + (wg & 7);
    c.bb = c.hb / a.H;
    c.hh = c.hb - c.bb * a.H;
    return wg < a.npw && c.hb < a.H * a.B;
  }
  remap_block(a.nqb, a.H, a.B, c.qb, c.hh, c.bb);
  c.hb = c.bb * a.H + c.hh;
  return true;
}

// the block after c on the persistent schedule; false past the workgroup's last one (always without npw)
F1_INLINE bool next_blk(const KArgs& a, const Blk& c, Blk& nb) {
  nb.qb = a.nqb - 1 - c.qb;
  nb.hb = c.hb + a.pstep;
  int hh = c.hh + a.pstep, bb = c.bb;
  while (hh >= a.H) {
    hh -= a.H;
    ++bb;
  }
  nb.hh = hh;
  nb.bb = bb;
  return a.npw > 0 && nb.hb < a.H * a.B;
}

// the keys the block's last query attends
F1_INLINE int kv_len_of(const KArgs& a, int qb) {
  const int wg_max_seq = min(qb * BM + BM - 1, a.Sq - 1);
  return max(min(wg_max_seq + (a.Skv - a.Sq) + 1, a.Skv), 1);
}

// the first block row of q tile i of wave w
F1_INLINE int qrow(int qb, int w, int i) {
  return qb * BM + 32 * w + 16 * i;
}

F1_INLINE unsigned q_region(uint32_t lds0, int w) {
  return w < NW / 2 ? lds0 + (unsigned)(w * QRB) : lds0 + (unsigned)(Q_HI_OFF + (w - NW / 2) * QRB);
}

// rows 16 ci .. 16 ci + 15 of tile t of block (hh, bb) (kv_len keys) into slot pp: K columns 0..127 and 128..191, V
F1_INLINE void copy_rows(const KArgs& a, uint32_t lds0, int ci, int hh, int bb, int kv_len, int t, int pp) {
  const unsigned long long krow = (unsigned long long)a.H * D_QK, vrow = (unsigned long long)a.H * D_V;
  const int row = t * NB + CROWS * ci;
  const int valid = max(kv_len - row, 0);
  const unsigned long long base = ((unsigned long long)bb * a.Skv + row) * a.H + hh;
  const unsigned long long gk = (unsigned long long)(uintptr_t)a.K + 2ull * base * D_QK;
  const unsigned long long gv = (unsigned long long)(uintptr_t)a.V + 2ull * base * D_V;
  const unsigned lk = lds0 + (unsigned)(K_SLOT_OFF + pp * SLOT + CROWS * ci * XK);
  const unsigned lv = lds0 + (unsigned)(V_SLOT_OFF + pp * SLOT + CROWS * ci * XV);
  tdm_load(tdm_g0(lk, gk), tdm_g1(W0_A, (unsigned)valid, 128, CROWS, krow));
  tdm_load(tdm_g0(lk + 256, gk + 256), tdm_g1(W0_B, (unsigned)valid, 64, CROWS, krow));
  tdm_load(tdm_g0(lv, gv), tdm_g1(W0_V, (unsigned)valid, 128, CROWS, vrow));
}

// the 32 Q rows of wave w of block (qb, hh, bb) into its region; rows past Sq zero-filled
F1_INLINE void copy_q(const KArgs& a, uint32_t lds0, int w, int qb, int hh, int bb) {
  const unsigned long long krow = (unsigned long long)a.H * D_QK;
  const unsigned rg = q_region(lds0, w);
  const int row0 = qrow(qb, w, 0);
  const int n_valid = max(a.Sq - row0, 0);
  const unsigned long long gq =
      (unsigned long long)(uintptr_t)a.Q + 2ull * ((((unsigned long long)bb * a.Sq + row0) * a.H + hh) * D_QK);
  tdm_load(tdm_g0(rg, gq), tdm_g1(W0_A, (unsigned)n_valid, 128, 16 * QT, krow));
  tdm_load(tdm_g0(rg + 256, gq + 256), tdm_g1(W0_B, (unsigned)n_valid, 64, 16 * QT, krow));
}

// a block's staging by low wave wv: the Q of wv and of its SIMD partner wv + 4, and its rows of the block's tile 0
// (slot pp)
F1_INLINE void stage(const KArgs& a, uint32_t lds0, int wv, const Blk& c, int pp) {
  copy_q(a, lds0, wv, c.qb, c.hh, c.bb);
  copy_q(a, lds0, wv + NW / 2, c.qb, c.hh, c.bb);
  copy_rows(a, lds0, wv, c.hh, c.bb, kv_len_of(a, c.qb), 0, pp);
}

// the Q fragments of wave w's two q tiles from its LDS region
F1_INLINE void read_q(uint32_t lds0, int w, int lane, bf16x16 (&qf)[QT][NKS]) {
  const lds_u8* qp = lds_ptr(q_region(lds0, w) + (unsigned)((lane & 15) * XK + (lane >> 4) * 16));
  static_for<QT>([&](auto i) {
    static_for<NKS>([&](auto ks) {
      qf[i][ks] = cat(lds_load_b128(qp, i * 16 * XK + ks * 64), lds_load_b128(qp, i * 16 * XK + ks * 64 + 32));
    });
  });
}

// One wave's blocks. Barriers: tile t (t > 0) waits for every wave's arrive of tile t - 1, which comes after the wave's
// K / V reads of tile t - 1 and, for a low wave, its TDM copy of tile t (so slot t % 2 holds tile t and slot
// (t + 1) % 2 is free for tile t + 1); a block ends with a wait for the arrives of its last tile, in which the low
// waves staged the next block's Q and tile 0 (into the slot of tile n - 2).
template <bool HI>
F1_INLINE void run(uint32_t lds0k, int wvk) {
  constexpr bool COPIER = !HI;  // copies the next tile, and in a block's last tile the next block's Q and tile 0
  Blk cur;
  bf16x16 qf[QT][NKS];
  {
    const KArgs a = kargs(0);
    if (!first_blk(a, cur)) return;  // an idle persistent workgroup (uniform)
    if (COPIER) stage(a, lds0k, wvk, cur, 0);
    // block 0's Q and tile 0 landed and visible
    drain_and_arrive();
    bar_wait();
    read_q(lds0k, wvk, (int)__builtin_amdgcn_mbcnt_lo(~0u, 0u), qf);
  }
  int par = 0;  // the slot of the block's tile 0
  for (;;) {
    // ---- the block's tiles (state: m, d, O, the slot bases) ---------------------------------------------------------
    f32x8 o[QT][NDT];
    float m[QT], d[QT];
    int n;
    {
      // values derived from an opaque zero are not hoisted out of the block loop (they would stay live across the
      // tile loops)
      const int z = opaque0();
      const KArgs a = kargs(z);
      const uint32_t lds0 = lds0k + (unsigned)z;
      const int wv = wvk + z;
      const int lane = (int)__builtin_amdgcn_mbcnt_lo(~0u, (unsigned)z);
      const int qb = cur.qb + z, hh = cur.hh + z, bb = cur.bb + z;
      const int Sq = a.Sq, Skv = a.Skv;
      const int causal_off = Skv - Sq;
      const int kv_len = kv_len_of(a, qb);
      n = (kv_len + NB - 1) / NB;

      const float c2 = a.scale * LOG2E;
      const float fz = __builtin_bit_cast(float, z);  // +0.0, opaque (a constant would be hoisted out of the loop)
      static_for<QT>([&](auto i) {
        m[i] = BIG_NEG;
        d[i] = fz;
        static_for<NDT>([&](auto dt) { o[i][dt] = f32x8{fz, fz, fz, fz, fz, fz, fz, fz}; });
      });
      const unsigned k_lane = (unsigned)((lane & 15) * XK + (lane >> 4) * 16);
      const unsigned v_lane = (unsigned)((((lane >> 4) * 8) + (lane & 7)) * XV + ((lane >> 3) & 1) * 16);
      unsigned kc = lds0 + (unsigned)(K_SLOT_OFF + par * SLOT) + k_lane;
      unsigned kn = lds0 + (unsigned)(K_SLOT_OFF + (par ^ 1) * SLOT) + k_lane;
      unsigned vc = lds0 + (unsigned)(V_SLOT_OFF + par * SLOT) + v_lane;
      unsigned vn = lds0 + (unsigned)(V_SLOT_OFF + (par ^ 1) * SLOT) + v_lane;
      const int khalf = lane >> 4;
      int kv_last[QT];  // the last key each lane's row of q tile i attends
      static_for<QT>([&](auto i) { kv_last[i] = min(qrow(qb, wv, i) + (lane & 15) + causal_off, Skv - 1); });

      auto tile = [&](int t, auto masked) __attribute__((always_inline)) {
        constexpr bool MASKED = decltype(masked)::value;
        const int nxt = t + 1;
        if (t != 0) bar_wait();
        bf16x8 kv8[NKV * NKS * 2];
        {
          const lds_u8* kp = lds_ptr(kc);
          static_for<NKS>([&](auto ks) {
            static_for<NKV>([&](auto kv) {
              static_for<2>([&](auto h) {
                kv8[(kv * NKS + ks) * 2 + h] = lds_load_b128(kp, kv * 16 * XK + (ks * 32 + h * 16) * 2);
              });
            });
          });
        }
        if constexpr (COPIER) {
          if (nxt < n) copy_rows(a, lds0, wv, hh, bb, kv_len, nxt, (par + nxt) & 1);
        }
        sched_barrier<0>();
        // S^T = K Q^T, k-step major in the order of the K reads, fenced so an LDS wait releases at most one k-step
        f32x8 sacc[QT][NKV];
        static_for<NKS>([&](auto ks) {
          if constexpr (ks > 0) sched_barrier<0>();
          static_for<NKV>([&](auto kv) {
            if constexpr (kv > 0) sched_barrier<0>();
            constexpr int j = (kv * NKS + ks) * 2;
            const bf16x16 kfr = cat(kv8[j], kv8[j + 1]);
            static_for<QT>([&](auto i) {
              if constexpr (ks == 0) {
                sacc[i][kv] = wmma(kfr, qf[i][ks], zero8());
              } else {
                sacc[i][kv] = wmma(kfr, qf[i][ks], sacc[i][kv]);
              }
            });
          });
        });
        sched_barrier<0>();
        bf16x8 v8[NDT * 2 * 2];
        {
          const lds_u8* vp = lds_ptr(vc);
          static_for<NDT>([&](auto dt) {
            static_for<2>([&](auto kt) {
              static_for<2>([&](auto h) {
                v8[(dt * 2 + kt) * 2 + h] = lds_tr16(vp, (kt * 32 + h * 16) * XV + dt * 32);
              });
            });
          });
        }
        sched_barrier<0>();
        // lane l holds query l % 16 of the q tile and kv rows (l / 16) * 8 + [0, 8) of each 16-row kv sub-tile
        float x[QT][32];
        auto scores = [&](auto i) __attribute__((always_inline)) {
          static_for<NKV>([&](auto kv) {
            static_for<8>([&](auto e) {
              float sv = sacc[i][kv][(int)e];
              if constexpr (MASKED) {
                const int kv_pos = t * NB + khalf * 8 + (kv * 16 + e);
                sv = kv_pos > kv_last[i] ? -__builtin_inff() : sv;
              }
              x[i][kv * 8 + e] = sv;
            });
          });
        };
        float mn[QT], corr[QT];
        bool resc[QT];
        auto rowmax = [&](auto i) __attribute__((always_inline)) {
          const float mx = tree_max<32>(x[i]);
          const float lm = __builtin_fmaxf(mx, peer(mx));
          const float need = __builtin_fmaf(lm, c2, -m[i]);
          resc[i] = __builtin_amdgcn_ballot_w32(need > LOG2E) != 0u;
          const float mfull = __builtin_fmaxf(m[i], lm * c2);
          mn[i] = resc[i] ? mfull : m[i];
          corr[i] = exp2_raw(m[i] - mn[i]);
        };
        bf16x8 pb[QT][NKV];
        const f32x2 c2v = {c2, c2};
        // q tile i's probabilities: bf16 for the PV WMMAs, fp32 into pp for the row sums
        auto probs = [&](auto i, auto& pp) __attribute__((always_inline)) {
          const f32x2 n2 = {-mn[i], -mn[i]};
          static_for<NKV>([&](auto kv) {
            f32x8 pv;
            static_for<4>([&](auto j) {
              constexpr int e = kv * 8 + 2 * j;
              const f32x2 sv = {x[i][e], x[i][e + 1]};
              const f32x2 av = __builtin_elementwise_fma(sv, c2v, n2);
              const float p0 = exp2_raw(av[0]), p1 = exp2_raw(av[1]);
              pv[2 * j] = p0;
              pv[2 * j + 1] = p1;
              pp[e][(int)i] = p0;
              pp[e + 1][(int)i] = p1;
            });
            pb[i][kv] = to_bf16x8(pv);
          });
        };
        auto rescale = [&](auto i) __attribute__((always_inline)) {
          if (resc[i]) {
            static_for<NDT>([&](auto dt) { o[i][dt] = o[i][dt] * corr[i]; });
          }
        };
        // q tile i's O^T += V^T P^T, 16 WMMAs
        auto pv = [&](auto i) __attribute__((always_inline)) {
          auto one = [&](auto dt, auto kt) __attribute__((always_inline)) {
            const bf16x16 vfr = cat(v8[(dt * 2 + kt) * 2], v8[(dt * 2 + kt) * 2 + 1]);
            const bf16x16 pfr = cat(pb[i][2 * kt], pb[i][2 * kt + 1]);
            o[i][dt] = wmma(vfr, pfr, o[i][dt]);
          };
          if constexpr (PV_KV_MAJOR) {
            static_for<2>([&](auto kt) { static_for<NDT>([&](auto dt) { one(dt, kt); }); });
          } else {
            static_for<NDT>([&](auto dt) { static_for<2>([&](auto kt) { one(dt, kt); }); });
          }
        };
        // q tile 0's softmax; its PV with q tile 1's softmax (independent: the scheduler interleaves them); q tile 1's
        // PV. Both rows' sums by one tree of (q tile 0, q tile 1) pairs.
        f32x2 pp[32];
        scores(ic<0>{});
        rowmax(ic<0>{});
        probs(ic<0>{}, pp);
        rescale(ic<0>{});
        if constexpr (HI) {
          drain_and_arrive();  // this wave's K / V reads are done (it copies nothing)
        } else {
          wait_ds0();
        }
        sched_barrier<0>();
        pv(ic<0>{});
        scores(ic<1>{});
        rowmax(ic<1>{});
        probs(ic<1>{}, pp);
        sched_barrier<0>();
        const f32x2 ls = tree_sum<f32x2, 32>(pp);
        static_for<QT>([&](auto i) {
          d[i] = __builtin_fmaf(corr[i], d[i], ls[(int)i] + peer(ls[(int)i]));
          m[i] = mn[i];
        });
        if constexpr (!HI) drain_and_arrive();  // also waits for this wave's TDM copy of the next tile
        rescale(ic<1>{});
        pv(ic<1>{});
        unsigned tmp = kc;
        kc = kn;
        kn = tmp;
        tmp = vc;
        vc = vn;
        vn = tmp;
      };

      // tiles [0, clean) need no mask for any of the wave's rows, [clean, wave_end) are masked (its diagonal, the kv
      // tail), from wave_end on its rows attend no key
      const int row0 = qrow(qb, wv, 0);
      const int clean = max(min((max(row0 + causal_off, 0) + 1) / NB, n - 1), 0);
      const int wave_q_max = max(min(row0 + 16 * QT - 1, Sq - 1) + causal_off, -1);
      const int wave_end = max(min((wave_q_max + NB) / NB, n), clean);
      int t = 0;
      for (; t < clean; ++t) tile(t, std::false_type{});
      for (; t < wave_end; ++t) tile(t, std::true_type{});
      // sync-only tiles: the barrier and the copies; a low wave stages the next block in the last one, before its
      // arrive (which waits for the copies)
      for (; t < n; ++t) {
        if (t != 0) bar_wait();
        if constexpr (COPIER) {
          if (t + 1 < n) {
            copy_rows(a, lds0, wv, hh, bb, kv_len, t + 1, (par + t + 1) & 1);
          } else {
            const int z2 = opaque0();
            const KArgs a2 = kargs(z2);
            Blk nb;
            if (next_blk(a2, cur, nb)) stage(a2, lds0k + (unsigned)z2, wvk + z2, nb, (par + n) & 1);
          }
        }
        drain_and_arrive();
      }
      // every wave's last tile arrived: the next block's Q and tile 0 landed and visible, unless a low wave needed the
      // last tile itself (bottom-right causal with Sq < Skv, partial blocks; the persistent grid has neither): it
      // stages now and every wave waits for it at one more barrier (wave 3 has the low waves' highest rows: a uniform
      // decision)
      bar_wait();
      const int r3 = qrow(qb, 3, QT - 1);
      const int end3 = min((max(min(r3 + 15, Sq - 1) + causal_off, -1) + NB) / NB, n);
      if (end3 >= n) {
        if constexpr (COPIER) {
          if (wave_end >= n) {
            const int z2 = opaque0();
            const KArgs a2 = kargs(z2);
            Blk nb;
            if (next_blk(a2, cur, nb)) stage(a2, lds0k + (unsigned)z2, wvk + z2, nb, (par + n) & 1);
          }
        }
        drain_and_arrive();
        bar_wait();
      }
    }

    // ---- the next block's Q fragments, then this block's output -----------------------------------------------------
    bool nx;
    Blk nxt_blk;
    {
      const int z = opaque0();
      const KArgs a = kargs(z);
      const uint32_t lds0 = lds0k + (unsigned)z;
      const int wv = wvk + z;
      const int lane = (int)__builtin_amdgcn_mbcnt_lo(~0u, (unsigned)z);
      const int qb = cur.qb + z, hh = cur.hh + z, bb = cur.bb + z;
      const int Sq = a.Sq, H = a.H;
      nx = next_blk(a, cur, nxt_blk);
      read_q(lds0, wv, lane, qf);  // (harmless past the last block)
      // the previous block's O stores have read this wave's O region
      __builtin_amdgcn_s_wait_asynccnt(0);
      const unsigned region =
          HI ? lds0 + (unsigned)(O_HI_OFF + (wv - NW / 2) * OWB) : lds0 + (unsigned)(O_LO_OFF + wv * OWB);
      u32x4 ob[QT][NDT];
      static_for<QT>([&](auto i) {
        const float inv = d[i] > 0.f ? 1.0f / d[i] : 0.f;
        static_for<NDT>([&](auto dt) { ob[i][dt] = __builtin_bit_cast(u32x4, to_bf16x8(o[i][dt] * inv)); });
      });
      {
        const uint32_t wp = region + (unsigned)((lane & 15) * XO + (lane >> 4) * 16);
        static_for<QT>([&](auto i) {
          static_for<NDT>([&](auto dt) { lds_store_b128(wp, i * 16 * XO + dt * 32, ob[i][dt]); });
        });
      }
      wait_ds0();
      // q tile i's 16 rows (Sq % 16 == 0: valid or past Sq as a whole): store k (0..7) copies rows 2k, 2k + 1, lane l
      // its 16-byte chunk l % 16 of row 2k + l / 16; global address = the row pair's (uniform) + the lane's offset
      const unsigned rowb = (unsigned)H * (D_V * 2);
      const unsigned goff = (unsigned)(lane >> 4) * rowb + (unsigned)(lane & 15) * 16u;
      const unsigned loff = region + (unsigned)((lane >> 4) * XO + (lane & 15) * 16);
      static_for<QT>([&](auto i) {
        const int r0 = qrow(qb, wv, i);
        if (r0 < Sq) {
          const unsigned long long g0 =
              (unsigned long long)(uintptr_t)a.O + ((((unsigned long long)bb * Sq + r0) * H + hh) * D_V) * 2ull;
          static_for<8>([&](auto k) {
            store_async_from_lds(g0 + (unsigned long long)(2 * k) * rowb + goff,
                                 loff + (unsigned)((i * 16 + 2 * k) * XO));
          });
        }
      });
      // lse by the lanes holding kv half 0; a masked lane stores at byte offset 0x7FFFFFFF, past the bound: dropped
      const bool khalf0 = (lane >> 4) == 0;
      const long long lse_bytes = ((long long)bb + 1) * H * Sq * 4;
      const __amdgpu_buffer_rsrc_t r_l = rsrc(a.LSE, (int)min(lse_bytes, 0x7FFFFFFFll));
      static_for<QT>([&](auto i) {
        const float lse = (m[i] + __builtin_log2f(d[i])) * LN2;
        const int seq = qrow(qb, wv, i) + (lane & 15);
        const bool mask = khalf0 && seq < Sq;
        const int off = ((bb * H + hh) * Sq + seq) * 4;
        __builtin_amdgcn_raw_buffer_store_b32(__builtin_bit_cast(unsigned, lse), r_l, mask ? off : 0x7FFFFFFF, 0,
                                              0);
      });
    }
    if (!nx) break;
    cur = nxt_blk;
    par = (par + n) & 1;
  }
}
}  // namespace

// grid (npw, 1, 1) persistent or (nqb, H, B), block 256 (8 waves, two per SIMD: 512 VGPRs each); the whole LDS
extern "C" __global__ __attribute__((amdgpu_flat_work_group_size(256, 256), amdgpu_waves_per_eu(2, 2)))
void k_mla_fwd(const bf16* __restrict__ Q, const bf16* __restrict__ K, const bf16* __restrict__ V,
               bf16* __restrict__ O, float* __restrict__ LSE, float scale, int Sq, int Skv, int H, int B, int nqb,
               int npw, int pstep) {
  __shared__ __attribute__((aligned(1024))) unsigned char smem[LDS_B];
  // the arguments are read from the kernarg segment where used (kargs)
  (void)Q, (void)K, (void)V, (void)O, (void)LSE, (void)scale, (void)Sq, (void)Skv, (void)H, (void)B, (void)nqb,
      (void)npw, (void)pstep;
  const uint32_t lds0 = lds_addr(smem);
  const int wv = wave_id((int)__builtin_amdgcn_workitem_id_x());
  if (wv < NW / 2) {
    run<false>(lds0, wv);
  } else {
    run<true>(lds0, wv);
  }
}
