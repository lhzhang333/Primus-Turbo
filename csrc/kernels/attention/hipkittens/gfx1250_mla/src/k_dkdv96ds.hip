/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// k_dkdv96ds -- dK and dV of the DeepSeek-V3 MLA attention backward on gfx1250 (qk 192 / v 128, bf16, bottom-right
// causal: query q sees kv <= q + cshift, cshift = Skv - Sq), plus the dS scratch: every bf16 dS^T tile the kernel
// feeds its dK WMMA is also stored into DS (dslayout.h), where k_dqds_c reads it back for dQ.
//   Q [B, Sq, Hq, 192], dO [B, Sq, Hq, 128], K / dK [B, Skv, Hkv, 192], V / dV [B, Skv, Hkv, 128] bf16; NL, ND
//   [B, Hq, Sq] fp32 (k_delta's row constants); G = Hq / Hkv q heads per kv head (the generated loop assumes 1).
//
// grid (HG, k3_nblk(Skv), B*Hkv/HG), block 64 = 2 waves (one per SIMD). A work item is one kv block of one kv head:
// blocks [0, n96) are 96-row blocks (wave w owns the 48 kv rows from 96 bid + 48 w: NKV 3 sub-tiles of 16), the 0..2
// blocks after them 64-row tail blocks (NKV 2): two bodies behind a uniform branch on the block index (built with
// -structurizecfg-skip-uniform-regions, so they stay separate regions). Ascending blocks = longest first.
//
// A workgroup runs the query tiles (32 queries) of its block: first the n_mask tiles on the diagonal (causal mask on),
// then the n_full tiles below it. The Q / dO tiles go through a 3-stage LDS ring filled by the TDM (a stage =
// [32][D_V] dO + [32][D_QK] Q, each wave moving its 16 rows); K / V of the wave's rows stay in registers, dV / dK
// accumulate in registers and leave through per-wave row-major images in LDS (whole-line global stores). Tile j lives
// in stage j mod 3; the TDM of tile j + 2 issues at the top of tile j (every prefetch clamped to the last tile); one
// barrier per tile, after the last LDS read of its stage: every read of a stage precedes the barrier and the TDM into
// it follows it (WAR), and each wave's half of the next tile has landed before the wave signals it (RAW).
//
// Marker shell: LLVM compiles the work-item fetch, the descriptors, the first two ring tiles, the barriers around the
// tile loop and the epilogue's addressing. The per-tile work stands at the inline-asm markers (; HKGEN INIT / LOOP /
// FINAL nkv=...): tools/bwd.py generates it (the numerics are there) and tools/hkgen.py splices it into this shell's
// .s. The generated code writes only VGPRs above every VGPR the shell references between INIT and FINAL, the
// clobbered SGPRs, vcc and scc. The code object's sha256 pins the shell's instructions too: so does every unused value
// below that is noted as kept.
#include "dkdv_prims.h"
#include "dkdv_idx.h"

using namespace hk1;

static_assert(DSL_VERSION == 2 && DSL_SCALED == 1, "k_dkdv96ds writes dslayout v2 with the scale inside dS");

// one wave's K / V staging area of the marker shell, after the ring: 16 * K3_NKV rows of V (K1_XV), then of K (K1_XK)
#define K3_KV_W (16 * K3_NKV * (K1_XV + K1_XK))
// SGPRs the generated code may use as temporaries (and vcc / scc): declared clobbered at every marker
#define HKGEN_CLOBBERS "memory", "vcc", "scc", "s90", "s91", "s92", "s93", "s94", "s95", "s96", "s97", "s98", \
                       "s99", "s100", "s101", "s102", "s103"
// the LOOP region keeps its loop state, TDM descriptors and per-tile scalars in s48..s103
#define HKGEN_LOOP_CLOBBERS HKGEN_CLOBBERS, "s48", "s49", "s50", "s51", "s52", "s53", "s54", "s55", "s56", "s57", \
    "s58", "s59", "s60", "s61", "s62", "s63", "s64", "s65", "s66", "s67", "s68", "s69", "s70", "s71", "s72", "s73",  \
    "s74", "s75", "s76", "s77", "s78", "s79", "s80", "s81", "s82", "s83", "s84", "s85", "s86", "s87", "s88", "s89"

extern "C" __global__ __launch_bounds__(64) void k_dkdv96ds(
    const bf16* __restrict__ Q, const bf16* __restrict__ K, const bf16* __restrict__ V,
    const bf16* __restrict__ DO, const float* __restrict__ NL, const float* __restrict__ ND,
    bf16* __restrict__ DV, bf16* __restrict__ DK, bf16* __restrict__ DS,
    float scale, int Sq, int Skv, int Hq, int Hkv, int G, int nqt, int cshift, int causal, int B, int HG,
    int ds_mode) {
  __shared__ __attribute__((aligned(1024))) unsigned char smem[K1_LDS + K1_NW * K3_KV_W];
  lock_simd();                           // SCHED_MODE.DISABLE_XDL_ARB_STALL
  const uint32_t lds0 = lds_addr(smem);
  const int tid = (int)threadIdx.x;
  const int lane = tid % 32;
  const int wv = wave_id(tid);
  const int half = lane / 16;
  const int row = lane % 16;
  // ---- marker shell: work item li = (bz * nblk + by) * HG + bx (the grid's order: per bz the longest kv blocks
  // first) from a ticket counter (an int 64 B before DS, in the scratch's front guard, zeroed by the host before each
  // launch): the workgroups take the items in the order they start. The dispatcher deals the workgroups to the 8
  // XCDs round robin, a fixed number each; on a slower XCD they start later and take the later, shorter items instead
  // of that XCD finishing last. Which workgroup runs an item does not change what it computes.
  __shared__ int s_item[1];
  if (tid == 0)
    s_item[0] = __hip_atomic_fetch_add(reinterpret_cast<int*>(reinterpret_cast<unsigned char*>(DS) - 64), 1,
                                       __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  lds_barrier();
  const int k3_item = __builtin_amdgcn_readfirstlane(s_item[0]);
#define K3_BX (k3_item % HG)
#define K3_BY (k3_item / HG % k3_nblk(Skv))
#define K3_BZ (k3_item / (HG * k3_nblk(Skv)))
  const int bat = k1_bat(K3_BZ, K3_BX, HG, Hkv);
  const int hkv = k1_hkv(K3_BZ, K3_BX, HG, Hkv);
  const int bid = K3_BY;
  const int n96 = k3_n96(Skv);

  // true byte extents on every buffer descriptor: an access past them is dropped (a load returns 0)
  const int nk_b = k1_nk_b(B, Skv, Hkv), nv_b = k1_nv_b(B, Skv, Hkv), nl_b = k1_nl_b(B, Hq, Sq);
  // r_k / r_v are unused, kept (see the top)
  const auto r_k = rsrc(K, nk_b), r_v = rsrc(V, nv_b), r_nl = rsrc(NL, nl_b), r_nd = rsrc(ND, nl_b);
  const int kbase = k1_kbase(bat, Skv, Hkv, hkv), vbase = k1_vbase(bat, Skv, Hkv, hkv);
  const float c1 = scale * K1_LOG2E;     // softmax exp2 multiplier
  const int nqt2 = nqt / 2;              // q pairs (nqt: 16-row query tiles)

  // ---- Q / dO ring TDM descriptors: group 1 per op (loop-invariant), per-op LDS / global bases
  i32x8 g1[3];
  uint32_t ldsb[3];
  uint64_t gb[3];
  int64_t gstep_q[3], gstep_g[3];
  static_for<3>([&](auto op) {
    const int w = k1_tdm_w(op);
    const int64_t st = (int64_t)k1_tdm_stride(Hq, op);
    const int ext = k1_tdm_ext();
    g1[op] = i32x8{(int)k1_tdm_w0(op), (int)0xFFFF0000u, (int)(((uint32_t)ext & 0xFFFFu) << 16 | 0x7FFFu),
                   (int)(((uint32_t)ext >> 16) & 0xFFFFu | ((uint32_t)w << 16)), K1_TDM_ROWS,
                   (int)(uint32_t)st, (int)((uint32_t)((uint64_t)st >> 32) & 0xFFFFu), 0};
    ldsb[op] = lds0 + (uint32_t)k1_tdm_lds(0, op, wv);
    const uint64_t base = (uint64_t)(uintptr_t)(op == 0 ? DO : Q);
    gb[op] = base + 2u * (uint64_t)k1_tdm_gelem(bat, Sq, 0, Hq, k1_qh(hkv, G, 0), op, wv);
    gstep_q[op] = 2 * (k1_tdm_gelem(bat, Sq, 32, Hq, k1_qh(hkv, G, 0), op, wv)
                       - k1_tdm_gelem(bat, Sq, 0, Hq, k1_qh(hkv, G, 0), op, wv));
    gstep_g[op] = 2 * (k1_tdm_gelem(bat, Sq, 0, Hq, k1_qh(hkv, G, 1), op, wv)
                       - k1_tdm_gelem(bat, Sq, 0, Hq, k1_qh(hkv, G, 0), op, wv));
  });
  // the TDM of tile (query pair qt, q head gh) into the stage at byte offset stage_off: this wave's 16 rows.
  // tdm_prep = the scalar math (group 0: LDS address, 64-bit global address), tdm_issue = the 3 ops
  struct TdmG0 { i32x4 g0[3]; };
  auto tdm_prep = [&](int qt, int gh, int stage_off) {
    TdmG0 t;
    static_for<3>([&](auto op) {
      const uint64_t ga = gb[op] + (uint64_t)((int64_t)qt * gstep_q[op] + (int64_t)gh * gstep_g[op]);
      t.g0[op] = i32x4{1, (int)(ldsb[op] + (uint32_t)stage_off), (int)(uint32_t)ga,
                       (int)((uint32_t)(ga >> 32) | 0x80000000u)};
    });
    return t;
  };
  auto tdm_issue = [&](const TdmG0& t) { static_for<3>([&](auto op) { tdm_load(t.g0[op], g1[op]); }); };
  auto tdm_tile = [&](int qt, int gh, int stage_off) { tdm_issue(tdm_prep(qt, gh, stage_off)); };

  // ---- NL / ND: the 8 vec4 loads of (query pair qt, q head gh); voffset 32*half + constant, soffset uniform
  const int voff_l = k1_nl_voff(lane);

  // ======================= one body per block height: NKV 3 (96-row blocks) or NKV 2 (64-row tail blocks) =======
  auto body = [&](auto nkv_c) __attribute__((always_inline)) {
    constexpr int NKV = decltype(nkv_c)::value;
    constexpr bool K3 = NKV == K3_NKV;
    constexpr int ROWS = 32 * NKV;
    const int kv0g = k3_kv0g(bid, n96);
    const int kv0 = k3_kv0(kv0g, wv, NKV);

    // an XCNT drain before the export's scalar math, 64-row tail body only (once per tail workgroup); no builtin
    // exists for s_wait_xcnt, the fence keeps the math below the wait
    if constexpr (!K3) {
      asm volatile("s_wait_xcnt 0x0" ::: "memory");
      sched_barrier<0>();
    }

    // ---- dS^T export (dslayout.h): per-wave descriptor (base: region bh0 = bat*Hq + hkv*G, row kt0 = kv0 / 16;
    // num_records: the wave's true extent, 0 for ds_mode 0), row constants, voffset
    const int kt0 = k3ds_kt0(kv0);
    const int64_t ds_base = k3ds_base(bat, Hq, hkv, G, kt0, Sq, Skv, causal);
    const auto r_ds = rsrc64((const unsigned char*)DS + ds_base, k3ds_nrec(ds_mode, G, kt0, NKV, Sq, Skv, causal));
    int ds_rc[NKV];
    static_for<NKV>([&](auto kh) { ds_rc[kh] = k3ds_rowc(kh, kt0, Sq, Skv, causal); });
    const int vds = k1ds_voff(lane);

    // ---- loop bounds: causal tile skip at workgroup level (block id only, both waves equal)
    const int qp_start = k1_qp_start(kv0g, cshift, causal);
    const int nmaskp = k3_nmaskp(kv0g, ROWS, cshift, causal, nqt);
    const int nqp_eff = k1_nqp_eff(kv0g, cshift, causal, nqt);
    const int n_mask = G * nmaskp;
    const int n_full = G * (nqp_eff - nmaskp);

    // ---- marker shell: the per-tile work as generated regions at inline-asm markers (tools/bwd.py; spliced into
    // this shell's .s at build time). The generated loop assumes G == 1 (the host admits MHA only): tile k is
    // q pair qp_start + k, in stage k mod 3. Marker contract: the generated code
    // writes only VGPRs above every VGPR this shell references between INIT and FINAL, the clobbered SGPRs, vcc and
    // scc; one workgroup barrier per tile.
    //   INIT  (c1, c1), (1 / scale, 1 / scale); the wave's K / V rows by TDM into its staging area, accumulators = 0,
    //         then the K / V fragments from the staging area
    //   LOOP  tile 0's hh0 operands and NL/ND, then every tile k = 0 .. N-1 (the first n_mask diagonal: causal mask
    //         on, the dS^T stores of unstored blocks dropped): TDM of tile k+2; S / dP, softmax, dV / dK; the dS^T
    //         stores of tile k; NL/ND of tile k+1; tensorcnt 3 + barrier after the last read of stage k; hh0
    //         operands of tile k+1
    //   FINAL dV * (1 / scale), dK -> bf16 images (epi_images' layout), whole-line readback, global stores
    {  // both block heights (NKV 3: 96 rows, NKV 2: the 64-row tail); the markers carry nkv
      const int N = n_mask + n_full;
      auto stg = [&](int t) { return (t % K1_DEPTH) * K1_QDO_B; };
      auto tq = [&](int t) { return k1_clampqt(qp_start + (t < N ? t : N - 1), nqt2); };
      auto nls = [&](int qt) { return uniform(k1_nl_soff(bat, Hq, k1_qh(hkv, G, 0), Sq, qt * 32)); };
      const uint32_t rdq_l = lds0 + (uint32_t)k1_rd_lane(1, lane), rdv_l = lds0 + (uint32_t)k1_rd_lane(0, lane);
      const uint32_t trv_l = lds0 + (uint32_t)k1_tr_lane(0, lane), trk_l = lds0 + (uint32_t)k1_tr_lane(1, lane);
      (void)kbase, (void)vbase;  // only the FINAL marker's operands read them: odr-used here, the lambda captures them
      // this wave's K / V staging area (16 * NKV rows of V, then of K); INIT moves the rows there by TDM (ops as a
      // ring stage's: V cols [0, 128), K cols [0, 128), K cols [128, 192)), waits for every TDM op, then reads the
      // fragments (ds_load_b128 at lane part + kh * 16 rows + dt * 64 + u * 32)
      const uint32_t kv_l = lds0 + K1_LDS + (uint32_t)(wv * K3_KV_W);
      const uint64_t kv_kg = (uint64_t)(uintptr_t)K + 2u * (uint64_t)(((int64_t)(bat * Skv + kv0) * Hkv + hkv) * K1_D_QK);
      const uint64_t kv_vg = (uint64_t)(uintptr_t)V + 2u * (uint64_t)(((int64_t)(bat * Skv + kv0) * Hkv + hkv) * K1_D_V);
      // the first two Q / dO tiles, ahead of INIT: their latency overlaps INIT's K / V loads
      if (N > 0) {
        sched_barrier<0>();
        tdm_tile(tq(0), 0, stg(0));
        tdm_tile(tq(1), 0, stg(1));
        sched_barrier<0>();
      }
      asm volatile("; HKGEN INIT nkv=%8 lk=%0 lv=%1 kg=%2 vg=%3 hkv=%4 kvl=%5 c1=%6 sc=%7"
                   ::"v"(kv_l + (uint32_t)(16 * NKV * K1_XV + (lane % 16) * K1_XK + (lane / 16) * 16)),
                   "v"(kv_l + (uint32_t)((lane % 16) * K1_XV + (lane / 16) * 16)), "s"(kv_kg), "s"(kv_vg), "s"(Hkv),
                   "s"(kv_l), "s"(c1), "s"(scale), "i"(NKV)
                   : HKGEN_CLOBBERS);  // its TDM descriptors in s92..s103, one op at a time
      if (N > 0) {
        sched_barrier<0>();
        tensor_wait<K1_TW>();                    // own half of tile 0
        sched_barrier<0>();
        lds_barrier();                           // RAW: both halves of tile 0
        sched_barrier<0>();
        // LOOP: the hh0 operands of tile 0 + its NL/ND, then every tile k = 0 .. N-1 (the first n_mask diagonal) in a
        // generated loop (unrolled by the ring depth: the stage is static per body). Every per-tile scalar is derived
        // inside from these loop-invariant inputs: no shell code runs between the tiles.
        //   tdm: per op o (dO, Q 0..127, Q 128..191): global byte base of tile 0's slice (lo, hi), bytes per tile,
        //        LDS byte of stage 0; then the outer strides (elements) of op 0 / ops 1, 2; N; n_mask
        //   sc:  qp_start, the last query pair, NL/ND soffset of pair 0, thr0 = cshift - kv0, qp_first of kv16 rows
        //        kt0 + kh (dS^T blocks stored iff qt >= it), 0
        //   rc:  the dS^T row constants k3ds_rowc(kh) (soffset of pair qt = 1024 qt + rc[kh]), 0
        const int dl = row - 8 * half;           // the mask's lane part: kv row - q row offset of this lane
        {
          i32x16 tdmv;
          static_for<3>([&](auto op) {
            const uint64_t ga0 = gb[op] + (uint64_t)((int64_t)qp_start * gstep_q[op]);
            tdmv[4 * op + 0] = (int)(uint32_t)ga0;
            tdmv[4 * op + 1] = (int)(uint32_t)(ga0 >> 32);
            tdmv[4 * op + 2] = (int)gstep_q[op];
            tdmv[4 * op + 3] = (int)ldsb[op];
          });
          tdmv[12] = (int)k1_tdm_stride(Hq, 0);
          tdmv[13] = (int)k1_tdm_stride(Hq, 1);
          tdmv[14] = N;
          tdmv[15] = n_mask;
          const i32x8 scv = {qp_start, nqt2 - 1, nls(0), cshift - kv0, k3ds_qpf(0, kt0, Sq, Skv, causal),
                             k3ds_qpf(1, kt0, Sq, Skv, causal), k3ds_qpf(NKV - 1, kt0, Sq, Skv, causal), 0};
          const i32x4 rcv = {ds_rc[0], ds_rc[1], ds_rc[NKV - 1], 0};
          asm volatile("; HKGEN LOOP nkv=%13 lq=%0 lv=%1 ltv=%2 ltk=%3 nlv=%4 vds=%5 dl=%6 tdm=%7 sc=%8 rc=%9 rnl=%10 "
                       "rnd=%11 rds=%12"
                       ::"v"(rdq_l), "v"(rdv_l), "v"(trv_l), "v"(trk_l), "v"(voff_l), "v"(vds), "v"(dl), "s"(tdmv),
                       "s"(scv), "s"(rcv), "s"(r_nl), "s"(r_nd), "s"(r_ds), "i"(NKV)
                       : HKGEN_LOOP_CLOBBERS);
        }
      }
      tensor_wait<0>();                          // no TDM may land after the ring is reused
      sched_barrier<0>();
      lds_barrier();                             // every wave's last ring read before the images overwrite it
      sched_barrier<0>();
      // FINAL: images (lane bases ev / ek), whole-line readback (rv / rk), global stores at byte lane
      // part gv / gk + row base sov / sok + rb * rsv / rsk (4 kv rows) + li * 128
      {
        const int rsv = Hkv * (K1_D_V / 8), rsk = Hkv * (K1_D_QK / 8);
        const auto r_dv = rsrc(DV, nv_b), r_dk = rsrc(DK, nk_b);
        asm volatile("; HKGEN FINAL nkv=%12 ev=%0 ek=%1 rv=%2 rk=%3 gv=%4 gk=%5 sov=%6 sok=%7 rsv=%8 rsk=%9 rdv=%10 "
                     "rdk=%11"
                     ::"v"(lds0 + (uint32_t)k3_epi_wr(NKV, wv, 0, 0, 0, lane)),
                     "v"(lds0 + (uint32_t)k3_epi_wr(NKV, wv, 1, 0, 0, lane)),
                     "v"(lds0 + (uint32_t)k3_epi_rd(NKV, wv, 0, 0, 0, lane)),
                     "v"(lds0 + (uint32_t)k3_epi_rd(NKV, wv, 1, 0, 0, lane)), "v"(16 * ((lane / 8) * rsv + lane % 8)),
                     "v"(16 * ((lane / 8) * rsk + lane % 8)), "s"(16 * (vbase + kv0 * rsv)), "s"(16 * (kbase + kv0 * rsk)),
                     "s"(64 * rsv), "s"(64 * rsk), "s"(r_dv), "s"(r_dk), "i"(NKV)
                     : HKGEN_CLOBBERS);
      }
      return;
    }
  };

  // uniform (SGPR) branch on the block id: 96-row blocks first, then the 64-row tail blocks
  if (bid < n96) {
    body(std::integral_constant<int, K3_NKV>{});
  } else {
    body(std::integral_constant<int, 2>{});
  }
#undef K3_BX
#undef K3_BY
#undef K3_BZ
}
