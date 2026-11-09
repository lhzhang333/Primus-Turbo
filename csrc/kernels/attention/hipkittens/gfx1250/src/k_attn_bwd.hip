/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// Flash-attention backward for gfx1250 (MI455X): head dim 128, bf16, Hq = 4 Hkv, bottom-right causal.
// Three kernels in one code object, launched in this order on one stream:
//
//   k_attn_bwd_prep  LD records {NL = -LSE log2e, delta = rowsum(dO o O)} of 64 query rows; zeroes the
//                    matching 32 KB of the fp32 dQ workspace dQ_acc        grid (Sq/64, Hq, B), block 256
//   k_attn_bwd_main  dK (scaled), dV, and dQ_acc += dS K (fp32 atomics)    grid (Hkv, Skv/128, B), block 128
//   k_attn_bwd_dq    dQ = bf16(scale dQ_acc)                              grid (Sq/32, Hq, B), block 256
//
// Every index comes from bwd_idx.h. Buffer accesses carry true extents, TDM tiles equal the tensor dims,
// global offsets are int64, LDS is static, barrier counts are uniform and TDM is drained before exit.
//
// k_attn_bwd_main, step s (unrolled x4: P4 = s & 3 is compile-time, so every ring slot, dS^T buffer and
// LDS offset is a constant; P = s & 1):
//   T    waves 0..2: TDM op(s+3) -> slot (s+3) & 3 (group 0 built in E2 of step s-1); wave 3: a dummy op
//   G0   S/dP(s) kv tile 0 (16 WMMA)   || dSa(s-1) 8 tr16 (buffer P^1) + dOT(s) 16 tr16
//   G1   S/dP(s) kv tile 1 (16 WMMA)   || softmax kv tile 0
//   C    dQ(s-1) = dSa K (16 WMMA)     || softmax kv tile 1 + QT(s) 16 tr16; s_wait_dscnt 16
//   ST   4 ds_store_b128: dS^T(s) -> buffer P
//   E1   dV^T(s) (16 WMMA)             || 16 of the 32 dQ(s-1) atomics + read-back(s+1) items 0..7
//   SIG  s_wait_tensorcnt 1, s_wait_dscnt NUNC, s_barrier_signal
//   E2   dK^T(s) (16 WMMA)             || read-back(s+1) items 8..23 + the other 16 atomics
//   WAIT s_barrier_wait
// Diagonal steps (s >= nf) branch on the wave kind (wave-uniform): DEAD waves (every pair masked) skip
// S/dP, softmax, dV/dK and the E operands and store a zero dS^T, with their memory ops as bursts; FULL /
// DIAG waves run the full schedule with the causal mask. Built in expert scheduling mode
// (-mllvm -amdgpu-expert-scheduling-mode): the compiler inserts the dependency waits.
#include "bwd_idx.h"
#include "hk_attn/ext_bwd.h"
#include "hk_attn/prims.h"

using namespace hk_attn;

// llvm.fake.use through an asm label: a register use with no instruction and no register-class
// constraint. The register allocator keeps the operand's VGPRs allocated up to this point; it only
// takes effect lexically in the kernel (the inliner drops the fake uses of an inlined callee).
extern "C" __device__ void hka_llvm_fake_use(...) __asm("llvm.fake.use");
#define HOLD_2X4(x)                                                                                                  \
    do {                                                                                                             \
        hka_llvm_fake_use((x)[0][0]);                                                                                \
        hka_llvm_fake_use((x)[0][1]);                                                                                \
        hka_llvm_fake_use((x)[0][2]);                                                                                \
        hka_llvm_fake_use((x)[0][3]);                                                                                \
        hka_llvm_fake_use((x)[1][0]);                                                                                \
        hka_llvm_fake_use((x)[1][1]);                                                                                \
        hka_llvm_fake_use((x)[1][2]);                                                                                \
        hka_llvm_fake_use((x)[1][3]);                                                                                \
    } while (0)
#define HOLD_8(x)                                                                                                    \
    do {                                                                                                             \
        hka_llvm_fake_use((x)[0]);                                                                                   \
        hka_llvm_fake_use((x)[1]);                                                                                   \
        hka_llvm_fake_use((x)[2]);                                                                                   \
        hka_llvm_fake_use((x)[3]);                                                                                   \
        hka_llvm_fake_use((x)[4]);                                                                                   \
        hka_llvm_fake_use((x)[5]);                                                                                   \
        hka_llvm_fake_use((x)[6]);                                                                                   \
        hka_llvm_fake_use((x)[7]);                                                                                   \
    } while (0)
#define HOLD_2X2(x)                                                                                                  \
    do {                                                                                                             \
        hka_llvm_fake_use((x)[0][0]);                                                                                \
        hka_llvm_fake_use((x)[0][1]);                                                                                \
        hka_llvm_fake_use((x)[1][0]);                                                                                \
        hka_llvm_fake_use((x)[1][1]);                                                                                \
    } while (0)

namespace {

constexpr int G  = bwd::G4;
constexpr int XP = bwd::XP, SP = bwd::SP;
using T_QDO = tdm_tile<bwd::QS, bwd::D, XP>;
using T_LD  = tdm_tile<1, 64, bwd::LD_B, 4>;
using T_KV  = tdm_tile<bwd::BKV, bwd::D, XP>;
static_assert(T_QDO::bytes == bwd::QT_B && T_KV::bytes == bwd::KV_B && T_LD::bytes == bwd::LD_B, "TDM footprints");
static_assert(bwd::RING == 4 && bwd::NDS == 2, "slot = k & 3, dS buffer = s & 1");

struct XSet {
    opnd_t qA[2][4], dA[2][4];
    f32x4  ld[2][4];
};

// one scheduling group per WMMA: {1 WMMA, VALU|TRANS, VMEM, DS_READ}; the XV / XA / XD instructions of
// a phase are spread evenly (bwd::share) over its NW groups, the first SKIPV groups take no VALU
template <int NW, int XV, int XA, int XD, int SKIPV = 0> HKA_DEV void pattern() {
    static_for<NW>([&](auto g) {
        sched_group_barrier<sg::WMMA, 1, 0>();
        if constexpr ((int) g >= SKIPV && XV > 0) {
            constexpr int nv = bwd::share(XV, NW - SKIPV, (int) g - SKIPV);
            if constexpr (nv > 0) sched_group_barrier<sg::VALU | sg::TRANS, nv, 0>();
        }
        if constexpr (XA > 0) {
            constexpr int na = bwd::share(XA, NW, g);
            if constexpr (na > 0) sched_group_barrier<sg::VMEM, na, 0>();
        }
        if constexpr (XD > 0) {
            constexpr int nd = bwd::share(XD, NW, g);
            if constexpr (nd > 0) sched_group_barrier<sg::DS_READ, nd, 0>();
        }
    });
}

// opaque copy of an LDS address (no instruction): the per-segment lane bases of the loop are computed once
// and stay live in VGPRs instead of being re-added inside the loop
HKA_DEV const lds_u8 *opaque_lds(const lds_u8 *p) {
    uint32_t a = (uint32_t) (uintptr_t) p;
    asm volatile("" : "+v"(a));
    return lds_ptr(a);
}

} // namespace

// ============================================================================================================
// k_attn_bwd_prep / k_attn_bwd_dq. The int64 (batch, seq) element strides of dO and O (prep) and of dQ
// (convert) follow the shape arguments.
HKA_DEV void prep_body(const bf16 *DO, const bf16 *O, const float *LSE, float *LD, float *DQA, int Sq, int Hq,
                       int B, int64_t do_sb, int64_t do_ss, int64_t o_sb, int64_t o_ss) {
    __shared__ __attribute__((aligned(16))) unsigned char sdelta[bwd::PREP_LDS_B];
    const int      t = (int) __builtin_amdgcn_workitem_id_x();
    const int      bx = (int) blockIdx.x, h = (int) blockIdx.y, b = (int) blockIdx.z;
    const int      q0 = bwd::PREP_ROWS * bx;
    const uint32_t lds0 = lds_addr(sdelta);
    // whole-tensor resources: B Sq Hq D bf16 in either layout
    const rsrc_t   r_do = make_rsrc(DO, bwd::qdo_bytes(B, Sq, Hq)), r_o = make_rsrc(O, bwd::qdo_bytes(B, Sq, Hq));
    const rsrc_t   r_lse = make_rsrc(LSE, (int64_t) B * Hq * Sq * 4), r_ld = make_rsrc(LD, bwd::ld_bytes(B, Hq, Sq));
    const int      c = bwd::prep_chunk(t);
    u32x4          vdo[4], vo[4];
    static_for<4>([&](auto p) {
        // dO and O each in their own layout (row byte < 2^31: host gate)
        const int row = q0 + bwd::prep_row(t, p);
        vdo[p] = buf_load_b128(r_do, (int) bwd::prep_row_byte_s(do_sb, do_ss, b, h, row) + 16 * c, 0);
        vo[p]  = buf_load_b128(r_o, (int) bwd::prep_row_byte_s(o_sb, o_ss, b, h, row) + 16 * c, 0);
    });
    static_for<4>([&](auto p) {
        const bf16x8 a = __builtin_bit_cast(bf16x8, vdo[p]), o = __builtin_bit_cast(bf16x8, vo[p]);
        float        e0 = 0.f, e1 = 0.f;
        static_for<4>([&](auto k) {
            e0 = __builtin_fmaf((float) a[2 * k], (float) o[2 * k], e0);
            e1 = __builtin_fmaf((float) a[2 * k + 1], (float) o[2 * k + 1], e1);
        });
        float v = e0 + e1;
        v       = v + xor16<1>(v);
        v       = v + xor16<2>(v);
        v       = v + xor16<4>(v);
        v       = v + xor16<8>(v);
        if (c == 0) ds_store_b32(lds_ptr(lds0), 4 * bwd::prep_row(t, p), __builtin_bit_cast(uint32_t, v));
    });
    lds_barrier();
    if (t < 16) {
        const u32x4 d4 = ds_load_b128_u32(lds_ptr(lds0), 16 * t);
        const f32x4 l4 = buf_load_f32x4(r_lse, (int) bwd::lse_byte(b, Hq, h, Sq, q0 + 4 * t), 0);
        const f32x4 nl = -(l4 * bwd::LOG2E);
        const int   rec = (int) bwd::ld_rec_byte(b, Hq, h, Sq, q0 + 4 * t);
        buf_store_b128(r_ld, rec, 0, __builtin_bit_cast(u32x4, nl));
        buf_store_b128(r_ld, rec + 16, 0, d4);
    }
    const rsrc_t r_dqa = make_rsrc(DQA, bwd::dqa_bytes(B, Hq, Sq));
    const int    base = (int) bwd::cvt_src_byte(b, Hq, h, bwd::nqp_of(Sq), 2 * bx);
    static_for<8>([&](auto i) { buf_store_b128(r_dqa, base + bwd::prep_zero_off(t, i), 0, zero_u32x4()); });
}

extern "C" __global__ __launch_bounds__(bwd::PREP_THREADS) void k_attn_bwd_prep(
    const bf16 *__restrict__ DO, const bf16 *__restrict__ O, const float *__restrict__ LSE, float *__restrict__ LD,
    float *__restrict__ DQA, int Sq, int Hq, int B, int64_t do_sb, int64_t do_ss, int64_t o_sb, int64_t o_ss) {
    prep_body(DO, O, LSE, LD, DQA, Sq, Hq, B, do_sb, do_ss, o_sb, o_ss);
}

extern "C" __global__ __launch_bounds__(bwd::CVT_THREADS) void k_attn_bwd_dq(const float *__restrict__ DQA,
                                                                           bf16 *__restrict__ DQ, float scale,
                                                                           int Sq, int Hq, int B, int64_t dq_sb,
                                                                           int64_t dq_ss) {
    __shared__ __attribute__((aligned(16))) unsigned char img[bwd::CVT_LDS_B];
    const int      t = (int) __builtin_amdgcn_workitem_id_x();
    const int      qt = (int) blockIdx.x, qh = (int) blockIdx.y, b = (int) blockIdx.z;
    const uint32_t lds0 = lds_addr(img);
    const rsrc_t   r_src = make_rsrc(DQA, bwd::dqa_bytes(B, Hq, Sq)), r_dq = make_rsrc(DQ, bwd::qdo_bytes(B, Sq, Hq));
    const int      src = (int) bwd::cvt_src_byte(b, Hq, qh, bwd::nqp_of(Sq), qt);
    f32x4          x[4];
    static_for<4>([&](auto i) { x[i] = buf_load_f32x4(r_src, src + bwd::cvt_src_off(t, i), 0); });
    static_for<4>([&](auto i) {
        const int   p = bwd::cvt_src_off(t, i) / 4;
        const f32x4 y = x[i] * scale;
        ds_store_b64(lds_ptr(lds0), bwd::cvt_lds(bwd::cvt_q(p), bwd::cvt_d(p)),
                     u32x2{cvt_pk_bf16(y[0], y[1]), cvt_pk_bf16(y[2], y[3])});
    });
    lds_barrier();
    static_for<2>([&](auto i) {
        const int row = bwd::cvt_row(t, i), chunk = bwd::cvt_chunk(t, i);
        // dQ in its own layout (the whole-tensor resource above covers both)
        buf_store_b128(r_dq, (int) bwd::cvt_dst_byte_s(dq_sb, dq_ss, b, qh, qt, row, chunk), 0,
                       ds_load_b128_u32(lds_ptr(lds0), row * bwd::CVT_PITCH + 16 * chunk));
    });
}

// ============================================================================================================
// k_attn_bwd_main. Kernel argument offsets: Q 0, K 8, V 16, DO 24, LD 32, DK 40, DV 48, DQA 56 (pointers),
// scale 64 (f32), Sq 68, Skv 72, Hq 76, Hkv 80, B 84, causal 88 (i32), then the int64 (batch, seq) element
// strides of Q 96, K 112, V 128, DO 144, DK 160, DV 176 (192 B).
extern "C" __global__ __launch_bounds__(bwd::BLOCK) void k_attn_bwd_main(
    const bf16 *__restrict__ Q, const bf16 *__restrict__ K, const bf16 *__restrict__ V,
    const bf16 *__restrict__ DO, const float *__restrict__ LD, bf16 *__restrict__ DK, bf16 *__restrict__ DV,
    float *__restrict__ DQA, float scale, int Sq, int Skv, int Hq, int Hkv, int B, int causal, int64_t q_sb,
    int64_t q_ss, int64_t k_sb, int64_t k_ss, int64_t v_sb, int64_t v_ss, int64_t do_sb, int64_t do_ss,
    int64_t dk_sb, int64_t dk_ss, int64_t dv_sb, int64_t dv_ss) {
    __shared__ __attribute__((aligned(1024))) unsigned char smem[bwd::LDS_BYTES];
    lock_simd();
    (void) B; // host-gated; every extent below is per workgroup
    const uint32_t lds0 = lds_addr(smem);
    const int      l = lane_id(), w = wave_id();
    const bwd::Work wk = bwd::wg_work((int) blockIdx.x, (int) blockIdx.y, (int) blockIdx.z);
    const int      kvw0 = bwd::kvw0_of(wk.blk), nqp = bwd::nqp_of(Sq), cshift = Skv - Sq;
    const int      n = bwd::n_all(kvw0, cshift, causal, nqp, G), nf = bwd::n_full(kvw0, cshift, causal, nqp, G);
    if (n < G) return; // all waves, before any TDM op or barrier; unreachable behind the host gate
    const int   rot = bwd::rot_of(wk.blk, G);
    const float c1 = scale * bwd::LOG2E;

    // ---- buffer resources (true extents: the workgroup's dQ_acc group, its dK / dV slabs) -------------------
    const rsrc_t  r_dqa = make_rsrc((const char *) DQA + bwd::dqa_grp_byte(wk.b, Hq, wk.hkv, G, Sq),
                                    bwd::dqa_grp_bytes(G, Sq));
    // dK / dV slabs, each in its own layout: base = row kvw0 of head hkv, extent = 127 seq strides + one row
    (void) Hkv;
    const rsrc_t  r_dk = make_rsrc((const char *) DK + bwd::kv_tile_byte_s(dk_sb, dk_ss, wk.b, wk.hkv, kvw0),
                                   bwd::epi_bytes_s(dk_ss));
    const rsrc_t  r_dv = make_rsrc((const char *) DV + bwd::kv_tile_byte_s(dv_sb, dv_ss, wk.b, wk.hkv, kvw0),
                                   bwd::epi_bytes_s(dv_ss));

    // ---- TDM roles: wave 0 dO, 1 Q, 2 LD records (one op per step each); wave 3 the K / V images, then per
    // step a 16-B dummy op (K row kvw0 -> its own cell bwd::DUMMY_OFF; constant descriptor, no role branch)
    const int      role = w;
    using T_DUM = tdm_tile<1, bwd::DUMMY_COLS, bwd::DUMMY_B>;
    const bool     dum  = role == bwd::ISS_KV;
    const char    *rten = role == bwd::ISS_DO ? (const char *) DO
                          : role == bwd::ISS_Q ? (const char *) Q
                          : role == bwd::ISS_LD ? (const char *) LD : (const char *) K;
    // the role tensor's strides (dO: wave 0, Q: wave 1; LD and the dummy: 0)
    const int64_t  rsb = role == bwd::ISS_DO ? do_sb : role == bwd::ISS_Q ? q_sb : 0;
    const int64_t  rss = role == bwd::ISS_DO ? do_ss : role == bwd::ISS_Q ? q_ss : 0;
    const uint64_t rbase = (uint64_t) (uintptr_t) rten +
                           (uint64_t) (dum ? bwd::dummy_src_byte_s(k_sb, k_ss, wk.b, wk.hkv, kvw0)
                                           : bwd::role_tile_byte_s(role, rsb, rss, wk.b, Sq, Hq, wk.hkv, G, 0, 0));
    const int64_t  rqstep = dum ? 0 : bwd::role_qstep_s(role, rss, Sq, Hq), rhstep = dum ? 0 : bwd::role_hstep_s(role, rss, Sq, Hq);
    const u32x8    rg1 = dum ? tdm_g1<T_DUM>(0, 1)
                         : role == bwd::ISS_LD ? tdm_g1<T_LD>(bwd::role_stride_el_s(role, rss), bwd::role_rows(role))
                                               : tdm_g1<T_QDO>(bwd::role_stride_el_s(role, rss), bwd::role_rows(role));
    const int      rlds0 = dum ? bwd::DUMMY_OFF : bwd::role_lds0(role), rlstep = dum ? 0 : bwd::role_lds_step(role);
    auto           ring_g0 = [&](int k) { // group 0 of the ring op of step k (tile clamp(k)) -> slot k & 3
        const int qt = bwd::tile_qt(k, n, G, nqp), gh = bwd::tile_gh(k, n, G, rot);
        return tdm_g0(lds0 + (uint32_t) bwd::role_lds_at(rlds0, rlstep, k & 3),
                      rbase + (uint64_t) bwd::role_step_byte(0, rqstep, rhstep, qt, gh));
    };

    // ---- prologue: ops 0..2 (waves 0..2), K / V images (wave 3); dS^T buffer 1 zeroed (dQ(-1) = 0); P0 ------
    if (w < bwd::ISS_KV) {
        tdm_load(ring_g0(0), rg1);
        tdm_load(ring_g0(1), rg1);
        tdm_load(ring_g0(2), rg1);
    } else {
        // K and V each in their own layout: tile base at row kvw0 of head hkv, TDM row stride = the seq stride
        tdm_load(tdm_g0(lds0 + bwd::OFF_K, (uint64_t) (uintptr_t) K + (uint64_t) bwd::kv_tile_byte_s(k_sb, k_ss, wk.b, wk.hkv, kvw0)),
                 tdm_g1<T_KV>(k_ss, bwd::BKV));
        tdm_load(tdm_g0(lds0 + bwd::OFF_V, (uint64_t) (uintptr_t) V + (uint64_t) bwd::kv_tile_byte_s(v_sb, v_ss, wk.b, wk.hkv, kvw0)),
                 tdm_g1<T_KV>(v_ss, bwd::BKV));
    }
    {
        lds_u8 *dr = rows_base<SP>(lds0 + bwd::lds_ds(1) + bwd::ds_rows_wave(w), l);
        static_for<2>([&](auto kh) {
            static_for<2>([&](auto hh) { ds_store_b128(dr, bwd::ct_imm(SP, 16 * kh, 16 * hh), zero_u32x4()); });
        });
    }
    if (w < bwd::ISS_KV) wait_tensorcnt<bwd::TCNT>(); // op(0), op(1) landed; op(2) flies
    else wait_tensorcnt<0>();
    lds_barrier();

    // ---- resident operands: K / V rows of the wave's kv rows, K^T for its dQ columns ----------------------------
    opnd_t kf[2][4], vf[2][4], kt[bwd::DQ_NJ][4];
    {
        const lds_u8 *kr = rows_base<XP>(lds0 + bwd::OFF_K + bwd::kimg_rows_wave(w), l);
        const lds_u8 *vr = rows_base<XP>(lds0 + bwd::OFF_V + bwd::kimg_rows_wave(w), l);
        const lds_u8 *kc = tr_base<XP>(lds0 + bwd::OFF_K + 128 * (w & 1), l);
        static_for<2>([&](auto kh) {
            static_for<4>([&](auto dt) {
                kf[kh][dt] = load_rows<XP>(kr, 16 * kh, 32 * dt);
                vf[kh][dt] = load_rows<XP>(vr, 16 * kh, 32 * dt);
            });
        });
        static_for<bwd::DQ_NJ>([&](auto j) { static_for<4>([&](auto c) { kt[j][c] = load_tr<XP>(kc, 32 * c, 16 * j); }); });
    }
    acc_t dVt[8][2], dKt[8][2];
    static_for<8>([&](auto dt) {
        static_for<2>([&](auto kh) {
            dVt[dt][kh] = zero_acc();
            dKt[dt][kh] = zero_acc();
        });
    });

    // ---- per-step helpers ---------------------------------------------------------------------------------------------
    // per-segment lane bases (loop-invariant VGPRs); every per-step slot / buffer offset below is a compile-time
    // constant (x4 unroll) that lands in the 16-bit DS offset field
    const lds_u8 *q_rows = opaque_lds(rows_base<XP>(lds0 + bwd::OFF_Q, l)), *q_tr = opaque_lds(tr_base<XP>(lds0 + bwd::OFF_Q, l));
    const lds_u8 *do_rows = opaque_lds(rows_base<XP>(lds0 + bwd::OFF_DO, l)), *do_tr = opaque_lds(tr_base<XP>(lds0 + bwd::OFF_DO, l));
    const lds_u8 *ld_b = opaque_lds(lds_ptr(lds0 + bwd::OFF_LD + bwd::ld_lane(l)));
    // the wave's q half (16 (w >> 1) columns of dS^T = 32 B) folded into the lane base
    const lds_u8 *ds_tr = opaque_lds(tr_base<SP>(lds0 + bwd::OFF_DS + 32 * (w >> 1), l));
    lds_u8       *ds_st = (lds_u8 *) opaque_lds(rows_base<SP>(lds0 + bwd::OFF_DS + bwd::ds_rows_wave(w), l));
    // read-back items [LO, HI) into X from ring slot SLOT (order: qA / dA by dt, then the LD records)
    auto readback = [&](auto lo_c, auto hi_c, auto slot_c, XSet &X) __attribute__((always_inline)) {
        constexpr int LO = decltype(lo_c)::value, HI = decltype(hi_c)::value, SL = decltype(slot_c)::value;
        static_for<bwd::RB_ITEMS>([&](auto it) {
            if constexpr ((int) it >= LO && (int) it < HI) {
                constexpr int kd = bwd::rb_kind(it), hh = bwd::rb_hh(it);
                if constexpr (kd == 0)
                    X.qA[hh][bwd::rb_dt(it)] = load_rows<XP>(q_rows + SL * bwd::QT_B, 16 * hh, 32 * bwd::rb_dt(it));
                else if constexpr (kd == 1)
                    X.dA[hh][bwd::rb_dt(it)] = load_rows<XP>(do_rows + SL * bwd::QT_B, 16 * hh, 32 * bwd::rb_dt(it));
                else
                    X.ld[hh][bwd::rb_k(it)] = __builtin_bit_cast(f32x4, ds_load_b128_u32(ld_b, SL * bwd::LD_B + bwd::ld_imm(hh, bwd::rb_k(it))));
            }
        });
    };
    opnd_t dSa[bwd::DQ_NQH][4]; // A operands of dQ(s - 1), read at the start of step s
    auto   load_dsa = [&](auto buf_c) {
        constexpr int BUF = decltype(buf_c)::value;
        static_for<bwd::DQ_NQH>([&](auto qh) {
            static_for<4>([&](auto c) { dSa[qh][c] = load_tr<SP>(ds_tr + BUF * bwd::DS_B, 32 * c, 16 * qh); });
        });
    };
    const int voff_lane = bwd::dqs_voff(l, w);
    auto      dq_wmma = [&](acc_t (&dq)[2][2]) { // dQ = dS K: 4 chains jj (d tiles 64 (w & 1) + 16 jj) of the wave's 16 q
        static_for<4>([&](auto c) {
            static_for<4>([&](auto jj) {
                constexpr int qh = jj / 2, j = jj % 2;
                if constexpr (c == 0) dq[qh][j] = wmma0(dSa[0][c], kt[jj][c]);
                else dq[qh][j] = wmma(dSa[0][c], kt[jj][c], dq[qh][j]);
            });
        });
    };
    // atomics [LO, HI) of the 32 (order qh, j, i), vo = their step voffset; each its own instruction
    auto atomics = [&](auto lo_c, auto hi_c, const acc_t (&dq)[2][2], int vo) __attribute__((always_inline)) {
        constexpr int LO = decltype(lo_c)::value, HI = decltype(hi_c)::value;
        static_for<bwd::NATOM>([&](auto a) {
            if constexpr ((int) a >= LO && (int) a < HI) {
                constexpr int qh = a / 16, j = (a / 8) % 2, i = a % 8;
                buf_atomic_add_f32(r_dqa, vo + bwd::dqs_imm(2 * qh + j, i), 0, dq[qh][j][i]);
            }
        });
    };

    constexpr std::integral_constant<int, 0> I0{};
    constexpr std::integral_constant<int, 1> I1{};

    // ---- one diagonal step (masked_c 1: the masked full body, 2: the DEAD-wave body; p4_c: s & 3) ----------------
    // X holds the operands of tile s on entry and of tile s + 1 on exit
    u32x4 g0n = ring_g0(bwd::LEAD); // group 0 of op(3), issued at the top of step 0
    auto  step = [&](auto masked_c, auto p4_c, int s, XSet &X) __attribute__((always_inline)) {
        constexpr bool MASKED = decltype(masked_c)::value;
        constexpr int  P4 = decltype(p4_c)::value, P = P4 & 1, NX = (P4 + 1) & 3;
        constexpr bool MDEAD = (int) decltype(masked_c)::value == 2;
#define BWD_KERNEL_SCOPE 0
#include "bwd_step.inc"
#undef BWD_KERNEL_SCOPE
    };

    // ---- main loop: x4 unrolled (n, nf multiples of G = 4), the full steps then the diagonal steps ---------------
    static_assert(G % 4 == 0, "x4 unroll needs n, nf multiples of 4");
    XSet X;
    readback(std::integral_constant<int, 0>{}, std::integral_constant<int, bwd::RB_ITEMS>{}, std::integral_constant<int, 0>{}, X);
    constexpr std::integral_constant<int, 0> Q0{};
    constexpr std::integral_constant<int, 1> Q1{};
    constexpr std::integral_constant<int, 2> Q2{};
    constexpr std::integral_constant<int, 3> Q3{};
    int s = 0;
    // The full steps are expanded at kernel scope (their register holds must be lexically in the kernel).
    // dQ(s - 1), its voffset and E2's operands are loop-carried; the initial values only feed the holds in G0
    // of step 0.
    acc_t dq_c[2][2];
    int   avo_c = 0;
    static_for<2>([&](auto qh) { static_for<2>([&](auto j) { dq_c[qh][j] = zero_acc(); }); });
    opnd_t qt_c[8];
    u32x4  du_c[2][2];
    static_for<8>([&](auto dt) { qt_c[dt] = opnd_t{}; });
    static_for<2>([&](auto hh) { static_for<2>([&](auto kh) { du_c[hh][kh] = zero_u32x4(); }); });
#define BWD_FULL_STEP_SCOPE(P4_, S_)                                                                                 \
    constexpr bool MASKED = false, MDEAD = false;                                                                    \
    constexpr int  P4 = P4_, P = P4 & 1, NX = (P4 + 1) & 3;                                                          \
    const int      s_at = S_;
#define BWD_KERNEL_SCOPE 1
    for (; s < nf; s += 4) {
        { BWD_FULL_STEP_SCOPE(0, s) { const int s = s_at;
#include "bwd_step.inc"
        } }
        { BWD_FULL_STEP_SCOPE(1, s + 1) { const int s = s_at;
#include "bwd_step.inc"
        } }
        { BWD_FULL_STEP_SCOPE(2, s + 2) { const int s = s_at;
#include "bwd_step.inc"
        } }
        { BWD_FULL_STEP_SCOPE(3, s + 3) { const int s = s_at;
#include "bwd_step.inc"
        } }
    }
#undef BWD_KERNEL_SCOPE
#undef BWD_FULL_STEP_SCOPE
    // diagonal steps: per step a wave-uniform kind branch between the DEAD body and the masked full body
    auto mstep = [&](auto p4_c, int sm_) __attribute__((always_inline)) {
        if (bwd::wave_kind(kvw0, w, bwd::tile_qt(sm_, n, G, nqp), cshift, causal) == bwd::K_DEAD)
            step(std::integral_constant<int, 2>{}, p4_c, sm_, X);
        else
            step(std::integral_constant<int, 1>{}, p4_c, sm_, X);
    };
    for (; s < n; s += 4) {
        mstep(Q0, s);
        mstep(Q1, s + 1);
        mstep(Q2, s + 2);
        mstep(Q3, s + 3);
    }

    // ---- drain: dQ(n - 1) (dSa from buffer (n - 1) & 1 = 1) and its atomics; epilogue; TDM drain + barrier -------
    {
        load_dsa(std::integral_constant<int, 1>{});
        acc_t dq[2][2];
        dq_wmma(dq);
        const int vo = voff_nonneg(voff_lane + bwd::dqa_voff(0, 0, bwd::atom_gh(n, n, G, rot), bwd::atom_qt(n, n, G, nqp), nqp));
        static_for<2>([&](auto qh) {
            static_for<2>([&](auto j) { atomic_add_c<512>(r_dqa, vo + bwd::dqs_imm(2 * qh + j, 0), 0, dq[qh][j]); });
        });
    }
    // dK in dK's layout, dV in dV's: per-lane voffsets with the int32 byte row strides
    const int dk_rb = (int) (2 * dk_ss), dv_rb = (int) (2 * dv_ss);
    static_for<2>([&](auto kh) {
        const int vk = voff_nonneg(bwd::epi_voff_kh_s(l, w, kh, dk_rb)), vv = voff_nonneg(bwd::epi_voff_kh_s(l, w, kh, dv_rb));
        static_for<8>([&](auto dt) {
            buf_store_ct_bf16(r_dv, vv + bwd::epi_imm(dt), 0, dVt[dt][kh]);
            buf_store_ct_bf16(r_dk, vk + bwd::epi_imm(dt), 0, scale_acc(dKt[dt][kh], scale));
        });
    });
    split_signal_tdm_lds<0>();
    split_wait_lds();
}
