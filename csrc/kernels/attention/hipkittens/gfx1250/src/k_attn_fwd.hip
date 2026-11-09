/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// k_attn_fwd -- flash-attention forward for gfx1250 (MI455X): head dim 128, bf16, GQA with
// G = Hq / Hkv in {4, 8, 16}, bottom-right causal (query p sees kv <= p + Skv - Sq).
//
//   O[b, p, hq, :] = softmax_kv(scale Q K^T) V (bf16),  LSE[b, hq, p] = ln sum_kv exp(scale s) (fp32)
//
// grid (W, 1, 1), block 128 (4 waves, one per SIMD), static LDS 320 KB (one workgroup per CU). A work
// item is 64 query positions x one head quad; the persistent schedule, the TDM ring and every index
// formula are in fwd_idx.h. Per wave: Q^T resident (16 positions x 4 heads), S^T = K Q^T and
// O^T += V^T P^T in 32-kv chunks with S^T double-buffered by chunk parity, exp2 softmax with a lazy
// rescale (only when a running max grows by more than 8 nat).
//
// Chunk c of the current item (stage T = T0 + c / 2):
//   B   QK(c+1) -> S[(c+1) & 1]  ||  exp of q tiles 0..2, P^T pack of tiles 0..1  (+ the op(T+3)
//       descriptor and the next stage's burst bases on even c)
//   D   even c: WAIT(P_{T+1}) + [Q(j+1)] + op(T+3);  odd c: tensorcnt 1 (2 in the last stage) + SIGNAL
//   E   even c: one LDS burst for the stage: K(c+2), V(c+1), K(c+3), V(c+2)
//   F1  PV(c) q tiles 0, 1  ||  SM1(c+1) (column max, rescale decision), P^T pack of q tile 2
//   F2  PV(c) q tiles 2, 3  ||  exp + pack of q tile 3
//   C   lazy rescale of chunk c+1 (a wave-uniform branch, rarely taken)
// Per item: prologue (Q^T burst, K(0), K(1), V(0), QK(0), SM1(0)), the stages (the steady loop runs two
// stages per iteration; stages n-2 and n-1 carry the causal mask and the next item's ops), epilogue
// (O through the wave's LDS image to global memory, LSE). All branches are workgroup-uniform.
//
// Q, K, V and O are each contiguous BSHD or a BSHD view of SBHD storage, chosen per tensor by the host
// through the element strides that follow the shape arguments.
#include "fwd_idx.h"
#include "hk_attn/ext_fwd.h"
#include "hk_attn/prims.h"

#pragma clang fp contract(off)

#define FWD_INL __attribute__((always_inline))

using namespace hk_attn;

namespace {

constexpr int NH = fwd::NH, NDK = fwd::NDK, NDT = fwd::NDT;
template <int N> using ic = std::integral_constant<int, N>;

static_assert(fwd::BLOCK == 128 && fwd::LDS_BYTES == 327680, "4 waves, one workgroup per CU");

struct Tdm { // one TDM op: groups 0 / 1 (wave-uniform SGPRs)
    u32x4 g0;
    u32x8 g1;
};
HKA_DEV Tdm make_tdm(uint32_t lds, uint64_t ga, const fwd::G1 &g) {
    return Tdm{tdm_g0(lds, ga), tdm_g1_words(g.w)};
}
HKA_DEV void issue(const Tdm &d) {
    tdm_load(d.g0, d.g1);
}

// max over the 16 values of one query in this lane (kv tiles 0 and 1): 7 v_max3 + 1 v_max
HKA_DEV float max16(const acc_t &a, const acc_t &b) {
    const float x0 = max3(a[0], a[1], a[2]), x1 = max3(a[3], a[4], a[5]), x2 = max3(a[6], a[7], b[0]);
    const float x3 = max3(b[1], b[2], b[3]), x4 = max3(b[4], b[5], b[6]);
    return __builtin_fmaxf(max3(x0, x1, x2), max3(x3, x4, b[7]));
}

// SGPR value fence (no instruction): nothing computed from x is placed before this point, which pins
// the op descriptor and the next stage's burst bases into region B.
HKA_DEV void sgpr_fence(int &x) {
    asm volatile("" : "+s"(x));
}

// The kernel arguments as laid out in the kernarg segment. Arguments needed only at item boundaries
// (schedule decode, output pointers) are re-read there through an opaque copy of the segment pointer,
// so they hold no SGPR across the stage loop.
struct KArgs {
    const __bf16 *Q, *K, *V;
    __bf16       *O;
    float        *LSE;
    long long     o_bytes, lse_bytes;
    int           B, Sq, Skv, Hq, Hkv, G, nqb, nhq4, nkv_st, cst, causal;
    float         c2, c2inv;
    int           W, mode;
    uint32_t      mW;   // Fdiv magic numbers of W, max(nqb / 2, 1), nhq4, B nhq4
    int           shW;
    uint32_t      mNP;
    int           shNP;
    uint32_t      mH4;
    int           shH4;
    uint32_t      mNH;
    int           shNH;
    long long     sQb, sQs, sKb, sKs, sVb, sVs, sOb, sOs; // element strides (batch, seq) of Q, K, V, O
};
static_assert(__builtin_offsetof(KArgs, B) == 56 && __builtin_offsetof(KArgs, c2) == 100 &&
                  __builtin_offsetof(KArgs, mode) == 112 && __builtin_offsetof(KArgs, shNH) == 144 &&
                  sizeof(KArgs) == 216,
              "kernarg layout");
static_assert(__builtin_offsetof(KArgs, sQb) == 152 && __builtin_offsetof(KArgs, sKb) == 168 &&
                  __builtin_offsetof(KArgs, sVb) == 184 && __builtin_offsetof(KArgs, sOs) == 208,
              "kernarg layout: 8 int64 strides at 152 .. 215");
typedef const __attribute__((address_space(4))) KArgs *kargs_p;
// the segment pointer, made opaque (no instruction): loads through it are neither merged with the
// entry's argument loads nor hoisted out of the item loop
HKA_DEV kargs_p kargs() {
    kargs_p p = (kargs_p) __builtin_amdgcn_kernarg_segment_ptr();
    asm volatile("" : "+s"(p));
    return p;
}

} // namespace

extern "C" __global__ __launch_bounds__(fwd::BLOCK) void k_attn_fwd(
    const __bf16 *Q, const __bf16 *K, const __bf16 *V, __bf16 *O, float *LSE, long long o_bytes,
    long long lse_bytes, int B, int Sq, int Skv, int Hq, int Hkv, int G, int nqb, int nhq4, int nkv_st,
    int cst, int causal, float c2, float c2inv, int W, int mode, uint32_t mW, int shW, uint32_t mNP, int shNP,
    uint32_t mH4, int shH4, uint32_t mNH, int shNH, long long sQb, long long sQs, long long sKb, long long sKs,
    long long sVb, long long sVs, long long sOb, long long sOs) {
    __shared__ __attribute__((aligned(1024))) unsigned char smem[fwd::LDS_BYTES];
    lock_simd();

    // ---- identity (wave-uniform except the lane) ------------------------------------------------------------
    const int      w       = wave_id();
    const int      l       = lane_id();
    const int      rv      = fwd::op_role_v(w); // 0: this wave moves K halves, 1: V halves (SALU selects)
    const uint32_t lds0    = lds_addr(smem);
    const uint64_t kv_base = (uint64_t) (uintptr_t) (rv != 0 ? V : K);
    // the seq stride of this wave's role tensor: the TDM row stride of its K / V ops, held across the stage
    // loop in one SGPR; the batch strides and the Q / O strides are re-read at item boundaries (lay_*)
    const uint32_t kv_ss = fwd::kv_ss32(rv != 0 ? sVs : sKs);
    const uint32_t sst   = fwd::kv_sst_s(kv_ss); // K / V elements per 64-row stage
    auto lay_kv = [&](kargs_p ka) FWD_INL { return rv != 0 ? fwd::Lay{ka->sVb, ka->sVs} : fwd::Lay{ka->sKb, ka->sKs}; };
    auto lay_q  = [&](kargs_p ka) FWD_INL { return fwd::Lay{ka->sQb, ka->sQs}; };
    auto lay_o  = [&](kargs_p ka) FWD_INL { return fwd::Lay{ka->sOb, ka->sOs}; };
    // schedule / shape parameters re-read from the kernarg segment (SGPR, short-lived)
    auto sched = [&](kargs_p ka) FWD_INL {
        return fwd::Sched{ka->mode, ka->W, ka->B, ka->nhq4, ka->nqb, fwd::Fdiv{ka->mW, ka->shW}, fwd::Fdiv{ka->mNP, ka->shNP},
                          fwd::Fdiv{ka->mH4, ka->shH4}, fwd::Fdiv{ka->mNH, ka->shNH}};
    };
    auto shape = [&](kargs_p ka) FWD_INL {
        return fwd::Shape{ka->B, ka->Sq, Skv, ka->Hq, Hkv, ka->G, ka->nqb, ka->nhq4, ka->nkv_st, ka->cst, causal};
    };
    const int g = (int) __builtin_amdgcn_workgroup_id_x();
    // ---- TDM descriptors (SALU only: both candidates, then scalar selects) -----------------------------------
    // K / V op (compact form fwd::OpC: d1 > 0 real, 0 dummy) at global stage T
    auto op_desc = [&](const fwd::OpC &o, int T) FWD_INL -> Tdm {
        const bool     real = o.d1 > 0;
        const fwd::G1  a = fwd::kv_g1_s(rv, kv_ss, o.d1), d = fwd::dummy_g1(rv);
        fwd::G1        s1{};
        for (int i = 0; i < 8; ++i) s1.w[i] = real ? a.w[i] : d.w[i];
        const uint32_t la_r = (uint32_t) fwd::kv_op_lds(w, T), la_d = (uint32_t) fwd::dummy_off(w);
        return make_tdm(lds0 + (real ? la_r : la_d), kv_base + 2ull * (uint64_t) o.el, s1);
    };
    // Q op (compact): the item's Q tile into the wave's Q region, or a 16-B dummy of Q row 0 into its Q dummy
    // cell
    auto q_desc = [&](const fwd::OpC &o) FWD_INL -> Tdm {
        const bool     real = o.d1 > 0;
        const kargs_p  ka = kargs();
        const fwd::G1  a = fwd::q_g1_s(ka->sQs, o.d1), d = fwd::dummy_g1(0);
        fwd::G1        s1{};
        for (int i = 0; i < 8; ++i) s1.w[i] = real ? a.w[i] : d.w[i];
        const uint32_t la_r = (uint32_t) fwd::q_op_lds(w), la_d = (uint32_t) fwd::q_dummy_off(w);
        return make_tdm(lds0 + (real ? la_r : la_d), (uint64_t) (uintptr_t) ka->Q + 2ull * (uint64_t) o.el, s1);
    };

    // ---- LDS operand bursts (per-lane base VGPR + compile-time DS immediates) --------------------------------
    // burst bases of global stage T (one v_add each; computed a stage ahead, off the DS address path)
    auto kbase = [&](int T) FWD_INL { return lds0 + (uint32_t) (fwd::k_slot_off(T) + fwd::k_lane(l)); };
    auto vbase = [&](int T) FWD_INL { return lds0 + (uint32_t) (fwd::v_slot_off(T) + fwd::v_lane(l)); };
    // K A operands of stage half HC at base kb (consumption order of qk: k outer, kt inner)
    auto ld_k = [&](opnd_t(&kk)[2][NDK], uint32_t kb, auto HC) FWD_INL {
        const lds_u8 *kp = lds_ptr(kb);
        static_for<NDK>([&](auto k) {
            static_for<2>([&](auto kt) {
                kk[kt][k] = cat(ds_load_b128(kp, fwd::k_imm(HC, kt, k, 0)), ds_load_b128(kp, fwd::k_imm(HC, kt, k, 1)));
            });
        });
    };
    // V^T A operands of stage half HC at base vb (tr16 at pitch 288)
    auto ld_v = [&](opnd_t(&vv)[NDT], uint32_t vb, auto HC) FWD_INL {
        const lds_u8 *vp = lds_ptr(vb);
        static_for<NDT>([&](auto dt) {
            vv[dt] = cat(ds_load_tr16_b128(vp, fwd::v_imm(HC, 0, dt)), ds_load_tr16_b128(vp, fwd::v_imm(HC, 1, dt)));
        });
    };

    // ---- state -------------------------------------------------------------------------------------------------
    opnd_t qb[NH][NDK];    // Q^T B operands (resident per item)
    opnd_t kab[2][2][NDK]; // K A operands by chunk parity: QK(c) reads kab[c & 1]; the even E(c) loads K(c+2), K(c+3)
    opnd_t vE[NDT], vO[NDT], vN[NDT]; // V^T A operands of PV(2t), PV(2t+1), PV(2t+2) (vE = vN after the odd chunk)
    opnd_t P[NH];         // P^T B operands
    acc_t  S[2][NH][2];   // S^T of chunks c (exp in place) and c + 1
    acc_t  oacc[NH][NDT]; // O^T
    f32x2  lsum[NH];      // per-lane partial row sums (halves added in the epilogue)
    float  m2[NH];        // running max (log2 domain)
    uint32_t need = 0u;   // ballot of "some q tile's own-half max exceeds its threshold" (wave-uniform)
    float  m2th[NH];      // (m2 + TH2) c2inv: the unscaled own-half max that triggers a rescale
    float  mxh[NH];       // own-half column max of chunk c + 1 (for the rescale block)
    // current item (n, T0, el0 in SGPRs; its b / hq4 / qb are decoded again for the epilogue) and lookahead
    int      n = 1, T0 = 0;
    uint32_t el0 = 0; // K / V element of the item's stage 0 for this wave's op (fwd::kv_el0_s)
    fwd::OpC nx[3];   // ops of stream stages T0 + n .. T0 + n + 2 (the next items' first stages, or dummies)
    fwd::OpC qn{};    // Q op of item j + 1 (issued in item j's last stage; a dummy after the last item)
    uint32_t kb0 = 0, kb1 = 0, vb0 = 0, vb1 = 0, kb2 = 0, vb2 = 0; // burst bases of stages T, T+1 (T+2)

    // items of this workgroup (>= 1 for every grid the host launches: W <= #units). A larger grid gives some
    // workgroups J = 0; they return here, before any TDM op, LDS access or barrier (J is workgroup-uniform)
    const int J = fwd::wg_items(sched(kargs()), g);
    if (J <= 0) return;
    // item j of this workgroup: its stage count and stage-0 K / V element (needed from its prologue on)
    auto set_cur = [&](int j) FWD_INL {
        const kargs_p  ka = kargs();
        const fwd::Work x = fwd::wg_item(sched(ka), g, j);
        n   = fwd::n_stages(x.qb, ka->cst, ka->nkv_st, causal);
        el0 = fwd::kv_el0_s(lay_kv(ka), x.b, fwd::hkv_of(x.hq4, ka->G), w);
    };
    // lookahead of item j (first used at its stage n-3): the ops of the 3 stream stages after it and Q(j+1).
    // Called right before the item's QK(0), in the same scheduling region, so its SALU fills WMMA issue gaps.
    auto set_ahead = [&](int j) FWD_INL {
        const kargs_p    ka = kargs();
        const fwd::Sched sc = sched(ka);
        const fwd::Shape sh = shape(ka);
        fwd::StRef r[3];
        fwd::stream3(sh, sc, g, J, j + 1, r);
        const fwd::Lay lk = lay_kv(ka);
        static_for<3>([&](auto k) { nx[k] = fwd::op_c_s(sh, lk, r[k], w); });
        qn = fwd::q_c_s(j + 1 < J, fwd::wg_item(sc, g, fwd::imin(j + 1, J - 1)), sh.Sq, lay_q(ka), w);
    };

    // ---- phases ----------------------------------------------------------------------------------------------
    auto qk = [&](acc_t(&s)[NH][2], const opnd_t(&kk)[2][NDK]) FWD_INL { // S^T(c') = K(c') Q^T
        static_for<NDK>([&](auto k) {
            static_for<2>([&](auto kt) {
                static_for<NH>([&](auto h) {
                    if constexpr (decltype(k)::value == 0) s[h][kt] = wmma0(kk[kt][0], qb[h][0]);
                    else s[h][kt] = wmma(kk[kt][k], qb[h][k], s[h][kt]);
                });
            });
        });
    };
    auto pv = [&](const opnd_t(&vv)[NDT], auto H) FWD_INL { // O^T += V^T P^T of q tile H
        static_for<NDT>([&](auto dt) { oacc[H][dt] = wmma(vv[dt], P[H], oacc[H][dt]); });
    };
    // SM1 of chunk c1 (S buffer s): [causal mask] -> column max -> rescale decision; FIRST (item prologue):
    // m2 = the chunk's max directly (O and l are 0)
    auto sm1 = [&](acc_t(&s)[NH][2], int c1, auto MASKED, auto FIRST) FWD_INL {
        if constexpr (decltype(MASKED)::value != 0) {
            const int off0 = fwd::mask_off(w, c1, n, 0, causal), off1 = fwd::mask_off(w, c1, n, 1, causal);
            static_for<NH>([&](auto h) {
                s[h][0] = mask_row_gt_col(s[h][0], l, off0);
                s[h][1] = mask_row_gt_col(s[h][1], l, off1);
            });
        }
        static_for<NH>([&](auto h) {
            const float mx = max16(s[h][0], s[h][1]);
            if constexpr (decltype(FIRST)::value != 0) {
                m2[h] = __builtin_fmaxf(mx, xhalf(mx)) * c2;
                m2th[h] = (m2[h] + fwd::TH2) * c2inv;
            } else {
                mxh[h] = mx;
            }
        });
        // one compare for the branch: max_h (mx_h - m2th_h) > 0 <=> some q tile's own-half max exceeds its
        // threshold (fp32 denormals are on, so x - y > 0 <=> x > y for every non-NaN x, y)
        if constexpr (decltype(FIRST)::value == 0) {
            const float d = __builtin_fmaxf(__builtin_fmaxf(mxh[0] - m2th[0], mxh[1] - m2th[1]),
                                            __builtin_fmaxf(mxh[2] - m2th[2], mxh[3] - m2th[3]));
            need          = ballot(d > 0.f);
        }
    };
    // p = exp2(s c2 - m2) in place (8 v_pk_fma + 16 v_exp), then the P^T pack (8 v_cvt_pk_bf16)
    auto tail_exp = [&](acc_t(&s)[NH][2], auto H) FWD_INL {
        constexpr int h  = decltype(H)::value;
        const f32x2   cc = {c2, c2}, nm = {-m2[h], -m2[h]};
        static_for<2>([&](auto kt) {
            static_for<4>([&](auto j) {
                constexpr int jj = decltype(j)::value;
                const f32x2   a  = pk_fma(acc_pair<jj>(s[h][kt]), cc, nm);
                s[h][kt][2 * jj]     = exp2_hw(a[0]);
                s[h][kt][2 * jj + 1] = exp2_hw(a[1]);
            });
        });
    };
    auto tail_pack = [&](acc_t(&s)[NH][2], auto H) FWD_INL {
        constexpr int h = decltype(H)::value;
        P[h]            = pack_pt_fwd(s[h][0], s[h][1]);
    };
    auto rsum = [&](acc_t(&s)[NH][2], auto H) FWD_INL { // 7 v_pk_add tree + 1 accumulate
        constexpr int h = decltype(H)::value;
        const f32x2   a = pk_add(pk_add(acc_pair<0>(s[h][0]), acc_pair<1>(s[h][0])),
                                 pk_add(acc_pair<2>(s[h][0]), acc_pair<3>(s[h][0])));
        const f32x2   b = pk_add(pk_add(acc_pair<0>(s[h][1]), acc_pair<1>(s[h][1])),
                                 pk_add(acc_pair<2>(s[h][1]), acc_pair<3>(s[h][1])));
        lsum[h]         = pk_add(lsum[h], pk_add(a, b));
    };
    auto scale_state = [&](auto H, float corr) FWD_INL {
        constexpr int h = decltype(H)::value;
        static_for<NDT>([&](auto dt) { oacc[h][dt] = oacc[h][dt] * corr; });
        lsum[h] = pk_mul(lsum[h], f32x2{corr, corr});
    };
    // C: lazy rescale (wave-uniform branch, rarely taken, laid out of line); the per-q-tile decisions are
    // recomputed inside the block, and corr = 1 exactly for q tiles whose m2 stays
    auto rescale = [&]() FWD_INL {
        if (__builtin_expect(any_set(need), 0)) {
            static_for<NH>([&](auto h) {
                const uint32_t b = ballot(mxh[h] > m2th[h]);
                const float mo   = m2[h];
                const float mf   = __builtin_fmaxf(mxh[h], xhalf(mxh[h]));
                const float mn   = any_set(b) ? __builtin_fmaxf(mo, mf * c2) : mo;
                const float corr = exp2_hw(mo - mn);
                scale_state(h, corr);
                m2[h] = mn;
                m2th[h] = (mn + fwd::TH2) * c2inv;
            });
        }
    };
    auto zero_state = [&]() FWD_INL {
        static_for<NH>([&](auto h) {
            static_for<NDT>([&](auto dt) { oacc[h][dt] = zero_acc(); });
            lsum[h] = f32x2{0.f, 0.f};
        });
    };

    // ---- one iteration on chunk c = 2 t + PAR of the current item (global stage T = T0 + t) --------------------
    // QKN: QK(c+1) present (false only in the item's last odd chunk); BAR: 1 WAIT(P_{T+1}) + op(T+3),
    // 2 tensorcnt 1 + SIGNAL(P_{T+2}), 3 WAIT + Q(j+1) + op(T+3) (last stage, even), 4 tensorcnt 2 + SIGNAL
    // (last stage, odd); SM: 0 none, 1 SM1(c+1) + C(c+1), 2 the same with the causal mask; OPK (BAR 1 / 3):
    // 0 own stage t+3 or nx[0], 1 nx[1], 2 nx[2]
    auto iter = [&](int t, auto PAR, auto QKN, auto BAR, auto SM, auto OPK) FWD_INL {
        constexpr int  par = decltype(PAR)::value, nxp = 1 - par;
        constexpr bool qkn = decltype(QKN)::value != 0;
        constexpr int  bar = decltype(BAR)::value, sm = decltype(SM)::value, opk = decltype(OPK)::value;
        constexpr bool last = bar == 3 || bar == 4;
        const int      c    = 2 * t + par;
        opnd_t(&va)[NDT] = *(par == 0 ? &vE : &vO);
        // B: QK(c+1) || exp of q tiles 0..2, pack of 0..1 (+ op(T+3)'s descriptor and the next stage's
        // burst bases on even c)
        Tdm dn{}, dq{};
        if constexpr (bar == 1 || bar == 3) {
            int tk = t; // fenced: computed here, in B
            sgpr_fence(tk);
            __builtin_assume(tk >= 0);
            fwd::OpC r = nx[opk];
            if constexpr (opk == 0) {
                const fwd::OpC own = fwd::own_c(Skv, el0, sst, w, tk + 3);
                r                  = tk + 3 < n ? own : nx[0];
            }
            dn = op_desc(r, T0 + tk + 3);
            if constexpr (bar == 3) dq = q_desc(qn);
            kb2 = kbase(T0 + tk + 2);
            vb2 = vbase(T0 + tk + 2);
        }
        if constexpr (qkn) qk(S[nxp], kab[nxp]); // K(c+1): loaded at the even E of the previous stage (front: K(1))
        tail_exp(S[par], ic<0>{});
        tail_pack(S[par], ic<0>{});
        tail_exp(S[par], ic<1>{});
        tail_pack(S[par], ic<1>{});
        tail_exp(S[par], ic<2>{});
        sched_barrier<0>();
        // D: barrier point
        if constexpr (bar == 1 || bar == 3) {
            split_wait_lds(); // WAIT(P_{T+1}): stage T+1 published, stage T-1 free
            if constexpr (bar == 3) issue(dq); // Q(j+1) (or a dummy) before op(T+3)
            issue(dn);                          // op(T+3) into slot (T+3) % 4
            sched_barrier<0>();
        } else if constexpr (bar == 2) {
            split_signal_tdm_lds<fwd::TCNT_STEADY>(); // SIGNAL(P_{T+2}): own op(T+2) landed, stage T read
        } else {
            split_signal_tdm_lds<fwd::TCNT_LAST>(); // the same; Q(j+1), op(T+3) may fly
        }
        // E
        if constexpr (par == 0) { // the stage's one burst, in the order of first use (in-order DS counter)
            if constexpr (!last) {
                ld_k(kab[0], kb1, ic<0>{}); // K(c+2): stage T+1, half 0 (QK in the odd B)
                sched_barrier<0>();
            }
            ld_v(vO, vb0, ic<1>{}); // V(c+1): stage T, half 1 (PV in the odd F)
            sched_barrier<0>();
            if constexpr (!last) {
                ld_k(kab[1], kb1, ic<1>{}); // K(c+3): stage T+1, half 1 (QK in the next even B)
                sched_barrier<0>();
                ld_v(vN, vb1, ic<0>{}); // V(c+2): stage T+1, half 0 (PV in the next even F)
                sched_barrier<0>();
            }
        }
        // F1: PV(c) q tiles 0, 1 || SM1(c+1), pack of q tile 2
        pv(va, ic<0>{});
        pv(va, ic<1>{});
        if constexpr (sm != 0) sm1(S[nxp], c + 1, ic<sm == 2>{}, ic<0>{});
        tail_pack(S[par], ic<2>{});
        rsum(S[par], ic<0>{});
        rsum(S[par], ic<1>{});
        // {1 WMMA, 3 VALU} per PV WMMA, so the SM1 max trees do not run ahead of the first PV WMMA
        static_for<2 * NDT>([&](auto) {
            sched_group_barrier<sg::WMMA, 1, 0>();
            sched_group_barrier<sg::VALU, 3, 0>();
        });
        sched_barrier<0>();
        // F2: PV(c) q tiles 2, 3 || exp + pack of q tile 3
        pv(va, ic<2>{});
        tail_exp(S[par], ic<3>{});
        tail_pack(S[par], ic<3>{});
        pv(va, ic<3>{});
        rsum(S[par], ic<2>{});
        rsum(S[par], ic<3>{});
        sched_barrier<0>();
        // C: rescale for chunk c+1
        if constexpr (sm != 0) rescale();
        if constexpr (par == 1) { // the next stage's burst bases
            kb0 = kb1;
            kb1 = kb2;
            vb0 = vb1;
            vb1 = vb2;
        }
        if constexpr (par == 1 && !last) static_for<NDT>([&](auto dt) { vE[dt] = vN[dt]; }); // V(2t+2): the next PV(even)
    };

    // ---- item phases -----------------------------------------------------------------------------------------
    // prologue loads of the current item (global stage T0 = this item's stage 0, published): own Q landed,
    // Q^T burst, K(0), K(1), V(0)
    auto front_loads = [&]() FWD_INL {
        wait_tensorcnt<fwd::TCNT_ITEM>(); // own Q(j) landed (op(T0+2) may fly)
        const lds_u8 *qp = lds_ptr(lds0 + (uint32_t) fwd::q_lane(w, l));
        static_for<NH>([&](auto h) {
            static_for<NDK>([&](auto k) {
                qb[h][k] = cat(ds_load_b128(qp, fwd::q_imm(h, k, 0)), ds_load_b128(qp, fwd::q_imm(h, k, 1)));
            });
        });
        ld_k(kab[0], kb0, ic<0>{}); // K(0), consumed by QK(0)
        ld_k(kab[1], kb0, ic<1>{}); // K(1), consumed by QK(1) in iteration 0
        ld_v(vE, vb0, ic<0>{});     // V(0), consumed by PV(0) in iteration 0
        sched_barrier<0>();
    };
    auto front_qk = [&]() FWD_INL {
        qk(S[0], kab[0]);
    };
    auto front_sm1 = [&]() FWD_INL {
        sm1(S[0], 0, ic<1>{}, ic<1>{}); // SM1(0): masked (effective iff n == 1), m2 = max
        // retire the front's loads here, after QK(0): otherwise V(0) is still pending in LLVM's DS score window at
        // the stage loop's entry and every iteration's PV(0) gets an s_wait_dscnt
        wait_dscnt<0>();
        sched_barrier<0>();
    };
    // stages of the current item
    auto stages = [&]() FWD_INL {
        const int nmain = fwd::imax(n - 2, 0);
        // 2 stages per loop iteration; an odd count runs stage 0 first (workgroup-uniform branch)
        int t0 = 0;
        if (nmain & 1) {
            iter(0, ic<0>{}, ic<1>{}, ic<1>{}, ic<1>{}, ic<0>{});
            iter(0, ic<1>{}, ic<1>{}, ic<2>{}, ic<1>{}, ic<0>{});
            t0 = 1;
        }
        for (int t = t0; t < nmain; t += 2) {
            iter(t, ic<0>{}, ic<1>{}, ic<1>{}, ic<1>{}, ic<0>{});
            iter(t, ic<1>{}, ic<1>{}, ic<2>{}, ic<1>{}, ic<0>{});
            iter(t + 1, ic<0>{}, ic<1>{}, ic<1>{}, ic<1>{}, ic<0>{});
            iter(t + 1, ic<1>{}, ic<1>{}, ic<2>{}, ic<1>{}, ic<0>{});
        }
        if (fwd::has_pen_stage(n)) { // stage n-2, its SM1(2n-2) masked
            const int t = n - 2;
            iter(t, ic<0>{}, ic<1>{}, ic<1>{}, ic<1>{}, ic<1>{});
            iter(t, ic<1>{}, ic<1>{}, ic<2>{}, ic<2>{}, ic<0>{});
        }
        {
            const int t = n - 1; // last stage: Q(j+1) + nx[2]; SM1(2n-1) masked; no QK(2n)
            iter(t, ic<0>{}, ic<1>{}, ic<3>{}, ic<2>{}, ic<2>{});
            iter(t, ic<1>{}, ic<0>{}, ic<4>{}, ic<0>{}, ic<0>{});
        }
    };
    // epilogue pieces (wave-private O region + VMEM; no barrier)
    auto epi_lt = [&](float (&lt)[NH]) FWD_INL {
        static_for<NH>([&](auto h) {
            const float ll = lsum[h][0] + lsum[h][1];
            lt[h]          = ll + xhalf(ll); // identical bits in lanes l and l ^ 16
        });
    };
    auto epi_norm = [&](u32x4(&pk)[NH][NDT], const float (&lt)[NH]) FWD_INL {
        static_for<NH>([&](auto h) {
            const float inv = lt[h] > 0.f ? rcp_ieee(lt[h]) : 0.f;
            static_for<NDT>([&](auto dt) { pk[h][dt] = cvt_pk_bf16(oacc[h][dt] * inv); });
        });
    };
    auto epi_image = [&](const u32x4(&pk)[NH][NDT]) FWD_INL {
        lds_u8 *ow = lds_ptr(lds0 + (uint32_t) fwd::ow_lane(w, l));
        static_for<NH>([&](auto h) { static_for<NDT>([&](auto dt) { ds_store_b128(ow, fwd::ow_imm(h, dt), pk[h][dt]); }); });
    };
    auto epi_out = [&](const fwd::Work &x, const float (&lt)[NH]) FWD_INL {
        compiler_fence(); // read-back after the image writes (same-wave DS ops complete in order)
        const kargs_p ka = kargs();
        const int     Sq = ka->Sq, Hq = ka->Hq;
        const rsrc_t  ro = make_rsrc(ka->O, ka->o_bytes); // the whole tensor (SBHD: a batch's rows interleave)
        const lds_u8 *rp = lds_ptr(lds0 + (uint32_t) fwd::or_lane(w, l));
        const fwd::Lay lo = lay_o(ka);
        static_for<2 * fwd::QW>([&](auto s) {
            const u32x4 v = ds_load_b128_u32(rp, fwd::or_imm(s));
            buf_store_b128(ro, (int) fwd::o_store_b_s(lo, x.b, x.hq4, x.qb, w, s, l), 0, v);
        });
        const rsrc_t rl = make_rsrc(ka->LSE, ka->lse_bytes);
        float        lse[NH];
        static_for<NH>([&](auto h) { lse[h] = (m2[h] + log2_hw(lt[h])) * fwd::LN2; });
        const bool hi = l >= 16;
        static_for<2>([&](auto k) {
            buf_store_f32(rl, (int) fwd::lse_store_b(x.b, Hq, Sq, x.hq4, x.qb, w, k, l), 0, hi ? lse[2 + k] : lse[k]);
        });
    };

    auto item_of = [&](int j) FWD_INL { return fwd::wg_item(sched(kargs()), g, j); };

    // ---- workgroup prologue -------------------------------------------------------------------------------------
    {
        const kargs_p    ka = kargs();
        const fwd::Sched sc = sched(ka);
        const fwd::Shape sh = shape(ka);
        fwd::StRef       r[3];
        fwd::stream3(sh, sc, g, J, 0, r); // global stages 0, 1, 2
        const fwd::Lay lk = lay_kv(ka);
        issue(q_desc(fwd::q_c_s(1, fwd::wg_item(sc, g, 0), sh.Sq, lay_q(ka), w)));
        issue(op_desc(fwd::op_c_s(sh, lk, r[0], w), 0));
        issue(op_desc(fwd::op_c_s(sh, lk, r[1], w), 1));
        sched_barrier<0>();
        wait_tensorcnt<fwd::TCNT_STEADY>(); // own Q(0) and op 0 landed
        lds_barrier();                      // P_0: every wave's op 0 (stage 0) published
        issue(op_desc(fwd::op_c_s(sh, lk, r[2], w), 2));
        split_signal_tdm_lds<fwd::TCNT_STEADY>(); // SIGNAL(P_1): own op 1 landed
    }
    kb0 = kbase(0);
    kb1 = kbase(1);
    vb0 = vbase(0);
    vb1 = vbase(1);
    for (int j = 0; j < J; ++j) {
        set_cur(j);
        front_loads();
        zero_state();
        set_ahead(j);
        front_qk();
        front_sm1();
        stages();
        float lt[NH];
        epi_lt(lt);
        u32x4 pk[NH][NDT];
        epi_norm(pk, lt);
        epi_image(pk);
        epi_out(item_of(j), lt);
        sched_barrier<0>();
        T0 += n;
    }
    // ---- exit: every TDM op of this wave landed (dummies included), then WAIT(P_{N+1}) ---------------------------
    sched_barrier<0>();
    wait_tensorcnt<0>();
    split_wait_lds();
}
