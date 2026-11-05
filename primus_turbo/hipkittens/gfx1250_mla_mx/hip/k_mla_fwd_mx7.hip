/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// k_mla_fwd_mx (rung m7) -- m6 with paired blocks: a workgroup runs q blocks nqb - 1 - p and p of one (b, h) (2 (16 - p)
// + 2 (p + 1) = 34 tiles for nqb 16: balanced), the ring index running on across the two blocks, so the first block's
// last three tile slots TDM-load the second block's first three tiles (the same kv tiles); half the workgroup launches.
// The pairs of a (b, h) on one XCD. B H % 8 != 0 or nqb odd: one block per workgroup (pair_mode() on host and device).
// m6's notes follow.
// k_mla_fwd_mx (rung m6) -- m4's tile / slot pipeline (QK(q_{i+1}) and PV(q_{i-1}) WMMAs with softmax(q_i)) and its
// operand recipe (Q8 of q * scale * log2 e, R2 scales, K centred), with a branch-free softmax:
//  - per lane (q column) integer shifts: lm = max S'' (S'' = S' - m out of the nope WMMA, C = -m); d = ceil(lm) when
//    lm > TAU2, else 0; the exp2 of S'' does not wait for the max: the shift goes into the P conversion's scale
//    (P = E4M3(exp2(S'') / 2^(d - PE))), so the 64 exps and the 32 max3 of a q tile are independent work;
//  - O and the row sum (the all-ones PV WMMA) accumulate relative to a fixed per-lane reference mref (the first
//    tile's shift): the PV WMMAs' E8M0 B scale is 2^(D - PE) per lane, D = m - mref (an integer). O is never rescaled,
//    no wave-uniform decision, no branch per q tile. A guard rescales O when D > DGUARD (any lane; once per tile).
//  - first tile: exact-integer path: d = ceil(lm) from the max, S'' -= d before the exp2, mref = d.
// lse = (mref + log2 l) ln 2. Emulation: mxlab.emu_fwd6 (o -0.25 dB vs m4 on randn: the max element of P is not a
// power of two after an integer shift).
// Scheduling: SCHED 0 = m4's sched_barrier fences (chunk c of a slot at step floor(c NSTEP / NCHUNK)); SCHED 1 = one
// region per slot, ordered by sched_group_barrier windows [WMMA, VALU, TRANS, VALU, TRANS (, TRANS)].
#include "mx_common.h"

using namespace mx;

namespace {
constexpr int D_QK = 192, D_V = 128;
constexpr int BM = 256;
constexpr int QT = 4;
constexpr int NB = 128;
constexpr int NKV = NB / 16;
constexpr int NDT = D_V / 16;
constexpr int XK = 208, XV = 144;
constexpr int K_B = NB * XK;
constexpr int KS_B = NB * 8;
constexpr int V_B = D_V * XV;
constexpr int VS_B = D_V * 4;
constexpr int STG_B = K_B + KS_B + V_B + VS_B;
constexpr int NSTG = 4;
constexpr int Q_OFF = NSTG * STG_B;
constexpr int QS_OFF = Q_OFF + BM * XK;
constexpr int LDS_RING_Q = QS_OFF + BM * 8;
#ifndef OSTAGE
#define OSTAGE 1  // epilogue: O staged in LDS (256 B rows), one TDM store per block
#endif
constexpr int XO = 2 * D_V;                        // staging row stride (bytes): no pad (the TDM store
                                                   // ignores LDS padding: rows after the first come out wrong)
constexpr int O_OFF = LDS_RING_Q;                  // staging: BM rows x XO
constexpr int LDS_B = OSTAGE ? O_OFF + BM * XO : LDS_RING_Q;
static_assert(LDS_B <= 327680, "LDS");
constexpr unsigned W0_KN = tdm_w0(0, 32, 20), W0_KR = tdm_w0(0, 16, 36), W0_V = tdm_w0(0, 32, 4), W0_S = tdm_w0(0, 0, 0);
#ifndef TAU2_X100
#define TAU2_X100 75
#endif
#ifndef PE
#define PE 8
#endif
#ifndef SCHED
#define SCHED 1
#endif
#ifndef NO_GUARD
#define NO_GUARD 0
#endif
#ifndef PAIR
#define PAIR 1
#endif
#ifndef PRO_PARTIAL
#define PRO_PARTIAL 1
#endif
#ifndef QKFIRST
#define QKFIRST 1
#endif
#ifndef XCD_MAP
#define XCD_MAP 1  // workgroups of one (b, h) on one XCD (dispatch: linear id L -> XCD L % 8): K / V reuse in its L2
#endif
#ifndef TDM_SPLIT
#define TDM_SPLIT 1
#endif
#ifndef TDM_ONLY
#define TDM_ONLY 0  // 1: only the TDM ring, its waits and the barriers (the data movement floor)
#endif
#ifndef FIRST_PEEL
#define FIRST_PEEL 0  // 1: a separate first-tile body (exact shift before the exps; costs ~300 spills)
#endif
constexpr float TAU2 = TAU2_X100 / 100.0f;
static_assert(TAU2_X100 + 100 * PE <= 880, "P = E4M3(p 2^PE) <= 2^(TAU2 + PE) must stay below 448 = 2^8.807");
constexpr int SC_ONE = 127 * 0x01010101;   // E8M0 2^0
constexpr int ONES = 0x38383838;           // E4M3 1.0 x 4
constexpr int DGUARD = 48;                 // rescale O when D = m - mref exceeds this (B scale range, fp32 range)
#ifndef GCHUNK
#define GCHUNK 4                           // tiles between two guard checks
#endif
constexpr unsigned PVS0 = (unsigned)(127 - PE) * 0x01010101u;  // the B scale bytes at D = 0: 2^-PE

typedef int32_t i32x32 __attribute__((ext_vector_type(32)));

struct KFrag {
    i32x32 x[NKV];  // lanes 0..23: [nope 16 | rope 8] of this lane's kv row half
    int sc[NKV];
};
struct VFrag {
    i32x16 t[NDT];
    int sc[NDT / 2];
};
struct QFrag {
    i32x32 x;
    int sc;
};
// the softmax state of one q tile (per lane = q column)
struct QState {
    f32x8 o[NDT];  // O^T accumulators (rows d), relative to mref
    f32x8 os;      // row sum (all-ones PV WMMA), relative to mref
    float m;       // the running max (log2 units); the nope WMMA's C = -m in all 8 rows (mtuple), so S'' = S' - m
                   // comes out of the WMMA. One VGPR per q tile, the 8-VGPR tuple only while its QK runs
    float mref;    // the reference (log2 units)
    int D;         // m - mref (integer); the PV WMMAs' B scale 2^(D - PE) is formed from it per PV group (pv_scale):
                   // a long-lived scale VGPR would have to live in v0..v255 (WMMA scale operands), which spills
};
MX_INLINE int pv_scale(int D) { return (int)((unsigned)D * 0x01010101u + PVS0); }

MX_INLINE i32x32 frag24(u32x4 a0, u32x4 a1, u32x4 a2, u32x4 a3, u32x4 r0, u32x4 r1) {
    const i32x16 n = {(int)a0[0], (int)a0[1], (int)a0[2], (int)a0[3], (int)a1[0], (int)a1[1], (int)a1[2], (int)a1[3],
                      (int)a2[0], (int)a2[1], (int)a2[2], (int)a2[3], (int)a3[0], (int)a3[1], (int)a3[2], (int)a3[3]};
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
MX_INLINE void fence() { __builtin_amdgcn_sched_barrier(0); }
// sched_group_barrier masks
constexpr int SG_VALU = 0x2, SG_WMMA = 0x8, SG_DSR = 0x100, SG_TRANS = 0x400;
template <int MASK, int N> MX_INLINE void sgb() {
    if constexpr (N > 0) __builtin_amdgcn_sched_group_barrier(MASK, N, 0);
}

struct KAddr {
    uint32_t n, r, s;
};
struct VAddr {
    uint32_t t, s;
};
MX_INLINE uint32_t pin(uint32_t x) {
    asm volatile("" : "+v"(x));
    return x;
}
MX_INLINE int lane_now() {
    int l;
    asm volatile("v_mbcnt_lo_u32_b32 %0, -1, 0" : "=v"(l));
    return l;
}
MX_INLINE KAddr kaddr(uint32_t stage_base, int lr, int lh) {
    return {pin(stage_base + (unsigned)(lr * XK + 64 * lh)), pin(stage_base + (unsigned)(lr * XK + 128 + 32 * lh)),
            pin(stage_base + (unsigned)(K_B + lr * 8 + 4 * lh))};
}
MX_INLINE VAddr vaddr(uint32_t stage_base, int lr, int lh, int lane) {
    return {pin(stage_base + (unsigned)(K_B + KS_B + lr * XV + 64 * lh)),
            pin(stage_base + (unsigned)(K_B + KS_B + V_B + lane * 4))};
}
template <int J>
MX_INLINE void load_k1(KFrag &k, const KAddr &a) {
    const lds_u8 *pn = lds_ptr(a.n), *pr = lds_ptr(a.r), *ps = lds_ptr(a.s);
    constexpr int o = 16 * J * XK;
    k.x[J] = frag24(lds_b128(pn, o), lds_b128(pn, o + 16), lds_b128(pn, o + 32), lds_b128(pn, o + 48),
                    lds_b128(pr, o), lds_b128(pr, o + 16));
    k.sc[J] = (int)lds_b32(ps, 16 * J * 8);
}
MX_INLINE void load_v(VFrag &v, const VAddr &a) {
    const lds_u8 *pt = lds_ptr(a.t), *ps = lds_ptr(a.s);
    static_for<NDT>([&](auto dd) {
        constexpr int o = 16 * (int)dd * XV;
        const u32x4 a0 = lds_b128(pt, o), a1 = lds_b128(pt, o + 16), a2 = lds_b128(pt, o + 32),
                    a3 = lds_b128(pt, o + 48);
        v.t[(int)dd] = i32x16{(int)a0[0], (int)a0[1], (int)a0[2], (int)a0[3], (int)a1[0], (int)a1[1], (int)a1[2],
                              (int)a1[3], (int)a2[0], (int)a2[1], (int)a2[2], (int)a2[3], (int)a3[0], (int)a3[1],
                              (int)a3[2], (int)a3[3]};
    });
    static_for<NDT / 2>([&](auto pp) { v.sc[(int)pp] = (int)lds_b32(ps, 32 * 4 * (int)pp); });
}
struct QAddr {
    uint32_t n, r, s;
};
template <int I>
MX_INLINE void load_q(QFrag &q, const QAddr &a) {
    const lds_u8 *pn = lds_ptr(a.n), *pr = lds_ptr(a.r);
    constexpr int o = 64 * I * XK;
    q.x = frag24(lds_b128(pn, o), lds_b128(pn, o + 16), lds_b128(pn, o + 32), lds_b128(pn, o + 48), lds_b128(pr, o),
                 lds_b128(pr, o + 16));
    q.sc = (int)lds_b32(lds_ptr(a.s), 64 * I * 8);
}

// QK WMMA w (0..15) of a q tile: w < 8 the nope WMMA of kv16 tile w (C = -m), w >= 8 the rope WMMA of kv16 tile w - 8
MX_INLINE f32x8 mtuple(float m) { return f32x8{-m, -m, -m, -m, -m, -m, -m, -m}; }
template <int W>
MX_INLINE void qk_w(f32x8 (&s)[NKV], const KFrag &k, const QFrag &q, const f32x8 &mt) {
    if constexpr (W < NKV) {
        s[W] = wmma_mx<0, 0>(lo16(k.x[W]), lo16(q.x), mt, k.sc[W], q.sc);
    } else {
        s[W - NKV] = wmma_mx<1, 1>(mid16(k.x[W - NKV]), mid16(q.x), s[W - NKV], k.sc[W - NKV], q.sc);
    }
}
// PV WMMA w (0..8) of a q tile: w < 8 V^T tile w, w == 8 the all-ones tile (the row sum); B scale = the lane's 2^(D-PE)
template <int W>
MX_INLINE void pv_w(QState &st, const VFrag &v, const i32x16 &pb, int pvs, const i32x16 &ones) {
    if constexpr (W == NDT)
        st.os = wmma_mx<0, 0>(ones, pb, st.os, SC_ONE, pvs);
    else if constexpr (W % 2 == 0)
        st.o[W] = wmma_mx<0, 0>(v.t[W], pb, st.o[W], v.sc[W / 2], pvs);
    else
        st.o[W] = wmma_mx<1, 0>(v.t[W], pb, st.o[W], v.sc[W / 2], pvs);
}

// ---- softmax pieces of one q tile (lane = q column; s = S'' of the 8 kv16 tiles, 8 rows each)
// mask (diagonal tiles): kv > q iff 16 j + r > thr
template <int J, bool DIAG>
MX_INLINE void sm_mask(f32x8 (&s)[NKV], int thr) {
    if constexpr (DIAG) {
        static_for<8>([&](auto rr) {
            constexpr int r = rr;
            s[J][r] = (16 * J + r > thr) ? -__builtin_inff() : s[J][r];
        });
    }
}
// partial max of kv16 tile J into pm[J / 2] (4 partials: 2 tiles each)
template <int J>
MX_INLINE void sm_max(float (&pm)[4], const f32x8 (&s)[NKV]) {
    float a = (J % 2 == 0) ? -__builtin_inff() : pm[J / 2];
    static_for<4>([&](auto hh) {
        constexpr int h2 = hh;
        a = __builtin_fmaxf(a, __builtin_fmaxf(s[J][2 * h2], s[J][2 * h2 + 1]));
    });
    pm[J / 2] = a;
}
template <int J>
MX_INLINE void sm_exp(f32x8 (&s)[NKV]) {
    static_for<8>([&](auto rr) { s[J][(int)rr] = exp2_raw(s[J][(int)rr]); });
}
// the lane's max over its 64 values and its peer's (the other 8 kv rows of every kv16 tile)
MX_INLINE float sm_lm(const float (&pm)[4]) {
    const float mx = __builtin_fmaxf(__builtin_fmaxf(pm[0], pm[1]), __builtin_fmaxf(pm[2], pm[3]));
    return __builtin_fmaxf(mx, peer(mx));
}
// non-first tile: d = lm > TAU2 ? ceil(lm) : 0; pscale = 2^(d - PE); D += d; m += d
MX_INLINE float sm_decide(QState &st, float lm, bool first = false) {
    // first (uniform; FIRST_PEEL 0): d = ceil(lm) also downwards, mref = d, D = 0 -- the exps ran unshifted, which
    // needs |S'| < 126 on the first tile (logits under 87 nats)
    const float thr = first ? -__builtin_inff() : TAU2;  // uniform: the compare stays a VALU compare
    const float d = lm > thr ? __builtin_ceilf(lm) : 0.f;
    const int di = (int)d;
    if (first) st.mref = d;
    st.D = first ? 0 : st.D + di;
    st.m += d;
    return __builtin_ldexpf(1.0f / (float)(1 << PE), di);
}
// first tile: d = ceil(lm) (lm finite: kv 0 is in every q row's first tile), S'' -= d, mref = d, D = 0
MX_INLINE void sm_first(QState &st, f32x8 (&s)[NKV], float lm) {
    const float d = __builtin_ceilf(lm);
    st.mref = d;
    st.D = 0;
    st.m += d;
    const f32x2 dv = {d, d};
    static_for<NKV>([&](auto jj) {
        static_for<4>([&](auto hh) {
            constexpr int j = jj, h2 = hh;
            f32x2 x = {s[j][2 * h2], s[j][2 * h2 + 1]};
            x = x - dv;
            s[j][2 * h2] = x[0];
            s[j][2 * h2 + 1] = x[1];
        });
    });
}
template <int J>
MX_INLINE void sm_cvt(i32x16 &pb, const f32x8 (&s)[NKV], float pscale) {
    const u32x2 q = cvt8_e4m3(s[J], pscale);
    pb[2 * J] = (int)q[0];
    pb[2 * J + 1] = (int)q[1];
}
// the whole softmax of a q tile, in dependency order: per kv16 tile mask, max, exp; the decision; the conversions.
// FIRST: max of all tiles first, the exact shift, then the exps.
template <bool FIRST, bool DIAG>
MX_INLINE void softmax(QState &st, f32x8 (&s)[NKV], i32x16 &pb, int thr, bool first = false) {
    float pm[4];
    float pscale;
    if constexpr (FIRST) {
        static_for<NKV>([&](auto jj) {
            sm_mask<(int)jj, DIAG>(s, thr);
            sm_max<(int)jj>(pm, s);
        });
        sm_first(st, s, sm_lm(pm));
        static_for<NKV>([&](auto jj) { sm_exp<(int)jj>(s); });
        pscale = 1.0f / (float)(1 << PE);
    } else {
        static_for<NKV>([&](auto jj) {
            sm_mask<(int)jj, DIAG>(s, thr);
            sm_max<(int)jj>(pm, s);
            sm_exp<(int)jj>(s);
        });
        pscale = sm_decide(st, sm_lm(pm), first);
    }
    static_for<NKV>([&](auto jj) { sm_cvt<(int)jj>(pb, s, pscale); });
}
// SCHED 0: the softmax in 13 chunks (8 x mask / max / exp of one kv16 tile, the decision, 4 x two conversions)
template <int C, bool FIRST, bool DIAG>
MX_INLINE void sm_chunk(QState &st, f32x8 (&s)[NKV], i32x16 &pb, int thr, float (&pm)[4], float &pscale,
                        bool first = false) {
    if constexpr (FIRST) {  // FIRST: all max first (chunks 0..7 mask + max; chunk 8 shift + 64 exps)
        if constexpr (C < NKV) {
            sm_mask<C, DIAG>(s, thr);
            sm_max<C>(pm, s);
        } else if constexpr (C == NKV) {
            sm_first(st, s, sm_lm(pm));
            static_for<NKV>([&](auto jj) { sm_exp<(int)jj>(s); });
            pscale = 1.0f / (float)(1 << PE);
        } else {
            sm_cvt<2 * (C - NKV - 1)>(pb, s, pscale);
            sm_cvt<2 * (C - NKV - 1) + 1>(pb, s, pscale);
        }
    } else {
        if constexpr (C < NKV) {
            sm_mask<C, DIAG>(s, thr);
            sm_max<C>(pm, s);
            sm_exp<C>(s);
        } else if constexpr (C == NKV) {
            pscale = sm_decide(st, sm_lm(pm), first);
        } else {
            sm_cvt<2 * (C - NKV - 1)>(pb, s, pscale);
            sm_cvt<2 * (C - NKV - 1) + 1>(pb, s, pscale);
        }
    }
}
constexpr int NCHUNK = NKV + 1 + NKV / 2;

// the overflow guard: D > DGUARD in any lane of any q tile (rare): O, the row sum *= 2^-D, mref += D, D = 0
MX_INLINE void guard_rescale(QState (&st)[QT]) {
    static_for<QT>([&](auto ii) {
        constexpr int i = ii;
        const float c = __builtin_ldexpf(1.0f, -st[i].D);
        const f32x2 cv = {c, c};
        static_for<NDT>([&](auto dd) {
            static_for<4>([&](auto hh) {
                constexpr int dt = dd, h2 = hh;
                f32x2 x = {st[i].o[dt][2 * h2], st[i].o[dt][2 * h2 + 1]};
                x = x * cv;
                st[i].o[dt][2 * h2] = x[0];
                st[i].o[dt][2 * h2 + 1] = x[1];
            });
        });
        static_for<4>([&](auto hh) {
            constexpr int h2 = hh;
            f32x2 x = {st[i].os[2 * h2], st[i].os[2 * h2 + 1]};
            x = x * cv;
            st[i].os[2 * h2] = x[0];
            st[i].os[2 * h2 + 1] = x[1];
        });
        st[i].mref += (float)st[i].D;
        st[i].D = 0;
    });
}
}  // namespace


MX_INLINE bool pair_mode(int nqb, int H, int B) { return PAIR && nqb % 2 == 0 && (B * H) % 8 == 0; }

extern "C" __global__ __attribute__((amdgpu_flat_work_group_size(128, 128), amdgpu_waves_per_eu(1, 1)))
void k_mla_fwd_mx(const unsigned char *__restrict__ Q8, const unsigned char *__restrict__ QS,
                  const unsigned char *__restrict__ K8, const unsigned char *__restrict__ KS,
                  const unsigned char *__restrict__ V8T, const unsigned char *__restrict__ VS, bf16 *__restrict__ O,
                  float *__restrict__ LSE, float scale, int S, int H, int B, long long o_sb, long long o_ss) {
    __shared__ __attribute__((aligned(1024))) unsigned char smem[LDS_B];
    (void)scale;  // folded into Q8
    lock_simd();
    const uint32_t lds0 = lds_addr(smem);
    const int tid = (int)__builtin_amdgcn_workitem_id_x();
    const int wv = wave_id(tid);
    const int lane = tid & 31, lr = lane & 15, lh = lane >> 4;
    const int nqb = S / BM;
    // blocks: paired (grid (nqb / 2, H, B)): the k-th workgroup of XCD c (k = L / 8) takes pair p = k % (nqb / 2) of
    // bh = c + 8 (k / (nqb / 2)): q blocks nqb - 1 - p, then p. Else (grid (nqb, H, B)): m6's XCD order, one block.
    const bool paired = pair_mode(nqb, H, B);
    int qbA, qbB, h, b;
    if (paired) {
        const int np = nqb / 2;
        const int L = (int)blockIdx.x + np * ((int)blockIdx.y + H * (int)blockIdx.z);
        const int k = L >> 3;
        const int bhx = (L & 7) + 8 * (k / np);
        qbA = nqb - 1 - k % np;
        qbB = k % np;
        h = bhx % H;
        b = bhx / H;
    } else if ((B * H) % 8 == 0) {
        const int L = (int)blockIdx.x + nqb * ((int)blockIdx.y + H * (int)blockIdx.z);
        const int k = L >> 3;
        const int bhx = (L & 7) + 8 * (k / nqb);
        qbA = nqb - 1 - k % nqb;
        qbB = -1;
        h = bhx % H;
        b = bhx / H;
    } else {
        qbA = nqb - 1 - (int)blockIdx.x;
        qbB = -1;
        h = (int)blockIdx.y;
        b = (int)blockIdx.z;
    }
    const int nA = 2 * (qbA + 1), nB = qbB >= 0 ? 2 * (qbB + 1) : 0;
    const unsigned long long bh = (unsigned long long)b * H + h;
    const unsigned long long gk = (unsigned long long)(uintptr_t)K8 + bh * S * D_QK;
    const unsigned long long gks = (unsigned long long)(uintptr_t)KS + bh * S * 8;
    const unsigned long long gv = (unsigned long long)(uintptr_t)V8T + bh * D_V * S;
    const unsigned long long gvs = (unsigned long long)(uintptr_t)VS + bh * (unsigned long long)(S / NB) * VS_B;

    // the tile's 5 TDM ops spread over the waves (m6 TDM_SPLIT): two per wave and tile
    constexpr int TDM_PER_TILE = 2;
    const unsigned a_lds = wv == 0 ? 0u : wv == 1 ? 128u : wv == 2 ? (unsigned)(K_B + KS_B) : (unsigned)(K_B + KS_B + V_B);
    const unsigned long long a_base = wv == 0 ? gk : wv == 1 ? gk + 128 : wv == 2 ? gv : gvs;
    const unsigned long long a_step = wv <= 1 ? (unsigned long long)NB * D_QK : wv == 2 ? (unsigned long long)NB
                                                                                       : (unsigned long long)VS_B;
    const i32x8 a_g1 = tdm_g1(wv == 0 ? W0_KN : wv == 1 ? W0_KR : wv == 2 ? W0_V : W0_S, 128,
                              wv == 0 ? 128u : wv == 1 ? 64u : wv == 2 ? 128u : 4u, 128,
                              wv <= 1 ? (unsigned long long)D_QK : wv == 2 ? (unsigned long long)S : 4ull);
    const i32x8 b_g1 = tdm_g1(W0_S, 128, 8, NB, 8);
    const int b_cnt = wv == 1 ? 1 : 0;
    // ring slot g (global over the workgroup's blocks): kv tile g of block A, then kv tile g - nA of block B (the same
    // (b, h): the same K / V), past the end a NULL descriptor
    auto issue_slot = [&](int g) {
        const int tb = g - nA;
        const bool real = g < nA || tb < nB;
        const int tt = g < nA ? g : (tb < nB ? tb : 0);
        const unsigned st = lds0 + (unsigned)((g & (NSTG - 1)) * STG_B);
        tdm_load(tdm_g0(st + a_lds, a_base + (unsigned long long)tt * a_step, real ? 1 : 0), a_g1);
        tdm_load(tdm_g0(st + K_B, gks + (unsigned long long)tt * (NB * 8), real ? b_cnt : 0), b_g1);
    };
    auto stage = [&](int g) { return lds0 + (unsigned)((g & (NSTG - 1)) * STG_B); };
    const i32x16 ones = {ONES, ONES, ONES, ONES, ONES, ONES, ONES, ONES,
                         ONES, ONES, ONES, ONES, ONES, ONES, ONES, ONES};

    // one q block: rb = its ring base (the global ring slot of its tile 0); its tiles 0..2 already issued
    auto run_block = [&](const int qb, const int rb) __attribute__((always_inline)) {
        const int q0 = qb * BM;
        const int n = (q0 + BM) / NB;
        const unsigned long long gq = (unsigned long long)(uintptr_t)Q8 + (bh * S + q0) * D_QK;
        const unsigned long long gqs = (unsigned long long)(uintptr_t)QS + (bh * S + q0) * 8;
        if (rb == 0) {
            // first block: its Q went out in the workgroup prologue ahead of tiles 0..2 (in-order completion): once
            // Q and tile 0 landed only tiles 1, 2 (2 ops each) may stay in flight, which the loop's B_{t+1} waits cover
#if PRO_PARTIAL
            tensor_wait<2 * TDM_PER_TILE>();
#else
            tensor_wait<0>();
#endif
        } else {
            // second block: its tiles 0..2 came during the first block; its Q now (the region is free: the first
            // block's last Q read was before its last tile's B_{t+1} wait)
            tdm_load(tdm_g0(lds0 + Q_OFF, gq, wv == 1 ? 1 : 0), tdm_g1(W0_KN, BM, 128, BM, D_QK));
            tdm_load(tdm_g0(lds0 + Q_OFF + 128, gq + 128, wv == 2 ? 1 : 0), tdm_g1(W0_KR, BM, 64, BM, D_QK));
            tdm_load(tdm_g0(lds0 + QS_OFF, gqs, wv == 3 ? 1 : 0), tdm_g1(W0_S, BM, 8, BM, 8));
            tensor_wait<0>();
        }
        bar_signal();
        bar_wait();  // B_0 of the block: its Q and tile 0 in LDS; the previous block's O store done

        auto qaddr = [&]() __attribute__((always_inline)) {
            const int ln = lane_now();
            const unsigned row = (unsigned)(16 * wv + (ln & 15)), hh = (unsigned)(ln >> 4);
            return QAddr{pin(lds0 + Q_OFF + row * XK + 64 * hh), pin(lds0 + Q_OFF + row * XK + 128 + 32 * hh),
                         pin(lds0 + QS_OFF + row * 8 + 4 * hh)};
        };
        QState st[QT];
        static_for<QT>([&](auto ii) {
            constexpr int i = ii;
            st[i].mref = 0.f;
            st[i].D = 0;
            st[i].m = 0.f;
            st[i].os = f32x8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            static_for<NDT>([&](auto dd) { st[i].o[(int)dd] = f32x8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}; });
        });
        KFrag kf;
        VFrag vf;
        {
            const KAddr ka = kaddr(stage(rb), lr, lh);
            static_for<NKV>([&](auto jj) { load_k1<(int)jj>(kf, ka); });
        }
        f32x8 sa[NKV], sb[NKV];
        i32x16 pa, pbb;
        QFrag qf[2];
        {
            const QAddr qa = qaddr();
            load_q<0>(qf[0], qa);
            load_q<1>(qf[1], qa);
        }
        {
            const f32x8 mt0 = mtuple(st[0].m);
            static_for<16>([&](auto ww) { qk_w<(int)ww>(sa, kf, qf[0], mt0); });
        }
        fence();
        // tile t: on entry sa = S''(q0) of tile t, qf[1] = Q(q1), kf = K(t). Slot 3 computes QK(q0) of tile t + 1 (not
        // LAST).
        auto tile = [&](int t, const QAddr &qa, auto first_c, auto diag_c, auto last_c) __attribute__((always_inline)) {
            constexpr bool FIRST = decltype(first_c)::value, DIAG = decltype(diag_c)::value,
                           LAST = decltype(last_c)::value;
            const VAddr va = vaddr(stage(rb + t), lr, lh, lane);
            const KAddr ka = kaddr(stage(rb + t + 1), lr, lh);
            fence();
            issue_slot(rb + t + 3);
            load_v(vf, va);
            fence();
            const int kvrow0 = t * NB + 8 * lh;
            static_for<QT>([&](auto ii) {
                constexpr int i = ii;
                f32x8(&scur)[NKV] = (i % 2 == 0) ? sa : sb;
                f32x8(&snext)[NKV] = (i % 2 == 0) ? sb : sa;
                i32x16 &pcur = (i % 2 == 0) ? pa : pbb;
                i32x16 &pprev = (i % 2 == 0) ? pbb : pa;
                const int qr_i = q0 + 64 * i + 16 * wv + lr;
                const int thr = qr_i - kvrow0;
                if constexpr (i == 1) {
                    tensor_wait<2 * TDM_PER_TILE>();
                    bar_signal();  // B_{t+1}
                    fence();
                }
                if constexpr (i == QT - 1) {
                    bar_wait();
                    if constexpr (!LAST) {
                        static_for<NKV>([&](auto jj) { load_k1<(int)jj>(kf, ka); });
                    }
                    fence();
                }
                constexpr int NPV = (i >= 1) ? NDT + 1 : 0;
                constexpr int NQK = (i + 1 < QT || !LAST) ? 16 : 0;
                constexpr int NW = NPV + NQK;
                constexpr int qn_mt = (i + 1) % QT;
                const f32x8 mtn = mtuple(st[qn_mt].m);
                constexpr bool LOADQ = !(LAST && i >= 2);
                const int pvsc = pv_scale(st[i - 1 < 0 ? 0 : i - 1].D);
                // one region: the slot's loads, WMMAs and softmax in program order, then the issue windows
                if constexpr (LOADQ) load_q<(i + 2) % QT>(qf[i % 2], qa);
                // WMMA order: QK(q_{i+1}) first where its K is resident (slots 1, 2: P(q_{i-1}), the PV's B operand, is
                // converted at the very end of slot i - 1 and needs the slack); slot 3 keeps PV first (its QK reads the
                // next tile's K fragments, loaded in this slot)
                constexpr bool QKF = QKFIRST && i >= 1 && i <= 2;
                static_for<NW>([&](auto kk) {
                    constexpr int k = kk;
                    if constexpr (QKF) {
                        if constexpr (k < NQK) {
                            qk_w<k>(snext, kf, qf[(i + 1) % 2], mtn);
                        } else {
                            pv_w<k - NQK>(st[i - 1 < 0 ? 0 : i - 1], vf, pprev, pvsc, ones);
                        }
                    } else if constexpr (k < NPV) {
                        pv_w<k>(st[i - 1 < 0 ? 0 : i - 1], vf, pprev, pvsc, ones);
                    } else {
                        qk_w<k - NPV>(snext, kf, qf[(i + 1) % 2], mtn);
                    }
                });
                softmax<FIRST, DIAG>(st[i], scur, pcur, thr, t == 0);
                // windows: the q loads up front, then per WMMA [VALU, TRANS, VALU, TRANS (, TRANS)]: 64 exps over the
                // slot's WMMAs (NW 16 or 25), the 55-odd VALU two per window
                if constexpr (LOADQ) sgb<SG_DSR, 7>();
                constexpr int NWIN = NW > 0 ? NW : 1;
                constexpr int T3 = 64 - 2 * NWIN > 0 ? 64 - 2 * NWIN : 0;  // windows with a third exp
                static_for<NWIN>([&](auto kk) {
                    constexpr int k = kk;
                    if constexpr (k < NW) sgb<SG_WMMA, 1>();
                    sgb<SG_VALU, 1>();
                    sgb<SG_TRANS, 1>();
                    sgb<SG_VALU, 1>();
                    sgb<SG_TRANS, 1>();
                    if constexpr (k < T3) sgb<SG_TRANS, 1>();
                });
                fence();
                (void)pprev;
            });
            // tail: PV(q3)
            const int pvs3 = pv_scale(st[QT - 1].D);
            static_for<NDT + 1>([&](auto ww) { pv_w<(int)ww>(st[QT - 1], vf, pbb, pvs3, ones); });
            fence();
        };
        {
            // the overflow guard between chunks of GCHUNK tiles: D grows by at most the row max's rise over a chunk, so
            // O stays finite unless the max climbs > ~56 log2 units (39 nats) within GCHUNK tiles after a check
            const QAddr qa = qaddr();
            for (int t0 = 0; t0 < n - 2; t0 += GCHUNK) {
                const int t1 = t0 + GCHUNK < n - 2 ? t0 + GCHUNK : n - 2;
                for (int t = t0; t < t1; ++t) tile(t, qa, std::false_type{}, std::false_type{}, std::false_type{});
    #if !NO_GUARD
                const int d01 = st[0].D > st[1].D ? st[0].D : st[1].D, d23 = st[2].D > st[3].D ? st[2].D : st[3].D;
                if (__builtin_amdgcn_ballot_w32((d01 > d23 ? d01 : d23) > DGUARD) != 0u) guard_rescale(st);
    #endif
            }
        }
        tile(n - 2, qaddr(), std::false_type{}, std::true_type{}, std::false_type{});
        tile(n - 1, qaddr(), std::false_type{}, std::true_type{}, std::true_type{});
        // epilogue: lse by plain stores (both lane halves, the same value: no exec change); O as bf16 rows into the LDS
        // staging block, then one TDM store of the block's 256 x 256 B by wave 0 (the scattered SBHD global stores held
        // their data VGPRs for hundreds of cycles each: s_wait_xcnt ~5.6k cycles per block)
        static_for<QT>([&](auto ii) {
            constexpr int i = ii;
            const int qr_i = q0 + 64 * i + 16 * wv + lr;
            const float l = st[i].os[0];
            LSE[(bh * S) + qr_i] = (st[i].mref + __builtin_log2f(l)) * LN2;
        });
        static_for<QT>([&](auto ii) {
            constexpr int i = ii;
            const float l = st[i].os[0];
            const float inv = l > 0.f ? 1.0f / l : 0.f;
            const int ln = lane_now();
            const unsigned rowb = lds0 + O_OFF + (unsigned)((64 * i + 16 * wv + (ln & 15)) * XO + 16 * (ln >> 4));
            static_for<NDT>([&](auto dd) {
                constexpr int dt = dd;
                bf16x8 w;
                static_for<8>([&](auto rr) { w[(int)rr] = (bf16)(st[i].o[dt][(int)rr] * inv); });
                *(__attribute__((address_space(3))) u32x4 *)(uintptr_t)(rowb + 32 * dt) = __builtin_bit_cast(u32x4, w);
            });
        });
        bar_signal();
        bar_wait();
        if (wv == 0) {
            const unsigned long long go = (unsigned long long)(uintptr_t)(O + (long long)b * o_sb + (long long)q0 * o_ss +
                                                                         (long long)h * D_V);
            tdm_store(tdm_g0(lds0 + O_OFF, go), tdm_g1(tdm_w0(0, 0, 0), BM, 2 * D_V, BM, 2ull * (unsigned long long)o_ss));
        }
    };

    // workgroup prologue: the first block's Q and tiles 0..2 (ring slots 0..2); the second block's Q in run_block
    {
        // the first block's Q, then its tiles 0..2 (ring slots 0..2)
        const unsigned long long gq = (unsigned long long)(uintptr_t)Q8 + (bh * S + (unsigned long long)qbA * BM) * D_QK;
        const unsigned long long gqs = (unsigned long long)(uintptr_t)QS + (bh * S + (unsigned long long)qbA * BM) * 8;
        tdm_load(tdm_g0(lds0 + Q_OFF, gq, wv == 1 ? 1 : 0), tdm_g1(W0_KN, BM, 128, BM, D_QK));
        tdm_load(tdm_g0(lds0 + Q_OFF + 128, gq + 128, wv == 2 ? 1 : 0), tdm_g1(W0_KR, BM, 64, BM, D_QK));
        tdm_load(tdm_g0(lds0 + QS_OFF, gqs, wv == 3 ? 1 : 0), tdm_g1(W0_S, BM, 8, BM, 8));
    }
    issue_slot(0);
    issue_slot(1);
    issue_slot(2);
    // one instantiation of the block body, run once or twice (two calls would duplicate it: ~260 VGPR spills)
    const int nblk = qbB >= 0 ? 2 : 1;
    for (int blk = 0; blk < nblk; ++blk) run_block(blk == 0 ? qbA : qbB, blk == 0 ? 0 : nA);
    tensor_wait<0>();
}
