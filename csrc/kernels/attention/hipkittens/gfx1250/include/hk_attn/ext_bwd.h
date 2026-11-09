/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// Primitives of the backward kernels on top of prims.h. In the backward's S = Q K^T orientation an
// accumulator row is a query, so the per-row terms (-LSE log2e, delta) arrive as two f32x4 per
// accumulator (rows 8h + 0..3 and 8h + 4..7) and pair (2j, 2j + 1) of the accumulator meets pair
// (2j % 4, 2j % 4 + 1) of an f32x4: one register pair per v_pk_* operand.
#pragma once
#include "frag.h"
#include "lds.h"
#include "sync.h"
#include "types.h"

namespace hk_attn {

namespace detail {
// element pair j (0..3) of a row-term pair (lo = rows 0..3, hi = rows 4..7)
HKA_DEV f32x2 row_pair(f32x4 lo, f32x4 hi, int j) {
    const f32x4 v = j < 2 ? lo : hi;
    return (j & 1) ? f32x2{v[2], v[3]} : f32x2{v[0], v[1]};
}
} // namespace detail

// exp2 argument of the backward softmax: t[i] = fma(s[i], c1, nl[row i]) with c1 = scale * log2(e)
// and nl = -LSE * log2(e) (4 x v_pk_fma_f32).
HKA_DEV acc_t softmax_arg_rows(acc_t s, float c1, f32x4 nl_lo, f32x4 nl_hi) {
    const f32x2 c2 = {c1, c1};
    acc_t       t;
    static_for<4>([&](auto j) {
        const f32x2 a = __builtin_elementwise_fma(f32x2{s[2 * j], s[2 * j + 1]}, c2,
                                                  detail::row_pair(nl_lo, nl_hi, j));
        t[2 * j]      = a[0];
        t[2 * j + 1]  = a[1];
    });
    return t;
}
// p[i] = exp2(t[i]) (8 x v_exp_f32; exp2(MASK_NEG) = +0).
HKA_DEV acc_t exp2_acc(acc_t t) {
    acc_t p;
    static_for<8>([&](auto i) { p[(int) i] = exp2_hw(t[(int) i]); });
    return p;
}
// dS = p * (dp - delta[row i]), unscaled (4 x v_pk_add_f32 + 4 x v_pk_mul_f32).
HKA_DEV acc_t dsoftmax(acc_t p, acc_t dp, f32x4 de_lo, f32x4 de_hi) {
    acc_t d;
    static_for<4>([&](auto j) {
        const f32x2 t = f32x2{dp[2 * j], dp[2 * j + 1]} - detail::row_pair(de_lo, de_hi, j);
        const f32x2 r = f32x2{p[2 * j], p[2 * j + 1]} * t;
        d[2 * j]      = r[0];
        d[2 * j + 1]  = r[1];
    });
    return d;
}
// c * s per element (4 x v_pk_mul_f32).
HKA_DEV acc_t scale_acc(acc_t c, float s) {
    const f32x2 s2 = {s, s};
    acc_t       o;
    static_for<4>([&](auto j) {
        const f32x2 r = f32x2{c[2 * j], c[2 * j + 1]} * s2;
        o[2 * j]      = r[0];
        o[2 * j + 1]  = r[1];
    });
    return o;
}
HKA_DEV u32x4 zero_u32x4() {
    return u32x4{0u, 0u, 0u, 0u};
}

HKA_DEV void ds_store_b64(lds_u8 *base, int off, u32x2 v) {
    *(__attribute__((address_space(3))) u32x2 *) (base + off) = v;
}
HKA_DEV void ds_store_b32(lds_u8 *base, int off, uint32_t v) {
    *(__attribute__((address_space(3))) uint32_t *) (base + off) = v;
}

// Split-barrier signal of a full backward step: retire all but this wave's TW newest TDM ops and
// all but its N newest LDS ops, then s_barrier_signal. The N newest LDS ops must be loads outside
// the barrier protocol (fully used ds_load_b128 / tr16, kept younger than the protocol stores by
// order_fence) that read no bytes another wave writes after the barrier; atomics and global stores
// stay in flight.
template <int TW, int N> HKA_DEV void split_signal_tdm_dscnt() {
    static_assert(TW >= 0 && N >= 0 && N < 64, "dscnt is a 6-bit field");
    sched_barrier<0>();
    compiler_fence();
    wait_tensorcnt<TW>();
    wait_dscnt<N>();
    barrier_signal();
    compiler_fence();
    sched_barrier<0>();
}

} // namespace hk_attn
