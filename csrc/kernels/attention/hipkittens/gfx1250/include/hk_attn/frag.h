/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// Register-only work on WMMA accumulators: C -> operand pack, causal masks, cross-half exchange and
// exp2. Lane (r, h) element i of an accumulator holds C[8h + i][r] (layout.h).
#pragma once
#include "types.h"

namespace hk_attn {

// c0 = rows [0, 16), c1 = rows [16, 32) of a 32 x 16 fp32 matrix X whose rows are the next GEMM's
// contraction index -> the operand with element e of lane (r, h) = bf16(X[op_k(l, e)][r]): B = X as
// a B operand, A = X^T as an A operand (4 v_cvt_pk_bf16_f32 per accumulator, RNE).
HKA_DEV opnd_t pack_k32(acc_t c0, acc_t c1) {
    return cat_u32(cvt_pk_bf16(c0), cvt_pk_bf16(c1));
}
// Forward: the S^T accumulators of kv rows [0, 16) and [16, 32) of a chunk (lane = query) -> the
// P^T B operand of O^T += V^T P^T.
HKA_DEV opnd_t pack_pt_fwd(acc_t st_kv_lo, acc_t st_kv_hi) {
    return pack_k32(st_kv_lo, st_kv_hi);
}

// Causal masks (v_cmp + v_cndmask per element, branch-free): masked elements become `neg`.
// The forward masks S^T with -inf before the row max (its running max starts finite). The backward
// masks the exp2 argument after the fma with the finite MASK_NEG: masking S first would turn a
// query row that sees no key (LSE = -inf, possible when Sq > Skv) into NaN.
inline constexpr float MASK_NEG = -3.0e38f;
HKA_DEV acc_t          mask_lt(acc_t c, int thr, float neg) {
    static_for<8>([&](auto i) { c[(int) i] = ((int) i < thr) ? neg : c[(int) i]; });
    return c;
}
HKA_DEV acc_t mask_gt(acc_t c, int thr, float neg) {
    static_for<8>([&](auto i) { c[(int) i] = ((int) i > thr) ? neg : c[(int) i]; });
    return c;
}
// Tile-local column n > row m + off -> neg (off wave-uniform). Bottom-right causal on S = Q K^T
// (rows q from m0, columns kv from n0): off = m0 + Skv - Sq - n0.
HKA_DEV acc_t mask_col_gt_row(acc_t c, int lane, int off, float neg = -__builtin_inff()) {
    return mask_lt(c, lay::thr_cgr(lane, off), neg);
}
// Tile-local row m > column n + off -> neg. Bottom-right causal on S^T = K Q^T (rows kv from m0,
// columns q from n0): off = n0 + Skv - Sq - m0.
HKA_DEV acc_t mask_row_gt_col(acc_t c, int lane, int off, float neg = -__builtin_inff()) {
    return mask_gt(c, lay::thr_rgc(lane, off), neg);
}

// The value of lane l ^ 16 (v_permlanex16_b32 with the identity selects).
HKA_DEV float xhalf(float v) {
    const int x = __builtin_bit_cast(int, v);
    return __builtin_bit_cast(
        float, __builtin_amdgcn_permlanex16(x, x, 0x76543210, (int) 0xFEDCBA98u, false, false));
}
// The value of lane l ^ M within a 16-lane row (DPP row_xmask:M).
template <int M> HKA_DEV float xor16(float v) {
    static_assert(M > 0 && M < 16, "row_xmask stays inside a 16-lane row");
    return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v),
                                                                 0x160 | M, 0xF, 0xF, false));
}

// v_exp_f32 (no denormal range reduction); exp2(-inf) = 0.
HKA_DEV float exp2_hw(float x) {
    return __builtin_amdgcn_exp2f(x);
}

} // namespace hk_attn
