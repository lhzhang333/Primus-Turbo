/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// Primitives of the forward kernel on top of prims.h.
#pragma once
#include "prims.h"

namespace hk_attn {

// TDM group 1 from 8 precomputed words (fwd_idx.h G1).
HKA_DEV u32x8 tdm_g1_words(const uint32_t (&w)[8]) {
    return u32x8{w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]};
}

// v_cmp -> lane mask in an SGPR (wave-uniform), and its uniform test.
HKA_DEV uint32_t ballot(bool p) {
    return __builtin_amdgcn_ballot_w32(p);
}
HKA_DEV bool any_set(uint32_t m) {
    return m != 0u;
}

// v_log_f32 (no denormal handling) and the correctly rounded IEEE reciprocal.
HKA_DEV float log2_hw(float x) {
    return __builtin_amdgcn_logf(x);
}
HKA_DEV float rcp_ieee(float x) {
    return 1.0f / x;
}

// v_pk_{fma,mul,add}_f32 on f32x2, v_max3_num_f32, and the f32x2 of accumulator elements 2J, 2J + 1
// (adjacent VGPRs, no move).
HKA_DEV f32x2 pk_fma(f32x2 a, f32x2 b, f32x2 c) {
    return __builtin_elementwise_fma(a, b, c);
}
HKA_DEV f32x2 pk_mul(f32x2 a, f32x2 b) {
    return a * b;
}
HKA_DEV f32x2 pk_add(f32x2 a, f32x2 b) {
    return a + b;
}
HKA_DEV float max3(float a, float b, float c) {
    return __builtin_fmaxf(__builtin_fmaxf(a, b), c);
}
template <int J> HKA_DEV f32x2 acc_pair(acc_t c) {
    static_assert(J >= 0 && J < 4, "accumulator pair index");
    return f32x2{c[2 * J], c[2 * J + 1]};
}

} // namespace hk_attn
