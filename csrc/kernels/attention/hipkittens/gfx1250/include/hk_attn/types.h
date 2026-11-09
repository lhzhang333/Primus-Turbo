/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// Register types and small helpers of the gfx1250 (MI455X, wave32) attention primitives.
#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <utility>

#include "layout.h"

namespace hk_attn {

typedef __bf16                 bf16;
typedef __bf16                 bf16x2 __attribute__((ext_vector_type(2)));
typedef __bf16                 bf16x8 __attribute__((ext_vector_type(8)));
typedef __bf16                 bf16x16 __attribute__((ext_vector_type(16)));
typedef float                  f32x2 __attribute__((ext_vector_type(2)));
typedef float                  f32x4 __attribute__((ext_vector_type(4)));
typedef float                  f32x8 __attribute__((ext_vector_type(8)));
typedef uint32_t               u32x2 __attribute__((ext_vector_type(2)));
typedef uint32_t               u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t               u32x8 __attribute__((ext_vector_type(8)));
typedef int32_t                i32x4 __attribute__((ext_vector_type(4)));
typedef int32_t                i32x8 __attribute__((ext_vector_type(8)));
typedef __amdgpu_buffer_rsrc_t rsrc_t;
typedef __attribute__((address_space(3))) unsigned char lds_u8;
typedef __attribute__((address_space(3))) u32x4         lds_u32x4;
typedef __attribute__((address_space(3))) bf16x8        lds_bf16x8;

// A or B operand of v_wmma_f32_16x16x32_bf16 (8 VGPRs) and its C / D accumulator (8 VGPRs); the
// lane maps are in layout.h.
typedef bf16x16 opnd_t;
typedef f32x8   acc_t;

#define HKA_DEV __device__ __forceinline__

// Compile-time loop: f(std::integral_constant<int, I>{}) for I in [0, N).
template <typename F, int... I>
HKA_DEV void static_for_impl(F &&f, std::integer_sequence<int, I...>) {
    (f(std::integral_constant<int, I>{}), ...);
}
template <int N, typename F> HKA_DEV void static_for(F &&f) {
    static_for_impl(static_cast<F &&>(f), std::make_integer_sequence<int, N>{});
}

// Operand concatenation: a register rename, no instruction when the halves are adjacent VGPRs.
HKA_DEV bf16x16 cat(bf16x8 lo, bf16x8 hi) {
    return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
}
HKA_DEV bf16x16 cat_u32(u32x4 lo, u32x4 hi) {
    return __builtin_bit_cast(bf16x16, __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7));
}
HKA_DEV acc_t zero_acc() {
    return acc_t{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
}

// fp32 -> bf16 with round to nearest even (v_cvt_pk_bf16_f32), packed per element pair: dword j =
// bf16(c[2j]) | bf16(c[2j + 1]) << 16. The pairs stay in the integer domain until an operand is
// formed (cat_u32); a vector fptrunc followed by the concat would be rewritten into one conversion
// at the head of the WMMA run that consumes it.
HKA_DEV uint32_t cvt_pk_bf16(float lo, float hi) {
    return __builtin_bit_cast(uint32_t, __builtin_convertvector(f32x2{lo, hi}, bf16x2));
}
HKA_DEV u32x4 cvt_pk_bf16(acc_t c) {
    return u32x4{cvt_pk_bf16(c[0], c[1]), cvt_pk_bf16(c[2], c[3]), cvt_pk_bf16(c[4], c[5]),
                 cvt_pk_bf16(c[6], c[7])};
}

// Lane index in the wave (blocks are 1-D).
HKA_DEV int lane_id() {
    return (int) __builtin_amdgcn_workitem_id_x() & (lay::WAVE - 1);
}
// Wave index in the workgroup as a uniform SGPR. The unsigned shift tells LLVM the value is
// non-negative where it feeds a buffer offset.
HKA_DEV int wave_id() {
    return (
        int) ((uint32_t) __builtin_amdgcn_readfirstlane((int) __builtin_amdgcn_workitem_id_x()) >>
              5);
}

} // namespace hk_attn
