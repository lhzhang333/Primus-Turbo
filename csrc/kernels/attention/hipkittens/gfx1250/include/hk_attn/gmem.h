/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// Global memory accesses of the attention kernels other than TDM (tdm.h): raw buffer loads, stores
// and fp32 atomics on buffer resources that carry the true byte extent of the tensor they address.
// An access at or above the extent is dropped by the hardware (a load returns 0); the kernels place
// every extent on an access boundary and keep each range-checked offset in voffset.
#pragma once
#include "types.h"

namespace hk_attn {

// make.buffer.rsrc flags operand: 0, the default range-check mode.
inline constexpr int RSRC_FLAGS = 0;
static_assert((RSRC_FLAGS & 0xF) == 0, "V# word-3 bits 31:28 must stay 0");

// Buffer resource over [base, base + nbytes) (the backend packs nbytes into the 45-bit num_records
// field exactly). base and nbytes must be wave-uniform.
HKA_DEV rsrc_t make_rsrc(const void *base, int64_t nbytes) {
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<void *>(base), (short) 0, (long) nbytes,
                                             RSRC_FLAGS);
}

HKA_DEV u32x4 buf_load_b128(rsrc_t r, int voff, int soff) {
    return __builtin_amdgcn_raw_buffer_load_b128(r, voff, soff, 0);
}
HKA_DEV f32x4 buf_load_f32x4(rsrc_t r, int voff, int soff) {
    return __builtin_bit_cast(f32x4, buf_load_b128(r, voff, soff));
}
HKA_DEV void buf_store_b32(rsrc_t r, int voff, int soff, uint32_t v) {
    __builtin_amdgcn_raw_buffer_store_b32(v, r, voff, soff, 0);
}
HKA_DEV void buf_store_b128(rsrc_t r, int voff, int soff, u32x4 v) {
    __builtin_amdgcn_raw_buffer_store_b128(v, r, voff, soff, 0);
}
HKA_DEV void buf_store_f32(rsrc_t r, int voff, int soff, float v) {
    buf_store_b32(r, voff, soff, __builtin_bit_cast(uint32_t, v));
}

// buffer_atomic_add_f32 without return, device scope (CPOL scope bits 4:3 = 16).
inline constexpr int CPOL_SCOPE_DEV = 16;
HKA_DEV void         buf_atomic_add_f32(rsrc_t r, int voff, int soff, float v) {
    (void) __builtin_amdgcn_raw_ptr_buffer_atomic_fadd_f32(v, r, voff, soff, CPOL_SCOPE_DEV);
}
// Declares 0 <= v, so the backend can fold a constant added to v into the instruction's offset
// field (it must prove the sum does not wrap). Never pass a negative voffset.
HKA_DEV int voff_nonneg(int v) {
    __builtin_assume(v >= 0);
    return v;
}
// Atomically adds the 8 elements of an accumulator in C-fragment order: element i to byte
// voff + i * SLOT_B. With voff = 4 * lane + a 128-B aligned base every instruction covers one
// aligned 128-B line, and i * SLOT_B lands in the offset field.
template <int SLOT_B> HKA_DEV void atomic_add_c(rsrc_t r, int voff, int soff, acc_t c) {
    static_assert(SLOT_B >= 0 && 7 * SLOT_B < (1 << 23), "element offsets fit the offset field");
    const int v = voff_nonneg(voff);
    static_for<8>([&](auto i) { buf_atomic_add_f32(r, v + (int) i * SLOT_B, soff, c[(int) i]); });
}

// C^T as bf16 straight from registers: lane (r, h) stores bf16(c[0..7]) (RNE) as 16 bytes at
// voff, which the caller sets to row n0 + r, columns m0 + 8h of the destination.
HKA_DEV void buf_store_ct_bf16(rsrc_t r, int voff, int soff, acc_t c) {
    buf_store_b128(r, voff, soff, cvt_pk_bf16(c));
}

} // namespace hk_attn
