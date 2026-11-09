/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// LDS accesses of the attention kernels: ds_load_b128 / ds_store_b128 / ds_load_tr16_b128 and the
// two WMMA operand loaders.
//
// A tile is a row-major bf16 image with a row pitch that is a multiple of 16 B. Bank conflicts: a
// "rows" load of 16 lanes touches 16 rows x one 16-B chunk (conflict-free for an odd multiple of
// 16 B, e.g. 272), a tr16 load touches 8 rows x two adjacent 16-B chunks (conflict-free for an odd
// multiple of 32 B, e.g. 288). A loader is a per-lane base computed once plus a compile-time
// immediate per fragment, so every load of a burst is `ds_load_* vD, vBASE offset:IMM`.
#pragma once
#include "types.h"

// Compile error when a constant fragment coordinate is not a multiple of 8 elements (a misaligned
// b128 / tr16 address would silently break the lane map); runtime arguments are not diagnosed.
#define HKA_MUL8(x, msg) __attribute__((diagnose_if(((x) & 7) != 0, msg, "error")))

namespace hk_attn {

// LDS byte address of a generic pointer to a __shared__ object, and back.
HKA_DEV uint32_t lds_addr(const void *p) {
    return (uint32_t) (uintptr_t) (const lds_u8 *) p;
}
HKA_DEV lds_u8 *lds_ptr(uint32_t a) {
    return (lds_u8 *) (uintptr_t) a;
}
HKA_DEV bf16x8 ds_load_b128(const lds_u8 *base, int off) {
    return *(const lds_bf16x8 *) (base + off);
}
HKA_DEV u32x4 ds_load_b128_u32(const lds_u8 *base, int off) {
    return *(const lds_u32x4 *) (base + off);
}
HKA_DEV void ds_store_b128(lds_u8 *base, int off, u32x4 v) {
    *(lds_u32x4 *) (base + off) = v;
}
// ds_load_tr16_b128: each aligned group of 8 lanes transposes the 8 x 8 bf16 block its lanes
// address (element j of lane 8g + c = element c of the bytes addressed by lane 8g + j).
HKA_DEV bf16x8 ds_load_tr16_b128(const lds_u8 *base, int off) {
    return __builtin_amdgcn_ds_load_tr16_b128_v8bf16((lds_bf16x8 *) (base + off));
}

// "rows" loader (2 x ds_load_b128): the tile's rows are the operand's lane dimension (an A operand
// from A [M][K], a B operand from B^T [N][K]):
//   element e of lane (r, h) = T[r0 + r][k0 + op_k(l, e)]
// rows_base once per tile, then load_rows per fragment with compile-time r0 / k0 (k0 % 8 == 0).
template <int PITCH> HKA_DEV lds_u8 *rows_base(uint32_t tile, int lane) {
    static_assert(PITCH % 16 == 0, "16-B aligned rows");
    return lds_ptr(tile + (uint32_t) lay::rows_lane(lane, PITCH));
}
template <int PITCH>
HKA_DEV opnd_t load_rows(const lds_u8 *base, int r0, int k0)
    HKA_MUL8(k0, "load_rows: k0 % 8 != 0 (16-B aligned rows chunk)") {
    const int off = lay::rows_imm(PITCH, r0, k0, 0);
    return cat(ds_load_b128(base, off), ds_load_b128(base, off + 32));
}

// "tr" loader (2 x ds_load_tr16_b128): the tile's rows are the contraction index (an A operand from
// A^T [K][M], a B operand from B [K][N]); element e of lane (r, h) = T[k0 + op_k(l, e)][r0 + r].
// tr_base once per tile, then load_tr per fragment with compile-time k0 / r0 (r0 % 8 == 0).
template <int PITCH> HKA_DEV lds_u8 *tr_base(uint32_t tile, int lane) {
    static_assert(PITCH % 16 == 0, "16-B aligned rows");
    return lds_ptr(tile + (uint32_t) lay::tr_lane(lane, PITCH));
}
template <int PITCH>
HKA_DEV opnd_t load_tr(const lds_u8 *base, int k0, int r0)
    HKA_MUL8(r0, "load_tr: r0 % 8 != 0 (16-B aligned tr16 chunk)") {
    return cat(ds_load_tr16_b128(base, lay::tr_imm(PITCH, k0, r0, 0)),
               ds_load_tr16_b128(base, lay::tr_imm(PITCH, k0, r0, 1)));
}

} // namespace hk_attn
