/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// Tensor Data Mover: a 2-D row-major tile from global memory into a padded LDS image
// (tensor_load_to_lds; descriptor groups 2..4 zero) and s_wait_tensorcnt.
//
// Descriptor group 0 (4 SGPRs): w0 = 1 (issue), w1 = LDS byte address, w2 / w3 = global byte
// address [31:0] / [56:32] | type 2 << 30. Group 1 (8 SGPRs): w0 = element size / padding (layout.h
// tdm_w0), w1 = tensor_dim0[15:0] << 16, w2 = tensor_dim0[31:16] | tensor_dim1[15:0] << 16, w3 =
// tensor_dim1[31:16] | tile_dim0 << 16, w4 = tile_dim1, w5 / w6 = row stride (elements, 48 bits),
// w7 = 0. dim0 = columns, dim1 = rows, measured from the tile origin: an element outside the tensor
// extents is not fetched and lands in LDS as zero. Row t of the tile lands at lds + t * pitch.
// Every descriptor word must be wave-uniform; the ops of one wave complete in order, so
// s_wait_tensorcnt N retires all but the N newest.
#pragma once
#include "types.h"

namespace hk_attn {

// A tile of TILE_ROWS x TILE_COLS elements of ELEM_B bytes landing in LDS with a row pitch of
// PITCH_B bytes.
template <int TILE_ROWS, int TILE_COLS, int PITCH_B, int ELEM_B = 2> struct tdm_tile {
    static_assert(
        lay::tdm_tile_ok(TILE_COLS, PITCH_B, ELEM_B),
        "TDM tile: row bytes a power of two in [8, 1024], pad a multiple of 4 B <= 512 B");
    static_assert(TILE_ROWS > 0 && TILE_ROWS < 65536, "tile_dim1 is 16 bits");
    static constexpr int      rows = TILE_ROWS, cols = TILE_COLS, pitch = PITCH_B;
    static constexpr int      bytes = TILE_ROWS * PITCH_B; // LDS footprint
    static constexpr uint32_t w0    = lay::tdm_w0(TILE_COLS, PITCH_B, ELEM_B);
};

// Group 1 of tile T with a row stride of `stride_el` elements and tensor_dim1 = rows_valid clamped
// to [0, T::rows]; tensor_dim0 = T::cols.
template <typename T> HKA_DEV u32x8 tdm_g1(int64_t stride_el, int rows_valid) {
    const uint32_t d0 = (uint32_t) T::cols;
    const uint32_t d1 = (uint32_t) (rows_valid < 0         ? 0
                                    : rows_valid > T::rows ? T::rows
                                                           : rows_valid);
    const uint64_t st = (uint64_t) stride_el;
    return u32x8{T::w0,
                 (d0 & 0xFFFFu) << 16,
                 (d0 >> 16) | ((d1 & 0xFFFFu) << 16),
                 (d1 >> 16) | ((uint32_t) T::cols << 16),
                 (uint32_t) T::rows,
                 (uint32_t) st,
                 (uint32_t) (st >> 32) & 0xFFFFu,
                 0u};
}
// Group 0: one tile at global byte address `gaddr` into LDS byte address `lds`.
HKA_DEV u32x4 tdm_g0(uint32_t lds, uint64_t gaddr, uint32_t pred = 1u) {
    return u32x4{pred, lds, (uint32_t) gaddr,
                 ((uint32_t) (gaddr >> 32) & 0x01FFFFFFu) | (2u << 30)};
}
// tensor_load_to_lds (increments TENSORCNT).
HKA_DEV void tdm_load(u32x4 g0, u32x8 g1) {
    const i32x4 z4 = {0, 0, 0, 0};
    const i32x8 z8 = {0, 0, 0, 0, 0, 0, 0, 0};
    __builtin_amdgcn_tensor_load_to_lds(__builtin_bit_cast(i32x4, g0),
                                        __builtin_bit_cast(i32x8, g1), z4, z4, z8, 0);
}
// s_wait_tensorcnt N: all but the N newest TDM ops of this wave have landed (visible to this wave;
// other waves also need a barrier).
template <int N> HKA_DEV void wait_tensorcnt() {
    __builtin_amdgcn_s_wait_tensorcnt((short) N);
}

} // namespace hk_attn
