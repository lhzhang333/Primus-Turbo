/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// Lane maps of the gfx1250 attention primitives, as constexpr functions of scalars; the device
// wrappers (lds.h, frag.h, gmem.h) compute their addresses with exactly these.
//
// Wave32: lane l in [0, 32), r = l % 16 (the operand / accumulator lane row), h = l / 16 (the lane
// half). v_wmma_f32_16x16x32_bf16 computes D[16x16] = A[16x32] B[32x16] + C with
//   A operand (8 VGPRs, bf16x16): lane (r, h) element e holds A[r][op_k(l, e)]
//   B operand (8 VGPRs, bf16x16): lane (r, h) element e holds B[op_k(l, e)][r]
//   C / D     (8 VGPRs, f32x8):   lane (r, h) element i holds C[8h + i][r]
//   op_k(l, e) = (e / 8) * 16 + 8h + e % 8
// so two C tiles stacked along M (rows 0..15, 16..31) are a B operand as they are (frag.h
// pack_k32): one v_cvt_pk_bf16_f32 per element pair, no data movement.
#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>

#define HKA_HD static inline constexpr __host__ __device__

namespace hk_attn {
namespace lay {

inline constexpr int WAVE = 32;

// "rows" operand loader (lds.h load_rows, 2 x ds_load_b128): the LDS tile is row-major with a
// pitch of `pitch` bytes and its rows are the operand's lane dimension. Lane (r, h) reads row r0 +
// r, bytes 2 k0 + 16h + [0, 16) (elements 0..7) and 32 bytes further (elements 8..15):
//   element e = T[r0 + r][k0 + op_k(l, e)]
HKA_HD int rows_lane(int lane, int pitch) {
    return (lane % 16) * pitch + (lane / 16) * 16;
}
HKA_HD int rows_imm(int pitch, int r0, int k0, int half) {
    return r0 * pitch + 2 * k0 + 32 * half;
}
// "tr" operand loader (lds.h load_tr, 2 x ds_load_tr16_b128): the LDS tile's rows are the
// contraction index. Lane l addresses row k0 + 8h + l % 8 (+ 16 for the second load), bytes
// 2 r0 + 16 ((l / 8) % 2); the instruction transposes each group of 8 lanes, so
//   element e = T[k0 + op_k(l, e)][r0 + r]
HKA_HD int tr_lane(int lane, int pitch) {
    return ((lane / 16) * 8 + lane % 8) * pitch + ((lane / 8) % 2) * 16;
}
HKA_HD int tr_imm(int pitch, int k0, int r0, int half) {
    return (k0 + 16 * half) * pitch + 2 * r0;
}
// Causal-mask thresholds on an accumulator (frag.h): tile-local row m = 8h + i, column n = r, `off`
// wave-uniform.
//   mask_col_gt_row: masked iff n > m + off  <=>  i < thr_cgr
//   mask_row_gt_col: masked iff m > n + off  <=>  i > thr_rgc
HKA_HD int thr_cgr(int lane, int off) {
    return lane % 16 - 8 * (lane / 16) - off;
}
HKA_HD int thr_rgc(int lane, int off) {
    return lane % 16 - 8 * (lane / 16) + off;
}

// Word 0 of TDM descriptor group 1 (tdm.h): bits 17:16 log2(element bytes), bit 20 pad enable,
// 24:22 log2(pad interval in dwords) - 1, 31:25 pad dwords - 1. The interval is the tile row
// (a power of two, 8..1024 B) and the pad is the LDS pitch minus the row (a multiple of 4 B,
// <= 512 B).
HKA_HD int ilog2(int x) {
    return x <= 1 ? 0 : 1 + ilog2(x / 2);
}
HKA_HD bool tdm_tile_ok(int tile_cols, int pitch_b, int elem_b) {
    const int  row_b = tile_cols * elem_b, pad_b = pitch_b - row_b;
    const bool pow2 = row_b >= 8 && row_b <= 1024 && (row_b & (row_b - 1)) == 0;
    const bool esz  = elem_b == 1 || elem_b == 2 || elem_b == 4 || elem_b == 8;
    return esz && pow2 && pad_b >= 0 && pad_b % 4 == 0 && pad_b <= 512;
}
HKA_HD uint32_t tdm_w0(int tile_cols, int pitch_b, int elem_b) {
    const int      row_b = tile_cols * elem_b, pad_b = pitch_b - row_b;
    const uint32_t ds = (uint32_t) ilog2(elem_b) << 16;
    return pad_b == 0 ? ds
                      : ds | (1u << 20) | ((uint32_t) (ilog2(row_b / 4) - 1) << 22) |
                            ((uint32_t) (pad_b / 4 - 1) << 25);
}

} // namespace lay
} // namespace hk_attn
