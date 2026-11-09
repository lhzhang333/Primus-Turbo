/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// v_wmma_f32_16x16x32_bf16, the only matrix instruction of the attention kernels:
//   D[m][n] = sum_{k < 32} A[m][k] * B[k][n] + C[m][n]    (fp32 accumulation of bf16 products)
// Orientations (lanes index the C column):
//   fwd  S^T = K Q^T      A = K rows             B = Q rows as B^T          C lane = q, rows = kv
//   fwd  O^T += V^T P^T   A = V^T (tr16 of V)    B = pack_k32(S^T tiles)    C lane = q, rows = d
//   bwd  S = Q K^T        A = Q rows             B = K rows as B^T          C lane = kv, rows = q
//   bwd  dV^T += dO^T P   A = dO^T (tr16 of dO)  B = P packed from S        C lane = kv, rows = d
//   bwd  dQ = dS K        A = dS (tr16 of dS^T)  B = K (tr16 of K)          C lane = d, rows = q
#pragma once
#include "types.h"

namespace hk_attn {

template <bool REUSE_A = false, bool REUSE_B = false>
HKA_DEV acc_t wmma(opnd_t a, opnd_t b, acc_t c) {
    return __builtin_amdgcn_wmma_f32_16x16x32_bf16(false, a, false, b, (short) 0, c, REUSE_A,
                                                   REUSE_B);
}

// D = A B (the zero accumulator is an inline constant).
template <bool REUSE_A = false, bool REUSE_B = false> HKA_DEV acc_t wmma0(opnd_t a, opnd_t b) {
    return wmma<REUSE_A, REUSE_B>(a, b, zero_acc());
}

} // namespace hk_attn
