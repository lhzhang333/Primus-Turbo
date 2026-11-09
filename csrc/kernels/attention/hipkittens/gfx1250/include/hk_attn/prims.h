/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// The gfx1250 (MI455X, wave32) attention primitive layer: thin wrappers that each lower to the
// instructions named in their comments.
//   layout.h  lane maps of the WMMA operands / accumulators and the LDS loaders
//   types.h   register types (opnd_t, acc_t), static_for, bf16 packing, lane / wave id
//   wmma.h    v_wmma_f32_16x16x32_bf16
//   lds.h     ds_load_b128 / ds_load_tr16_b128 operand loaders, ds_store_b128
//   frag.h    C -> operand pack, causal masks, lane exchange, exp2
//   gmem.h    buffer resources with true extents, buffer loads / stores, fp32 buffer atomics
//   tdm.h     tensor_load_to_lds descriptors, s_wait_tensorcnt
//   sync.h    s_wait_dscnt, split barriers with LDS-only fences, scheduling controls
#pragma once
#include "frag.h"
#include "gmem.h"
#include "layout.h"
#include "lds.h"
#include "sync.h"
#include "tdm.h"
#include "types.h"
#include "wmma.h"
