###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""DeepSeek-V3 MLA attention on MXFP8 operands for gfx1250 (MI455X): HIP kernels (development package).

``mx.attn_fwd`` / ``mx.attn_bwd`` run the quantization kernels and the MX attention kernels; the kernels are compiled
from ``hip/`` at first use (amdclang++) into a per-user cache. See benchmark/hk_mla_mx for the plan, the rungs, the
numerics and the tests."""
