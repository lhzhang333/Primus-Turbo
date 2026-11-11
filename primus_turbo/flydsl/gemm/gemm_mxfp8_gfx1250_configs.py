###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Static per-shape configs of the gfx1250 MXFP8 dense GEMM (``gemm_mxfp8_kernel_gfx1250``). No runtime autotune."""

from typing import NamedTuple


class MXGemmConfig(NamedTuple):
    """One kernel instance: a ``tile_m x tile_n`` block per workgroup of 2x2 waves, ``num_buffers`` LDS slots of
    ``kpair`` 128-wide K steps each, and a ``cluster_m x cluster_n`` workgroup cluster that multicasts the shared
    A (along N) and B (along M) tiles; 1x1 is the no-multicast config. ``scale_4d`` reads the scales in aiter's
    ``[R/32][K/128][32][4]`` layout instead of the raw ``[R, K/32]`` one; ``b_preshuffle`` reads B in aiter's 16x16
    preshuffled layout (lab A/B options)."""

    tile_m: int = 256
    tile_n: int = 256
    num_buffers: int = 2
    kpair: int = 2
    cluster_m: int = 1
    cluster_n: int = 4
    scale_4d: bool = False
    b_preshuffle: bool = False


DEFAULT = MXGemmConfig()

# (M, N, K) -> config, measured on MI455X (scratch_mxd/results/k1_*.md).
_TABLE: dict = {}


def pick_config(M: int, N: int, K: int) -> MXGemmConfig:
    cfg = _TABLE.get((M, N, K))
    if cfg is not None:
        return cfg
    if N <= 1024 or M <= 1024:
        return MXGemmConfig(cluster_n=1)
    return DEFAULT
