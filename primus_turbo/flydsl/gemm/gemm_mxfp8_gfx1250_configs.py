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
    A (along N) and B (along M) tiles; 1x1 is the no-multicast config. ``scale_4d`` describes the input (the entry
    point sets it from the scale tensors' rank)."""

    tile_m: int = 256
    tile_n: int = 256
    num_buffers: int = 2
    kpair: int = 2
    cluster_m: int = 2
    cluster_n: int = 2
    scale_4d: bool = False


DEFAULT = MXGemmConfig()

# (M, N, K) -> config, measured on MI455X (scratch_mxd/results/k1_*.md).
_TABLE: dict = {}


def pick_config(M: int, N: int, K: int, num_cus: int = 256) -> MXGemmConfig:
    """The measured config of (M, N, K), else by grid fill: 256x256 while it gives at least ~3/4 of a wave, then
    128x256, then 128x128 (two workgroups per CU). The entry point shrinks the cluster to what the grid divides."""
    cfg = _TABLE.get((M, N, K))
    if cfg is not None:
        return cfg
    tiles = lambda tm, tn: -(-M // tm) * -(-N // tn)  # noqa: E731
    if tiles(256, 256) >= 3 * num_cus // 4:
        return DEFAULT
    if tiles(128, 256) >= 3 * num_cus // 4:
        return MXGemmConfig(128, 256, 3, 2, 1, 4)
    return MXGemmConfig(128, 128, 2, 2, 1, 1)
