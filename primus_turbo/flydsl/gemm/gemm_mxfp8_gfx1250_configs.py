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
    A (along N) and B (along M) tiles; 1x1 is the no-multicast config. ``scale_4d`` / ``b_preshuffle`` describe the
    inputs (the entry point sets them from the scale tensors' rank and its ``b_preshuffled`` flag)."""

    tile_m: int = 256
    tile_n: int = 256
    num_buffers: int = 2
    kpair: int = 2
    cluster_m: int = 2
    cluster_n: int = 2
    scale_4d: bool = False
    b_preshuffle: bool = False


DEFAULT = MXGemmConfig()

C = MXGemmConfig
# (M, N, K) -> fastest measured config with packed scales, DSv3 at T = 8192 (MI455X, scratch_mxd/results/k1_perf.md);
# fwd (T, N, K), dgrad (T, K, N), wgrad (N, K, T); experts: one expert of 2048 rows. Configs within ~2% are noise.
_TABLE: dict = {
    # MLA projections
    (8192, 1536, 7168): C(256, 256, 2, 2, 2, 2),  # q_down fwd
    (8192, 7168, 1536): C(256, 256, 2, 2, 2, 2),  # q_down dgrad
    (1536, 7168, 8192): C(256, 256, 4, 1, 1, 1),  # q_down wgrad
    (8192, 24576, 1536): C(256, 256, 2, 2, 2, 2),  # q_up fwd
    (8192, 1536, 24576): C(256, 256, 2, 2, 2, 2),  # q_up dgrad
    (24576, 1536, 8192): C(256, 256, 2, 2, 2, 2),  # q_up wgrad
    (8192, 576, 7168): C(256, 128, 3, 2, 1, 1),  # kv_down fwd
    (8192, 7168, 576): C(256, 256, 2, 2, 4, 1),  # kv_down dgrad
    (576, 7168, 8192): C(128, 256, 3, 2, 1, 4),  # kv_down wgrad
    (8192, 32768, 512): C(256, 256, 2, 2, 4, 1),  # kv_up fwd
    (8192, 512, 32768): C(128, 128, 4, 2, 2, 2),  # kv_up dgrad
    (32768, 512, 8192): C(256, 256, 2, 2, 2, 2),  # kv_up wgrad
    (8192, 7168, 16384): C(256, 256, 2, 2, 2, 2),  # o_proj fwd
    (8192, 16384, 7168): C(256, 256, 2, 2, 2, 2),  # o_proj dgrad
    (7168, 16384, 8192): C(256, 256, 2, 2, 4, 1),  # o_proj wgrad
    # shared expert
    (8192, 4096, 7168): C(256, 256, 2, 2, 2, 2),  # fc1 fwd
    (8192, 7168, 4096): C(256, 256, 2, 2, 2, 2),  # fc1 dgrad
    (4096, 7168, 8192): C(256, 256, 2, 2, 2, 2),  # fc1 wgrad
    (8192, 7168, 2048): C(256, 256, 2, 2, 2, 2),  # fc2 fwd
    (8192, 2048, 7168): C(256, 256, 2, 2, 1, 4),  # fc2 dgrad
    (7168, 2048, 8192): C(256, 256, 2, 2, 4, 1),  # fc2 wgrad
    # one routed expert (2048 rows)
    (2048, 4096, 7168): C(128, 256, 3, 2, 1, 4),  # fc1 fwd
    (2048, 7168, 4096): C(256, 256, 2, 2, 2, 2),  # fc1 dgrad
    (2048, 7168, 2048): C(256, 256, 2, 2, 2, 2),  # fc2 fwd
    (2048, 2048, 7168): C(128, 128, 4, 2, 2, 2),  # fc2 dgrad
    (4096, 7168, 2048): C(256, 256, 2, 2, 2, 2),  # fc1 wgrad
    (7168, 2048, 2048): C(256, 256, 2, 2, 4, 1),  # fc2 wgrad
}


# With a 16x16-preshuffled B the deep kpair-1 pipelines become usable (row-major B at kpair 1 is a 128-byte TDM row per
# tile row): (M, N, K) -> config where that beats the row-major table (scratch_mxd/results/k1_perf.md).
_TABLE_BSH: dict = {
    (8192, 1536, 7168): C(256, 256, 4, 1, 2, 2),  # q_down fwd    28.2 us (row-major B 31.0)
    (1536, 7168, 8192): C(256, 256, 4, 1, 2, 2),  # q_down wgrad  31.8 (36.2)
    (8192, 1536, 24576): C(256, 256, 4, 1, 2, 2),  # q_up dgrad   88.7 (97.8)
    (24576, 1536, 8192): C(256, 256, 4, 1, 2, 2),  # q_up wgrad   97.5 (101.0)
    (8192, 7168, 576): C(256, 256, 3, 1, 2, 2),  # kv_down dgrad   22.6 (28.6)
    (8192, 7168, 16384): C(256, 256, 4, 1, 1, 4),  # o_proj fwd  263.9 (281.1)
    (7168, 2048, 8192): C(256, 256, 4, 1, 1, 4),  # shared fc2 wgrad 35.6 (36.7)
}


def pick_config(M: int, N: int, K: int, b_preshuffled: bool = False, num_cus: int = 256) -> MXGemmConfig:
    """The measured config of (M, N, K), else by grid fill: 256x256 while it gives at least ~3/4 of a wave, then
    128x256, then 128x128 (two workgroups per CU). The entry point shrinks the cluster to what the grid divides."""
    cfg = (b_preshuffled and _TABLE_BSH.get((M, N, K))) or _TABLE.get((M, N, K))
    if cfg is not None:
        return cfg
    tiles = lambda tm, tn: -(-M // tm) * -(-N // tn)  # noqa: E731
    if tiles(256, 256) >= 3 * num_cus // 4:
        # long K with a preshuffled B: the deep kpair-1 pipeline (2-4% on the 32-expert-as-one-GEMM shapes)
        return MXGemmConfig(256, 256, 4, 1, 2, 2) if b_preshuffled and K >= 4096 else DEFAULT
    if tiles(128, 256) >= 3 * num_cus // 4:
        return MXGemmConfig(128, 256, 3, 2, 1, 4)
    return MXGemmConfig(128, 128, 2, 2, 1, 1)


def pick_grouped_config(mode: int, M: int, N: int, K: int, b_preshuffled: bool = True) -> MXGemmConfig:
    """Routed experts. ``mode`` 1 (grouped-M fwd / dgrad: M = rows per expert, multicast along N only) or 2
    (variable-K wgrad: K = tokens per expert)."""
    if mode == 1:
        return MXGemmConfig(256, 256, 4, 1, 1, 4) if b_preshuffled and K >= 4096 else MXGemmConfig(256, 256, 2, 2, 1, 4)
    return MXGemmConfig(256, 256, 2, 2, 2, 2)
