###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

"""Static configs of the gfx1250 MXFP8 GEMMs (``gemm_mxfp8_kernel_gfx1250``): dense per (M, N, K), the routed
experts' grouped-M and variable-K modes per shape class. Measured on MI455X (DSv3 shapes); no runtime autotune."""

from typing import NamedTuple

DENSE, GROUPED_M, VARIABLE_K = 0, 1, 2  # the GEMM modes
_GRID_CUS = 256  # MI455X: the CU count of pick_config's grid-fill rule


class MXGemmConfig(NamedTuple):
    """One kernel instance: a ``tile_m x tile_n`` block per workgroup of 2x2 waves, ``num_buffers`` LDS slots of
    ``kpair`` 128-wide K steps each, and a ``cluster_m x cluster_n`` workgroup cluster that multicasts the shared A
    (along N) and B (along M) tiles (1x1: none). ``persist``: persistent workgroups, which load the next tile during
    the stores; it needs 2 slots of kpair 2 (else the call raises), and a call with fp32 out, or beta = 1 where the
    early old-C fetch does not fit, runs non-persistent. ``c_pre`` (persistent beta = 1 into bf16): the old C's first
    rows come during the previous tile's epilogue; ignored elsewhere."""

    tile_m: int = 256
    tile_n: int = 256
    num_buffers: int = 2
    kpair: int = 2
    cluster_m: int = 2
    cluster_n: int = 2
    persist: bool = False
    c_pre: bool = False


DEFAULT = MXGemmConfig()

C = MXGemmConfig
# (M, N, K) -> the fastest config measured with packed scales and a row-major B, DSv3 at T = 8192 tokens: fwd
# (T, N, K), dgrad (T, K, N), wgrad (N, K, T); routed experts: one expert of 2048 rows.
_TABLE: dict = {
    # MLA projections
    (8192, 1536, 7168): C(256, 256, 2, 2, 2, 2),  # q_down fwd
    (8192, 7168, 1536): C(256, 256, 2, 2, 2, 2),  # q_down dgrad
    (1536, 7168, 8192): C(256, 256, 4, 1, 1, 1),  # q_down wgrad
    (8192, 24576, 1536): C(256, 256, 2, 2, 4, 1, persist=True),  # q_up fwd (many tiles, short K: persistent)
    (8192, 1536, 24576): C(256, 256, 2, 2, 2, 2),  # q_up dgrad
    (24576, 1536, 8192): C(256, 256, 2, 2, 2, 2),  # q_up wgrad
    (8192, 576, 7168): C(256, 128, 3, 2, 1, 1),  # kv_down fwd
    (8192, 7168, 576): C(256, 256, 2, 2, 4, 1),  # kv_down dgrad
    (576, 7168, 8192): C(128, 256, 3, 2, 1, 4),  # kv_down wgrad
    (8192, 32768, 512): C(256, 256, 2, 2, 4, 1, persist=True),  # kv_up fwd (many tiles, short K: persistent)
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


# With a 16x16-preshuffled B: where the deep kpair-1 pipeline, or a larger multicast cluster (the mainloop is bound by
# the global -> LDS traffic, most of it A's), beats the row-major table's config.
_TABLE_BSH: dict = {
    (8192, 1536, 7168): C(256, 256, 4, 1, 2, 2),  # q_down fwd
    (1536, 7168, 8192): C(256, 256, 4, 1, 2, 2),  # q_down wgrad
    (8192, 1536, 24576): C(256, 256, 4, 1, 2, 2),  # q_up dgrad
    (24576, 1536, 8192): C(256, 256, 4, 1, 2, 2),  # q_up wgrad
    (8192, 7168, 576): C(256, 256, 3, 1, 2, 2),  # kv_down dgrad
    (8192, 7168, 16384): C(256, 256, 4, 1, 2, 4),  # o_proj fwd
    (8192, 16384, 7168): C(256, 256, 4, 1, 4, 4),  # o_proj dgrad
    (7168, 2048, 8192): C(256, 256, 4, 1, 1, 4),  # shared fc2 wgrad
    (8192, 7168, 4096): C(256, 256, 4, 1, 2, 2),  # shared fc1 dgrad
    (8192, 2048, 7168): C(256, 256, 4, 1, 4, 4),  # shared fc2 dgrad
    (8192, 4096, 7168): C(256, 256, 4, 1, 4, 4),  # shared fc1 fwd
}


# beta = 1 into a bf16 out (a wgrad into main_grad): kpair 2 for the old C fetched during the last K step, persistent
# with the old C's first rows fetched during the previous tile's epilogue (c_pre).
_TABLE_ACC: dict = {
    (1536, 7168, 8192): C(256, 256, 2, 2, 2, 2, persist=True, c_pre=True),  # q_down wgrad
    (24576, 1536, 8192): C(256, 256, 2, 2, 1, 4, persist=True, c_pre=True),  # q_up wgrad
    (7168, 16384, 8192): C(256, 256, 2, 2, 2, 2, persist=True, c_pre=True),  # o_proj wgrad
    (7168, 2048, 8192): C(256, 256, 2, 2, 2, 2, persist=True, c_pre=True),  # shared fc2 wgrad
}


def pick_config(M: int, N: int, K: int, b_preshuffled: bool = False, beta1: bool = False) -> MXGemmConfig:
    """The measured config of (M, N, K), else by grid fill: 256x256 while it gives at least ~3/4 of a wave, then
    128x256, then 128x128 (two workgroups per CU). The entry point shrinks the cluster to what the grid divides.
    ``beta1`` (accumulating into a bf16 out): the old C fetched early wants kpair 2, so a kpair-1 pick becomes the
    persistent c2x2 with c_pre."""
    cfg = beta1 and _TABLE_ACC.get((M, N, K))
    if cfg:
        return cfg
    cfg = _pick(M, N, K, b_preshuffled)
    if beta1 and cfg.kpair == 1:
        return MXGemmConfig(256, 256, 2, 2, 2, 2, persist=True, c_pre=True)
    return cfg


def _pick(M: int, N: int, K: int, b_preshuffled: bool) -> MXGemmConfig:
    cfg = (b_preshuffled and _TABLE_BSH.get((M, N, K))) or _TABLE.get((M, N, K))
    if cfg is not None:
        return cfg
    tiles = lambda tm, tn: -(-M // tm) * -(-N // tn)  # noqa: E731
    if tiles(256, 256) >= 3 * _GRID_CUS // 4:
        # long K with a preshuffled B: the deep kpair-1 pipeline
        return MXGemmConfig(256, 256, 4, 1, 2, 2) if b_preshuffled and K >= 4096 else DEFAULT
    if tiles(128, 256) >= 3 * _GRID_CUS // 4:
        return MXGemmConfig(128, 256, 3, 2, 1, 4)
    return MXGemmConfig(128, 128, 2, 2, 1, 1)


def pick_grouped_config(
    mode: int, N: int, K: int, b_preshuffled: bool = True, bf16_out: bool = True
) -> MXGemmConfig:
    """Routed experts. ``GROUPED_M`` (fwd / dgrad, ``N`` x ``K`` per expert): with a preshuffled B and K >= 2048 the
    deep kpair-1 pipeline in clusters of 2 m x 4 n tiles (B multicast inside an expert), else persistent kpair 2.
    ``VARIABLE_K`` (wgrad, ``K`` = tokens per expert): persistent with the early old-C rows (c_pre), in 2x2 clusters
    except a narrow N into bf16 (N <= 2048: 1x1)."""
    if mode == GROUPED_M:
        if b_preshuffled and K >= 2048:
            return MXGemmConfig(256, 256, 4, 1, 2, 4)
        return MXGemmConfig(256, 256, 2, 2, 1, 4, persist=True)
    if mode != VARIABLE_K:
        raise ValueError(f"pick_grouped_config: mode {mode} is not GROUPED_M or VARIABLE_K")
    c = 1 if N <= 2048 and bf16_out else 2
    return MXGemmConfig(256, 256, 2, 2, c, c, persist=True, c_pre=True)
