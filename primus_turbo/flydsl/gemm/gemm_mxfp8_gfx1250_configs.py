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
    inputs (the entry point sets them from the scale tensors' rank and its ``b_preshuffled`` flag). ``persist``:
    persistent workgroups (bf16 out, kpair 2 x 2 slots; the next tile loads during the stores); ``stagger``: the (first wave of) workgroups start in that many phases; ``xcd_remap``: grouped-M m tiles
    grouped per XCD; ``c_pre`` (persistent beta = 1 into bf16 with the early old-C fetch): the old C's first rows (as
    many as the LDS past the slots holds, 80 at 256x256) come during the previous tile's epilogue, only the rest
    during the last K step. They are dropped where they do not apply."""

    tile_m: int = 256
    tile_n: int = 256
    num_buffers: int = 2
    kpair: int = 2
    cluster_m: int = 2
    cluster_n: int = 2
    scale_4d: bool = False
    b_preshuffle: bool = False
    persist: bool = False
    stagger: int = 0
    xcd_remap: bool = False
    c_pre: bool = False


DEFAULT = MXGemmConfig()

C = MXGemmConfig
# (M, N, K) -> fastest measured config with packed scales, DSv3 at T = 8192 (MI455X, scratch_mxd/results/k1_perf.md;
# persistent entries: the in-job A/B scratch_mxd/results/kg3c_pd.json, 11-12); fwd (T, N, K), dgrad (T, K, N),
# wgrad (N, K, T); experts: one expert of 2048 rows. Configs within ~2% are noise.
_TABLE: dict = {
    # MLA projections
    (8192, 1536, 7168): C(256, 256, 2, 2, 2, 2),  # q_down fwd
    (8192, 7168, 1536): C(256, 256, 2, 2, 2, 2),  # q_down dgrad
    (1536, 7168, 8192): C(256, 256, 4, 1, 1, 1),  # q_down wgrad
    (8192, 24576, 1536): C(256, 256, 2, 2, 4, 1, persist=True),  # q_up fwd: 106.0 us (c2x2 115.0, c4x1 111.4)
    (8192, 1536, 24576): C(256, 256, 2, 2, 2, 2),  # q_up dgrad
    (24576, 1536, 8192): C(256, 256, 2, 2, 2, 2),  # q_up wgrad
    (8192, 576, 7168): C(256, 128, 3, 2, 1, 1),  # kv_down fwd
    (8192, 7168, 576): C(256, 256, 2, 2, 4, 1),  # kv_down dgrad
    (576, 7168, 8192): C(128, 256, 3, 2, 1, 4),  # kv_down wgrad
    (8192, 32768, 512): C(256, 256, 2, 2, 4, 1, persist=True),  # kv_up fwd: 74.5 us (not persistent 84.8)
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
# tile row): (M, N, K) -> config where that beats the row-major table (scratch_mxd/results/k1_perf.md). q_down /
# kv_down re-checked 11-12 in one job against the row-major table's kpair-2 configs on the same preshuffled B
# (results/kg3c_qkv_psh*.json): q_down fwd 29.0 vs 31.4 us, wgrad 31.8 vs 36.9, kv_down dgrad 22.5 vs 23.3.
_TABLE_BSH: dict = {
    (8192, 1536, 7168): C(256, 256, 4, 1, 2, 2),  # q_down fwd    28.2 us (row-major B 31.0)
    (1536, 7168, 8192): C(256, 256, 4, 1, 2, 2),  # q_down wgrad  31.8 (36.2)
    (8192, 1536, 24576): C(256, 256, 4, 1, 2, 2),  # q_up dgrad   88.7 (97.8)
    (24576, 1536, 8192): C(256, 256, 4, 1, 2, 2),  # q_up wgrad   97.5 (101.0)
    (8192, 7168, 576): C(256, 256, 3, 1, 2, 2),  # kv_down dgrad   22.6 (28.6)
    (8192, 7168, 16384): C(256, 256, 4, 1, 2, 4),  # o_proj fwd: 260.2 (c1x4 280.2; kcl1, 11-13: 8-WG multicast)
    (7168, 2048, 8192): C(256, 256, 4, 1, 1, 4),  # shared fc2 wgrad 35.6 (36.7)
    # 11-13 in-job A/B across pipeline depth x clusters (scratch_mxd/results/kprobe2.json, 2 rounds): the deep kpair-1
    # pipeline with 2x2 multicast beats the table's kpair-2 c2x2 where the loop is bound by the global -> LDS traffic
    (8192, 16384, 7168): C(256, 256, 4, 1, 4, 4),  # o_proj dgrad 240.0 (c2x2 260.1, nb2 kp2 c2x2 283.8; kcl1)
    # kfin7 (results/kfin7_bsh4.json, the old pick first, 2 rounds)
    (8192, 7168, 4096): C(256, 256, 4, 1, 2, 2),  # shared fc1 dgrad 76.3 (nb2 kp2 c2x2 83.8)
    (8192, 2048, 7168): C(256, 256, 4, 1, 4, 4),  # shared fc2 dgrad 33.2 (c2x2 35.5, kcl3; nb2 kp2 c1x4 38.9)
    (8192, 4096, 7168): C(256, 256, 4, 1, 4, 4),  # shared fc1 fwd 66.6 (nb2 kp2 c2x2 70.1; kcl1 0.966, kcl3 0.940)
}


# beta = 1 into a bf16 out (a wgrad fused into main_grad): (M, N, K) -> config. The kpair-1 entries above have no early
# old-C fetch there (their epilogue reads it at the stores, ~1.4x slower); in-job A/B of the 21 dense GEMMs as the op
# layer runs them (scratch_mxd/results/kd21_p.json, 11-13: the pick first, 2 rounds).
_TABLE_ACC: dict = {
    (1536, 7168, 8192): C(256, 256, 2, 2, 2, 2, persist=True, c_pre=True),  # q_down wgrad 45.3 us (kpair 1: 63.4)
    (24576, 1536, 8192): C(256, 256, 2, 2, 1, 4, persist=True, c_pre=True),  # q_up wgrad 122.2 (182.2)
    (7168, 16384, 8192): C(256, 256, 2, 2, 2, 2, persist=True, c_pre=True),  # o_proj wgrad 305.9 (c4x1: 313.5)
    (7168, 2048, 8192): C(256, 256, 2, 2, 2, 2, persist=True, c_pre=True),  # shared fc2 wgrad 47.2 (kpair 1: 67.0)
}


def pick_config(M: int, N: int, K: int, b_preshuffled: bool = False, num_cus: int = 256,
                beta1: bool = False) -> MXGemmConfig:
    """The measured config of (M, N, K), else by grid fill: 256x256 while it gives at least ~3/4 of a wave, then
    128x256, then 128x128 (two workgroups per CU). The entry point shrinks the cluster to what the grid divides.
    ``beta1``: accumulating into a bf16 out (the old C fetched early wants kpair 2: a kpair-1 pick becomes the
    persistent c2x2 with c_pre)."""
    cfg = beta1 and _TABLE_ACC.get((M, N, K))
    if cfg:
        return cfg
    cfg = _pick(M, N, K, b_preshuffled, num_cus)
    if beta1 and cfg.kpair == 1:
        return MXGemmConfig(256, 256, 2, 2, 2, 2, persist=True, c_pre=True)
    return cfg


def _pick(M: int, N: int, K: int, b_preshuffled: bool, num_cus: int) -> MXGemmConfig:
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


def pick_grouped_config(mode: int, M: int, N: int, K: int, b_preshuffled: bool = True,
                        bf16_out: bool = True) -> MXGemmConfig:
    """Routed experts. ``mode`` 1 (grouped-M fwd / dgrad: M = rows per expert, multicast along N only) or 2
    (variable-K wgrad: K = tokens per expert). In-job A/B on the DSv3 experts, 32 x 2048 rows balanced and skewed
    (scratch_mxd/results/kg3c_*.json, 11-12; kg5_*.json, 11-13): grouped-M with a preshuffled B runs best as the deep
    kpair-1 pipeline, K = 7168 (fc1 fwd, fc2 dgrad) and the shorter K too (fc1 dgrad 4096: 0.913 / 0.921 balanced /
    skewed, fc2 fwd 2048: 0.923 / 0.952 vs persistent nb2 kp2 c1x4; below K 2048 unmeasured, persistent kept);
    variable-K into a bf16 main_grad persistent (6-10%: the next tile loads while a tile stores) with the old C's first
    rows fetched during the previous tile's epilogue (c_pre: 2.5-3.2% balanced, 2-8% skewed; 11-13
    results/kw1_vk_*.json), c2x2 except for a narrow N into bf16 (fc2 wgrad, N = 2048: c1x1, 8% faster skewed; fp32
    out is not persistent, where c2x2 stays 8% faster)."""
    if mode == 1:
        if b_preshuffled and K >= 2048:
            return MXGemmConfig(256, 256, 4, 1, 1, 4)
        return MXGemmConfig(256, 256, 2, 2, 1, 4, persist=True)
    c = 1 if N <= 2048 and bf16_out else 2
    return MXGemmConfig(256, 256, 2, 2, c, c, persist=True, c_pre=True)
