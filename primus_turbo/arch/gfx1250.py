###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Manifest of gfx1250 (MI455X): the code Primus-Turbo has for it and the features built on it.

Data only; importing it imports nothing else. ``tools/check_arch_layout.py`` (pre-commit) checks
that everything listed exists and that every path under ``primus_turbo/`` and ``csrc/`` whose name
carries the ``gfx1250`` tag is listed here, so this file stays the complete answer to "where is the
gfx1250 code". Register a new gfx1250 kernel here in the pull request that adds it.
"""

ARCH = "gfx1250"
PRODUCTS = ("MI455X",)

# FlyDSL kernel packages and modules, by kernel family.
KERNELS = {
    # dense multi-head / grouped-query attention, bf16, head dim 128
    "attention.mha": "primus_turbo.flydsl.attention.gfx1250",
    # multi-head latent attention at the DeepSeek-V3 head dims (qk 192, v 128)
    "attention.mla": "primus_turbo.flydsl.attention.gfx1250_mla",
    # DeepSeek-V4 sparse MLA
    "attention.sparse_mla": "primus_turbo.flydsl.attention.gfx1250_sparse_mla",
    # bf16 grouped GEMM: NT, NN and the variable-K weight gradient
    "grouped_gemm.bf16": "primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_kernel_gfx1250",
    # MegaMoE: dispatch + GEMM + combine in fused kernels (shared infrastructure and bf16/)
    "mega_moe": "primus_turbo.flydsl.mega.gfx1250",
    # FlyDSL intranode DeepEP (V2). One source for every architecture, parameterized by
    # flydsl/deep_ep/arch.py; gfx1250 is the architecture that runs it, so it carries no tag.
    "deep_ep.intranode": "primus_turbo.flydsl.deep_ep.intranode_v2_kernel",
}

# Implementation-layer modules (primus_turbo/pytorch/kernels): what the ops call on gfx1250.
IMPLS = {
    "attention.mha.flydsl": "primus_turbo.pytorch.kernels.attention.attention_flydsl_gfx1250_impl",
    "attention.mla.flydsl": "primus_turbo.pytorch.kernels.attention.attention_flydsl_gfx1250_mla_impl",
    "attention.mla.hipkittens": "primus_turbo.pytorch.kernels.attention.attention_hipkittens_gfx1250_mla_impl",
    "mega_moe.bf16": "primus_turbo.pytorch.kernels.fused_mega_moe.fused_mega_moe_gfx1250_impl",
}

# Tagged paths that are not Python modules of the package: sources and build products.
SOURCES = (
    "csrc/kernels/attention/hipkittens/gfx1250_mla",  # HipKittens kernels of the MLA backward
    "primus_turbo/lib/hk_gfx1250_mla",  # their code objects (built, not tracked)
)

# Features other code may ask for (primus_turbo.capabilities): the kernel families and
# implementation modules each one needs, by their names above; "<name>:<symbol>" also names a
# symbol the module must define.
FEATURES = {
    "attention.flydsl.mha": ("attention.mha", "attention.mha.flydsl"),
    "attention.flydsl.mla": ("attention.mla", "attention.mla.flydsl"),
    "attention.flydsl.sparse_mla": ("attention.sparse_mla",),
    "attention.hipkittens.mla": ("attention.mla.hipkittens",),
    "mega_moe.bf16": ("mega_moe", "mega_moe.bf16"),
    "grouped_gemm.flydsl.bf16.wgrad_accum": (
        "grouped_gemm.bf16:grouped_gemm_bf16_variable_k_accum_supported",
    ),
    "deep_ep.flydsl": ("deep_ep.intranode",),
}
