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

"""FlyDSL kernels written for gfx1250 (MI455X): wave32, WMMA, the TDM engine.

One package per operation, with the layout the packages next to this one have
(``primus_turbo/flydsl/<op>/``, which hold the gfx950 kernels and the shared code):

    ``attention/``     ``mha`` (dense, head dim 128), ``mla`` (DeepSeek-V3 head dims), ``sparse_mla`` (DeepSeek-V4)
    ``grouped_gemm/``  bf16 grouped GEMM (NT, NN, variable-K)
    ``mega/``          MegaMoE: dispatch + GEMM + combine in fused kernels

Nothing is imported here. The packages can be imported on any architecture (the implementation layer,
``primus_turbo/pytorch/kernels``, imports some of them at module level), but their kernels launch on
gfx1250 only; the implementation layer checks the device before it calls them. The FlyDSL DeepEP is
not here: it is one source for every architecture (``flydsl/deep_ep``, parameterized by ``arch.py``).
Code outside Primus-Turbo asks ``primus_turbo.capabilities`` instead of looking in this directory.

A launcher and the plain helpers it calls stay in one directory: FlyDSL folds a helper's source into a
kernel's disk-cache key only when the helper sits in the launcher's directory.
"""
