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

"""FlyDSL attention kernels for gfx1250 (MI455X), one package per kernel family:

    ``mha``         dense multi-head / grouped-query attention, bf16, head dim 128
    ``mla``         multi-head latent attention at the DeepSeek-V3 head dims (qk 192, v 128)
    ``sparse_mla``  DeepSeek-V4 sparse MLA

This package module imports nothing. A family keeps its launchers and the helpers they call in
its own directory: FlyDSL folds a helper's source into a kernel's disk-cache key only when the
helper sits in the launcher's directory.
"""
