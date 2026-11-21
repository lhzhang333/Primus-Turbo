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

"""Old path of ``primus_turbo.flydsl.grouped_gemm.gfx1250.grouped_gemm_bf16_kernel``, kept for one release.

Importing this name gives the moved module itself (the same object, so its functions, caches and
anything patched on it are shared), for callers outside Primus-Turbo that still import the old
path. New code imports ``primus_turbo.flydsl.grouped_gemm.gfx1250.grouped_gemm_bf16_kernel``.
"""

import sys

from primus_turbo.flydsl.grouped_gemm.gfx1250 import grouped_gemm_bf16_kernel

sys.modules[__name__] = grouped_gemm_bf16_kernel
