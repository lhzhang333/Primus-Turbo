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

"""FlyDSL flash attention for gfx1250 (MI455X) at head dim 128 with GQA, on generated tile schedules.

The design of the DeepSeek-V3 MLA kernels (``gfx1250_mla``) at q / k / v head dim 128. Forward (fwd): 256-row q
blocks, on persistent workgroups where the grid allows them, the tile math generated (fwd_sched). Backward (bwd):
k_dkdv computes S and dP once, produces dK / dV and stores every dS^T tile into a scratch, k_dqds computes dQ = dS K
from it. The assembly model (``gfx1250_mla.asm``), the shell helpers (``gfx1250_mla.common``) and k_delta_bshd are
shared with that package.
"""

import hashlib
import pathlib

_PKG = pathlib.Path(__file__).resolve().parent
_MLA = _PKG.parents[1] / "gfx1250_mla"
_UTILS = _PKG.parents[2] / "utils"
# FlyDSL's disk-cache key misses code reached via nested helpers or other directories; launchers read this.
SOURCE_DIGEST = hashlib.sha256(
    b"".join(
        p.read_bytes()
        for p in (
            *sorted(_PKG.glob("*.py")),
            _MLA / "asm.py",
            _MLA / "common.py",
            _UTILS / "buffer_ops.py",
            _UTILS / "prims.py",
        )
    )
).hexdigest()
