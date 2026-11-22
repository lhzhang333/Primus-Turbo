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

"""Nothing lives here any more: the gfx1250 kernels moved to ``primus_turbo.flydsl.gfx1250.attention``.

The directory stays for one release. Primus releases that predate ``primus_turbo.capabilities``
decide whether to use these kernels by checking that this directory exists; without it they would
switch the feature off without an error. New code asks ``primus_turbo.capabilities``.
"""
