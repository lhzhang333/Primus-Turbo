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

"""RMSNorm forward + MXFP8 dual quantization for gfx1250 (the DSv3 norms that feed MX GEMMs).

Two launches, no bf16 round trip between the norm and the quantizer:

* Turbo's RMSNorm forward kernel without its y store (``rmsnorm_impl.rmsnorm_rstd_impl``): the same rstd, bit
  for bit, as ``rmsnorm_fwd_impl``. Reads x once.
* the tile kernel of ``mxfp8_quant_gfx1250`` with the ``P_NORM`` producer: the x tile comes in by TDM, is
  normalized in LDS to ``y = bf16((x * rstd[row]) * gamma[col])`` (Turbo's RMSNorm formula), optionally
  stored, and quantized like any bf16 tile, in any layout (the second read of x mostly hits the MALL).

So y is Turbo's RMSNorm output bit for bit and the codes are exactly ``mxfp8_ref.quant_dual`` of it: the fused
path computes what Turbo's RMSNorm followed by the MXFP8 dual quantization computes. x may have a row stride
(a split view, e.g. MLA's ``kv_compressed`` inside ``[T, 576]``); H % 8 == 0.
"""

from __future__ import annotations

import torch

from primus_turbo.flydsl.quantization import mxfp8_quant_gfx1250 as Q
from primus_turbo.pytorch.kernels.normalization.rmsnorm_impl import rmsnorm_rstd_impl


def rmsnorm_quant_mxfp8(
    x, gamma, eps=1e-6, out_dtype=torch.float8_e4m3fn, rule="nosat", pad=False, layout=0, write_y=True
):
    """RMSNorm + MXFP8 dual cast of x [M, H] bf16 (last dim contiguous, row stride % 8 == 0, H % 8 == 0) with
    gamma [H] bf16: (y [M, H] bf16 or None, rstd [M] fp32, q_row [M, H'], s_row, q_col [H, M'], s_col), the copies
    as ``mxfp8_quant_gfx1250.quant_mxfp8_raw(y, ..., pad=pad, layout=layout)`` would write them (``pad``: HIP's
    128-padding of K and M). y and rstd are ``rmsnorm_fwd_impl``'s, bit for bit. ``write_y`` False: y is not
    stored (None), only quantized."""
    assert x.ndim == 2 and x.is_cuda and x.dtype == torch.bfloat16 and gamma.dtype == torch.bfloat16
    M, H = x.shape
    assert (
        gamma.shape == (H,)
        and H % 8 == 0
        and x.stride(1) == 1
        and x.stride(0) % 8 == 0
        and x.data_ptr() % 16 == 0
    )
    assert out_dtype == torch.float8_e4m3fn and rule in Q.RULES
    rstd = rmsnorm_rstd_impl(x, H, eps)
    y = torch.empty(M, H, dtype=torch.bfloat16, device=x.device) if write_y else None
    KO, Mc = (Q._ceil128(H), Q._ceil128(M)) if pad else (H, M)
    qr, sr = Q._bufs((), M, KO, layout & Q.PACK_ROW, layout & Q.SHUF_ROW, out_dtype, x)
    qc, sc = Q._bufs((), H, Mc, layout & Q.PACK_COL, layout & Q.SHUF_COL, out_dtype, x)
    Q._run(
        Q.P_NORM,
        Q.O_ROWS | Q.O_COLS,
        rule,
        x,
        (qr, sr, qc, sc, None, None),
        M,
        Mc,
        -(-M // Q.BM),
        H,
        KO,
        0,
        int(pad),
        x2=gamma.contiguous(),
        p=rstd,
        dbg=y,
        t=x.stride(0),
        lay=layout,
    )
    return y, rstd, qr, sr, qc, sc
