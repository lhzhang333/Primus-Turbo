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

* ``rstd`` kernel: one wave per row, each lane sums the squares of 8-element chunks ``8 (lane + 32 j)`` in
  fp32, a butterfly over the wave adds the lanes; ``rstd = rsqrt(sum / H + eps)`` (fp32, what the backward
  needs). Reads x once.
* the tile kernel of ``mxfp8_quant_gfx1250`` with the ``P_NORM`` producer: the x tile comes in by TDM, is
  normalized in LDS to ``y = bf16((x * rstd[row]) * gamma[col])`` (Turbo's RMSNorm formula), optionally
  stored, and quantized like any bf16 tile, in any layout (the second read of x mostly hits the MALL).

So the codes are exactly ``mxfp8_ref.quant_dual`` of the bf16 ``y`` this writes. x may have a row stride
(a split view, e.g. MLA's ``kv_compressed`` inside ``[T, 576]``); H % 8 == 0.
"""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import math as fm
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import Constexpr, T
from flydsl.expr.typing import Vector as Vec

from primus_turbo.flydsl.quantization import mxfp8_quant_gfx1250 as Q
from primus_turbo.flydsl.utils.buffer_ops import buffer_load, buffer_store

I32, I64, F32 = fx.Int32, fx.Int64, fx.Float32
ROWS = 8  # rows per workgroup: one wave each


def _f32(v):
    return Vec.from_elements([v], I32).bitcast(F32)[0]


@flyc.jit
def _launch_rstd(arg_x: fx.Pointer, arg_r: fx.Pointer, stream: fx.Stream, i32_m: fx.Int32, i32_xs: fx.Int32,
                 i32_grid: fx.Int32, K: Constexpr[int], EPS: Constexpr[float]):
    @flyc.kernel(name="rmsnorm_rstd_gfx1250", known_block_size=[Q.NT, 1, 1])
    def kern(arg_x: fx.Pointer, arg_r: fx.Pointer, i32_m: fx.Int32, i32_xs: fx.Int32):
        t = I32(fx.thread_idx.x)
        lane = t % I32(32)
        row = I32(fx.block_idx.x) * I32(ROWS) + rocdl.readfirstlane(T.i32, t // I32(32))
        if row < i32_m:
            xs = Q._rsrc(Q._addr(arg_x) + I64(row) * I64(i32_xs) * I64(2), I32(K * 2))
            acc = F32(0.0)
            for j in range_constexpr((K + 255) // 256):  # past H: the bound reads 0
                v = Vec(buffer_load(xs, I32(j * 128) + lane * I32(4), vec_width=4, dtype=T.i32))
                v = v.bitcast(fx.BFloat16).to(F32)
                for e in range_constexpr(8):
                    acc = acc + v[e] * v[e]
            for ctrl in (0xB1, 0x4E, 0x164, 0x168):  # xor 1, 2, 4, 8 within each 16-lane row
                acc = acc + _f32(Q._dpp(Vec.from_elements([acc], F32).bitcast(I32)[0], ctrl))
            bits = Vec.from_elements([acc], F32).bitcast(I32)[0]
            tot = _f32(I32(rocdl.readlane(T.i32, bits, 0))) + _f32(I32(rocdl.readlane(T.i32, bits, 16)))
            rstd = fm.rsqrt(tot / F32(float(K)) + F32(EPS))
            buffer_store(rstd, Q._rsrc(Q._addr(arg_r), i32_m * I32(4)), row, mask=lane == I32(0))

    kern(arg_x, arg_r, i32_m, i32_xs).launch(grid=(i32_grid, 1, 1), block=(Q.NT, 1, 1), stream=stream)


_COMPILED: dict = {}


def rmsnorm_rstd(x: torch.Tensor, eps: float, rstd: torch.Tensor | None = None) -> torch.Tensor:
    """``rsqrt(mean(x^2) + eps)`` per row of x [M, H] bf16 (rows may be strided), fp32 [M]."""
    M, H = x.shape
    rstd = torch.empty(M, dtype=torch.float32, device=x.device) if rstd is None else rstd
    if M == 0:
        return rstd
    key = (H, float(eps))
    fn = _COMPILED.get(key)
    if fn is not None:  # the compiled launcher with data_ptr ints and the raw stream (as mxfp8_quant_gfx1250._run)
        dev = x.get_device()
        stream = Q._RAW_STREAM(dev) if Q._RAW_STREAM is not None else torch.cuda.current_stream()
        fn(x.data_ptr(), rstd.data_ptr(), stream, M, x.stride(0), -(-M // ROWS), *key)
        return rstd
    fn = flyc.compile(_launch_rstd, Q._ptr(x), Q._ptr(rstd), torch.cuda.current_stream(), M, x.stride(0),
                      -(-M // ROWS), *key)  # compiles, then runs this call
    if fn is not None:
        _COMPILED[key] = fn
    return rstd


def rmsnorm_quant_mxfp8(x, gamma, eps=1e-6, out_dtype=torch.float8_e4m3fn, rule="nosat", pad=False, layout=0,
                        write_y=True):
    """RMSNorm + MXFP8 dual cast of x [M, H] bf16 (last dim contiguous, row stride % 8 == 0, H % 8 == 0) with
    gamma [H] bf16: (y [M, H] bf16 or None, rstd [M] fp32, q_row [M, H'], s_row, q_col [H, M'], s_col), the copies
    as ``mxfp8_quant_gfx1250.quant_mxfp8_raw(y, ..., pad=pad, layout=layout)`` would write them (``pad``: HIP's
    128-padding of K and M). ``write_y`` False: y is not stored (None), only quantized."""
    assert x.ndim == 2 and x.is_cuda and x.dtype == torch.bfloat16 and gamma.dtype == torch.bfloat16
    M, H = x.shape
    assert gamma.shape == (H,) and H % 8 == 0 and x.stride(1) == 1 and x.stride(0) % 8 == 0 and x.data_ptr() % 16 == 0
    assert out_dtype == torch.float8_e4m3fn and rule in Q.RULES
    rstd = rmsnorm_rstd(x, eps)
    y = torch.empty(M, H, dtype=torch.bfloat16, device=x.device) if write_y else None
    KO, Mc = (Q._ceil128(H), Q._ceil128(M)) if pad else (H, M)
    qr, sr = Q._bufs((), M, KO, layout & Q.PACK_ROW, layout & Q.SHUF_ROW, out_dtype, x)
    qc, sc = Q._bufs((), H, Mc, layout & Q.PACK_COL, layout & Q.SHUF_COL, out_dtype, x)
    Q._run(Q.P_NORM, Q.O_ROWS | Q.O_COLS, rule, x, (qr, sr, qc, sc, None, None), M, Mc, -(-M // Q.BM), H, KO, 0,
           int(pad), x2=gamma.contiguous(), p=rstd, dbg=y, t=x.stride(0), lay=layout)
    return y, rstd, qr, sr, qc, sc
