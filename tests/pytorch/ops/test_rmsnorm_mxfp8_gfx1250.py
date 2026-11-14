###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""gfx1250 RMSNorm forward + MXFP8 dual quantization (primus_turbo/flydsl/quantization/rmsnorm_mxfp8_gfx1250.py):
rstd and the bf16 y bit for bit Turbo's RMSNorm forward (``rmsnorm_fwd_impl``), both MXFP8 copies bit for bit
tests/pytorch/ref/mxfp8_ref.py's cast of that y, in every layout. (The op, ``ops.normalization.rmsnorm_mxfp8``:
tests/pytorch/ops/test_mxfp8_ops_gfx1250.py.)"""

import pytest
import torch

from primus_turbo.pytorch.core.utils import is_gfx1250
from tests.pytorch.ref import mxfp8_ref as R
from tests.pytorch.ref.mxfp8_ref import assert_bytes_equal as _eq
from tests.pytorch.ref.mxfp8_ref import padded as _padded
from tests.pytorch.test_utils import tensor_at_mapping_end

pytestmark = pytest.mark.skipif(not is_gfx1250(), reason="gfx1250 kernels")
if is_gfx1250():
    from primus_turbo.flydsl.quantization import rmsnorm_mxfp8_gfx1250 as N
    from primus_turbo.flydsl.quantization.mxfp8_quant_gfx1250 import (
        PACK_COL,
        PACK_ROW,
        SHUF_COL,
        SHUF_ROW,
    )

    # the layouts by name: raw; the op's default (packed rows = A, packed + 16x16 columns = a wgrad B); all
    LAYOUTS = {
        "raw": 0,
        "mx_ab": PACK_ROW | PACK_COL | SHUF_COL,
        "all": PACK_ROW | PACK_COL | SHUF_ROW | SHUF_COL,
    }
EPS = 1e-6


def _inputs(M, H, stride_pad=0, seed=0):
    """x [M, H] bf16 (a column slice of [M, H + stride_pad] when stride_pad), rows scaled over 2^+-20, a zero
    row, a row whose squares overflow fp32 (rstd 0), a constant row; gamma [H] bf16 around 1."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    full = torch.randn(M, H + stride_pad, generator=g, device="cuda")
    full = full * torch.exp2(torch.randint(-20, 20, (M, 1), generator=g, device="cuda").float())
    if M > 3:
        full[1], full[2], full[3] = 0.0, 3.0e19, -0.75
    x = full.to(torch.bfloat16)[:, :H]
    gamma = (1 + 0.25 * torch.randn(H, generator=g, device="cuda")).to(torch.bfloat16)
    return x, gamma


def _check(x, gamma, rule, pad, lay, write_y):
    from primus_turbo.pytorch.kernels.normalization.rmsnorm_impl import rmsnorm_fwd_impl

    M, H = x.shape
    y, rstd, *q = N.rmsnorm_quant_mxfp8(x, gamma, EPS, rule=rule, pad=pad, layout=lay, write_y=write_y)
    y_ref, _, rstd_ref = rmsnorm_fwd_impl(x, gamma, EPS)[:3]  # Turbo's RMSNorm: the fused path's y and rstd
    _eq(rstd, rstd_ref, "rstd")
    y_ref = y_ref.cpu()
    if write_y:
        _eq(y, y_ref, "y")
    else:
        assert y is None
    KO, Mc = (-(-H // 128) * 128, -(-M // 128) * 128) if pad else (H, M)
    shapes = ((M, KO), (M, -(-KO // 32)), (H, Mc), (H, -(-Mc // 32)))
    ref = [_padded(t, sh) for t, sh in zip(R.quant_dual(y_ref, rule), shapes)]
    flags = dict(
        pack_row=lay & PACK_ROW, shuf_row=lay & SHUF_ROW, pack_col=lay & PACK_COL, shuf_col=lay & SHUF_COL
    )
    ref = R.to_layout(ref, **{k: bool(v) for k, v in flags.items()})
    for got, exp, what in zip(q, ref, ("q_row", "s_row", "q_col", "s_col")):
        _eq(got, exp, what)


@pytest.mark.parametrize("rule", ["nosat", "even"])
@pytest.mark.parametrize("lay,pad", [("raw", False), ("mx_ab", True), ("all", True)])
@pytest.mark.parametrize(
    "M,H,stride_pad",
    [(1, 512, 0), (33, 512, 64), (100, 1536, 0), (300, 200, 8), (257, 7168, 0), (128, 256, 0)],
)
def test_kernel(M, H, stride_pad, rule, lay, pad):
    lay = LAYOUTS[lay]
    if (lay & SHUF_ROW and M % 16) or (lay & SHUF_COL and H % 16):
        pytest.skip("16x16 codes need the copy's rows % 16 == 0")
    x, gamma = _inputs(M, H, stride_pad, seed=M + H)
    _check(x, gamma, rule, pad, lay, write_y=True)


@pytest.mark.parametrize("M,H,stride_pad", [(8192, 512, 64), (4096, 1536, 576), (8192, 7168, 0)])
def test_dsv3_shapes(M, H, stride_pad):
    """The DSv3 norms, one each: kv_layernorm (512, a view of [T, 576]), q_layernorm (1536, a view of the merged
    q / kv down output [T, 2112]), input / pre-MLP layernorm (7168); the op's layouts."""
    x, gamma = _inputs(M, H, stride_pad, seed=H)
    _check(x, gamma, "nosat", True, LAYOUTS["mx_ab"], write_y=H == 7168)


def test_skip_y():
    x, gamma = _inputs(300, 1536, 0, seed=3)
    _check(x, gamma, "nosat", True, LAYOUTS["mx_ab"], write_y=False)


def test_tail_at_mapping_end():
    """x [M, 200] (H % 128 != 0) ending where its GPU mapping does: the last row's partial k-tile must not be read
    by the TDM (it addresses whole 128-column rows). Same outputs as x in ordinary storage."""
    x, gamma = _inputs(257, 200, 0, seed=5)
    ref = N.rmsnorm_quant_mxfp8(x, gamma, EPS, pad=True, layout=LAYOUTS["mx_ab"])
    with tensor_at_mapping_end(x.shape, x.dtype) as e:
        e.copy_(x)
        got = N.rmsnorm_quant_mxfp8(e, gamma, EPS, pad=True, layout=LAYOUTS["mx_ab"])
        for k, (a, b) in enumerate(zip(got, ref)):
            _eq(a, b, f"output {k}")
