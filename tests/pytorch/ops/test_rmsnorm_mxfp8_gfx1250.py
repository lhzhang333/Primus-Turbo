###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""gfx1250 RMSNorm forward + MXFP8 dual quantization (primus_turbo/flydsl/quantization/rmsnorm_mxfp8_gfx1250.py)
and its op (``primus_turbo.pytorch.ops.normalization.rmsnorm_mxfp8``), against tests/pytorch/ref/mxfp8_ref.py:
rstd, the bf16 y and both MXFP8 copies bit for bit (reference on the CPU)."""

import os
import sys

import pytest
import torch

from primus_turbo.pytorch.core.utils import is_gfx1250

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ref"))
import mxfp8_ref as R  # noqa: E402

pytestmark = pytest.mark.skipif(not is_gfx1250(), reason="gfx1250 kernels")
PACK_ROW, PACK_COL, SHUF_ROW, SHUF_COL = 1, 2, 4, 8
MX_AB = PACK_ROW | PACK_COL | SHUF_COL  # the op's default: packed rows (A), packed + 16x16 columns (wgrad B)
EPS = 1e-6


@pytest.fixture(scope="module")
def N():
    from primus_turbo.flydsl.quantization import rmsnorm_mxfp8_gfx1250

    return rmsnorm_mxfp8_gfx1250


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


def _eq(a, b, what):
    a, b = a.contiguous().view(torch.uint8).cpu(), b.contiguous().view(torch.uint8).cpu()
    assert a.shape == b.shape, f"{what}: shape {tuple(a.shape)} vs {tuple(b.shape)}"
    bad = (a != b).nonzero()
    assert bad.numel() == 0, f"{what}: {bad.shape[0]} bytes differ, first at {bad[0].tolist()}"


def _padded(t, shape):
    out = torch.zeros(shape, dtype=torch.uint8)
    out[tuple(slice(0, d) for d in t.shape)] = t.cpu().view(torch.uint8)
    return out


def _pack(s):
    return R.pack_scales(s.cpu().view(torch.uint8))


def _check(N, x, gamma, rule, pad, lay, write_y):
    M, H = x.shape
    y, rstd, *q = N.rmsnorm_quant_mxfp8(x, gamma, EPS, rule=rule, pad=pad, layout=lay, write_y=write_y)
    rstd_ref = R.rmsnorm_rstd_ref(x.cpu(), EPS)
    _eq(rstd, rstd_ref, "rstd")
    y_ref = R.rmsnorm_ref(x.cpu(), gamma.cpu(), rstd_ref)
    if write_y:
        _eq(y, y_ref, "y")
    else:
        assert y is None
    KO, Mc = (-(-H // 128) * 128, -(-M // 128) * 128) if pad else (H, M)
    shapes = ((M, KO), (M, -(-KO // 32)), (H, Mc), (H, -(-Mc // 32)))
    ref = [_padded(t, sh) for t, sh in zip(R.quant_dual(y_ref, rule), shapes)]
    ref = (R.preshuffle_b16(ref[0]) if lay & SHUF_ROW else ref[0], _pack(ref[1]) if lay & PACK_ROW else ref[1],
           R.preshuffle_b16(ref[2]) if lay & SHUF_COL else ref[2], _pack(ref[3]) if lay & PACK_COL else ref[3])
    for got, exp, what in zip(q, ref, ("q_row", "s_row", "q_col", "s_col")):
        _eq(got, exp, what)


@pytest.mark.parametrize("rule", ["nosat", "even"])
@pytest.mark.parametrize("lay,pad", [(0, False), (MX_AB, True), (PACK_ROW | PACK_COL | SHUF_ROW | SHUF_COL, True)])
@pytest.mark.parametrize(
    "M,H,stride_pad", [(1, 512, 0), (33, 512, 64), (100, 1536, 0), (300, 200, 8), (257, 7168, 0), (128, 256, 0)]
)
def test_kernel(N, M, H, stride_pad, rule, lay, pad):
    if (lay & SHUF_ROW and M % 16) or (lay & SHUF_COL and H % 16):
        pytest.skip("16x16 codes need the copy's rows % 16 == 0")
    x, gamma = _inputs(M, H, stride_pad, seed=M + H)
    _check(N, x, gamma, rule, pad, lay, write_y=True)


@pytest.mark.parametrize(
    "M,H,stride_pad", [(8192, 512, 64), (4096, 512, 64), (8192, 1536, 0), (4096, 1536, 576), (4096, 7168, 0),
                       (8192, 7168, 0)]
)
def test_dsv3_shapes(N, M, H, stride_pad):
    """The DSv3 norms at T = 8192 / 4096: kv_layernorm (512, a view of [T, 576]), q_layernorm (1536; as a view of
    the merged q / kv down output [T, 2112] too), input / pre-MLP layernorm (7168); the op's layouts."""
    x, gamma = _inputs(M, H, stride_pad, seed=H)
    _check(N, x, gamma, "nosat", True, MX_AB, write_y=H == 7168)


def test_skip_y(N):
    x, gamma = _inputs(300, 1536, 0, seed=3)
    _check(N, x, gamma, "nosat", True, MX_AB, write_y=False)


@pytest.mark.parametrize("M,H,N_out,stride_pad", [(256, 512, 1024, 64), (300, 1536, 512, 0), (512, 7168, 576, 0)])
def test_op(M, H, N_out, stride_pad, monkeypatch):
    """``rmsnorm_mxfp8`` feeding ``gemm_fp8`` with its pair: no re-quantization (strict mode), the GEMM output equal
    to ``gemm_fp8`` of the op's own bf16 y, and the gradients (through y and through the pair) equal to the
    unfused path's within its rstd rounding."""
    import primus_turbo.common.envs as envs
    from primus_turbo.pytorch.core.low_precision import (
        Float8QuantConfig,
        Format,
        ScaleDtype,
        ScalingGranularity,
        ScalingRecipe,
    )
    from primus_turbo.pytorch.ops.gemm_fp8 import gemm_fp8
    from primus_turbo.pytorch.ops.normalization import rmsnorm, rmsnorm_mxfp8

    cfg = Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.MX_BLOCKWISE,
                            scale_dtype=ScaleDtype.E8M0, block_size=32)
    g = torch.Generator(device="cuda").manual_seed(M)
    w = (torch.randn(N_out, H, generator=g, device="cuda") / H**0.5).to(torch.bfloat16)
    try:  # the op layer's choice of operand layouts (the W line's mx_gemm_layout), raw before it exists
        from primus_turbo.pytorch.ops.utils import MX_A, MX_B, mx_gemm_layout

        rr, rc = (MX_A, MX_B) if mx_gemm_layout(w) else (ScalingRecipe(), ScalingRecipe())
    except ImportError:
        rr, rc = ScalingRecipe(), ScalingRecipe()
    x0, gamma0 = _inputs(M, H, stride_pad, seed=7)
    v = torch.randn(M, H, generator=g, device="cuda")
    monkeypatch.setenv("PRIMUS_TURBO_MX_STRICT", "1")
    envs.reset_cache()

    def run(fused):
        x, gamma = x0.detach().clone().requires_grad_(), gamma0.detach().clone().requires_grad_()
        if fused:
            y, pair = rmsnorm_mxfp8(x, gamma, EPS, row_recipe=rr, col_recipe=rc)
            out = gemm_fp8(pair, w, trans_b=True, config=cfg)
        else:
            y = rmsnorm(x, gamma, EPS)
            out = gemm_fp8(y, w, trans_b=True, config=cfg)
        ((out.float() ** 2).sum() + (y.float() * v).sum()).backward()
        return y.detach(), out.detach(), x.grad, gamma.grad

    y, out, dx, dg = run(True)
    assert torch.equal(out, gemm_fp8(y, w, trans_b=True, config=cfg)), "the pair is the MXFP8 cast of y"
    y2, out2, dx2, dg2 = run(False)
    for a, b, what in ((y, y2, "y"), (out, out2, "out"), (dx, dx2, "dx"), (dg, dg2, "dgamma")):
        err = float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30))
        assert err < 2e-2, f"{what}: relative error {err:.3g} vs the unfused path"
