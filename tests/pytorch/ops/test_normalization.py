###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch
import torch.nn.functional as F

from primus_turbo.pytorch.ops.normalization import rmsnorm, rmsnorm_residual
from tests.pytorch.test_utils import get_tolerances


def _dgamma_tolerances(dtype):
    """Tolerances for ``dgamma``, which is a reduction over all ``B`` rows."""
    if dtype == torch.float32:
        return dict(rtol=1e-3, atol=1e-3)
    return get_tolerances(dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("outer_shape", [(1,), (511,), (4096,), (8192,)])
@pytest.mark.parametrize("inner_shape", [33, 128, 512, 513, 1536, 4096, 5120, 7168, 8192])
def test_rmsnorm_ops(dtype, outer_shape, inner_shape):
    torch.manual_seed(1)
    device = "cuda:0"
    eps = 1e-6

    shape = outer_shape + (inner_shape,)
    x = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    gamma = torch.randn(inner_shape, dtype=dtype, device=device, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()
    gamma_ref = gamma.detach().clone().requires_grad_()

    # Forward
    y_ref = F.rms_norm(x_ref, [inner_shape], gamma_ref, eps)
    y = rmsnorm(x, gamma, eps)

    torch.testing.assert_close(y_ref, y, **get_tolerances(dtype))

    # Backward
    grad_out = torch.randn_like(y)
    y.backward(grad_out)
    y_ref.backward(grad_out)

    torch.testing.assert_close(x.grad, x_ref.grad, **get_tolerances(dtype))
    torch.testing.assert_close(gamma.grad, gamma_ref.grad, **_dgamma_tolerances(dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("outer_shape", [(1,), (511,), (4096,)])
@pytest.mark.parametrize("inner_shape", [128, 4096, 8192])
def test_rmsnorm_zero_centered_ops(dtype, outer_shape, inner_shape):
    # zero_centered=True applies a (1 + gamma) gain. Reference folds the +1 into the weight.
    torch.manual_seed(3)
    device = "cuda:0"
    eps = 1e-6

    shape = outer_shape + (inner_shape,)
    x = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    gamma = torch.randn(inner_shape, dtype=dtype, device=device, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()
    gamma_ref = gamma.detach().clone().requires_grad_()

    # FWD — reference uses (1 + gamma) as the gain.
    y_ref = F.rms_norm(x_ref, [inner_shape], 1.0 + gamma_ref, eps)
    y = rmsnorm(x, gamma, eps, zero_centered=True)

    torch.testing.assert_close(y_ref, y, **get_tolerances(dtype))

    # BWD — d(1 + gamma)/d(gamma) = 1, so the gamma grad is unchanged and
    # flows through (1 + gamma_ref) to the gamma_ref leaf.
    grad_out = torch.randn_like(y)
    y.backward(grad_out)
    y_ref.backward(grad_out)

    torch.testing.assert_close(x.grad, x_ref.grad, **get_tolerances(dtype))
    torch.testing.assert_close(gamma.grad, gamma_ref.grad, **_dgamma_tolerances(dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("outer_shape", [(1,), (511,), (4096,)])
@pytest.mark.parametrize("inner_shape", [128, 4096, 8192])
def test_rmsnorm_residual_ops(dtype, outer_shape, inner_shape):
    torch.manual_seed(2)
    device = "cuda:0"
    eps = 1e-6

    shape = outer_shape + (inner_shape,)
    x = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    residual = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    gamma = torch.randn(inner_shape, dtype=dtype, device=device, requires_grad=True)

    x_ref = x.detach().clone().requires_grad_()
    r_ref = residual.detach().clone().requires_grad_()
    gamma_ref = gamma.detach().clone().requires_grad_()

    # Forward
    h_ref = x_ref + r_ref
    y_ref = F.rms_norm(h_ref, [inner_shape], gamma_ref, eps)

    y, x_plus_r = rmsnorm_residual(x, residual, gamma, eps)

    torch.testing.assert_close(x_plus_r, h_ref.detach(), **get_tolerances(dtype))
    torch.testing.assert_close(y, y_ref, **get_tolerances(dtype))

    # Backward — only flow gradient through y so dxpr = 0.
    grad_out = torch.randn_like(y)
    y.backward(grad_out)
    y_ref.backward(grad_out)

    torch.testing.assert_close(x.grad, x_ref.grad, **get_tolerances(dtype))
    torch.testing.assert_close(residual.grad, r_ref.grad, **get_tolerances(dtype))
    # The residual variant doubles the input magnitude entering the dgamma sum,
    # so for low-precision dtypes we use a slightly looser tolerance to absorb
    # the extra bf16/fp16 reduction noise.
    dg_tol = _dgamma_tolerances(dtype)
    if dtype in (torch.float16, torch.bfloat16):
        dg_tol = dict(rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(gamma.grad, gamma_ref.grad, **dg_tol)


@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("B,H", [(8192, 512), (8192, 1536), (4096, 7168), (300, 2880)])
def test_rmsnorm_bwd_tiled(B, H, residual):
    """The tiled backward at the DSv3 norm widths (kv / q / hidden) and a ragged shape: dx / dgamma vs torch
    (bf16), the residual variant with gradients through both outputs, and bitwise the same on a rerun (fixed
    launch configs, no autotune)."""
    torch.manual_seed(B + H)
    eps = 1e-6
    x = torch.randn(B, H, dtype=torch.bfloat16, device="cuda")
    r = torch.randn_like(x)
    gamma = torch.randn(H, dtype=torch.bfloat16, device="cuda")
    gy, gr = torch.randn_like(x), torch.randn_like(x)

    def run(ref):
        leaves = [t.clone().requires_grad_() for t in ((x, r, gamma) if residual else (x, gamma))]
        if residual:
            if ref:
                h = leaves[0] + leaves[1]
                outs = (F.rms_norm(h, [H], leaves[2], eps), h)
            else:
                outs = rmsnorm_residual(leaves[0], leaves[1], leaves[2], eps)
            torch.autograd.backward(outs, (gy, gr))
        else:
            y = F.rms_norm(leaves[0], [H], leaves[1], eps) if ref else rmsnorm(leaves[0], leaves[1], eps)
            y.backward(gy)
        return [t.grad for t in leaves]

    got, ref = run(False), run(True)
    for a, b in zip(got[:-1], ref[:-1]):
        torch.testing.assert_close(a, b, **get_tolerances(torch.bfloat16))
    torch.testing.assert_close(got[-1], ref[-1], rtol=3e-2, atol=3e-2)
    for a, b in zip(got, run(False)):
        assert torch.equal(a, b), "the backward is not bitwise the same on a rerun"
