###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import json
import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

import primus_turbo
from primus_turbo.pytorch.core.utils import is_gfx1250
from primus_turbo.pytorch.ops.normalization import rmsnorm, rmsnorm_residual
from tests.pytorch.test_utils import get_tolerances


def _dgamma_tolerances(dtype):
    """Tolerances for ``dgamma``, which is a reduction over all ``B`` rows."""
    if dtype == torch.float32:
        return dict(rtol=1e-3, atol=1e-3)
    return get_tolerances(dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("outer_shape", [(1,), (511,), (4096,), (8192,)])
@pytest.mark.parametrize("inner_shape", [33, 128, 513, 4096, 5120, 7168, 8192, 12288, 16384])
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
@pytest.mark.parametrize("inner_shape", [128, 4096, 8192, 16384])
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
@pytest.mark.parametrize("inner_shape", [128, 4096, 8192, 12288, 16384])
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


# Prints {case: sha256 of each input's grad} for the bf16 backward of the [B, H, residual] cases in argv[1].
_BWD_DIGESTS = r"""
import hashlib, json, sys
import torch
from primus_turbo.pytorch.ops.normalization import rmsnorm, rmsnorm_residual

out = {}
for B, H, residual in json.loads(sys.argv[1]):
    g = torch.Generator(device="cuda").manual_seed(B * 7 + H)
    x, r, gy, gr = (torch.randn(B, H, generator=g, device="cuda").to(torch.bfloat16) for _ in range(4))
    gamma = torch.randn(H, generator=g, device="cuda").to(torch.bfloat16)
    leaves = [t.requires_grad_() for t in ((x, r, gamma) if residual else (x, gamma))]
    if residual:
        torch.autograd.backward(rmsnorm_residual(x, r, gamma, 1e-6), (gy, gr))
    else:
        rmsnorm(x, gamma, 1e-6).backward(gy)
    out[f"{B}x{H}{'r' if residual else ''}"] = [
        hashlib.sha256(t.grad.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest() for t in leaves
    ]
print(json.dumps(out))
"""


def _bwd_digests_in_new_process(cases):
    root = os.path.dirname(os.path.dirname(os.path.abspath(primus_turbo.__file__)))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (root, os.environ.get("PYTHONPATH")) if p))
    proc = subprocess.run(
        [sys.executable, "-c", _BWD_DIGESTS, json.dumps(cases)],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(not is_gfx1250(), reason="fixed launch configs on gfx1250 only (autotuned elsewhere)")
def test_rmsnorm_bwd_bitwise_across_processes():
    """dx / dgamma (/ dresidual) bit for bit the same in two processes, for each backward kernel (multi-row,
    grid-stride). An autotune per process could pick different num_warps, i.e. a different reduction order, in
    each of them."""
    cases = [
        [8192, 128, False],
        [8192, 512, False],
        [8192, 1536, True],
        [4096, 7168, False],
        [300, 2880, True],
        [2048, 12288, False],
        [1024, 16384, True],
    ]
    first, second = (_bwd_digests_in_new_process(cases) for _ in range(2))
    assert len(first) == len(cases)
    assert first == second, {k: (first[k], second[k]) for k in first if first[k] != second[k]}
