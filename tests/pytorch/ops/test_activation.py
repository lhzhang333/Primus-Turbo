###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import random
from functools import partial

import pytest
import torch
import torch.nn.functional as F

from primus_turbo.pytorch.ops.activation import (
    geglu_with_probs,
    swiglu,
    swiglu_with_probs,
)
from tests.pytorch.test_utils import get_tolerances

torch.manual_seed(42)
random.seed(42)

CLAMP_LIMIT = 1.0


def _clamp_ref(x: torch.Tensor, min=None, max=None):
    """``torch.clamp`` that keeps the gradient at ``x == bound``, as the Turbo and TE clamped GLU kernels do.

    torch>=2.14 zeroes the ``torch.clamp`` gradient at a scalar bound (pytorch/pytorch#185831).
    """
    if max is not None:
        x = torch.where(x > max, max, x)
    if min is not None:
        x = torch.where(x < min, min, x)
    return x


# NOTE: Align precision with torch.compile
@torch.compile
def swiglu_with_probs_ref(x: torch.Tensor, probs: torch.Tensor):
    dtype = x.dtype
    x = torch.chunk(x, 2, dim=-1)
    res = F.silu(x[0]) * x[1]
    return (res * probs).to(dtype)


# NOTE: Align precision with torch.compile
@torch.compile
def clamped_swiglu_with_probs_ref(x: torch.Tensor, probs: torch.Tensor, clamp_limit: float):
    dtype = x.dtype
    gate, up = torch.chunk(x, 2, dim=-1)
    gate = _clamp_ref(gate, max=clamp_limit)
    up = _clamp_ref(up, min=-clamp_limit, max=clamp_limit)
    res = F.silu(gate) * up
    return (res * probs).to(dtype)


def generate_tokens_per_expert_list(num_experts: int, num_tokens: int):

    if num_experts == 1:
        return [num_tokens]

    parts = []
    remaining = num_tokens
    for _ in range(num_experts - 1):
        val = random.randint(0, remaining)
        parts.append(val)
        remaining -= val

    parts.append(remaining)
    return parts


# NOTE: Align precision with torch.compile
@torch.compile
def geglu_with_probs_ref(x: torch.Tensor, probs: torch.Tensor):
    dtype = x.dtype
    x = torch.chunk(x, 2, dim=-1)
    res = F.gelu(x[0]) * x[1]
    return (res * probs).to(dtype)


# NOTE: Align precision with torch.compile
@torch.compile
def clamped_geglu_with_probs_ref(x: torch.Tensor, probs: torch.Tensor, clamp_limit: float):
    dtype = x.dtype
    gate, up = torch.chunk(x, 2, dim=-1)
    gate = _clamp_ref(gate, max=clamp_limit)
    up = _clamp_ref(up, min=-clamp_limit, max=clamp_limit)
    res = F.gelu(gate) * up
    return (res * probs).to(dtype)


@pytest.mark.parametrize(
    "batch_size",
    [1, 8],
)
@pytest.mark.parametrize(
    "sequence_length",
    [
        1,
        128,
        2048,
        2025,
        8192,
    ],
)
@pytest.mark.parametrize(
    "hidden_size",
    [
        128,
        256,
        2048,
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("with_tokens_per_expert", [False, True])
@pytest.mark.parametrize("act_type", ["swiglu", "geglu", "clamped_swiglu", "clamped_geglu"])
def test_glu_with_probs(batch_size, sequence_length, hidden_size, dtype, with_tokens_per_expert, act_type):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    if act_type == "swiglu":
        func = swiglu_with_probs
        ref_func = swiglu_with_probs_ref
    elif act_type == "geglu":
        func = geglu_with_probs
        ref_func = geglu_with_probs_ref
    elif act_type == "clamped_swiglu":
        func = partial(swiglu_with_probs, clamp_limit=CLAMP_LIMIT)
        ref_func = partial(clamped_swiglu_with_probs_ref, clamp_limit=CLAMP_LIMIT)
    elif act_type == "clamped_geglu":
        func = partial(geglu_with_probs, clamp_limit=CLAMP_LIMIT)
        ref_func = partial(clamped_geglu_with_probs_ref, clamp_limit=CLAMP_LIMIT)

    device = "cuda"

    probs_dtype = torch.float32
    x = torch.randn(
        batch_size, sequence_length, hidden_size * 2, device=device, dtype=dtype, requires_grad=True
    )
    probs = torch.rand(batch_size, sequence_length, device=device, dtype=probs_dtype, requires_grad=True)

    num_tokens = batch_size * sequence_length

    x_ref = x.view(-1, x.size(-1)).clone().detach()
    x_ref.requires_grad_()
    probs_ref = probs.view(-1).unsqueeze(-1).clone().detach()
    probs_ref.requires_grad_()

    if with_tokens_per_expert:
        num_experts = 64
        tokens_per_expert = torch.tensor(
            generate_tokens_per_expert_list(num_experts, num_tokens), device=device, requires_grad=False
        )
        row_mask = torch.zeros(num_tokens, device=device, dtype=torch.int64, requires_grad=False)
        row_mask[: torch.sum(tokens_per_expert)] = 1
    else:
        row_mask = None

    out = func(x, probs, row_mask)
    out_ref = ref_func(x_ref, probs_ref)
    torch.testing.assert_close(out, out_ref, **get_tolerances(dtype))

    out.backward(torch.ones_like(out))
    grad_x = x.grad.clone()
    grad_probs = probs.grad.clone()

    out_ref.backward(torch.ones_like(out_ref))
    grad_x_ref = x_ref.grad.clone()
    grad_probs_ref = probs_ref.grad.clone()

    torch.testing.assert_close(grad_x, grad_x_ref.view_as(grad_x), **get_tolerances(dtype))
    torch.testing.assert_close(grad_probs, grad_probs_ref.view_as(grad_probs), **get_tolerances(probs_dtype))


def _swiglu_ref(x: torch.Tensor):
    """Megatron's bias-free SwiGLU: fp32 math, rounded once."""
    gate, up = torch.chunk(x.float(), 2, dim=-1)
    return (F.silu(gate) * up).to(x.dtype)


def _swiglu_bwd_ref(grad: torch.Tensor, x: torch.Tensor):
    """Megatron's swiglu_back: fp32 math, rounded once."""
    gate, up = torch.chunk(x.float(), 2, dim=-1)
    grad = grad.float()
    sig = torch.sigmoid(gate)
    return torch.cat((grad * sig * (1 + gate * (1 - sig)) * up, grad * (gate * sig)), dim=-1).to(x.dtype)


@pytest.mark.parametrize(
    "shape",
    [(1, 2), (3, 16), (5, 2 * 1000), (64, 2 * 1024), (130, 2 * 14336), (2, 37, 2 * 4096), (4, 8, 2 * 1408)],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_swiglu(shape, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    x = (torch.randn(shape, device="cuda", dtype=torch.float32) * 3).to(dtype).requires_grad_()
    out = swiglu(x)
    assert out.shape == (*shape[:-1], shape[-1] // 2) and out.dtype == dtype
    torch.testing.assert_close(out, _swiglu_ref(x.detach()), **get_tolerances(dtype))
    grad = torch.randn_like(out)
    out.backward(grad)
    torch.testing.assert_close(x.grad, _swiglu_bwd_ref(grad, x.detach()), **get_tolerances(dtype))


def test_swiglu_strided_rows():
    """A row stride larger than the row (a column slice): unit last-dim stride is all the kernels need."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    base = torch.randn(96, 3000, device="cuda", dtype=torch.bfloat16)
    x = base[:, 100:2148].detach().requires_grad_()  # [96, 2048], row stride 3000
    out = swiglu(x)
    torch.testing.assert_close(out, _swiglu_ref(x.detach()), **get_tolerances(torch.bfloat16))
    grad = torch.randn(96, 2000, device="cuda", dtype=torch.bfloat16)[:, :1024]  # strided grad
    out.backward(grad)
    torch.testing.assert_close(x.grad, _swiglu_bwd_ref(grad, x.detach()), **get_tolerances(torch.bfloat16))


def test_swiglu_empty():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    x = torch.empty(0, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = swiglu(x)
    assert out.shape == (0, 32)
    out.sum().backward()
    assert x.grad.shape == (0, 64)
