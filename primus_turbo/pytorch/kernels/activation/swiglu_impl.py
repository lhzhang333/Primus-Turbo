###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional

import torch
import triton

from primus_turbo.triton.activation.swiglu_kernel import (
    swiglu_bwd_kernel,
    swiglu_dense_bwd_kernel,
    swiglu_dense_fwd_kernel,
    swiglu_fwd_kernel,
    swiglu_with_mask_bwd_kernel,
    swiglu_with_mask_fwd_kernel,
)

# Tile of the dense kernels: BLOCK_M rows x BLOCK_N columns of each half.
_DENSE_BLOCK_M = 4
_DENSE_BLOCK_N = 1024
_DENSE_NUM_WARPS = 4


def _dense_grid(num_rows: int, half_cols: int):
    return (triton.cdiv(num_rows, _DENSE_BLOCK_M), triton.cdiv(half_cols, _DENSE_BLOCK_N))


def swiglu_fwd(x: torch.Tensor) -> torch.Tensor:
    """``silu(x[:, :N]) * x[:, N:]`` of a 2-D ``x`` with ``2 * N`` columns (unit column stride)."""
    assert x.ndim == 2 and x.size(1) % 2 == 0, f"x must be 2-D with an even last dim, got {tuple(x.shape)}"
    num_rows, half_cols = x.size(0), x.size(1) // 2
    out = torch.empty(num_rows, half_cols, dtype=x.dtype, device=x.device)
    if out.numel() == 0:
        return out
    assert x.stride(1) == 1, "x must have a unit column stride"
    swiglu_dense_fwd_kernel[_dense_grid(num_rows, half_cols)](
        x,
        out,
        num_rows,
        half_cols,
        x.stride(0),
        out.stride(0),
        BLOCK_M=_DENSE_BLOCK_M,
        BLOCK_N=_DENSE_BLOCK_N,
        num_warps=_DENSE_NUM_WARPS,
    )
    return out


def swiglu_bwd(grad_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Gradient of :func:`swiglu_fwd` with respect to ``x``, both halves in one pass."""
    assert x.ndim == 2 and grad_out.ndim == 2, "x and grad_out must be 2-D"
    num_rows, half_cols = x.size(0), x.size(1) // 2
    assert grad_out.shape == (num_rows, half_cols), f"grad_out {tuple(grad_out.shape)} vs x {tuple(x.shape)}"
    grad_x = torch.empty(num_rows, 2 * half_cols, dtype=x.dtype, device=x.device)
    if grad_x.numel() == 0:
        return grad_x
    assert x.stride(1) == 1 and grad_out.stride(1) == 1, "x and grad_out must have a unit column stride"
    swiglu_dense_bwd_kernel[_dense_grid(num_rows, half_cols)](
        grad_out,
        x,
        grad_x,
        num_rows,
        half_cols,
        grad_out.stride(0),
        x.stride(0),
        grad_x.stride(0),
        BLOCK_M=_DENSE_BLOCK_M,
        BLOCK_N=_DENSE_BLOCK_N,
        num_warps=_DENSE_NUM_WARPS,
    )
    return grad_x


def swiglu_fwd_with_probs(
    x: torch.Tensor,
    probs: torch.Tensor,
    row_mask: Optional[torch.Tensor] = None,
    clamp_limit: Optional[float] = None,
):
    num_tokens, double_hidden_size = x.size()

    probs = probs.unsqueeze(-1)

    out = torch.empty(num_tokens, double_hidden_size // 2, dtype=x.dtype, device=x.device)

    has_clamp = clamp_limit is not None

    if row_mask is None:
        grid = (num_tokens,)
        swiglu_fwd_kernel[grid](
            x,
            probs,
            out,
            num_tokens=num_tokens,
            stride_x_token=x.stride(0),
            stride_probs_token=probs.stride(0),
            stride_out_token=out.stride(0),
            clamp_limit=float(clamp_limit) if has_clamp else 0.0,
            LOAD_WIDTH=triton.next_power_of_2(double_hidden_size // 2),
            HAS_CLAMP=has_clamp,
        )
    else:
        assert row_mask.is_cuda, "row_mask must be a CUDA tensor"

        BLOCK_SIZE = 8192
        grid = (BLOCK_SIZE,)
        swiglu_with_mask_fwd_kernel[grid](
            x,
            probs,
            row_mask,
            out,
            num_tokens=num_tokens,
            stride_x_token=x.stride(0),
            stride_probs_token=probs.stride(0),
            stride_out_token=out.stride(0),
            clamp_limit=float(clamp_limit) if has_clamp else 0.0,
            LOAD_WIDTH=triton.next_power_of_2(double_hidden_size // 2),
            BLOCK_SIZE=BLOCK_SIZE,
            HAS_CLAMP=has_clamp,
        )

    return out


def swiglu_bwd_with_probs(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    probs: torch.Tensor,
    row_mask: Optional[torch.Tensor] = None,
    clamp_limit: Optional[float] = None,
):
    num_tokens, hidden_size = grad_out.size()

    grad_x = torch.empty_like(x)
    grad_probs = torch.empty_like(probs)

    has_clamp = clamp_limit is not None

    if row_mask is None:
        grid = (num_tokens,)
        swiglu_bwd_kernel[grid](
            grad_out,
            x,
            probs,
            grad_x,
            grad_probs,
            num_tokens=num_tokens,
            stride_grad_out_token=grad_out.stride(0),
            stride_x_token=x.stride(0),
            stride_probs_token=probs.stride(0),
            stride_grad_x_token=grad_x.stride(0),
            stride_grad_probs_token=grad_probs.stride(0),
            clamp_limit=float(clamp_limit) if has_clamp else 0.0,
            LOAD_WIDTH=triton.next_power_of_2(hidden_size),
            HAS_CLAMP=has_clamp,
        )
    else:
        assert row_mask.is_cuda, "tokens_per_expert must be a CUDA tensor"

        BLOCK_SIZE = 8192
        grid = (BLOCK_SIZE,)
        swiglu_with_mask_bwd_kernel[grid](
            grad_out,
            x,
            probs,
            row_mask,
            grad_x,
            grad_probs,
            num_tokens=num_tokens,
            stride_grad_out_token=grad_out.stride(0),
            stride_x_token=x.stride(0),
            stride_probs_token=probs.stride(0),
            stride_grad_x_token=grad_x.stride(0),
            stride_grad_probs_token=grad_probs.stride(0),
            clamp_limit=float(clamp_limit) if has_clamp else 0.0,
            LOAD_WIDTH=triton.next_power_of_2(hidden_size),
            BLOCK_SIZE=BLOCK_SIZE,
            HAS_CLAMP=has_clamp,
        )

    return grad_x, grad_probs
