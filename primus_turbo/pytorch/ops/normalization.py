###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Triton-backed RMSNorm ops (standard + fused residual variant).

Public API:
    - ``rmsnorm(x, gamma, eps=1e-6, zero_centered=False) -> y``
    - ``rmsnorm_residual(x, residual, gamma, eps=1e-6) -> (y, x_plus_r)``
    - ``rmsnorm_residual_fp4(...) -> (y, x_plus_r, QuantizedTensorPair)``
    - ``rmsnorm_mxfp8(x, gamma, eps=1e-6) -> (y, QuantizedTensorPair)``
"""

from __future__ import annotations

from typing import Tuple, Union

import torch

from primus_turbo.pytorch.core.low_precision import (
    MXFP8_BLOCK_SIZE,
    Float4QuantConfig,
    ScalingGranularity,
    ScalingRecipe,
    check_mxfp4_support,
    float4_e2m1fn_x2,
    float8_e4m3,
)
from primus_turbo.pytorch.core.quantized_tensor import (
    QuantizedTensor,
    QuantizedTensorPair,
)
from primus_turbo.pytorch.kernels.normalization.rmsnorm_impl import (
    rmsnorm_bwd_impl,
    rmsnorm_bwd_residual_impl,
    rmsnorm_fwd_impl,
    rmsnorm_fwd_residual_impl,
)

__all__ = ["rmsnorm", "rmsnorm_residual", "rmsnorm_residual_fp4", "rmsnorm_mxfp8"]


class _RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6, zero_centered: bool = False):
        assert x.is_cuda and gamma.is_cuda, "rmsnorm: x and gamma must be CUDA tensors"
        orig_shape = x.shape
        H = gamma.shape[0]
        assert orig_shape[-1] == H, (
            f"rmsnorm: last dim of x ({orig_shape[-1]}) must equal gamma.shape[0] ({H})"
        )

        y, x2, rstd, BLOCK_H, ROWS, num_warps, num_stages = rmsnorm_fwd_impl(x, gamma, eps, zero_centered)

        ctx.save_for_backward(x2, gamma, rstd)
        ctx.eps = eps
        ctx.zero_centered = zero_centered
        ctx.orig_shape = orig_shape
        ctx.BLOCK_H = BLOCK_H
        ctx.ROWS = ROWS
        ctx.num_warps = num_warps
        ctx.num_stages = num_stages
        return y.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x2, gamma, rstd = ctx.saved_tensors
        dx, dg = rmsnorm_bwd_impl(
            grad_out,
            x2,
            gamma,
            rstd,
            ctx.BLOCK_H,
            ctx.ROWS,
            ctx.num_warps,
            ctx.num_stages,
            ctx.zero_centered,
        )
        return dx.reshape(ctx.orig_shape), dg, None, None


class _RMSNormResidualFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, residual: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6):
        assert x.is_cuda and residual.is_cuda and gamma.is_cuda, (
            "rmsnorm_residual: x, residual and gamma must be CUDA tensors"
        )
        assert x.shape == residual.shape, (
            f"rmsnorm_residual: shape mismatch {tuple(x.shape)} vs {tuple(residual.shape)}"
        )
        orig_shape = x.shape
        H = gamma.shape[0]
        assert orig_shape[-1] == H

        y, x_plus_r, rstd, BLOCK_H, ROWS, num_warps, num_stages = rmsnorm_fwd_residual_impl(
            x, residual, gamma, eps
        )

        ctx.save_for_backward(x_plus_r, gamma, rstd)
        ctx.eps = eps
        ctx.orig_shape = orig_shape
        ctx.BLOCK_H = BLOCK_H
        ctx.ROWS = ROWS
        ctx.num_warps = num_warps
        ctx.num_stages = num_stages
        return y.reshape(orig_shape), x_plus_r.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor, grad_xpr: torch.Tensor):
        x_plus_r, gamma, rstd = ctx.saved_tensors
        dx, dg = rmsnorm_bwd_residual_impl(
            grad_y,
            grad_xpr,
            x_plus_r,
            gamma,
            rstd,
            ctx.BLOCK_H,
            ctx.ROWS,
            ctx.num_warps,
            ctx.num_stages,
        )
        dx_out = dx.reshape(ctx.orig_shape)
        # Jacobian of add() is [I, I] -> both x and residual get the same grad.
        return dx_out, dx_out, dg, None


class _RMSNormResidualWithRstdFunction(torch.autograd.Function):
    """Same as residual RMSNorm but also returns Triton's ``rstd`` (f32, non-diff).

    R3 reads ``rstd`` for the FlyDSL dual under ``no_grad``. Returning Float4 from
    this Function hits unimplemented ``fill_cuda``; f32 ``rstd`` is safe.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        residual: torch.Tensor,
        gamma: torch.Tensor,
        eps: float,
        skip_y_store: bool,
    ):
        assert x.is_cuda and residual.is_cuda and gamma.is_cuda
        assert x.shape == residual.shape
        orig_shape = x.shape
        H = gamma.shape[0]
        assert orig_shape[-1] == H

        skip = bool(skip_y_store)
        y, x_plus_r, rstd, BLOCK_H, ROWS, num_warps, num_stages = rmsnorm_fwd_residual_impl(
            x, residual, gamma, eps, skip_y_store=skip
        )
        ctx.save_for_backward(x_plus_r, gamma, rstd)
        ctx.eps = eps
        ctx.orig_shape = orig_shape
        ctx.BLOCK_H = BLOCK_H
        ctx.ROWS = ROWS
        ctx.num_warps = num_warps
        ctx.num_stages = num_stages
        y_out = y.reshape(orig_shape)
        xpr_out = x_plus_r.reshape(orig_shape)
        # y stays a differentiable STE for a consumer of the quantized pair, so it must
        # not be marked non-differentiable even when its store is skipped.
        ctx.mark_non_differentiable(rstd)
        return y_out, xpr_out, rstd

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor, grad_xpr: torch.Tensor, grad_rstd: torch.Tensor):
        del grad_rstd
        x_plus_r, gamma, rstd = ctx.saved_tensors
        if grad_y is None:
            grad_y = torch.zeros(ctx.orig_shape, device=x_plus_r.device, dtype=x_plus_r.dtype)
        dx, dg = rmsnorm_bwd_residual_impl(
            grad_y,
            grad_xpr,
            x_plus_r,
            gamma,
            rstd,
            ctx.BLOCK_H,
            ctx.ROWS,
            ctx.num_warps,
            ctx.num_stages,
        )
        dx_out = dx.reshape(ctx.orig_shape)
        return dx_out, dx_out, dg, None, None


def rmsnorm(
    x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6, zero_centered: bool = False
) -> torch.Tensor:
    """RMSNorm.

    Args:
        x: input tensor; normalization is over the last dim.
        gamma: learnable gain of shape ``[x.shape[-1]]``.
        eps: variance epsilon.
        zero_centered: if True, the effective gain is ``(1 + gamma)`` (computed in
            fp32). Initialize ``gamma`` to zeros in this mode.
    """
    return _RMSNormFunction.apply(x, gamma, eps, zero_centered)


def rmsnorm_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _RMSNormResidualFunction.apply(x, residual, gamma, eps)


class _RMSNormFP4STE(torch.autograd.Function):
    """Quantize ``y`` without reading it, and keep it on the gradient path.

    The pair is computed from ``xpr``/``rstd``/``gamma`` in f32 registers, so the
    values never depend on ``y`` -- that is the whole point, and it is what lets the
    caller skip the bf16 ``y`` store. But a consumer of the pair still has to send its
    dgrad back to the norm, and a ``QuantizedTensor`` built outside the graph is a leaf.
    Taking ``y`` as the input and handing its gradient straight through makes the pair a
    graph node whose backward lands on ``y``, i.e. a straight-through estimator.

    The Float4 outputs are safe here because they are consumed: autograd never has to
    synthesise a gradient for them (which is where ``fill_cuda`` is unimplemented).
    """

    @staticmethod
    def forward(ctx, y, xpr2, rstd, gamma, config, row_recipe, col_recipe, dual_quant):
        with torch.no_grad():
            gamma_f32 = gamma.detach().to(dtype=torch.float32).contiguous()
            row, row_scale, col, col_scale = dual_quant(
                xpr2,
                rstd.reshape(-1).contiguous(),
                gamma_f32,
                float4_e2m1fn_x2,
                col_rht=col_recipe.use_rht,
                scale_rounding_mode=config.scale_rounding_mode,
            )

        def _wrap(data, scale_inv, shape, recipe, axis):
            return QuantizedTensor(
                data,
                scale_inv,
                shape=shape,
                orig_dtype=y.dtype,
                dest_dtype=float4_e2m1fn_x2,
                granularity=config.granularity,
                block_size=config.block_size,
                scaling_recipe=recipe,
                scale_rounding_mode=config.scale_rounding_mode,
                quantized_axis=axis,
                requires_grad=y.requires_grad,
            )

        M, K = xpr2.shape
        return (
            _wrap(row, row_scale, torch.Size((M, K)), row_recipe, -1),
            _wrap(col, col_scale, torch.Size((K, M)), col_recipe, -2),
        )

    @staticmethod
    def backward(ctx, grad_row, grad_col):
        # The consumer's dgrad arrives on the row-wise operand; the col-wise one is a
        # wgrad operand, which produces no gradient of its own input.
        del grad_col
        return grad_row, None, None, None, None, None, None, None


def rmsnorm_residual_fp4(
    x: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = 1e-6,
    config: Union[Float4QuantConfig, None] = None,
    skip_y_store: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, QuantizedTensorPair]:
    """Residual RMSNorm whose ``y`` comes back already quantized, without a second pass.

    Returns ``(y, x_plus_r, y_fp4)``. ``y_fp4.data`` is the row-wise operand and
    ``y_fp4.data_t`` the col-wise (RHT) one, so it feeds :func:`gemm_fp4` or
    :func:`mlp_fp4` as ``a`` / ``x`` with no further quantization.

    ``y`` and ``x_plus_r`` both carry autograd; ``y`` stays the STE so a consumer
    of ``y_fp4`` still sends its dgrad back through RMSNorm. Triton's ``rstd`` is
    a non-differentiable f32, which is what lets the dual run under ``no_grad``
    (a Float4 tensor must not enter ``run_backward``: ``fill_cuda`` is
    unimplemented).

    ``skip_y_store``: skip the bf16 ``y`` HBM store. Legal only when every consumer
    reads ``y_fp4``; a consumer that quantizes bf16 ``y`` itself (the final LN
    feeding the LM head) must leave this False.

    The scales come out of the norm's own registers rather than from a reload of
    ``y``, so this is not bit-identical to quantizing a materialized bf16 ``y``:
    ``rstd * gamma`` is applied in f32 and the bf16 rounding never happens. It is
    the more accurate of the two.
    """
    from primus_turbo.flydsl.quantization.mxfp4_quant_kernel import flydsl_rmsnorm_dual_quant

    if config is None:
        config = Float4QuantConfig()
    supported, reason = check_mxfp4_support()
    assert supported, reason
    assert config.granularity == ScalingGranularity.MX_BLOCKWISE, (
        f"rmsnorm_residual_fp4 currently covers MX_BLOCKWISE only, got {config.granularity}"
    )
    assert not config.use_preshuffle, "the fused dual emits raw (non-preshuffled) scales"
    assert x.dtype == torch.bfloat16, f"the fused dual is bf16-only, got {x.dtype}"

    row_recipe = ScalingRecipe()
    col_recipe = ScalingRecipe(use_rht=True)

    y, x_plus_r, rstd = _RMSNormResidualWithRstdFunction.apply(x, residual, gamma, eps, skip_y_store)
    H = gamma.shape[0]
    xpr2 = x_plus_r.reshape(-1, H).contiguous()
    row_qt, col_qt = _RMSNormFP4STE.apply(
        y, xpr2, rstd, gamma, config, row_recipe, col_recipe, flydsl_rmsnorm_dual_quant
    )
    return y, x_plus_r, QuantizedTensorPair(row_qt, col_qt)


def _rmsnorm_mxfp8_fwd(x2, gamma, eps, write_y, row_recipe, col_recipe):
    """(y or None, rstd, q_row, s_row, q_col, s_col): gfx1250's fused FlyDSL kernels when they take the call
    (bf16, H % 8 == 0 and H > 512, 16-byte aligned rows), else Turbo's RMSNorm then the MXFP8 dual
    quantization of its output."""
    from primus_turbo.pytorch.kernels.quantization.quantization_impl import (
        _gfx1250_mxfp8,
        quantize_mxfp8_impl,
    )

    H = x2.shape[1]
    blocks_1d = not (row_recipe.use_2d_block or col_recipe.use_2d_block)
    q = _gfx1250_mxfp8(x2, float8_e4m3, row_recipe, col_recipe, blocks_1d)
    if (
        q is not None
        and gamma.dtype == torch.bfloat16
        and H % 8 == 0
        and H > 512  # narrower (kv_layernorm): two launches cost more than the bf16 round trip saves
        and x2.stride(1) == 1
        and x2.stride(0) % 8 == 0
        and x2.data_ptr() % 16 == 0
    ):
        from primus_turbo.flydsl.quantization.rmsnorm_mxfp8_gfx1250 import (
            rmsnorm_quant_mxfp8,
        )

        _, rule, lay = q
        return rmsnorm_quant_mxfp8(x2, gamma, eps, rule=rule, pad=True, layout=lay, write_y=write_y)
    y, _, rstd, *_ = rmsnorm_fwd_impl(x2, gamma, eps)
    out = quantize_mxfp8_impl(y, float8_e4m3, None, MXFP8_BLOCK_SIZE, True, row_recipe, col_recipe)
    return (y, rstd, *out)


class _RMSNormMXFP8Function(torch.autograd.Function):
    """RMSNorm forward that also returns its output MXFP8-quantized; one backward for both.

    The row copy is the consumer GEMM's A operand, so the dgrad that GEMM returns for it is a gradient of ``y``:
    the backward adds it to ``y``'s own (other consumers) and runs Turbo's RMSNorm backward with the saved
    ``rstd``. The column copy (wgrad's operand) gets none. Missing gradients stay ``None`` (no zero fill).
    """

    @staticmethod
    def forward(ctx, x, gamma, eps, write_y, row_recipe, col_recipe):
        ctx.set_materialize_grads(False)
        H = gamma.shape[0]
        assert x.shape[-1] == H, (
            f"rmsnorm_mxfp8: last dim of x ({x.shape[-1]}) must equal gamma.shape[0] ({H})"
        )
        x2 = x.reshape(-1, H)
        y, rstd, qr, sr, qc, sc = _rmsnorm_mxfp8_fwd(x2, gamma, eps, write_y, row_recipe, col_recipe)
        ctx.save_for_backward(x2, gamma, rstd)
        ctx.orig_shape = x.shape
        common = dict(
            shape=x2.shape,
            orig_dtype=x.dtype,
            dest_dtype=float8_e4m3,
            granularity=ScalingGranularity.MX_BLOCKWISE,
            block_size=MXFP8_BLOCK_SIZE,
            requires_grad=x.requires_grad,
        )
        row = QuantizedTensor(qr, sr, scaling_recipe=row_recipe, quantized_axis=-1, **common)
        col = QuantizedTensor(qc, sc, scaling_recipe=col_recipe, quantized_axis=-2, **common)
        return (y.view(x.shape) if write_y else None), row, col

    @staticmethod
    def backward(ctx, grad_y, grad_row, grad_col):
        del grad_col
        x2, gamma, rstd = ctx.saved_tensors
        grads = [g.reshape(x2.shape) for g in (grad_y, grad_row) if g is not None]
        if not grads:
            return None, None, None, None, None, None
        dy = grads[0] if len(grads) == 1 else grads[0] + grads[1]
        dx, dg = rmsnorm_bwd_impl(dy, x2, gamma, rstd, 0, 0, 0, 0)
        return dx.reshape(ctx.orig_shape), dg, None, None, None, None


def rmsnorm_mxfp8(
    x: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = 1e-6,
    *,
    write_y: bool = True,
    row_recipe: Union[ScalingRecipe, None] = None,
    col_recipe: Union[ScalingRecipe, None] = None,
) -> Tuple[Union[torch.Tensor, None], QuantizedTensorPair]:
    """RMSNorm (``(x * rstd) * gamma``, fp32 math) whose output also comes back MXFP8-quantized.

    Returns ``(y, QuantizedTensorPair(data=row, data_t=col))``: E4M3, one E8M0 scale per 32 elements, K (= H) and M
    padded to 128 as the HIP quantizer pads them. ``row`` (``quantized_axis=-1``) is a consumer GEMM's A operand,
    ``col`` (``-2``) its wgrad operand: pass the pair as ``a`` to ``gemm_fp8`` and nothing is quantized again.
    Gradients flow back through both ``y`` and ``row``.

    ``row_recipe`` / ``col_recipe``: the layouts the consumer reads. Default: gfx1250's MX GEMM layouts for bf16 x
    there (``ScalingRecipe(shuffle_scale=True)`` rows, ``ScalingRecipe(shuffle_scale=True, shuffle_out=True)``
    columns), raw ``ScalingRecipe()`` otherwise.

    The outputs are those of Turbo's ``rmsnorm`` followed by ``quantize_fp8_with_trans``, bit for bit: one fused pass
    on gfx1250 (bf16 x and gamma, H % 8 == 0 and H > 512, 16-byte aligned rows; x may be a strided row view), those
    two ops otherwise. ``write_y=False`` returns ``None`` for ``y`` (every consumer must read the pair).
    """
    from primus_turbo.pytorch.core.utils import is_gfx1250

    if row_recipe is None or col_recipe is None:
        # gfx1250's GEMM layouts where its quantizers write them (bf16 x), raw ones otherwise
        mx = is_gfx1250() and x.dtype == torch.bfloat16
        row_recipe = row_recipe or ScalingRecipe(shuffle_scale=mx)
        col_recipe = col_recipe or ScalingRecipe(shuffle_scale=mx, shuffle_out=mx)
    y, row, col = _RMSNormMXFP8Function.apply(x, gamma, eps, write_y, row_recipe, col_recipe)
    return y, QuantizedTensorPair(row, col)
