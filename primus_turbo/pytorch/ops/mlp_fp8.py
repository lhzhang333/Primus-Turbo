###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Dense GLU MLP in MXFP8 on gfx1250 (e.g. a shared expert): ``fc2(f(gate) * up)`` in one autograd Function.

fc1 and fc2 are gemm_fp8's MXFP8 linears (``_mx_linear_fwd`` / ``_mx_linear_bwd``); the GLU and its gradient are
fused with the MXFP8 dual cast of their outputs, so the activation and fc1's output gradient are never written in
bf16. The results equal ``gemm_fp8 -> GLU -> gemm_fp8`` bit for bit when that GLU computes the kernels' bf16 values.
"""

from typing import Union

import torch

from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
    ScalingRecipe,
)
from primus_turbo.pytorch.core.quantized_tensor import (
    QuantizedTensor,
    QuantizedTensorPair,
)
from primus_turbo.pytorch.core.utils import is_gfx1250
from primus_turbo.pytorch.kernels.quantization.quantization_impl import _gfx1250_mxfp8
from primus_turbo.pytorch.ops.gemm_fp8 import _mx_linear_bwd, _mx_linear_fwd
from primus_turbo.pytorch.ops.grouped_mlp_fp8 import _check_activation
from primus_turbo.pytorch.ops.utils import (
    MX_A,
    MX_B,
    _ensure_contiguous_grad_out,
    _get_fp8_dtype,
    _setup_fused_grad_accum,
    mx_gemm_layout,
)

__all__ = ["mlp_fp8"]


def _glu_quantizer(t, dtype, row, col):
    """(the gfx1250 quantizer module, its scale rule, the layout bits of the ``row`` / ``col`` recipes)."""
    q = _gfx1250_mxfp8(t, dtype, row, col)
    if q is None:
        raise NotImplementedError(f"mlp_fp8 (MXFP8): no gfx1250 fused GLU kernel for {t.dtype} -> {dtype}")
    return q


class FP8MLPMXFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x, w1, w2, x_t, w1_t, w2_t, activation, clamp_limit, out_dtype, config, fuse_wgrad_accum_pattern
    ):
        fuse1, main_grad1 = _setup_fused_grad_accum(w1, fuse_wgrad_accum_pattern)
        fuse2, main_grad2 = _setup_fused_grad_accum(w2, fuse_wgrad_accum_pattern)
        l1, saved1, lay1 = _mx_linear_fwd(x, w1, x_t, w1_t, out_dtype, config)
        M, I = l1.shape[0], l1.shape[1] // 2
        assert l1.shape[1] == 2 * I and I % 128 == 0, f"mlp_fp8: fc1 width 2I, I % 128 == 0: {l1.shape[1]}"
        # the activation is fc2's A operand: its two copies in the layout fc2 reads
        lay2 = mx_gemm_layout(w2, config=config, out_dtype=out_dtype)
        recipes = (MX_A, MX_B) if lay2 else (ScalingRecipe(), ScalingRecipe())
        qdtype = _get_fp8_dtype(config.format, True)
        Q, rule, bits = _glu_quantizer(l1, qdtype, *recipes)
        qr, sr, qc, sc = Q.glu_quant_mxfp8(
            l1, None, activation, clamp_limit, qdtype, rule, pad=True, layout=bits
        )
        common = dict(
            shape=(M, I),
            orig_dtype=l1.dtype,
            dest_dtype=qdtype,
            granularity=ScalingGranularity.MX_BLOCKWISE,
            block_size=config.block_size,
        )
        act = QuantizedTensor(qr, sr, scaling_recipe=recipes[0], quantized_axis=-1, **common)
        act_t = QuantizedTensor(qc, sc, scaling_recipe=recipes[1], quantized_axis=-2, **common)
        out, saved2, lay2 = _mx_linear_fwd(act, w2, act_t, w2_t, out_dtype, config)
        ctx.save_for_backward(*saved1, l1, *saved2)
        ctx.lays, ctx.fuse, ctx.main_grads = (lay1, lay2), (fuse1, fuse2), (main_grad1, main_grad2)
        ctx.act = (activation, clamp_limit)
        ctx.out_dtype, ctx.config = out_dtype, config
        return out

    @staticmethod
    def backward(ctx, grad_out):
        saved = ctx.saved_tensors
        saved1, l1, saved2 = saved[:4], saved[4], saved[5:]
        (lay1, lay2), (fuse1, fuse2), (mg1, mg2) = ctx.lays, ctx.fuse, ctx.main_grads
        cfg, out_dtype = ctx.config, ctx.out_dtype
        grad_out = _ensure_contiguous_grad_out(grad_out)
        d_act, grad_w2 = _mx_linear_bwd(grad_out, saved2, lay2, cfg, out_dtype, fuse2, mg2)
        # grad_l1 is fc1's output gradient: two A-layout copies for its dgrad and wgrad
        gdtype = _get_fp8_dtype(cfg.format, False)
        g = MX_A if lay1 else ScalingRecipe()
        Q, rule, bits = _glu_quantizer(d_act, gdtype, g, g)
        _, *grad_l1 = Q.dglu_quant_mxfp8(d_act, l1, None, *ctx.act, gdtype, rule, pad=True, layout=bits)
        grad_x, grad_w1 = _mx_linear_bwd(tuple(grad_l1), saved1, lay1, cfg, out_dtype, fuse1, mg1)
        return grad_x, grad_w1, grad_w2, None, None, None, None, None, None, None, None


@torch._dynamo.disable(recursive=True, reason="builds QuantizedTensor wrappers inside its autograd.Function")
def mlp_fp8(
    x: Union[torch.Tensor, QuantizedTensor, QuantizedTensorPair],
    w1: Union[torch.Tensor, QuantizedTensor, QuantizedTensorPair],
    w2: Union[torch.Tensor, QuantizedTensor, QuantizedTensorPair],
    trans_w1: bool = True,
    trans_w2: bool = True,
    out_dtype: Union[None, torch.dtype] = None,
    config: Union[Float8QuantConfig, None] = None,
    fuse_wgrad_accum_pattern: Union[None, str] = None,
    activation: Union[None, str] = None,
    clamp_limit: Union[None, float] = None,
) -> torch.Tensor:
    """Dense GLU MLP in MXFP8: ``fc2(f(gate) * up)``. gfx1250 only (NotImplementedError elsewhere): the fused GLU /
    dGLU + MXFP8 dual-cast kernels are that arch's.

    Args:
        x: [M, K] activations, or a pre-quantized pair (e.g. ``rmsnorm_mxfp8``'s; ``data_t`` the column copy) in
            the recipes fc1 reads (gemm_fp8's: ``MX_A`` / ``MX_B`` where ``mx_gemm_layout`` holds, raw else).
        w1: [2I, K] fc1 weight (gate rows, then up), I % 128 == 0; w2: [K_out, I] fc2 weight. Plain, or cached
            pairs (``create_quantized_weight(..., need_weight_transpose_cache=True)``). NT only: ``trans_w1`` /
            ``trans_w2`` must be True.
        out_dtype: bfloat16 (the default for bf16 inputs).
        config: MX_BLOCKWISE Float8QuantConfig, format E4M3 (the fused kernels quantize to E4M3 both ways).
        fuse_wgrad_accum_pattern: ``"megatron"``: each wgrad accumulates into its weight's ``main_grad``
            (bf16 or fp32; ``main_grad`` / ``grad_added_to_main_grad`` on the weight or its pair's ``data``).
        activation: ``"silu"`` (default) or ``"gelu"`` (tanh); ``clamp_limit``: DeepSeek-V4's clamp, or None.
    """
    if config is None or config.granularity != ScalingGranularity.MX_BLOCKWISE:
        raise ValueError("mlp_fp8: an MX_BLOCKWISE Float8QuantConfig is required")
    if config.format != Format.E4M3:
        raise ValueError(
            f"mlp_fp8: format E4M3 only (its fused GLU / dGLU quantize to E4M3), got {config.format}"
        )
    if not (trans_w1 and trans_w2):
        raise ValueError("mlp_fp8: NT only (trans_w1 = trans_w2 = True)")
    if not is_gfx1250():
        raise NotImplementedError("mlp_fp8: gfx1250 only (its fused GLU / dGLU + MXFP8 dual-cast kernels)")
    activation = activation or "silu"
    clamp_limit = _check_activation(activation, clamp_limit)
    x, x_t = (x.data, x.data_t) if isinstance(x, QuantizedTensorPair) else (x, None)
    w1, w1_t = (w1.data, w1.data_t) if isinstance(w1, QuantizedTensorPair) else (w1, None)
    w2, w2_t = (w2.data, w2.data_t) if isinstance(w2, QuantizedTensorPair) else (w2, None)
    if x.ndim != 2 or w1.ndim != 2 or w2.ndim != 2:
        raise ValueError(
            f"mlp_fp8: 2-D x, w1, w2 (flatten the leading dims), got {x.shape}, {w1.shape}, {w2.shape}"
        )
    if out_dtype is None:
        assert w1.dtype == w2.dtype, "w1 and w2 must have the same dtype"
        out_dtype = torch.promote_types(x.dtype, w1.dtype)
    if out_dtype != torch.bfloat16:
        raise ValueError(f"mlp_fp8: bfloat16 output only (the fused GLU / dGLU read bf16), got {out_dtype}")
    return FP8MLPMXFunc.apply(
        x, w1, w2, x_t, w1_t, w2_t, activation, clamp_limit, out_dtype, config, fuse_wgrad_accum_pattern
    )
