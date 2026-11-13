###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Dense SwiGLU MLP in MXFP8 on gfx1250 (the shared expert, a dense FFN): ``fc2(f(gate) * up)``.

    fwd  fc1 : l1 (bf16)                    = gemm(x, w1)                        (gemm_fp8's linear)
         glu : act_row, act_col             = glu_quant_mxfp8(l1)                (one pass: act never in bf16)
         fc2 : out                          = gemm(act_row, w2)
    bwd  fc2 : d_act (bf16), dw2            = fc2's dgrad / wgrad
         dglu: dl1_row, dl1_col             = dglu_quant_mxfp8(d_act, l1)        (grad_l1 never in bf16)
         fc1 : dx, dw1                      = fc1's dgrad / wgrad on the dGLU's copies

One autograd Function owns both linears, so neither fc2's activation nor fc1's output gradient is quantized
twice or kept in bf16; ``l1`` is the one bf16 tensor saved. The linears are gemm_fp8's (``_mx_linear_fwd`` /
``_mx_linear_bwd``): the same quantization, GEMMs, cached-weight and fused-wgrad handling, so the op equals
``gemm_fp8 -> SwiGLU -> gemm_fp8`` bit for bit when that SwiGLU computes the kernels' bf16 values.
"""

from typing import Union

import torch

from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
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
        lay2 = mx_gemm_layout(w2)
        recipes = (MX_A, MX_B) if lay2 else (ScalingRecipe(), ScalingRecipe())
        qdtype = _get_fp8_dtype(config.format, True)
        Q, rule, bits = _glu_quantizer(l1, qdtype, *recipes)
        qr, sr, qc, sc = Q.glu_quant_mxfp8(
            l1, None, activation, clamp_limit, qdtype, rule, pad=True, layout=bits
        )
        common = dict(shape=(M, I), orig_dtype=l1.dtype, dest_dtype=qdtype,
                      granularity=ScalingGranularity.MX_BLOCKWISE, block_size=config.block_size)  # fmt: skip
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
    num_cu: int | None = None,
    fuse_wgrad_accum_pattern: Union[None, str] = None,
    activation: Union[None, str] = None,
    clamp_limit: Union[None, float] = None,
) -> torch.Tensor:
    """Dense SwiGLU MLP in MXFP8 (gfx1250): ``fc2(f(gate) * up)``, the API of ``mlp_fp4``.

    Args:
        x: [M, K] bf16 activations, or a pre-quantized pair (``rmsnorm_mxfp8``'s; ``data_t`` the column
            copy) in the recipes fc1 reads (gemm_fp8's rule: MX_A / MX_B on gfx1250's GEMM layout).
        w1: [2I, K] fc1 weight (gate rows, then up), I % 128 == 0; w2: [K_out, I] fc2 weight. bf16, or
            cached pairs (``create_quantized_weight(..., need_weight_transpose_cache=True,
            scaling_recipe=mx_weight_layout_recipe(False, w.shape))``). NT only: ``trans_w1`` / ``trans_w2``.
        config: MX_BLOCKWISE Float8QuantConfig (block 32).
        num_cu: signature parity with ``mlp_fp4``; unused.
        fuse_wgrad_accum_pattern: ``"megatron"``: each wgrad accumulates into its weight's ``main_grad``
            (bf16 or fp32; ``main_grad`` / ``grad_added_to_main_grad`` on the weight or its pair's ``data``).
        activation: ``"silu"`` (default) or ``"gelu"`` (tanh); ``clamp_limit``: DeepSeek-V4's clamp, or None.
    """
    del num_cu
    if config is None or config.granularity != ScalingGranularity.MX_BLOCKWISE:
        raise ValueError("mlp_fp8: an MX_BLOCKWISE Float8QuantConfig is required")
    if not (trans_w1 and trans_w2):
        raise ValueError("mlp_fp8: NT only (trans_w1 = trans_w2 = True)")
    if not is_gfx1250():
        raise NotImplementedError("mlp_fp8: gfx1250 only (its fused GLU / dGLU + MXFP8 dual-cast kernels)")
    activation = activation or "silu"
    clamp_limit = _check_activation(activation, clamp_limit)
    x, x_t = (x.data, x.data_t) if isinstance(x, QuantizedTensorPair) else (x, None)
    w1, w1_t = (w1.data, w1.data_t) if isinstance(w1, QuantizedTensorPair) else (w1, None)
    w2, w2_t = (w2.data, w2.data_t) if isinstance(w2, QuantizedTensorPair) else (w2, None)
    if out_dtype is None:
        assert w1.dtype == w2.dtype, "w1 and w2 must have the same dtype"
        out_dtype = torch.promote_types(x.dtype, w1.dtype)
    return FP8MLPMXFunc.apply(
        x, w1, w2, x_t, w1_t, w2_t, activation, clamp_limit, out_dtype, config, fuse_wgrad_accum_pattern
    )
