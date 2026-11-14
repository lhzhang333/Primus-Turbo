###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Helpers shared across the GEMM / grouped-GEMM op implementations."""

from typing import Optional

import torch

from primus_turbo.pytorch.core.backend import (
    BackendType,
    GlobalBackendManager,
    PrecisionType,
)
from primus_turbo.pytorch.core.low_precision import (
    Format,
    ScalingRecipe,
    float8_e4m3,
    float8_e5m2,
)
from primus_turbo.pytorch.core.utils import is_gfx942, is_gfx950, is_gfx1250

_ALL, _BF16_FP32 = (torch.bfloat16, torch.float16, torch.float32), (torch.bfloat16, torch.float32)
# main_grad dtypes an MXFP8 wgrad with a beta=1 epilogue (fuse_bgrad_accum_pattern) adds into, per (arch, grouped,
# backend); dense = gemm_fp8's wgrad, grouped = grouped_gemm_fp8's variable-K one. gfx950 FlyDSL: 16-bit only.
# gfx1250: FlyDSL, and Triton as its fallback (no hipBLASLt MX GEMM there).
_MX_WGRAD_ACCUM = {
    ("gfx950", False, BackendType.HIPBLASLT): _ALL,
    ("gfx950", False, BackendType.FLYDSL): _ALL[:2],
    ("gfx950", True, BackendType.TRITON): _ALL,
    ("gfx950", True, BackendType.FLYDSL): _ALL[:2],
    ("gfx1250", False, BackendType.FLYDSL): _BF16_FP32,
    ("gfx1250", True, BackendType.FLYDSL): _BF16_FP32,
    ("gfx1250", False, BackendType.TRITON): _ALL,
    ("gfx1250", True, BackendType.TRITON): _ALL,
}


def mx_wgrad_accum_backend(grouped: bool) -> BackendType:
    """Default backend of the MXFP8 wgrad that accumulates into ``main_grad`` (beta=1): FlyDSL on gfx1250, else
    dense hipBLASLt / grouped Triton (which takes the fp32 main_grad Megatron allocates by default)."""
    if is_gfx1250():
        return BackendType.FLYDSL
    return BackendType.TRITON if grouped else BackendType.HIPBLASLT


def mx_wgrad_accum_supported(grouped: bool, main_grad_dtype: torch.dtype) -> bool:
    """Whether the MXFP8 wgrad of ``grouped_gemm_fp8`` (``grouped``) / ``gemm_fp8`` can add into a ``main_grad`` of
    ``main_grad_dtype`` with the backend it would run (pinned through GlobalBackendManager or its env variable, else
    ``mx_wgrad_accum_backend``). Frameworks ask before passing fuse_bgrad_accum_pattern; where it is False they add
    the wgrad themselves."""
    if is_gfx942():
        return False
    get = GlobalBackendManager.get_grouped_gemm_backend if grouped else GlobalBackendManager.get_gemm_backend
    backend = getattr(get(PrecisionType.FP8), "backend", None) or mx_wgrad_accum_backend(grouped)
    arch = "gfx1250" if is_gfx1250() else "gfx950" if is_gfx950() else None
    return main_grad_dtype in _MX_WGRAD_ACCUM.get((arch, grouped, backend), ())


# gfx1250's MX GEMM layouts, written by the quantizers when a recipe's shuffle flags ask for them: packed E8M0 scales
# for every operand (A: shuffle_scale), 16x16 blocks for the B operands too (W, Wᵀ, wgrad's activation columns)
MX_A, MX_B = ScalingRecipe(shuffle_scale=True), ScalingRecipe(shuffle_scale=True, shuffle_out=True)
MX_W = ScalingRecipe(use_2d_block=True, shuffle_scale=True, shuffle_out=True)


def _mx1250_layout_ok(grouped: bool) -> bool:
    """Whether gfx1250's FlyDSL MX GEMMs of this op kind (dense / ``grouped``) run here: on gfx1250, with no other
    FP8 GEMM backend pinned."""
    if not is_gfx1250():
        return False
    get = GlobalBackendManager.get_grouped_gemm_backend if grouped else GlobalBackendManager.get_gemm_backend
    return getattr(get(PrecisionType.FP8), "backend", None) in (None, BackendType.FLYDSL)


def mx_weight_layout_recipe(grouped: bool, shape=None) -> ScalingRecipe:
    """Scaling recipe of a cached MXFP8 weight (both copies, ``create_quantized_weight(scaling_recipe=)``) of
    ``gemm_fp8`` / (``grouped``) the grouped ops: gfx1250's MX GEMM layout ``MX_W`` when its FlyDSL MX GEMMs of that
    kind take it here (and ``shape``'s last two dims are % 16, if given), else the raw 2-D block recipe. The op layer
    decides the same way (``mx_gemm_layout``), so a cache made with this recipe is what the op expects."""
    ok = _mx1250_layout_ok(grouped) and (shape is None or (shape[-1] % 16 == 0 and shape[-2] % 16 == 0))
    return MX_W if ok else ScalingRecipe(use_2d_block=True)


def mx_gemm_layout(w, grouped: bool = False, a=None) -> bool:
    """Whether the MX GEMMs of a ``gemm_fp8`` / (``grouped``) grouped op with weight ``w`` and input ``a`` run on
    gfx1250's GEMM layouts: a cached ``w``'s recipe says, else a quantized ``a``'s, else a plain ``w`` takes them when
    ``mx_weight_layout_recipe`` would cache it so."""
    from primus_turbo.pytorch.core.quantized_tensor import QuantizedTensor

    for t, flag in ((w, "shuffle_out"), (a, "shuffle_scale")):
        if isinstance(t, QuantizedTensor):
            return bool(t.scaling_recipe is not None and getattr(t.scaling_recipe, flag))
    return mx_weight_layout_recipe(grouped, w.shape) == MX_W


def mx_check_operand_recipe(who: str, name: str, t, want: ScalingRecipe, lay: bool) -> None:
    """Raise (not assert) when a pre-quantized MX operand ``t`` (e.g. ``rmsnorm_mxfp8``'s pair) was written in
    another layout than the consumer reads: ``want`` = ``MX_A`` / ``MX_B`` (gfx1250 GEMM layout, ``lay``) or raw."""
    got = t.scaling_recipe
    if got != want:
        raise ValueError(
            f"{who}: {name} was quantized with {got}, but this GEMM reads the "
            f"{'gfx1250 MX GEMM layout' if lay else 'raw MX layout'} and expects {want}; quantize it with the "
            f"recipes the consumer uses (row MX_A / column MX_B where mx_gemm_layout holds, raw ScalingRecipe() else)"
        )


def _get_fp8_dtype(format: Format, is_fwd_stage: bool):
    if format == Format.E4M3:
        return float8_e4m3
    elif format == Format.E5M2:
        return float8_e5m2
    elif format == Format.HYBRID:
        return float8_e4m3 if is_fwd_stage else float8_e5m2
    else:
        raise ValueError(f"Unsupported FP8 format: {format}")


def _ensure_contiguous_grad_out(grad_out: torch.Tensor) -> torch.Tensor:
    # Some upstream reductions can produce expanded zero-stride grad_out views.
    # Custom grouped GEMM kernels expect dense layouts.
    return grad_out if grad_out.is_contiguous() else grad_out.contiguous()


def _setup_fused_grad_accum(b, fuse_bgrad_accum_pattern: Optional[str]):
    """Resolve the weight's gradient-accumulation buffer for the fused wgrad path.

    Returns ``(enabled, main_grad)``. When enabled, the wgrad GEMM writes straight
    into ``main_grad`` and the Function must return no gradient for ``b``.

    Call this at the top of ``forward``: the flag has to be set while the weight
    object is still in hand, because the backward pass only sees the saved tensors.
    """
    if fuse_bgrad_accum_pattern is None:
        return False, None

    assert fuse_bgrad_accum_pattern in ["megatron"], (
        "Only megatron support gradient accumulation fusion currently"
    )

    assert hasattr(b, "grad_added_to_main_grad"), (
        "b.grad_added_to_main_grad must be set up before the backward pass."
    )
    assert hasattr(b, "main_grad"), "b.main_grad must be set up before the backward pass."
    assert isinstance(b.main_grad, torch.Tensor) and (b.main_grad.shape == b.shape), (
        "b.main_grad must be a tensor with the same shape as b"
    )

    # Set in forward, not backward: autograd hands backward the saved tensors, not
    # the parameter object that carries this attribute.
    b.grad_added_to_main_grad = True
    return True, b.main_grad


_dummy_wgrads = {}


def _get_dummy_wgrad(shape: list, dtype: torch.dtype, zero=False) -> torch.Tensor:
    """Returns a dummy tensor of given shape.

    Supports arbitrary rank (2D for plain Linear weights, 3D for stacked
    grouped-linear weights ``(num_gemms, out_features, in_features)``, etc.).
    One flat buffer per dtype serves every shape: the dummy is a contiguous view of
    its first ``numel`` elements, so the dummies cost the largest weight once instead
    of one persistent buffer per distinct weight shape (DeepSeek-V3 with MXFP8 experts
    and dense linears: 1.75 GiB instead of 3.1 GiB). The buffer only grows; views
    handed out earlier keep the old one alive until autograd drops them (the
    framework's hook sets ``param.grad = None`` right after). The contents are never
    read (``zero=True``: zeroed for frameworks that add the dummy, e.g. Megatron's
    ``zero_out_wgrad``); views of one buffer may alias each other, which is harmless
    for the same reason. A contiguous view keeps the gradient layout contract, so
    AccumulateGrad still takes it without a copy.
    """
    global _dummy_wgrads
    numel = 1
    for d in shape:
        numel *= int(d)
    key = (dtype, torch.cuda.current_device())
    buf = _dummy_wgrads.get(key)
    if buf is None or buf.numel() < numel:
        buf = _dummy_wgrads[key] = torch.empty(numel, dtype=dtype, device="cuda", requires_grad=False)
    view = buf[:numel].view(*shape)
    if zero:
        view.fill_(0)
    return view.detach()
