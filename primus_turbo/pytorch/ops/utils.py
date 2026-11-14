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


# gfx1250's MX GEMM layouts, written by its quantizers when a recipe's shuffle flags ask for them: packed E8M0 scales
# for every operand (A: shuffle_scale), 16x16 blocks for the B operands too (W, Wᵀ, wgrad's activation columns). On
# other archs the same flags mean other layouts (gfx950: the CDNA4 shuffles), so only gfx1250 reads them as these.
MX_A, MX_B = ScalingRecipe(shuffle_scale=True), ScalingRecipe(shuffle_scale=True, shuffle_out=True)
MX_W = ScalingRecipe(use_2d_block=True, shuffle_scale=True, shuffle_out=True)


def _mx1250_layout_ok(grouped: bool) -> bool:
    """Whether gfx1250's FlyDSL MX GEMMs of this op kind (dense / ``grouped``) run here: on gfx1250, with no other
    FP8 GEMM backend pinned."""
    if not is_gfx1250():
        return False
    get = GlobalBackendManager.get_grouped_gemm_backend if grouped else GlobalBackendManager.get_gemm_backend
    return getattr(get(PrecisionType.FP8), "backend", None) in (None, BackendType.FLYDSL)


def _mx1250_layout_dtypes_ok(config=None, out_dtype=None) -> bool:
    """The formats the gfx1250 MX GEMM layouts are written in (its quantizers take bf16 -> E4M3 only): E4M3 forward
    and backward (not E5M2 / HYBRID) and a bf16 output, each where given."""
    return (config is None or config.format == Format.E4M3) and out_dtype in (None, torch.bfloat16)


def mx_weight_layout_recipe(grouped: bool, shape, *, config=None, dtype=None) -> ScalingRecipe:
    """Scaling recipe for the cached MXFP8 copies of a weight of ``shape`` (both copies,
    ``create_quantized_weight(scaling_recipe=)``) of ``gemm_fp8`` / (``grouped``) the grouped ops: gfx1250's MX GEMM
    layout ``MX_W`` where its FlyDSL MX GEMMs of that kind run (gfx1250, no other FP8 GEMM backend pinned) and take
    the shape (N and K % 16; grouped: % 32), for an E4M3 ``config`` and a bf16 weight ``dtype`` where given; else the
    raw 2-D block recipe. ``mx_gemm_layout`` decides the same way, so the op reads such a cache in the layout it
    expects."""
    align = 32 if grouped else 16
    ok = _mx1250_layout_ok(grouped) and shape[-1] % align == 0 and shape[-2] % align == 0
    ok = ok and _mx1250_layout_dtypes_ok(config) and dtype in (None, torch.bfloat16)
    return MX_W if ok else ScalingRecipe(use_2d_block=True)


def mx_gemm_layout(w, grouped: bool = False, a=None, config=None, out_dtype=None) -> bool:
    """Whether the MX GEMMs of a ``gemm_fp8`` / (``grouped``) grouped op with weight ``w``, input ``a``, ``config``
    and ``out_dtype`` run on gfx1250's MX GEMM layouts (``MX_A`` / ``MX_B`` / ``MX_W``) rather than the raw ones.
    Never off gfx1250. A cached ``w``'s recipe decides, else a quantized ``a``'s; ValueError when the layout they
    force cannot run here (another FP8 GEMM backend pinned, a format other than E4M3, a non-bf16 output, or a
    non-bf16 operand to quantize into it). Else a plain bf16 ``w`` (and ``a``) takes the layout where
    ``mx_weight_layout_recipe`` would cache it so."""
    if not is_gfx1250():
        return False
    from primus_turbo.pytorch.core.quantized_tensor import QuantizedTensor

    def bf16(t):  # an operand the op quantizes itself: gfx1250's quantizers write the layouts from bf16 only
        return t is None or isinstance(t, QuantizedTensor) or t.dtype == torch.bfloat16

    ok = _mx1250_layout_ok(grouped) and _mx1250_layout_dtypes_ok(config, out_dtype) and bf16(w) and bf16(a)
    for t, flag, what in ((w, "shuffle_out", "cached weight"), (a, "shuffle_scale", "quantized input")):
        if isinstance(t, QuantizedTensor):
            if t.scaling_recipe is None or not getattr(t.scaling_recipe, flag):
                return False
            if not ok:
                raise ValueError(
                    f"MXFP8 {'grouped ' if grouped else ''}GEMM: the {what} is in gfx1250's MX GEMM layout "
                    f"({t.scaling_recipe}), which only gfx1250's FlyDSL MX GEMMs read (no other FP8 GEMM backend "
                    f"pinned), for E4M3 data and bf16 activations and output; got format "
                    f"{getattr(config, 'format', None)}, input {getattr(a, 'dtype', None)}, weight {w.dtype}, "
                    f"out_dtype {out_dtype}. Quantize it with the raw recipe (mx_weight_layout_recipe says which)."
                )
            return True
    return ok and mx_weight_layout_recipe(grouped, w.shape) == MX_W


def mx_check_operand_recipes(who: str, lay: bool, **copies) -> None:
    """gfx1250: raise ValueError when a pre-quantized MX operand copy (``name=(tensor or None, the recipe it must
    carry)``, e.g. ``a`` / ``a_t`` / ``b`` / ``b_t``) was written in another layout than the GEMMs of ``who`` read
    (``lay``: gfx1250's MX GEMM layout, else the raw one). Elsewhere a no-op: the ops keep their assertions there."""
    if not is_gfx1250():
        return
    from primus_turbo.pytorch.core.quantized_tensor import QuantizedTensor

    for name, (t, want) in copies.items():
        if isinstance(t, QuantizedTensor) and t.scaling_recipe != want:
            raise ValueError(
                f"{who}: {name} was quantized with {t.scaling_recipe}, but this GEMM reads the "
                f"{'gfx1250 MX GEMM layout' if lay else 'raw MX layout'} and expects {want}; quantize it with the "
                f"recipes the op uses (MX_A rows / MX_B columns / MX_W weights where mx_gemm_layout holds, the "
                f"raw ones else)"
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
