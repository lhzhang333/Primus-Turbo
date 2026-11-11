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
    float8_e4m3,
    float8_e5m2,
)
from primus_turbo.pytorch.core.utils import is_gfx942, is_gfx950, is_gfx1250

# MXFP8 wgrad GEMMs with a beta=1 epilogue (fuse_bgrad_accum_pattern) and the main_grad dtypes each one accumulates
# into: dense = the wgrad of gemm_fp8, grouped = the variable-K wgrad of grouped_gemm_fp8. FlyDSL's are gfx950-only and
# write 16-bit only.
_MX_WGRAD_ACCUM_OUT_DTYPES = {
    False: {
        BackendType.HIPBLASLT: (torch.bfloat16, torch.float16, torch.float32),
        BackendType.TRITON: (torch.bfloat16, torch.float16, torch.float32),
        BackendType.FLYDSL: (torch.bfloat16, torch.float16),
    },
    True: {
        BackendType.TRITON: (torch.bfloat16, torch.float16, torch.float32),
        BackendType.FLYDSL: (torch.bfloat16, torch.float16),
    },
}


def mx_wgrad_accum_backend(grouped: bool) -> BackendType:
    """Default backend of the MXFP8 wgrad that accumulates into ``main_grad`` (beta=1), in one place so a new kernel
    switches it here: dense hipBLASLt (Triton on gfx1250, where hipBLASLt's MX GEMM is wrong); grouped Triton, which
    also takes the fp32 main_grad Megatron allocates by default (FlyDSL's gfx950 epilogue writes only 16-bit; pin it
    with GlobalBackendManager for a bf16 main_grad)."""
    if grouped or is_gfx1250():
        return BackendType.TRITON
    return BackendType.HIPBLASLT


def mx_wgrad_accum_supported(grouped: bool, main_grad_dtype: torch.dtype) -> bool:
    """Whether the MXFP8 wgrad of ``grouped_gemm_fp8`` (``grouped``) or ``gemm_fp8`` can add into a ``main_grad`` of
    ``main_grad_dtype`` on this GPU with the backend it would run: the FP8 backend pinned through GlobalBackendManager
    (or its env variable), else ``mx_wgrad_accum_backend``. Auto-tune only picks among backends that accept the
    accumulating call, so it is answered for the default. Frameworks ask before passing fuse_bgrad_accum_pattern."""
    if is_gfx942():  # no MXFP8
        return False
    pinned = (GlobalBackendManager.get_grouped_gemm_backend if grouped else GlobalBackendManager.get_gemm_backend)(
        PrecisionType.FP8
    )
    backend = getattr(pinned, "backend", None) or mx_wgrad_accum_backend(grouped)
    if backend == BackendType.FLYDSL and not is_gfx950():
        return False
    if backend == BackendType.HIPBLASLT and is_gfx1250():  # GEMMFP8HipBLASLtBackend refuses MX there
        return False
    if backend == BackendType.TRITON and not grouped and not is_gfx1250():  # the dense Triton MX path is gfx1250's
        return False
    return main_grad_dtype in _MX_WGRAD_ACCUM_OUT_DTYPES[grouped].get(backend, ())


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
    Tensors are cached by ``(shape, dtype)`` so each distinct weight layout
    only allocates one persistent buffer that gets reused across steps.
    """
    global _dummy_wgrads
    key = (tuple(shape), dtype)
    if key not in _dummy_wgrads:
        _dummy_wgrads[key] = torch.empty(
            shape,
            dtype=dtype,
            device="cuda",
            requires_grad=False,
        )
    if zero:
        _dummy_wgrads[key].fill_(0)
    return _dummy_wgrads[key].detach()
