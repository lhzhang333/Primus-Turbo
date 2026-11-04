###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""HipKittens BF16 GEMM for gfx1250 (MI455X).

``hk_gemm_tn(a, b)`` returns ``b @ a.T`` for ``a`` [M, K] and ``b`` [N, K], both K-contiguous BF16: the
forward GEMM of a linear layer with ``a`` the weight and ``b`` the activations (and the input-gradient
GEMM on a transposed weight). The result is row-major [N, M], i.e. ``C = A B^T`` column-major.

``hk_gemm_nt(a, b)`` returns ``b.T @ a`` for ``a`` [K, M] and ``b`` [K, N], both with K outermost: the
weight gradient of a linear layer, ``dW = dY^T x`` with ``a = x`` [tokens, in] and ``b = dY``
[tokens, out], with neither operand transposed. The result is row-major [N, M] (``dW`` [out, in]);
``accumulate=True`` adds it to ``out`` instead (``main_grad += dW``, one rounding, as a GEMM with beta = 1).

The kernels (``hk_gemm_gfx1250.cpp``) cover a leading block of C whose sides are multiples of a tile
cluster (TN: 896 or 1024 along M, 1024 along N; NT: 512 or 1024 along M, 1024 along N) with K a multiple
of 128; ``hk_gemm_tn`` / ``hk_gemm_nt`` run that block with the variant that fills the GPU best and
finish a ragged M or N edge with ``torch.mm``. A grid must fill the GPU: at least as many workgroups as
CUs (one tile per workgroup), and four waves or more unless the last is 85% full. A TN GEMM whose tiles
cover at most half the CUs but whose K is long splits K across 2-8 workgroups per tile (each split at least
2048 deep; FP32 partial tiles summed by a second kernel). They refuse (``hk_gemm_tn_supported`` /
``hk_gemm_nt_supported`` is False) when K is not a multiple of 128, the block would cover less than
three quarters of C, or no grid fills the GPU.
"""

from __future__ import annotations

import ctypes
import functools
import math
from typing import Optional, Tuple

import torch

from .build import HipKittensGemmUnavailable, hipkittens_include, load_library

__all__ = [
    "HipKittensGemmUnavailable",
    "hipkittens_include",
    "hk_gemm_available",
    "hk_gemm_nt",
    "hk_gemm_nt_out_ok",
    "hk_gemm_nt_plan",
    "hk_gemm_nt_supported",
    "hk_gemm_tn",
    "hk_gemm_tn_plan",
    "hk_gemm_tn_supported",
]

_MAX_BYTES = 2**32  # TDM tensor descriptors carry 32-bit extents: keep every operand under 4 GiB
_MIN_COVER = 0.75


@functools.lru_cache(maxsize=None)
def hk_gemm_available() -> bool:
    """gfx1250 and the kernel library builds and loads."""
    try:
        from primus_turbo.pytorch.core.utils import is_gfx1250

        if not (torch.cuda.is_available() and is_gfx1250()):
            return False
        load_library()
        return True
    except HipKittensGemmUnavailable:
        return False


@functools.lru_cache(maxsize=None)
def _variants(kind: str = "tn") -> dict:
    lib = load_library()
    query = lib.hk_gemm_tn_variant if kind == "tn" else lib.hk_gemm_nt_variant
    out, v = {}, 0
    while True:
        vals = [ctypes.c_int() for _ in range(4)]
        if query(v, *[ctypes.byref(x) for x in vals]) != 0:
            return out
        out[v] = tuple(x.value for x in vals)  # (BM, BN, CM, CN)
        v += 1


@functools.lru_cache(maxsize=None)
def _num_cus() -> int:
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


_SPLITS = (2, 3, 4, 6, 8)
_MIN_K_BLOCKS_PER_SPLIT = 16  # 2048 along K per split: the partial-tile traffic stays small next to the MMAs


def _grid_ok(units: int) -> Tuple[bool, float]:
    """Whether a grid of ``units`` workgroups (one per CU at a time) fills the GPU, and its last-wave fill."""
    if units < _num_cus():
        return False, 0.0
    waves = math.ceil(units / _num_cus())
    fill = units / (waves * _num_cus())
    # a short grid whose last wave is mostly empty: a library kernel with other tile shapes does better
    return not (waves < 4 and fill < 0.85), fill


def _plan(kind: str, M: int, N: int, K: int) -> Optional[Tuple[int, int, int, int]]:
    """``(variant, M_block, N_block, splits)`` among the ``kind`` variants: the one with the most useful work
    per CU slot (covered area times the occupancy of the last wave of workgroups, one workgroup per CU).
    TN shapes with too few tiles to fill the GPU and a long K split K across workgroups (``splits`` > 1)."""
    if K <= 0 or K % 128 or M <= 0 or N <= 0:
        return None
    best = None
    split_best = None
    for v, (bm, bn, cm, cn) in _variants(kind).items():
        mb, nb = M - M % (bm * cm), N - N % (bn * cn)
        if mb == 0 or nb == 0:
            continue
        tiles = (mb // bm) * (nb // bn)
        ok, fill = _grid_ok(tiles)
        if ok:
            score = (mb * nb) * fill
            if best is None or score > best[0] * 1.0001:
                best = (score, v, mb, nb, 1)
            continue
        if kind != "tn" or 2 * tiles > _num_cus():
            # split K only where the tiles cover at most half the CUs: with more, the FP32 partial tiles
            # cost more than a library kernel with smaller tiles loses (q_up's dgrad: 192 tiles, 0.86x)
            continue
        # one tile per workgroup leaves most CUs idle for the whole (long-K) GEMM: split K instead
        for splits in _SPLITS:
            if K // 128 < splits * _MIN_K_BLOCKS_PER_SPLIT:
                break
            ok, fill = _grid_ok(tiles * splits)
            if ok:
                score = (mb * nb) * fill
                if split_best is None or score > split_best[0] * 1.0001:
                    split_best = (score, v, mb, nb, splits)
                break
    if best is None:
        best = split_best
    if best is None or best[2] * best[3] < _MIN_COVER * M * N:
        return None
    return best[1], best[2], best[3], best[4]


@functools.lru_cache(maxsize=4096)
def hk_gemm_tn_plan(M: int, N: int, K: int) -> Optional[Tuple[int, int, int, int]]:
    """``(variant, M_block, N_block, splits)`` for ``hk_gemm_tn``, or None if it does not take this shape."""
    return _plan("tn", M, N, K)


@functools.lru_cache(maxsize=4096)
def hk_gemm_nt_plan(M: int, N: int, K: int) -> Optional[Tuple[int, int, int, int]]:
    """``(variant, M_block, N_block, splits)`` for ``hk_gemm_nt`` (splits is 1), or None if it does not take
    this shape."""
    return _plan("nt", M, N, K)


def hk_gemm_tn_supported(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether ``hk_gemm_tn(a, b)`` runs here (shape, dtype, layout, alignment, size)."""
    if not (
        a.dim() == 2
        and b.dim() == 2
        and a.dtype == torch.bfloat16
        and b.dtype == torch.bfloat16
        and a.is_cuda
        and b.is_cuda
        and a.device == b.device
        and a.is_contiguous()
        and b.is_contiguous()
        and a.shape[1] == b.shape[1]
        and a.data_ptr() % 256 == 0
        and b.data_ptr() % 256 == 0
        and max(a.numel(), b.numel(), a.shape[0] * b.shape[0]) * 2 < _MAX_BYTES
    ):
        return False
    return hk_gemm_available() and hk_gemm_tn_plan(a.shape[0], b.shape[0], a.shape[1]) is not None


def hk_gemm_tn(a: torch.Tensor, b: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``b @ a.T`` (row-major [N, M]) for BF16 ``a`` [M, K] and ``b`` [N, K]; see the module docstring."""
    if not hk_gemm_tn_supported(a, b):
        raise ValueError(f"hk_gemm_tn does not take a {tuple(a.shape)} x {tuple(b.shape)} {a.dtype} GEMM here")
    M, K = a.shape
    N = b.shape[0]
    if out is None:
        out = torch.empty((N, M), dtype=torch.bfloat16, device=a.device)
    elif not (out.shape == (N, M) and out.dtype == torch.bfloat16 and out.is_contiguous() and out.data_ptr() % 256 == 0):
        raise ValueError("out must be a contiguous, 256-byte aligned BF16 tensor of shape [N, M]")
    v, mb, nb, splits = hk_gemm_tn_plan(M, N, K)
    bm, bn, _, _ = _variants("tn")[v]
    stream = torch.cuda.current_stream(a.device).cuda_stream
    if splits > 1:
        ws = torch.empty(splits * mb * nb, dtype=torch.float32, device=a.device)
        rc = load_library().hk_gemm_tn_splitk(
            v, a.data_ptr(), b.data_ptr(), out.data_ptr(), ws.data_ptr(), M, N, K, mb // bm, nb // bn, splits, stream
        )
    else:
        rc = load_library().hk_gemm_tn(v, a.data_ptr(), b.data_ptr(), out.data_ptr(), M, N, K, mb // bm, nb // bn, stream)
    if rc != 0:
        raise RuntimeError(f"hk_gemm_tn launch failed ({rc}) for M={M} N={N} K={K} variant {v}")
    if mb < M:  # columns [mb, M) of the row-major result
        out[:, mb:] = torch.mm(b, a[mb:].t())
    if nb < N:  # rows [nb, N)
        torch.mm(b[nb:], a.t(), out=out[nb:])
    return out


def hk_gemm_nt_supported(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether ``hk_gemm_nt(a, b)`` runs here (shape, dtype, layout, alignment, size)."""
    if not (
        a.dim() == 2
        and b.dim() == 2
        and a.dtype == torch.bfloat16
        and b.dtype == torch.bfloat16
        and a.is_cuda
        and b.is_cuda
        and a.device == b.device
        and a.is_contiguous()
        and b.is_contiguous()
        and a.shape[0] == b.shape[0]
        and a.shape[1] % 8 == 0  # 16-byte rows for the TDM
        and b.shape[1] % 8 == 0
        and a.data_ptr() % 256 == 0
        and b.data_ptr() % 256 == 0
        and max(a.numel(), b.numel(), a.shape[1] * b.shape[1]) * 2 < _MAX_BYTES
    ):
        return False
    return hk_gemm_available() and hk_gemm_nt_plan(a.shape[1], b.shape[1], a.shape[0]) is not None


def hk_gemm_nt_out_ok(out: torch.Tensor, a: torch.Tensor, b: torch.Tensor, accumulate: bool = False) -> bool:
    """Whether ``out`` can take ``hk_gemm_nt(a, b)``: a contiguous BF16 [N, M] tensor on the device of ``a``,
    256-byte aligned to be overwritten, 16-byte aligned to be accumulated into (the alignment of a
    parameter's slice of a gradient buffer)."""
    return (
        out.shape == (b.shape[1], a.shape[1])
        and out.dtype == torch.bfloat16
        and out.device == a.device
        and out.is_contiguous()
        and out.data_ptr() % (16 if accumulate else 256) == 0
    )


def hk_gemm_nt(
    a: torch.Tensor, b: torch.Tensor, out: Optional[torch.Tensor] = None, accumulate: bool = False
) -> torch.Tensor:
    """``b.T @ a`` (row-major [N, M]) for BF16 ``a`` [K, M] and ``b`` [K, N]; see the module docstring.

    With ``accumulate`` the product is added to ``out`` (``out += b.T @ a``, the sum taken in FP32 and rounded
    to BF16 once, as a GEMM with beta = 1): a weight gradient accumulated straight into ``main_grad``."""
    if not hk_gemm_nt_supported(a, b):
        raise ValueError(f"hk_gemm_nt does not take a {tuple(a.shape)} x {tuple(b.shape)} {a.dtype} GEMM here")
    K, M = a.shape
    N = b.shape[1]
    if out is None:
        if accumulate:
            raise ValueError("accumulate needs out")
        out = torch.empty((N, M), dtype=torch.bfloat16, device=a.device)
    elif not hk_gemm_nt_out_ok(out, a, b, accumulate):
        raise ValueError(
            "out must be a contiguous BF16 tensor of shape [N, M], 256-byte aligned (16-byte to accumulate into)"
        )
    v, mb, nb, _ = hk_gemm_nt_plan(M, N, K)
    bm, bn, _, _ = _variants("nt")[v]
    stream = torch.cuda.current_stream(a.device).cuda_stream
    launch = load_library().hk_gemm_nt_accumulate if accumulate else load_library().hk_gemm_nt
    rc = launch(v, a.data_ptr(), b.data_ptr(), out.data_ptr(), M, N, K, mb // bm, nb // bn, stream)
    if rc != 0:
        raise RuntimeError(f"hk_gemm_nt launch failed ({rc}) for M={M} N={N} K={K} variant {v}")
    if accumulate:  # the edges as GEMMs with beta = 1; the corner [nb:, mb:] once
        if mb < M:
            out[:, mb:].addmm_(b.t(), a[:, mb:])
        if nb < N:
            out[nb:, :mb].addmm_(b[:, nb:].t(), a[:, :mb])
        return out
    if mb < M:  # columns [mb, M) of the row-major result
        out[:, mb:] = torch.mm(b.t(), a[:, mb:])
    if nb < N:  # rows [nb, N)
        torch.mm(b[:, nb:].t(), a, out=out[nb:])
    return out
