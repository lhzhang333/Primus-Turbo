###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""HipKittens BF16 GEMM for gfx1250 (MI455X).

``hk_gemm_tn(a, b)`` returns ``b @ a.T`` for ``a`` [M, K] and ``b`` [N, K], both K-contiguous BF16: the
forward GEMM of a linear layer with ``a`` the weight and ``b`` the activations (and the input-gradient
GEMM on a transposed weight). The result is row-major [N, M], i.e. ``C = A B^T`` column-major.

The kernels (``hk_gemm_gfx1250.cpp``) cover a leading block of C whose sides are multiples of a tile
cluster (896 or 1024 along M, 1024 along N) with K a multiple of 128; ``hk_gemm_tn`` runs that block
with the variant that fills the GPU best and finishes a ragged M or N edge with ``torch.mm``. It
refuses (``hk_gemm_tn_supported`` is False) when K is not a multiple of 128, the block would cover
less than three quarters of C, or it has fewer tiles than the GPU has CUs (one tile per workgroup
and no split-K, so such a GEMM would leave CUs idle throughout), or a grid of fewer than four waves of
workgroups whose last wave is less than 85% full.
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
def _variants() -> dict:
    lib = load_library()
    out, v = {}, 0
    while True:
        vals = [ctypes.c_int() for _ in range(4)]
        if lib.hk_gemm_tn_variant(v, *[ctypes.byref(x) for x in vals]) != 0:
            return out
        out[v] = tuple(x.value for x in vals)  # (BM, BN, CM, CN)
        v += 1


@functools.lru_cache(maxsize=None)
def _num_cus() -> int:
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


@functools.lru_cache(maxsize=4096)
def hk_gemm_tn_plan(M: int, N: int, K: int) -> Optional[Tuple[int, int, int]]:
    """``(variant, M_block, N_block)`` for ``hk_gemm_tn``, or None if it does not take this shape.

    Among the variants, the one with the most useful work per CU slot: covered area times the
    occupancy of the last wave of workgroups (one workgroup per CU).
    """
    if K <= 0 or K % 128 or M <= 0 or N <= 0:
        return None
    best = None
    for v, (bm, bn, cm, cn) in _variants().items():
        mb, nb = M - M % (bm * cm), N - N % (bn * cn)
        if mb == 0 or nb == 0:
            continue
        tiles = (mb // bm) * (nb // bn)
        if tiles < _num_cus():
            # one tile per workgroup and no split-K: fewer tiles than CUs leaves part of the GPU idle
            # for the whole (often long-K) GEMM
            continue
        waves = math.ceil(tiles / _num_cus())
        fill = tiles / (waves * _num_cus())
        if waves < 4 and fill < 0.85:
            # a short grid whose last wave is mostly empty: a library kernel with split-K or other
            # tile shapes does better there
            continue
        score = (mb * nb) * fill
        if best is None or score > best[0] * 1.0001:
            best = (score, v, mb, nb)
    if best is None or best[2] * best[3] < _MIN_COVER * M * N:
        return None
    return best[1], best[2], best[3]


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
    v, mb, nb = hk_gemm_tn_plan(M, N, K)
    bm, bn, _, _ = _variants()[v]
    stream = torch.cuda.current_stream(a.device).cuda_stream
    rc = load_library().hk_gemm_tn(v, a.data_ptr(), b.data_ptr(), out.data_ptr(), M, N, K, mb // bm, nb // bn, stream)
    if rc != 0:
        raise RuntimeError(f"hk_gemm_tn launch failed ({rc}) for M={M} N={N} K={K} variant {v}")
    if mb < M:  # columns [mb, M) of the row-major result
        out[:, mb:] = torch.mm(b, a[mb:].t())
    if nb < N:  # rows [nb, N)
        torch.mm(b[nb:], a.t(), out=out[nb:])
    return out
