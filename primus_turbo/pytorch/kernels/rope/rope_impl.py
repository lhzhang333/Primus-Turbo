###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Python-side wrappers that launch the fused QKV RoPE kernels: Triton or FlyDSL.

``PRIMUS_TURBO_ROPE_BACKEND`` picks the backend (TRITON or FLYDSL); unset, gfx1250 runs the Triton
kernels and other architectures the FlyDSL ones. Both take the same inputs and round each output once
from fp32; their cos / sin can differ in the last fp32 bit, so an output can differ in the last bit.
The Triton backward reads dq / dk / dv through their strides (a unit last-dim stride is all it needs);
the FlyDSL one takes contiguous copies.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import triton

from primus_turbo.common import envs
from primus_turbo.flydsl.rope.rope_kernel import (
    ROPE_HEAD_DIM,
    ROPE_ROW_GROUP,
    flydsl_qkv_rope_backward,
    flydsl_qkv_rope_forward,
)
from primus_turbo.pytorch.core.utils import is_gfx1250
from primus_turbo.triton.rope.qkv_rope_kernel import (
    qkv_rope_bwd_kernel,
    qkv_rope_fwd_kernel,
)

_BACKENDS = ("TRITON", "FLYDSL")
# (s, b) rows of one KV group per Triton program, and its warps: the best of a gfx1250 sweep at the
# Llama-3.1-8B shape (S=8192, B=4, 8 groups of 4 q heads, D=128), forward and backward alike.
_TRITON_ROWS = 4
_TRITON_NUM_WARPS = 2


def rope_backend() -> str:
    """The backend :func:`rope_fwd_impl` / :func:`rope_bwd_impl` run: ``PRIMUS_TURBO_ROPE_BACKEND``,
    else TRITON on gfx1250 and FLYDSL elsewhere."""
    value = envs.PRIMUS_TURBO_ROPE_BACKEND
    if value is None:
        return "TRITON" if is_gfx1250() else "FLYDSL"
    backend = value.strip().upper()
    if backend not in _BACKENDS:
        raise ValueError(f"PRIMUS_TURBO_ROPE_BACKEND must be one of {_BACKENDS}, got {value!r}")
    return backend


def _freq_table(freqs: torch.Tensor, S: int) -> torch.Tensor:
    table = freqs.reshape(-1, freqs.shape[-1])
    if table.shape[0] < S:
        raise ValueError(f"fused_qkv_rope: the frequency table has {table.shape[0]} positions, S is {S}")
    return table


def _triton_rope_forward(qkv, q_freqs, k_freqs, qkv_split_arg_list):
    S, B, NG, _ = qkv.shape
    q_size, D, _ = qkv_split_arg_list
    npg = q_size // D
    q = torch.empty(S, B, NG * npg, D, device=qkv.device, dtype=qkv.dtype)
    k = torch.empty(S, B, NG, D, device=qkv.device, dtype=qkv.dtype)
    v = torch.empty_like(k)
    n_rows = S * B
    if n_rows == 0 or NG == 0:
        return q, k, v
    qf = _freq_table(q_freqs, S)
    kf = _freq_table(k_freqs, S)
    qkv_rope_fwd_kernel[(triton.cdiv(n_rows, _TRITON_ROWS), NG)](
        qkv,
        qf,
        kf,
        q,
        k,
        v,
        n_rows,
        B,
        qf.stride(0),
        kf.stride(0),
        NG=NG,
        NPG=npg,
        NPG_P2=triton.next_power_of_2(npg),
        D=D,
        ROWS=_TRITON_ROWS,
        num_warps=_TRITON_NUM_WARPS,
    )
    return q, k, v


def _triton_rope_backward(dq, dk, dv, q_freqs, k_freqs, qkv_split_arg_list):
    S, B, _, D = dq.shape
    NG = dk.shape[2]
    npg = qkv_split_arg_list[0] // D
    dq, dk, dv = (t if t.stride(-1) == 1 else t.contiguous() for t in (dq, dk, dv))
    dqkv = torch.empty(S, B, NG, (npg + 2) * D, device=dq.device, dtype=dq.dtype)
    n_rows = S * B
    if n_rows == 0 or NG == 0:
        return dqkv
    qf = _freq_table(q_freqs, S)
    kf = _freq_table(k_freqs, S)
    qkv_rope_bwd_kernel[(triton.cdiv(n_rows, _TRITON_ROWS), NG)](
        dq,
        dk,
        dv,
        qf,
        kf,
        dqkv,
        n_rows,
        B,
        qf.stride(0),
        kf.stride(0),
        dq.stride(0),
        dq.stride(1),
        dq.stride(2),
        dk.stride(0),
        dk.stride(1),
        dk.stride(2),
        dv.stride(0),
        dv.stride(1),
        dv.stride(2),
        NG=NG,
        NPG=npg,
        NPG_P2=triton.next_power_of_2(npg),
        D=D,
        ROWS=_TRITON_ROWS,
        num_warps=_TRITON_NUM_WARPS,
    )
    return dqkv


def rope_shape_error(qkv, q_freqs, k_freqs, qkv_split_arg_list) -> str | None:
    """The reason this shape is unsupported, or None when the kernels accept it."""
    if qkv.ndim != 4:
        return f"qkv must be 4-D [S, B, H, q+k+v], got {tuple(qkv.shape)}"
    if qkv.dtype != torch.bfloat16:
        return f"qkv must be bfloat16, got {qkv.dtype}"
    if not qkv.is_contiguous():
        return "qkv must be contiguous"
    if len(qkv_split_arg_list) != 3:
        return f"qkv_split_arg_list must be [q, k, v], got {list(qkv_split_arg_list)}"
    q_size, k_size, v_size = qkv_split_arg_list
    if k_size != ROPE_HEAD_DIM or v_size != ROPE_HEAD_DIM or q_size % ROPE_HEAD_DIM:
        return f"k and v must be {ROPE_HEAD_DIM} and q a multiple of it, got {list(qkv_split_arg_list)}"
    if qkv.shape[-1] != q_size + k_size + v_size:
        return f"qkv last dim {qkv.shape[-1]} != q+k+v {q_size + k_size + v_size}"
    if q_freqs.shape[-1] != ROPE_HEAD_DIM or k_freqs.shape[-1] != ROPE_HEAD_DIM:
        return f"freqs last dim must be {ROPE_HEAD_DIM}"
    # No per-row predicate, and the descriptors span all of memory: a partial last
    # row group would read and write out of bounds rather than be clamped.
    rows = qkv.shape[0] * qkv.shape[1]
    if rows % ROPE_ROW_GROUP:
        return f"S*B must be a multiple of {ROPE_ROW_GROUP}, got {rows}"
    return None


def rope_fwd_impl(
    qkv: torch.Tensor,
    q_freqs: torch.Tensor,
    k_freqs: torch.Tensor,
    qkv_split_arg_list: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rotate the q and k halves of a packed QKV tensor; v is copied through."""
    why = rope_shape_error(qkv, q_freqs, k_freqs, qkv_split_arg_list)
    if why is not None:
        raise ValueError(f"fused_qkv_rope: unsupported input ({why})")
    if rope_backend() == "TRITON":
        return _triton_rope_forward(qkv, q_freqs, k_freqs, qkv_split_arg_list)
    return flydsl_qkv_rope_forward(qkv, q_freqs, k_freqs, qkv_split_arg_list)


def rope_bwd_impl(
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    q_freqs: torch.Tensor,
    k_freqs: torch.Tensor,
    qkv_split_arg_list: Sequence[int],
) -> torch.Tensor:
    """Packed dQKV for :func:`rope_fwd_impl`: the forward rotation transposed."""
    if rope_backend() == "TRITON":
        return _triton_rope_backward(dq, dk, dv, q_freqs, k_freqs, qkv_split_arg_list)
    return flydsl_qkv_rope_backward(
        dq.contiguous(), dk.contiguous(), dv.contiguous(), q_freqs, k_freqs, qkv_split_arg_list
    )
