###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

"""Host entry of the gfx1250 head-dim-128 attention forward on generated tile schedules (GQA)."""

import contextlib
from typing import Optional, Tuple

import flydsl.compiler as flyc
import torch

from primus_turbo.flydsl.attention.gfx1250_mla import NUM_XCD
from primus_turbo.flydsl.attention.gfx1250_mla.common import num_cus

from .fwd_kernel import BM, D_QK, D_V, launch_d128_fwd

_COMPILED = {}  # device index -> flyc.compile's CompiledFunction (every shape-dependent argument is a runtime scalar)
# Persistent workgroups (one per CU, each serving the head-batches of its XCD) where the grid allows them; False: one
# workgroup per q block.
PERSIST = True


def launch_grid(q: torch.Tensor, k: torch.Tensor) -> Tuple[int, int]:
    """(npw, nqb) of the kernel for q [B, Sq, H, 128], k [B, Skv, Hkv, 128]: nqb q blocks of 256 rows per (batch, q head),
    and npw > 0 workgroups (one per CU) on the static persistent schedule, or 0 for one workgroup per q block. The
    schedule covers seqlen_q == seqlen_kv in whole q blocks when every XCD group of CUs takes a whole number of
    q-block rows and an even number of steps (two steps pair q block i with nqb - 1 - i: equal work per workgroup)."""
    b, sq, h, _ = q.shape
    nqb = -(-sq // BM)
    # (whole q blocks of seqlen_q == seqlen_kv also give every block of a persistent workgroup BM / NB = 4 kv tiles
    # or more; the kernel's prologue needs 3: the assertion in fwd_kernel)
    if not PERSIST or k.shape[1] != sq or sq % BM:
        return 0, nqb
    cus = num_cus(q.device)
    per_group, hb = cus // NUM_XCD, b * h
    if cus % NUM_XCD or per_group % nqb or hb % NUM_XCD:
        return 0, nqb
    steps, rem = divmod(hb // NUM_XCD, per_group // nqb)
    return (cus if rem == 0 and steps % 2 == 0 else 0), nqb


def shape_unsupported_reason(
    batch: int, seqlen_q: int, seqlen_kv: int, heads_q: int, heads_kv: int
) -> Optional[str]:
    """Why this forward cannot run these sizes (causal), or None."""
    if batch <= 0 or heads_q <= 0 or heads_kv <= 0 or heads_q % heads_kv:
        return f"batch and heads must be positive, heads_q a multiple of heads_kv, got {batch} / {heads_q} / {heads_kv}"
    g = heads_q // heads_kv
    if g & (g - 1):  # the kernel takes kv head = q head >> log2(G)
        return f"heads_q / heads_kv must be a power of two, got {heads_q} / {heads_kv}"
    if (
        batch >= 1 << 16 or heads_q >= 1 << 16
    ):  # the grid (nqb, heads, batch): its last two dimensions wrap at 65536
        return f"batch and heads_q must be under 65536 each (16-bit dimensions of the launch grid), got {batch} / {heads_q}"
    if seqlen_q <= 0 or seqlen_q % 64 or seqlen_kv % 64 or seqlen_q > seqlen_kv:
        return (
            "seqlen_q and seqlen_kv must be positive multiples of 64 with seqlen_q <= seqlen_kv, got "
            f"{seqlen_q} / {seqlen_kv}"
        )
    if batch * heads_q * seqlen_q * D_V * 2 >= 2**31:  # (lse, 4 bytes per row, is then far below its 2 GiB)
        return f"out must be under 2 GiB (the kernel's 32-bit byte counts), got {batch} * {seqlen_q} * {heads_q} rows"
    return None


def flash_attn_fwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Causal attention forward at head dim 128 with GQA on gfx1250.

    q [B, Sq, Hq, 128], k / v [B, Skv, Hkv, 128] (Hq / Hkv a power of two): contiguous bf16 on one gfx1250
    device, Sq and Skv multiples of 64 (the kernel's kv tiles and its mask's diagonal tiles), Sq <= Skv,
    out under 2 GiB, B and Hq under 65536. The mask is bottom-right: query i attends keys j <= i + Skv - Sq. Returns
    o [B, Sq, Hq, 128] bf16 and lse [B, Hq, Sq] fp32 (natural log).
    """
    for name, t, dim in (("q", q, D_QK), ("k", k, D_QK), ("v", v, D_V)):
        if t.dtype != torch.bfloat16 or t.dim() != 4 or t.shape[-1] != dim or not t.is_contiguous():
            raise ValueError(f"{name} must be a contiguous [B, S, H, {dim}] bfloat16 tensor")
    if k.device != q.device or v.device != q.device:
        raise ValueError(f"q/k/v must be on one device, got {q.device}/{k.device}/{v.device}")
    b, sq, h, _ = q.shape
    skv, hkv = k.shape[1], k.shape[2]
    if tuple(k.shape[:2]) != (b, skv) or tuple(v.shape[:3]) != (b, skv, hkv):
        raise ValueError(
            f"q {tuple(q.shape)}, k {tuple(k.shape)}, v {tuple(v.shape)}: batch and kv heads must match"
        )
    reason = shape_unsupported_reason(b, sq, skv, h, hkv)
    if reason is not None:
        raise ValueError(reason)

    out = torch.empty((b, sq, h, D_V), dtype=q.dtype, device=q.device)
    lse = torch.empty((b, h, sq), dtype=torch.float32, device=q.device)
    # q's device as the current one: its stream and compiled kernel
    on_q_device = q.device.type != "cuda" or q.device.index == torch.cuda.current_device()
    with contextlib.nullcontext() if on_q_device else torch.cuda.device(q.device):
        npw, nqb = launch_grid(q, k)
        gsh = (h // hkv).bit_length() - 1
        args = (
            out,
            q,
            k,
            v,
            lse,
            float(softmax_scale),
            sq,
            skv,
            h,
            hkv,
            gsh,
            b,
            nqb,
            npw,
            torch.cuda.current_stream(),
        )
        fn = _COMPILED.get(q.device.index)
        if fn is not None:
            fn(*args)
        else:
            # flyc.compile launches the kernel itself; it returns None under COMPILE_ONLY.
            fn = flyc.compile(launch_d128_fwd, *args)
            if fn is not None:
                _COMPILED[q.device.index] = fn
    return out, lse
