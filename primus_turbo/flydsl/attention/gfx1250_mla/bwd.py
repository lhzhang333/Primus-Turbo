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

"""Host side of the gfx1250 MLA flash-attention backward: shape rules, launch plan, entry point.

k_delta_bshd (bwd_delta: delta and the row constants NL / ND), then per head chunk k_dkdv96ds (bwd_dkdv96: dK and dV
over 96-row kv blocks, and every bf16 dS^T tile it feeds its dK WMMA also into a scratch laid out as dslayout v2) and
k_dqds (bwd_dqds: dQ = dS K from that scratch, so S and dP are computed once).

The whole dS^T scratch is about batch * heads * seqlen_q * seqlen_kv bytes under the causal mask (4.33 GB at b2 s4096
h128). The backward runs the heads in chunks instead, alternating between the current stream and a side stream with a
scratch each: one chunk's kernels fill the CUs the other chunk's tail leaves idle, and two chunks' scratch is held at
once (ds_chunkings). It takes the most chunks whose k_dqds still has two workgroups per CU, within
PRIMUS_TURBO_MLA_BWD_DS_SCRATCH_MB; a problem too small for two such chunks runs whole, and a budget below two heads'
scratch runs one head at a time on one stream. The scratch is the last buffer it allocates, from the caching allocator
on the current stream; one that cannot be allocated sends the call to smaller chunks. Every head's arithmetic is the
same however the heads are chunked, so the gradients are too.
"""

import math
from typing import List, Optional, Tuple

import flydsl.compiler as flyc
import torch

import primus_turbo.common.envs as envs
from primus_turbo.common.logger import logger

from . import bwd_delta as _delta
from . import bwd_dkdv96 as _d
from . import bwd_dqds as _q

_NUM_CU = 256  # MI455X
# device -> the smallest dS scratch whose allocation failed there: a failed allocation costs the caching allocator a
# synchronize and its whole cache, so a scratch at least as large is not tried there again (except the smallest).
_DS_OOM_BYTES = {}
# device -> the side stream of a chunked backward
_SIDE_STREAMS = {}
# The smallest head chunk the backward launches by choice: its k_dqds keeps two workgroups per CU busy (smaller chunks
# leave CUs idle in each kernel's tail faster than the other stream's chunk can fill them).
_CHUNK_MIN_DQ_WORKGROUPS = 2 * _NUM_CU

# flyc.compile's CompiledFunction per (launcher, tensor dtypes/ranks, device). Launching through it
# skips FlyDSL's per-call key derivation; all shape-dependent kernel arguments are runtime scalars.
_COMPILED = {}


def shape_unsupported_reason(batch: int, seqlen_q: int, seqlen_kv: int, num_heads: int) -> Optional[str]:
    """Why the backward cannot run these MHA sizes, or None."""
    if batch <= 0 or num_heads <= 0:
        return f"batch and num_heads must be positive, got {batch} / {num_heads}"
    if seqlen_q <= 0 or seqlen_q % 64:  # k_dkdv96ds walks query pairs of 32 rows two at a time
        return f"seqlen_q must be a positive multiple of 64, got {seqlen_q}"
    if seqlen_kv <= 0 or seqlen_kv % 64:
        return f"seqlen_kv must be a positive multiple of 64, got {seqlen_kv}"
    if seqlen_q > seqlen_kv:
        # bottom-right causal leaves the first seqlen_q - seqlen_kv queries without keys
        return f"causal needs seqlen_q <= seqlen_kv, got {seqlen_q} > {seqlen_kv}"
    # The sizes the attention is verified for: q / dq up to 1 GiB, lse up to 256 MiB, k under 2 GiB (the kernels'
    # own 32-bit byte offsets would allow q / dq and k up to 2 GiB).
    rows_q, rows_kv = batch * seqlen_q * num_heads, batch * seqlen_kv * num_heads
    if rows_q * _d.D_QK * 2 > (1 << 30) or rows_q * 4 > (1 << 28) or rows_kv * _d.D_QK * 2 >= (1 << 31):
        return "q/k/lse too large for the backward's verified sizes"
    reason = _d.unsupported_reason(batch, seqlen_q, seqlen_kv, num_heads, num_heads)
    return reason or _q.dqds_unsupported_reason(batch, seqlen_q, seqlen_kv, num_heads, num_heads)


def _head_group(nh, target):
    """Heads per grid row: nh up to ``target``, else the largest multiple of 8 (one XCD per head under
    round-robin dispatch) that divides nh and is at most ``target``, else nh."""
    if nh <= target:
        return nh
    return next((d for d in range(target, 7, -8) if nh % d == 0), nh)


def ds_scratch_bytes(batch: int, seqlen_q: int, seqlen_kv: int, num_heads: int, causal: bool = True) -> int:
    """Bytes of the whole dS^T scratch of these MHA sizes (the backward holds at most two head chunks of it)."""
    return batch * num_heads * _d.ds_blocks_per_head(seqlen_q, seqlen_kv, causal) * _d.DSL_BLK_B


def ds_chunkings(
    batch: int, seqlen_q: int, seqlen_kv: int, num_heads: int, budget: int
) -> List[Tuple[int, int, int]]:
    """[(heads per chunk, streams, scratch bytes)] the backward tries in order: all heads in one scratch; chunks of
    ceil(heads / 2^i) heads alternating between two streams with a scratch each; last, one head at a time on one
    stream. It starts within ``budget`` bytes at the most chunks whose k_dqds still has _CHUNK_MIN_DQ_WORKGROUPS (else
    at the fewest chunks) and goes on to more, smaller chunks, for a scratch that cannot be allocated. When nothing
    fits the budget, only the last."""
    per_head = ds_scratch_bytes(batch, seqlen_q, seqlen_kv, 1)
    dq_workgroups = batch * sum(_q.dq_split(seqlen_q)[1:])  # k_dqds workgroups of one head
    opts, hc = [], num_heads
    while True:
        streams = 1 if hc == num_heads else 2
        opts.append((hc, streams, per_head * hc * streams))
        if hc == 1:
            break
        hc = -(-hc // 2)
    if num_heads > 1:
        opts.append((1, 1, per_head))
    fit = [o for o in opts if o[2] <= budget]  # the scratch shrinks along opts: a suffix
    if not fit:
        return opts[-1:]
    busy = [i for i, (hc, _, _) in enumerate(fit) if hc * dq_workgroups >= _CHUNK_MIN_DQ_WORKGROUPS]
    return fit[busy[-1] if busy else 0 :]


def _side_stream(device):
    stream = _SIDE_STREAMS.get(device)
    if stream is None:
        stream = _SIDE_STREAMS[device] = torch.cuda.Stream(device=device)
    return stream


def _launches(dout, q, k, v, out, lse, softmax_scale, causal, stream):
    """(launcher, args) of every launch in issue order, and the (dq, dk, dv) they write: k_delta_bshd, then
    k_dkdv96ds and k_dqds per head chunk. torch.OutOfMemoryError when even one head's scratch cannot be allocated."""
    b, sq, h, _ = q.shape
    skv = k.shape[1]
    c = int(bool(causal))
    budget = envs.PRIMUS_TURBO_MLA_BWD_DS_SCRATCH_MB << 20
    tries = ds_chunkings(b, sq, skv, h, budget)
    if tries[0][2] > budget:
        logger.warning(
            f"MLA backward: one head's dS scratch, {tries[0][2]} bytes, is over the {budget}-byte budget "
            "(PRIMUS_TURBO_MLA_BWD_DS_SCRATCH_MB); the heads run one at a time",
            once=True,
        )
    # The scratch is the last allocation: when it fits, the backward fits.
    delta = torch.empty((b, h, sq), device=q.device, dtype=torch.float32)
    nl = torch.empty_like(delta)  # -LOG2E * lse
    nd = torch.empty_like(delta)  # -scale * delta
    dq, dk, dv = q.new_empty(q.shape), k.new_empty(k.shape), v.new_empty(v.shape)
    for i, (hc, streams, nbytes) in enumerate(tries):
        last = i == len(tries) - 1
        if not last and nbytes >= _DS_OOM_BYTES.get(q.device, math.inf):
            continue
        try:
            ds = torch.empty((nbytes // 2,), device=q.device, dtype=torch.bfloat16)
            break
        except torch.OutOfMemoryError:
            if last:
                raise
            _DS_OOM_BYTES[q.device] = nbytes
            logger.warning(
                f"MLA backward: no memory for a {nbytes}-byte dS scratch on {q.device}; smaller head chunks run (a "
                "scratch at least as large is not tried again)",
                once=True,
            )
    scale = float(softmax_scale)
    q_off, nhead, ntile = _q.dq_split(sq)
    nblk = _d.ds_blocks_per_head(sq, skv, c)
    scratches = ds.view(streams, -1)
    side = (
        stream if streams == 1 or q.device.type != "cuda" else _side_stream(q.device)
    )  # meta tensors: one stream
    delta_args = (
        dout,
        out,
        delta,
        lse,
        nl,
        nd,
        scale,
        sq,
        h,
        b * sq * h,
        sq // _delta.ROWS_DELTA,
        b * h,
        stream,
    )
    launches = [(_delta.launch_delta, delta_args)]
    for i, h0 in enumerate(range(0, h, hc)):
        nh, st, ds_c = min(hc, h - h0), (stream, side)[i % 2], scratches[i % streams]
        hg_kv, hg_q = _head_group(nh, _d.HEAD_GROUP), _head_group(nh, _q.HEAD_GROUP)
        # Hq, Hkv, G = h, h, 1: MHA; the heads [h0, h0 + nh). The kernels take only the scratch's base address:
        # k_dkdv96ds gets a [blocks, 512] view (FlyDSL packs tensor sizes as int32), k_dqds the address itself.
        dkdv = (
            q,
            k,
            v,
            dout,
            nl,
            nd,
            dv,
            dk,
            ds_c.view(-1, _d.DSL_BLK_B // 2),
            scale,
            sq,
            skv,
            h,
            h,
            1,
            sq // 16,
        )
        dkdv += (
            skv - sq,
            c,
            b,
            hg_kv,
            1,
            h0,
            nh,
            _d.nblk96(skv),
            b * nh // hg_kv,
            st,
        )  # ds_mode 1: dS^T stores on
        dqa = (
            ds_c.data_ptr(),
            k,
            dq,
            sq,
            skv,
            h,
            h,
            1,
            skv // _q.KV32,
            skv - sq,
            c,
            q_off,
            ntile,
            hg_q,
            nblk,
        )
        dqa += (b * sq * h * _d.D_QK * 2, h0, nh, ntile + nhead, b * nh // hg_q, st)
        launches += [(_d.launch_dkdv96ds, dkdv), (_q.launch_dqds, dqa)]
    return launches, (dq, dk, dv)


def _launch(launcher, args):
    tensors = [a for a in args if isinstance(a, torch.Tensor)]
    key = (launcher, tuple((a.dtype, a.dim()) for a in tensors), tensors[0].device.index)
    fn = _COMPILED.get(key)
    if fn is not None:
        fn(*args)
        return
    fn = flyc.compile(launcher, *args)  # compiles and issues this launch
    if fn is not None:  # None under COMPILE_ONLY
        _COMPILED[key] = fn


def flash_attn_mla_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float,
    causal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gradients (dq, dk, dv) of MLA flash attention on gfx1250, laid out like q, k, v.

    q [B, Sq, H, 192], k [B, Skv, H, 192], v [B, Skv, H, 128], out/dout [B, Sq, H, 128], contiguous
    bf16 on one device; lse [B, H, Sq] fp32 natural log, as the forward returns it. Causal only,
    bottom-right.
    """
    if not causal:
        raise ValueError("the gfx1250 MLA attention backward supports causal attention only")
    named = (
        ("dout", dout, _d.D_V),
        ("q", q, _d.D_QK),
        ("k", k, _d.D_QK),
        ("v", v, _d.D_V),
        ("out", out, _d.D_V),
    )
    for name, t, dim in named:
        if t.dtype != torch.bfloat16 or t.dim() != 4 or t.shape[-1] != dim or not t.is_contiguous():
            raise ValueError(f"{name} must be a contiguous [B, S, H, {dim}] bfloat16 tensor")
    if any(t.device != q.device for t in (dout, k, v, out, lse)):
        raise ValueError("dout/q/k/v/out/lse must be on one device")
    b, sq, h, _ = q.shape
    skv = k.shape[1]
    if k.shape[:3] != (b, skv, h) or v.shape[:3] != (b, skv, h):
        raise ValueError(
            f"q {tuple(q.shape)}, k {tuple(k.shape)}, v {tuple(v.shape)}: batch and heads must match (MHA)"
        )
    if out.shape[:3] != (b, sq, h) or dout.shape != out.shape or lse.shape != (b, h, sq):
        raise ValueError(
            f"out/dout/lse shapes {tuple(out.shape)}/{tuple(dout.shape)}/{tuple(lse.shape)} "
            f"do not match q {tuple(q.shape)}"
        )
    reason = shape_unsupported_reason(b, sq, skv, h)
    if reason is not None:
        raise ValueError(reason)
    lse = lse.contiguous().float()
    stream = torch.cuda.current_stream()
    launches, grads = _launches(dout, q, k, v, out, lse, softmax_scale, causal, stream)
    _launch(*launches[0])  # k_delta_bshd: the row constants every later launch reads
    side = next((args[-1] for _, args in launches[1:] if args[-1] is not stream), None)
    if (
        side is not None
    ):  # a chunked dS plan: the side stream starts after k_delta_bshd, the caller's after both
        side.wait_stream(stream)
    for launcher, args in launches[1:]:
        _launch(launcher, args)
    if side is not None:
        stream.wait_stream(side)
    return grads
