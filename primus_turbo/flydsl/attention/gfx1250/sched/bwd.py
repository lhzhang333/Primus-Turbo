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

"""Host side of the gfx1250 head-dim-128 flash-attention backward on generated tile schedules: shape rules, launch
plan, entry point. The plan of the MLA backward (gfx1250_mla.bwd) with GQA:

k_delta_bshd (the MLA package's: the row constants NL / ND), then per chunk of kv heads k_dkdv (bwd_dkdv_kernel: dK
and dV over 192-row kv blocks, the G = Hq / Hkv q heads of a kv head one after the other on the same accumulators, and
every bf16 dS^T tile also into a scratch) and k_dqds (bwd_dq_kernel: dQ = dS K from that scratch), so S and dP are
computed once and nothing is added with atomics: every output element is written once, the gradients are bitwise
reproducible.

The whole dS^T scratch is about batch * heads_q * seqlen_q * seqlen_kv bytes under the causal mask (8.2 GiB at b4
s8192 hq32). The backward runs all heads on one scratch when that is within PRIMUS_TURBO_MLA_BWD_DS_SCRATCH_MB;
over the budget it runs the kv heads in chunks, alternating between the current stream and a side stream with a
scratch each (ds_chunkings).
"""

import contextlib
import inspect
import math
from typing import List, Optional, Tuple

import flydsl.compiler as flyc
import torch

import primus_turbo.common.envs as envs
from primus_turbo.common.logger import logger
from primus_turbo.flydsl.attention.gfx1250_mla import bwd_delta_kernel as _delta
from primus_turbo.flydsl.attention.gfx1250_mla.bwd import _head_group
from primus_turbo.flydsl.attention.gfx1250_mla.common import num_cus

from . import bwd_dkdv_items as _items
from . import bwd_dkdv_kernel as _dkdv
from . import bwd_dq_kernel as _dq

D = _dkdv.D_QK  # head dim of q, k and v
_DS_OOM_BYTES = {}  # device -> the smallest dS scratch whose allocation failed there
_SIDE_STREAMS = {}  # (device index, caller stream) -> the side stream of its chunked backwards
_PARAMS = {}  # launcher -> its parameter names
_COMPILED = {}  # (launcher, tensor dtypes / ranks, device) -> flyc.compile's CompiledFunction


def shape_unsupported_reason(
    batch: int, seqlen_q: int, seqlen_kv: int, heads_q: int, heads_kv: int
) -> Optional[str]:
    """Why this backward cannot run these sizes, or None."""
    if batch <= 0 or heads_q <= 0 or heads_kv <= 0 or heads_q % heads_kv:
        return f"batch and heads must be positive, heads_q a multiple of heads_kv, got {batch} / {heads_q} / {heads_kv}"
    if batch * heads_q >= 1 << 16:  # a dimension of k_delta_bshd's launch grid
        return f"batch * heads_q must be under 65536, got {batch} * {heads_q}"
    if seqlen_q <= 0 or seqlen_q % 64 or seqlen_kv <= 0 or seqlen_kv % 64:
        return f"seqlen_q and seqlen_kv must be positive multiples of 64, got {seqlen_q} / {seqlen_kv}"
    if seqlen_q > seqlen_kv:
        return f"causal needs seqlen_q <= seqlen_kv, got {seqlen_q} > {seqlen_kv}"
    # 32-bit byte offsets of q / dq (k_dqds, k_dkdv's Q and dO tiles) and of k / dk
    if batch * max(heads_q * seqlen_q, heads_kv * seqlen_kv) * D * 2 >= 1 << 31:
        return (
            "q and k must be under 2 GiB each (the kernels' 32-bit byte offsets), got batch * seqlen * heads = "
            f"{batch} * {seqlen_q} * {heads_q} / {batch} * {seqlen_kv} * {heads_kv} rows of {D * 2} bytes"
        )
    # k_dkdv leaves the dK / dV stores of the last kv block's rows past seqlen_kv to the buffer range check: their
    # 32-bit offsets must not wrap back into range
    padded_kv = _items.nblk_kv(seqlen_kv) * _items.ROWS_B
    if batch * heads_kv * padded_kv * D * 2 >= 1 << 32:
        return f"k padded to k_dkdv's {_items.ROWS_B}-row kv blocks must be under 4 GiB, got {batch} * {padded_kv} * {heads_kv} rows"
    return None


def kv_blocks(seqlen_kv: int) -> int:
    """k_dkdv's kv blocks of one (batch, kv head): its items."""
    return _items.nblk_kv(seqlen_kv)


def ds_scratch_bytes(batch: int, seqlen_q: int, seqlen_kv: int, heads_q: int) -> int:
    """Bytes of the whole dS^T scratch (chunked, the backward holds two chunks of it)."""
    return batch * heads_q * _items.ds_blocks_per_head(seqlen_q, seqlen_kv) * _items.DSL_BLK_B


def ds_chunkings(
    batch: int, seqlen_q: int, seqlen_kv: int, heads_q: int, heads_kv: int, budget: int
) -> List[Tuple[int, int, int]]:
    """[(kv heads per chunk, streams, scratch bytes)] the backward tries in order (gfx1250_mla.bwd.ds_chunkings with
    kv heads, each with its heads_q / heads_kv q heads, as the unit)."""
    per_head = ds_scratch_bytes(batch, seqlen_q, seqlen_kv, heads_q // heads_kv)
    opts, hc = [(heads_kv, 1, per_head * heads_kv)], heads_kv
    while hc > 1:
        hc = -(-hc // 2)
        opts.append((hc, 2, per_head * hc * 2))
    opts.append((1, 1, per_head))
    tries = []
    for opt in opts:
        if opt[2] <= budget and (not tries or opt[2] < tries[-1][2]):
            tries.append(opt)
    return tries or opts[-1:]


def _side_stream(device, stream):
    key = (device.index, stream.cuda_stream)
    side = _SIDE_STREAMS.get(key)
    if side is None:
        side = _SIDE_STREAMS[key] = torch.cuda.Stream(device=device)
    return side


def _positional(launcher, **kwargs):
    """``launcher``'s arguments in its parameter order (FlyDSL's compiled call takes them positionally)."""
    names = _PARAMS.get(launcher)
    if names is None:
        names = _PARAMS[launcher] = tuple(inspect.signature(launcher.func).parameters)
    if len(kwargs) != len(names):
        raise TypeError(f"{launcher.func.__name__} takes {names}, got {tuple(kwargs)}")
    return tuple(kwargs[n] for n in names)


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


def _delta_launch(dout, q, k, v, out, lse, softmax_scale, stream):
    """The k_delta_bshd launch and the buffers of the whole backward but the scratch: (NL, ND, dq, dk, dv)."""
    b, sq, hq, _ = q.shape
    nl = torch.empty((b, hq, sq), device=q.device, dtype=torch.float32)  # -LOG2E * lse
    nd = torch.empty_like(nl)  # -scale * rowsum(dO * O)
    bufs = (nl, nd, q.new_empty(q.shape), k.new_empty(k.shape), v.new_empty(v.shape))
    launch = _positional(
        _delta.launch_delta,
        DO=dout,
        O=out,
        LSE=lse,
        NL=nl,
        ND=nd,
        scale=float(softmax_scale),
        S=sq,
        H=hq,
        n_rows=b * sq * hq,
        nsblk=-(-sq // _delta.ROWS_DELTA),
        nbh=b * hq,
        stream=stream,
    )
    return (_delta.launch_delta, launch), bufs


def _chunk_launches(dout, q, k, v, softmax_scale, stream, bufs):
    """The dS scratch (the backward's last allocation) and the k_dkdv / k_dqds launches of each chunk of kv heads, in
    issue order. torch.OutOfMemoryError when even one kv head's scratch cannot be allocated."""
    nl, nd, dq, dk, dv = bufs
    b, sq, hq, _ = q.shape
    skv, hkv = k.shape[1], k.shape[2]
    g = hq // hkv
    ncu = num_cus(q.device)
    budget = envs.PRIMUS_TURBO_MLA_BWD_DS_SCRATCH_MB << 20
    tries = ds_chunkings(b, sq, skv, hq, hkv, budget)
    for i, (hc, streams, nbytes) in enumerate(tries):
        last = i == len(tries) - 1
        if not last and nbytes >= _DS_OOM_BYTES.get(q.device, math.inf):
            continue
        tables = [
            _items.item_table(b, sq, skv, hq, hkv, h0, min(hc, hkv - h0), ncu, q.device)
            for h0 in range(0, hkv, hc)
        ]
        try:
            ds = torch.empty((nbytes // 2,), device=q.device, dtype=torch.bfloat16)
            break
        except torch.OutOfMemoryError:
            if last:
                raise
            _DS_OOM_BYTES[q.device] = nbytes
            logger.warning(
                f"attention backward: no memory for a {nbytes}-byte dS scratch on {q.device}; smaller head chunks run",
                once=True,
            )
    scale = float(softmax_scale)
    q_off, nhead, ntile = _dq.dq_split(sq)
    ds_blocks = _items.ds_blocks_per_head(sq, skv)
    scratches = ds.view(streams, -1)
    side = stream if streams == 1 or q.device.type != "cuda" else _side_stream(q.device, stream)
    launches = []
    for i, h0 in enumerate(range(0, hkv, hc)):
        nh, st, ds_c = min(hc, hkv - h0), (stream, side)[i % 2], scratches[i % streams]
        nhq = nh * g  # the chunk's q heads [h0 * g, h0 * g + nhq)
        hg_q = _head_group(nhq, _dq.HEAD_GROUP)
        table, nwg, nslot = tables[i]
        dkdv = _positional(
            _dkdv.launch_dkdv,
            Q=q,
            K=k,
            V=v,
            DO=dout,
            NL=nl,
            ND=nd,
            DV_=dv,
            DK=dk,
            DS=ds_c.view(-1, _items.DSL_BLK_B // 2),
            TBL=table,
            scale=scale,
            Hq=hq,
            Hkv=hkv,
            nl_bytes=b * hq * sq * 4,
            nslot=nslot,
            nwg=nwg,
            stream=st,
        )
        dqds = _positional(
            _dq.launch_dqds,
            DS=ds_c.data_ptr(),
            K=k,
            DQ=dq,
            Sq=sq,
            Skv=skv,
            Hq=hq,
            Hkv=hkv,
            G=g,
            nkvt=skv // _dq.KV32,
            cshift=skv - sq,
            causal=1,
            q_off=q_off,
            ntile=ntile,
            HG=hg_q,
            nblk=ds_blocks,
            dq_bytes=b * sq * hq * D * 2,
            H0=h0 * g,
            HC=nhq,
            ny=ntile + nhead,
            ngz=b * nhq // hg_q,
            stream=st,
        )
        launches += [(_dkdv.launch_dkdv, dkdv), (_dq.launch_dqds, dqds)]
    return launches


@contextlib.contextmanager
def _joined(stream, side):
    """Around the launches of a backward that also runs on ``side`` (None: it does not): ``side`` starts after what
    ``stream`` has issued, and ``stream`` goes on after ``side``, also when a launch raised."""
    if side is None:
        yield
        return
    side.wait_stream(stream)
    try:
        yield
    finally:
        stream.wait_stream(side)


def flash_attn_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gradients (dq, dk, dv) of causal flash attention at head dim 128 on gfx1250, laid out like q, k, v.

    q / out / dout [B, Sq, Hq, 128], k / v [B, Skv, Hkv, 128] (Hq a multiple of Hkv), contiguous bf16 on one device;
    lse [B, Hq, Sq] fp32 natural log, as the forward returns it. The mask is bottom-right causal.
    """
    b, sq, hq, _ = q.shape
    skv, hkv = k.shape[1], k.shape[2]
    reason = shape_unsupported_reason(b, sq, skv, hq, hkv)
    if reason is not None:
        raise ValueError(reason)
    lse = lse.contiguous().float()
    stream = torch.cuda.current_stream()
    # k_delta_bshd (the row constants every later launch reads) goes out first: the chunks' host work, the scratch
    # allocation included, then overlaps it
    delta_launch, bufs = _delta_launch(dout, q, k, v, out, lse, softmax_scale, stream)
    _launch(*delta_launch)
    launches = _chunk_launches(dout, q, k, v, softmax_scale, stream, bufs)
    streams = {args[_PARAMS[launcher].index("stream")] for launcher, args in launches} - {stream}
    with _joined(stream, streams.pop() if streams else None):
        for launcher, args in launches:
            _launch(launcher, args)
    return bufs[2:]
