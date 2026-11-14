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

"""k_delta_bshd of the gfx1250 MLA attention backward: the row constants of the dK / dV kernel, NL = -LOG2E * lse and
ND = -scale * delta with delta = rowsum(dO * O).

o / do are [B, S, H, D_V] bf16; lse (natural log), NL and ND are [B, H, S] fp32.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx

# UNSTABLE(gfx1250): the raw LLVM LDS loads/stores and the workgroup-only gpu.barrier attribute have no stable API
# in FlyDSL 0.3.4.1.
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import range_constexpr

from primus_turbo.flydsl.utils.prims import LOG2E

from . import SOURCE_DIGEST
from .fwd_staging import lds_ptr

D_V = 128  # v/o head dim
DV8_V = D_V // 8  # 8-element vectors per v/o row
WAVE = 32
DELTA_THREADS = 256
LANES_PER_ROW = D_V // 8
ROWS_PER_PASS = DELTA_THREADS // LANES_PER_ROW
ROWS_DELTA = 128  # queries per k_delta workgroup (32 or 64: 1.3x / 2x slower at the DSv3 shape)
PASSES_PER_WG = ROWS_DELTA // ROWS_PER_PASS


def _bv(t, nb, dt, vec=1):
    """Buffer view of `t` as rows of `vec` elements of `dt`, with `nb` bytes in bounds."""
    nt = (1 << 31) // (vec * (dt.width // 8))
    b = fx.rocdl.make_buffer_tensor(t, num_records_bytes=fx.Int64(nb))
    return fx.Tensor(fx.make_view(fx.get_iter(b), fx.make_layout((nt, vec), (vec, 1))))


def _atom(dt, vec):
    return fx.make_copy_atom(fx.rocdl.BufferCopy(vec * dt.width), dt)


def _ldv(buf, tile, dt, vec):
    f = fx.make_rmem_tensor(fx.make_layout(vec, 1), dt)
    fx.copy_atom_call(_atom(dt, vec), fx.slice(buf, (tile, None)), f)
    return f.load()


def _ld1(buf, idx, dt):
    f = fx.make_rmem_tensor(fx.make_layout(1, 1), dt)
    fx.copy_atom_call(_atom(dt, 1), fx.slice(buf, (idx, None)), f)
    return f.load()[0]


def _st1(v, buf, idx, dt):
    f = fx.make_rmem_tensor(fx.make_layout(1, 1), dt)
    fx.memref_store_vec(fx.Vector.from_elements([v], dtype=dt), f)
    fx.copy_atom_call(_atom(dt, 1), f, fx.slice(buf, (idx, None)))


def _lds_barrier():
    """Workgroup barrier fenced on LDS only, so in-flight global loads need not drain."""
    fx.gpu.barrier(address_spaces=ir.Attribute.parse("[#gpu.address_space<workgroup>]"))


@flyc.kernel(name="k_mla_bwd_delta_bshd", known_block_size=[DELTA_THREADS, 1, 1])
def k_delta_bshd(
    DO: fx.Tensor,
    O: fx.Tensor,
    LSE: fx.Tensor,
    NL: fx.Tensor,
    ND: fx.Tensor,
    scale: fx.Float32,
    S: fx.Int32,
    H: fx.Int32,
    n_rows: fx.Int32,
):
    """delta = sum_d dO * O, NL = -LOG2E * lse and ND = -scale * delta, all [B, H, S] fp32.

    Workgroup (sblk, bh) owns queries [32 sblk, 32 sblk + 32) of batch-major head bh, so each output
    is one 128-B line; 16 lanes sum a row and the row sums meet in LDS."""
    tid = fx.Int32(fx.thread_idx.x)
    sblk = fx.Int32(fx.block_idx.x)
    bh = fx.Int32(fx.block_idx.y)
    b = bh // H
    h = bh - b * H
    g_do = _bv(DO, n_rows * (D_V * 2), fx.BFloat16, 8)
    g_o = _bv(O, n_rows * (D_V * 2), fx.BFloat16, 8)
    g_lse = _bv(LSE, n_rows * 4, fx.Float32)
    g_nl = _bv(NL, n_rows * 4, fx.Float32)
    g_nd = _bv(ND, n_rows * 4, fx.Float32)
    smem = fx.SharedAllocator().allocate(ROWS_DELTA * 4)
    lds0 = fx.Int32(fx.ptrtoint(smem.peek().ptr))

    lane_in_row = tid % fx.Int32(LANES_PER_ROW)
    row_in_group = tid // fx.Int32(LANES_PER_ROW)
    s0 = sblk * fx.Int32(ROWS_DELTA)
    # pass u reads query s0 + u*ROWS_PER_PASS + row_in_group, i.e. BSHD row (b*S + s)*H + h
    tiles = [
        ((b * S + s0 + fx.Int32(u * ROWS_PER_PASS) + row_in_group) * H + h) * fx.Int32(DV8_V) + lane_in_row
        for u in range_constexpr(PASSES_PER_WG)
    ]
    do_vecs = [_ldv(g_do, tiles[u], fx.BFloat16, 8).ir_value() for u in range_constexpr(PASSES_PER_WG)]
    o_vecs = [_ldv(g_o, tiles[u], fx.BFloat16, 8).ir_value() for u in range_constexpr(PASSES_PER_WG)]

    for u in range_constexpr(PASSES_PER_WG):
        do8 = fx.Vector(do_vecs[u])
        o8 = fx.Vector(o_vecs[u])
        e0 = fx.Float32(0.0)
        e1 = fx.Float32(0.0)
        for c in range_constexpr(4):  # the lane's 8 elements in 2 FMA chains
            e0 = e0 + fx.Float32(do8[2 * c]) * fx.Float32(o8[2 * c])
            e1 = e1 + fx.Float32(do8[2 * c + 1]) * fx.Float32(o8[2 * c + 1])
        acc = e0 + e1
        for sft in range_constexpr(LANES_PER_ROW.bit_length() - 1):
            acc = acc + fx.gpu.shuffle_xor(acc, 1 << sft, WAVE)
        # the xor butterfly leaves the row sum in all 16 lanes of the row; they all store it
        r = fx.Int32(u * ROWS_PER_PASS) + row_in_group
        llvm.store(fx.as_ir_value(acc), lds_ptr(lds0 + r * fx.Int32(4)))
    _lds_barrier()
    # Lane l of every wave reads row l; only wave 0 stores, the other waves store to index n_rows,
    # past the end of every buffer.
    l32 = tid % fx.Int32(ROWS_DELTA)
    v = fx.Float32(llvm.load(fx.Float32.ir_type, lds_ptr(lds0 + l32 * fx.Int32(4))))
    dst = bh * S + s0 + l32
    # (seqlen is a multiple of 64, not of ROWS_DELTA: the last block's rows past S are not stored)
    w0 = (tid < fx.Int32(ROWS_DELTA)) & (s0 + l32 < S)
    _st1(v * (fx.Float32(0.0) - scale), g_nd, w0.select(dst, n_rows), fx.Float32)
    lse_r = _ld1(g_lse, dst, fx.Float32)
    _st1(lse_r * fx.Float32(-LOG2E), g_nl, w0.select(dst, n_rows), fx.Float32)


@flyc.jit
def launch_delta(
    DO,
    O,
    LSE,
    NL,
    ND,
    scale: fx.Float32,
    S: fx.Int32,
    H: fx.Int32,
    n_rows: fx.Int32,
    nsblk: fx.Int32,
    nbh: fx.Int32,
    stream: fx.Stream,
):
    _ = SOURCE_DIGEST  # FlyDSL disk-cache key
    # grid = (ceil(S / ROWS_DELTA) query blocks, B * H heads)
    k_delta_bshd(DO, O, LSE, NL, ND, scale, S, H, n_rows).launch(
        grid=(nsblk, nbh, 1), block=(DELTA_THREADS, 1, 1), stream=stream
    )
