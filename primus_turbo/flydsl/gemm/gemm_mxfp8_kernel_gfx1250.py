###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Adapted from aiter (MIT, https://github.com/ROCm/aiter): aiter/ops/flydsl/kernels/gemm_a8w8_256x256_gfx1250.py
# (FlyDSL Project Contributors, Apache-2.0), gemm_a8w8_gfx1250.py and gemm_common_gfx1250.py.
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

"""MXFP8 NT GEMMs for gfx1250, one mainloop for three modes, fp32 accumulation, ``out`` bf16 or fp32, optionally
``out += result`` (beta = 1, rounded once):

* dense: ``out[M, N] = (a * sa)[M, K] @ (b * sb)[N, K]^T``;
* grouped-M (routed experts fwd / dgrad): ``out[rows_g] = a[rows_g] @ b[g]^T``, a's rows of expert g padded to 32
  (``offs_pad``), out tight (``offs``);
* variable-K (routed experts wgrad): ``out[g] (+)= a[:, k_g] @ b[:, k_g]^T`` over each expert's token range ``k_g``
  (``offs``, multiples of 128).

Operands: E4M3, K contiguous. Scales: one E8M0 byte per 32 K, packed ``[ceil(R/32)][ceil(K/128)][32][4]`` (fast) or
raw ``[rows, ceil(K/32)]`` (one 4 * kpair-byte TDM row per tile row: 20-45% slower). B: 16x16-preshuffled
``[N/16][K/16][16][16]`` (what the quantizers emit: the deep kpair-1 pipelines need it) or row-major (fallback; the
same speed at kpair 2).

The mainloop is aiter's compute-bound gfx1250 tile: 2x2 waves of up to 128x128, ``v_wmma_scale_f32_16x16x128_f8f6f4``
with one scale VGPR feeding two WMMAs (ROW0 / ROW1 lane halves), ``num_buffers`` LDS slots of ``kpair`` 128-wide K
steps filled by persistent per-wave TDM descriptors (wave 0: A, 1: B, 2: A scales, 3: B scales), a split
signal / wait barrier per slot, cluster multicast of the shared tiles. Changes: row-major A (and B when not
preshuffled) in padded LDS rows, mx32 scales, M / N / K tails by TDM out-of-bounds zero fill (no padding of the data;
K tails clamp the inner extent of every load), an LDS-staged epilogue with 16-byte buffer stores (no TDM store: the
suspect of the grouped kernel's rare silent error). No branch skips the TDM pipeline: a dead tile (past the
experts' tiles) runs one zero-extent K step and drops its stores; a cluster never mixes experts.
"""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.rocdl import cluster, tdm_ops
from flydsl.expr.typing import Constexpr, T, as_ir_value
from flydsl.expr.typing import Vector as Vec

from primus_turbo.flydsl.gemm.gemm_mxfp8_gfx1250_configs import (
    MXGemmConfig,
    pick_config,
    pick_grouped_config,
)
from primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_kernel_gfx1250 import (
    _make_lds_copy_ops,
    _require_gfx1250,
)
from primus_turbo.flydsl.utils.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)

WAVE = 32
WMMA_K = 128  # K per WMMA = per scale word (4 E8M0 bytes)
LDS_PAD = 16  # bytes after every A / B LDS row: conflict-free ds_load_b128 fragment reads
C_PAD_B = 16  # bytes after every row of the epilogue's C staging
DENSE, GROUPED_M, VARIABLE_K = 0, 1, 2
LDS_BYTES = 320 * 1024
CLUSTER_SYNC_REVS = 8  # a cluster barrier every this many pipeline revolutions keeps multicast partners close
_WG_BARRIER = -1
_E8M0 = (torch.uint8, getattr(torch, "float8_e8m0fnu", torch.uint8))


def geometry(c: MXGemmConfig, stage_f32: bool = False):
    """LDS plan of a config: (row bytes, stage bytes of A / B / SA / SB, base offsets of the four regions, arena);
    ``stage_f32``: the epilogue stages fp32 (fp32 out or beta = 1) instead of bf16."""
    super_k = WMMA_K * c.kpair
    row = super_k + LDS_PAD
    sc = (lambda r: r // 32 * super_k) if c.scale_4d else (lambda r: r * 4 * c.kpair)
    st = (c.tile_m * row, c.tile_n * (super_k if c.b_preshuffle else row), sc(c.tile_m), sc(c.tile_n))
    nb = c.num_buffers
    base_b = 0  # B first, as aiter (64 KiB aligned base)
    base_a = base_b + nb * st[1]
    base_sa = base_a + nb * st[0]
    base_sb = base_sa + nb * st[2]
    arena = max(base_sb + nb * st[3], c.tile_m * (c.tile_n * (4 if stage_f32 else 2) + C_PAD_B))
    return row, st, (base_a, base_b, base_sa, base_sb), arena


def _tdm_group1(mask, tile0: int, tile1: int, dim0, dim1, stride, pad: int):
    """GROUP1 of a 2-D TDM load of 1-byte elements, inner dim0 (``tile0`` bytes per row), outer dim1 (``tile1`` rows),
    early timeout on; ``pad`` bytes of LDS padding after every row; ``dim0`` / ``dim1`` are the (>= 0) extents from
    the descriptor's address, past which the load zero-fills."""
    cfg = 1 << 21
    if pad:
        cfg |= (1 << 20) | (((tile0 // 4).bit_length() - 2) << 22) | ((pad // 4 - 1) << 25)
    return [
        fx.Int32(cfg) | (fx.Int32(mask) & 0xFFFF),
        (dim0 & 0xFFFF) << 16,
        ((dim0 >> 16) & 0xFFFF) | ((dim1 & 0xFFFF) << 16),
        ((dim1 >> 16) & 0xFFFF) | fx.Int32(tile0 << 16),
        fx.Int32(tile1),
        fx.Int32(stride),
        fx.Int32(0),
        fx.Int32(0),
    ]


def _i64_table(arg, n: int):
    """The low words of an int64 table of ``n`` entries as uniform values (scalar buffer loads)."""
    rsrc = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(arg)), num_records_bytes=8 * n)
    vals = []
    for q in range_constexpr((2 * n + 3) // 4):
        v = Vec(buffer_load(rsrc, fx.Int32(4 * q), vec_width=4, is_scalar=True))
        vals += [fx.Int32(v[0]), fx.Int32(v[2])]
    return vals[:n]


@flyc.jit
def _launch_gemm_mxfp8(
    arg_c: fx.Pointer,
    arg_a: fx.Pointer,
    arg_b: fx.Pointer,
    arg_sa: fx.Pointer,
    arg_sb: fx.Pointer,
    arg_g0: fx.Pointer,
    arg_g1: fx.Pointer,
    i32_m: fx.Int32,
    i32_n: fx.Int32,
    i32_k: fx.Int32,
    i32_lda: fx.Int32,
    i32_ldb: fx.Int32,
    i32_ldc: fx.Int32,
    i32_ldsa: fx.Int32,
    i32_ldsb: fx.Int32,
    stream: fx.Stream,
    tile_m: Constexpr[int],
    tile_n: Constexpr[int],
    num_buffers: Constexpr[int],
    kpair: Constexpr[int],
    cluster_m: Constexpr[int],
    cluster_n: Constexpr[int],
    scale_4d: Constexpr[bool],
    b_shuf: Constexpr[bool],
    k_tail: Constexpr[bool],
    mode: Constexpr[int],
    n_groups: Constexpr[int],
    out_f32: Constexpr[bool],
    beta1: Constexpr[bool],
):
    """``arg_g0`` / ``arg_g1``: grouped-M: the tight and the 32-padded row offsets of the experts (int64, G + 1);
    variable-K: the 128-padded token offsets (``arg_g0``). Unused in dense mode."""
    cfg = MXGemmConfig(tile_m, tile_n, num_buffers, kpair, cluster_m, cluster_n, scale_4d, b_shuf)
    STG = 4 if out_f32 or beta1 else 2  # bytes per staged output element
    OB = 4 if out_f32 else 2  # bytes per output element
    LDS_ROW, STAGE, BASE, ARENA = geometry(cfg, STG == 4)
    G = n_groups
    assert tile_m in (128, 256) and tile_n in (128, 256) and kpair in (1, 2) and num_buffers in (2, 3, 4)
    assert ARENA <= LDS_BYTES, f"{cfg} needs {ARENA} B of LDS"
    use_cluster = cluster_m * cluster_n > 1
    NB, KPAIR = num_buffers, kpair
    SUPER_K = WMMA_K * KPAIR
    UNROLL = KPAIR * NB
    WM, WN = tile_m // 2, tile_n // 2  # 2x2 waves
    m_rep, n_rep = WM // 16, WN // 16
    half_m, half_n = m_rep // 2, n_rep // 2
    n_acc = m_rep * n_rep
    N_SA, N_SB = WM // 32, WN // 32  # scale VGPRs: one per 32 rows (ROW0 / ROW1 feed two WMMAs)
    BLOCK = 4 * WAVE
    DS_PER_FRAG = 4  # ds_load_b128 per 16-dword fragment
    # scale slot: raw [R, K/32] -> tile rows of 4 * KPAIR bytes; 4-D [R/32][K/128][32][4] -> tile/32 rows of SUPER_K
    SC_PAR, SC_SM = (WMMA_K, SUPER_K) if scale_4d else (4, 32 * 4 * KPAIR)
    SC_TILE = (lambda rows: (rows // 32, SUPER_K)) if scale_4d else (lambda rows: (rows, 4 * KPAIR))
    C_ROW_B = tile_n * STG + C_PAD_B
    # fragment geometry (16-row step, 16-byte K-chunk step, K-step offset): row-major rows, or aiter's 16x16
    # preshuffled B (per 16 rows [K/16][16 rows][16 B]: one TDM row of SUPER_K * 16 bytes per 16 rows)
    GEO_A = (16 * LDS_ROW, 32, WMMA_K)
    GEO_B = (SUPER_K * 16, 512, WMMA_K * 16) if b_shuf else GEO_A
    kname = f"gemm_mxfp8_gfx1250_{('dense', f'grouped{G}', f'vark{G}')[mode]}_t{tile_m}x{tile_n}_nb{NB}_kp{KPAIR}" \
        f"_c{cluster_m}x{cluster_n}" + ("_s4d" if scale_4d else "") + ("_bsh" if b_shuf else "") + \
        ("_kt" if k_tail else "") + ("_f32" if out_f32 else "") + ("_acc" if beta1 else "")

    @flyc.kernel(name=kname, known_block_size=[BLOCK, 1, 1])
    def kernel_gemm_mxfp8(
        arg_c: fx.Pointer,
        arg_a: fx.Pointer,
        arg_b: fx.Pointer,
        arg_sa: fx.Pointer,
        arg_sb: fx.Pointer,
        arg_g0: fx.Pointer,
        arg_g1: fx.Pointer,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        i32_k: fx.Int32,
        i32_lda: fx.Int32,
        i32_ldb: fx.Int32,
        i32_ldc: fx.Int32,
        i32_ldsa: fx.Int32,
        i32_ldsb: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        bid_x, bid_y, bid_z = fx.block_idx
        wave = rocdl.readfirstlane(T.i32, tid // WAVE)
        lane = tid % WAVE
        lane16, kgrp = lane % 16, lane // 16
        wave_m, wave_n = wave // 2, wave % 2
        zero = fx.Int32(0)
        blk_n = fx.Int32(bid_y) * tile_n
        n_valid = i32_n - blk_n
        expert, k0, klen = zero, zero, i32_k
        if const_expr(mode == DENSE):
            blk_m = (fx.Int32(bid_z) * fx.Int32(fx.grid_dim.x) + fx.Int32(bid_x)) * tile_m
            a_row0, out_row0, m_valid = blk_m, blk_m, i32_m - blk_m  # <= 0 for a workgroup past the edge
            supers = (i32_k + (SUPER_K - 1)) // SUPER_K
        elif const_expr(mode == GROUPED_M):
            # m-tile t of the experts' tiles in order (ceil(len_g / tile_m) each); a tile past them is dead
            t = fx.Int32(bid_z) * fx.Int32(fx.grid_dim.x) + fx.Int32(bid_x)
            offs, offs_pad = _i64_table(arg_g0, G + 1), _i64_table(arg_g1, G + 1)
            acc, first, len_g, in0, out0 = zero, zero, zero, zero, zero
            for g in range_constexpr(G):
                n_g = (offs[g + 1] - offs[g] + (tile_m - 1)) // tile_m
                hit = (t >= acc) & (t < acc + n_g)
                expert, first = hit.select(fx.Int32(g), expert), hit.select(acc, first)
                len_g = hit.select(offs[g + 1] - offs[g], len_g)
                in0, out0 = hit.select(offs_pad[g], in0), hit.select(offs[g], out0)
                acc = acc + n_g
            local = (t - first) * tile_m
            rows = len_g - local
            m_valid = (rows < tile_m).select(rows, fx.Int32(tile_m))
            m_valid = (t < acc).select(m_valid, zero)  # dead: no row
            a_row0, out_row0 = in0 + local, out0 + local
            supers = (t < acc).select((i32_k + (SUPER_K - 1)) // SUPER_K, fx.Int32(1))
        else:  # VARIABLE_K: workgroup (x, y, g) = output tile of expert g over its tokens [offs[g], offs[g + 1])
            blk_m = fx.Int32(bid_x) * tile_m
            a_row0, out_row0, m_valid = blk_m, blk_m, i32_m - blk_m
            expert = fx.Int32(bid_z)
            rsrc = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(arg_g0)), num_records_bytes=8 * (G + 1))
            o2 = Vec(buffer_load(rsrc, expert * 2, vec_width=4, is_scalar=True))
            k0, klen = fx.Int32(o2[0]), fx.Int32(o2[2]) - fx.Int32(o2[0])
            supers = (klen + (SUPER_K - 1)) // SUPER_K
            supers = (supers > zero).select(supers, fx.Int32(1))  # an empty expert: one zero K step
        a_mask = b_mask = 0
        if const_expr(use_cluster):
            lx = fx.Int32(rocdl.cluster_workgroup_id_x())
            ly = fx.Int32(rocdl.cluster_workgroup_id_y())
            # flat id in the cluster = x + y * cluster_m; A is shared along y (same M tile), B along x
            a_mask = fx.Int32(sum(1 << (y * cluster_m) for y in range(cluster_n))) << lx
            b_mask = fx.Int32((1 << cluster_m) - 1) << (ly * cluster_m)

        arena = fx.SharedAllocator(static=False)
        arena.allocate(ARENA)
        base_ptr = arena.base_ptr
        lds0 = fx.Int32(fx.ptrtoint(base_ptr))

        # ---- per-wave persistent TDM descriptor (wave 0: A, 1: B, 2: A scales, 3: B scales) ----
        def _owner(w):
            """[addr lo, addr hi | type, LDS slot 0, slot step, K step, inner extent, GROUP1 words 0, 2 (dim1
            half), 3, 4, 5] of operand ``w``; the K step and inner extent are in the operand's bytes."""
            rows_valid, blk = (m_valid, a_row0) if w % 2 == 0 else (n_valid, blk_n)
            rows = tile_m if w % 2 == 0 else tile_n
            ld, ptr, mask = ((i32_lda, arg_a, a_mask), (i32_ldb, arg_b, b_mask), (i32_ldsa, arg_sa, a_mask),
                             (i32_ldsb, arg_sb, b_mask))[w]
            pad = LDS_PAD if w < 2 else 0
            eoff = fx.Int64(0)
            if const_expr(mode == GROUPED_M and w % 2 == 1):  # b [G, N, K], its scales [G, N(/32), ...]
                eoff = fx.Int64(expert) * fx.Int64(i32_n // (32 if w == 3 and scale_4d else 1)) * fx.Int64(ld)
            if const_expr(w == 1 and b_shuf):  # [N/16][K * 16]
                rows, inner, step, total, row0, ld, pad, kb = tile_n // 16, SUPER_K * 16, SUPER_K * 16, klen * 16, \
                    blk // 16, i32_ldb * 16, 0, k0 * 16
                rows_valid = (rows_valid + 15) // 16
            elif const_expr(w < 2):
                inner, step, total, row0, kb = SUPER_K, SUPER_K, klen, blk, k0
            elif const_expr(scale_4d):  # (variable-K token offsets are multiples of 128)
                (rows, inner), step, total, row0 = SC_TILE(rows), SUPER_K, (klen + 127) // 128 * 128, blk // 32
                rows_valid, kb = (rows_valid + 31) // 32, k0
            else:
                (rows, inner), step, total, row0, kb = SC_TILE(rows), 4 * KPAIR, (klen + 31) // 32, blk, k0 // 32
            ext1 = (rows_valid > zero).select(rows_valid, zero)
            addr = fx.Int64(fx.ptrtoint(ptr)) + eoff + fx.Int64(row0) * fx.Int64(ld) + fx.Int64(kb)
            g1 = _tdm_group1(mask, inner, rows, total, ext1, ld, pad)
            return [fx.Int32(addr), fx.Int32(addr >> 32) | fx.Int32(-(1 << 31)), lds0 + BASE[w], fx.Int32(STAGE[w]),
                    fx.Int32(step), total, g1[0], g1[2] & fx.Int32(-65536), g1[3], g1[4], g1[5]]

        d = [zero] * 11
        if wave == 0:
            d = _owner(0)
        elif wave == 1:
            d = _owner(1)
        elif wave == 2:
            d = _owner(2)
        else:
            d = _owner(3)
        t_lo, t_hi, t_lds, t_lds_step, t_step, t_total, t_g10, t_g12_hi, t_g13, t_g14, t_g15 = d
        last_delta = (supers - 1) * t_step

        def _prepare(slot, delta):
            """GROUP0 / GROUP1 of the load of K offset ``delta`` (this wave's units) into ``slot``."""
            lo, hi = tdm_ops.add_addr_with_carry(t_lo, t_hi, delta)
            g0 = Vec.from_elements([fx.Int32(1), t_lds + t_lds_step * slot, fx.Int32(lo), fx.Int32(hi)], fx.Int32)
            d0 = t_total - delta if const_expr(k_tail) else t_total
            d0 = (d0 > zero).select(d0, zero)
            g1 = Vec.from_elements(
                [t_g10, (d0 & 0xFFFF) << 16, ((d0 >> 16) & 0xFFFF) | t_g12_hi, t_g13, t_g14, t_g15, zero, zero],
                fx.Int32,
            )
            return tdm_ops.TDMDescriptor2D(as_ir_value(g0), as_ir_value(g1))

        wmb, wnb = wave_m * WM, wave_n * WN
        lds_load_b32, _ = _make_lds_copy_ops(32)
        lds_load_b128, _ = _make_lds_copy_ops(128)

        # ---- fragment / scale read addresses per unrolled stage (displacements stay DS immediates) ----
        a_byte = (wmb + lane16) * LDS_ROW + kgrp * 16
        b_byte = (wnb // 16) * GEO_B[0] + kgrp * 256 + lane16 * 16 if b_shuf else (wnb + lane16) * LDS_ROW + kgrp * 16
        if const_expr(scale_4d):
            sa_byte, sb_byte = (wmb // 32) * SUPER_K + lane * 4, (wnb // 32) * SUPER_K + lane * 4
        else:
            sa_byte, sb_byte = (wmb + lane) * (4 * KPAIR), (wnb + lane) * (4 * KPAIR)
        def _planar(off, lane_byte):
            # a pointer offset, not an integer add: LLVM keeps one base VGPR per stage and the fragment
            # displacements in the 16-bit DS offset instead of folding the stage base into every load
            return fx.index_cast(T.index, fx.ptrtoint(fx.add_offset(base_ptr, off))) + fx.index_cast(T.index, lane_byte)

        addr_a, addr_b, addr_sa, addr_sb = [], [], [], []
        for st in range_constexpr(UNROLL):
            slot, par = st // KPAIR, st % KPAIR
            addr_a.append(_planar(BASE[0] + slot * STAGE[0] + par * WMMA_K, a_byte))
            addr_b.append(_planar(BASE[1] + slot * STAGE[1] + par * GEO_B[2], b_byte))
            addr_sa.append(_planar(BASE[2] + slot * STAGE[2] + par * SC_PAR, sa_byte))
            addr_sb.append(_planar(BASE[3] + slot * STAGE[3] + par * SC_PAR, sb_byte))

        def _join(v):
            return v[0].shuffle(v[1], list(range(8))).shuffle(v[2].shuffle(v[3], list(range(8))), list(range(16)))

        def _load_frag(addr, idx, geo=GEO_A):
            return _join([Vec(lds_load_b128(addr, idx * geo[0] + geo[1] * j)) for j in range_constexpr(4)])

        def _rmem(n, v):
            t = fx.make_rmem_tensor(n, fx.Int32)
            t.store(v)
            return t

        wmma = [
            [
                fx.make_mma_atom(
                    fx.rocdl.WMMAScale(16, 16, WMMA_K, fx.Float8E4M3FN, fx.Float8E4M3FN, fx.Float32,
                                       opsel_a=sb_sel, opsel_b=sa_sel)
                )
                for sa_sel in range_constexpr(2)
            ]
            for sb_sel in range_constexpr(2)
        ]
        c_frags = [fx.make_rmem_tensor(8, fx.Float32) for _ in range_constexpr(n_acc)]
        for cf in c_frags:
            cf.store(Vec.filled(8, 0.0, fx.Float32))

        def _mma(wm, wn, act, wt, sa_k, sb_k):
            # WMMA "A" = the B fragment (N rows), "B" = the A fragment: each lane's 8 results are 8 consecutive N
            idx = wm * n_rep + wn
            fx.gemm(wmma[wn % 2][wm % 2], c_frags[idx], wt, act, c_frags[idx],
                    scale_a=sb_k[wn // 2], scale_b=sa_k[wm // 2])

        seed_a = [fx.make_rmem_tensor(16, fx.Int32) for _ in range_constexpr(half_m)]
        seed_b = [fx.make_rmem_tensor(16, fx.Int32) for _ in range_constexpr(half_n)]
        seed_sa = [fx.make_rmem_tensor(1, fx.Int32) for _ in range_constexpr(N_SA)]
        seed_sb = [fx.make_rmem_tensor(1, fx.Int32) for _ in range_constexpr(N_SB)]

        def _seed_thunks(stage, parity=0):
            """Producers of the next stage's scales and first-half fragments: (early, tail)."""
            head = [
                (lambda i=i: seed_sa[i].store(Vec.from_elements([lds_load_b32(addr_sa[stage], i * SC_SM)[0]],
                                                                fx.Int32)))
                for i in range_constexpr(N_SA)
            ] + [
                (lambda i=i: seed_sb[i].store(Vec.from_elements([lds_load_b32(addr_sb[stage], i * SC_SM)[0]],
                                                                fx.Int32)))
                for i in range_constexpr(N_SB)
            ]
            a_th = [(lambda i=i: seed_a[i].store(_load_frag(addr_a[stage], i))) for i in range_constexpr(half_m)]
            b_th = [(lambda i=i: seed_b[i].store(_load_frag(addr_b[stage], i, GEO_B))) for i in range_constexpr(half_n)]
            return (head + a_th, b_th) if parity == 0 else (head + b_th, a_th)

        def _chain(d, pos, t):
            d[pos] = t if pos not in d else (lambda p=d[pos], q=t: (p(), q()))

        def _stage(stage, next_stage, future_slot, future_delta, fence_out, has_next, boundary, parity=0):
            """One 128-wide K step: four quadrants of the wave tile, at most three fragment groups live; the
            second-half fragments of this step and the first half of the next are read in between the WMMAs."""
            sa_k = [seed_sa[i].load()[0] for i in range_constexpr(N_SA)]
            sb_k = [seed_sb[i].load()[0] for i in range_constexpr(N_SB)]
            a0, b0 = seed_a, seed_b
            rocdl.sched_barrier(0)
            refill = boundary and future_delta is not None
            prepared = _prepare(future_slot, future_delta) if const_expr(refill) else None
            n_slots = half_m * half_n
            nxt = {}

            def _quad(wm0, wn0, act, wt, produce, n_fast, pre=None):
                if const_expr(pre is not None):
                    pre()
                last = max(produce) if produce else -1
                for pos in range_constexpr(n_slots):
                    i, j = (pos // len(wt), pos % len(wt)) if n_fast else (pos % len(act), pos // len(act))
                    _mma(wm0 + i, wn0 + j, act[i], wt[j], sa_k, sb_k)
                    if const_expr(pos in produce):
                        produce[pos]()
                    if const_expr(pos == last):
                        for _ in range_constexpr(2 * len(produce)):
                            rocdl.sched_mfma(1)
                            rocdl.sched_dsrd(DS_PER_FRAG // 2)
                        rocdl.sched_barrier(0)

            def _mk(kind, key):
                n = half_m if kind == "a" else half_n
                addr, geo = (addr_a[stage], GEO_A) if kind == "a" else (addr_b[stage], GEO_B)
                nxt[key] = [None] * n

                def _go(i):
                    nxt[key][i] = _rmem(16, _load_frag(addr, n + i, geo))

                return {i: (lambda i=i: _go(i)) for i in range_constexpr(n)}

            def _sig():
                rocdl.sched_barrier(0)
                rocdl.s_wait_dscnt(0)
                rocdl.sched_barrier(0)
                if const_expr(boundary and has_next):
                    tdm_ops.tensor_wait(fence_out)
                    rocdl.s_barrier_signal(_WG_BARRIER)
                rocdl.sched_barrier(0)

            def _wait_refill():
                if const_expr(boundary and has_next):
                    # LLVM hoists the wait above the slack WMMAs; pinning them in between measured +0.2% (noise)
                    rocdl.s_barrier_wait(_WG_BARRIER)
                    if const_expr(refill):
                        tdm_ops.tensor_load_2d(prepared)
                rocdl.sched_barrier(0)

            slack = max(1, n_slots // 4)
            early, tail = _seed_thunks(next_stage, parity) if const_expr(has_next) else ([], [])
            q2, q3 = {slack - 1: _wait_refill}, {}
            for i, t in enumerate(early):  # Q2 after the slack, then the front of Q3 ...
                p = slack + i
                _chain(q2, p, t) if p < n_slots else _chain(q3, min(p - n_slots, n_slots - 1), t)
            for i, t in enumerate(tail):  # ... and the last fragment group's reuse slots at the end of Q3
                _chain(q3, n_slots - len(tail) + i, t)
            fence = lambda: rocdl.sched_barrier(0)  # noqa: E731
            if const_expr(parity == 0):  # (A0,B0) -> (A0,B1) -> (A1,B1) -> (A1,B0)
                _quad(0, 0, a0, b0, _mk("b", "b1"), False)
                _quad(0, half_n, a0, nxt["b1"], _mk("a", "a1"), True, pre=fence)
                _quad(half_m, half_n, nxt["a1"], nxt["b1"], q2, True, pre=_sig)
                _quad(half_m, 0, nxt["a1"], b0, q3, True, pre=fence)
            else:  # (A0,B0) -> (A1,B0) -> (A1,B1) -> (A0,B1)
                _quad(0, 0, a0, b0, _mk("a", "a1"), True)
                _quad(half_m, 0, nxt["a1"], b0, _mk("b", "b1"), False, pre=fence)
                _quad(half_m, half_n, nxt["a1"], nxt["b1"], q2, False, pre=_sig)
                _quad(0, half_n, a0, nxt["b1"], q3, False, pre=fence)

        # ---- prologue: fill every slot (past the last K step: reload it), wait for slot 0, read its seeds ----
        if const_expr(use_cluster):
            cluster.cluster_barrier()
        for i in range_constexpr(NB):
            d_pro = fx.Int32(i) * t_step
            tdm_ops.tensor_load_2d(_prepare(i, (d_pro < last_delta).select(d_pro, last_delta)))
        tdm_ops.tensor_wait(NB - 1)
        gpu.barrier()
        for group in _seed_thunks(0):
            for thunk in group:
                thunk()

        n_full = (supers + (NB - 1)) // NB - 1
        drain = supers - n_full * NB  # 1..NB slots left after the steady revolutions
        slot_delta = [fx.Int32(c + NB) * t_step for c in range_constexpr(NB)]

        def _run_steady(parity):
            for rev in range(n_full):
                rev_delta = rev * NB * t_step
                for g in range_constexpr(UNROLL):
                    delta = rev_delta + slot_delta[g // KPAIR]
                    delta = (delta < last_delta).select(delta, last_delta)
                    _stage(g, (g + 1) % UNROLL, g // KPAIR, delta, NB - 2, True, g % KPAIR == KPAIR - 1, parity)
                if const_expr(use_cluster):
                    if (rev % CLUSTER_SYNC_REVS) == (CLUSTER_SYNC_REVS - 1):
                        cluster.cluster_barrier()

        # odd waves run the quadrants B-first: the two wave pairs read different fragment halves at a time
        wave_parity = fx.Int32(
            _llvm.inline_asm(T.i32, [as_ir_value(rocdl.wave_id())], "s_and_b32 $0, $1, 1", "=s,s,~{scc}",
                             has_side_effects=True)
        )
        if wave_parity == 0:
            _run_steady(0)
        else:
            _run_steady(1)
        rocdl.s_wait_dscnt(0)
        rocdl.sched_barrier(0)
        for g in range_constexpr(UNROLL):
            if g < drain * KPAIR:
                _stage(g, (g + 1) % UNROLL, g // KPAIR, None, NB - 2 - g // KPAIR, g < UNROLL - 1,
                       g % KPAIR == KPAIR - 1)

        # ---- epilogue: accumulators -> rows in LDS (bf16, or fp32 for fp32 out / beta = 1) -> 16-byte buffer
        # stores (beta = 1: + the old C, read at the same offset, rounded once); rows / columns past M / N dropped
        accs = [c_frags[i].load() for i in range_constexpr(n_acc)]
        tdm_ops.tensor_wait(0)
        if const_expr(use_cluster):
            cluster.cluster_barrier()  # no multicast into this LDS is still in flight
        else:
            gpu.barrier()
        for wm in range_constexpr(m_rep):
            for wn in range_constexpr(n_rep):
                h = accs[wm * n_rep + wn] if const_expr(STG == 4) else accs[wm * n_rep + wn].to(fx.BFloat16)
                row, col = wmb + wm * 16 + lane16, wnb + wn * 16 + kgrp * 8
                fx.ptr_store(h.bitcast(fx.Int8), base_ptr + row * C_ROW_B + col * STG)
        gpu.barrier()
        m_ext = (m_valid < tile_m).select(m_valid, fx.Int32(tile_m))
        m_ext = (m_ext > zero).select(m_ext, zero)
        n_ext = (n_valid < tile_n).select(n_valid, fx.Int32(tile_n))
        c_addr = fx.Int64(fx.ptrtoint(arg_c)) + (fx.Int64(out_row0) * fx.Int64(i32_ldc) + fx.Int64(blk_n)) * OB
        if const_expr(mode == VARIABLE_K):  # out [G, M, ldc]
            c_addr = c_addr + fx.Int64(expert) * fx.Int64(i32_m) * fx.Int64(i32_ldc) * OB
        ldc_b = i32_ldc * OB
        c_res = create_buffer_resource_from_addr(c_addr, num_records_bytes=m_ext * ldc_b)
        # lane tid stores 16-byte chunk tid % CH of rows tid // CH + i * RPI: one column per lane, so the column mask
        # is folded into the lane's base (past num_records, and every later row of it too: dropped)
        EPC = 16 // OB  # output elements per chunk
        CH = tile_n // EPC
        RPI = BLOCK // CH
        c16, r0 = tid % CH, tid // CH
        lane_b = (c16 * EPC < n_ext).select(r0 * ldc_b + c16 * 16, m_ext * ldc_b)
        step_b = RPI * ldc_b
        GRP = 65536 // (RPI * C_ROW_B)  # iterations per LDS base: displacements stay 16-bit DS offsets
        for i in range_constexpr(tile_m // RPI):
            if const_expr(i % GRP == 0):
                rd = _planar(i * RPI * C_ROW_B, r0 * C_ROW_B + c16 * EPC * STG)
            off = lane_b + step_b * i
            if const_expr(STG == 2):
                v = lds_load_b128(rd, (i % GRP) * RPI * C_ROW_B)
            else:
                acc_v = [Vec(lds_load_b128(rd, (i % GRP) * RPI * C_ROW_B + 16 * j)) for j in range_constexpr(STG // OB)]
                acc_v = acc_v[0].shuffle(acc_v[1], list(range(8))) if const_expr(OB == 2) else acc_v[0]
                acc_v = acc_v.bitcast(fx.Float32)
                if const_expr(beta1):
                    old = Vec(buffer_load(c_res, off // 4, vec_width=4, dtype=fx.Int32))
                    acc_v = acc_v + (old.bitcast(fx.BFloat16).to(fx.Float32) if const_expr(OB == 2)
                                     else old.bitcast(fx.Float32))
                v = (acc_v.to(fx.BFloat16) if const_expr(OB == 2) else acc_v).bitcast(fx.Int32)
            buffer_store(v, c_res, off, offset_is_bytes=True)

    # Launch: grid.x runs up to 32 M tiles (a power of two dividing them), grid.y the N tiles, grid.z the M chunks
    # (grouped-M: the m tiles of all experts, at most ceil(M / tile_m) + G; variable-K: grid.z = the expert).
    gx = (i32_m + (tile_m - 1)) // tile_m + (G if mode == GROUPED_M else 0)
    gx = (gx + (cluster_m - 1)) // cluster_m * cluster_m
    gy = ((i32_n + (tile_n - 1)) // tile_n + (cluster_n - 1)) // cluster_n * cluster_n
    pow2 = gx & -gx
    capped = (pow2 < 32).select(pow2, fx.Int32(32))
    m_run = ((gx > 32) & (pow2 >= 8) & ((capped % cluster_m) == 0)).select(capped, gx)
    if const_expr(mode == VARIABLE_K):
        m_run = gx
    attrs = {"rocdl.cluster_dims": f"{cluster_m},{cluster_n},1"} if use_cluster else {}
    kernel_gemm_mxfp8(
        arg_c, arg_a, arg_b, arg_sa, arg_sb, arg_g0, arg_g1, i32_m, i32_n, i32_k, i32_lda, i32_ldb, i32_ldc,
        i32_ldsa, i32_ldsb, value_attrs=attrs,
    ).launch(
        grid=(m_run, gy, G if mode == VARIABLE_K else gx // m_run),
        block=(BLOCK, 1, 1),
        stream=stream,
        cluster=(cluster_m, cluster_n, 1) if use_cluster else None,
    )


_launch_gemm_mxfp8.compile_hints["llvm_options"] = {
    "amdgpu-expert-scheduling-mode": True,
    "amdgpu-kernarg-preload": True,
    "amdgpu-kernarg-preload-count": 16,
}

_COMPILED: dict = {}
_F16_OUT = {torch.bfloat16: False, torch.float32: True}


def _divisor_cluster(tiles: int, c: int) -> int:
    """The largest power of two <= ``c`` that divides ``tiles``."""
    while tiles % c:
        c //= 2
    return c


def _check_scales(sa, sb, a_rows: int, b_rows: int, K: int, who: str, b_batch: tuple = ()):
    """Both raw ``[rows, ceil(K/32)]`` (K/32 contiguous) or both packed ``[ceil(rows/32), ceil(K/128), 32, 4]``
    (contiguous); ``b_batch``: leading expert dims of b's scales. Returns (packed, ld_sa, ld_sb)."""
    if sa.dtype not in _E8M0 or sb.dtype not in _E8M0:
        raise ValueError(f"{who}: scales must be E8M0 bytes, got {sa.dtype}, {sb.dtype}")
    packed = sa.dim() == 4
    if packed:
        kb4 = (K + 127) // 128
        want_a, want_b = ((a_rows + 31) // 32, kb4, 32, 4), b_batch + ((b_rows + 31) // 32, kb4, 32, 4)
        if tuple(sa.shape) != want_a or tuple(sb.shape) != want_b or not (sa.is_contiguous() and sb.is_contiguous()):
            raise ValueError(f"{who}: packed scales must be contiguous {want_a}, {want_b}, got "
                             f"{tuple(sa.shape)}, {tuple(sb.shape)}")
        return True, kb4 * 128, kb4 * 128
    kb = (K + 31) // 32
    if tuple(sa.shape) != (a_rows, kb) or tuple(sb.shape) != b_batch + (b_rows, kb) or sa.stride(-1) != 1 \
            or sb.stride(-1) != 1 or (b_batch and not sb.is_contiguous()):
        raise ValueError(f"{who}: scales must be [{a_rows}, {kb}], {b_batch + (b_rows, kb)} (K/32 contiguous), got "
                         f"{tuple(sa.shape)}, {tuple(sb.shape)}")
    return False, sa.stride(0), sb.stride(-2)


def _launch(who, mode, cfg, out, a, b, sa, sb, g0, g1, M, N, K, lda, ldb, ldsa, ldsb, G, b_preshuffled, beta1,
            k_tail):
    """Validate the common layout rules, then run the cached compiled launcher of this constexpr tuple."""
    if a.dtype != torch.float8_e4m3fn or b.dtype != torch.float8_e4m3fn:
        raise ValueError(f"{who}: a / b must be float8_e4m3fn, got {a.dtype}, {b.dtype}")
    if a.stride(-1) != 1 or b.stride(-1) != 1 or lda % 16 or ldb % 16 or K % 16:
        raise ValueError(f"{who}: K ({K}) and the a / b row strides must be multiples of 16 bytes, K contiguous")
    if N % 8 or out.stride(-1) != 1 or out.stride(-2) % 8:
        raise ValueError(f"{who}: N ({N}) and the out row stride must be multiples of 8 (16-byte output rows)")
    if b_preshuffled and (N % 16 or ldb != K):
        raise ValueError(f"{who}: a preshuffled b needs N % 16 == 0 and dense [N/16][K/16][16][16] blocks")
    for t in (a, b, out):
        if t.data_ptr() % 16:
            raise ValueError(f"{who}: a, b and out must be 16-byte aligned")
    # a cluster must tile the grid exactly: a workgroup wholly past M or N would sit in a multicast group;
    # grouped-M m tiles belong to different experts, so it multicasts A along N only
    cm = 1 if mode == GROUPED_M else _divisor_cluster((M + cfg.tile_m - 1) // cfg.tile_m, cfg.cluster_m)
    cn = _divisor_cluster((N + cfg.tile_n - 1) // cfg.tile_n, cfg.cluster_n)
    const = (cfg.tile_m, cfg.tile_n, cfg.num_buffers, cfg.kpair, cm, cn, bool(sa.dim() == 4), bool(b_preshuffled),
             bool(k_tail), mode, G, _F16_OUT[out.dtype], bool(beta1))
    tabs = (g0 if g0 is not None else out, g1 if g1 is not None else out)
    args = (out.data_ptr(), a.data_ptr(), b.data_ptr(), sa.data_ptr(), sb.data_ptr(), tabs[0].data_ptr(),
            tabs[1].data_ptr(), M, N, K, lda, ldb, out.stride(-2), ldsa, ldsb, torch.cuda.current_stream(), *const)
    fn = _COMPILED.get(const)
    if fn is None:
        ptr = lambda t: flyc.from_c_void_p(fx.Int8, t.data_ptr(), assumed_align=16)  # noqa: E731
        first = (ptr(out), ptr(a), ptr(b), ptr(sa), ptr(sb), ptr(tabs[0]), ptr(tabs[1])) + args[7:]
        fn = flyc.compile(_launch_gemm_mxfp8, *first)  # compiles and runs this call
        if fn is not None:
            _COMPILED[const] = fn
        return out
    fn(*args)
    return out


def _out(out, shape, out_dtype, accumulate, device, who):
    if out_dtype not in _F16_OUT:
        raise NotImplementedError(f"{who}: out_dtype {out_dtype} (bf16 or fp32)")
    if out is None:
        if accumulate:
            raise ValueError(f"{who}: accumulate=True needs out")
        return torch.empty(shape, dtype=out_dtype, device=device)
    if out.dtype != out_dtype or tuple(out.shape) != tuple(shape):
        raise ValueError(f"{who}: out must be {out_dtype} {tuple(shape)}, got {out.dtype} {tuple(out.shape)}")
    return out


def gemm_mxfp8_gfx1250(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    out: torch.Tensor | None = None,
    config: MXGemmConfig | None = None,
    b_preshuffled: bool = False,
    accumulate: bool = False,
) -> torch.Tensor:
    """``out[M, N] (+)= (a * a_scale) @ (b * b_scale)^T`` with E4M3 ``a [M, K]``, ``b [N, K]`` (K contiguous, row
    strides multiples of 16 bytes) and E8M0 scales (uint8 or float8_e8m0fnu), both packed
    ``[ceil(rows/32), ceil(K/128), 32, 4]`` (``mxfp8_ref.pack_scales``, the fast layout) or both raw
    ``[rows, ceil(K/32)]``. ``b_preshuffled``: ``b`` in the 16x16 order ``[N/16][K/16][16][16]`` (N, K % 16 == 0).
    ``out`` bf16 or fp32, N a multiple of 8; ``accumulate``: ``out = round(out + fp32 result)``.
    ``config`` overrides the static per-shape table."""
    _require_gfx1250(a.device)
    M, K = a.shape
    N = b.shape[0]
    if b.shape[1] != K:
        raise ValueError(f"gemm_mxfp8_gfx1250: a {tuple(a.shape)} and b {tuple(b.shape)} disagree on K")
    out = _out(out, (M, N), out_dtype, accumulate, a.device, "gemm_mxfp8_gfx1250")
    if M == 0 or N == 0:
        return out
    if K == 0:
        return out if accumulate else out.zero_()
    cfg = config or pick_config(M, N, K, b_preshuffled)
    packed, ldsa, ldsb = _check_scales(a_scale, b_scale, M, N, K, "gemm_mxfp8_gfx1250")
    k_tail = K % (WMMA_K * cfg.kpair) != 0
    return _launch("gemm_mxfp8_gfx1250", DENSE, cfg, out, a, b, a_scale, b_scale, None, None, M, N, K, a.stride(0),
                   b.stride(0), ldsa, ldsb, 1, b_preshuffled, accumulate, k_tail)


def grouped_gemm_mxfp8_gfx1250(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    group_offs: torch.Tensor,
    group_offs_padded: torch.Tensor,
    m_out: int,
    out_dtype: torch.dtype = torch.bfloat16,
    out: torch.Tensor | None = None,
    config: MXGemmConfig | None = None,
    b_preshuffled: bool = True,
) -> torch.Tensor:
    """Routed experts fwd / dgrad: ``out[offs[g] + i] = a[offs_pad[g] + i] @ b[g]^T`` for ``i < offs[g + 1] -
    offs[g]``. ``a [M_pad, K]`` holds each expert's rows from ``group_offs_padded[g]`` (int64 [G + 1], multiples of
    32: Turbo's rowwise-padded MX layout), ``b [G, N, K]`` (16x16-preshuffled per expert by default), scales packed
    ``[ceil(M_pad/32), ..]`` / ``[G, N/32, ..]`` (or raw), ``group_offs`` (int64 [G + 1]) the tight offsets of the
    ``m_out`` output rows. Device tables only: no host sync."""
    _require_gfx1250(a.device)
    who = "grouped_gemm_mxfp8_gfx1250"
    G, N, K = b.shape
    if a.shape[1] != K or group_offs.numel() != G + 1 or group_offs_padded.numel() != G + 1:
        raise ValueError(f"{who}: a {tuple(a.shape)}, b {tuple(b.shape)}, offsets {group_offs.numel()} / "
                         f"{group_offs_padded.numel()} entries")
    if group_offs.dtype != torch.int64 or group_offs_padded.dtype != torch.int64:
        raise ValueError(f"{who}: group offsets must be int64")
    if not b.is_contiguous():
        raise ValueError(f"{who}: b must be contiguous [G, N, K]")
    out = _out(out, (m_out, N), out_dtype, False, a.device, who)
    if m_out == 0 or N == 0:
        return out
    if K == 0:
        return out.zero_()
    cfg = config or pick_grouped_config(GROUPED_M, m_out // max(G, 1), N, K, b_preshuffled)
    packed, ldsa, ldsb = _check_scales(a_scale, b_scale, a.shape[0], N, K, who, (G,))
    if packed and N % 32:
        raise ValueError(f"{who}: packed b scales need N % 32 == 0")
    return _launch(who, GROUPED_M, cfg, out, a, b, a_scale, b_scale, group_offs, group_offs_padded, m_out, N, K,
                   a.stride(0), b.stride(1), ldsa, ldsb, G, b_preshuffled, False, K % (WMMA_K * cfg.kpair) != 0)


def grouped_gemm_mxfp8_variable_k_gfx1250(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    group_offs: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    out: torch.Tensor | None = None,
    accumulate: bool = False,
    config: MXGemmConfig | None = None,
    b_preshuffled: bool = True,
) -> torch.Tensor:
    """Routed experts wgrad: ``out[g] (+)= a[:, T_g] @ b[:, T_g]^T`` over expert g's tokens
    ``T_g = [group_offs[g], group_offs[g + 1])`` (int64 [G + 1], multiples of 128: the column copies' layout).
    ``a [M, T]`` (dY^T), ``b [N, T]`` (X^T, 16x16-preshuffled by default), scales packed ``[M/32, T/128, 32, 4]`` /
    ``[N/32, ..]`` (or raw); ``out [G, M, N]`` bf16 or fp32, ``accumulate``: ``out[g] = round(out[g] + result)``
    (the fused main_grad update; an empty expert leaves its slice unchanged, or zero without ``accumulate``)."""
    _require_gfx1250(a.device)
    who = "grouped_gemm_mxfp8_variable_k_gfx1250"
    (M, T), N = a.shape, b.shape[0]
    G = group_offs.numel() - 1
    if b.shape[1] != T or group_offs.dtype != torch.int64 or T % 128:
        raise ValueError(f"{who}: a {tuple(a.shape)}, b {tuple(b.shape)}, token count a multiple of 128, int64 offsets")
    out = _out(out, (G, M, N), out_dtype, accumulate, a.device, who)
    if out.stride(0) != M * out.stride(1):
        raise ValueError(f"{who}: out must be [G, M, ldc] with expert stride M * ldc")
    if M == 0 or N == 0 or G == 0:
        return out
    cfg = config or pick_grouped_config(VARIABLE_K, M, N, T // max(G, 1), b_preshuffled)
    packed, ldsa, ldsb = _check_scales(a_scale, b_scale, M, N, T, who)
    return _launch(who, VARIABLE_K, cfg, out, a, b, a_scale, b_scale, group_offs, None, M, N, T, a.stride(0),
                   b.stride(0), ldsa, ldsb, G, b_preshuffled, accumulate, True)
