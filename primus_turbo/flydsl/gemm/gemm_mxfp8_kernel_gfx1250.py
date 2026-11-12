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
``out += result`` (beta = 1, rounded once). Entry points (signatures stable, positional order as listed):

* ``gemm_mxfp8_gfx1250(a, a_scale, b, b_scale, out_dtype=bf16, out=None, config=None, b_preshuffled=False,
  accumulate=False)``: dense ``out[M, N] (+)= (a * sa)[M, K] @ (b * sb)[N, K]^T``;
* ``grouped_gemm_mxfp8_gfx1250(a, a_scale, b, b_scale, group_offs, group_offs_padded, m_out, out_dtype=bf16,
  out=None, config=None, b_preshuffled=True)``: routed experts fwd / dgrad, ``out[offs[g] + i] = a[offs_pad[g] + i]
  @ b[g]^T`` (``a``'s rows of expert g padded to 32 from ``offs_pad``, out tight under ``offs``, int64 [G + 1]
  device tables, no host sync);
* ``grouped_gemm_mxfp8_variable_k_gfx1250(a, a_scale, b, b_scale, group_offs, out_dtype=bf16, out=None,
  accumulate=False, config=None, b_preshuffled=True)``: routed experts wgrad, ``out[g] (+)= a[:, T_g] @ b[:, T_g]^T``
  over expert g's tokens ``T_g`` (``offs``, multiples of 128: the column copies), ``out [G, M, N]`` (e.g. main_grad).

Operands: E4M3, K contiguous, 16-byte rows. Scales: one E8M0 byte per 32 K, packed ``[ceil(R/32)][ceil(K/128)][32][4]``
(fast) or raw ``[rows, ceil(K/32)]`` (20-45% slower); both scales of a call in the same layout. B: 16x16-preshuffled
``[N/16][K/16][16][16]`` (what the quantizers emit) or row-major. K tails: a partial last K step reads the last row of
an operand up to a step past its data (a TDM load's inner extent clamps the data, not the addressing): the quantizers
allocate >= 2 KiB of storage slack after a preshuffled B (and raw scales); without it the entry points use kpair 1
(128-aligned K has no partial step) and refuse K % 128 != 0. The module switches ``C_EARLY_ON`` (beta = 1 into bf16:
fetch the old C during the last K step) and ``PERSIST_ON`` / ``XCD_REMAP_ON`` / ``STAGGER_ON`` (None: as the config's
``persist`` / ``xcd_remap`` / ``stagger`` fields say) select kernel variants for A/B runs; read at the call.

The mainloop is aiter's compute-bound gfx1250 tile: 2x2 waves of up to 128x128, ``v_wmma_scale_f32_16x16x128_f8f6f4``
with one scale VGPR feeding two WMMAs (ROW0 / ROW1 lane halves), ``num_buffers`` LDS slots of ``kpair`` 128-wide K
steps filled by per-wave TDM descriptors (wave 0: A, 1: B, 2: A scales, 3: B scales), a split signal / wait barrier
per slot, cluster multicast of the shared tiles. Changes: row-major A (and B when not preshuffled) in padded LDS rows,
mx32 scales, M / N / K tails by TDM out-of-bounds zero fill (no padding of the data; a half last K step loads half
rows), an LDS-staged epilogue with 16-byte buffer stores (no TDM store: the suspect of the grouped kernel's rare silent
error). No branch skips the TDM pipeline inside a tile; a grouped-M tile past the experts' ends its workgroup at once
(whole clusters: a cluster never mixes experts).
"""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.arith import _to_raw as _raw
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
NUM_CUS = 256  # MI455X CUs (the persistent grid uses the device's count; this where there is no device)
STAGGER_SLEEP = 48  # stagger: s_sleep units (64 clocks) per phase step (~1.4 us)
CLUSTER_SYNC_REVS = 8  # a cluster barrier every this many pipeline revolutions keeps multicast partners close
_WG_BARRIER = -1
_E8M0 = (torch.uint8, getattr(torch, "float8_e8m0fnu", torch.uint8))


def geometry(c: MXGemmConfig, stage_f32: bool = False, slot_major: bool = False):
    """LDS plan of a config: (row bytes, slot step of A / B / SA / SB, base offsets of the four regions, arena,
    A + B slot pitch); ``stage_f32``: the epilogue stages fp32 (fp32 out or beta = 1 without ``c_early``) instead of
    bf16. ``slot_major``: slot j of A and B is one contiguous ``[B_j | A_j]`` block (the early old-C load reuses a
    whole free slot), else aiter's order (all B slots, then all A slots)."""
    super_k = WMMA_K * c.kpair
    row = super_k + LDS_PAD
    sc = (lambda r: r // 32 * super_k) if c.scale_4d else (lambda r: r * 4 * c.kpair)
    st = (c.tile_m * row, c.tile_n * (super_k if c.b_preshuffle else row), sc(c.tile_m), sc(c.tile_n))
    nb = c.num_buffers
    pitch = st[0] + st[1]
    if slot_major:
        base_b, base_a = 0, st[1]
        step = (pitch, pitch, st[2], st[3])
    else:
        base_b = 0  # B first, as aiter (64 KiB aligned base)
        base_a = base_b + nb * st[1]
        step = st
    base_sa = nb * pitch
    base_sb = base_sa + nb * st[2]
    arena = max(base_sb + nb * st[3], c.tile_m * (c.tile_n * (4 if stage_f32 else 2) + C_PAD_B))
    return row, step, (base_a, base_b, base_sa, base_sb), arena, pitch


def c_early_fits(c: MXGemmConfig) -> bool:
    """beta = 1 into a bf16 C can fetch the old C tile (rows padded like the staging) during the last K step into the
    A + B slot that step does not read: kpair 2 (one free slot of a whole tile). With a cluster, a cluster barrier
    first: every partner is past the previous step, so none of its multicasts into that slot is still in flight."""
    return c.kpair == 2 and c.tile_m * (c.tile_n * 2 + C_PAD_B) <= geometry(c, slot_major=True)[4]


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
    c_early: Constexpr[bool],
    persist: Constexpr[bool],
    xcd_remap: Constexpr[bool],
    stagger: Constexpr[int],
    n_cus: Constexpr[int],
):
    """``arg_g0`` / ``arg_g1``: grouped-M: the tight and the 32-padded row offsets of the experts (int64, G + 1);
    variable-K: the 128-padded token offsets (``arg_g0``). Unused in dense mode. ``c_early`` (beta = 1, bf16 out,
    ``c_early_fits``, whole N tiles): the old C tile comes into the free A + B slot during the last K step."""
    cfg = MXGemmConfig(tile_m, tile_n, num_buffers, kpair, cluster_m, cluster_n, scale_4d, b_shuf)
    assert not c_early or (beta1 and not out_f32 and c_early_fits(cfg))
    # persistent: the next tile's slots load during this one's epilogue, which stages bf16 in one slot
    assert not persist or (num_buffers == 2 and c_early_fits(cfg) and not out_f32 and (c_early or not beta1))
    STG = 4 if out_f32 or (beta1 and not c_early) else 2  # bytes per staged output element
    OB = 4 if out_f32 else 2  # bytes per output element
    LDS_ROW, STAGE, BASE, ARENA, PITCH = geometry(cfg, STG == 4, c_early or persist)
    G = n_groups
    assert tile_m in (128, 256) and tile_n in (128, 256) and kpair in (1, 2) and num_buffers in (2, 3, 4)
    assert ARENA <= LDS_BYTES, f"{cfg} needs {ARENA} B of LDS"
    use_cluster = cluster_m * cluster_n > 1
    NB, KPAIR = num_buffers, kpair
    SUPER_K = WMMA_K * KPAIR
    UNROLL = KPAIR * NB
    HALF_TAIL = KPAIR == 2 and k_tail  # a last K step of <= 128 loads half a slot and skips its second WMMA step
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
        ("_kt" if k_tail else "") + ("_f32" if out_f32 else "") + ("_acc" if beta1 else "") + ("_ce" if c_early else "") + \
        ("_p" if persist else "") + ("_x" if xcd_remap else "") + (f"_s{stagger}" if stagger else "")

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

        vk_tab = None
        if const_expr(mode == VARIABLE_K and G <= 32):
            # the experts' token offsets, lane e holding offs[e] / offs[e + 1] (read once; a tile then needs no load)
            rs_vk = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(arg_g0)), num_records_bytes=8 * (G + 1))
            vk_tab = (fx.Int32(buffer_load(rs_vk, lane * 2, vec_width=1, dtype=T.i32)),
                      fx.Int32(buffer_load(rs_vk, lane * 2 + 2, vec_width=1, dtype=T.i32)))

        def _decode_vk(bx, by, e):
            """variable-K: output tile (bx, by) of expert e over its tokens [offs[e], offs[e + 1])."""
            blk_m, blk_n = bx * tile_m, by * tile_n
            if const_expr(vk_tab is not None):
                k0 = fx.Int32(rocdl.readlane(T.i32, vk_tab[0], e))
                klen = fx.Int32(rocdl.readlane(T.i32, vk_tab[1], e)) - k0
            else:
                rsrc = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(arg_g0)), num_records_bytes=8 * (G + 1))
                o2 = Vec(buffer_load(rsrc, e * 2, vec_width=4, is_scalar=True))
                k0, klen = fx.Int32(o2[0]), fx.Int32(o2[2]) - fx.Int32(o2[0])
            supers = (klen + (SUPER_K - 1)) // SUPER_K
            supers = (supers > zero).select(supers, fx.Int32(1))  # an empty expert: one zero K step
            # (a_row0, out_row0, m_valid, blk_n, n_valid, expert, k0, klen, supers)
            return blk_m, blk_m, i32_m - blk_m, blk_n, i32_n - blk_n, e, k0, klen, supers

        def _gm_table():
            """grouped-M, once per workgroup: lane l of chunk c holds expert 32 c + l's first m tile (a DPP prefix sum
            of the tile counts ceil(len_g / tile_m)), tile count, rows and row offsets; and the number of m tiles."""
            rs0 = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(arg_g0)), num_records_bytes=8 * (G + 1))
            rs1 = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(arg_g1)), num_records_bytes=8 * (G + 1))

            def _scan(v):  # inclusive prefix over the wave: row_shr within 16-lane rows, then row 0's total
                for k in (1, 2, 4, 8):
                    v = v + fx.Int32(_rocdl.update_dpp(T.i32, _raw(zero), _raw(v), 0x110 + k, 0xF, 0xF, True))
                return v + (lane >= 16).select(fx.Int32(rocdl.readlane(T.i32, v, 15)), zero)

            acc, chunks = zero, []
            for ch in range_constexpr((G + 31) // 32):
                g = lane + ch * 32
                o0 = fx.Int32(buffer_load(rs0, g * 2, vec_width=1, dtype=T.i32))  # low words (int64 tables)
                ln = (g < G).select(fx.Int32(buffer_load(rs0, g * 2 + 2, vec_width=1, dtype=T.i32)) - o0, zero)
                p0 = fx.Int32(buffer_load(rs1, g * 2, vec_width=1, dtype=T.i32))
                n_g = (ln + (tile_m - 1)) // tile_m
                incl = _scan(n_g)
                chunks.append((acc + incl - n_g, n_g, ln, p0, o0))
                acc = acc + fx.Int32(rocdl.readlane(T.i32, incl, 31))
            return chunks, acc

        def _decode_gm(tab, t, blk_n):
            """grouped-M: m tile t of the experts' tiles in order; a tile past them is dead (no rows, no B). A ballot
            of the expert whose tiles hold t, its entries by readlane (no load)."""
            chunks, acc = tab
            first, len_g, in0, out0, expert = zero, zero, zero, zero, zero
            for ch, (excl, n_g, ln, p0, o0) in enumerate(chunks):
                hit = fx.Int32(rocdl.ballot(T.i32, (t >= excl) & (t < excl + n_g)))
                src = fx.Int32(_llvm.call_intrinsic(T.i32, "llvm.cttz.i32", [_raw(hit), _raw(fx.Boolean(False))],
                                                    [], []))
                found = hit != zero
                expert = found.select(src + ch * 32, expert)
                first = found.select(fx.Int32(rocdl.readlane(T.i32, excl, src)), first)
                len_g = found.select(fx.Int32(rocdl.readlane(T.i32, ln, src)), len_g)
                in0 = found.select(fx.Int32(rocdl.readlane(T.i32, p0, src)), in0)
                out0 = found.select(fx.Int32(rocdl.readlane(T.i32, o0, src)), out0)
            local = (t - first) * tile_m
            rows = len_g - local
            m_valid = (rows < tile_m).select(rows, fx.Int32(tile_m))
            m_valid = (t < acc).select(m_valid, zero)
            n_valid = (t < acc).select(i32_n - blk_n, zero)
            supers = (t < acc).select((i32_k + (SUPER_K - 1)) // SUPER_K, fx.Int32(1))
            return in0 + local, out0 + local, m_valid, blk_n, n_valid, expert, zero, i32_k, supers

        if const_expr(mode == DENSE and not persist):
            blk_n = fx.Int32(bid_y) * tile_n
            blk_m = (fx.Int32(bid_z) * fx.Int32(fx.grid_dim.x) + fx.Int32(bid_x)) * tile_m
            supers = (i32_k + (SUPER_K - 1)) // SUPER_K
            # m_valid <= 0 for a workgroup past the edge
            dec = (blk_m, blk_m, i32_m - blk_m, blk_n, i32_n - blk_n, zero, zero, i32_k, supers)
        elif const_expr(mode == GROUPED_M and not persist):
            t_x = fx.Int32(bid_x)
            if const_expr(xcd_remap):
                # workgroups go round-robin over the XCDs, so x and x + 8 share one: give an XCD consecutive m tiles
                # (an expert's B tiles then stay in fewer L2s); grid.x is a multiple of 8 here
                gdx = fx.Int32(fx.grid_dim.x)
                t_x = (gdx % 8 == 0).select((t_x % 8) * (gdx // 8) + t_x // 8, t_x)
            t = fx.Int32(bid_z) * fx.Int32(fx.grid_dim.x) + t_x
            gm_tab = _gm_table()
            acc = gm_tab[1]
            dec = _decode_gm(gm_tab, t, fx.Int32(bid_y) * tile_n)
            # a tile past the experts' (the grid is an upper bound) ends its workgroup here: uniform, and a cluster
            # (along N) shares the m tile, so whole clusters end
            _llvm.inline_asm(None, [as_ir_value(t), as_ir_value(acc)],
                             "s_cmp_lt_u32 $0, $1\n\ts_cbranch_scc1 1f\n\ts_endpgm\n\t1:", "s,s,~{scc},~{memory}",
                             has_side_effects=True)
        elif const_expr(mode == VARIABLE_K and not persist):
            dec = _decode_vk(fx.Int32(bid_x), fx.Int32(bid_y), fx.Int32(bid_z))
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

        # ---- per-wave TDM descriptor of a tile (wave 0: A, 1: B, 2: A scales, 3: B scales) ----
        def _owner(w, dec):
            """[addr lo, addr hi | type, LDS slot 0, slot step, K step, inner extent, GROUP1 words 0, 2 (dim1
            half), 3, 4, 5, and words 0, 3 of a half K step] of operand ``w`` of the tile ``dec``; the K step and
            inner extent are in the operand's bytes."""
            a_row0, _, m_valid, blk_n, n_valid, expert, k0, klen, _ = dec
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
            # a half K step (HALF_TAIL): half the row, the rest of the LDS row skipped as padding, so nothing past
            # the data is addressed (the inner extent of a TDM load clamps the data, not the addressing). The pad
            # field holds at most 512 B, the interval 1 KiB: not a preshuffled B (2 KiB halves), nor raw scales (4 B)
            halvable = w == 0 or (w == 1 and not b_shuf) or (w >= 2 and scale_4d)
            g1h = _tdm_group1(mask, inner // 2, rows, total, ext1, ld, pad + inner // 2) if halvable else g1
            return [fx.Int32(addr), fx.Int32(addr >> 32) | fx.Int32(-(1 << 31)), lds0 + BASE[w], fx.Int32(STAGE[w]),
                    fx.Int32(step), total, g1[0], g1[2] & fx.Int32(-65536), g1[3], g1[4], g1[5], g1h[0], g1h[3]]

        def _desc(dec):
            d = [zero] * 13
            if wave == 0:
                d = _owner(0, dec)
            elif wave == 1:
                d = _owner(1, dec)
            elif wave == 2:
                d = _owner(2, dec)
            else:
                d = _owner(3, dec)
            return d

        def _make_prepare(d, supers, rot):
            """The load of K offset ``delta`` (this wave's units, clamped to the last K step) into logical slot
            ``slot`` of a tile (descriptor words ``d``); slot j is physical slot j ^ ``rot`` (persistent tiles).
            c_early: loads past the last K step (re-loads of it) go into the last step's own slot (the same bytes),
            never into the slot the old C is fetched into."""
            t_lo, t_hi, t_lds, t_lds_step, t_step, t_total, t_g10, t_g12_hi, t_g13, t_g14, t_g15, t_g10h, t_g13h = d
            last_delta = (supers - 1) * t_step
            last_slot = (supers - 1) % NB

            def _prepare(slot, delta):
                past = delta > last_delta
                delta = past.select(last_delta, delta)
                if const_expr(c_early or persist):
                    slot = past.select(last_slot, fx.Int32(slot))
                if const_expr(persist):
                    slot = fx.Int32(slot) ^ rot
                lo, hi = tdm_ops.add_addr_with_carry(t_lo, t_hi, delta)
                g0 = Vec.from_elements([fx.Int32(1), t_lds + t_lds_step * slot, fx.Int32(lo), fx.Int32(hi)], fx.Int32)
                d0 = t_total - delta if const_expr(k_tail) else t_total
                d0 = (d0 > zero).select(d0, zero)
                w0, w3 = t_g10, t_g13
                if const_expr(HALF_TAIL):
                    half = d0 <= (t_step >> 1)
                    w0, w3 = half.select(t_g10h, t_g10), half.select(t_g13h, t_g13)
                g1 = Vec.from_elements(
                    [w0, (d0 & 0xFFFF) << 16, ((d0 >> 16) & 0xFFFF) | t_g12_hi, w3, t_g14, t_g15, zero, zero],
                    fx.Int32,
                )
                return tdm_ops.TDMDescriptor2D(as_ir_value(g0), as_ir_value(g1))

            return _prepare, t_step, last_slot

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

        def _stage_addrs(rot):
            """Per unrolled stage: the fragment / scale base addresses of A, B, SA, SB (physical slot = slot ^ rot)."""
            addrs = ([], [], [], [])
            for st in range_constexpr(UNROLL):
                slot, par = st // KPAIR, st % KPAIR
                for w, lb, po in ((0, a_byte, WMMA_K), (1, b_byte, GEO_B[2]), (2, sa_byte, SC_PAR),
                                  (3, sb_byte, SC_PAR)):
                    if const_expr(persist):
                        addrs[w].append(_planar(BASE[w] + par * po, lb + (fx.Int32(slot) ^ rot) * STAGE[w]))
                    else:
                        addrs[w].append(_planar(BASE[w] + slot * STAGE[w] + par * po, lb))
            return addrs

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

        def _mma(wm, wn, act, wt, sa_k, sb_k):
            # WMMA "A" = the B fragment (N rows), "B" = the A fragment: each lane's 8 results are 8 consecutive N
            idx = wm * n_rep + wn
            fx.gemm(wmma[wn % 2][wm % 2], c_frags[idx], wt, act, c_frags[idx],
                    scale_a=sb_k[wn // 2], scale_b=sa_k[wm // 2])

        seed_a = [fx.make_rmem_tensor(16, fx.Int32) for _ in range_constexpr(half_m)]
        seed_b = [fx.make_rmem_tensor(16, fx.Int32) for _ in range_constexpr(half_n)]
        seed_sa = [fx.make_rmem_tensor(1, fx.Int32) for _ in range_constexpr(N_SA)]
        seed_sb = [fx.make_rmem_tensor(1, fx.Int32) for _ in range_constexpr(N_SB)]

        def _chain(d, pos, t):
            d[pos] = t if pos not in d else (lambda p=d[pos], q=t: (p(), q()))

        # epilogue store geometry: lane tid stores 16-byte chunk tid % CH of rows tid // CH + i * RPI
        EPC = 16 // OB  # output elements per chunk
        CH = tile_n // EPC
        RPI = BLOCK // CH
        c16, r0 = tid % CH, tid // CH
        GRP = 65536 // (RPI * C_ROW_B)  # iterations per LDS base: displacements stay 16-bit DS offsets
        wave_parity = fx.Int32(
            _llvm.inline_asm(T.i32, [as_ir_value(rocdl.wave_id())], "s_and_b32 $0, $1, 1", "=s,s,~{scc}",
                             has_side_effects=True)
        )

        def _tile(dec, rot, nxt=None):
            """One output tile ``dec``: mainloop + epilogue. Persistent (``nxt`` = (next tile, whether there is
            one)): this tile's first two slots were loaded already (physical slot = logical ^ ``rot``), and the
            epilogue loads the next tile's while it stores; returns the next tile's ``rot``."""
            a_row0, out_row0, m_valid, blk_n, n_valid, expert, k0, klen, supers = dec
            _prepare, t_step, last_slot = _make_prepare(_desc(dec), supers, rot)
            addr_a, addr_b, addr_sa, addr_sb = _stage_addrs(rot)
            for cf in c_frags:
                cf.store(Vec.filled(8, 0.0, fx.Float32))

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
                b_th = [(lambda i=i: seed_b[i].store(_load_frag(addr_b[stage], i, GEO_B)))
                        for i in range_constexpr(half_n)]
                return (head + a_th, b_th) if parity == 0 else (head + b_th, a_th)

            def _stage(stage, next_stage, future_slot, future_delta, fence_out, has_next, boundary, parity=0,
                       more=None):
                """One 128-wide K step: four quadrants of the wave tile, at most three fragment groups live; the
                second-half fragments of this step and the first half of the next are read in between the WMMAs.
                ``more`` (drain): whether a later step runs, else its slot fence and barrier are skipped."""
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
                    if const_expr(boundary and has_next and more is None):
                        tdm_ops.tensor_wait(fence_out)
                        rocdl.s_barrier_signal(_WG_BARRIER)
                    elif const_expr(boundary and has_next):
                        if more:
                            tdm_ops.tensor_wait(fence_out)
                            rocdl.s_barrier_signal(_WG_BARRIER)
                    rocdl.sched_barrier(0)

                def _wait_refill():
                    if const_expr(boundary and has_next and more is None):
                        # LLVM hoists the wait above the slack WMMAs; pinning them in between measured +0.2% (noise)
                        rocdl.s_barrier_wait(_WG_BARRIER)
                        if const_expr(refill):
                            tdm_ops.tensor_load_2d(prepared)
                    elif const_expr(boundary and has_next):
                        if more:
                            rocdl.s_barrier_wait(_WG_BARRIER)
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

            # the output tile: rows / columns past M / N dropped by the stores (and not read by the old-C load)
            m_ext = (m_valid < tile_m).select(m_valid, fx.Int32(tile_m))
            m_ext = (m_ext > zero).select(m_ext, zero)
            n_ext = (n_valid < tile_n).select(n_valid, fx.Int32(tile_n))
            c_addr = fx.Int64(fx.ptrtoint(arg_c)) + (fx.Int64(out_row0) * fx.Int64(i32_ldc) + fx.Int64(blk_n)) * OB
            if const_expr(mode == VARIABLE_K):  # out [G, M, ldc]
                c_addr = c_addr + fx.Int64(expert) * fx.Int64(i32_m) * fx.Int64(i32_ldc) * OB
            free_slot = (last_slot + (NB - 1)) % NB  # c_early: the previous K step's slot (logical)
            if const_expr(persist):
                free_slot = free_slot ^ rot

            # ---- prologue: fill every slot (past the last K step: reload it), wait for slot 0, read its seeds ----
            if const_expr(not persist):  # (persistent: loaded by the previous tile's epilogue)
                if const_expr(use_cluster):
                    cluster.cluster_barrier()
                for i in range_constexpr(NB):
                    tdm_ops.tensor_load_2d(_prepare(i, fx.Int32(i) * t_step))
            tdm_ops.tensor_wait(NB - 1)
            gpu.barrier()
            for group in _seed_thunks(0):
                for thunk in group:
                    thunk()

            n_full = (supers + (NB - 1)) // NB - 1
            drain = supers - n_full * NB  # 1..NB slots left after the steady revolutions
            slot_delta = [fx.Int32(c + NB) * t_step for c in range_constexpr(NB)]
            n_drain = drain * KPAIR  # K steps of the drain (a half last step skips its second one)
            if const_expr(HALF_TAIL):
                n_drain = n_drain - (klen - (supers - 1) * SUPER_K <= SUPER_K // 2).select(fx.Int32(1), zero)

            def _run_steady(parity):
                for rev in range(n_full):
                    rev_delta = rev * NB * t_step
                    for g in range_constexpr(UNROLL):
                        delta = rev_delta + slot_delta[g // KPAIR]
                        _stage(g, (g + 1) % UNROLL, g // KPAIR, delta, NB - 2, True, g % KPAIR == KPAIR - 1, parity)
                    if const_expr(use_cluster):
                        if (rev % CLUSTER_SYNC_REVS) == (CLUSTER_SYNC_REVS - 1):
                            cluster.cluster_barrier()

            def _c_load():
                """c_early: the old C tile (bf16 rows padded to C_ROW_B) into the free slot (the previous K
                step's, never refilled: see ``_prepare``); waited for by the epilogue's tensor_wait(0)."""
                g0 = Vec.from_elements([fx.Int32(1), lds0 + free_slot * PITCH, fx.Int32(c_addr),
                                        fx.Int32(c_addr >> 32) | fx.Int32(-(1 << 31))], fx.Int32)
                g1 = Vec.from_elements(_tdm_group1(0, tile_n * 2, tile_m, fx.Int32(tile_n * 2), m_ext, i32_ldc * 2,
                                                   C_PAD_B), fx.Int32)
                tdm_ops.tensor_load_2d(tdm_ops.TDMDescriptor2D(as_ir_value(g0), as_ir_value(g1)))

            rot_n = rot
            if const_expr(persist):
                # the next tile: its first slot goes into the slot the epilogue does not stage in (c_early: the last
                # K step's, after the mainloop; else the free one, at the last K step's start), its second one into
                # the staging slot once the stores have read it
                nd, more_tiles = nxt
                last_phys = last_slot ^ rot
                rot_n = last_phys if const_expr(c_early) else last_phys ^ 1
                prep_n, t_step_n, _ = _make_prepare(_desc(nd), nd[8], rot_n)

            # odd waves run the quadrants B-first: the two wave pairs read different fragment halves at a time
            if wave_parity == 0:
                _run_steady(0)
            else:
                _run_steady(1)
            rocdl.s_wait_dscnt(0)
            rocdl.sched_barrier(0)
            for g in range_constexpr(UNROLL):
                if g < n_drain:
                    if const_expr((c_early or persist) and g % KPAIR == 0):
                        if fx.Int32(g) == (drain - 1) * KPAIR:  # the last K step begins: the free slot is free
                            if const_expr(use_cluster):
                                cluster.cluster_barrier()  # (in every partner: loads multicast into it)
                            if const_expr(c_early):
                                if wave == 3:
                                    _c_load()
                            else:
                                if more_tiles:
                                    tdm_ops.tensor_load_2d(prep_n(0, zero))
                    _stage(g, (g + 1) % UNROLL, g // KPAIR, None, NB - 2 - g // KPAIR, g < UNROLL - 1,
                           g % KPAIR == KPAIR - 1,
                           more=(fx.Int32(g) < n_drain - 1) if const_expr(c_early or persist) else None)

            # ---- epilogue: accumulators -> rows in LDS (bf16, or fp32 for fp32 out / beta = 1 without c_early)
            # -> 16-byte buffer stores; beta = 1: + the old C (c_early: from the free slot, added in registers and
            # staged in place; else read at the store's offset), rounded once; rows / columns past M / N dropped
            accs = [c_frags[i].load() for i in range_constexpr(n_acc)]
            rocdl.s_wait_dscnt(0)
            tdm_ops.tensor_wait(0)
            if const_expr(use_cluster):
                cluster.cluster_barrier()  # no multicast into this LDS is still in flight
            else:
                gpu.barrier()
            stg_off = zero
            if const_expr(c_early):  # staged in place over the old C
                stg_off = free_slot * PITCH
            if const_expr(persist):
                if const_expr(c_early):
                    if more_tiles:
                        tdm_ops.tensor_load_2d(prep_n(0, zero))
                else:
                    stg_off = last_phys * PITCH
            for wm in range_constexpr(m_rep):
                for wn in range_constexpr(n_rep):
                    h = accs[wm * n_rep + wn]
                    row, col = wmb + wm * 16 + lane16, wnb + wn * 16 + kgrp * 8
                    if const_expr(c_early):
                        old = Vec(lds_load_b128(lds0 + stg_off + (wmb + lane16) * C_ROW_B + (wnb + kgrp * 8) * 2,
                                                wm * 16 * C_ROW_B + wn * 32))
                        h = h + old.bitcast(fx.BFloat16).to(fx.Float32)
                    h = h if const_expr(STG == 4) else h.to(fx.BFloat16)
                    fx.ptr_store(h.bitcast(fx.Int8), base_ptr + stg_off + row * C_ROW_B + col * STG)
            gpu.barrier()
            ldc_b = i32_ldc * OB
            c_res = create_buffer_resource_from_addr(c_addr, num_records_bytes=m_ext * ldc_b)
            # one column per lane, so the column mask is folded into the lane's base (past num_records, and every
            # later row of it too: dropped)
            lane_b = (c16 * EPC < n_ext).select(r0 * ldc_b + c16 * 16, m_ext * ldc_b)
            step_b = RPI * ldc_b
            for i in range_constexpr(tile_m // RPI):
                if const_expr(i % GRP == 0):
                    rd = _planar(i * RPI * C_ROW_B, stg_off + r0 * C_ROW_B + c16 * EPC * STG)
                off = lane_b + step_b * i
                if const_expr(STG == 2):
                    v = lds_load_b128(rd, (i % GRP) * RPI * C_ROW_B)
                else:
                    acc_v = [Vec(lds_load_b128(rd, (i % GRP) * RPI * C_ROW_B + 16 * j))
                             for j in range_constexpr(STG // OB)]
                    acc_v = acc_v[0].shuffle(acc_v[1], list(range(8))) if const_expr(OB == 2) else acc_v[0]
                    acc_v = acc_v.bitcast(fx.Float32)
                    if const_expr(beta1):
                        old = Vec(buffer_load(c_res, off // 4, vec_width=4, dtype=fx.Int32))
                        acc_v = acc_v + (old.bitcast(fx.BFloat16).to(fx.Float32) if const_expr(OB == 2)
                                         else old.bitcast(fx.Float32))
                    v = (acc_v.to(fx.BFloat16) if const_expr(OB == 2) else acc_v).bitcast(fx.Int32)
                buffer_store(v, c_res, off, offset_is_bytes=True)
            if const_expr(persist):
                rocdl.s_wait_dscnt(0)
                # the staging slot is read (in every workgroup of the cluster: the loads multicast into it)
                if const_expr(use_cluster):
                    cluster.cluster_barrier()
                else:
                    gpu.barrier()
                if more_tiles:
                    tdm_ops.tensor_load_2d(prep_n(1, t_step_n))
            return rot_n

        if const_expr(persist):
            # persistent: cluster c (a workgroup without one) runs cluster tiles c, c + NC, ...; its workgroup
            # (lx, ly) takes tile (xg * cluster_m + lx, yg * cluster_n + ly) of cluster tile (xg, yg)
            c_id, n_cl = fx.Int32(bid_x) // cluster_m, fx.Int32(fx.grid_dim.x) // cluster_m
            lxp, lyp = fx.Int32(bid_x) % cluster_m, fx.Int32(bid_y)
            mtg = (i32_m + (tile_m - 1)) // tile_m // cluster_m  # (the entry's cluster divides the tile counts)
            ntg = (i32_n + (tile_n - 1)) // tile_n // cluster_n
            if const_expr(mode == VARIABLE_K):  # m tile fastest, then n tile, then expert
                n_tiles = mtg * ntg * G

                def _decode_lin(tau):
                    xg, yg = tau % mtg, (tau // mtg) % ntg
                    return _decode_vk(xg * cluster_m + lxp, yg * cluster_n + lyp, tau // (mtg * ntg))
            elif const_expr(mode == DENSE):  # chunks of 32 m tiles x all n groups (the grid's order), the last short
                n_tiles = mtg * ntg
                RX = max(1, 32 // cluster_m)
                full = mtg // RX * RX
                rem = mtg - full
                rem = (rem > zero).select(rem, fx.Int32(1))
                supers_d = (i32_k + (SUPER_K - 1)) // SUPER_K

                def _decode_lin(tau):
                    head = tau < full * ntg
                    tail = tau - full * ntg
                    xg = head.select(tau // (RX * ntg) * RX + tau % RX, full + tail % rem)
                    yg = head.select((tau // RX) % ntg, tail // rem)
                    blk_m, blk_n = (xg * cluster_m + lxp) * tile_m, (yg * cluster_n + lyp) * tile_n
                    return blk_m, blk_m, i32_m - blk_m, blk_n, i32_n - blk_n, zero, zero, i32_k, supers_d
            else:  # GROUPED_M (cluster_m == 1): chunks of 32 m tiles x all n groups (the grid's order), the last short
                gm_tab = _gm_table()
                n_m = gm_tab[1]
                n_tiles = n_m * ntg
                full = n_m // 32 * 32
                rem = n_m - full
                rem = (rem > zero).select(rem, fx.Int32(1))

                def _decode_lin(tau):
                    head = tau < full * ntg
                    tail = tau - full * ntg
                    t = head.select(tau // (32 * ntg) * 32 + tau % 32, full + tail % rem)
                    yg = head.select((tau // 32) % ntg, tail // rem)
                    return _decode_gm(gm_tab, t, (yg * cluster_n + lyp) * tile_n)

            n_iter = (n_tiles - c_id + (n_cl - 1)) // n_cl
            # a cluster with no tile ends here (uniform over it)
            _llvm.inline_asm(None, [as_ir_value(c_id), as_ir_value(n_tiles)],
                             "s_cmp_lt_u32 $0, $1\n\ts_cbranch_scc1 1f\n\ts_endpgm\n\t1:", "s,s,~{scc},~{memory}",
                             has_side_effects=True)
            if const_expr(stagger):
                # start the clusters in ``stagger`` phases a fraction of a tile apart, so their epilogues (old-C reads,
                # stores: the memory-bound end of every tile) do not all fall into the same microseconds
                for _ in range(c_id % stagger):
                    rocdl.s_sleep(STAGGER_SLEEP)
            if const_expr(use_cluster):
                cluster.cluster_barrier()  # every partner resident before the first multicast
            d0 = _decode_lin(c_id)
            prep0, t_step0, _ = _make_prepare(_desc(d0), d0[8], zero)
            for i in range_constexpr(NB):
                tdm_ops.tensor_load_2d(prep0(i, fx.Int32(i) * t_step0))
            rot = zero
            for it in range(n_iter):
                tau = c_id + it * n_cl
                more_tiles = it + 1 < n_iter
                nxt_tau = more_tiles.select(tau + n_cl, tau)
                rot = _tile(_decode_lin(tau), rot, (_decode_lin(nxt_tau), more_tiles))
        else:
            if const_expr(stagger):
                # the first wave of workgroups starts in ``stagger`` phases (by cluster), and the waves after it
                # inherit the offsets: the memory-bound epilogues of the CUs stop falling into the same microseconds
                gdx, gdy = fx.Int32(fx.grid_dim.x), fx.Int32(fx.grid_dim.y)
                lin = fx.Int32(bid_x) + gdx * (fx.Int32(bid_y) + gdy * fx.Int32(bid_z))
                phase = (lin < NUM_CUS).select((fx.Int32(bid_x) // cluster_m) % stagger, zero)
                for _ in range(phase):
                    rocdl.s_sleep(STAGGER_SLEEP)
            _tile(dec, 0)

    # Launch: grid.x runs up to 32 M tiles (a power of two dividing them), grid.y the N tiles, grid.z the M chunks
    # (grouped-M: the m tiles of all experts, at most ceil(M / tile_m) + G; variable-K: grid.z = the expert;
    # persistent: NUM_CUS workgroups).
    gx = (i32_m + (tile_m - 1)) // tile_m + (G if mode == GROUPED_M else 0)
    gx = (gx + (cluster_m - 1)) // cluster_m * cluster_m
    gy = ((i32_n + (tile_n - 1)) // tile_n + (cluster_n - 1)) // cluster_n * cluster_n
    pow2 = gx & -gx
    capped = (pow2 < 32).select(pow2, fx.Int32(32))
    m_run = ((gx > 32) & (pow2 >= 8) & ((capped % cluster_m) == 0)).select(capped, gx)
    grid = (m_run, gy, gx // m_run)
    if const_expr(mode == VARIABLE_K):
        grid = (gx, gy, G)
    if const_expr(persist):  # n_cus workgroups in clusters along x; variable-K: at most one cluster per tile
        n_cl = n_cus // (cluster_m * cluster_n)
        if const_expr(mode != GROUPED_M):
            n_ct = gx // cluster_m * (gy // cluster_n) * (G if mode == VARIABLE_K else 1)
            n_cl = (n_ct < n_cl).select(n_ct, fx.Int32(n_cl))
        grid = (n_cl * cluster_m, cluster_n, 1)
    attrs = {"rocdl.cluster_dims": f"{cluster_m},{cluster_n},1"} if use_cluster else {}
    kernel_gemm_mxfp8(
        arg_c, arg_a, arg_b, arg_sa, arg_sb, arg_g0, arg_g1, i32_m, i32_n, i32_k, i32_lda, i32_ldb, i32_ldc,
        i32_ldsa, i32_ldsb, value_attrs=attrs,
    ).launch(
        grid=grid,
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


# A/B switches over the configs' choices (None: as the config says)
C_EARLY_ON = True  # beta = 1 into bf16: fetch the old C during the last K step where it fits
PERSIST_ON = None  # bf16 out, kpair 2 x 2 slots: persistent workgroups, the next tile loaded during the stores
XCD_REMAP_ON = None  # grouped-M: consecutive m tiles of a grid chunk on one XCD
STAGGER_ON = None  # start the (first wave of) workgroups in this many phases (0: together)


def _sw(switch, default):
    return default if switch is None else switch


_NUM_CUS: list = []


def _num_cus() -> int:
    """The device's CUs (persistent launches start one workgroup per CU); NUM_CUS where there is no device."""
    if not _NUM_CUS:
        dev = torch.cuda.is_available() and torch.cuda.device_count() > 0
        _NUM_CUS.append(torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
                        if dev else NUM_CUS)
    return _NUM_CUS[0]


def constexprs(mode, cfg, M, N, G, packed, b_preshuffled, k_tail, out_f32, beta1) -> tuple:
    """The constexpr tail of ``_launch_gemm_mxfp8`` (and the compiled-kernel key) of a call. A cluster must tile the
    grid exactly (a workgroup wholly past M or N would sit in a multicast group): it shrinks to what the tiles
    divide; grouped-M m tiles belong to different experts, so it multicasts A along N only."""
    cm = 1 if mode == GROUPED_M else _divisor_cluster((M + cfg.tile_m - 1) // cfg.tile_m, cfg.cluster_m)
    cn = _divisor_cluster((N + cfg.tile_n - 1) // cfg.tile_n, cfg.cluster_n)
    c_early = C_EARLY_ON and beta1 and not out_f32 and N % cfg.tile_n == 0 and \
        c_early_fits(cfg._replace(cluster_m=cm, cluster_n=cn))
    persist = _sw(PERSIST_ON, cfg.persist) and cfg.num_buffers == 2 and not out_f32 and c_early_fits(cfg) and \
        (c_early or not beta1)
    return (cfg.tile_m, cfg.tile_n, cfg.num_buffers, cfg.kpair, cm, cn, bool(packed), bool(b_preshuffled),
            bool(k_tail), mode, G, bool(out_f32), bool(beta1), bool(c_early), bool(persist),
            bool(_sw(XCD_REMAP_ON, cfg.xcd_remap) and mode == GROUPED_M and not persist), int(_sw(STAGGER_ON, cfg.stagger)),
            _num_cus() if persist else 0)


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
    const = constexprs(mode, cfg, M, N, G, sa.dim() == 4, b_preshuffled, k_tail, _F16_OUT[out.dtype], beta1)
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


def _slack(t: torch.Tensor) -> int:
    """Bytes of ``t``'s storage after its last element."""
    end = t.storage_offset() + sum((n - 1) * st for n, st in zip(t.shape, t.stride())) + 1
    return t.untyped_storage().nbytes() - end * t.element_size()


def _tail_safe(cfg: MXGemmConfig, v: int, tensors, b_preshuffled: bool, packed: bool, who: str) -> MXGemmConfig:
    """A partial last K step of ``v`` (mod 128 * kpair) loads whole tile rows and lets the TDM clamp the data (its
    inner extent bounds the data, not the addressing): the last row of a, b and the scales is read up to a step past
    its data. A half step (kpair 2, v <= 128) loads only its first half of a, a row-major b and packed scales, which
    then read nothing past the data at 128-aligned K; a preshuffled b (2 KiB group rows) and raw scales still do. So:
    ``cfg`` if the storages have that slack (the quantizers allocate it), else kpair 1 when K is 128-aligned (no
    partial step); K % 128 != 0 without the slack is refused (pad K to 128, as the MX quantizers do)."""
    v %= WMMA_K * cfg.kpair
    if v == 0:
        return cfg
    width = WMMA_K if cfg.kpair == 1 or v <= WMMA_K else 2 * WMMA_K
    need = (width - v, (WMMA_K * cfg.kpair - v) * 16 if b_preshuffled else width - v,
            0 if packed else 4 * cfg.kpair - (v + 31) // 32)
    need = need + need[2:]
    if all(_slack(t) >= n for t, n in zip(tensors, need)):
        return cfg
    if v % WMMA_K == 0:  # kpair 2 with a half step: kpair 1 has none
        return cfg._replace(kpair=1, num_buffers=min(4, 2 * cfg.num_buffers))
    raise ValueError(f"{who}: K % 128 = {v % WMMA_K} would read up to {max(need)} B past the last row of a / b / the "
                     f"scales, more than their storages hold: pad K to a multiple of 128 or give them that slack")


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
    cfg = _tail_safe(cfg, K, (a, b, a_scale, b_scale), b_preshuffled, packed, "gemm_mxfp8_gfx1250")
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
    cfg = _tail_safe(cfg, K, (a, b, a_scale, b_scale), b_preshuffled, packed, who)
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
    cfg = _tail_safe(cfg, WMMA_K, (a, b, a_scale, b_scale), b_preshuffled, packed, who)  # any expert: 128 * odd
    return _launch(who, VARIABLE_K, cfg, out, a, b, a_scale, b_scale, group_offs, None, M, N, T, a.stride(0),
                   b.stride(0), ldsa, ldsb, G, b_preshuffled, accumulate, True)
