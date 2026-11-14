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

"""MXFP8 quantization for gfx1250 (MI455X): E4M3 codes, one E8M0 scale per 32 elements.

Layouts, per copy (``layout`` bits; the same kernel, only the store addressing differs): codes row-major, or
(``SHUF_ROW`` / ``SHUF_COL``) in 16x16 blocks ``[rows/16][cols/16][16][16]``, same shape (rows, cols % 16 == 0;
``mxfp8_ref.preshuffle_b16``); scales raw ``[rows, ceil(cols/32)]`` uint8 viewed as ``float8_e8m0fnu``, or
(``PACK_ROW`` / ``PACK_COL``) packed ``[ceil(rows/32), ceil(cols/128), 32, 4]`` (``mxfp8_ref.pack_scales``: one
int32 is a row's 4 scales of a 128-wide step, pad bytes 0). Packed scales and 16x16 B operands are the fast
inputs of the gfx1250 MX GEMM. Batched inputs put the batch dim first. One kernel family, one 128 x 128 bf16
tile per workgroup (8 waves):

* the tile comes in by one TDM load into LDS rows padded by 16 bytes; rows and columns past the
  tensor (or past an expert's tokens) are zero-filled by the descriptor's bounds;
* rows phase: a lane takes half a 32-block of a row (two ``ds_load_b128``), |x| max on the bf16
  bits, the other half's max by DPP, the E8M0 byte by the scale rule, codes by
  ``v_cvt_scalef32_pk8_fp8_bf16``, one 16-byte store (8 lanes fill a 128-byte line);
* cols phase: the same on half a 32-block of a column, read by two ``ds_load_tr16_b128`` (the
  LDS transposes 8 x 8 bf16 blocks), giving the transposed copy ``q_col [cols, rows]``; its codes
  (staged in the consumed bf16 tile) and all scale bytes go through LDS, then out as whole 128-byte
  lines and one scale dword per row (byte and half-line stores cost the first version ~45%);
* ``2d`` (weights): one scale per 32 x 32 block, its max from a pre-pass into LDS; both copies use
  it, so ``q_t == q.T``;
* ``grouped`` (MoE): the row copy per expert zero-padded to 32 rows, the column copy to 128 tokens (pad
  rows / blocks: scale 127, Turbo's grouped contract); HIP's 8-tuple with both padded lens / offsets.
* producers of the tile besides a plain load: fused GLU / dGLU (MoE, or dense: ``glu_quant_mxfp8``),
  gather (permute), RMSNorm (``P_NORM``, ``rmsnorm_mxfp8_gfx1250``: the tile normalized by a precomputed
  ``rstd``).

Scale rules (compile-time): ``nosat`` (default: the smallest 2^e with amax / 2^e <= 448) and
``even`` (Turbo's HIP rule). ``tests/pytorch/ref/mxfp8_ref.py`` is the bitwise reference.
"""

from __future__ import annotations

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl
from flydsl._mlir.dialects import vector as _vector
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.arith import ArithValue
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import Constexpr, T
from flydsl.expr.typing import Vector as Vec

from primus_turbo.flydsl.utils.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)
from primus_turbo.flydsl.utils.gemm_epilogue_helper import (
    _check_activation,
    _check_clamp_limit,
    _glu_act,
    _glu_act_grad,
    _glu_clamp,
    _glu_kept,
)
from primus_turbo.flydsl.utils.prims import _row16_sum_f32

I32, I64 = fx.Int32, fx.Int64
BM = BK = 128  # tile rows x columns
PITCH = 2 * BK + 16  # LDS bytes per bf16 tile row
CPITCH = BM + 32  # LDS bytes per staged column-copy row (2-way bank conflicts on the stage writes)
NT = 256  # 8 waves
# TDM group-1 word 0: 2-byte elements, LDS padding 4 dwords after every 64 (one tile row)
TDM_W0 = (1 << 16) | (1 << 20) | (5 << 22) | (3 << 25)
RULES = {"nosat": 0x1F, "even": 0x08}  # added to |amax| bf16 bits before taking the exponent
P_DENSE, P_GROUPED, P_GLU, P_DGLU, P_GATHER, P_NORM = range(6)  # producers of the bf16 tile
O_ROWS, O_COLS = 1, 2  # outputs
PAD_SCALE = 127


def _addr(p):
    return I64(fx.ptrtoint(p))


def _rsrc(a, nbytes):
    """Buffer resource at byte address ``a`` with an access bound: a masked store (offset 0x7FFFFFFF)
    is dropped only when ``nbytes`` < 2^31, never without a bound."""
    return create_buffer_resource_from_addr(a, num_records_bytes=nbytes)


def _lds_ptr(a):
    return _llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), _raw(a))


def _lds_v4(a):
    return Vec(_llvm.LoadOp(T.vec(4, T.i32), _lds_ptr(a), alignment=16).result)


def _lds_tr16(a):
    return Vec(_rocdl.ds_load_tr16_b128(ir.VectorType.get([8], T.i16), _lds_ptr(a))).bitcast(I32)


def _lds_st(a, v, align):
    _llvm.StoreOp(_raw(v), _lds_ptr(a), alignment=align)


def _dpp(v, ctrl):
    return I32(_rocdl.update_dpp(T.i32, _raw(v), _raw(v), ctrl, 0xF, 0xF, False))


def _max(a, b):
    return (a > b).select(a, b)


def _amax(words):
    """max |x| bits (bf16, in an i32) of the bf16 pairs in ``words`` (vector<4xi32> list): unsigned max
    of the bits shifted left by one (drops the sign, one packed shift per pair), shifted back."""
    m = None
    for w in words:
        a = Vec(w).bitcast(fx.Uint16) << fx.Uint16(1)
        m = a if m is None else (a > m).select(a, m)
    return m.reduce("max").to(I32) >> I32(1)


def _e8m0(amax, rule_c):
    """E8M0 byte of a block from its max |x| bf16 bits; Inf / NaN -> 0xFF."""
    byte = ((amax + I32(rule_c)) >> I32(7)) - I32(8)
    byte = _max(byte, I32(0))
    return (amax >= I32(0x7F80)).select(I32(0xFF), byte)


def _cvt16(words, byte, rule_c, swap=None):
    """16 bf16 (two vector<4xi32>) / 2^(byte - 127) -> 16 E4M3 codes (vector<4xi32>; ``swap``: the
    words' halves in reverse order). The conversion uses the scale's exponent field only (0 reads as
    2^-127), keeps denormals, and turns anything rounding past 448 into NaN: ``even`` clamps |x| to
    448 * 2^(byte - 127) first."""
    if const_expr(rule_c == RULES["even"]):
        bound = (byte >= I32(247)).select(I32(0x7F7F), ((byte + I32(8)) << I32(7)) | I32(0x60)).to(fx.Int16)
        clamped = []
        for w in words:
            v = Vec(w).bitcast(fx.Int16)
            a = v & fx.Int16(0x7FFF)
            clamped.append(((a > bound).select(bound, a) | (v & fx.Int16(-0x8000))).bitcast(I32))
        words = clamped
    scale = I32(byte << I32(23)).bitcast(fx.Float32)
    out = [
        Vec(_rocdl.cvt_scalef32_pk8_fp8_bf16(T.vec(2, T.i32), _raw(Vec(w).bitcast(fx.BFloat16)), _raw(scale)))
        for w in words
    ]
    if const_expr(swap is None):
        return out[0].shuffle(out[1], [0, 1, 2, 3])
    return swap.select(out[1].shuffle(out[0], [0, 1, 2, 3]), out[0].shuffle(out[1], [0, 1, 2, 3]))


def _store16(rsrc, off, codes, ok, room, width):
    """16 code bytes at byte offset ``off``; ``width`` < 16 stores only the first ``room`` bytes."""
    if const_expr(width == 16):
        buffer_store(codes, rsrc, off, mask=ok, offset_is_bytes=True)
        return
    vals = codes.bitcast(fx.Int32 if width == 4 else fx.Int8)
    for s in range_constexpr(16 // width):
        buffer_store(vals[s], rsrc, off + I32(s * width), mask=ok & (I32((s + 1) * width) <= room),
                     offset_is_bytes=True)


def _store_byte(rsrc, off, byte, ok):
    buffer_store(ArithValue(_raw(byte)).trunci(T.i8), rsrc, off, mask=ok, offset_is_bytes=True)


def _tdm_load(lds, gaddr, cols, rows, stride):
    """TDM load of a BM x BK bf16 tile at ``gaddr`` (row stride ``stride`` elements) into LDS at
    ``lds``; ``cols`` / ``rows`` (from the tile start) bound it, the rest of the tile is zero."""
    g0 = [I32(1), lds, I32(gaddr), I32(gaddr >> I64(32)) | I32(-(1 << 31))]
    g1 = [
        I32(TDM_W0),
        (cols & I32(0xFFFF)) << I32(16),
        ((cols >> I32(16)) & I32(0xFFFF)) | ((rows & I32(0xFFFF)) << I32(16)),
        ((rows >> I32(16)) & I32(0xFFFF)) | I32(BK << 16),
        I32(BM),
        I32(stride),
        I32(0),
        I32(0),
    ]
    vec = [_vector.from_elements(T.vec(len(g), T.i32), [_raw(v) for v in g]) for g in (g0, g1)]
    z4, z8 = _raw(Vec.filled(4, 0, I32)), _raw(Vec.filled(8, 0, I32))
    rocdl.tensor_load_to_lds(vec[0], vec[1], z4, z4, z8)


def _group_scan(arg_go, mc0, G, lane):
    """Expert of the column-copy tile at padded token ``mc0``, lane-parallel: lane l of chunk c takes
    expert 32 c + l, DPP prefix sums give the offsets of the column copy (experts padded to 128 tokens)
    and of the row copy (to 32 rows), a ballot the tile's expert. Returns (first input row, real rows,
    first row-copy row, [(lane's expert, (len, offset) of the column copy, (len, offset) of the rows)]
    per chunk); expert G's offsets are the padded totals."""
    rs = _rsrc(_addr(arg_go), I32(8 * (G + 1)))
    acc, acc32, in0, nreal, rb32, per = I32(0), I32(0), I32(0), I32(0), I32(0), []

    def scan(v):  # inclusive prefix over the wave: row_shr within 16-lane rows, then row 0's total
        for k in (1, 2, 4, 8):
            v = v + I32(_rocdl.update_dpp(T.i32, _raw(I32(0)), _raw(v), 0x110 + k, 0xF, 0xF, True))
        return v + (lane >= I32(16)).select(I32(rocdl.readlane(T.i32, v, 15)), I32(0))

    for ch in range_constexpr(G // 32 + 1):
        g = I32(ch * 32) + lane
        o0 = I32(buffer_load(rs, g * I32(2), vec_width=1, dtype=T.i32))
        ln = (g < I32(G)).select(I32(buffer_load(rs, g * I32(2) + I32(2), vec_width=1, dtype=T.i32)) - o0, I32(0))
        pl, pl32 = ((ln + I32(127)) >> I32(7)) << I32(7), ((ln + I32(31)) >> I32(5)) << I32(5)
        x, x32 = scan(pl), scan(pl32)
        excl, excl32 = acc + x - pl, acc32 + x32 - pl32
        hit = rocdl.ballot(T.i32, (mc0 >= excl) & (mc0 < excl + pl))
        src = I32(_llvm.call_intrinsic(T.i32, "llvm.cttz.i32", [_raw(I32(hit)), _raw(fx.Boolean(False))], [], []))
        found = I32(hit) != I32(0)
        in0 = found.select(I32(rocdl.readlane(T.i32, o0 + mc0 - excl, src)), in0)
        nreal = found.select(I32(rocdl.readlane(T.i32, ln - (mc0 - excl), src)), nreal)
        rb32 = found.select(I32(rocdl.readlane(T.i32, excl32 + mc0 - excl, src)), rb32)
        per.append((g, (pl, excl), (pl32, excl32)))
        acc, acc32 = acc + I32(rocdl.readlane(T.i32, x, 31)), acc32 + I32(rocdl.readlane(T.i32, x32, 31))
    return in0, nreal, rb32, per


def _act8(g, u, prob, act, clamp):
    """GLU forward on 8 columns (f32), the gfx950 phase-2 math: clamp, silu as g / (1 + exp2(-g log2 e))
    (gelu via its sigmoid form), times up, times the row's prob; the caller rounds to bf16."""
    gc, uc, _, _ = _glu_clamp(g, u, clamp if clamp else None, lanes=8)
    if const_expr(act == "silu"):
        ex = Vec.from_elements(
            [fx.Float32(rocdl.exp2(T.f32, _raw(v * fx.Float32(-1.4426950408889634)))) for v in [gc[i] for i in range(8)]],
            fx.Float32,
        )
        gate = gc / (ex + fx.Float32(1.0))
    else:
        gate = _glu_act(gc, act, lanes=8)
    return (gate * uc) * prob


def _dact8(d, g, u, prob, act, clamp):
    """GLU backward on 8 columns, the gfx950 phase-2 math: (dgate, dup, sum of d * f(g) * up)."""
    gc, uc, kept_g, kept_u = _glu_clamp(g, u, clamp if clamp else None, lanes=8)
    a, da = _glu_act_grad(gc, act, lanes=8)
    gp = fx.Float32(0.0)
    for i in range_constexpr(8):
        gp = gp + d[i] * a[i] * uc[i]
    dp = d * prob
    return _glu_kept(dp * uc * da, kept_g, lanes=8), _glu_kept(dp * a, kept_u, lanes=8), gp


def _bf8(v):
    return v.to(fx.BFloat16).bitcast(I32)


@flyc.jit
def _launch(
    arg_x: fx.Pointer,
    arg_x2: fx.Pointer,
    arg_qr: fx.Pointer,
    arg_sr: fx.Pointer,
    arg_qc: fx.Pointer,
    arg_sc: fx.Pointer,
    arg_go: fx.Pointer,
    arg_lc: fx.Pointer,
    arg_oc: fx.Pointer,
    arg_p: fx.Pointer,
    arg_gp: fx.Pointer,
    arg_dbg: fx.Pointer,
    arg_z: fx.Pointer,
    stream: fx.Stream,
    i32_m: fx.Int32,
    i32_mc: fx.Int32,
    i32_grid: fx.Int32,
    i32_batch: fx.Int32,
    i32_t: fx.Int32,
    K: Constexpr[int],
    PROD: Constexpr[int],
    OUT: Constexpr[int],
    B2D: Constexpr[int],
    RULE_C: Constexpr[int],
    G: Constexpr[int],
    WM: Constexpr[int],
    KO: Constexpr[int],
    PADM: Constexpr[int],
    ACT: Constexpr[str],
    CLAMP: Constexpr[float],
    AUX: Constexpr[int],
    LAY: Constexpr[int],
    NOP: Constexpr[int],
):
    NKT = (K + BK - 1) // BK
    # expert-padded copies: the MoE producers; GLU / dGLU without group offsets (G = 0) are dense
    EXP = PROD in (P_GROUPED, P_GATHER) or (PROD in (P_GLU, P_DGLU) and G > 0)
    HALVES = 2 if PROD == P_DGLU else 1  # dGLU quantizes dgate (columns [0, K)) and dup ([K, 2K))
    OW = 2 * K if PROD == P_DGLU else KO  # row-copy width (KO > K: zero columns to a 128 pad)
    KV = K if PROD in (P_GLU, P_DGLU) else KO  # columns of a half the row copy holds
    KBO = (OW + 31) // 32
    WK = 16 if OW % 16 == 0 else (4 if OW % 4 == 0 else 1)
    NTILE = {P_GLU: 2, P_DGLU: 3}.get(PROD, 1)
    TILE = BM * PITCH
    O_SR = NTILE * TILE  # scale bytes: the row copy's [BM][4], then the column copy's [BK][4]
    O_SC, O_TAB = O_SR + BM * 4, O_SR + (BM + BK) * 4
    FULLM = EXP or PADM  # the column copy holds whole 128-token tiles
    # LAY: packed scales [rows/32][K/128][32][4] for the row (1) / column (2) copy, 16x16-preshuffled codes
    # [rows/16][K/16][16][16] for the row (4) / column (8) copy (the gfx1250 MX GEMM's fast layouts)
    PR, PC, SR, SC = (LAY & 1, LAY & 2, LAY & 4, LAY & 8)
    RT = (OW + BK - 1) // BK  # 128-column steps of the row copy (packed scales)

    @flyc.kernel(name=f"mxfp8_quant_gfx1250_p{PROD}o{OUT}", known_block_size=[NT, 1, 1])
    def kern(
        arg_x: fx.Pointer,
        arg_x2: fx.Pointer,
        arg_qr: fx.Pointer,
        arg_sr: fx.Pointer,
        arg_qc: fx.Pointer,
        arg_sc: fx.Pointer,
        arg_go: fx.Pointer,
        arg_lc: fx.Pointer,
        arg_oc: fx.Pointer,
        arg_p: fx.Pointer,
        arg_gp: fx.Pointer,
        arg_dbg: fx.Pointer,
        arg_z: fx.Pointer,
        i32_m: fx.Int32,
        i32_mc: fx.Int32,
        i32_t: fx.Int32,
    ):
        arena = fx.SharedAllocator(static=False)
        arena.allocate(O_TAB + 64)
        base = I32(I64(fx.ptrtoint(arena.base_ptr)))
        t = I32(fx.thread_idx.x)
        lane = t % I32(32)
        wave = rocdl.readfirstlane(T.i32, t // I32(32))
        pid = I32(fx.block_idx.x)
        bat = I32(fx.block_idx.y)
        mt = pid // I32(NKT)
        kt = pid - mt * I32(NKT)
        k0 = kt * I32(BK)
        mc0 = mt * I32(BM)  # the tile's first row in the column copy's token space
        if const_expr(EXP):
            in0, nreal, rb32, per = _group_scan(arg_go, mc0, G, lane)
            if (pid == I32(0)) & (wave == I32(0)):  # padded lens / offsets (int64 [2, G + 1]) of both copies
                for a, w in ((arg_lc, 1), (arg_oc, 2)):  # column copy (128 tokens), row copy (32 rows)
                    lo = _rsrc(_addr(a), I32(16 * (G + 1)))
                    for g, *pairs in per:
                        ln, ex = pairs[w - 1]
                        buffer_store(Vec.from_elements([ln, I32(0)], I32), lo, g * I32(2), mask=g < I32(G))
                        buffer_store(Vec.from_elements([ex, I32(0)], I32), lo, (g + I32(G + 1)) * I32(2),
                                     mask=g <= I32(G))
        else:
            in0, rb32 = mc0, I32(0)
            nreal = i32_m - mc0
        nreal = (nreal > I32(BM)).select(I32(BM), nreal)
        if nreal > I32(0):
            row0 = I64(bat) * I64(i32_m) + I64(in0)  # the tile's first input (= output) row
            cols = I32(K) - k0
            # ---- producer: the bf16 tile(s) into LDS, rows / columns past the data zero
            if const_expr(PROD in (P_DENSE, P_GROUPED, P_NORM)):
                xs = i32_t if PROD == P_NORM else I32(K)  # row stride of x (elements)
                if wave == I32(0):
                    _tdm_load(base, _addr(arg_x) + (row0 * I64(xs) + I64(k0)) * I64(2), cols, nreal, xs)
                    rocdl.s_wait_tensorcnt(0)
            elif const_expr(PROD in (P_GLU, P_DGLU)):  # gate / up halves of l1, then dact
                l1 = _addr(arg_x2 if PROD == P_DGLU else arg_x) + (row0 * I64(2 * K) + I64(k0)) * I64(2)
                if wave < I32(NTILE):
                    src = [l1, l1 + I64(2 * K), _addr(arg_x) + (row0 * I64(K) + I64(k0)) * I64(2)]
                    for w in range_constexpr(NTILE):
                        if wave == I32(w):
                            _tdm_load(base + I32(w * TILE), src[w], cols, nreal, K if w == 2 else 2 * K)
                    rocdl.s_wait_tensorcnt(0)
            else:  # gather: row r is x[dest2src[in0 + r]], one 16-byte async load per lane and chunk
                ri = _rsrc(_addr(arg_x2), i32_m * I32(4))
                rp = _rsrc(_addr(arg_p), i32_m * I32(4))
                for i in range_constexpr(BM * BK // 8 // NT):
                    q = t + I32(i * NT)
                    r, c16 = q // I32(16), q % I32(16)
                    src = I32(buffer_load(ri, in0 + r, vec_width=1, dtype=T.i32))
                    ok = (r < nreal) & (src >= I32(0)) & (src < i32_t) & (k0 + c16 * I32(8) < I32(K))
                    if const_expr(AUX):  # zero the rows whose prob is 0 (the permute's padding rule)
                        ok = ok & (fx.Float32(buffer_load(rp, in0 + r, vec_width=1, dtype=T.f32)) != fx.Float32(0.0))
                    ga = ok.select(_addr(arg_x) + (I64(src) * I64(K) + I64(k0 + c16 * I32(8))) * I64(2), _addr(arg_z))
                    gp = _llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), _raw(ga))
                    _rocdl.global_load_async_to_lds_b128(gp, _lds_ptr(base + r * I32(PITCH) + c16 * I32(16)), 0)
                rocdl.s_wait_asynccnt(0)
            gpu.barrier()
            if const_expr(PROD in (P_GLU, P_DGLU)):  # elementwise GLU / dGLU, rounded to bf16 in place
                rp = _rsrc(_addr(arg_p), i32_m * I32(4))
                dbg = _rsrc(_addr(arg_dbg) + row0 * I64(OW * 2), nreal * I32(OW * 2))
                gpr = _rsrc(_addr(arg_gp) + I64(kt) * I64(i32_m) * I64(4), i32_m * I32(4))
                for i in range_constexpr(BM * BK // 8 // NT):
                    q = t + I32(i * NT)
                    r, c8 = q // I32(16), (q % I32(16)) * I32(8)
                    off = base + r * I32(PITCH) + c8 * I32(2)
                    prob = fx.Float32(1.0) if NOP else fx.Float32(buffer_load(rp, in0 + r, vec_width=1, dtype=T.f32))
                    gv, uv = [Vec(_lds_v4(off + I32(w * TILE))).bitcast(fx.BFloat16).to(fx.Float32) for w in range(2)]
                    dok = (r < nreal) & (k0 + c8 < I32(K))
                    if const_expr(PROD == P_GLU):
                        res = [_bf8(_act8(gv, uv, prob, ACT, CLAMP))]
                    else:
                        dv = Vec(_lds_v4(off + I32(2 * TILE))).bitcast(fx.BFloat16).to(fx.Float32)
                        dg, du, gsum = _dact8(dv, gv, uv, prob, ACT, CLAMP)
                        res = [_bf8(dg), _bf8(du)]
                        gsum = _row16_sum_f32(gsum)  # lane 15 of each 16-lane row: the row's sum over the tile
                        if const_expr(not NOP):
                            buffer_store(gsum, gpr, in0 + r, mask=(r < nreal) & (q % I32(16) == I32(15)))
                    for w in range_constexpr(len(res)):
                        _lds_st(off + I32(w * TILE), res[w], 16)
                        if const_expr(AUX):  # the bf16 values (tests)
                            buffer_store(res[w], dbg, (r * I32(OW) + I32(w * K) + k0 + c8) * I32(2), mask=dok,
                                         offset_is_bytes=True)
                gpu.barrier()
            if const_expr(PROD == P_NORM):  # RMSNorm: y = bf16((x * rstd[row]) * gamma[col]) in place; AUX: y out
                rp = _rsrc(_addr(arg_p), i32_m * I32(4))
                ys = _rsrc(_addr(arg_dbg) + row0 * I64(K * 2), nreal * I32(K * 2))
                c8 = (t % I32(16)) * I32(8)  # a lane's 8 columns, the same in every pass
                gam = Vec(buffer_load(_rsrc(_addr(arg_x2), I32(K * 2)), (k0 + c8) >> I32(1), vec_width=4, dtype=T.i32))
                gam = gam.bitcast(fx.BFloat16).to(fx.Float32)
                for i in range_constexpr(BM * BK // 8 // NT):
                    r = t // I32(16) + I32(i * 16)
                    off = base + r * I32(PITCH) + c8 * I32(2)
                    rstd = fx.Float32(buffer_load(rp, in0 + r, vec_width=1, dtype=T.f32))
                    yv = _bf8((Vec(_lds_v4(off)).bitcast(fx.BFloat16).to(fx.Float32) * rstd) * gam)
                    _lds_st(off, yv, 16)
                    if const_expr(AUX):
                        buffer_store(yv, ys, (r * I32(K) + k0 + c8) * I32(2), mask=(r < nreal) & (k0 + c8 < I32(K)),
                                     offset_is_bytes=True)
                gpu.barrier()
            if const_expr(B2D):  # 32 x 32 block maxima: 16 lanes per block, 2 rows each
                bi, sub = t // I32(16), t % I32(16)
                a0 = base + ((bi // I32(4)) * I32(32) + sub * I32(2)) * I32(PITCH) + (bi % I32(4)) * I32(64)
                a = _amax([_lds_v4(a0 + I32(r * PITCH + 16 * q)) for r in range(2) for q in range(4)])
                for ctrl in (0xB1, 0x4E, 0x164, 0x168):  # xor 1, 2, 4, 8
                    a = _max(a, _dpp(a, ctrl))
                _lds_st(base + I32(O_TAB) + bi * I32(4), a, 4)
                gpu.barrier()

            def blk_amax(words, tab, partner):
                if const_expr(B2D):
                    return I32(_llvm.LoadOp(T.i32, _lds_ptr(base + I32(O_TAB) + tab * I32(4)), alignment=4).result)
                a = _amax(words)
                return _max(a, _dpp(a, partner))

            for hf in range_constexpr(HALVES):
                tb = base + I32((hf if PROD == P_DGLU else 0) * TILE)  # the bf16 tile of this half
                sb = base + I32((2 - 2 * hf if PROD == P_DGLU else 0) * TILE)  # a consumed tile: the stage
                ocol = I32(hf * K)  # the half's first column of the row copy / row of the column copy
                # ---- rows phase: lane = (row, 32-block kb, half h), 4 rows per wave and pass; codes
                # straight to q_row (8 lanes = one 128-byte line), scale bytes to LDS. 16x16 codes: 16 rows x
                # one kb per wave and pass, so each 16-lane half stores a whole 256-byte block. Expert row copies
                # hold whole 32-row groups: rows past the tokens are zero codes, scale 127.
                nrows = ((nreal + I32(31)) >> I32(5)) << I32(5) if EXP else nreal
                rowbase = I64(rb32) if EXP else I64(bat) * I64(i32_m) + I64(in0)
                if const_expr(OUT & O_ROWS):
                    qr = _rsrc(_addr(arg_qr) + rowbase * I64(OW), nrows * I32(OW))
                    h = lane % I32(2)
                    for p in range_constexpr(BM // 32):
                        if const_expr(SR):
                            u = I32(p * 8) + wave
                            r, kb = (u >> I32(2)) * I32(16) + lane // I32(2), u & I32(3)
                        else:
                            r, kb = I32(p * 32) + wave * I32(4) + lane // I32(8), (lane % I32(8)) // I32(2)
                        col = k0 + kb * I32(32) + h * I32(16)
                        a0 = tb + r * I32(PITCH) + (kb * I32(2) + h) * I32(32)
                        words = [_lds_v4(a0), _lds_v4(a0 + I32(16))]
                        byte = _e8m0(blk_amax(words, (r // I32(32)) * I32(4) + kb, 0xB1), RULE_C)
                        if const_expr(EXP):  # pad rows: 127 over the real columns, 0 in the K padding
                            byte = ((r >= nreal) & (k0 + kb * I32(32) < I32(K))).select(I32(PAD_SCALE), byte)
                        codes = _cvt16(words, byte, RULE_C)
                        ok = (r < nrows) & (col < I32(KV))
                        if const_expr(SR):
                            off = ((r >> I32(4)) * I32(OW // 16) + ((ocol + col) >> I32(4))) * I32(256) + (r & I32(15)) * I32(16)
                            _store16(qr, off, codes, ok, I32(16), 16)
                        else:
                            _store16(qr, r * I32(OW) + ocol + col, codes, ok, I32(KV) - col, WK)
                        if h == I32(0):
                            _lds_st(base + I32(O_SR) + r * I32(4) + kb, ArithValue(_raw(byte)).trunci(T.i8), 1)
                # ---- cols phase: lane = (column of an 8-column chunk, 16 rows of a 64-row slab); codes and
                # scale bytes staged in LDS. Groups 16 rows apart hit the same banks whatever the row pitch:
                # odd groups read their second 8 rows first (2-way conflicts, the minimum).
                if const_expr(OUT & O_COLS):
                    g, j = lane // I32(8), lane % I32(8)
                    odd = (g % I32(2)) == I32(1)
                    first = odd.select(I32(8 * PITCH), I32(0))
                    staged = []
                    for cp in range_constexpr(BK // 64):
                        c = (I32(cp * 8) + wave) * I32(8) + j
                        for rp in range_constexpr(BM // 64):
                            m = I32(rp * 64) + g * I32(16)  # the lane's first row in the tile
                            a0 = tb + (m + j) * I32(PITCH) + (c - j) * I32(2)
                            words = [_lds_tr16(a0 + first), _lds_tr16(a0 + I32(8 * PITCH) - first)]
                            blk = I32(rp * 2) + g // I32(2)
                            byte = _e8m0(blk_amax(words, blk * I32(4) + c // I32(32), 0x168), RULE_C)
                            if const_expr(EXP):  # blocks of padding only
                                byte = (blk * I32(32) >= nreal).select(I32(PAD_SCALE), byte)
                            staged.append((c * I32(CPITCH) + m, _cvt16(words, byte, RULE_C, odd), c * I32(4) + blk, byte))
                    gpu.barrier()  # every wave is done with the tile: stage the column copy in a consumed one
                    for qo, codes, so, byte in staged:
                        _lds_st(sb + qo, codes, 16)
                        if g % I32(2) == I32(0):
                            _lds_st(base + I32(O_SC) + so, ArithValue(_raw(byte)).trunci(T.i8), 1)
                gpu.barrier()

                # ---- stores from LDS: the column copy by whole 128-byte lines (16x16 blocks: 256 bytes),
                # scales one row per lane (packed: a 32-row group's 128 bytes per 32 lanes)
                ncol = I32(K) - k0
                ncol = (ncol > I32(BK)).select(I32(BK), ncol)
                mcb = (i32_mc + I32(31)) // I32(32)
                cbase = I64(bat) * I64(HALVES * K) + I64(ocol) + I64(k0)  # the tile's first column-copy row
                if const_expr(OUT & O_COLS):
                    qc = _rsrc(_addr(arg_qc) + cbase * I64(i32_mc), ncol * i32_mc)
                    width, mroom = (16, I32(BM)) if FULLM else (WM, nreal)
                    for i in range_constexpr(BK * BM // 16 // NT):
                        idx = t + I32(i * NT)
                        if const_expr(SC):  # 16 lanes per 16x16 block: 256 contiguous bytes
                            c, q16 = (idx // I32(128)) * I32(16) + idx % I32(16), ((idx // I32(16)) % I32(8)) * I32(16)
                            off = ((c >> I32(4)) * (i32_mc >> I32(4)) + ((mc0 + q16) >> I32(4))) * I32(256) + (c & I32(15)) * I32(16)
                            w = 16
                        else:
                            c, q16 = idx // I32(8), (idx % I32(8)) * I32(16)
                            off, w = c * i32_mc + mc0 + q16, width
                        v = _lds_v4(sb + c * I32(CPITCH) + q16)
                        _store16(qc, off, v, (c < ncol) & (q16 < mroom), mroom - q16, w)
                rs = t % I32(BM)  # lanes 0..127: row rs of the row copy's scales; 128..255: of the column copy's
                sbyte = I32(_llvm.LoadOp(T.i32, _lds_ptr(base + I32(O_SR) + t * I32(4)), alignment=4).result)
                if t < I32(BM):
                    if const_expr(OUT & O_ROWS):
                        kbg = (ocol + k0) // I32(32)
                        if const_expr(PR):  # rows past the data in the last 32-row group: 0 (2d blocks would not be)
                            gbase = I64(rb32 >> I32(5)) if EXP else I64(bat) * I64((i32_m + I32(31)) >> I32(5)) + I64(in0 >> I32(5))
                            n32 = ((nreal + I32(31)) >> I32(5)) << I32(5)
                            sr = _rsrc(_addr(arg_sr) + gbase * I64(RT * 128), (n32 >> I32(5)) * I32(RT * 128))
                            off = ((rs >> I32(5)) * I32(RT) + kbg // I32(4)) * I32(128) + (rs & I32(31)) * I32(4)
                            buffer_store((rs < nrows).select(sbyte, I32(0)), sr, off, mask=rs < n32, offset_is_bytes=True)
                        else:
                            sr = _rsrc(_addr(arg_sr) + rowbase * I64(KBO), nrows * I32(KBO))
                            if const_expr(OW % BK == 0 and KV % BK == 0):  # 4 whole, dword-aligned bytes per row
                                buffer_store(sbyte, sr, rs * I32(KBO) + kbg, mask=rs < nrows, offset_is_bytes=True)
                            else:
                                for b in range_constexpr(4):
                                    ok = (rs < nrows) & ((k0 + I32(32 * b)) < I32(KV))
                                    _store_byte(sr, rs * I32(KBO) + kbg + I32(b), sbyte >> I32(8 * b), ok)
                elif const_expr(OUT & O_COLS):
                    if const_expr(PC):  # rows past the columns of x: zero bytes (as mxfp8_ref.pack_scales pads)
                        ct = (mcb + I32(3)) >> I32(2)  # 128-token steps of the column copy
                        gbase = I64(bat) * I64((HALVES * K + 31) // 32) + I64((ocol + k0) >> I32(5))
                        c32 = ((ncol + I32(31)) >> I32(5)) << I32(5)
                        sc = _rsrc(_addr(arg_sc) + gbase * I64(ct) * I64(128), (c32 >> I32(5)) * ct * I32(128))
                        off = ((rs >> I32(5)) * ct + mt) * I32(128) + (rs & I32(31)) * I32(4)
                        buffer_store((rs < ncol).select(sbyte, I32(0)), sc, off, mask=rs < c32, offset_is_bytes=True)
                    else:
                        sc = _rsrc(_addr(arg_sc) + cbase * I64(mcb), ncol * mcb)
                        if (mcb % I32(4) == I32(0)) & ((nreal == I32(BM)) | fx.Boolean(bool(FULLM))):
                            buffer_store(sbyte, sc, rs * mcb + mt * I32(4), mask=rs < ncol, offset_is_bytes=True)
                        else:
                            for b in range_constexpr(4):
                                ok = (rs < ncol) & (I32(32 * b) < (I32(BM) if FULLM else nreal))
                                _store_byte(sc, rs * mcb + mt * I32(4) + I32(b), sbyte >> I32(8 * b), ok)
                if const_expr(hf + 1 < HALVES):
                    gpu.barrier()  # the scale bytes and the stage are rewritten by the next half

    kern(arg_x, arg_x2, arg_qr, arg_sr, arg_qc, arg_sc, arg_go, arg_lc, arg_oc, arg_p, arg_gp, arg_dbg, arg_z,
         i32_m, i32_mc, i32_t).launch(grid=(i32_grid, i32_batch, 1), block=(NT, 1, 1), stream=stream)


_COMPILED: dict = {}
_E8 = getattr(torch, "float8_e8m0fnu", torch.uint8)
_ZERO: dict = {}
PACK_ROW, PACK_COL, SHUF_ROW, SHUF_COL = 1, 2, 4, 8  # ``layout`` bits (the kernel's LAY)
ROW_BITS, COL_BITS = PACK_ROW | SHUF_ROW, PACK_COL | SHUF_COL


def _ptr(t):
    return flyc.from_c_void_p(fx.Int8, t.data_ptr(), assumed_align=16)


def _width(n):
    return 16 if n % 16 == 0 else (4 if n % 4 == 0 else 1)


def _ceil128(n):
    return -(-n // BK) * BK


_RAW_STREAM = getattr(torch._C, "_cuda_getCurrentRawStream", None)


def _run(prod, out, rule, x, outs, m, mc, m_tiles, k, ko=0, b2d=0, padm=0, go=None, x2=None, p=None, gp=None,
         dbg=None, t=0, act="silu", clamp=0.0, lay=0):
    """Launch one kernel of the family. ``outs`` = (q_row, s_row, q_col, s_col, lens / offs of the column copy,
    of the row copy (int64 [2, G + 1]: lens, offsets)), any None. The compiled launcher of a constexpr tail is
    called with the tensors' data_ptr() ints (no per-call pointer objects); compiling runs the call itself."""
    batch = 1 if x.ndim == 2 else x.shape[0]
    grid = m_tiles * -(-k // BK)
    if grid == 0 or batch == 0:
        return
    tail = (k, prod, out, b2d, RULES[rule], 0 if go is None else go.shape[0] - 1, _width(mc), ko or k, padm, act,
            float(clamp or 0.0), int(dbg is not None or (prod == P_GATHER and p is not None)), lay,
            int(p is None and prod in (P_GLU, P_DGLU)))
    dev = x.get_device()
    zero = _ZERO.get(dev)
    if zero is None:
        zero = _ZERO[dev] = torch.zeros(64, dtype=torch.uint8, device=x.device)
    tensors = (x, x2, *outs[:4], go, *outs[4:], p, gp, dbg, zero)
    fn = _COMPILED.get(tail)
    if fn is not None:
        xp = x.data_ptr()
        fn(*[xp if a is None else a.data_ptr() for a in tensors],
           _RAW_STREAM(dev) if _RAW_STREAM is not None else torch.cuda.current_stream(), m, mc, grid, batch, t, *tail)
        return
    fn = flyc.compile(_launch, *[_ptr(x if a is None else a) for a in tensors], torch.cuda.current_stream(), m, mc,
                      grid, batch, t, *tail)  # compiles, then runs this call
    if fn is not None:  # (None under COMPILE_ONLY)
        _COMPILED[tail] = fn


def _check(x, out_dtype, rule, ndims):
    assert x.ndim in ndims and x.is_cuda and x.dtype == torch.bfloat16, f"bf16 {ndims}-D CUDA input, got {x.dtype}"
    assert out_dtype == torch.float8_e4m3fn, f"E4M3 codes only, got {out_dtype}"
    assert rule in RULES, f"rule must be one of {tuple(RULES)}, got {rule!r}"
    return x if x.is_contiguous() else x.contiguous()


# Storage past every GEMM operand the quantizers write: a TDM load's inner extent clamps the data, not the
# addressing, so the gfx1250 MX GEMM's last K step may address up to ~2 KiB past an operand's end.
GEMM_SLACK = 4096
_ALIGN = 256  # every carved tensor starts 256-byte aligned
_ESIZE = {torch.float8_e4m3fn: 1, _E8: 1, torch.uint8: 1, torch.int64: 8}
_PLANS: dict = {}


def _carve(specs, device):
    """The tensors of ``specs`` (``(shape, dtype)`` each) carved from one ``torch.empty``: contiguous, 256-byte
    aligned, each followed by ``GEMM_SLACK`` bytes of the allocation. The offsets are planned once per ``specs``.
    (One allocation per copy, not per dual cast: a copy's codes and scales live and die together, while the row
    and column copies of an activation do not -- the column copy is kept for the wgrad.)"""
    plan = _PLANS.get(specs)
    if plan is None:
        off, parts = 0, []
        for shape, dtype in specs:
            size = _ESIZE[dtype]
            parts.append((dtype, shape, tuple(math.prod(shape[i + 1:]) for i in range(len(shape))), off // size))
            off = -(-(off + math.prod(shape) * size + GEMM_SLACK) // _ALIGN) * _ALIGN
        if len(_PLANS) > 4096:  # many distinct shapes: start over rather than grow without bound
            _PLANS.clear()
        plan = _PLANS[specs] = (off, tuple(parts))
    total, parts = plan
    buf = torch.empty(total, dtype=torch.uint8, device=device)
    views = {torch.uint8: buf}
    out = []
    for dtype, shape, strides, eoff in parts:
        v = views.get(dtype)
        if v is None:
            v = views[dtype] = buf.view(dtype)
        out.append(v.as_strided(shape, strides, eoff))
    return out


def _bufs(lead, rows, cols, pack, shuf, dtype, like):
    """A copy's codes [*lead, rows, cols] (16x16 blocks when ``shuf``) and E8M0 scales, raw [*lead, rows,
    ceil(cols/32)] or (``pack``) [*lead, ceil(rows/32), ceil(cols/128), 32, 4], in one allocation (``_carve``)."""
    assert not shuf or rows % 16 == cols % 16 == 0, f"16x16-preshuffled codes need rows, cols % 16 == 0: {rows}, {cols}"
    sshape = (-(-rows // 32), -(-cols // BK), 32, 4) if pack else (rows, -(-cols // 32))
    return _carve(((tuple(lead) + (rows, cols), dtype), (tuple(lead) + sshape, _E8)), like.device)


def quant_mxfp8_rows(x, out_dtype=torch.float8_e4m3fn, rule="nosat", blk_2d=False, pad=False, layout=0):
    """Row copy only: x [M, K] or [B, M, K] bf16 -> (q [.., M, K'], s [.., M, K'/32] e8m0); K' = K, or
    ceil128(K) with zero codes / byte-0 scales past K when ``pad`` (the HIP layout). ``layout``: its row bits."""
    x = _check(x, out_dtype, rule, (2, 3))
    *b, M, K = x.shape
    KO = _ceil128(K) if pad else K
    q, s = _bufs(b, M, KO, layout & PACK_ROW, layout & SHUF_ROW, out_dtype, x)
    _run(P_DENSE, O_ROWS, rule, x, (q, s, None, None, None, None), M, M, -(-M // BM), K, KO, int(blk_2d),
         lay=layout & ROW_BITS)
    return q, s


def quant_mxfp8_cols(x, out_dtype=torch.float8_e4m3fn, rule="nosat", blk_2d=False, pad=False, layout=0):
    """Column copy only: x [M, K] or [B, M, K] -> (q_col [.., K, M'], s_col [.., K, M'/32]); M' = M or ceil128(M).
    ``layout``: its column bits."""
    x = _check(x, out_dtype, rule, (2, 3))
    *b, M, K = x.shape
    Mc = _ceil128(M) if pad else M
    q, s = _bufs(b, K, Mc, layout & PACK_COL, layout & SHUF_COL, out_dtype, x)
    _run(P_DENSE, O_COLS, rule, x, (None, None, q, s, None, None), M, Mc, -(-M // BM), K, K, int(blk_2d), int(pad),
         lay=layout & COL_BITS)
    return q, s


def quant_mxfp8_raw_batched(x, out_dtype=torch.float8_e4m3fn, row_2d=False, col_2d=False, rule="nosat", pad=False,
                            layout=0):
    """Dual cast of x [B, M, K] bf16, read once: (q_row [B, M, K], s_row, q_col [B, K, M], s_col); raw scales
    [B, M, ceil(K/32)] / [B, K, ceil(M/32)] unless ``layout`` packs them. ``row_2d == col_2d == True``: 32 x 32
    blocks (weights), one scale per block shared by both copies; otherwise 1 x 32 blocks along each copy's rows.
    ``pad``: the HIP layout, K and M padded to 128 (zero codes, byte-0 scales)."""
    assert row_2d == col_2d, "the 32x32 block scale is shared by both copies"
    x = _check(x, out_dtype, rule, (3,))
    B, M, K = x.shape
    KO, Mc = (_ceil128(K), _ceil128(M)) if pad else (K, M)
    qr, sr = _bufs((B,), M, KO, layout & PACK_ROW, layout & SHUF_ROW, out_dtype, x)
    qc, sc = _bufs((B,), K, Mc, layout & PACK_COL, layout & SHUF_COL, out_dtype, x)
    _run(P_DENSE, O_ROWS | O_COLS, rule, x, (qr, sr, qc, sc, None, None), M, Mc, -(-M // BM), K, KO, int(row_2d),
         int(pad), lay=layout)
    return qr, sr, qc, sc


def quant_mxfp8_raw(x, out_dtype=torch.float8_e4m3fn, row_2d=False, col_2d=False, rule="nosat", pad=False, layout=0):
    """Dual cast of x [M, K]: ``quant_mxfp8_raw_batched`` with B = 1 (2-D outputs, 4-D packed scales)."""
    assert row_2d == col_2d, "the 32x32 block scale is shared by both copies"
    x = _check(x, out_dtype, rule, (2,))
    M, K = x.shape
    KO, Mc = (_ceil128(K), _ceil128(M)) if pad else (K, M)
    qr, sr = _bufs((), M, KO, layout & PACK_ROW, layout & SHUF_ROW, out_dtype, x)
    qc, sc = _bufs((), K, Mc, layout & PACK_COL, layout & SHUF_COL, out_dtype, x)
    _run(P_DENSE, O_ROWS | O_COLS, rule, x, (qr, sr, qc, sc, None, None), M, Mc, -(-M // BM), K, KO, int(row_2d),
         int(pad), lay=layout)
    return qr, sr, qc, sc


def _grouped(prod, x, group_lens, group_offs, out_dtype, rule, k, M, rows=True, cols=True, pad=False, layout=0, **kw):
    G = group_lens.shape[0]
    assert group_offs.dtype == torch.int64 and group_offs.shape[0] == G + 1 and G > 0
    OW = 2 * k if prod == P_DGLU else (_ceil128(k) if pad else k)
    assert not (prod == P_DGLU and rows and layout & PACK_ROW and k % BK), f"packed dGLU row scales need I % 128 == 0: {k}"
    Mr, Mp = -(-(M + 32 * G) // 32) * 32, -(-(M + BM * G) // BM) * BM  # bounds of the padded rows / tokens
    qr, sr = _bufs((), Mr, OW, layout & PACK_ROW, layout & SHUF_ROW, out_dtype, x) if rows else (None, None)
    C = OW if prod == P_DGLU else k
    qc, sc = _bufs((), C, Mp, layout & PACK_COL, layout & SHUF_COL, out_dtype, x) if cols else (None, None)
    lc, lr = _carve((((2, G + 1), torch.int64), ((2, G + 1), torch.int64)), x.device)
    out = (O_ROWS if rows else 0) | (O_COLS if cols else 0)
    lay = layout & ((ROW_BITS if rows else 0) | (COL_BITS if cols else 0))
    _run(prod, out, rule, x, (qr, sr, qc, sc, lc, lr), M, Mp, Mp // BM, k, OW if prod != P_DGLU else k,
         go=group_offs.contiguous(), lay=lay, **kw)
    return qr, sr, qc, sc, lr[0, :G], lr[1], lc[0, :G], lc[1]


def grouped_quant_mxfp8_raw(x, group_lens, group_offs, out_dtype=torch.float8_e4m3fn, rule="nosat", pad=False,
                            rows=True, cols=True, layout=0):
    """MoE dual cast of x [M, K] (experts' tokens consecutive, ``group_offs`` [G + 1] int64), HIP's
    ``grouped_quantize_mxfp8_dual`` contract (``mxfp8_ref.quant_grouped``): the row copy per expert padded to 32
    rows, the column copy per expert padded to 128 tokens (pad rows / blocks: zero codes, scale 127; ``pad``: K
    padded to 128). Returns (q_row [Mr, K], s_row, q_col [K, Mp], s_col, lens_row [G], offs_row [G + 1],
    lens_col [G], offs_col [G + 1]) with Mr = ceil32(M + 32 G), Mp = ceil128(M + 128 G): bounds (no host sync);
    rows past offs_row[G] / columns past offs_col[G] are not written, nor are tokens past group_offs[G].
    ``rows`` / ``cols`` False leave that copy out (None); ``layout``: packed scales / 16x16 codes per copy."""
    x = _check(x, out_dtype, rule, (2,))
    return _grouped(P_GROUPED, x, group_lens, group_offs, out_dtype, rule, x.shape[1], x.shape[0], rows, cols, pad,
                    layout)


def grouped_glu_quant_mxfp8(l1, probs, group_lens, group_offs, activation="silu", clamp_limit=None,
                            out_dtype=torch.float8_e4m3fn, rule="nosat", act_out=None, layout=0):
    """GLU forward + MoE dual cast: act = bf16(f(min(gate, L)) * clamp(up, -L, L) * probs[row]) from
    l1 [M, 2I] bf16 (gate | up, Turbo's fc1 layout; probs [M] fp32), quantized as
    ``grouped_quant_mxfp8_raw`` would quantize it (the 8-tuple, act [M, I]). ``act_out`` [M, I] bf16
    (tests): also receives act."""
    l1 = _check(l1, out_dtype, rule, (2,))
    M, I = l1.shape[0], l1.shape[1] // 2
    assert l1.shape[1] == 2 * I and probs.shape == (M,) and probs.dtype == torch.float32
    return _grouped(P_GLU, l1, group_lens, group_offs, out_dtype, rule, I, M, p=probs.contiguous(), dbg=act_out,
                    act=_check_activation(activation), clamp=_check_clamp_limit(clamp_limit), layout=layout)


def grouped_dglu_quant_mxfp8(grad_act, l1, probs, group_lens, group_offs, activation="silu", clamp_limit=None,
                             out_dtype=torch.float8_e4m3fn, rule="nosat", grad_out=None, layout=0):
    """GLU backward + MoE dual cast: from grad_act [M, I] bf16 (the fc2 dgrad), l1 [M, 2I] and probs [M]:
    grad_l1 = [dgate | dup] [M, 2I] (bf16, then quantized like ``grouped_quant_mxfp8_raw``; the column
    copy [2I, Mp]) and grad_probs [M] fp32 = sum_j grad_act * f(gate) * up (per-128-column partials,
    summed in a fixed order). Returns (grad_probs, *the 8-tuple). ``grad_out`` [M, 2I] bf16 (tests):
    also receives grad_l1. Packed row scales need I % 128 == 0."""
    grad_act, l1 = _check(grad_act, out_dtype, rule, (2,)), _check(l1, out_dtype, rule, (2,))
    M, I = grad_act.shape
    assert l1.shape == (M, 2 * I) and probs.shape == (M,) and probs.dtype == torch.float32 and I % 32 == 0
    part = torch.zeros(-(-I // BK), M, dtype=torch.float32, device=l1.device)
    out = _grouped(P_DGLU, grad_act, group_lens, group_offs, out_dtype, rule, I, M, x2=l1, p=probs.contiguous(),
                   gp=part, dbg=grad_out, act=_check_activation(activation), clamp=_check_clamp_limit(clamp_limit),
                   layout=layout)
    return (part.sum(0), *out)


def _dense_glu(prod, x, l1, probs, I, out_dtype, rule, pad, layout, **kw):
    """Dense GLU / dGLU + dual cast: x the tile source ([M, 2I] l1 for GLU, [M, I] grad_act for dGLU)."""
    M, W = l1.shape[0], (2 * I if prod == P_DGLU else I)  # W: columns of the quantized output
    assert probs is None or (probs.shape == (M,) and probs.dtype == torch.float32), "probs: fp32 [M] or None"
    assert not pad or I % BK == 0, f"the HIP padding needs I % 128 == 0 here: {I}"
    assert not (prod == P_DGLU and layout & PACK_ROW and I % BK), f"packed dGLU row scales need I % 128 == 0: {I}"
    Mc = _ceil128(M) if pad else M
    qr, sr = _bufs((), M, W, layout & PACK_ROW, layout & SHUF_ROW, out_dtype, x)
    qc, sc = _bufs((), W, Mc, layout & PACK_COL, layout & SHUF_COL, out_dtype, x)
    _run(prod, O_ROWS | O_COLS, rule, x, (qr, sr, qc, sc, None, None), M, Mc, -(-M // BM), I, W if prod != P_DGLU
         else I, 0, int(pad), p=None if probs is None else probs.contiguous(), lay=layout, **kw)
    return qr, sr, qc, sc


def glu_quant_mxfp8(l1, probs=None, activation="silu", clamp_limit=None, out_dtype=torch.float8_e4m3fn,
                    rule="nosat", pad=False, act_out=None, layout=0):
    """Dense GLU forward + dual cast (a dense SwiGLU MLP's fc2 operands, e.g. the shared expert): act =
    bf16(f(min(gate, L)) * clamp(up, -L, L) [* probs[row]]) from l1 [M, 2I] bf16, the grouped kernel's math, quantized
    as ``quant_mxfp8_raw(act, pad=pad, layout=layout)`` would: (q_row [M, I], s_row, q_col [I, M'], s_col).
    ``probs`` None: no row weights (bit-identical to all ones). ``act_out`` [M, I] bf16 (tests): also receives act."""
    l1 = _check(l1, out_dtype, rule, (2,))
    I = l1.shape[1] // 2
    assert l1.shape[1] == 2 * I
    return _dense_glu(P_GLU, l1, l1, probs, I, out_dtype, rule, pad, layout, dbg=act_out,
                      act=_check_activation(activation), clamp=_check_clamp_limit(clamp_limit))


def dglu_quant_mxfp8(grad_act, l1, probs=None, activation="silu", clamp_limit=None, out_dtype=torch.float8_e4m3fn,
                     rule="nosat", pad=False, grad_out=None, layout=0):
    """Dense GLU backward + dual cast (the fc1 dgrad / wgrad operands): grad_l1 = [dgate | dup] [M, 2I] from
    grad_act [M, I] (the fc2 dgrad) and l1 [M, 2I], quantized as ``quant_mxfp8_raw(grad_l1, pad=pad, layout=layout)``
    would. Returns (grad_probs [M] fp32 or None when ``probs`` is None, q_row [M, 2I], s_row, q_col [2I, M'],
    s_col). ``grad_out`` [M, 2I] bf16 (tests): also receives grad_l1."""
    grad_act, l1 = _check(grad_act, out_dtype, rule, (2,)), _check(l1, out_dtype, rule, (2,))
    M, I = grad_act.shape
    assert l1.shape == (M, 2 * I) and I % 32 == 0
    part = None if probs is None else torch.empty(-(-I // BK), M, dtype=torch.float32, device=l1.device)
    out = _dense_glu(P_DGLU, grad_act, l1, probs, I, out_dtype, rule, pad, layout, x2=l1, gp=part, dbg=grad_out,
                     act=_check_activation(activation), clamp=_check_clamp_limit(clamp_limit))
    return (None if part is None else part.sum(0), *out)


def grouped_gather_quant_mxfp8(x, dest2src, group_lens, group_offs, permuted_probs=None,
                               out_dtype=torch.float8_e4m3fn, rule="nosat", total_m=None, pad=False, layout=0):
    """Permute (gather) + MoE dual cast: row d of the permuted activation is x[dest2src[d]] (x [T, H]
    bf16, dest2src [M] int32, read by per-row async loads); a row whose dest2src is out of [0, T), or
    whose ``permuted_probs`` is exactly 0 (the routing padding rule), is zero. Returns the 8-tuple of
    ``grouped_quant_mxfp8_raw`` applied to the permuted [M, H] activation. Serves the backward too
    (dOut rows -> dY2, no probs)."""
    x = _check(x, out_dtype, rule, (2,))
    M = int(total_m if total_m is not None else dest2src.shape[0])
    assert dest2src.dtype == torch.int32 and dest2src.numel() >= M and x.shape[1] % 8 == 0
    pp = permuted_probs
    assert pp is None or (pp.dtype == torch.float32 and pp.numel() >= M)
    return _grouped(P_GATHER, x, group_lens, group_offs, out_dtype, rule, x.shape[1], M, pad=pad, layout=layout,
                    x2=dest2src.contiguous(), p=None if pp is None else pp.contiguous(), t=x.shape[0])
