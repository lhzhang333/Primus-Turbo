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

Raw layouts throughout: codes row-major, scales ``[rows, ceil(cols / 32)]`` uint8 viewed as
``float8_e8m0fnu``. One kernel family, one 128 x 128 bf16 tile per workgroup (8 waves):

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
* ``grouped`` (MoE): the column copy per expert, zero-padded to 128 tokens (pad blocks: scale 127,
  Turbo's grouped contract); the row copy is the dense one (``offs_row = group_offs``).

Scale rules (compile-time): ``nosat`` (default: the smallest 2^e with amax / 2^e <= 448) and
``even`` (Turbo's HIP rule). ``tests/pytorch/ref/mxfp8_ref.py`` is the bitwise reference.
"""

from __future__ import annotations

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

I32, I64 = fx.Int32, fx.Int64
BM = BK = 128  # tile rows x columns
PITCH = 2 * BK + 16  # LDS bytes per bf16 tile row
CPITCH = BM + 32  # LDS bytes per staged column-copy row (2-way bank conflicts on the stage writes)
NT = 256  # 8 waves
# LDS: the bf16 tile (once consumed: the column copy's staged codes, rows of CPITCH bytes), the scale bytes
# (the row copy's [BM][4], then the column copy's [BK][4]), the 2d block maxima
O_SR = BM * PITCH
O_SC = O_SR + BM * 4
O_TAB = O_SC + BK * 4
LDS_BYTES = O_TAB + 64
# TDM group-1 word 0: 2-byte elements, LDS padding 4 dwords after every 64 (one tile row)
TDM_W0 = (1 << 16) | (1 << 20) | (5 << 22) | (3 << 25)
RULES = {"nosat": 0x1F, "even": 0x08}  # added to |amax| bf16 bits before taking the exponent
ROWS, DUAL, BLK2D, GROUPED = range(4)  # kernel modes
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


def _store16(rsrc, off, codes, ok, room, width, cm=0):
    """16 code bytes at byte offset ``off``; ``width`` < 16 stores only the first ``room`` bytes."""
    if const_expr(width == 16):
        buffer_store(codes, rsrc, off, mask=ok, offset_is_bytes=True, cache_modifier=cm)
        return
    vals = codes.bitcast(fx.Int32 if width == 4 else fx.Int8)
    for s in range_constexpr(16 // width):
        buffer_store(vals[s], rsrc, off + I32(s * width), mask=ok & (I32((s + 1) * width) <= room), cache_modifier=cm,
                     offset_is_bytes=True)


def _store_byte(rsrc, off, byte, ok, cm=0):
    buffer_store(ArithValue(_raw(byte)).trunci(T.i8), rsrc, off, mask=ok, offset_is_bytes=True, cache_modifier=cm)


def _tdm_load(lds, gaddr, cols, rows, stride, cm=0):
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
    rocdl.tensor_load_to_lds(vec[0], vec[1], z4, z4, z8, cache_policy=cm)


def _group_scan(arg_go, mc0, G, lane):
    """Expert of the column-copy tile at padded token ``mc0`` (experts padded to 128 tokens), lane-
    parallel: lane l of chunk c takes expert 32 c + l, a DPP prefix sum gives the padded offsets, a
    ballot the tile's expert. Returns (first input row, real rows, [(lane's expert, padded len,
    padded offset)] per chunk); expert G's offset is the padded total."""
    rs = _rsrc(_addr(arg_go), I32(8 * (G + 1)))
    acc, in0, nreal, per = I32(0), I32(0), I32(0), []
    for ch in range_constexpr(G // 32 + 1):
        g = I32(ch * 32) + lane
        o0 = I32(buffer_load(rs, g * I32(2), vec_width=1, dtype=T.i32))
        ln = (g < I32(G)).select(I32(buffer_load(rs, g * I32(2) + I32(2), vec_width=1, dtype=T.i32)) - o0, I32(0))
        pl = ((ln + I32(BM - 1)) >> I32(7)) << I32(7)
        x = pl
        for k in (1, 2, 4, 8):  # row_shr k: inclusive sums within each 16-lane row
            x = x + I32(_rocdl.update_dpp(T.i32, _raw(I32(0)), _raw(x), 0x110 + k, 0xF, 0xF, True))
        x = x + (lane >= I32(16)).select(I32(rocdl.readlane(T.i32, x, 15)), I32(0))
        excl = acc + x - pl
        hit = rocdl.ballot(T.i32, (mc0 >= excl) & (mc0 < excl + pl))
        src = I32(_llvm.call_intrinsic(T.i32, "llvm.cttz.i32", [_raw(I32(hit)), _raw(fx.Boolean(False))], [], []))
        found = I32(hit) != I32(0)
        in0 = found.select(I32(rocdl.readlane(T.i32, o0 + mc0 - excl, src)), in0)
        nreal = found.select(I32(rocdl.readlane(T.i32, ln - (mc0 - excl), src)), nreal)
        per.append((g, pl, excl))
        acc = acc + I32(rocdl.readlane(T.i32, x, 31))
    return in0, nreal, per


@flyc.jit
def _launch(
    arg_x: fx.Pointer,
    arg_qr: fx.Pointer,
    arg_sr: fx.Pointer,
    arg_qc: fx.Pointer,
    arg_sc: fx.Pointer,
    arg_go: fx.Pointer,
    arg_lc: fx.Pointer,
    arg_oc: fx.Pointer,
    stream: fx.Stream,
    i32_m: fx.Int32,
    i32_mc: fx.Int32,
    i32_grid: fx.Int32,
    i32_batch: fx.Int32,
    K: Constexpr[int],
    MODE: Constexpr[int],
    RULE_C: Constexpr[int],
    G: Constexpr[int],
    WM: Constexpr[int],
    CPOL: Constexpr[int],
):
    NKT = (K + BK - 1) // BK
    KB = (K + 31) // 32
    WK = 16 if K % 16 == 0 else (4 if K % 4 == 0 else 1)
    COLS = MODE != ROWS
    CST = CPOL & 31  # output stores' cache policy (gfx12 TH | SCOPE << 3); CPOL >> 5: the TDM load's

    @flyc.kernel(name=f"mxfp8_quant_gfx1250_m{MODE}", known_block_size=[NT, 1, 1])
    def kern(
        arg_x: fx.Pointer,
        arg_qr: fx.Pointer,
        arg_sr: fx.Pointer,
        arg_qc: fx.Pointer,
        arg_sc: fx.Pointer,
        arg_go: fx.Pointer,
        arg_lc: fx.Pointer,
        arg_oc: fx.Pointer,
        i32_m: fx.Int32,
        i32_mc: fx.Int32,
    ):
        arena = fx.SharedAllocator(static=False)
        arena.allocate(LDS_BYTES)
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
        if const_expr(MODE == GROUPED):
            in0, nreal, per = _group_scan(arg_go, mc0, G, lane)
            if (pid == I32(0)) & (wave == I32(0)):  # the padded lens / offsets of the column copy (int64)
                lc, oc = _rsrc(_addr(arg_lc), I32(8 * G)), _rsrc(_addr(arg_oc), I32(8 * G + 8))
                for g, pl, excl in per:
                    buffer_store(Vec.from_elements([pl, I32(0)], I32), lc, g * I32(2), mask=g < I32(G))
                    buffer_store(Vec.from_elements([excl, I32(0)], I32), oc, g * I32(2), mask=g <= I32(G))
        else:
            in0 = mc0
            nreal = i32_m - mc0
        nreal = (nreal > I32(BM)).select(I32(BM), nreal)
        if nreal > I32(0):
            row0 = I64(bat) * I64(i32_m) + I64(in0)  # the tile's first input row
            if wave == I32(0):
                gx = _addr(arg_x) + (row0 * I64(K) + I64(k0)) * I64(2)
                _tdm_load(base, gx, I32(K) - k0, nreal, K, CPOL >> 5)
                rocdl.s_wait_tensorcnt(0)
            gpu.barrier()
            if const_expr(MODE == BLK2D):  # 32 x 32 block maxima: 16 lanes per block, 2 rows each
                bi, sub = t // I32(16), t % I32(16)
                a0 = base + ((bi // I32(4)) * I32(32) + sub * I32(2)) * I32(PITCH) + (bi % I32(4)) * I32(64)
                a = _amax([_lds_v4(a0 + I32(r * PITCH + 16 * q)) for r in range(2) for q in range(4)])
                for ctrl in (0xB1, 0x4E, 0x164, 0x168):  # xor 1, 2, 4, 8
                    a = _max(a, _dpp(a, ctrl))
                _lds_st(base + I32(O_TAB) + bi * I32(4), a, 4)
                gpu.barrier()

            def blk_amax(words, tab, partner):
                if const_expr(MODE == BLK2D):
                    return I32(_llvm.LoadOp(T.i32, _lds_ptr(base + I32(O_TAB) + tab * I32(4)), alignment=4).result)
                a = _amax(words)
                return _max(a, _dpp(a, partner))

            # ---- rows phase: lane = (row, 32-block kb, half h), 4 rows per wave and pass; codes straight
            # to q_row (8 lanes = one 128-byte line), scale bytes to LDS
            qr = _rsrc(_addr(arg_qr) + row0 * I64(K), nreal * I32(K))
            kb, h = (lane % I32(8)) // I32(2), lane % I32(2)
            col = k0 + kb * I32(32) + h * I32(16)
            for p in range_constexpr(BM // 32):
                r = I32(p * 32) + wave * I32(4) + lane // I32(8)
                a0 = base + r * I32(PITCH) + (kb * I32(2) + h) * I32(32)
                words = [_lds_v4(a0), _lds_v4(a0 + I32(16))]
                byte = _e8m0(blk_amax(words, (r // I32(32)) * I32(4) + kb, 0xB1), RULE_C)
                codes = _cvt16(words, byte, RULE_C)
                _store16(qr, r * I32(K) + col, codes, (r < nreal) & (col < I32(K)), I32(K) - col, WK, CST)
                if h == I32(0):
                    _lds_st(base + I32(O_SR) + r * I32(4) + kb, ArithValue(_raw(byte)).trunci(T.i8), 1)

            # ---- cols phase: lane = (column of an 8-column chunk, 16 rows of a 64-row slab); codes and
            # scale bytes to LDS. Groups 16 rows apart hit the same banks whatever the row pitch: odd
            # groups read their second 8 rows first (2-way conflicts, the minimum).
            if const_expr(COLS):
                g, j = lane // I32(8), lane % I32(8)
                odd = (g % I32(2)) == I32(1)
                first = odd.select(I32(8 * PITCH), I32(0))
                staged = []
                for cp in range_constexpr(BK // 64):
                    c = (I32(cp * 8) + wave) * I32(8) + j
                    for rp in range_constexpr(BM // 64):
                        m = I32(rp * 64) + g * I32(16)  # the lane's first row in the tile
                        a0 = base + (m + j) * I32(PITCH) + (c - j) * I32(2)
                        words = [_lds_tr16(a0 + first), _lds_tr16(a0 + I32(8 * PITCH) - first)]
                        blk = I32(rp * 2) + g // I32(2)
                        byte = _e8m0(blk_amax(words, blk * I32(4) + c // I32(32), 0x168), RULE_C)
                        if const_expr(MODE == GROUPED):  # blocks of padding only
                            byte = (blk * I32(32) >= nreal).select(I32(PAD_SCALE), byte)
                        staged.append((c * I32(CPITCH) + m, _cvt16(words, byte, RULE_C, odd), c * I32(4) + blk, byte))
                gpu.barrier()  # every wave is done with the bf16 tile: stage the column copy in its space
                for qo, codes, so, byte in staged:
                    _lds_st(base + qo, codes, 16)
                    if g % I32(2) == I32(0):
                        _lds_st(base + I32(O_SC) + so, ArithValue(_raw(byte)).trunci(T.i8), 1)
            gpu.barrier()

            # ---- stores from LDS: the column copy by whole 128-byte lines, scales one row per lane
            if const_expr(COLS):
                mcb = (i32_mc + I32(31)) // I32(32)
                cbase = I64(bat) * I64(K) + I64(k0)  # the tile's first row of the column copy
                ncol = I32(K) - k0
                ncol = (ncol > I32(BK)).select(I32(BK), ncol)
                qc = _rsrc(_addr(arg_qc) + cbase * I64(i32_mc), ncol * i32_mc)
                width, mroom = (16, I32(BM)) if MODE == GROUPED else (WM, nreal)
                for i in range_constexpr(BK * BM // 16 // NT):
                    c, q16 = (t + I32(i * NT)) // I32(8), (t % I32(8)) * I32(16)
                    v = _lds_v4(base + c * I32(CPITCH) + q16)
                    _store16(qc, c * i32_mc + mc0 + q16, v, (c < ncol) & (q16 < mroom), mroom - q16, width, CST)
            rs = t % I32(BM)  # lanes 0..127: row rs of the row copy's scales; 128..255: of the column copy's
            sb = I32(_llvm.LoadOp(T.i32, _lds_ptr(base + I32(O_SR) + t * I32(4)), alignment=4).result)
            if t < I32(BM):
                sr = _rsrc(_addr(arg_sr) + row0 * I64(KB), nreal * I32(KB))
                if const_expr(K % BK == 0):  # 4 whole, dword-aligned scale bytes per row
                    buffer_store(sb, sr, rs * I32(KB) + kt * I32(4), mask=rs < nreal, offset_is_bytes=True, cache_modifier=CST)
                else:
                    for b in range_constexpr(4):
                        ok = (rs < nreal) & (kt * I32(4) + I32(b) < I32(KB))
                        _store_byte(sr, rs * I32(KB) + kt * I32(4) + I32(b), sb >> I32(8 * b), ok, CST)
            elif const_expr(COLS):
                sc = _rsrc(_addr(arg_sc) + cbase * I64(mcb), ncol * mcb)
                if const_expr(MODE == GROUPED):
                    buffer_store(sb, sc, rs * mcb + mt * I32(4), mask=rs < ncol, offset_is_bytes=True, cache_modifier=CST)
                else:
                    if (nreal == I32(BM)) & (mcb % I32(4) == I32(0)):
                        buffer_store(sb, sc, rs * mcb + mt * I32(4), mask=rs < ncol, offset_is_bytes=True, cache_modifier=CST)
                    else:
                        for b in range_constexpr(4):
                            ok = (rs < ncol) & (I32(32 * b) < nreal)
                            _store_byte(sc, rs * mcb + mt * I32(4) + I32(b), sb >> I32(8 * b), ok, CST)

    kern(arg_x, arg_qr, arg_sr, arg_qc, arg_sc, arg_go, arg_lc, arg_oc, i32_m, i32_mc).launch(
        grid=(i32_grid, i32_batch, 1), block=(NT, 1, 1), stream=stream
    )


_COMPILED: dict = {}
CACHE_POLICY = 0  # stores | TDM load << 5 (gfx12 TH | SCOPE << 3 each); 0 = default write-back
_E8 = getattr(torch, "float8_e8m0fnu", torch.uint8)


def _ptr(t):
    return flyc.from_c_void_p(fx.Int8, t.data_ptr(), assumed_align=16)


def _width(n):
    return 16 if n % 16 == 0 else (4 if n % 4 == 0 else 1)


def _run(mode, rule, x, qr, sr, qc=None, sc=None, go=None, lc=None, oc=None, mc=0, m_tiles=0):
    batch, M, K = x.shape
    grid = m_tiles * -(-K // BK)
    if grid == 0 or batch == 0:
        return
    args = [_ptr(a if a is not None else x) for a in (x, qr, sr, qc, sc, go, lc, oc)]
    args += [torch.cuda.current_stream(), M, mc, grid, batch, K, mode, RULES[rule]]
    args += [0 if go is None else go.shape[0] - 1, _width(mc), CACHE_POLICY]
    key = tuple(args[13:])
    fn = _COMPILED.get(key)
    if fn is None:
        fn = _COMPILED[key] = flyc.compile(_launch, *args)
    if fn is not None:
        fn(*args)


def _check(x, out_dtype, rule, ndim):
    assert x.ndim == ndim and x.is_cuda and x.dtype == torch.bfloat16, f"bf16 [{ndim}-D] CUDA input, got {x.dtype}"
    assert out_dtype == torch.float8_e4m3fn, f"E4M3 codes only, got {out_dtype}"
    assert rule in RULES, f"rule must be one of {tuple(RULES)}, got {rule!r}"
    return x.contiguous()


def quant_mxfp8_rows(x, out_dtype=torch.float8_e4m3fn, rule="nosat"):
    """Row copy only: x [M, K] bf16 -> (q [M, K] e4m3, s [M, ceil(K/32)] e8m0)."""
    x = _check(x, out_dtype, rule, 2)
    M, K = x.shape
    q = torch.empty(M, K, dtype=out_dtype, device=x.device)
    s = torch.empty(M, -(-K // 32), dtype=torch.uint8, device=x.device)
    _run(ROWS, rule, x.unsqueeze(0), q, s, mc=M, m_tiles=-(-M // BM))
    return q, s.view(_E8)


def quant_mxfp8_raw_batched(x, out_dtype=torch.float8_e4m3fn, row_2d=False, col_2d=False, rule="nosat"):
    """Dual cast of x [B, M, K] bf16, read once: (q_row [B, M, K], s_row [B, M, ceil(K/32)],
    q_col [B, K, M], s_col [B, K, ceil(M/32)]). ``row_2d == col_2d == True``: 32 x 32 blocks (weights),
    one scale per block shared by both copies; otherwise 1 x 32 blocks along each copy's rows."""
    assert row_2d == col_2d, "the 32x32 block scale is shared by both copies"
    x = _check(x, out_dtype, rule, 3)
    B, M, K = x.shape
    qr = torch.empty(B, M, K, dtype=out_dtype, device=x.device)
    sr = torch.empty(B, M, -(-K // 32), dtype=torch.uint8, device=x.device)
    qc = torch.empty(B, K, M, dtype=out_dtype, device=x.device)
    sc = torch.empty(B, K, -(-M // 32), dtype=torch.uint8, device=x.device)
    _run(BLK2D if row_2d else DUAL, rule, x, qr, sr, qc, sc, mc=M, m_tiles=-(-M // BM))
    return qr, sr.view(_E8), qc, sc.view(_E8)


def quant_mxfp8_raw(x, out_dtype=torch.float8_e4m3fn, row_2d=False, col_2d=False, rule="nosat"):
    """Dual cast of x [M, K]: ``quant_mxfp8_raw_batched`` with B = 1 (2-D outputs)."""
    assert x.ndim == 2, f"expects 2-D, got {x.ndim}-D"
    return tuple(t[0] for t in quant_mxfp8_raw_batched(x.unsqueeze(0), out_dtype, row_2d, col_2d, rule))


def grouped_quant_mxfp8_raw(x, group_lens, group_offs, out_dtype=torch.float8_e4m3fn, rule="nosat"):
    """MoE dual cast of x [M, K] (experts' tokens consecutive, ``group_offs`` [G + 1] int64):
    the row copy as ``quant_mxfp8_rows`` (no padding), the column copy per expert padded to 128
    tokens (``mxfp8_ref.quant_grouped``). Returns ``grouped_quantize_mxfp8_dual``'s 8-tuple:
    (q_row [M, K], s_row [M, ceil(K/32)], q_col [K, Mp], s_col [K, Mp / 32], lens_row = group_lens,
    offs_row = group_offs, lens_col [G], offs_col [G + 1]); Mp bounds offs_col[G] (no host sync), the
    columns past offs_col[G] are not written, nor are rows of x past group_offs[G] (tokens of no expert)."""
    x = _check(x, out_dtype, rule, 2)
    M, K = x.shape
    G = group_lens.shape[0]
    assert group_offs.dtype == torch.int64 and group_offs.shape[0] == G + 1 and G > 0
    Mp = -(-(M + G * BM) // BM) * BM
    qr = torch.empty(M, K, dtype=out_dtype, device=x.device)
    sr = torch.empty(M, -(-K // 32), dtype=torch.uint8, device=x.device)
    qc = torch.empty(K, Mp, dtype=out_dtype, device=x.device)
    sc = torch.empty(K, Mp // 32, dtype=torch.uint8, device=x.device)
    lc = torch.empty(G, dtype=torch.int64, device=x.device)
    oc = torch.empty(G + 1, dtype=torch.int64, device=x.device)
    _run(GROUPED, rule, x.unsqueeze(0), qr, sr, qc, sc, group_offs.contiguous(), lc, oc, mc=Mp, m_tiles=Mp // BM)
    return qr, sr.view(_E8), qc, sc.view(_E8), group_lens, group_offs, lc, oc
