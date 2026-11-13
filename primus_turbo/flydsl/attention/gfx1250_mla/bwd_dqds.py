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

"""gfx1250 MLA backward dQ from a dS scratch: dQ = dS K, S and dP are not recomputed.

dS is the scratch the dK / dV kernel writes (bf16, softmax scale included, packed causal), in the layout of
``csrc/kernels/attention/hipkittens/gfx1250_mla/include/dslayout.h`` (version 2):
  * one region per (batch, q head) bh = bat * Hq + qh, regions back to back, ``bwd_dkdv96.ds_blocks_per_head`` blocks each;
  * inside a region the kv16 rows kt = 0 .. Skv / 16 - 1 in order; row kt holds the q pairs (32 queries)
    qp = qp_first(kt) .. Sq / 32 - 1, qp_first(kt) = causal ? max(0, kt // 2 - (Skv - Sq) // 32) : 0;
  * a block is a [16 kv][32 q] bf16 tile, 1 KiB: chunk (kv, c) = queries [8 c, 8 c + 8) of kv row kv at block byte
    (c // 2) * 512 + ((c % 2) * 16 + kv) * 16; masked entries hold +-0.

k_dqds: grid (HG, ntile + nhead, B * Hq / HG), block 128 = 4 waves (one per SIMD, one workgroup per CU).
grid.y < ntile is a 384-query tile of [q_off, Sq) (4 waves x 96 queries, NQW 6 sub-tiles of 16 per wave), longest
first; grid.y == ntile (q_off == 256 only) the head [0, 256) as 4 waves x 64 queries (NQW 4). q pairs are dealt to
the waves in snake order (wave w: pairs w, 7 - w, 8 + w of the workgroup), so the fully masked pairs of the
diagonal spread over all waves.

Per 64-kv unit u (kv32 rows 2u, 2u + 1) every wave runs NQW x 12 WMMAs per kv32 row, dQ^T[16 d][16 q] +=
K^T[16 d][32 kv] dS^T[32 kv][16 q], on operands burst-loaded (ds_load_tr16_b128) one row ahead into the other
register set. The K and dS tiles come through two 3-stage LDS rings filled by the Tensor Data Mover; TDM issue is
split by role, one op per wave per unit: waves 0 / 3 move the dS of kv32 row 2u + w // 2 for all q pairs of the
workgroup (one 2-D op), waves 1 / 2 the K rows [64 u + 32 (w // 2), +32). One split workgroup barrier per unit:
arrive after this wave's next op landed and its ring reads retired, wait after the unit's first kv32 row run, then
the op of unit u + 3 goes into the freed stages. Fully masked q pairs are never loaded, and in the diagonal units a
wave-uniform branch skips their WMMAs (exact: they would add +-0). dQ leaves through per-wave row-major LDS images
as whole 128-B lines.

The arithmetic is that of ``k_dqds_c`` (and of the FlyDSL k_dqg dQ): every accumulator chains its kv32 rows in
ascending order from +0, so dQ is bitwise that of those kernels on the same dS.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx

# UNSTABLE(gfx1250): WMMA, ds_load_tr16, the raw TDM load and tensor_wait, s_wait_dscnt, the split barrier,
# sched barriers, readfirstlane, disable_xdl_arb_stall and the raw LLVM LDS loads / stores / inline asm have no
# stable API in FlyDSL 0.3.4.1.
from flydsl._mlir.dialects import llvm
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.rocdl import tdm_ops

from primus_turbo.flydsl.utils.buffer_ops import buffer_store, create_buffer_resource

from . import SOURCE_DIGEST
from .fwd_staging import lds_ptr

D_QK = 192
DV8_QK = D_QK // 8  # 16-B vectors per dq row
NDT = D_QK // 16  # 16-wide d tiles of dQ^T
WAVE = 32
NWAVE = 4  # waves per workgroup, one per SIMD
BLOCK = NWAVE * WAVE
KV32 = 32  # kv rows per WMMA step (kv32 row r = dS block rows 2r, 2r + 1)
MAIN_NQW = 6  # main tile: 4 waves x 96 queries
HEAD_NQW = 4  # merged head tile: 4 waves x 64 queries
MAIN_ROWS = 16 * MAIN_NQW * NWAVE  # 384
HEAD_ROWS = 16 * HEAD_NQW * NWAVE  # 256
DS_BLK_EL = 512  # bf16 per dS block [16 kv][32 q]
# dS block images in LDS: the TDM pads 16 B after every 256 data bytes, so the 32 tr16 lanes of one operand hit
# every bank exactly twice.
DSI_GRP_B = 256 + 16
DSI_BLK_B = 4 * DSI_GRP_B  # 1088
TDIM0 = 0x7FFFFFFF  # TDM inner extent: never clamps
DQ_IMG_ROW_B = 400  # dQ epilogue image row: 384 B data + 16 B pad
DQ_LINE_EL = 64  # bf16 per 128-B dq line
# K ring: 3 stages of 64 rows x 480 B (32 B of pad after every 128 B); half h = rows [32h, 32h + 32) of a unit.
C_KROW_B = 480
C_KHALF_B = KV32 * C_KROW_B
C_KSTG_B = 2 * C_KHALF_B
C_KDEPTH = 3
C_DSDEPTH = 3
C_DS_OFF = C_KDEPTH * C_KSTG_B  # the dS ring follows the K ring
C_TW = 1  # tensorcnt before a barrier: only the newest op may stay in flight
# Steady-state schedule: split barrier after SPLIT_AT WMMAs of a row run; the operand bursts interleaved
# {1 tr16, 1 WMMA} after IL_LEAD_A (first row) / IL_LEAD_C (second row) leading WMMAs.
SPLIT_AT = 66
IL_LEAD_A = 8
IL_LEAD_C = 4
# Steady units per loop iteration: a multiple of the ring depth, so every burst address is a precomputed base VGPR
# plus an immediate.
STEADY_UNROLL = 3
assert STEADY_UNROLL % C_KDEPTH == 0
# TDM group-1 word 0: bf16, LDS padding on; dS: interval 64 dwords (enc 5), amount 4 dwords (enc 3); K: interval
# 32 dwords (enc 4), amount 8 dwords (enc 7).
DS_W0 = (1 << 16) | (1 << 20) | (5 << 22) | (3 << 25)
BK_W0 = (1 << 16) | (1 << 20) | (4 << 22) | (7 << 25)


def _nbg(nqw):
    """dS q pairs (blocks) per workgroup."""
    return NWAVE * nqw // 2


def _slot_b(nqw):
    """One kv32 row x the workgroup's block images."""
    return 2 * _nbg(nqw) * DSI_BLK_B


def _dsstg_b(nqw):
    return 2 * _slot_b(nqw)


LDS_C = C_DS_OFF + C_DSDEPTH * _dsstg_b(MAIN_NQW)  # the rings
EPI_IMG_B = 16 * DQ_IMG_ROW_B
LDS_BYTES = LDS_C + NWAVE * EPI_IMG_B  # 274432: one workgroup per CU
assert LDS_BYTES <= 327680 < 2 * LDS_BYTES


def _k_imm(dt, h):
    """K^T tr16 immediate of d tile dt, kv half h (rows h*16 .. of the kv32 row)."""
    return dt * 32 + (dt // 4) * 32 + h * 16 * C_KROW_B


def _ds_half(qh):
    """Image offset of the 16-query half qh of a dS block (its second 512 data bytes)."""
    return qh * 2 * DSI_GRP_B


def _c32(v):
    """Python int as a signed 32-bit DSL constant."""
    v &= 0xFFFFFFFF
    return fx.Int32(v - (1 << 32) if v >= 1 << 31 else v)


def _ir(v):
    return fx.as_ir_value(v)


# ------------------------------------------------------------------------------------------- device helpers
def _tr16(addr):
    v8b = fx.Vector.make_type(8, fx.BFloat16)
    return fx.Vector(rocdl.ds_load_tr16_b128(v8b, lds_ptr(addr)))


def _cat(a, b):
    return a.shuffle(b, list(range(16)))


def _wmma(a, b, c):
    v8f = fx.Vector.make_type(8, fx.Float32)
    return rocdl.wmma_f32_16x16x32_bf16(v8f, _ir(a), _ir(b), _ir(c), reuseA=False, reuseB=False).result


def _keep(vals):
    """An empty asm statement that reads `vals`: they stay allocated (live) up to this point."""
    llvm.inline_asm(None, [_ir(v) for v in vals], "", ",".join(["v"] * len(vals)), has_side_effects=True)


def _sgb(mask, n):
    rocdl.sched_group_barrier(mask, n, 0)


def _sgb_interleave(lead, ratio, nt, nw):
    """{lead WMMA}, then {1 DS read, ratio WMMA} x nt, then the rest of the nw WMMAs."""
    assert lead + nt * ratio <= nw
    if lead:
        _sgb(0x008, lead)
    for _ in range(nt):
        _sgb(0x100, 1)
        _sgb(0x008, ratio)
    if nw - lead - nt * ratio:
        _sgb(0x008, nw - lead - nt * ratio)


def _split_arrive():
    fx.llvm.memory_fence(syncscope="workgroup", ordering=fx.AtomicOrdering.Release)
    rocdl.s_barrier_signal(-1)


def _split_wait():
    rocdl.s_barrier_wait(-1)
    fx.llvm.memory_fence(syncscope="workgroup", ordering=fx.AtomicOrdering.Acquire)


def _wg_sync():
    rocdl.sched_barrier(0)
    rocdl.s_wait_dscnt(0)
    _split_arrive()
    _split_wait()
    rocdl.sched_barrier(0)


def _vec_i32(vals):
    return fx.Vector.from_elements([v if isinstance(v, fx.Int32) else fx.Int32(v) for v in vals], fx.Int32)


def _tdm_issue(g0, g1):
    z4 = fx.Vector.filled(4, 0, fx.Int32)
    z8 = fx.Vector.filled(8, 0, fx.Int32)
    rocdl.tensor_load_to_lds(_ir(g0), _ir(g1), _ir(z4), _ir(z4), _ir(z8), cache_policy=0)


@flyc.jit
def _wmma_if(live, accs, a_ops, b_ops):
    """The WMMAs of one q pair behind a wave-uniform branch (masked units)."""
    res = accs
    if live:
        res = [_wmma(a, b, c) for a, b, c in zip(a_ops, b_ops, accs)]
    return res


# --------------------------------------------------------------------------------------------- the body
def _dqds_body(nqw, DS, K, DQ, Sq, Skv, Hq, Hkv, G, nkvt, cshift, causal, q0g, HG, nblk, dq_bytes, lds0):
    nb_g = _nbg(nqw)
    nacc = nqw * NDT
    slot_b, dsstg_b = _slot_b(nqw), _dsstg_b(nqw)
    nt_burst = 2 * NDT + 2 * nqw  # tr16 loads of one operand burst

    # ---- identity (wv, dsw, hh, hb, bat, qh, hkv are wave-uniform)
    tid = fx.Int32(fx.thread_idx.x)
    lane = tid % fx.Int32(WAVE)
    wv = fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, (tid // fx.Int32(WAVE)).ir_value()))
    dsw = (wv & fx.Int32(1)) == (wv >> fx.Int32(1))  # waves 0 / 3 issue dS ops, 1 / 2 K ops
    hh = wv >> fx.Int32(1)  # the kv32 row / K half of a unit this wave moves
    gx = fx.Int32(fx.block_idx.x)
    gz = fx.Int32(fx.block_idx.z)
    # HG % 8 == 0: every head's workgroups run on XCD x % 8, each XCD gets a contiguous range of heads
    xr = (HG % fx.Int32(8) == fx.Int32(0)).select(
        (gx % fx.Int32(8)) * (HG // fx.Int32(8)) + gx // fx.Int32(8), gx
    )
    hb = gz * HG + xr  # bat * Hq + qh = the dS region
    bat = hb // Hq
    qh = hb - bat * Hq
    hkv = qh // G
    qp0g = q0g // fx.Int32(32)
    mshift = cshift // fx.Int32(32)
    nqp = Sq // fx.Int32(32)
    is_causal = causal != fx.Int32(0)

    # ---- kv range: kv32 rows with an entry attended by one of the workgroup's queries; 64-kv units
    lim = (q0g + fx.Int32(16 * nqw * NWAVE) + cshift + fx.Int32(KV32 - 1)) // fx.Int32(KV32)
    lim = fx.min(fx.max(lim, fx.Int32(1)), nkvt)
    n32 = is_causal.select(lim, nkvt)
    nu = (n32 + fx.Int32(1)) // fx.Int32(2)
    # units [0, nfu) have no fully masked block in the workgroup's pairs in either kv32 row
    t_fu = (qp0g + mshift + fx.Int32(1)) // fx.Int32(2)
    nfu = is_causal.select(fx.max(fx.min(t_fu, nu), fx.Int32(0)), nu)
    # units run with real ops only (u + 3 < nu)
    nf = fx.min(fx.min(nfu, nu - fx.Int32(1)), fx.max(nu - fx.Int32(3), fx.Int32(0)))
    nloop = nu - fx.Int32(1)

    # ---- lanes
    lane_r = (lane // fx.Int32(16)) * fx.Int32(8) + lane % fx.Int32(8)  # tr16 row
    lane_c = ((lane // fx.Int32(8)) % fx.Int32(2)) * fx.Int32(8)  # tr16 column (elements)
    k_lane = lane_r * fx.Int32(C_KROW_B) + lane_c * fx.Int32(2)
    ds_lane = (lane_c // fx.Int32(8)) * fx.Int32(DSI_GRP_B) + lane_r * fx.Int32(16)

    def wave_pair(k):
        """q pair (0-based in the workgroup) of the wave's k-th pair: snake order."""
        return fx.Int32(4 * k + 3) - wv if k % 2 else fx.Int32(4 * k) + wv

    wp_off = [wave_pair(k) * fx.Int32(DSI_BLK_B) for k in range(nqw // 2)]

    # ---- TDM descriptors (all fields wave-uniform)
    ga_ds = fx.Int64(DS)  # the scratch's address (it can exceed the 2^31 elements of a tensor argument)
    ga_k = fx.Int64(fx.ptrtoint(fx.get_iter(K)))
    kst = fx.Int64(Hkv) * fx.Int64(D_QK)  # elements between consecutive k rows
    ga_dummy = ga_k + (fx.Int64(bat * Skv) * fx.Int64(Hkv) + fx.Int64(hkv)) * fx.Int64(D_QK * 2)

    def desc(lds_off, ga, w0, tdim1, tile0, tile1, stride):
        lo = fx.Int32(ga)
        hi = fx.Int32(ga >> fx.Int64(32)) | _c32(0x80000000)
        g0 = [fx.Int32(1), lds0 + lds_off, lo, hi]
        g1 = [
            w0,
            _c32((TDIM0 & 0xFFFF) << 16),
            ((tdim1 & fx.Int32(0xFFFF)) << fx.Int32(16)) | fx.Int32(TDIM0 >> 16),
            ((tdim1 >> fx.Int32(16)) & fx.Int32(0xFFFF)) | (tile0 << fx.Int32(16)),
            tile1,
            fx.Int32(stride),
            fx.Int32(stride >> fx.Int64(32)) & fx.Int32(0xFFFF),
            fx.Int32(0),
        ]
        return g0, g1

    def c_desc(u, real, stage=None):
        """This wave's role op of unit u. real: the caller guarantees u < nu and the op is a real one (steady
        units); else ops past the last unit, or of a row without stored pairs, are 16-B dummies. stage: u % 3 when
        known at trace time."""
        if stage is None:
            us = u % fx.Int32(3)
            k_st = us * fx.Int32(C_KSTG_B)
            ds_st = fx.Int32(C_DS_OFF) + us * fx.Int32(dsstg_b)
        else:
            k_st = fx.Int32(stage * C_KSTG_B)
            ds_st = fx.Int32(C_DS_OFF + stage * dsstg_b)
        # K role: rows [64u + 32hh, +32), all 192 columns
        kv0 = u * fx.Int32(64) + hh * fx.Int32(KV32)
        gel = (fx.Int64(bat * Skv + kv0) * fx.Int64(Hkv) + fx.Int64(hkv)) * fx.Int64(D_QK)
        k_lds = k_st + hh * fx.Int32(C_KHALF_B)
        k_ga = ga_k + gel * fx.Int64(2)
        k_dim1 = fx.max(Skv - kv0, fx.Int32(0))
        # dS role: kv32 row 2u + hh x the workgroup's stored q pairs [v0, qp0g + nb_g)
        r = u * fx.Int32(2) + hh
        slot = ds_st + hh * fx.Int32(slot_b)
        f2r = is_causal.select(fx.max(r - mshift, fx.Int32(0)), fx.Int32(0))  # qp_first(2r)
        nv = qp0g + fx.Int32(nb_g) - fx.max(qp0g, f2r)
        t = fx.Int64(is_causal.select(fx.max(r - mshift - fx.Int32(1), fx.Int32(0)), fx.Int32(0)))
        row_off = fx.Int64(r) * fx.Int64(2) * fx.Int64(nqp) - t * (t + fx.Int64(1))
        src = (
            fx.Int64(hb) * fx.Int64(nblk) + row_off + fx.Int64(fx.max(qp0g - f2r, fx.Int32(0)))
        ) * fx.Int64(DS_BLK_EL)
        ds_stride = fx.Int64(nqp - f2r) * fx.Int64(DS_BLK_EL)
        ds_lds = slot + (fx.Int32(nb_g) - nv) * fx.Int32(DSI_BLK_B)
        ds_ga = ga_ds + src * fx.Int64(2)
        if real:
            sel = dsw
            lds_off = sel.select(ds_lds, k_lds)
            ga = sel.select(ds_ga, k_ga)
            w0 = sel.select(_c32(DS_W0), _c32(BK_W0))
            tdim1 = sel.select(fx.Int32(2), k_dim1)
            tile0 = sel.select(nv * fx.Int32(DS_BLK_EL), fx.Int32(D_QK))
            tile1 = sel.select(fx.Int32(2), fx.Int32(KV32))
            stride = sel.select(ds_stride, kst)
            return desc(lds_off, ga, w0, tdim1, tile0, tile1, stride)
        live = u < nu
        go = live & (nv >= fx.Int32(1))
        # the K and dS forms, each with its dummy
        k_lds_s = k_lds
        k_ga_s = live.select(k_ga, ga_dummy)
        k_dim1_s = live.select(k_dim1, fx.Int32(1))
        k_t0 = live.select(fx.Int32(D_QK), fx.Int32(8))
        k_t1 = live.select(fx.Int32(KV32), fx.Int32(1))
        k_str = live.select(kst, fx.Int64(0))
        d_lds = go.select(ds_lds, slot)
        d_ga = go.select(ds_ga, ga_dummy)
        d_dim1 = go.select(fx.Int32(2), fx.Int32(1))
        d_t0 = go.select(nv * fx.Int32(DS_BLK_EL), fx.Int32(8))
        d_t1 = go.select(fx.Int32(2), fx.Int32(1))
        d_str = go.select(ds_stride, fx.Int64(0))
        sel = dsw
        return desc(
            sel.select(d_lds, k_lds_s),
            sel.select(d_ga, k_ga_s),
            sel.select(_c32(DS_W0), _c32(BK_W0)),
            sel.select(d_dim1, k_dim1_s),
            sel.select(d_t0, k_t0),
            sel.select(d_t1, k_t1),
            sel.select(d_str, k_str),
        )

    def issue(d):
        g0, g1 = d
        _tdm_issue(_vec_i32(g0), _vec_i32(g1))

    # ---- operand bursts: (K^T of the 12 d tiles, dS^T of the wave's nqw sub-tiles)
    def burst(u, h):
        """The burst of kv32 row 2u + h (u traced, h a Python int)."""
        us = u % fx.Int32(3)
        kb = lds0 + us * fx.Int32(C_KSTG_B) + fx.Int32(h * C_KHALF_B) + k_lane
        kt = [
            _cat(_tr16(kb + fx.Int32(_k_imm(dt, 0))), _tr16(kb + fx.Int32(_k_imm(dt, 1))))
            for dt in range(NDT)
        ]
        r = u * fx.Int32(2) + fx.Int32(h)
        f2r = is_causal.select(fx.max(r - mshift, fx.Int32(0)), fx.Int32(0))
        nv = qp0g + fx.Int32(nb_g) - fx.max(qp0g, f2r)
        slot = fx.Int32(C_DS_OFF) + us * fx.Int32(dsstg_b) + fx.Int32(h * slot_b)
        d0 = lds0 + slot + ds_lane
        d1 = d0 + fx.max(nv, fx.Int32(0)) * fx.Int32(DSI_BLK_B)  # row-1 images (the op landed right-aligned)
        ds = []
        for s in range(nqw):
            po = wp_off[s // 2] + fx.Int32(_ds_half(s % 2))
            ds.append(_cat(_tr16(d0 + po), _tr16(d1 + po)))
        return kt, ds

    # steady-unit bursts: precomputed bases, every address a base VGPR + immediate; K stages 0 / 1 off kb01,
    # stage 2 off kb2; dS stage s off db[s] (the wave's pairs 0 and 2 = pair 0 + 8 blocks) and db1[s] (pair 1)
    kb01 = lds0 + k_lane
    kb2 = kb01 + fx.Int32(2 * C_KSTG_B)
    dl = lds0 + ds_lane + wp_off[0]
    db = [dl + fx.Int32(C_DS_OFF + s * dsstg_b) for s in range(3)]
    if nqw > 2:
        d1o = wp_off[1] - wp_off[0]
        db1 = [db[s] + d1o for s in range(3)]

    def burst_stage(s, h):
        """The burst of kv32 row 2u + h of a steady unit u with u % 3 == s (all pairs stored)."""
        ko = (0 if s == 2 else s * C_KSTG_B) + h * C_KHALF_B
        kb = kb2 if s == 2 else kb01
        kt = [
            _cat(_tr16(kb + fx.Int32(ko + _k_imm(dt, 0))), _tr16(kb + fx.Int32(ko + _k_imm(dt, 1))))
            for dt in range(NDT)
        ]
        so = h * slot_b
        r1 = nb_g * DSI_BLK_B
        ds = []
        for s2 in range(nqw):
            k = s2 // 2
            assert k != 2 or (4 * 2 + 0) - 0 == 8  # pair 2 = pair 0 + 8 blocks
            base = db1[s] if k == 1 else db[s]
            po = (8 * DSI_BLK_B if k == 2 else 0) + _ds_half(s2 % 2)
            ds.append(_cat(_tr16(base + fx.Int32(so + po)), _tr16(base + fx.Int32(so + r1 + po))))
        return kt, ds

    # ---- WMMA runs: acc[s * NDT + dt] += K^T[dt] dS^T[s]
    def run_wmma(acc, S, lo, hi):
        for i in range(lo, hi):
            s, dt = divmod(i, NDT)
            acc[i] = _wmma(S[0][dt], S[1][s], acc[i])

    def pair_live(k, r):
        """The wave's k-th q pair has an attended entry at kv32 row r (wave-uniform)."""
        f = r - mshift
        qpf = (is_causal & (f > fx.Int32(0))).select(f, fx.Int32(0))
        return (r < n32) & ((qp0g + wave_pair(k)) >= qpf)

    def run_part(acc, S, mask, r, lo, hi):
        if mask:
            for k in range(nqw // 2):
                a = max(2 * k * NDT, lo)
                bb = min(2 * (k + 1) * NDT, hi)
                if a < bb:
                    idx = list(range(a, bb))
                    new = _wmma_if(
                        pair_live(k, r),
                        [acc[i] for i in idx],
                        [S[0][i % NDT] for i in idx],
                        [S[1][i // NDT] for i in idx],
                    )
                    for i, v in zip(idx, new):
                        acc[i] = v
            rocdl.sched_barrier(0)
            return
        run_wmma(acc, S, lo, hi)
        rocdl.sched_barrier(0)

    # ---- state threaded through the unit loops: acc, then the X operand set (kt, ds)
    def pack(acc, X):
        return [_ir(v) for v in acc] + [_ir(v) for v in X[0]] + [_ir(v) for v in X[1]]

    def unpack(st):
        st = list(st)
        acc = [fx.Vector(v) for v in st[:nacc]]
        kt = [fx.Vector(v) for v in st[nacc : nacc + NDT]]
        ds = [fx.Vector(v) for v in st[nacc + NDT :]]
        return acc, (kt, ds)

    def steady_unit(st, u, s):
        """Main-tile steady unit u (u % 3 == s), all ops real: region A = burst Y (row 2u + 1) under the first
        SPLIT_AT WMMAs of run X; split barrier; op(u + 3); region C = burst X (row 2u + 2) under run Y."""
        acc, X = unpack(st)
        ns = min(SPLIT_AT, nacc)
        Y = burst_stage(s, 1)
        dn = c_desc(u + fx.Int32(3), True, stage=s)
        run_wmma(acc, X, 0, ns)
        _sgb_interleave(IL_LEAD_A, 1, nt_burst, ns)
        rocdl.sched_barrier(0)
        _keep(X[1])
        tdm_ops.tensor_wait(C_TW)
        rocdl.s_wait_dscnt(0)
        _split_arrive()  # P_{u+1}: op(u + 1) landed, unit u's ring reads retired
        rocdl.sched_barrier(0)
        run_wmma(acc, X, ns, nacc)
        rocdl.sched_barrier(0)
        _split_wait()  # unit u + 1 published, unit u's stages free
        rocdl.sched_barrier(0)
        issue(dn)  # op(u + 3) into unit u's stages
        rocdl.sched_barrier(0)
        X2 = burst_stage((s + 1) % 3, 0)
        run_wmma(acc, Y, 0, nacc)
        _sgb_interleave(IL_LEAD_C, 1, nt_burst, nacc)
        rocdl.sched_barrier(0)
        _keep(Y[1])
        return pack(acc, X2)

    def generic_unit(st, u, mask, real):
        """Head-tile steady units and the masked units: split barrier only."""
        acc, X = unpack(st)
        ns = SPLIT_AT * nacc // (MAIN_NQW * NDT)
        r0 = u * fx.Int32(2)
        Y = burst(u, 1)
        rocdl.sched_barrier(0)
        dn = c_desc(u + fx.Int32(3), real)
        run_part(acc, X, mask, r0, 0, ns)
        tdm_ops.tensor_wait(C_TW)
        rocdl.s_wait_dscnt(0)
        _split_arrive()
        rocdl.sched_barrier(0)
        run_part(acc, X, mask, r0, ns, nacc)
        _split_wait()
        rocdl.sched_barrier(0)
        issue(dn)
        rocdl.sched_barrier(0)
        X2 = burst(u + fx.Int32(1), 0)
        rocdl.sched_barrier(0)
        run_part(acc, Y, mask, r0 + fx.Int32(1), 0, nacc)
        return pack(acc, X2)

    @flyc.jit
    def loop_steady(state, n3):
        final = state
        for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n3), 1, init=state):
            u = fx.Int32(it) * fx.Int32(STEADY_UNROLL)
            st = list(carried)
            for j in range_constexpr(STEADY_UNROLL):
                st = steady_unit(st, u + fx.Int32(j), j % C_KDEPTH)
            final = yield st
        return final

    @flyc.jit
    def steady_tail(state, u0, nfs, s):
        res = state
        if u0 < nfs:
            res = steady_unit(list(state), u0, s)
        return res

    @flyc.jit
    def loop_generic(state, lo, hi, mask, real):
        final = state
        for it, carried in range(fx.Index(lo), fx.Index(hi), 1, init=state):
            final = yield generic_unit(list(carried), fx.Int32(it), mask, real)
        return final

    # ---- prologue: ops 0, 1; P_0; op 2; burst of kv32 row 0
    acc0 = [fx.Vector.filled(8, 0.0, fx.Float32) for _ in range(nacc)]
    issue(c_desc(fx.Int32(0), False))
    issue(c_desc(fx.Int32(1), False))
    rocdl.sched_barrier(0)
    tdm_ops.tensor_wait(C_TW)
    _wg_sync()
    issue(c_desc(fx.Int32(2), False))
    rocdl.sched_barrier(0)
    X0 = burst(fx.Int32(0), 0)
    rocdl.sched_barrier(0)
    state = pack(acc0, X0)

    u = fx.Int32(0)
    if nqw == MAIN_NQW:
        # the unrolled steady units burst row 2u + 2 assuming all of its pairs are stored; when (Skv - Sq) % 64
        # == 32 the last steady unit's next row starts mid-row: it runs as a generic unit
        nfs = (((cshift & fx.Int32(63)) == fx.Int32(32)) & (nf > fx.Int32(0))).select(nf - fx.Int32(1), nf)
        n3 = nfs // fx.Int32(STEADY_UNROLL)
        state = list(loop_steady(state, n3))
        u = n3 * fx.Int32(STEADY_UNROLL)
        for j in range(STEADY_UNROLL - 1):
            state = list(steady_tail(state, u + fx.Int32(j), nfs, j % C_KDEPTH))
        u = nfs
    state = list(loop_generic(state, u, nf, False, True))
    state = list(loop_generic(state, nf, nloop, True, False))
    # the last unit: no barrier, no op, no burst past the end
    acc, X = unpack(state)
    Y = burst(nloop, 1)
    rocdl.sched_barrier(0)
    r0 = nloop * fx.Int32(2)
    run_part(acc, X, True, r0, 0, nacc)
    run_part(acc, Y, True, r0 + fx.Int32(1), 0, nacc)

    # ---- exit: every wave's TDM landed (dummies included), every ring read retired
    tdm_ops.tensor_wait(0)
    _wg_sync()
    rsrc = create_buffer_resource(DQ, num_records_bytes=dq_bytes)
    rocdl.sched_barrier(0)
    # dQ epilogue per 16-row sub-tile: 12 dQ^T fragments (lane = q row, 8 consecutive d) -> this wave's row-major
    # image after the rings -> whole 128-B lines back
    img = lds0 + fx.Int32(LDS_C) + wv * fx.Int32(EPI_IMG_B)
    w_lane = img + (lane % fx.Int32(16)) * fx.Int32(DQ_IMG_ROW_B) + (lane // fx.Int32(16)) * fx.Int32(16)
    r_lane = img + (lane // fx.Int32(8)) * fx.Int32(DQ_IMG_ROW_B) + (lane % fx.Int32(8)) * fx.Int32(16)
    v8b = fx.Vector.make_type(8, fx.BFloat16)
    for s in range(nqw):
        for dt in range(NDT):
            llvm.store(_ir(acc[s * NDT + dt].to(fx.BFloat16)), lds_ptr(w_lane + fx.Int32(dt * 32)))
        qs = q0g + wave_pair(s // 2) * fx.Int32(32) + fx.Int32(16 * (s % 2))
        for rb in range(4):
            row = (bat * Sq + qs + fx.Int32(rb * 4) + lane // fx.Int32(8)) * Hq + qh
            for li in range(D_QK // DQ_LINE_EL):
                vv = fx.Vector(llvm.load(v8b, lds_ptr(r_lane + fx.Int32(rb * 4 * DQ_IMG_ROW_B + li * 128))))
                vec8 = row * fx.Int32(DV8_QK) + lane % fx.Int32(8) + fx.Int32(li * (DQ_LINE_EL // 8))
                buffer_store(_ir(vv), rsrc, vec8 * fx.Int32(16), offset_is_bytes=True)


@flyc.kernel(name="k_mla_bwd_dqds", known_block_size=[BLOCK, 1, 1])
def k_dqds(
    DS: fx.Int64,
    K: fx.Tensor,
    DQ: fx.Tensor,
    Sq: fx.Int32,
    Skv: fx.Int32,
    Hq: fx.Int32,
    Hkv: fx.Int32,
    G: fx.Int32,
    nkvt: fx.Int32,
    cshift: fx.Int32,
    causal: fx.Int32,
    q_off: fx.Int32,
    ntile: fx.Int32,
    HG: fx.Int32,
    nblk: fx.Int32,
    dq_bytes: fx.Int32,
):
    rocdl.disable_xdl_arb_stall()  # no WMMA multicycle arbitration stall (SCHED_MODE)
    lds0 = fx.Int32(fx.ptrtoint(fx.SharedAllocator().allocate(LDS_BYTES).peek().ptr))
    gy = fx.Int32(fx.block_idx.y)
    args = (DS, K, DQ, Sq, Skv, Hq, Hkv, G, nkvt, cshift, causal)
    if gy < ntile:
        # the 384-query tiles of [q_off, Sq), longest first
        _dqds_body(
            MAIN_NQW,
            *args,
            q_off + (ntile - fx.Int32(1) - gy) * fx.Int32(MAIN_ROWS),
            HG,
            nblk,
            dq_bytes,
            lds0,
        )
    else:
        # the head [0, q_off) = [0, 256)
        _dqds_body(HEAD_NQW, *args, q_off - fx.Int32(HEAD_ROWS), HG, nblk, dq_bytes, lds0)


@flyc.jit
def launch_dqds(
    DS: fx.Int64,
    K,
    DQ,
    Sq: fx.Int32,
    Skv: fx.Int32,
    Hq: fx.Int32,
    Hkv: fx.Int32,
    G: fx.Int32,
    nkvt: fx.Int32,
    cshift: fx.Int32,
    causal: fx.Int32,
    q_off: fx.Int32,
    ntile: fx.Int32,
    HG: fx.Int32,
    nblk: fx.Int32,
    dq_bytes: fx.Int32,
    ny: fx.Int32,
    ngz: fx.Int32,
    stream: fx.Stream,
):
    _ = SOURCE_DIGEST  # FlyDSL disk-cache key
    # grid = (HG q heads of a head group, ntile main tiles + the head tile, B*Hq/HG head groups)
    k_dqds(DS, K, DQ, Sq, Skv, Hq, Hkv, G, nkvt, cshift, causal, q_off, ntile, HG, nblk, dq_bytes).launch(
        grid=(HG, ny, ngz), block=(BLOCK, 1, 1), stream=stream
    )


# Expert scheduling mode: the hardware dependency interlocks off, LLVM inserts the dependency-counter waits (a DS
# read then does not wait for earlier WMMAs).
launch_dqds.compile_hints["llvm_options"] = {"amdgpu-expert-scheduling-mode": True}


# ------------------------------------------------------------------------------------------------ host entry
HEAD_GROUP = 32  # head-group target of the grid (bwd._head_group)
_MAX_BH_S = 512 * 4096  # largest batch * heads * seqlen the 32-bit offsets are checked for


def dq_split(sq):
    """(q_off, nhead, ntile): 384-query tiles over [q_off, sq), one merged 256-query head tile over [0, q_off)."""
    q_off = sq % MAIN_ROWS
    return q_off, int(q_off > 0), (sq - q_off) // MAIN_ROWS


def dqds_unsupported_reason(b, sq, skv, hq, hkv, causal=True):
    """Why k_dqds cannot take these sizes, or None."""
    if not causal:
        return "causal only"
    if b <= 0 or hq <= 0 or hkv <= 0 or hq % hkv:
        return f"heads must be positive and hq % hkv == 0, got {hq} / {hkv}"
    if sq <= 0 or skv < sq:
        return f"needs 0 < seqlen_q <= seqlen_kv, got {sq} / {skv}"
    if sq % 32 or skv % 32:
        return f"seqlen_q / seqlen_kv must be multiples of 32, got {sq} / {skv}"
    if sq % MAIN_ROWS not in (0, HEAD_ROWS):
        return f"seqlen_q % {MAIN_ROWS} must be 0 or {HEAD_ROWS}, got {sq}"
    if b * hq * skv > _MAX_BH_S or b * sq * hq * D_QK * 2 >= 1 << 31:
        return "too large for the 32-bit offsets"
    return None
