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

"""dK / dV of the gfx1250 MLA attention backward over 96-row kv blocks, plus the dS^T scratch (k_dkdv96ds).

The FlyDSL form of the HipKittens kernel k_dkdv96ds (csrc/kernels/attention/hipkittens/gfx1250_mla): grid
(HG, nblk(Skv), B*HC/HG) over the heads [H0, H0 + HC) (the host launches the heads in chunks to bound the scratch),
block 64 = 2 waves (one per SIMD). Blocks [0, n96) are 96-row kv blocks (wave w owns the 48 kv rows from
96 bid + 48 w: NKV 3 sub-tiles of 16), the 0..2 blocks after them 64-row tail blocks (NKV 2), two bodies behind a
uniform branch. A workgroup runs one sequence of query tiles (q pairs of 32 queries; MHA, the
kernel is specialized for G = Hq / Hkv = 1) over its block: first the n_mask tiles that straddle the diagonal
(causal mask on), then the n_full tiles below it. Q / dO tiles go through a 3-stage LDS ring filled by the TDM (a
stage = [32][D_V] dO + [32][D_QK] Q, each wave moving its 16 rows); K / V of the wave's rows stay in registers (the
S / dP B operands); dV / dK accumulate in registers and leave through per-wave row-major images in LDS (whole-line
global stores).

Every bf16 dS^T tile the kernel feeds its dK WMMA is also stored into the dS scratch (the HipKittens dslayout.h,
version 2): one region per (batch, q head) of the launch, inside it the kv16 rows in order, row kt holding the q
pairs from qp_first(kt) = max(0, kt // 2 - (Skv - Sq) // 32) on as 1 KiB blocks [16 kv][32 q] in the dK operand's
chunk order, with the softmax scale inside dS. k_dqds reads it back for dQ = dS K.

Numerics: P = exp2(c1 S + NL), c1 = scale * log2(e), dS = P (scale dP + ND), with k_delta_bshd's NL = -log2(e) lse
and ND = -scale delta, in the order of the HipKittens k_dkdv96ds, so dK and dV are bitwise that kernel's.

Ring protocol (one barrier per tile; tile j lives in stage sigma(j) = (j - n_mask) mod 3; every prefetch is clamped
to the last tile T_{N-1}):
  prologue: TDM T_0, T_1 -> tensorcnt 3 -> BARRIER -> readback of sigma(0) (NKV 3: hh0 only) -> NKV 2: TDM T_2 |
    NKV 3: T_2 becomes the pending prefetch.
  NKV 3 tile j: readback hh1 of cur, then the pending TDM (T_{j+2}); S / dP; tr16 dO^T, Q^T of cur; prep of T_{j+3}
    -> cur (pending, issued in the next tile); tensorcnt 3; BARRIER (split: arrive, 24 dK WMMAs, wait); readback hh0
    of the next stage.
  NKV 2 tile j: S / dP; tr16 of cur; dV / dK kh0; tensorcnt 3; BARRIER; readback of the next stage; TDM of T_{j+3}
    into cur; dV / dK kh1.
  exit: tensorcnt 0 -> BARRIER -> the epilogue images overwrite the ring.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx

# UNSTABLE(gfx1250): WMMA, ds_load_tr16, the raw TDM load and tensor_wait, the split barrier, sched barriers,
# readfirstlane, raw buffer loads / stores and disable_xdl_arb_stall have no stable API in FlyDSL 0.3.4.1.
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr.rocdl import tdm_ops

from primus_turbo.flydsl.utils.buffer_ops import (
    buffer_store,
    create_buffer_resource,
    get_element_ptr,
)
from primus_turbo.flydsl.utils.prims import LOG2E

from . import SOURCE_DIGEST
from .fwd_staging import lds_ptr

D_QK = 192
D_V = 128
NDT_QK = D_QK // 32  # 6 WMMA k-steps of S = Q K^T
NDT_V = D_V // 32  # 4 of dP = dO V^T
NDO_V = D_V // 16  # 8 dV^T output tiles
NDO_QK = D_QK // 16  # 12 dK^T output tiles
NQE = 2 * NDT_QK  # 12 Q readback entries per hh
NHB = NQE + 2 * NDT_V  # 20 entries per hh
NRB = 2 * NHB  # 40 entries of a stage
NLD = 8  # NL / ND vec4 entries
XK = D_QK * 2 + 16  # 400 B: padded LDS row of a Q tile / dK image
XV = D_V * 2 + 16  # 272 B: padded LDS row of a dO tile / dV image
QOFF = 32 * XV  # the Q tile inside a ring stage
QDO_B = 32 * (XV + XK)  # one ring stage
DEPTH = 3
RING_B = DEPTH * QDO_B  # 64512: the kernel's LDS (the epilogue images reuse it)
NW = 2  # waves per workgroup
TDM_ROWS = 16  # rows per wave per TDM op
TW = 3  # tensorcnt that retires all but the newest tile's 3 ops
NEG = -3.0e38  # masked logit: exp2 -> +0
NKV3 = 3
ROWS3 = 32 * NKV3  # 96
KH0_SALU = 2  # SALU per WMMA of a TDM-prep run
SB_ALU = 0x002 | 0x004 | 0x400  # sched_barrier: VALU, SALU and transcendental may cross
# Head-group target of the grid (x = the heads of a group, fastest): 8 = the XCD count, so under the round-robin
# dispatch each XCD runs one head's kv blocks at a time and that head's Q / dO tiles stay in its L2.
HEAD_GROUP = 8
# The 96-row body below was chosen by measurement (MI455X, b2 s4096 h128): hh0's S / dP chains run k-step by k-step
# with hh1's WMMAs interleaved into hh0's softmax, emitted stage by stage; the 40 dV / dK A-operand tr16 loads issue
# under hh1's softmax into the dead S / dP operand registers (a load into a register an in-flight WMMA still reads
# costs a va_vdst drain in expert scheduling mode); the ring barrier waits after dK sub-tile 1, so the next stage's
# hh0 readback runs under sub-tile 2. The HipKittens schedule and its generator's folded numerics measured slower.
DS_TH = 1  # TH_STORE_NT for the dS^T stores: written once and read once by k_dqds, they need not stay in L2
# dS scratch layout (HipKittens dslayout.h, version 2)
DSL_KV = 16
DSL_Q = 32
DSL_BLK_B = 1024
DS_ST_B = 512
# TDM ops of a ring stage: 0 dO [32][128], 1 Q cols [0, 128), 2 Q cols [128, 192)
TDM_W = (128, 128, 64)
TDM_D = (D_V, D_QK, D_QK)
TDM_C0 = (0, 0, 128)
TDM_XROW = (XV, XK, XK)
assert NW * 16 * NKV3 * (XV + XK) <= RING_B, "the epilogue images must fit in the ring"


def _ilog2(x):
    return 0 if x <= 1 else 1 + _ilog2(x // 2)


def _tdm_w0(op):
    """Group 1 word 0: data size 2 B, pad enable, pad interval log2(w / 2 dwords) - 1, pad amount pad / 2 dwords - 1."""
    w = TDM_W[op]
    pad = (TDM_XROW[op] - 2 * w) // 2
    return (1 << 16) | (1 << 20) | ((_ilog2(w // 2) - 1) << 22) | ((pad // 2 - 1) << 25)


assert tuple(_tdm_w0(op) for op in range(3)) == (0x07510000, 0x47510000, 0x87110000)


def _i32(x):
    """Python int -> i32 constant (two's complement for [2**31, 2**32))."""
    if x >= 1 << 31:
        x -= 1 << 32
    return fx.Int32(x)


def _ir(v):
    return fx.as_ir_value(v)


def _exp2(x):
    return fx.math.exp2(x, fastmath=arith.FastMathFlags.afn)


def _sel(c, a, b):
    return c.select(a, b)


def _min(a, b):
    return _sel(a < b, a, b)


def _max(a, b):
    return _sel(a > b, a, b)


def _uniform(x):
    return fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, _ir(x)))


def _lds_fence(release):
    """Workgroup fence on LDS only (MMRA amdgpu-synchronize-as = local, as gpu.barrier's LDS form): the release
    waits for this wave's DS ops (s_wait_dscnt 0), not for its in-flight global loads / stores."""
    op = llvm.FenceOp(
        llvm.AtomicOrdering.release if release else llvm.AtomicOrdering.acquire, syncscope="workgroup"
    )
    op.operation.attributes["llvm.mmra"] = ir.Attribute.parse(
        '#llvm.mmra_tag<"amdgpu-synchronize-as":"local">'
    )


def _lds_arrive():
    """Split LDS barrier, arrive: release fence on LDS, then s_barrier_signal."""
    _lds_fence(True)
    rocdl.s_barrier_signal(-1)


def _lds_wait():
    rocdl.s_barrier_wait(-1)
    _lds_fence(False)


def _lds_barrier():
    _lds_arrive()
    _lds_wait()


# ------------------------------------------------------------------------------------- index math (dkdv_idx.h)
def _clampqt(t, nqt2):
    t = _sel(t < nqt2, t, nqt2 - fx.Int32(1))
    return _sel(t < fx.Int32(0), fx.Int32(0), t)


def _ncur(cur):
    return _sel(cur == fx.Int32(2 * QDO_B), fx.Int32(0), cur + fx.Int32(QDO_B))


def _qp_start(kv0g, cshift, causal):
    c = kv0g - cshift
    qs = _sel(c < fx.Int32(0), fx.Int32(0), c) // fx.Int32(32)
    return _sel(causal != fx.Int32(0), qs, fx.Int32(0))


def _nmaskp(kv0g, rows, cshift, causal, nqt):
    nqt2 = nqt // fx.Int32(2)
    qs = _qp_start(kv0g, cshift, causal)
    u = kv0g + fx.Int32(rows - 1) - cshift
    qsf = _sel(u < fx.Int32(0), fx.Int32(0), (u + fx.Int32(31)) // fx.Int32(32))
    qsf = _min(qsf, nqt2)
    nm = _max(qsf - qs, fx.Int32(0))
    nm = _min(nm, nqt2 - qs)
    return _sel(causal != fx.Int32(0), nm, fx.Int32(0))


def _n96(Skv):
    n = Skv // fx.Int32(ROWS3)
    r = Skv - n * fx.Int32(ROWS3)
    return _sel(r == fx.Int32(32), n - fx.Int32(1), n)


# dS scratch row arithmetic (branch-free forms of dslayout.h; m = cshift / 32)
def _tri(k):
    """sum_{j < k} floor(j / 2), k >= 0 (i32: block counts)."""
    p = k >> fx.Int32(1)
    return p * (p - fx.Int32(1)) + (k & fx.Int32(1)) * p


def _skip(kt, m, causal):
    """Blocks the causal packing removes from kv16 rows 0 .. kt - 1 (m = cshift / 32 >= 0: skv >= sq)."""
    r = kt - fx.Int32(2) * m
    return _sel(causal != fx.Int32(0), _tri(_sel(r > fx.Int32(0), r, fx.Int32(0))), fx.Int32(0))


def _qpf_kt(kt, m, causal):
    f = (kt >> fx.Int32(1)) - m
    return _sel((causal != fx.Int32(0)) & (f > fx.Int32(0)), f, fx.Int32(0))


def _row_off(kt, nqp, m, causal):
    """Blocks before kv16 row kt inside one (batch, q head) region (i32)."""
    return kt * nqp - _skip(kt, m, causal)


def ds_blocks_per_head(sq, skv, causal=True):
    """DSL_BLK_B blocks of one (batch, q head) region of the dS^T scratch (host integers, skv >= sq)."""
    nqp, nkt, m = sq // DSL_Q, skv // DSL_KV, (skv - sq) // DSL_Q
    # under the mask, kv16 row k >= 2m (seqlen_kv >= seqlen_q: m >= 0) skips its first k // 2 - m q pairs
    r = max(nkt - 2 * m, 0)
    skip = (r // 2) * (r // 2 - 1) + (r % 2) * (r // 2) if causal else 0
    return nkt * nqp - skip


# ------------------------------------------------------------------------------------------------------ kernel
def _dkdv96_impl(
    Q, K, V, DO, NL, ND, DV_, DK, DS, scale, Sq, Skv, Hq, Hkv, G, nqt, cshift, causal, B_, HG, ds_mode, H0, HC
):
    rocdl.disable_xdl_arb_stall()  # SCHED_MODE.DISABLE_XDL_ARB_STALL
    v8b = fx.Vector.make_type(8, fx.BFloat16)
    v8f = fx.Vector.make_type(8, fx.Float32)
    v4f = fx.Vector.make_type(4, fx.Float32)
    smem = fx.SharedAllocator().allocate(RING_B)
    lds0 = fx.Int32(fx.ptrtoint(smem.peek().ptr))
    tid = fx.Int32(fx.thread_idx.x)
    lane = tid & fx.Int32(31)
    wv = _uniform(tid) >> fx.Int32(5)
    half = lane >> fx.Int32(4)
    row = lane & fx.Int32(15)
    # The launch covers the heads [H0, H0 + HC) of every batch; hb = bat * HC + (head - H0) is the dS region of the
    # (batch, head) in the launch's scratch.
    hb = fx.Int32(fx.block_idx.z) * HG + fx.Int32(fx.block_idx.x)
    bat = hb // HC
    hkv = H0 + hb - bat * HC
    bid = fx.Int32(fx.block_idx.y)
    n96 = _n96(Skv)
    nqt2 = nqt // fx.Int32(2)
    c1 = scale * fx.Float32(LOG2E)

    # true byte extents: an access past them is dropped (a load returns 0)
    nk_b = B_ * Skv * Hkv * fx.Int32(D_QK * 2)
    nv_b = B_ * Skv * Hkv * fx.Int32(D_V * 2)
    nl_b = B_ * Hq * Sq * fx.Int32(4)
    r_k = create_buffer_resource(K, num_records_bytes=nk_b)
    r_v = create_buffer_resource(V, num_records_bytes=nv_b)
    r_nl = create_buffer_resource(NL, num_records_bytes=nl_b)
    r_nd = create_buffer_resource(ND, num_records_bytes=nl_b)
    rs_k = Hkv * fx.Int32(D_QK // 8)  # vec8 between consecutive k rows
    rs_v = Hkv * fx.Int32(D_V // 8)
    kbase = bat * Skv * rs_k + hkv * fx.Int32(D_QK // 8)
    vbase = bat * Skv * rs_v + hkv * fx.Int32(D_V // 8)

    # ---- Q / dO ring TDM descriptors: group 1 per op (loop-invariant), per-op LDS / global bases
    qh0 = hkv  # MHA (G == 1): the q head of the kv head; the host rejects G > 1
    row0 = fx.Int64(bat * Sq) * fx.Int64(Hq) + fx.Int64(qh0)  # element row of (bat, q 0, qh0)
    g1, ldsb, gb, gsq = [], [], [], []
    for op in range_constexpr(3):
        st = fx.Int64(Hq) * fx.Int64(TDM_D[op])  # row stride (elements)
        g1.append(
            fx.Vector.from_elements(
                [
                    _i32(_tdm_w0(op)),
                    _i32(0xFFFF0000),
                    _i32((TDM_ROWS << 16) | 0x7FFF),
                    _i32(TDM_W[op] << 16),
                    fx.Int32(TDM_ROWS),
                    fx.Int32(st),
                    fx.Int32(st >> fx.Int64(32)) & fx.Int32(0xFFFF),
                    fx.Int32(0),
                ],
                fx.Int32,
            )
        )
        xrow = TDM_XROW[op]
        ldsb.append(
            lds0 + fx.Int32((0 if op == 0 else QOFF) + TDM_C0[op] * 2) + wv * fx.Int32(TDM_ROWS * xrow)
        )
        base = fx.Int64(fx.ptrtoint(fx.get_iter(DO if op == 0 else Q)))
        gel = row0 * fx.Int64(TDM_D[op]) + fx.Int64(TDM_C0[op]) + fx.Int64(wv * fx.Int32(TDM_ROWS)) * st
        gb.append(base + fx.Int64(2) * gel)
        gsq.append(fx.Int64(2 * 32) * st)  # bytes per q pair
    z4 = _ir(fx.Vector.filled(4, 0, fx.Int32))
    z8 = _ir(fx.Vector.filled(8, 0, fx.Int32))

    def tdm_prep(qt, stage_off, live=None):
        """Group 0 of the 3 ops of tile (q pair qt) into the stage at stage_off (this wave's rows); the NULL descriptor
        (count 0) unless `live`."""
        cnt = fx.Int32(1) if live is None else live.select(fx.Int32(1), fx.Int32(0))
        out = []
        for op in range_constexpr(3):
            ga = gb[op] + fx.Int64(qt) * gsq[op]
            out.append(
                _ir(
                    fx.Vector.from_elements(
                        [
                            cnt,
                            ldsb[op] + stage_off,
                            fx.Int32(ga),
                            fx.Int32(ga >> fx.Int64(32)) | _i32(0x80000000),
                        ],
                        fx.Int32,
                    )
                )
            )
        return out

    def tdm_issue(g0s):
        for op in range_constexpr(3):
            rocdl.tensor_load_to_lds(g0s[op], _ir(g1[op]), z4, z4, z8, cache_policy=0)

    def tdm_tile(qt, stage_off, live=None):
        tdm_issue(tdm_prep(qt, stage_off, live))

    # ---- NL / ND: 8 vec4 loads of (q pair qt); voffset 32 half + constant, soffset uniform
    voff_l = half * fx.Int32(32)
    nl_base = (bat * Hq + qh0) * Sq * fx.Int32(4)

    def ldl(qt):
        sb = _uniform(nl_base + qt * fx.Int32(128))
        out = []
        for e in range_constexpr(NLD):
            rs_ = r_nl if (e % 4) // 2 == 0 else r_nd
            vo = voff_l + fx.Int32((e // 4) * 64 + (e % 2) * 16)
            out.append(rocdl.raw_ptr_buffer_load(v4f, rs_, _ir(vo), _ir(sb)))
        return out

    # ---- readback of a stage's S / dP A operands (Q, dO rows): entry hh * 20 + (Q dt * 2 + u | dO 12 + dt * 2 + u)
    rd_lane_q = row * fx.Int32(XK) + half * fx.Int32(16)
    rd_lane_v = row * fx.Int32(XV) + half * fx.Int32(16)

    def rd_base(stage_off):
        return (lds_ptr(lds0 + stage_off + rd_lane_q), lds_ptr(lds0 + stage_off + rd_lane_v))

    def _ld(base, off):
        p = base if off == 0 else get_element_ptr(base, static_byte_offset=off)
        return llvm.load(v8b, p)

    def rdqd_hh(rb, hh):
        out = [None] * NHB
        for dt in range_constexpr(NDT_QK):
            for u in range_constexpr(2):
                out[dt * 2 + u] = _ld(rb[0], QOFF + hh * 16 * XK + dt * 64 + u * 32)
        for dt in range_constexpr(NDT_V):
            for u in range_constexpr(2):
                out[NQE + dt * 2 + u] = _ld(rb[1], hh * 16 * XV + dt * 64 + u * 32)
        return out

    # ---- dV / dK A operands dO^T / Q^T (ds_load_tr16_b128): pieces p = 0, 1 (rows + 16 p) form a v16 operand
    lane_r = half * fx.Int32(8) + (lane & fx.Int32(7))
    lane_c = ((lane >> fx.Int32(3)) & fx.Int32(1)) * fx.Int32(8)
    tr_lane_v = lane_r * fx.Int32(XV) + lane_c * fx.Int32(2)
    tr_lane_q = lane_r * fx.Int32(XK) + lane_c * fx.Int32(2)

    def tr_base(stage_off, isq):
        return lds_ptr(lds0 + stage_off + (tr_lane_q if isq else tr_lane_v))

    def tr(base, isq, d):
        x = XK if isq else XV
        o0 = (QOFF if isq else 0) + d * 32
        p0 = base if o0 == 0 else get_element_ptr(base, static_byte_offset=o0)
        p1 = get_element_ptr(base, static_byte_offset=o0 + 16 * x)
        a = fx.Vector(rocdl.ds_load_tr16_b128(v8b, p0))
        b = fx.Vector(rocdl.ds_load_tr16_b128(v8b, p1))
        return a.shuffle(b, list(range(16)))

    def cat(a, b):
        return fx.Vector(a).shuffle(fx.Vector(b), list(range(16)))

    def wmma(a, b, c, ra=False, rb=False):
        return rocdl.wmma_f32_16x16x32_bf16(v8f, _ir(a), _ir(b), _ir(c), reuseA=ra, reuseB=rb).result

    zero8 = _ir(fx.Vector.filled(8, 0.0, fx.Float32))

    # ---- dS^T export: DS is the scratch (only its base address is used); the per-wave descriptor (base region bh0,
    # row kt0) is built in the body; ds_rc[kh] row constants; voffset lane * 16
    nqp = Sq // fx.Int32(DSL_Q)
    mm = cshift >> fx.Int32(5)
    nblk_bh = _row_off(Skv // fx.Int32(DSL_KV), nqp, mm, causal)
    vds = lane * fx.Int32(16)

    def body(NKV):
        K3 = NKV == NKV3
        ROWS = 32 * NKV
        NST = NKV * (NDO_V + NDO_QK)
        DK0 = NKV * NDO_V
        NCAR = NHB if K3 else NRB
        kv0g = _sel(bid < n96, bid * fx.Int32(ROWS3), n96 * fx.Int32(ROWS3) + (bid - n96) * fx.Int32(64))
        kv0 = kv0g + wv * fx.Int32(16 * NKV)

        # ---- causal split at workgroup level (MHA: the tile sequence is the q pairs qp_start .. nqt2 - 1)
        qp_start = _qp_start(kv0g, cshift, causal)
        nmaskp = _nmaskp(kv0g, ROWS, cshift, causal, nqt)
        n_mask = nmaskp
        n_full = nqt2 - qp_start - nmaskp
        n_seq = n_mask + n_full
        qt0 = qp_start + nmaskp
        one = fx.Int32(1)

        def clampn(t, n):
            """min(t, n - 1)"""
            return _sel(t < n, t, n - one)

        # ---- prologue: T_0 -> sigma(0), T_1 -> sigma(1); NL/ND of T_0; tensorcnt 3; BARRIER; readback of sigma(0)
        r_ = n_mask % fx.Int32(DEPTH)
        s0 = _sel(r_ == fx.Int32(0), fx.Int32(0), (fx.Int32(DEPTH) - r_) * fx.Int32(QDO_B))
        s1 = _ncur(s0)
        s2 = _ncur(s1)

        # K / V fragments of this wave's 16 NKV kv rows (the S / dP B operands)
        kf, vf = [], []
        for kh in range_constexpr(NKV):
            kr = kbase + (kv0 + fx.Int32(kh * 16) + row) * rs_k + half
            vr = vbase + (kv0 + fx.Int32(kh * 16) + row) * rs_v + half
            kf.append(
                [
                    cat(
                        rocdl.raw_ptr_buffer_load(
                            v8b, r_k, _ir((kr + fx.Int32(dt * 4)) * fx.Int32(16)), _ir(fx.Int32(0))
                        ),
                        rocdl.raw_ptr_buffer_load(
                            v8b,
                            r_k,
                            _ir((kr + fx.Int32(dt * 4 + 2)) * fx.Int32(16)),
                            _ir(fx.Int32(0)),
                        ),
                    )
                    for dt in range_constexpr(NDT_QK)
                ]
            )
            vf.append(
                [
                    cat(
                        rocdl.raw_ptr_buffer_load(
                            v8b, r_v, _ir((vr + fx.Int32(dt * 4)) * fx.Int32(16)), _ir(fx.Int32(0))
                        ),
                        rocdl.raw_ptr_buffer_load(
                            v8b,
                            r_v,
                            _ir((vr + fx.Int32(dt * 4 + 2)) * fx.Int32(16)),
                            _ir(fx.Int32(0)),
                        ),
                    )
                    for dt in range_constexpr(NDT_V)
                ]
            )

        # dS^T export of this wave: descriptor over [region bh0 row kt0, +true extent)
        kt0 = kv0 >> fx.Int32(4)
        ro = [_row_off(kt0 + fx.Int32(kh), nqp, mm, causal) for kh in range_constexpr(NKV + 1)]
        ds_off = (fx.Int64(hb) * fx.Int64(nblk_bh) + fx.Int64(ro[0])) * fx.Int64(DSL_BLK_B)
        ext = (ro[NKV] - ro[0]) * fx.Int32(DSL_BLK_B)  # rows kt0 .. kt0 + NKV - 1
        nrec = fx.Int64(_sel(ds_mode != fx.Int32(0), ext, fx.Int32(0)))
        r_ds = create_buffer_resource(DS, num_records_bytes=nrec, base_byte_offset=ds_off)
        ds_qpf = [_qpf_kt(kt0 + fx.Int32(kh), mm, causal) for kh in range_constexpr(NKV)]
        ds_rc = [(ro[kh] - ro[0] - ds_qpf[kh]) * fx.Int32(DSL_BLK_B) for kh in range_constexpr(NKV)]

        def ds_store(a, so):
            a = fx.Vector(a)
            buffer_store(
                _ir(a.shuffle(a, list(range(8)))),
                r_ds,
                vds,
                cache_modifier=DS_TH,
                soffset_bytes=so,
                offset_is_bytes=True,
            )
            buffer_store(
                _ir(a.shuffle(a, list(range(8, 16)))),
                r_ds,
                vds + fx.Int32(DS_ST_B),
                cache_modifier=DS_TH,
                soffset_bytes=so,
                offset_is_bytes=True,
            )

        # ---- S / dP + softmax / dS of one (hh, kh)
        def sdp_mm(kh, qfr, dfr):
            s = zero8
            p = zero8
            for dt in range_constexpr(NDT_QK):
                s = wmma(qfr[dt], kf[kh][dt], s)
                if const_expr(dt < NDT_V):
                    p = wmma(dfr[dt], vf[kh][dt], p)
            return s, p

        def sdp_one(do_mask, hh, kh, qfr, dfr, pre, q0, ph, dh):
            s, p = sdp_mm(kh, qfr, dfr)
            smx(do_mask, hh, kh, s, p, pre, q0, ph, dh)

        def smx(do_mask, hh, kh, s, p, pre, q0, ph, dh):
            """softmax / dS of one (hh, kh) from its raw S / dP C fragments."""
            qb = q0 + fx.Int32(hh * 16) + half * fx.Int32(8)
            sv, pv = fx.Vector(s), fx.Vector(p)
            kvl = kv0 + fx.Int32(kh * 16) + row
            for si in range_constexpr(8):
                nl = fx.Vector(pre[hh * 4 + si // 4])[si % 4]
                nd = fx.Vector(pre[hh * 4 + 2 + si // 4])[si % 4]
                tt = fx.Float32(fx.fma(sv[si], c1, nl))
                if const_expr(do_mask):
                    tt = (kvl > qb + fx.Int32(si) + cshift).select(fx.Float32(NEG), tt)
                pf = _exp2(tt)
                ph[kh][hh * 8 + si] = pf.to(fx.BFloat16)
                dh[kh][hh * 8 + si] = (pf * fx.Float32(fx.fma(pv[si], scale, nd))).to(fx.BFloat16)

        def smx_bf(do_mask, hh, khs, sp, pre, q0, ph, dh):
            """smx of half hh for the kv sub-tiles khs (sp[kh] = (S, dP) raw C fragments), stage by stage."""
            qb = q0 + fx.Int32(hh * 16) + half * fx.Int32(8)
            tt, pf, dpv = {}, {}, {}

            def dp_stage():
                for kh in khs:
                    pv = fx.Vector(sp[kh][1])
                    for si in range_constexpr(8):
                        nd = fx.Vector(pre[hh * 4 + 2 + si // 4])[si % 4]
                        dpv[kh, si] = fx.Float32(fx.fma(pv[si], scale, nd))

            for kh in khs:
                sv = fx.Vector(sp[kh][0])
                kvl = kv0 + fx.Int32(kh * 16) + row
                for si in range_constexpr(8):
                    nl = fx.Vector(pre[hh * 4 + si // 4])[si % 4]
                    t = fx.Float32(fx.fma(sv[si], c1, nl))
                    if const_expr(do_mask):
                        t = (kvl > qb + fx.Int32(si) + cshift).select(fx.Float32(NEG), t)
                    tt[kh, si] = t
            for kh in khs:
                for si in range_constexpr(8):
                    pf[kh, si] = _exp2(tt[kh, si])
            dp_stage()
            for kh in khs:
                for si in range_constexpr(8):
                    ph[kh][hh * 8 + si] = pf[kh, si].to(fx.BFloat16)
            for kh in khs:
                for si in range_constexpr(8):
                    dh[kh][hh * 8 + si] = (pf[kh, si] * dpv[kh, si]).to(fx.BFloat16)

        def frags(ops):
            qfr = [cat(ops[dt * 2], ops[dt * 2 + 1]) for dt in range_constexpr(NDT_QK)]
            dfr = [cat(ops[NQE + dt * 2], ops[NQE + dt * 2 + 1]) for dt in range_constexpr(NDT_V)]
            return qfr, dfr

        def pack(ph, dh):
            a_pk = [_ir(fx.Vector.from_elements(ph[kh], dtype=fx.BFloat16)) for kh in range_constexpr(NKV)]
            a_dsk = [_ir(fx.Vector.from_elements(dh[kh], dtype=fx.BFloat16)) for kh in range_constexpr(NKV)]
            return a_pk, a_dsk

        def sdp2(do_mask, ops, pre, q0):
            """64-row tail body: both hh halves from the carried 40-entry readback."""
            ph = [[None] * 16 for _ in range_constexpr(NKV)]
            dh = [[None] * 16 for _ in range_constexpr(NKV)]
            for hh in range_constexpr(2):
                qfr, dfr = frags(ops[hh * NHB : (hh + 1) * NHB])
                for kh in range_constexpr(NKV):
                    sdp_one(do_mask, hh, kh, qfr, dfr, pre, q0, ph, dh)
            return pack(ph, dh)

        def sdp3(do_mask, op0, rb1, pre, q0, after_burst, mid):
            """96-row body S / dP + softmax / dS: hh0 from the carried readback op0 (its six WMMA chains k-step by
            k-step); hh1's 20 entries burst-read from rb1 at the top; hh1's WMMAs (kv sub-tile by sub-tile) with hh0's
            softmax interleaved {1 WMMA, 2 VALU, 1 TRANS}; then `mid` (the dV / dK operand loads) and hh1's softmax."""
            ph = [[None] * 16 for _ in range_constexpr(NKV)]
            dh = [[None] * 16 for _ in range_constexpr(NKV)]
            qfr, dfr = frags(op0)
            rocdl.sched_barrier(SB_ALU)
            op1 = rdqd_hh(rb1, 1)
            rocdl.sched_barrier(SB_ALU)
            after_burst()
            # hh0: the 2 NKV chains k-step by k-step (each WMMA's dependence 2 NKV - 1 WMMAs back)
            s0, p0 = [zero8] * NKV, [zero8] * NKV
            for dt in range_constexpr(NDT_QK):
                for kh in range_constexpr(NKV):
                    s0[kh] = wmma(qfr[dt], kf[kh][dt], s0[kh])
                    if const_expr(dt < NDT_V):
                        p0[kh] = wmma(dfr[dt], vf[kh][dt], p0[kh])
            rocdl.sched_barrier(0)
            # hh1's WMMAs (kh-major) with hh0's softmax / dS interleaved: {1 WMMA, 2 VALU, 1 TRANS}
            qfr1, dfr1 = frags(op1)
            mm1 = [sdp_mm(kh, qfr1, dfr1) for kh in range_constexpr(NKV)]
            smx_bf(do_mask, 0, list(range(NKV)), list(zip(s0, p0)), pre, q0, ph, dh)
            nvw = 2 if const_expr(not do_mask) else 4
            for i in range_constexpr(NKV * (NDT_QK + NDT_V)):
                rocdl.sched_group_barrier(0x008, 1, 0)
                rocdl.sched_group_barrier(0x002, nvw, 0)
                rocdl.sched_group_barrier(0x400, 1, 0)
            rocdl.sched_barrier(0)
            mid()
            rocdl.sched_barrier(SB_ALU)
            smx_bf(do_mask, 1, list(range(NKV)), mm1, pre, q0, ph, dh)
            return pack(ph, dh)

        def dv_kh(kh, acc, b_do, a_pk):
            for d in range_constexpr(NDO_V):
                acc[kh * NDO_V + d] = wmma(b_do[d], a_pk[kh], acc[kh * NDO_V + d], rb=(d > 0))

        def dk_rng(kh, d0, d1, acc, b_q, a_dsk):
            for d in range_constexpr(d0, d1):
                acc[DK0 + kh * NDO_QK + d] = wmma(b_q[d], a_dsk[kh], acc[DK0 + kh * NDO_QK + d], rb=(d > 0))

        def dk_kh(kh, acc, b_q, a_dsk):
            dk_rng(kh, 0, NDO_QK, acc, b_q, a_dsk)

        rocdl.sched_barrier(0)
        tdm_tile(_clampqt(qp_start, nqt2), s0)
        tdm_tile(_clampqt(qp_start + clampn(one, n_seq), nqt2), s1, one < n_seq)
        preA = ldl(_clampqt(qp_start, nqt2))
        rocdl.sched_barrier(0)
        tdm_ops.tensor_wait(TW)
        rocdl.sched_barrier(0)
        _lds_barrier()  # RAW: every wave's half of sigma(0) landed
        rocdl.sched_barrier(0)
        t2 = _clampqt(qp_start + clampn(fx.Int32(2), n_seq), nqt2)
        if const_expr(K3):
            qd = rdqd_hh(rd_base(s0), 0)
            rocdl.sched_barrier(0)
            tdp = tdm_prep(t2, s2, fx.Int32(2) < n_seq)  # issued at the top of iteration 0
        else:
            rb0 = rd_base(s0)
            qd = rdqd_hh(rb0, 0) + rdqd_hh(rb0, 1)
            rocdl.sched_barrier(0)
            tdm_tile(t2, s2, fx.Int32(2) < n_seq)
            tdp = []
        rocdl.sched_barrier(0)

        acc0 = [zero8] * NST
        NTD = 3 if K3 else 0

        def unpack(st):
            acc = list(st[:NST])
            qd_ = list(st[NST : NST + NCAR])
            pre = list(st[NST + NCAR : NST + NCAR + NLD])
            tdp_ = list(st[NST + NCAR + NLD : NST + NCAR + NLD + NTD])
            cur = fx.Int32(st[NST + NCAR + NLD + NTD])
            return acc, qd_, pre, tdp_, cur

        def pack_state(acc, qd_, pre, tdp_, cur):
            return (
                [_ir(x) for x in acc]
                + [_ir(x) for x in qd_]
                + [_ir(x) for x in pre]
                + list(tdp_)
                + [_ir(cur)]
            )

        def tile(acc, qd_, pre, tdp_, cur, do_mask, ii, n, qbase):
            """Tile ii of a loop of n tiles (q pair qbase + ii) in ring stage cur: returns the new (acc, qd, NL/ND of
            tile ii + 1, pending TDM, stage)."""
            acc = list(acc)
            ncur = _ncur(cur)
            rocdl.sched_barrier(0)
            if const_expr(not K3):
                nxt = ldl(qbase + clampn(ii + one, n))
            pft = qbase + clampn(ii + fx.Int32(3), n)
            pf_live = ii + fx.Int32(3) < n
            qt = qbase + ii
            q0 = qt * fx.Int32(32)
            if const_expr(K3):
                # S / dP + softmax (sdp3) with the 40 dV / dK A-operand tr16 issued under hh1's softmax into the dead
                # S / dP operand registers (no register still read by an in-flight WMMA is reused); the dS^T stores;
                # dV; the next tile's NL / ND (loaded here: register room); dK kh0; tensorcnt 3; split barrier around
                # dK kh1; the next stage's hh0 readback under dK kh2
                tv, tk = tr_base(cur, False), tr_base(cur, True)
                rbb = rd_base(ncur)
                ops = {}

                def mid():
                    ops["do"] = [tr(tv, False, d) for d in range_constexpr(NDO_V)]
                    ops["q"] = [tr(tk, True, d) for d in range_constexpr(NDO_QK)]

                a_pk, a_dsk = sdp3(do_mask, qd_, rd_base(cur), pre, q0, lambda: tdm_issue(tdp_), mid)
                b_do, b_q = ops["do"], ops["q"]
                rocdl.sched_barrier(0)
                if const_expr(not do_mask):
                    # this iteration's dS^T blocks (all NKV stored below the diagonal), ahead of the dV run
                    so = qt * fx.Int32(DSL_BLK_B)
                    for kh in range_constexpr(NKV):
                        ds_store(a_dsk[kh], so + ds_rc[kh])
                    rocdl.sched_barrier(0)
                tdp_n = tdm_prep(pft, cur, pf_live)
                for d in range_constexpr(NDO_V):
                    for kh in range_constexpr(NKV):
                        acc[kh * NDO_V + d] = wmma(b_do[d], a_pk[kh], acc[kh * NDO_V + d], ra=(kh > 0))
                for i in range_constexpr(NKV * NDO_V):
                    rocdl.sched_group_barrier(0x008, 1, 0)
                    rocdl.sched_group_barrier(0x004, KH0_SALU, 0)
                rocdl.sched_barrier(0)
                nxt = ldl(qbase + clampn(ii + one, n))
                rocdl.sched_barrier(0)
                dk_kh(0, acc, b_q, a_dsk)
                rocdl.sched_barrier(0)
                tdm_ops.tensor_wait(TW)
                rocdl.sched_barrier(0)
                _lds_arrive()
                rocdl.sched_barrier(0)
                dk_kh(1, acc, b_q, a_dsk)
                rocdl.sched_barrier(0)
                _lds_wait()
                rocdl.sched_barrier(0)
                qd_ = rdqd_hh(rbb, 0)
                dk_kh(2, acc, b_q, a_dsk)
                for _ in range_constexpr(NDO_QK - 2):
                    rocdl.sched_group_barrier(0x008, 1, 0)
                    rocdl.sched_group_barrier(0x100, 2, 0)
                rocdl.sched_barrier(0)
            else:
                a_pk, a_dsk = sdp2(do_mask, qd_, pre, q0)
                tv, tk = tr_base(cur, False), tr_base(cur, True)
                rbb = rd_base(ncur)
                rocdl.sched_barrier(0)
                b_do = [tr(tv, False, d) for d in range_constexpr(NDO_V)]
                rocdl.sched_barrier(0)
                b_q = [tr(tk, True, d) for d in range_constexpr(NDO_QK)]
                rocdl.sched_barrier(0)
                tdp_c = tdm_prep(pft, cur, pf_live)
                dv_kh(0, acc, b_do, a_pk)
                dk_kh(0, acc, b_q, a_dsk)
                for _ in range_constexpr(NDO_V + NDO_QK):
                    rocdl.sched_group_barrier(0x008, 1, 0)
                    rocdl.sched_group_barrier(0x004, KH0_SALU, 0)
                rocdl.sched_barrier(0)
                tdm_ops.tensor_wait(TW)
                rocdl.sched_barrier(0)
                _lds_barrier()
                rocdl.sched_barrier(0)
                qd_ = rdqd_hh(rbb, 0) + rdqd_hh(rbb, 1)
                rocdl.sched_barrier(0x004)  # readback before the TDM; SALU may cross
                tdm_issue(tdp_c)
                rocdl.sched_barrier(0)
                for kh in range_constexpr(1, NKV):
                    dv_kh(kh, acc, b_do, a_pk)
                    dk_kh(kh, acc, b_q, a_dsk)
                tdp_n = []
            if const_expr(do_mask or not K3):
                # export: a block only if its kv16 row has an attended entry in pair qt
                rocdl.sched_barrier(0)
                so = qt * fx.Int32(DSL_BLK_B)
                for kh in range_constexpr(NKV):
                    _store_if(qt >= ds_qpf[kh], a_dsk[kh], so + ds_rc[kh])
                rocdl.sched_barrier(0)
            return acc, qd_, nxt, tdp_n, ncur

        @flyc.jit
        def _store_if(cond, a, so):
            if cond:
                ds_store(a, so)

        def loop1(do_mask, n, qbase, nclamp):
            """n tiles, one per trip (the NL/ND set is copied across the back edge); prefetches are clamped to tile
            nclamp - 1 of the sequence from qbase."""

            @flyc.jit
            def run(state):
                final = state
                for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n), 1, init=state):
                    out = tile(*unpack(list(carried)), do_mask, fx.Int32(it), nclamp, qbase)
                    final = yield pack_state(*out)
                return final

            return run

        @flyc.jit
        def tail_if(cond, state):
            res = state
            if cond:
                out = tile(*unpack(list(state)), False, n_full - one, n_full, qt0)
                res = pack_state(*out)
            return res

        @flyc.jit
        def full_loop2(state, npair):
            """Two tiles per trip: tile 2 it reads NL/ND set A and loads set B, tile 2 it + 1 the reverse, so the
            sets alternate without copies."""
            final = state
            for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(npair), 1, init=state):
                ii = fx.Int32(it) * fx.Int32(2)
                out = tile(*unpack(list(carried)), False, ii, n_full, qt0)
                out = tile(*out, False, ii + one, n_full, qt0)
                final = yield pack_state(*out)
            return final

        # the masked loop's ring stage starts at sigma(0) = s0, the full loop's at stage 0
        st = pack_state(acc0, qd, preA, tdp, s0)
        st = list(loop1(True, n_mask, qp_start, n_seq)(st))
        acc, qd, pre, tdp, _ = unpack(st)
        st = pack_state(acc, qd, pre, tdp, fx.Int32(0))
        if const_expr(K3):
            st = list(full_loop2(st, n_full >> one))
            st = list(tail_if((n_full & one) != fx.Int32(0), st))
        else:
            st = list(loop1(False, n_full, qt0, n_full)(st))
        acc = st[:NST]
        # retire the last (clamped) prefetch before the epilogue reuses the ring
        tdm_ops.tensor_wait(0)
        rocdl.sched_barrier(0)
        _lds_barrier()
        rocdl.sched_barrier(0)

        # ---- epilogue: C fragments -> this wave's row-major dV / dK images -> whole-line stores
        EPI_W = 16 * NKV * (XV + XK)
        EPI_DK = 16 * NKV * XV
        wr = lds_ptr(lds0 + wv * fx.Int32(EPI_W) + row * fx.Int32(XV) + half * fx.Int32(16))
        wrk = lds_ptr(
            lds0 + wv * fx.Int32(EPI_W) + fx.Int32(EPI_DK) + row * fx.Int32(XK) + half * fx.Int32(16)
        )
        for kh in range_constexpr(NKV):
            for d in range_constexpr(NDO_QK):
                if const_expr(d < NDO_V):
                    o = kh * 16 * XV + d * 32
                    dvv = fx.Vector(acc[kh * NDO_V + d])
                    llvm.store(
                        _ir(dvv.to(fx.BFloat16)),
                        wr if o == 0 else get_element_ptr(wr, static_byte_offset=o),
                    )
                o = kh * 16 * XK + d * 32
                llvm.store(
                    _ir(fx.Vector(acc[DK0 + kh * NDO_QK + d]).to(fx.BFloat16)),
                    wrk if o == 0 else get_element_ptr(wrk, static_byte_offset=o),
                )
        rocdl.sched_barrier(0)
        r_dv = create_buffer_resource(DV_, num_records_bytes=nv_b)
        r_dk = create_buffer_resource(DK, num_records_bytes=nk_b)
        lr = lane >> fx.Int32(3)
        lc = lane & fx.Int32(7)
        rdv = lds_ptr(lds0 + wv * fx.Int32(EPI_W) + lr * fx.Int32(XV) + lc * fx.Int32(16))
        rdk = lds_ptr(lds0 + wv * fx.Int32(EPI_W) + fx.Int32(EPI_DK) + lr * fx.Int32(XK) + lc * fx.Int32(16))
        gv = vbase + (kv0 + lr) * rs_v + lc
        gk = kbase + (kv0 + lr) * rs_k + lc
        for rb in range_constexpr(4 * NKV):
            for li in range_constexpr(D_V * 2 // 128):
                o = rb * 4 * XV + li * 128
                vv = llvm.load(v8b, rdv if o == 0 else get_element_ptr(rdv, static_byte_offset=o))
                buffer_store(
                    vv,
                    r_dv,
                    (gv + fx.Int32(rb * 4) * rs_v + fx.Int32(li * 8)) * fx.Int32(16),
                    offset_is_bytes=True,
                )
            for li in range_constexpr(D_QK * 2 // 128):
                o = rb * 4 * XK + li * 128
                kk = llvm.load(v8b, rdk if o == 0 else get_element_ptr(rdk, static_byte_offset=o))
                buffer_store(
                    kk,
                    r_dk,
                    (gk + fx.Int32(rb * 4) * rs_k + fx.Int32(li * 8)) * fx.Int32(16),
                    offset_is_bytes=True,
                )

    @flyc.jit
    def by_height(is96):
        if is96:
            body(NKV3)
        else:
            body(2)

    by_height(bid < n96)


@flyc.kernel(name="k_mla_bwd_dkdv96ds", known_block_size=[32 * NW, 1, 1])
def k_dkdv96ds(
    Q: fx.Tensor,
    K: fx.Tensor,
    V: fx.Tensor,
    DO: fx.Tensor,
    NL: fx.Tensor,
    ND: fx.Tensor,
    DV_: fx.Tensor,
    DK: fx.Tensor,
    DS: fx.Tensor,
    scale: fx.Float32,
    Sq: fx.Int32,
    Skv: fx.Int32,
    Hq: fx.Int32,
    Hkv: fx.Int32,
    G: fx.Int32,
    nqt: fx.Int32,
    cshift: fx.Int32,
    causal: fx.Int32,
    B_: fx.Int32,
    HG: fx.Int32,
    ds_mode: fx.Int32,
    H0: fx.Int32,
    HC: fx.Int32,
):
    _dkdv96_impl(
        Q,
        K,
        V,
        DO,
        NL,
        ND,
        DV_,
        DK,
        DS,
        scale,
        Sq,
        Skv,
        Hq,
        Hkv,
        G,
        nqt,
        cshift,
        causal,
        B_,
        HG,
        ds_mode,
        H0,
        HC,
    )


@flyc.jit
def launch_dkdv96ds(
    Q,
    K,
    V,
    DO,
    NL,
    ND,
    DV_,
    DK,
    DS,
    scale: fx.Float32,
    Sq: fx.Int32,
    Skv: fx.Int32,
    Hq: fx.Int32,
    Hkv: fx.Int32,
    G: fx.Int32,
    nqt: fx.Int32,
    cshift: fx.Int32,
    causal: fx.Int32,
    B_: fx.Int32,
    HG: fx.Int32,
    ds_mode: fx.Int32,
    H0: fx.Int32,
    HC: fx.Int32,
    nblk: fx.Int32,
    ngz: fx.Int32,
    stream: fx.Stream,
):
    _ = SOURCE_DIGEST  # FlyDSL disk-cache key
    # grid = (HG heads of a head group, kv blocks, B * HC / HG head groups): the heads [H0, H0 + HC)
    k_dkdv96ds(
        Q,
        K,
        V,
        DO,
        NL,
        ND,
        DV_,
        DK,
        DS,
        scale,
        Sq,
        Skv,
        Hq,
        Hkv,
        G,
        nqt,
        cshift,
        causal,
        B_,
        HG,
        ds_mode,
        H0,
        HC,
    ).launch(grid=(HG, nblk, ngz), block=(32 * NW, 1, 1), stream=stream)


launch_dkdv96ds.compile_hints["llvm_options"] = {
    "structurizecfg-skip-uniform-regions": True,
    "amdgpu-expert-scheduling-mode": True,
    "amdgpu-enable-delay-alu": False,
    "greedy-regclass-priority-trumps-globalness": True,
}
launch_dkdv96ds.compile_hints["waves_per_eu"] = 1


# ------------------------------------------------------------------------------------------------------- host
def nblk96(skv):
    """kv blocks: 96-row ones, then one 64-row tail for skv % 96 == 64 or two for 32 (the last 128 rows)."""
    return -(-skv // ROWS3)


def unsupported_reason(b, sq, skv, hq, hkv, causal=True):
    """Why k_dkdv96ds cannot take this problem, or None."""
    if hq != hkv:
        return "MHA only (heads_q == heads_kv): the kernel is specialized for G = 1"
    if not causal:
        return "causal only"
    if skv < sq:
        return "seqlen_kv < seqlen_q"
    if sq % 64 or skv % 32 or sq <= 0 or skv < 64:
        return "seqlen_q must be a positive multiple of 64 and seqlen_kv a multiple of 32 (>= 64)"
    if b * hkv * skv * D_QK * 2 >= 1 << 31 or b * hq * sq * 4 >= 1 << 31:
        return "k / NL too large for the kernel's 32-bit buffer offsets"
    return None
