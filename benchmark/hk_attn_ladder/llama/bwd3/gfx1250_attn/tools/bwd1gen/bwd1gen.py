###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""bwd1gen -- the generated body of k_hk_bwd1_g4 (HK gfx1250 d128 GQA attention backward v1).

The marker shell (hip/src/k_attn_bwd_v1.hip, -DHKGEN=1) runs the work decode, the TDM prologue (ops 0..2, the K / V
images, dS^T buffer 1 zeroed) and the first barrier, computes the loop-invariant scalars and lane bases, and marks
one region, BODY, which this module generates (build.py splices it in):

  INIT   K / V B operands (kf, vf: rows) and the dQ B operands (kt: tr16) from the K / V images, accumulators = 0,
         readback(0) (the S / dP A operands and the NL / delta records of tile 0), dO^T(0) operands 0..3
  FULL   steps 0 .. nf-1, unrolled by 4 (P4 = s & 3: the ring slot, the dS^T buffer and the TDM descriptor set are
         static per body); one step, WMMA order:
           A  S / dP of kv tile 0 (16)  || dS^T(s-1) tr16 (buffer P ^ 1), Q^T(s) operands 0..3
           B  S / dP of kv tile 1 (16)  || softmax / dS of kv tile 0; Q^T 4..7, dO^T 4..7; readback(s+1) starts as
                                           the kv tile 1 WMMAs release the S / dP A operands (dt by dt)
           C  dQ(s-1) = dSa K (16)      || softmax / dS of kv tile 1; readback(s+1)
           D  dV^T += dO^T P (16)       || the dS^T(s) stores, the 32 dQ(s-1) atomics, the LD records of s+1,
                                           s_wait_tensorcnt 1 + s_wait_dscnt N + s_barrier_signal (every read of slot s
                                           and the stores done; readback loads may stay in flight)
           E  dK^T += Q^T dS (16)       || dO^T(s+1) operands 0..3 (dV released them); s_barrier_wait at the end
         The TDM op of tile s+3 starts every step (after the previous step's barrier).
  DIAG   steps nf .. n-1: the same body with the causal mask on the exp2 argument (every wave kind computes; a DEAD
         wave's P and dS are exact zeros, so dK / dV keep their bits and the dS^T it stores adds nothing to dQ)
  FINAL  dQ(n-1) and its atomics, then dV^T and scale dK^T to bf16, whole-row buffer stores

Numerics are v1's (bwd1/v1_step.inc, hk_attn/ext_bwd.h) instruction for instruction: S and dP chains over dt from +0,
t = fma(S, c1, NL) (v_pk_fma_f32), masked t = MASK_NEG, P = exp2(t) (v_exp_f32), dS = (dP - delta) * P (v_pk_add_f32
with neg, v_pk_mul_f32), RNE bf16 (v_cvt_pk_bf16_f32); dQ chains over the kv chunk c from +0; every dV / dK
accumulator takes one WMMA per step in step order. So dK / dV are bitwise those of the HIP kernel.

The node list of a step is in program order (the semantics); schedule_slot reorders it along the dependence graph
(asmkit's parser: RAW / WAR / WAW on VGPRs, SGPRs, scc, vcc) plus order edges (window deadlines, the barrier
protocol). The output is real instructions only: asmkit.emit adds s_set_vgpr_msb, the expert-mode waits and v_nops."""

import re

from asmkit.emit import make_item

XP, SP = 272, 96  # LDS pitches (bytes) of the Q / dO / K / V tiles and of the dS^T exchange tile
QT_B = 32 * XP  # one Q or dO ring slot tile
LD_B = 256  # the NL / delta records of one step
DS_B = 128 * SP  # one dS^T exchange buffer
NEG = "0xff61b1e6"  # MASK_NEG = -3.0e38f
SCOPE = "scope:SCOPE_DEV"

# SGPRs of the generated state (the marker clobbers s72..s103)
SG_DESC = (72, 76)  # TDM group-0 descriptor sets (step parity): {1, LDS address, global address lo, hi}
SG_A = (80, 82, 84, 86)  # per P4: the global address of the next op of that P4 (64-bit, bit 63 set)
SG_V = ("s88", "s89", "s90", "s91")  # per P4: the dQ_acc tile scalar of the tile step P4 computes
SG_U = "s92"  # the dQ_acc tile scalar of the next P4 = 0 step's atomics (the previous iteration's P4 = 3 tile)
SG_S = "s93"  # step base s of the iteration
SG_M = "s94"  # causal mask base M of the iteration
SG_T = "s95"  # temporary
SCALARS = {f"s{i}" for i in range(72, 104)} | {"vcc", "scc"}


def vr(r, n=1):
    """Operand text of the n VGPRs at true register r: the encoded low 8 bits, the true numbers in a comment."""
    if r // 256 != (r + n - 1) // 256:
        raise ValueError(f"tuple v[{r}:{r + n - 1}] crosses a 256-register block")
    lo = r % 256
    if n == 1:
        return f"v{lo}" if r < 256 else f"v{lo} /*v{r}*/"
    return f"v[{lo}:{lo + n - 1}]" if r < 256 else f"v[{lo}:{lo + n - 1}] /*v[{r}:{r + n - 1}]*/"


def offset(off):
    """The offset modifier of a memory instruction (none for 0)."""
    if not 0 <= off < 65536:
        raise ValueError(f"offset {off} out of range")
    return f" offset:{off}" if off else ""


def wmma(d, a, b, c):
    """D = A B + C (v_wmma_f32_16x16x32_bf16) on 8-VGPR tuples; c None: from +0."""
    return f"v_wmma_f32_16x16x32_bf16 {vr(d, 8)}, {vr(a, 8)}, {vr(b, 8)}, {'0' if c is None else vr(c, 8)}"


def sbase(op):
    """First SGPR of a marker operand s[a:b] or sN."""
    m = re.fullmatch(r"s\[(\d+):(\d+)\]|s(\d+)", op)
    if not m:
        raise ValueError(f"{op}: not an SGPR operand")
    return int(m.group(1) or m.group(3))


def sreg(op, i):
    """SGPR i of a tuple operand."""
    return f"s{sbase(op) + i}"


def spair(op, i):
    """SGPRs i, i + 1 of a tuple operand as a pair."""
    b = sbase(op) + i
    if b % 2:
        raise ValueError(f"{op}: pair at odd SGPR s{b}")
    return f"s[{b}:{b + 1}]"


def rows_imm(r0, k0, half, pitch=XP):
    return r0 * pitch + 2 * k0 + 32 * half


def tr_imm(k0, r0, half, pitch=XP):
    return (k0 + 16 * half) * pitch + 2 * r0


def dqs_imm(jj, i):
    """dQ_acc byte offset of element i of dQ fragment jj (BWD1_OPT_DQSPLIT, bwd1/idx.h dqs_imm)."""
    return ((jj % 2) * 8 + i) * 512 + (jj // 2) * 128


class Layout:
    """VGPR map (true register numbers) of the generated state, all at or above `base`; an 8-VGPR tuple never crosses
    a 256-VGPR block.
      accV[dt][kh], accK[dt][kh]  dV^T / dK^T accumulators (d tile dt, kv tile kh)
      kf[kh][dt], vf[kh][dt]      S / dP B operands (K / V rows of the wave)
      kt[jj][c]                   dQ B operands (K^T: d tile jj of the wave's 64 d, kv chunk c)
      qA[hh][dt], dA[hh][dt]      S / dP A operands of the step (Q / dO rows), ld[hh][k] the NL / delta quads
      S[hh][kh], dP[hh][kh]       raw S / dP; the softmax runs in place (t, P in S; dP - delta, dS in dP)
      P[kh], DS[kh]               bf16 P / dS B operands (dwords 4 hh + j)
      dOT[dt], QT[dt]             dV / dK A operands (tr16), dSa[c] dQ A operands, dq[jj] the dQ fragments
      c1                          (c1, c1); FINAL: (scale, scale)
      vo[2]                       the atomics' voffset (step parity), thr[3] the mask thresholds (hh - kh + 1)"""

    def __init__(self, base):
        used = [[] for _ in range(4)]

        def take(n, align):
            for blk in range(4):
                r = max([max(base, blk * 256)] + [a + m for a, m in used[blk]])
                r += (-r) % align
                if r + n <= (blk + 1) * 256:
                    used[blk].append((r, n))
                    return r
            raise ValueError(f"layout does not fit above base v{base}")

        t8 = lambda: take(8, 8)  # noqa: E731
        self.accV = [[t8() for _ in range(2)] for _ in range(8)]
        self.accK = [[t8() for _ in range(2)] for _ in range(8)]
        self.kf = [[t8() for _ in range(4)] for _ in range(2)]
        self.vf = [[t8() for _ in range(4)] for _ in range(2)]
        self.kt = [[t8() for _ in range(4)] for _ in range(4)]
        self.qA = [[t8() for _ in range(4)] for _ in range(2)]
        self.dA = [[t8() for _ in range(4)] for _ in range(2)]
        self.dOT = [t8() for _ in range(8)]
        self.QT = [t8() for _ in range(8)]
        self.S = [[t8() for _ in range(2)] for _ in range(2)]
        self.dP = [[t8() for _ in range(2)] for _ in range(2)]
        self.dSa = [t8() for _ in range(4)]
        self.dq = [t8() for _ in range(4)]
        self.P = [t8() for _ in range(2)]
        self.DS = [t8() for _ in range(2)]
        self.ld = [[take(4, 4) for _ in range(4)] for _ in range(2)]
        self.c1 = take(2, 2)
        self.vo = [take(1, 1) for _ in range(2)]
        self.thr = [take(1, 1) for _ in range(3)]
        self.top = max(a + m for blk in used for a, m in blk)
        self.acc = [self.accV[dt][kh] for dt in range(8) for kh in range(2)] + [
            self.accK[dt][kh] for dt in range(8) for kh in range(2)
        ]


# --------------------------------------------------------------------------------------------------------------
# dependence graph and slot scheduler (tools/bwdgen's, with the split barrier and a VMEM issue rate)
# --------------------------------------------------------------------------------------------------------------
LAT = {
    "WMMA": 20,
    "VALU": 4,
    "TRANS": 8,
    "DS_LD": 90,
    "DS_ST": 4,
    "VM_LD": 800,
    "VM_ST": 4,
    "TDM": 4,
    "SALU": 2,
    "BAR_S": 1,
    "BAR_W": 1,
    "WAIT_CNT": 1,
    "OTHER": 1,
    "BR": 1,
}
WMMA_CHAIN = 16
WAR_XDL, MIN_XDL_AFTER = 24, 3
DS_MAX = 48
VALU_SLOTS, MEM_SLOTS = (2, 3, 6, 7), (1, 4, 5)
LDS_LAT, LDS_RATE = 70, 8
VM_RATE = 4
BAR_GAP = 12
MEM = ("DS_LD", "DS_ST", "VM_LD", "VM_ST", "TDM")


class Node:
    __slots__ = ("text", "kind", "vdefs", "vuses", "sdefs", "suses", "preds", "war_xdl", "tag")


def mk(text, tag=None):
    it = make_item("\t" + text)
    n = Node()
    n.text, n.kind, n.tag = text, it.kind, tag
    n.vdefs, n.vuses = set(it.vdefs), set(it.vuses)
    n.sdefs, n.suses = set(it.sdefs), set(it.suses)
    n.preds, n.war_xdl = [], set()
    return n


def build_dag(nodes):
    """Edges in list order: RAW (the producer's latency), WAR / WAW (order; WAR_XDL for a non-VALU writer behind a WMMA
    that reads the register); n.preds = [(pred, latency)]."""
    last_w, readers = {}, {}
    for i, n in enumerate(nodes):
        uses = [("v", x) for x in n.vuses] + [("s", x) for x in n.suses]
        defs = [("v", x) for x in n.vdefs] + [("s", x) for x in n.sdefs]
        preds = dict(n.preds)
        for r in uses:
            if r in last_w:
                p = last_w[r]
                lat = WMMA_CHAIN if (nodes[p].kind == "WMMA" and n.kind == "WMMA") else LAT[nodes[p].kind]
                preds[p] = max(preds.get(p, 0), lat)
        for r in defs:
            if r in last_w:
                preds[last_w[r]] = max(preds.get(last_w[r], 0), 1)
            for q in readers.get(r, ()):
                if q != i:
                    war_xdl = nodes[q].kind == "WMMA" and n.kind not in ("WMMA", "VALU", "TRANS")
                    preds[q] = max(preds.get(q, 0), WAR_XDL if war_xdl else 1)
                    if war_xdl:
                        n.war_xdl.add(q)
        n.preds = sorted(preds.items())
        for r in uses:
            readers.setdefault(r, set()).add(i)
        for r in defs:
            last_w[r] = i
            readers[r] = set()


def add_order(nodes, a, b, lat=1):
    """An order edge a -> b (a before b in list order) of at least lat cycles."""
    if not a < b:
        raise ValueError(f"order edges must point forward in program order ({a} -> {b})")
    d = dict(nodes[b].preds)
    d[a] = max(d.get(a, 0), lat)
    nodes[b].preds = sorted(d.items())


def schedule_slot(nodes, wseq, bar_deps=()):
    """(ordered nodes, modelled cycles): a cycle-stepped list scheduler around the in-order WMMA stream wseq, one
    instruction per cycle. A WMMA issues once the previous one left the XDL pipe (8 cycles); inside its window a VALU /
    TRANS issues only at VALU_SLOTS, a memory op only at MEM_SLOTS; one LDS op per LDS_RATE cycles, completing in order
    LDS_LAT after issue; one VMEM op per VM_RATE cycles; a TRANS blocks VALU / TRANS the next cycle; s_barrier_signal
    waits until the LDS ops bar_deps completed (its s_wait_dscnt fence), s_barrier_wait BAR_GAP WMMAs after it; at most
    DS_MAX LDS loads in flight. Ties: barrier and wait nodes first, then the earliest deadline (the first WMMA that needs
    the node, transitively), the longest critical path, program order."""
    n = len(nodes)
    succ = [[] for _ in range(n)]
    for i, nd in enumerate(nodes):
        for p, l in nd.preds:
            succ[p].append((i, l))
    wpos = {i: k for k, i in enumerate(wseq)}
    dl = [10**6] * n
    for i in range(n - 1, -1, -1):
        for s_, _ in succ[i]:
            dl[i] = min(dl[i], wpos[s_] if s_ in wpos else dl[s_])
    cp = [0] * n
    for i in range(n - 1, -1, -1):
        cp[i] = (8 if nodes[i].kind == "WMMA" else 1) + max((l + cp[s] for s, l in succ[i]), default=0)
    npred = [len(nd.preds) for nd in nodes]
    earliest = [0] * n
    ready = set(i for i in range(n) if npred[i] == 0)
    out, wmma_at, done = [], {}, {}
    t, last_w, wk, trans_at, ds_next, ds_last_done, vm_next = 0, -100, 0, -100, 0, 0, 0
    ds_issued, ds_retired, ds_seq = 0, -1, {}
    bs_wk = 10**6
    bar_deps = set(bar_deps)

    def xdl_ok(i):
        rs = nodes[i].war_xdl
        if not rs:
            return True
        if not all(r in wmma_at for r in rs):
            return False
        return wk - 1 - max(wmma_at[r] for r in rs) >= MIN_XDL_AFTER or wk >= len(wseq)

    def avail(i):
        k = nodes[i].kind
        if earliest[i] > t:
            return False
        o = t - last_w
        inwin = 0 <= o < 8
        if k in ("VALU", "TRANS"):
            return (not inwin or o in VALU_SLOTS) and t != trans_at + 1
        if k in MEM:
            if not xdl_ok(i):
                return False
            if k in ("DS_LD", "DS_ST") and (t < ds_next or ds_issued - 1 - ds_retired >= DS_MAX):
                return False
            if k in ("VM_LD", "VM_ST") and t < vm_next:
                return False
            return not inwin or o in MEM_SLOTS
        if k == "BAR_S":
            return o != 0 and all(j in done and done[j] <= t for j in bar_deps)
        if k == "BAR_W":
            return o != 0 and (wk >= bs_wk + BAR_GAP or wk >= len(wseq))
        return o != 0

    while len(out) < n:
        pick = None
        nw = wseq[wk] if wk < len(wseq) else None
        if nw is not None and nw in ready and earliest[nw] <= t and t >= last_w + 8:
            pick = nw
        else:
            cand = [i for i in ready if nodes[i].kind != "WMMA" and avail(i)]
            if cand:
                if 0 <= t - last_w < 8 and t - last_w in VALU_SLOTS:
                    cand = [i for i in cand if nodes[i].kind in ("VALU", "TRANS")] or cand
                pick = max(
                    cand, key=lambda i: (nodes[i].kind in ("BAR_S", "BAR_W", "WAIT_CNT"), -dl[i], cp[i], -i)
                )
        if pick is None:
            t += 1
            if t > 10**6:
                raise RuntimeError("slot scheduler deadlock")
            continue
        nd = nodes[pick]
        ready.discard(pick)
        k = nd.kind
        done_at = None
        if k in ("DS_LD", "DS_ST"):
            ds_next = max(t, ds_next) + LDS_RATE
            done_at = max(t + LDS_LAT, ds_last_done + LDS_RATE)
            ds_last_done = done_at
            ds_seq[pick] = ds_issued
            ds_issued += 1
            done[pick] = done_at
        if k in ("VM_LD", "VM_ST"):
            vm_next = t + VM_RATE
        for p, _ in nd.preds:
            if nodes[p].kind == "DS_LD" and p in ds_seq and nd.vuses & nodes[p].vdefs:
                ds_retired = max(ds_retired, ds_seq[p])
        if k == "BAR_S":
            ds_retired = max([ds_retired] + [ds_seq[j] for j in bar_deps if j in ds_seq])
            bs_wk = wk
        out.append(nd)
        if k == "WMMA":
            last_w = t
            wmma_at[pick] = wk
            wk += 1
        if k == "TRANS":
            trans_at = t
        for s_, l in succ[pick]:
            if k == "DS_LD" and nodes[s_].vuses & nd.vdefs:
                lat = done_at - t
            elif k == "WMMA":
                lat = WMMA_CHAIN if nodes[s_].kind == "WMMA" and (nodes[s_].vuses & nd.vdefs) else max(l, 1)
            else:
                lat = max(l, LAT[k]) if (nodes[s_].vuses & nd.vdefs or nodes[s_].suses & nd.sdefs) else l
            earliest[s_] = max(earliest[s_], t + lat)
            npred[s_] -= 1
            if npred[s_] == 0:
                ready.add(s_)
        t += 1
    return out, max(t, last_w + 8)


# --------------------------------------------------------------------------------------------------------------
# instruction texts
# --------------------------------------------------------------------------------------------------------------
def ld_rows(dst, base, off):
    return [f"ds_load_b128 {vr(dst + 4 * u, 4)}, {base}{offset(off + 32 * u)}" for u in range(2)]


def ld_tr(dst, base, off0, pitch):
    return [f"ds_load_tr16_b128 {vr(dst + 4 * u, 4)}, {base}{offset(off0 + 16 * u * pitch)}" for u in range(2)]


def rb_texts(L, ops, slot, dt):
    """readback of the step in ring slot `slot`: the qA / dA operands of k-step dt (4 tuples, 8 ds_load_b128)."""
    out = []
    for hh in range(2):
        out += ld_rows(L.qA[hh][dt], ops["qr"], slot * QT_B + rows_imm(16 * hh, 32 * dt, 0))
        out += ld_rows(L.dA[hh][dt], ops["dr"], slot * QT_B + rows_imm(16 * hh, 32 * dt, 0))
    return out


def ld_texts(L, ops, slot):
    """the NL / delta records of the step in ring slot `slot` (8 ds_load_b128)."""
    return [
        f"ds_load_b128 {vr(L.ld[hh][k], 4)}, {ops['ldb']}{offset(slot * LD_B + 128 * hh + 16 * k)}"
        for hh in range(2)
        for k in range(4)
    ]


def tr_texts(L, ops, which, slot, dts):
    """dO^T (which 'd') / Q^T ('q') operands dts of ring slot `slot` (2 tr16 each)."""
    reg, base = (L.dOT, ops["dtr"]) if which == "d" else (L.QT, ops["qtr"])
    out = []
    for dt in dts:
        out += ld_tr(reg[dt], base, slot * QT_B + 32 * dt, XP)
    return out


def dsa_texts(L, ops, buf):
    """the dQ A operands dSa[c] from dS^T buffer `buf` (2 tr16 each)."""
    out = []
    for c in range(4):
        out += ld_tr(L.dSa[c], ops["dsr"], buf * DS_B + 32 * c * SP, SP)
    return out


def softmax_texts(L, hh, kh, masked):
    """softmax / dS of block (hh, kh) in place: t = fma(S, c1, NL) [masked: i < thr -> MASK_NEG], P = exp2(t),
    dS = (dP - delta) * P, bf16 P / dS into dwords 4 hh + j of P[kh] / DS[kh]."""
    out = []
    s_, d_ = L.S[hh][kh], L.dP[hh][kh]
    thr = L.thr[hh - kh + 1]
    for j in range(4):
        nl = L.ld[hh][2 * (j // 2)] + 2 * (j % 2)
        de = L.ld[hh][2 * (j // 2) + 1] + 2 * (j % 2)
        s2, d2 = s_ + 2 * j, d_ + 2 * j
        out.append(f"v_pk_fma_f32 {vr(s2, 2)}, {vr(s2, 2)}, {vr(L.c1, 2)}, {vr(nl, 2)}")
        if masked:
            for e in range(2):
                out.append(f"v_cmp_ge_i32 vcc_lo, {2 * j + e}, {vr(thr)}")
                out.append(f"v_cndmask_b32 {vr(s2 + e)}, {NEG}, {vr(s2 + e)}, vcc_lo")
        out.append(f"v_exp_f32 {vr(s2)}, {vr(s2)}")
        out.append(f"v_exp_f32 {vr(s2 + 1)}, {vr(s2 + 1)}")
        out.append(f"v_pk_add_f32 {vr(d2, 2)}, {vr(d2, 2)}, {vr(de, 2)} neg_lo:[0,1] neg_hi:[0,1]")
        out.append(f"v_pk_mul_f32 {vr(d2, 2)}, {vr(d2, 2)}, {vr(s2, 2)}")
        out.append(f"v_cvt_pk_bf16_f32 {vr(L.P[kh] + 4 * hh + j)}, {vr(s2)}, {vr(s2 + 1)}")
        out.append(f"v_cvt_pk_bf16_f32 {vr(L.DS[kh] + 4 * hh + j)}, {vr(d2)}, {vr(d2 + 1)}")
    return out


def sdp_wmmas(L, kh):
    """S / dP of kv tile kh: dt outer, (S hh0, S hh1, dP hh0, dP hh1) inner (each chain's next WMMA 4 later)."""
    out = []
    for dt in range(4):
        for hh in range(2):
            s_ = L.S[hh][kh]
            out.append(wmma(s_, L.qA[hh][dt], L.kf[kh][dt], None if dt == 0 else s_))
        for hh in range(2):
            d_ = L.dP[hh][kh]
            out.append(wmma(d_, L.dA[hh][dt], L.vf[kh][dt], None if dt == 0 else d_))
    return out


def dq_wmmas(L):
    """dQ = dSa K^T over the 4 kv chunks c (c outer, the wave's 4 d tiles jj inner), from +0."""
    return [
        wmma(L.dq[jj], L.dSa[c], L.kt[jj][c], None if c == 0 else L.dq[jj]) for c in range(4) for jj in range(4)
    ]


def atomic_texts(L, ops, vo):
    return [
        f"buffer_atomic_add_f32 {vr(L.dq[jj] + i)}, {vr(vo)}, {ops['rq']}, null offen{offset(dqs_imm(jj, i))} {SCOPE}"
        for jj in range(4)
        for i in range(8)
    ]


# --------------------------------------------------------------------------------------------------------------
# one step
# --------------------------------------------------------------------------------------------------------------
# readback deadlines: k-step dt of the next step's S / dP operands before WMMA RB_BEFORE[dt] of this step, the LD
# records before LD_BEFORE; the dQ atomic a before WMMA 49 + a (one per window in D and E)
RB_BEFORE = (44, 52, 60, 68)
LD_BEFORE = 76
SIG_BEFORE = 60  # within D (48..63): the signal is listed among the dV WMMAs
ST_BEFORE = 50  # the dS^T stores early in D: the signal's fence then waits for little more than them
QA_BEFORE, QB_BEFORE = 16, 40  # Q^T 0..3 in A, Q^T 4..7 and dO^T 4..7 by WMMA 40
RB_GAP = 8  # WMMAs between a register's last WMMA reader and the load that overwrites it


def step_nodes(L, ops, p4, masked):
    """(nodes, WMMA order, barrier deps) of one step with P4 = p4 (program order = semantics)."""
    nodes = []

    def add(text, tag=None):
        nodes.append(mk(text, tag))
        return len(nodes) - 1

    P = p4 & 1
    slot, nslot = p4, (p4 + 1) & 3
    desc = SG_DESC[P]
    tdm, sc = ops["tdm"], ops["sc"]
    n_ = sreg(sc, 0)
    W = []
    proto = []  # LDS ops that complete before the barrier signal
    # entry: the TDM op of tile s + 3 (clamped to tile n - 1), the next iteration's address of this P4, the voffset
    add(f"s_add_co_i32 {SG_T}, {SG_S}, {p4 + 3}")
    add(f"s_cmp_ge_i32 {SG_T}, {n_}")
    add(f"s_cselect_b64 s[{desc + 2}:{desc + 3}], {spair(tdm, 10)}, s[{SG_A[p4]}:{SG_A[p4] + 1}]")
    add(f"s_mov_b32 s{desc + 1}, {sreg(tdm, 12 + p4)}")
    tdm_i = add(f"tensor_load_to_lds s[{desc}:{desc + 3}], {ops['g1']}")
    add(f"s_add_nc_u64 s[{SG_A[p4]}:{SG_A[p4] + 1}], s[{SG_A[p4]}:{SG_A[p4] + 1}], {spair(tdm, 8)}")
    vsel = SG_U if p4 == 0 else SG_V[p4 - 1]
    add(f"v_add_nc_u32 {vr(L.vo[P])}, {vsel}, {ops['vl']}")
    if masked:
        add(f"v_subrev_nc_u32 {vr(L.thr[1])}, {SG_M}, {ops['dl']}")
        add(f"v_add_nc_u32 {vr(L.thr[0])}, 16, {vr(L.thr[1])}")
        add(f"v_add_nc_u32 {vr(L.thr[2])}, -16, {vr(L.thr[1])}")
    # A: dS^T(s - 1) (buffer P ^ 1, written before the previous barrier), Q^T(s) 0..3
    for t in dsa_texts(L, ops, P ^ 1):
        proto.append(add(t))
    qt_a = [add(t) for t in tr_texts(L, ops, "q", slot, range(4))]
    proto += qt_a
    sdp0, sdp1 = sdp_wmmas(L, 0), sdp_wmmas(L, 1)
    for t in sdp0:
        W.append(add(t))
    # B: Q^T 4..7, dO^T 4..7 (dO^T 0..3 came in the previous step)
    qt_b = [add(t) for t in tr_texts(L, ops, "q", slot, range(4, 8)) + tr_texts(L, ops, "d", slot, range(4, 8))]
    proto += qt_b
    # readback(s + 1) k-step dt is listed (and ordered) RB_GAP WMMAs after the last kv tile 1 WMMA reading it: the
    # va_vdst bound of its WAR against those WMMAs is then >= RB_GAP, which in-flight VALU rarely exceed
    rb, rb_at = {}, {16 + 4 * dt + 3 + RB_GAP: dt for dt in range(4)}

    def wm(t):
        W.append(add(t))
        dt = rb_at.get(len(W) - 1)
        if dt is not None:
            rb[dt] = [add(x) for x in rb_texts(L, ops, nslot, dt)]

    for t in sdp1:
        wm(t)
    soft0 = [add(t) for hh in range(2) for t in softmax_texts(L, hh, 0, masked)]
    # C: dQ(s - 1) || softmax kv tile 1
    for t in dq_wmmas(L):
        wm(t)
    soft1 = [add(t) for hh in range(2) for t in softmax_texts(L, hh, 1, masked)]
    # D: the dS^T(s) stores (buffer P) after the whole softmax (one va_vdst(0) covers them all), then the dQ(s - 1)
    # atomics (their WMMAs retired by that wait) and the LD records of s + 1 (the softmax read them)
    stores = []
    for kh in range(2):
        for hh in range(2):
            o = P * DS_B + 16 * kh * SP + 32 * hh
            stores.append(add(f"ds_store_b128 {ops['dss']}, {vr(L.DS[kh] + 4 * hh, 4)}{offset(o)}"))
    proto += stores
    atoms = [add(t) for t in atomic_texts(L, ops, L.vo[P])]
    lds = [add(t) for t in ld_texts(L, ops, nslot)]
    for kh in range(2):
        for dt in range(8):
            if len(W) == SIG_BEFORE:  # the signal's nodes stand before the WMMA they must precede (edges point forward)
                tw = add("s_wait_tensorcnt 0x1")
                fence = add("s_wait_dscnt 0x0", tag="fence")
                sig = add("s_barrier_signal -1")
            wm(wmma(L.accV[dt][kh], L.dOT[dt], L.P[kh], L.accV[dt][kh]))
    # E: dK^T; dO^T(s + 1) 0..3 RB_GAP WMMAs after the last dV WMMA reading them (W 56 + 3)
    dnext = []
    for kh in range(2):
        for dt in range(8):
            wm(wmma(L.accK[dt][kh], L.QT[dt], L.DS[kh], L.accK[dt][kh]))
            if len(W) - 1 == 59 + RB_GAP:
                dnext = [add(t) for t in tr_texts(L, ops, "d", nslot, range(4))]
    if p4 == 3:  # iteration end: the scalars of the next iteration
        add(f"s_mov_b32 {SG_U}, {SG_V[3]}")
        for v in SG_V:
            add(f"s_add_co_i32 {v}, {v}, 0xffffc000")
        if masked:
            add(f"s_add_co_i32 {SG_M}, {SG_M}, -32")
    bw = add("s_barrier_wait -1")
    build_dag(nodes)
    # protocol: every read of slot s, the dS^T(s - 1) reads and the stores complete before the signal
    for i in proto:
        add_order(nodes, i, fence)
    add_order(nodes, tdm_i, tw)  # tensorcnt 1 after this step's op: op(s + 2) landed before the signal
    add_order(nodes, tw, fence)
    add_order(nodes, fence, sig)
    add_order(nodes, sig, W[SIG_BEFORE])
    for j in soft1:
        for i in stores:
            add_order(nodes, j, i)
    for i in stores:
        add_order(nodes, i, atoms[0])
        for j in lds:
            add_order(nodes, i, j)
        add_order(nodes, i, W[ST_BEFORE])
    # deadlines: the slot-s operands early (the signal's fence then waits for the stores only), the next step's
    # operands, the atomics one per window from WMMA 49
    for i in qt_a:
        add_order(nodes, i, W[QA_BEFORE])
    for i in qt_b:
        add_order(nodes, i, W[QB_BEFORE])
    for dt in range(4):
        for i in rb[dt]:
            add_order(nodes, i, W[RB_BEFORE[dt]])
    for i in lds:
        add_order(nodes, i, W[LD_BEFORE])
    for a, i in enumerate(atoms):
        if 49 + a < len(W):
            add_order(nodes, i, W[49 + a])
    # the barrier wait ends the step
    for i in range(len(nodes) - 1):
        if i != bw:
            add_order(nodes, i, bw)
    return nodes, W, proto, fence


def fix_fence(order, proto_nodes, fence_node):
    """Set the dscnt of the signal's fence: the LDS ops issued after the last protocol op may stay in flight."""
    pos = order.index(fence_node)
    last = max(order.index(p) for p in proto_nodes)
    if last > pos:
        raise RuntimeError("a protocol LDS op after the fence")
    n = sum(1 for x in order[last + 1 : pos] if x.kind in ("DS_LD", "DS_ST"))
    fence_node.text = f"s_wait_dscnt 0x{n:x}"


def region_step(L, ops, p4, masked):
    """(instruction texts, modelled cycles) of one step."""
    nodes, W, proto, fence = step_nodes(L, ops, p4, masked)
    order, cycles = schedule_slot(nodes, W, bar_deps=proto)
    fix_fence(order, [nodes[i] for i in proto], nodes[fence])
    return [n.text for n in order], cycles


# --------------------------------------------------------------------------------------------------------------
# INIT, the loops, FINAL
# --------------------------------------------------------------------------------------------------------------
def region_init(L, ops):
    """INIT: (c1, c1); accumulators = 0; the resident B operands; readback(0); dO^T(0) 0..3; the loop scalars."""
    tdm, sc = ops["tdm"], ops["sc"]
    out = [f"v_mov_b32 {vr(L.c1)}, {sreg(sc, 7)}", f"v_mov_b32 {vr(L.c1 + 1)}, {sreg(sc, 7)}"]
    for t in L.acc:
        for q in range(0, 8, 2):
            out.append(f"v_mov_b64 {vr(t + q, 2)}, 0")
    for kh in range(2):
        for dt in range(4):
            out += ld_rows(L.kf[kh][dt], ops["kr"], rows_imm(16 * kh, 32 * dt, 0))
            out += ld_rows(L.vf[kh][dt], ops["vr"], rows_imm(16 * kh, 32 * dt, 0))
    for jj in range(4):
        for c in range(4):
            out += ld_tr(L.kt[jj][c], ops["kc"], tr_imm(32 * c, 16 * jj, 0), XP)
    for dt in range(4):
        out += rb_texts(L, ops, 0, dt)
    out += ld_texts(L, ops, 0)
    out += tr_texts(L, ops, "d", 0, range(4))
    for d in SG_DESC:
        out.append(f"s_mov_b32 s{d}, 1")
    for p4 in range(4):
        out.append(f"s_mov_b64 s[{SG_A[p4]}:{SG_A[p4] + 1}], {spair(tdm, 2 * p4)}")
        out.append(f"s_mov_b32 {SG_V[p4]}, {sreg(sc, 2 + p4)}")
    out += [f"s_mov_b32 {SG_U}, {sreg(sc, 2)}", f"s_mov_b32 {SG_S}, 0", f"s_mov_b32 {SG_M}, {sreg(sc, 6)}"]
    return out


def region_final(L, ops):
    """FINAL: dQ(n - 1) from dS^T buffer 1 and its atomics (tile n - 1: SG_U), then dV^T and scale * dK^T to bf16
    (in place) and whole-row buffer stores (lane voffsets ev / ek per kv tile, + 32 dt)."""
    out = dsa_texts(L, ops, 1)
    out += dq_wmmas(L)
    out.append(f"v_add_nc_u32 {vr(L.vo[0])}, {SG_U}, {ops['vl']}")
    out += atomic_texts(L, ops, L.vo[0])
    out += [f"v_mov_b32 {vr(L.c1)}, {ops['scale']}", f"v_mov_b32 {vr(L.c1 + 1)}, {ops['scale']}"]
    for kh in range(2):
        for isk in (0, 1):
            for dt in range(8):
                a = (L.accK if isk else L.accV)[dt][kh]
                if isk:
                    out += [f"v_pk_mul_f32 {vr(a + q, 2)}, {vr(a + q, 2)}, {vr(L.c1, 2)}" for q in (0, 2, 4, 6)]
                out += [f"v_cvt_pk_bf16_f32 {vr(a + q)}, {vr(a + 2 * q)}, {vr(a + 2 * q + 1)}" for q in range(4)]
                vo, rs = (ops[f"ek{kh}"], ops["rk"]) if isk else (ops[f"ev{kh}"], ops["rv"])
                out.append(f"buffer_store_b128 {vr(a, 4)}, {vo}, {rs}, null offen{offset(32 * dt)}")
    return out


def region_body(L, ops, lp):
    """[(name, label, texts, successors)] of the region in layout order I F D M X, and the modelled cycles per step.
    I (INIT) skips to D when nf = 0; F (the FULL steps, 4 per trip) loops while s < nf; D leaves for X once s = n;
    M (the diagonal steps, 4 per trip) loops while s < n; X is FINAL."""
    sc = ops["sc"]
    n_, nf = sreg(sc, 0), sreg(sc, 1)
    cyc = {}
    full, diag = [], []
    for p4 in range(4):
        t, cyc[f"F{p4}"] = region_step(L, ops, p4, False)
        full += t
        t, cyc[f"M{p4}"] = region_step(L, ops, p4, True)
        diag += t
    init = region_init(L, ops) + [f"s_cmp_eq_u32 {nf}, 0", f"s_cbranch_scc1 {lp}_D"]
    full += [f"s_add_co_i32 {SG_S}, {SG_S}, 4", f"s_cmp_lt_i32 {SG_S}, {nf}", f"s_cbranch_scc1 {lp}_F"]
    head = [f"s_cmp_ge_i32 {SG_S}, {n_}", f"s_cbranch_scc1 {lp}_X"]
    diag += [f"s_add_co_i32 {SG_S}, {SG_S}, 4", f"s_cmp_lt_i32 {SG_S}, {n_}", f"s_cbranch_scc1 {lp}_M"]
    blocks = [
        ("I", init, ["F", "D"]),
        ("F", full, ["F", "D"]),
        ("D", head, ["M", "X"]),
        ("M", diag, ["M", "X"]),
        ("X", region_final(L, ops), []),
    ]
    return [(name, f"{lp}_{name}", texts, succ) for name, texts, succ in blocks], cyc
