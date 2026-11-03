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


def alloc_sgprs(avoid):
    """Place the loop state in s72..s103 away from `avoid` (SGPRs an in-flight shell TDM / VMEM op may still read at
    the region entry: writing one would need s_wait_xcnt 0, i.e. wait for that op). Two 4-aligned descriptor sets,
    four 2-aligned address pairs, eight singles; falls back to the fixed map when they do not fit."""
    global SG_DESC, SG_A, SG_V, SG_U, SG_S, SG_M, SG_T
    free = [i for i in range(72, 104) if f"s{i}" not in avoid]
    used = set()

    def take(n, align):
        for i in free:
            if i % align == 0 and all(j in free and j not in used for j in range(i, i + n)):
                used.update(range(i, i + n))
                return i
        return None

    desc = [take(4, 4), take(4, 4)]
    pairs = [take(2, 2) for _ in range(4)]
    singles = [take(1, 1) for _ in range(8)]
    if None in desc + pairs + singles:
        return False
    SG_DESC, SG_A = tuple(desc), tuple(pairs)
    SG_V = tuple(f"s{i}" for i in singles[:4])
    SG_U, SG_S, SG_M, SG_T = (f"s{i}" for i in singles[4:])
    return True


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
    "WMMA": 36,
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
# RAW latencies beyond LAT that tools/fwd1gen turns on (ATT of the generated forward, 2026-11-03; None: off, the
# backward's model): a packed-f32 VALU result (v_pk_*) is read PK_LAT cycles after its issue at the earliest; an SGPR /
# vcc a VALU writes (v_cmp_*_e64 s, ...) reaches a SALU / branch / memory reader SDST_LAT cycles after it
PK_LAT = None
SDST_LAT = None
# the TRANS unit completes one instruction per TRANS_RATE cycles (wave32 v_exp: quarter rate), each LAT["TRANS"] after
# its issue at the earliest (ATT fit of the forward: about 18 and 4); 0: no throughput model
TRANS_RATE = 0


def raw_lat(p, c):
    """Cycles from the issue of producer p to the issue of c that reads p's result (register dependence)."""
    if p.kind == "WMMA" and c.kind == "WMMA":
        return WMMA_CHAIN
    if p.kind == "VALU":
        if SDST_LAT and (c.suses & p.sdefs) and c.kind not in ("VALU", "TRANS", "WMMA"):
            return SDST_LAT
        if PK_LAT and p.text.startswith("v_pk_"):
            return max(PK_LAT, LAT["VALU"])
    return LAT[p.kind]
WAR_XDL, MIN_XDL_AFTER = 24, 14  # va_vdst(14) on a WAR against a WMMA: free whatever VALU is in flight
DS_MAX = 48
# issue model (one wave per SIMD; issue-to-issue medians of the generated g3 step, ATT 2026-11-03): one instruction per
# cycle; a WMMA every WMMA_CYC cycles at most; behind a memory instruction (LDS or VMEM) the next one of the same class
# (or an s_wait) issues the next cycle, anything else DS_BLOCK cycles after it (L->L 1, A->A 1, L->W / L->V / L->A /
# A->L / A->W 4); the LDS completes the wave's operations in order, one per LDS_RATE cycles, each at least LDS_LAT
# after its issue
WMMA_CYC = 8
DS_BLOCK = 4
LDS_LAT, LDS_RATE = 120, 8
BAR_GAP = 12
VM_BURST, VM_GAP = None, 8  # int: at most VM_BURST VMEM ops in a row, then VM_GAP cycles without one
MEM = ("DS_LD", "DS_ST", "VM_LD", "VM_ST", "TDM")


def set_model(**kw):
    """Override model parameters (build.py options)."""
    for k, v in kw.items():
        if v is not None:
            globals()[k] = v


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
                lat = raw_lat(nodes[p], n)
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


def schedule_slot(nodes, wseq, fence=None, bar_deps=()):
    """(ordered nodes, modelled cycles): a cycle-stepped list scheduler around the in-order WMMA stream wseq on the issue
    model above. An LDS instruction issues only if its DS_BLOCK fits before the next WMMA's cycle (unless that WMMA is
    not ready anyway); the fence node (the barrier signal's s_wait_dscnt) issues once the LDS ops bar_deps completed;
    s_barrier_wait BAR_GAP WMMAs after the signal; at most DS_MAX LDS loads in flight. Ties: barrier and wait nodes
    first, then the earliest deadline (the first WMMA that needs the node, transitively), the longest critical path,
    program order."""
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
        cp[i] = (WMMA_CYC if nodes[i].kind == "WMMA" else 1) + max((l + cp[s] for s, l in succ[i]), default=0)
    npred = [len(nd.preds) for nd in nodes]
    earliest = [0] * n
    ready = set(i for i in range(n) if npred[i] == 0)
    out, wmma_at, done = [], {}, {}
    t, last_w, wk, trans_at, ds_last_done = 0, -100, 0, -100, 0
    trans_done = -100
    mem_t, mem_c, ds_next = -100, None, 0
    vm_run, vm_last = 0, -100
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

    def next_wmma_ready():
        nw = wseq[wk] if wk < len(wseq) else None
        return nw is not None and nw in ready

    def mclass(k):
        return "DS" if k in ("DS_LD", "DS_ST") else "VM" if k in ("VM_LD", "VM_ST") else k

    def port_ok(k):
        """the issue port behind the last memory instruction (at mem_t, class mem_c)"""
        if mem_c is None:
            return True
        if mclass(k) == mem_c or k == "WAIT_CNT":
            return t >= mem_t + 1
        return t >= mem_t + DS_BLOCK

    def avail(i):
        k = nodes[i].kind
        if earliest[i] > t or not port_ok(k):
            return False
        if k == "TRANS" and t == trans_at + 1:
            return False
        if k in MEM:
            if not xdl_ok(i):
                return False
            if k in ("DS_LD", "DS_ST") and sum(1 for x in done.values() if x > t) >= DS_MAX:
                return False
            if k in ("VM_LD", "VM_ST") and VM_BURST and vm_run >= VM_BURST and t < vm_last + VM_GAP:
                return False
            # the WMMA after it waits DS_BLOCK: only if it fits before the WMMA's cycle (or the WMMA is not ready)
            if t + DS_BLOCK > last_w + WMMA_CYC and next_wmma_ready() and wk < len(wseq):
                return False
            return True
        if i == fence:
            return all(j in done and done[j] <= t for j in bar_deps)
        if k == "BAR_W":  # BAR_GAP WMMAs after the signal, unless its deadline WMMA comes first
            if not any(nodes[j].kind == "BAR_S" for j in range(n)):  # its signal is in an earlier block
                return True
            return wk >= bs_wk + BAR_GAP or wk >= len(wseq) or (bs_wk < 10**6 and wk >= WAIT_BEFORE)
        return True

    while len(out) < n:
        pick = None
        nw = wseq[wk] if wk < len(wseq) else None
        if nw is not None and nw in ready and earliest[nw] <= t and t >= last_w + WMMA_CYC and port_ok("WMMA"):
            pick = nw
        else:
            cand = [i for i in ready if nodes[i].kind != "WMMA" and avail(i)]
            if cand:
                pick = max(
                    cand, key=lambda i: (nodes[i].kind in ("BAR_S", "BAR_W", "WAIT_CNT"), -dl[i], cp[i], -i)
                )
        if pick is None:
            t += 1
            if t > 10**6:
                nw = wseq[wk] if wk < len(wseq) else None
                miss = [p for p, _ in nodes[nw].preds if p not in set(id(x) for x in out) and nodes[p] not in out][:5] if nw is not None else []
                raise RuntimeError(f"slot scheduler deadlock at WMMA {wk}: ready {[(i, nodes[i].text[:40]) for i in sorted(ready)][:6]}, "
                                   f"next WMMA missing preds {[(p, nodes[p].text[:50]) for p in miss]}")
            continue
        nd = nodes[pick]
        ready.discard(pick)
        k = nd.kind
        done_at = None
        if k in ("DS_LD", "DS_ST"):
            done_at = max(t + LDS_LAT, ds_last_done + LDS_RATE)
            ds_last_done = done_at
            ds_seq[pick] = ds_issued
            ds_issued += 1
            done[pick] = done_at
        if k in MEM:
            mem_t, mem_c = t, mclass(k)
        if k in ("VM_LD", "VM_ST"):
            vm_run = vm_run + 1 if t - vm_last < VM_GAP else 1
            vm_last = t
        for p, _ in nd.preds:
            if nodes[p].kind == "DS_LD" and p in ds_seq and nd.vuses & nodes[p].vdefs:
                ds_retired = max(ds_retired, ds_seq[p])
        if pick == fence:
            ds_retired = max([ds_retired] + [ds_seq[j] for j in bar_deps if j in ds_seq])
        if k == "BAR_S":
            bs_wk = wk
        out.append(nd)
        if k == "WMMA":
            last_w = t
            wmma_at[pick] = wk
            wk += 1
        if k == "TRANS":
            trans_at = t
            if TRANS_RATE:
                trans_done = max(t + LAT["TRANS"], trans_done + TRANS_RATE)
        for s_, l in succ[pick]:
            if k == "DS_LD" and nodes[s_].vuses & nd.vdefs:
                lat = done_at - t
            elif k == "TRANS" and TRANS_RATE and nodes[s_].vuses & nd.vdefs:
                lat = trans_done - t
            elif k == "WMMA":
                lat = WMMA_CHAIN if nodes[s_].kind == "WMMA" and (nodes[s_].vuses & nd.vdefs) else max(l, 1)
            else:
                lat = max(l, raw_lat(nd, nodes[s_])) if (nodes[s_].vuses & nd.vdefs or nodes[s_].suses & nd.sdefs) else l
            earliest[s_] = max(earliest[s_], t + lat)
            npred[s_] -= 1
            if npred[s_] == 0:
                ready.add(s_)
        t += 1
    return out, max(t, last_w + WMMA_CYC)


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
# deadlines (WMMA indices of the step): Q^T 0..3 by QA_BEFORE, Q^T 4..7 and dO^T 4..7 by QB_BEFORE (the signal's
# fence then waits for the stores only); readback(s + 1) k-step dt by RB_BEFORE[dt], its LD records by LD_BEFORE; the
# dS^T stores by ST_BEFORE; the dQ atomic a by WMMA 49 + a; the signal by SIG_BEFORE, the barrier wait by WAIT_BEFORE.
# A load that overwrites a WMMA operand register stands RB_GAP WMMAs after the WMMA's last reader.
QA_BEFORE, QB_BEFORE = 24, 40
QA_AFTER, QB_AFTER = 8, 14
ATOM_AFTER = None  # int: the dQ atomics listed behind this WMMA (>= 14 XDL after the last dQ WMMA, 47 -> 61), spread to the end
RB_BEFORE = (44, 52, 60, 68)
LD_BEFORE = 64
ST_BEFORE = 52
SIG_BEFORE = 60
WAIT_BEFORE = 72
TW_AFTER = 50  # the signal's s_wait_tensorcnt after this WMMA
RB_GAP = 14  # va_vdst(14): no wait whatever VALU is in flight


def step_nodes(L, ops, p4, masked):
    """(nodes, WMMA order, protocol LDS ops, fence) of one step with P4 = p4 (program order = semantics). The step
    starts with the dQ(s - 1) operands in VGPRs (dS^T(s - 1) read behind the previous step's barrier wait) and op(s + 3)
    in flight; it ends with readback(s + 1), dO^T(s + 1) 0..3, dS^T(s) and op(s + 4) issued."""
    nodes = []

    def add(text, tag=None):
        nodes.append(mk(text, tag))
        return len(nodes) - 1

    P = p4 & 1
    slot, nslot = p4, (p4 + 1) & 3
    desc = SG_DESC[P]
    tdm, sc = ops["tdm"], ops["sc"]
    n_ = sreg(sc, 0)
    W, proto = [], []
    vsel = SG_U if p4 == 0 else SG_V[p4 - 1]
    add(f"v_add_nc_u32 {vr(L.vo[P])}, {vsel}, {ops['vl']}")
    if masked:
        add(f"v_subrev_nc_u32 {vr(L.thr[1])}, {SG_M}, {ops['dl']}")
        add(f"v_add_nc_u32 {vr(L.thr[0])}, 16, {vr(L.thr[1])}")
        add(f"v_add_nc_u32 {vr(L.thr[2])}, -16, {vr(L.thr[1])}")
    # A: S / dP of kv tile 0 || Q^T(s) 0..3
    # (listed QA_AFTER / QB_AFTER WMMAs into the step: they overwrite Q^T / dO^T(s - 1), which the previous step's
    # last dK / dV WMMAs read; the emitter's va_vdst bound grows with every WMMA in between)
    sdp0, sdp1 = sdp_wmmas(L, 0), sdp_wmmas(L, 1)
    qt_a, qt_b = [], []
    hooks = {}

    def load_qa():
        qt_a.extend(add(t) for t in tr_texts(L, ops, "q", slot, range(4)))

    def load_qb():
        qt_b.extend(add(t) for t in tr_texts(L, ops, "q", slot, range(4, 8)) + tr_texts(L, ops, "d", slot, range(4, 8)))

    hooks[QA_AFTER], hooks[QB_AFTER] = load_qa, load_qb
    for t in sdp0:
        k = len(W)
        if k in hooks:
            hooks[k]()
        W.append(add(t))
    # B: S / dP of kv tile 1 || softmax kv tile 0; Q^T 4..7, dO^T 4..7; readback(s + 1) dt by dt, RB_GAP WMMAs after
    # the last kv tile 1 WMMA reading k-step dt
    rb, rb_at = {}, {16 + 4 * dt + 3 + RB_GAP: dt for dt in range(4)}

    def wm(t):
        k = len(W)
        if k in hooks:
            hooks[k]()
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
    # atomics and the LD records of s + 1 (the softmax read them)
    stores = []
    for kh in range(2):
        for hh in range(2):
            o = P * DS_B + 16 * kh * SP + 32 * hh
            stores.append(add(f"ds_store_b128 {ops['dss']}, {vr(L.DS[kh] + 4 * hh, 4)}{offset(o)}"))
    proto += stores
    atoms = []

    def atomics():
        atoms.extend(add(t) for t in atomic_texts(L, ops, L.vo[P]))

    lds = [add(t) for t in ld_texts(L, ops, nslot)]
    sigs = {}

    def signal():
        sigs["tw"] = add("s_wait_tensorcnt 0x1")  # op(s + 2) landed (op(s + 3) may fly) before the signal
        sigs["fence"] = add("s_wait_dscnt 0x0", tag="fence")
        sigs["sig"] = add("s_barrier_signal -1")

    post = {}

    def wait():
        post["bw"] = add("s_barrier_wait -1")
        # op(s + 4) into slot P4 (tile s's slot: every wave's reads of it preceded the signal), clamped to tile n - 1
        add(f"s_add_co_i32 {SG_T}, {SG_S}, {p4 + 4}")
        add(f"s_cmp_ge_i32 {SG_T}, {n_}")
        add(f"s_cselect_b64 s[{desc + 2}:{desc + 3}], {spair(tdm, 10)}, s[{SG_A[p4]}:{SG_A[p4] + 1}]")
        add(f"s_mov_b32 s{desc + 1}, {sreg(tdm, 12 + p4)}")
        post["tdm"] = add(f"tensor_load_to_lds s[{desc}:{desc + 3}], {ops['g1']}")
        add(f"s_add_nc_u64 s[{SG_A[p4]}:{SG_A[p4] + 1}], s[{SG_A[p4]}:{SG_A[p4] + 1}], {spair(tdm, 8)}")
        # dS^T(s) (buffer P, every wave's stores preceded the signal): the next step's dQ operands
        post["dsa"] = [add(t) for t in dsa_texts(L, ops, P)]

    hooks[SIG_BEFORE] = signal
    hooks[WAIT_BEFORE] = wait
    if ATOM_AFTER is None:
        atomics()  # g6b: right behind the stores, the a-th by WMMA 49 + a
    else:
        hooks[ATOM_AFTER] = atomics
    for kh in range(2):
        for dt in range(8):
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
    build_dag(nodes)
    tw, fence, sig, bw = sigs["tw"], sigs["fence"], sigs["sig"], post["bw"]
    proto += qt_a + qt_b
    # the Q^T / dO^T(s) loads stay behind QA_AFTER / QB_AFTER WMMAs (cross-step WAR against the previous dK / dV)
    if QA_AFTER > 8 or QB_AFTER > 14:
        for i in qt_a:
            add_order(nodes, W[QA_AFTER - 1], i)
        for i in qt_b:
            add_order(nodes, W[QB_AFTER - 1], i)
    # protocol: every read of slot s and the dS^T(s) stores complete before the signal; the wait after it
    for i in proto:
        add_order(nodes, i, fence)
    add_order(nodes, W[TW_AFTER], tw)  # late: op(s + 2) has (mostly) landed by then
    add_order(nodes, tw, fence)
    add_order(nodes, fence, sig)
    add_order(nodes, sig, W[SIG_BEFORE])
    add_order(nodes, sig, bw)
    add_order(nodes, bw, W[WAIT_BEFORE])
    add_order(nodes, bw, post["tdm"])
    for i in post["dsa"]:
        add_order(nodes, bw, i)
    for j in soft1:
        for i in stores:
            add_order(nodes, j, i)
    for i in stores:
        if atoms and i < atoms[0]:
            add_order(nodes, i, atoms[0])
        for j in lds:
            add_order(nodes, i, j)
        add_order(nodes, i, W[ST_BEFORE])
    for i in qt_a:
        add_order(nodes, i, W[QA_BEFORE])
    for i in qt_b:
        add_order(nodes, i, W[QB_BEFORE])
    for dt in range(4):
        for i in rb[dt]:
            add_order(nodes, i, W[RB_BEFORE[dt]])
    for i in lds:
        add_order(nodes, i, W[LD_BEFORE])
    na = len(atoms)
    for a, i in enumerate(atoms):  # spread over the WMMAs after ATOM_AFTER (an atomic's issue holds the next memory op)
        k = 49 + a if ATOM_AFTER is None else ATOM_AFTER + 1 + (a * (len(W) - ATOM_AFTER - 1)) // na
        if k < len(W):
            add_order(nodes, i, W[k])
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
    order, cycles = schedule_slot(nodes, W, fence=fence, bar_deps=proto)
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
    # op 3 into slot 3 (the prologue's barrier already ordered it after every wave's start; step 1's descriptor set),
    # and dS^T(-1) = 0 (buffer 1, zeroed by the shell): step 0's dQ operands
    d = SG_DESC[1]
    out += [f"s_mov_b32 s{d + 1}, {sreg(tdm, 15)}", f"s_mov_b64 s[{d + 2}:{d + 3}], {spair(sc, 8)}",
            f"tensor_load_to_lds s[{d}:{d + 3}], {ops['g1']}"]  # fmt: skip
    out += dsa_texts(L, ops, 1)
    return out


def region_final(L, ops):
    """FINAL: dQ(n - 1) (its operands dS^T(n - 1) read behind the last step's barrier wait) and its atomics (tile n - 1:
    SG_U), then dV^T and scale * dK^T to bf16 (in place) and whole-row buffer stores (lane voffsets ev / ek per kv
    tile, + 32 dt)."""
    out = dq_wmmas(L)
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
