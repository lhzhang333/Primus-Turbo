#!/usr/bin/env python3
"""resched.py -- post-RA re-scheduling of the main loop of an LLVM-compiled MX kernel (lab tool, CPU only).

The kernel source is compiled with the lab's xi flags plus -amdgpu-enable-vopd=0 (asmkit has no VOPD); the main loop
body of each named kernel (a label, its back branch, no label or branch in between) is taken apart into its real
instructions (LLVM's s_set_vgpr_msb, s_wait_alu, v_nop and register-driven s_wait_*cnt dropped), put in a new order
(--mode identity | list) that keeps every register / scalar / memory dependence of LLVM's register allocation, and
re-emitted through asmkit (tools/asmkit of the gfx1250_mla package), which derives the s_set_vgpr_msb, the expert-mode
waits and the hazard v_nops of the new order for every flow that can reach the block (fixed point over the back edge).
The loop entry and exit get a full drain (every counter, va_vdst / vm_vsrc 0, 5 v_nop): the code around the loop
keeps LLVM's waits, which assume LLVM's order inside the loop.

    resched.py --src k_mla_bwd_mx5.hip --kernels k_dkdv_mx,k_dkdv_mxr --tag 5r0 [--mode list] [-D NAME=V ...]

writes kern/build/k_mla_bwd_mx<tag>.hsaco (mxlab loads it with tag=<tag>) and kern/build/rs/<tag>/ (the .s files,
the listing, a report).
"""

import argparse
import collections
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KERN = HERE.parent / "kern"
BUILD = KERN / "build"
TOOLS = HERE.parents[2] / "primus_turbo" / "hipkittens" / "gfx1250_mla" / "tools"
sys.path.insert(0, str(TOOLS))
from asmkit import build as B  # noqa: E402
from asmkit import emit as EM  # noqa: E402
from asmkit import expert as E  # noqa: E402
from asmkit import isa  # noqa: E402
from asmkit import verify_obj as VO  # noqa: E402

FLAGS = ["-x", "hip", "--offload-arch=gfx1250", "--cuda-device-only", "--no-gpu-bundle-output", "-O3", "-std=c++20",
         "-mllvm", "-amdgpu-expert-scheduling-mode", "-mllvm", "-greedy-regclass-priority-trumps-globalness",
         "-mllvm", "-misched=gcn-max-ilp", "-mllvm", "-amdgpu-enable-vopd=0"]
DERIVED_CNT = {"dscnt", "loadcnt", "storecnt", "xcnt"}
DRAIN = ["\ts_wait_loadcnt 0x0", "\ts_wait_storecnt 0x0", "\ts_wait_dscnt 0x0", "\ts_wait_xcnt 0x0",
         "\ts_wait_alu depctr_va_vdst(0) depctr_vm_vsrc(0)"] + ["\tv_nop"] * 5


def cxx():
    import shutil
    return shutil.which("amdclang++") or "amdclang++"


def compile_s(src, defines, extra, tmp):
    tmp.mkdir(parents=True, exist_ok=True)
    for f in tmp.glob("*"):
        f.unlink()
    cmd = [cxx(), *FLAGS, *extra, "-save-temps=obj", *[f"-D{d}" for d in defines], "-o", str(tmp / "cc.hsaco"),
           str(KERN / src)]
    r = subprocess.run(cmd, cwd=KERN, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"compile failed:\n{r.stderr[-4000:]}")
    return cmd, next(tmp.glob("*-gfx1250.s"))


# ------------------------------------------------------------------------------------------------- loop extraction
def main_loop(ins, kernel):
    """(label line, back-branch line) of the largest loop of kernel whose body is one straight-line block."""
    k0 = next(i for i, x in enumerate(ins) if x.label == kernel)
    k1 = next(i for i in range(k0, len(ins)) if ins[i].text.strip().startswith(".Lfunc_end"))
    labels, best = {}, None
    for i in range(k0, k1):
        x = ins[i]
        if x.cls == "LABEL":
            labels[x.label] = i
        elif x.mn.startswith(("s_cbranch", "s_branch")):
            a = labels.get(isa.strip_comment(x.ops[0]).strip())
            if a is None:
                continue
            if any(ins[j].cls == "LABEL" or ins[j].mn.startswith(("s_cbranch", "s_branch")) for j in range(a + 1, i)):
                continue
            n = sum(1 for j in range(a + 1, i + 1) if ins[j].mn)
            if best is None or n > best[2]:
                best = (a, i, n)
    if best is None:
        raise RuntimeError(f"{kernel}: no single-block loop")
    return best[0], best[1]


def is_derived(x):
    if x.cls in ("MSB", "VNOP", "CLAUSE", "DELAY"):
        return True
    if x.cls == "WAIT_ALU":
        f = isa.wait_fields(x)
        if set(f) - {"va_vdst", "vm_vsrc"}:
            raise RuntimeError(f"s_wait_alu field not re-derived by asmkit: {x.text.strip()}")
        return True
    if x.cls == "WAIT_CNT":
        f = isa.cnt_fields(x)
        return bool(f) and set(f) <= DERIVED_CNT
    return False


class RItem(EM.Item):
    """asmkit's Item plus attributes of the scheduler (req: the s_set_vgpr_msb fields it needs)."""


def body_items(ins, a, b):
    items = []
    for j in range(a + 1, b + 1):
        x = ins[j]
        if not x.mn or x.cls in ("DIR", "LABEL") or is_derived(x):
            continue
        if "::" in x.mn:
            raise RuntimeError(f"VOPD in the loop: {x.text.strip()}")
        if x.mn == "s_nop":
            raise RuntimeError(f"s_nop in the loop (hazard not modelled): {x.text.strip()}")
        n0 = EM.make_item(x.text)
        n = RItem()
        for k in EM.Item.__slots__:
            setattr(n, k, getattr(n0, k))
        err = isa.coverage_errors(n.x)
        if err:
            raise RuntimeError(f"coverage {err}: {x.text.strip()}")
        n.pos = len(items)
        items.append(n)
    if items[-1].kind != "BR":
        raise RuntimeError("the body does not end with its branch")
    return items


# ------------------------------------------------------------------------------------------------- copy coalescing
def vtok(lo, n):
    """LLVM's operand text of VGPRs lo .. lo + n - 1 (the low-bank name, the true registers in a comment above v255)."""
    hi = lo + n - 1
    if lo // 256 != hi // 256:
        return None
    if n == 1:
        return f"v{lo}" if lo < 256 else f"v{lo % 256} /*v{lo}*/"
    return f"v[{lo}:{hi}]" if lo < 256 else f"v[{lo % 256}:{hi % 256}] /*v[{lo}:{hi}]*/"


def remake(it, text):
    n0 = EM.make_item(text)
    n = RItem()
    for k in EM.Item.__slots__:
        setattr(n, k, getattr(n0, k))
    n.pos = it.pos
    return n


def coalesce(items, log):
    """Remove v_mov_b32 / v_mov_b64 copies whose source the body defines with one instruction X (all of X's VGPR defs,
    no other reader of the source, the destination untouched between X and the copy): X writes the destination."""
    removed = 0
    while True:
        n = len(items)
        uses = collections.defaultdict(list)
        defs = collections.defaultdict(list)
        for k, it in enumerate(items):
            for r in it.vuses:
                uses[r].append(k)
            for r in it.vdefs:
                defs[r].append(k)
        done = None
        for m, it in enumerate(items):
            mn = it.x.mn
            if mn not in ("v_mov_b32_e32", "v_mov_b64_e32") or it.x.rmw or len(it.x.ops) != 2:
                continue
            src, dst = sorted(set(it.vuses)), sorted(set(it.vdefs))
            if not src or len(src) != len(dst) or it.suses or "::" in mn:
                continue
            if isa.regs_of(isa.first_tok(it.x.ops[1])) != src or isa.regs_of(isa.first_tok(it.x.ops[0])) != dst:
                continue  # modifiers / constants
            ks = [max([k for k in defs[r] if k < m], default=-1) for r in src]
            x = ks[0]
            if x < 0 or any(k != x for k in ks):
                continue
            X = items[x]
            if sorted(set(X.vdefs)) != src or X.x.rmw or X.kind not in ("VALU", "TRANS", "DS_LD") or "::" in X.x.mn:
                continue
            if isa.regs_of(isa.first_tok(X.x.ops[0])) != src:
                continue
            # the source: read only by the copy until its next def; not read across the back edge
            bad = False
            for r in src:
                nd = min([k for k in defs[r] if k > m], default=None)
                for u in uses[r]:
                    if u == m or u <= x:
                        if u <= x and nd is None and u < min(defs[r]):  # read before its first def: a carried value
                            bad = True
                        continue
                    if nd is None or u <= nd:
                        bad = True
                if nd is None and any(u < min(defs[r]) for u in uses[r]):
                    bad = True
            # the destination: neither read nor written strictly between X and the copy
            for r in dst:
                if any(x < k < m for k in uses[r]) or any(x < k < m for k in defs[r]):
                    bad = True
            tok = vtok(dst[0], len(dst))
            if bad or tok is None or dst != list(range(dst[0], dst[0] + len(dst))):
                continue
            code = isa.strip_comment(X.text)
            if code.rstrip() != X.text.rstrip() and "/*" not in X.text:
                continue
            ops = list(X.x.ops)
            first = isa.first_tok(ops[0])
            ops[0] = tok + ops[0][len(first):]
            new_text = "\t" + X.x.mn + " " + ", ".join(ops)
            done = (x, m, remake(X, new_text))
            break
        if done is None:
            break
        x, m, nx = done
        items[x] = nx
        del items[m]
        removed += 1
    for k, it in enumerate(items):
        it.pos = k
    log(f"  coalesce: removed {removed} copies")
    return items


# ------------------------------------------------------------------------------------------------- dependence graph
# result latency (cycles from issue until a dependent instruction can issue without waiting)
LAT_RAW = {"WMMA": 16, "TRANS": 6, "VALU": 3, "DS_LD": 96, "VM_LD": 600, "SALU": 1, "TDM": 1, "OTHER": 1}
FENCE = ("WAIT_CNT", "BAR_S", "BAR_W")


def order_class(n):
    """Memory / ordering class of an item for the ordering edges."""
    if n.kind in FENCE or n.x.mn.startswith(("s_setprio", "s_sleep", "s_sendmsg", "s_trap", "s_icache", "s_wait_kmcnt")):
        return "FENCE"
    if n.kind in ("DS_LD", "DS_ST", "VM_LD", "VM_ST", "TDM"):
        return n.kind
    if n.x.mn.startswith(("s_load", "s_buffer_load")):
        return "SMEM"
    return None


def build_dag(items):
    """succ[i] = {j: latency}: every dependence i -> j (i before j in the original order) that a new order must keep."""
    n = len(items)
    succ = [dict() for _ in range(n)]

    def edge(i, j, lat):
        if i != j and succ[i].get(j, -1) < lat:
            succ[i][j] = lat

    last_def, uses_since = {}, collections.defaultdict(list)  # register -> item, register -> readers since its def
    for j, it in enumerate(items):
        regs_u = [("v", r) for r in it.vuses] + [("s", r) for r in it.suses]
        regs_d = [("v", r) for r in it.vdefs] + [("s", r) for r in it.sdefs]
        for r in regs_u:  # RAW
            i = last_def.get(r)
            if i is not None:
                p = items[i]
                lat = LAT_RAW.get(p.kind, 1)
                if p.kind == "WMMA" and it.kind == "WMMA":
                    _, ab = EM.wmma_regs(it.x)
                    lat = 16 if (r[0] == "v" and r[1] in ab) else 8  # D -> A / B: result + 5 wait states; D -> C: chained
                edge(i, j, lat)
        for r in regs_d:  # WAR, WAW
            for i in uses_since.get(r, []):
                p = items[i]
                if p.kind == "WMMA" and it.kind in ("DS_LD", "VM_LD"):
                    lat = 40  # a load into registers an in-flight WMMA still reads waits ~37-47 cycles
                elif p.kind == "WMMA":
                    lat = 5
                elif p.kind in ("DS_LD", "DS_ST", "VM_LD", "VM_ST", "TDM"):
                    lat = 8  # vm_vsrc
                else:
                    lat = 1
                edge(i, j, lat)
            i = last_def.get(r)
            if i is not None:
                p = items[i]
                edge(i, j, LAT_RAW.get(p.kind, 1) if p.kind in ("DS_LD", "VM_LD") else 1)
        for r in regs_d:
            last_def[r] = j
            uses_since[r] = []
        for r in regs_u:
            uses_since[r].append(j)
    # ordering: fences against every memory op and each other; LDS stores against LDS ops; VMEM / TDM in order
    oc = [order_class(it) for it in items]
    for j in range(n):
        if oc[j] is None:
            continue
        for i in range(j):
            if oc[i] is None:
                continue
            a, b = oc[i], oc[j]
            keep = (
                a == "FENCE" or b == "FENCE"
                or ("DS_ST" in (a, b) and a.startswith("DS") and b.startswith("DS"))
                or (a in ("VM_LD", "VM_ST", "TDM") and b in ("VM_LD", "VM_ST", "TDM"))
                or (a == "SMEM" and b == "SMEM")
            )
            if keep:
                edge(i, j, 1)
    # fences also keep their place against every instruction (the barrier region of the loop top)
    for j in range(n):
        if items[j].kind in ("BAR_S", "BAR_W"):
            for i in range(j):
                edge(i, j, 1)
            for k in range(j + 1, n):
                edge(j, k, 1)
    for i in range(n - 1):  # the branch is last
        edge(i, n - 1, 0)
    return succ


def carried_edges(items):
    """Loop-carried dependences (distance 1): [(i, j, latency, kind)] from item i of iteration k to item j of k + 1."""
    n = len(items)
    first_use, first_def, last_use, last_def = {}, {}, {}, {}
    for j, it in enumerate(items):
        regs_u = [("v", r) for r in it.vuses] + [("s", r) for r in it.suses]
        regs_d = [("v", r) for r in it.vdefs] + [("s", r) for r in it.sdefs]
        for r in regs_u:
            if r not in first_def and r not in first_use:
                first_use[r] = j  # read before any def in the body: reads the previous iteration's value
            last_use[r] = j
        for r in regs_d:
            first_def.setdefault(r, j)
            last_def[r] = j
    out = []
    for r, j in first_use.items():  # RAW: last def (k) -> first use (k + 1)
        i = last_def.get(r)
        if i is not None:
            p = items[i]
            lat = LAT_RAW.get(p.kind, 1)
            if p.kind == "WMMA" and items[j].kind == "WMMA":
                lat = 8
            out.append((i, j, lat, "RAW"))
    for r, j in first_def.items():  # WAR: last use (k) -> first def (k + 1); WAW: last def -> first def
        i = last_use.get(r)
        if i is not None and i >= j:  # a use after the first def in the body is of this iteration's value
            p = items[i]
            lat = 40 if (p.kind == "WMMA" and items[j].kind in ("DS_LD", "VM_LD")) else (5 if p.kind == "WMMA" else 1)
            out.append((i, j, lat, "WAR"))
        i = last_def.get(r)
        if i is not None:
            out.append((i, j, 1, "WAW"))
    return out


def rec_mii(items, succ, carried):
    """The recurrence bound: max over carried edges u -> v (distance 1) of latency(u -> v) + longest path v ->* u."""
    import heapq  # noqa: F401
    n = len(items)
    best = (0, None)
    by_v = collections.defaultdict(list)
    for u, v, lat, k in carried:
        by_v[v].append((u, lat, k))
    for v, lst in by_v.items():
        dist = [-1] * n
        dist[v] = 0
        for i in range(v, n):
            if dist[i] < 0:
                continue
            for j, l in succ[i].items():
                if dist[i] + l > dist[j]:
                    dist[j] = dist[i] + l
        for u, lat, k in lst:
            if dist[u] >= 0 and dist[u] + lat > best[0]:
                best = (dist[u] + lat, (u, v, k))
    return best


# ------------------------------------------------------------------------------------------------- list scheduler
WIN_OK = (2, 3, 6, 7)  # issue offsets of non-WMMA instructions inside a WMMA's 8-cycle window
PRIO = "height"
DS_GAP = 2  # DS instructions back to back: ~2.1-2.4 cycles apart (ATT)


class Machine:
    """Per-wave issue model (calibrated on ATT of the b5 dK / dV loop): one instruction per cycle; a WMMA keeps the
    XDL 8 cycles and lets the others issue at offsets 2, 3, 6, 7 of its window; TRANS one per 2 cycles; LDS
    instructions at least DS_GAP cycles apart and at most one per LDS_GAP cycles on average (a bucket of LDS_BURST);
    an instruction that needs another s_set_vgpr_msb mode issues >= 4 cycles after a DS instruction, >= 3 after a
    WMMA (the mode switch waits until they have read their operands)."""

    def __init__(self, lds_gap, lds_burst, msb_cost=True):
        self.t = 0
        self.xdl_start = -100
        self.trans_free = 0
        self.lds_gap, self.lds_burst = lds_gap, lds_burst
        self.lds_tokens, self.lds_t = float(lds_burst), -100
        self.mode = {f: 0 for f in EM.FIELDS}
        self.last_kind, self.last_t = None, -100
        self.msb_cost = msb_cost
        self.nmsb = 0

    def tokens(self, t):
        return min(self.lds_burst, self.lds_tokens + (t - self.lds_t) / self.lds_gap)

    def switch(self, n):
        return any(self.mode[f] != v for f, v in n.req.items())

    def earliest(self, n, t):
        """The first cycle >= t at which n can issue."""
        while True:
            t0 = t
            k = n.kind
            dt = t - self.xdl_start
            if k == "WMMA":
                if dt < 8:
                    t = self.xdl_start + 8
            elif 0 <= dt < 8 and dt not in WIN_OK:
                t += 1
                continue
            if k == "TRANS" and t < self.trans_free:
                t = self.trans_free
            if k in ("DS_LD", "DS_ST"):
                if self.last_kind in ("DS_LD", "DS_ST") and t < self.last_t + DS_GAP:
                    t = self.last_t + DS_GAP
                if self.tokens(t) < 1:
                    t = max(t, int(self.lds_t + (1 - self.lds_tokens) * self.lds_gap + 0.999))
            if self.msb_cost and n.req and self.switch(n):
                if self.last_kind in ("DS_LD", "DS_ST") and t < self.last_t + 4:
                    t = self.last_t + 4
                elif self.last_kind == "WMMA" and t < self.last_t + 3:
                    t = self.last_t + 3
            if t == t0:
                return t

    def issue(self, n, t):
        k = n.kind
        if k == "WMMA":
            self.xdl_start = t
        if k == "TRANS":
            self.trans_free = t + 2
        if k in ("DS_LD", "DS_ST"):
            self.lds_tokens, self.lds_t = self.tokens(t) - 1, t
        if n.req and self.switch(n):
            self.mode.update(n.req)
            self.nmsb += 1
        if n.kind == "BR":
            self.mode = {f: 0 for f in EM.FIELDS}
        self.last_kind, self.last_t = k, t
        self.t = t + 1


def heights(items, succ):
    h = [0] * len(items)
    for i in range(len(items) - 1, -1, -1):
        h[i] = max([lat + h[j] for j, lat in succ[i].items()], default=0)
    return h


def list_schedule(items, succ, lds_gap=6.0, lds_burst=4.0, window=None, msb_pref=1.0):
    """Top-down list scheduling on the Machine model: at each step the ready item that can issue first, ties broken
    by the longest latency path to the end (height), a mode-compatible item preferred (msb_pref cycles of height)."""
    n = len(items)
    pred_n = [0] * n
    for i in range(n):
        for j in succ[i]:
            pred_n[j] += 1
    h = heights(items, succ)
    ready_t = [0] * n
    avail = [i for i in range(n) if pred_n[i] == 0]
    m = Machine(lds_gap, lds_burst)
    order, t = [], 0
    while avail:
        best = None
        for i in avail:
            if window is not None and i > len(order) + window:
                continue
            e = m.earliest(items[i], max(t, ready_t[i]))
            sw = 1 if (items[i].req and m.switch(items[i])) else 0
            if PRIO == "height":
                key = (-e, h[i] - msb_pref * sw, -i)
            elif PRIO == "orig":  # LLVM's order, stalls filled with the earliest ready later item
                key = (-e, -i - 64 * msb_pref * sw)
            else:  # "ready": first ready first
                key = (-e, -ready_t[i], -i)
            if best is None or key > best[0]:
                best = (key, i, e)
        _, i, e = best
        avail.remove(i)
        m.issue(items[i], e)
        order.append(i)
        for j, lat in succ[i].items():
            ready_t[j] = max(ready_t[j], e + lat)
            pred_n[j] -= 1
            if pred_n[j] == 0:
                avail.append(j)
        t = e + 1
    if len(order) != n:
        raise RuntimeError("cycle in the dependence graph")
    return order, t


def model_times(items, succ, order, lds_gap=6.0, lds_burst=4.0, msb_cost=True):
    """Issue cycle of each item (by position in order) under the Machine model."""
    ready_t = [0] * len(items)
    m = Machine(lds_gap, lds_burst, msb_cost)
    t, out = 0, []
    for i in order:
        e = m.earliest(items[i], max(t, ready_t[i]))
        m.issue(items[i], e)
        out.append(e)
        for j, lat in succ[i].items():
            ready_t[j] = max(ready_t[j], e + lat)
        t = e + 1
    return out


def calibrate(items, succ, order, ui, log, lds_gap, lds_burst):
    """Compare the model's issue times of the items (in order) with the median trip of an ATT trace of the same code."""
    import statistics
    d = Path(ui)
    code = json.load(open(d / "code.json"))["code"]
    text = [c[0] or "" for c in code]
    fn = json.load(open(d / "filenames.json"))["wave_filenames"]
    files = sorted(ff for se, sms in fn.items() for smk, sl in sms.items() for slot, wvs in sl.items() for wv, (ff, b, e) in wvs.items())
    w = json.load(open(d / files[0]))["wave"]["instructions"]
    hist = collections.Counter(x[4] for x in w)
    top = max(hist.values())
    lo = min(i for i, c in hist.items() if c >= top * 0.5)
    starts = [k for k, x in enumerate(w) if x[4] == lo]
    trips = [(starts[k], starts[k + 1]) for k in range(len(starts) - 1)]
    lens = [w[b][0] - w[a][0] for a, b in trips]
    a, b = trips[sorted(range(len(trips)), key=lambda k: lens[k])[len(trips) // 2]]
    t0 = w[a][0]
    meas = [(w[k][0] - t0, text[w[k][4]]) for k in range(a, b)]
    # match: the items in order against the issued instructions that are not derived (v_nop, waits)
    pred = model_times(items, succ, order, lds_gap, lds_burst)
    seq = [(items[i], pred[k]) for k, i in enumerate(order)]
    mi = [(t, tx) for t, tx in meas if not tx.startswith(("v_nop", "s_wait_dscnt", "s_wait_loadcnt", "s_wait_xcnt", "s_wait_storecnt"))]
    if len(mi) != len(seq):
        log(f"  calib: {len(mi)} issued vs {len(seq)} items")
    nmis = 0
    rows = []
    for k, ((it, tp), (tm, tx)) in enumerate(zip(seq, mi)):
        if it.x.mn.split("_e32")[0].split("_e64")[0] != tx.split()[0].split("_e32")[0].split("_e64")[0]:
            nmis += 1
            if nmis < 4:
                log(f"  mismatch at {k}: item {it.text.strip()[:50]} vs trace {tx[:50]}")
        rows.append((k, it, tp, tm))
    log(f"  calib: trip {lens[sorted(range(len(trips)), key=lambda k: lens[k])[len(trips) // 2]]} cycles measured, "
        f"model {pred[-1] + 1}; mismatches {nmis}")
    # per segment of 50 items: measured vs model span; the worst extra gaps (measured gap - model gap)
    for q in range(0, len(rows), 60):
        seg = rows[q : q + 60]
        dm, dp = seg[-1][3] - seg[0][3], seg[-1][2] - seg[0][2]
        kinds = collections.Counter(r[1].kind for r in seg)
        log(f"   items {q:4d}..{q + len(seg) - 1:4d}: measured {dm:5d} model {dp:5d}  {dict(kinds)}")
    ex = []
    for k in range(1, len(rows)):
        gm = rows[k][3] - rows[k - 1][3]
        gp = rows[k][2] - rows[k - 1][2]
        ex.append((gm - gp, k, gm, gp))
    ex.sort(reverse=True)
    by = collections.Counter()
    for e, k, gm, gp in ex:
        if e > 0:
            by[(rows[k - 1][1].kind, rows[k][1].kind)] += e
    log("   extra cycles (measured - model gap) by (prev, next) kind: " + str(by.most_common(14)))
    for e, k, gm, gp in ex[:25]:
        log(f"   +{e:4d} at {k}: gap measured {gm} model {gp}: {rows[k - 1][1].text.strip()[:45]} -> {rows[k][1].text.strip()[:60]}")


def model_cycles(items, succ, order, lds_gap=6.0, lds_burst=4.0, msb_cost=True):
    """(cycles, msb switches) of the Machine model for a fixed order (in-order issue)."""
    ready_t = [0] * len(items)
    m = Machine(lds_gap, lds_burst, msb_cost)
    t = 0
    for i in order:
        e = m.earliest(items[i], max(t, ready_t[i]))
        m.issue(items[i], e)
        for j, lat in succ[i].items():
            ready_t[j] = max(ready_t[j], e + lat)
        t = e + 1
    return t, m.nmsb


# ------------------------------------------------------------------------------------------------- emission
def emit_loop(items, fmap, rule, log):
    entry0 = EM.Flow([E.State(unknown=False)], [EM.XState(False)], [[EM._NOP] * 5])
    fl_in = entry0
    for it in range(30):
        lines, fl_out = EM.emit_block(items, fl_in, fmap, rule)
        nxt = EM.Flow.join([entry0, fl_out])
        if repr(nxt.sig()) == repr(fl_in.sig()):
            log(f"  fixed point after {it + 1} passes")
            return lines
        fl_in = nxt
    raise RuntimeError("no fixed point")


def msb_required(x, fmap):
    """asmkit's msb_required, plus operands mapped to {"-"}: not remapped by s_set_vgpr_msb (they must be in v0..v255)."""
    req = {}
    for p, b in EM.op_blocks(x):
        fs = fmap.get((x.mn, p))
        if fs == {"-"}:
            if b != 0:
                raise RuntimeError(f"operand {p} must be in v0..v255: {x.text.strip()}")
            continue
        if not fs:
            raise RuntimeError(f"no MSB field for {x.mn} operand {p}: {x.text.strip()}")
        if len(fs) > 1 and x.cls in ("XDL", "CSMACC", "DPMACC", "TRANS"):
            dflt = ("dst", "src0", "src1", "src2")[p] if p < 4 else None
            if dflt in fs:
                fs = {dflt}
        if len(fs) > 1:
            raise RuntimeError(f"ambiguous MSB field {sorted(fs)} for {x.mn} operand {p}: {x.text.strip()}")
        f = next(iter(fs))
        if f in req and req[f] != b:
            raise RuntimeError(f"conflicting MSB needs in {x.text.strip()}")
        req[f] = b
    return req


EM.msb_required = msb_required


def ds_fields(fmap, ins):
    for x in ins:
        if x.mn.startswith("v_wmma_scale"):
            for p in (4, 5):
                if not fmap.get((x.mn, p)):
                    fmap[(x.mn, p)] = {"-"}
        if x.cls == "DS":
            if x.mn.startswith(("ds_load", "ds_read")):
                fmap[(x.mn, 0)], fmap[(x.mn, 1)] = {"dst"}, {"src0"}
            elif x.mn.startswith(("ds_store", "ds_write")):
                fmap[(x.mn, 0)] = {"src0"}
                for p in range(1, len(x.ops)):
                    if isa.regs_of(isa.first_tok(x.ops[p])):
                        fmap[(x.mn, p)] = {"src1"} if p == 1 else {"src2"}
    return fmap


def verify_all(new, objdump, out, log):
    """verify_obj.compare per kernel function (it compares the first function of a file)."""
    od = objdump.read_text().split("\n")
    heads = [i for i, l in enumerate(od) if re.match(r"^[0-9a-f]{16} <[\w.$]+>:", l)]
    bad = []
    for k, h in enumerate(heads):
        name = re.match(r"^[0-9a-f]{16} <([\w.$]+)>:", od[h]).group(1)
        end = heads[k + 1] if k + 1 < len(heads) else len(od)
        while end > h and (not od[end - 1].strip() or od[end - 1].strip().startswith("s_nop 0 ")):
            end -= 1  # alignment padding after s_endpgm
        (out / f"od_{name}.s").write_text("\n".join(od[:6] + od[h:end]) + "\n")
        i0 = next(i for i, l in enumerate(new) if l.startswith(name + ":"))
        i1 = next(i for i in range(i0, len(new)) if new[i].strip().startswith(".Lfunc_end"))
        (out / f"cc_{name}.s").write_text("\n".join(new[i0 : i1 + 1]) + "\n")
        log(f"  {name}:")
        bad += VO.compare(str(out / f"cc_{name}.s"), str(out / f"od_{name}.s"), log=log)
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--kernels", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mode", default="identity", choices=("identity", "list", "llvm"))
    ap.add_argument("-D", dest="defines", action="append", default=[])
    ap.add_argument("--flag", action="append", default=[], help="extra compiler flag")
    ap.add_argument("--rule", default="llvm", choices=E.VA_RULES)
    ap.add_argument("--lds-gap", type=float, default=6.0)
    ap.add_argument("--lds-burst", type=float, default=4.0)
    ap.add_argument("--window", type=int, default=None, help="list mode: look ahead at most this many items")
    ap.add_argument("--msb-pref", type=float, default=1.0)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--coalesce", action="store_true", help="remove copies (the producer writes the destination)")
    ap.add_argument("--prio", default="height", choices=("height", "orig", "ready"))
    ap.add_argument("--calib", default="", help="ATT ui dir of the identity build: compare the model with it")
    ap.add_argument("--drop", default="", help="timing probe (wrong results): MN:K drops every K-th item of mnemonic MN")
    ap.add_argument("--lat-ds", type=int, default=96)
    a = ap.parse_args()
    LAT_RAW["DS_LD"] = a.lat_ds
    global PRIO
    PRIO = a.prio
    out = BUILD / "rs" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    rep = {"args": vars(a), "kernels": {}}
    logf = open(out / "log.txt", "w")

    def log(*s):
        print(*s, flush=True)
        print(*s, file=logf, flush=True)

    cmd, s_path = compile_s(a.src, a.defines, a.flag, out / "tmp")
    lines = s_path.read_text().rstrip("\n").split("\n")
    (out / "llvm.s").write_text("\n".join(lines) + "\n")
    ins = isa.parse(lines)
    fmap = ds_fields(EM.learn_msb_map(ins), ins)
    edits = []  # (a, b, new lines)
    for kn in a.kernels.split(","):
        la, lb = main_loop(ins, kn)
        items = body_items(ins, la, lb)
        if a.coalesce:
            items = coalesce(items, log)
        for spec in filter(None, a.drop.split(",")):
            mn, k = spec.split(":")
            k, c, keep = int(k), 0, []
            for it in items:
                if it.x.mn == mn:
                    c += 1
                    if c % k == 0:
                        continue
                keep.append(it)
            log(f"  probe: dropped {len(items) - len(keep)} {mn}")
            items = keep
            for q, it in enumerate(items):
                it.pos = q
        for it in items:
            it.req = {} if it.kind == "BR" else msb_required(it.x, fmap)
        succ = build_dag(items)
        ident = list(range(len(items)))
        c0, m0 = model_cycles(items, succ, ident, a.lds_gap, a.lds_burst)
        if a.calib:
            calibrate(items, succ, ident, a.calib, log, a.lds_gap, a.lds_burst)
            return 0
        if a.analyze:
            car = carried_edges(items)
            rm, arg = rec_mii(items, succ, car)
            log(f"  carried edges {len(car)} ({collections.Counter(k for *_, k in car)}); RecMII {rm} via {arg}")
            if arg:
                u, v, k = arg
                log(f"    u {u}: {items[u].text.strip()[:90]}")
                log(f"    v {v}: {items[v].text.strip()[:90]}")
            # the critical recurrences: per carried edge kind, the top 8 cycle lengths
            n = len(items)
            res = []
            byv = collections.defaultdict(list)
            for u, v, lat, k in car:
                byv[v].append((u, lat, k))
            for v, lst in byv.items():
                dist = [-1] * n
                dist[v] = 0
                for i in range(v, n):
                    if dist[i] < 0:
                        continue
                    for j, l in succ[i].items():
                        if dist[i] + l > dist[j]:
                            dist[j] = dist[i] + l
                for u, lat, k in lst:
                    if dist[u] >= 0:
                        res.append((dist[u] + lat, k, u, v))
            res.sort(reverse=True)
            for r in res[:12]:
                log(f"    cycle {r[0]:5d} {r[1]} {r[2]} -> {r[3]}: {items[r[2]].x.mn} -> {items[r[3]].x.mn}")
            return 0
        if a.mode == "list":
            order, _ = list_schedule(items, succ, a.lds_gap, a.lds_burst, a.window, a.msb_pref)
        else:
            order = ident
        c1, m1 = model_cycles(items, succ, order, a.lds_gap, a.lds_burst)
        kinds = collections.Counter(it.kind for it in items)
        log(f"{kn}: loop lines {la}..{lb}, {len(items)} items {dict(kinds)}; model cycles LLVM order {c0} (msb {m0}), new {c1} (msb {m1})")
        label = ins[la].label
        new_items = [items[i] for i in order]
        br = new_items[-1]
        br.text = br.text.replace(label, label + "_rs")
        body = emit_loop(new_items, fmap, a.rule, log)
        nmsb = sum(1 for l in body if "s_set_vgpr_msb" in l)
        nwait = sum(1 for l in body if l.strip().startswith("s_wait_alu") or l.strip().startswith(("s_wait_dscnt", "s_wait_loadcnt", "s_wait_xcnt")))
        nnop = sum(1 for l in body if l.strip() == "v_nop")
        llvm_body = [ins[j] for j in range(la + 1, lb + 1) if ins[j].mn]
        lm = collections.Counter(x.cls for x in llvm_body)
        log(f"  emitted {len(body)} lines: msb {nmsb}, waits {nwait}, v_nop {nnop} (LLVM: msb {lm['MSB']}, "
            f"wait_alu {lm['WAIT_ALU']}, v_nop {lm['VNOP']}, lines {len(llvm_body)})")
        rep["kernels"][kn] = {"loop": [la, lb], "items": len(items), "model_llvm": c0, "model_new": c1,
                              "emitted": len(body), "msb": nmsb, "waits": nwait, "nops": nnop}
        edits.append((la, lb, [lines[la]] + DRAIN + [label + "_rs:"] + body + DRAIN))
    new = list(lines)
    for la, lb, nl in sorted(edits, reverse=True):
        new[la : lb + 1] = nl
    spl = out / "new.s"
    spl.write_text("\n".join(new) + "\n")
    cc1as, lld = B.driver_steps(cmd, str(KERN))
    stem = Path(a.src).stem
    rung = re.match(r"k_mla_\w+_mx(\d+)$", stem)
    base = stem[: -len(rung.group(1))] if rung else stem
    hs = BUILD / f"{base}{a.tag}.hsaco"
    meta = B.assemble(spl, hs, cc1as, lld, cwd=str(KERN))
    bad = verify_all(new, Path(str(hs.with_suffix("")) + ".objdump.s"), out, log)
    rep["hsaco"], rep["verify"] = meta, ("ok" if not bad else [str(x) for x in bad])
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
    log(f"hsaco {hs} {meta}")
    log("verify_obj:", "ok" if not bad else bad[:4])
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
