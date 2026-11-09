###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Post-RA re-scheduling of a kernel's main loop: the forward kernel's tile loop (one straight-line block of ~790
instructions) in a list-scheduled order instead of LLVM's (forward -7%, bit-exact results).

The kernel source is compiled with the package flags plus ``-amdgpu-enable-vopd=0`` (asmkit has no VOPD) and
``-save-temps=obj``; the loop body (a label, its back branch, no label or branch in between) is taken apart into its
real instructions (LLVM's ``s_set_vgpr_msb``, ``s_wait_alu``, ``v_nop`` and register-driven ``s_wait_*cnt`` dropped),
put in a new order that keeps every register / scalar / memory dependence of LLVM's register allocation (barriers and
TDM / tensorcnt fences in place), and emitted through asmkit (``gfx1250_mla/tools/asmkit``), which derives
``s_set_vgpr_msb``, the expert-mode waits and the hazard ``v_nop``s of the new order for every flow reaching the block
(a fixed point over the back edge). The loop entry and exit drain every counter (the code around the loop keeps
LLVM's waits). The spliced ``.s`` is assembled with the driver's own commands and checked instruction by instruction
against the code object (``verify_obj``). The order: a top-down list schedule on a per-wave issue model (one
instruction per cycle; a WMMA keeps the matrix unit 8 cycles and lets others issue at offsets 2, 3, 6, 7 of its
window; TRANS one per 2 cycles; LDS instructions >= 2 cycles apart and at most one per LDS_GAP cycles on average),
first-ready-first. Standard library + asmkit only; the lab version with the model tools is
``benchmark/hk_mla_mx/py/resched.py``."""

import collections
import importlib
import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

_ASMKIT = Path(__file__).resolve().parent.parent / "gfx1250_mla" / "tools" / "asmkit"
_PKG = "_primus_turbo_gfx1250_asmkit"


def _asmkit():
    """asmkit's modules (build, emit, expert, isa, verify_obj), loaded from the gfx1250_mla tools directory."""
    if _PKG not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            _PKG, _ASMKIT / "__init__.py", submodule_search_locations=[str(_ASMKIT)]
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[_PKG] = mod
        spec.loader.exec_module(mod)
    return tuple(importlib.import_module(f"{_PKG}.{m}") for m in ("build", "emit", "expert", "isa", "verify_obj"))


def available():
    return (_ASMKIT / "emit.py").is_file()


SDST_RE = re.compile(r"v_(add|sub|subrev)_co(_ci)?_u32_e64$|v_div_scale_f(32|64)$|v_mad_(u|i)64_(u|i)32")
DERIVED_CNT = {"dscnt", "loadcnt", "storecnt", "xcnt"}
DRAIN = ["\ts_wait_loadcnt 0x0", "\ts_wait_storecnt 0x0", "\ts_wait_dscnt 0x0", "\ts_wait_xcnt 0x0",
         "\ts_wait_alu depctr_va_vdst(0) depctr_vm_vsrc(0)"] + ["\tv_nop"] * 5  # fmt: skip
LAT_RAW = {"WMMA": 16, "TRANS": 6, "VALU": 3, "DS_LD": 32, "VM_LD": 600, "SALU": 1, "TDM": 1, "OTHER": 1}
WIN_OK = (2, 3, 6, 7)
# asmkit's va_vdst rule: LLVM's, except that a write-after-read on an in-flight WMMA keeps the per-class bound (the bf16
# HK backward generator's rule; forward 0.606 vs 0.610 ms with LLVM's rule)
VA_RULE = "llvm_war_xdl"
DS_GAP = 2


class _Tools:
    def __init__(self):
        self.B, self.EM, self.E, self.isa, self.VO = _asmkit()


def _main_loop(T, ins, kernel):
    k0 = next(i for i, x in enumerate(ins) if x.label == kernel)
    k1 = next(i for i in range(k0, len(ins)) if ins[i].text.strip().startswith(".Lfunc_end"))
    labels, best = {}, None
    for i in range(k0, k1):
        x = ins[i]
        if x.cls == "LABEL":
            labels[x.label] = i
        elif x.mn.startswith(("s_cbranch", "s_branch")):
            a = labels.get(T.isa.strip_comment(x.ops[0]).strip())
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


def _largest_block(T, ins, kernel, skip):
    """(first line, last line) of the kernel's largest straight-line block with WMMAs outside the lines skip: a label,
    then no label or branch but its last instruction (the peeled tiles + epilogue of the forward)."""
    k0 = next(i for i, x in enumerate(ins) if x.label == kernel)
    k1 = next(i for i in range(k0, len(ins)) if ins[i].text.strip().startswith(".Lfunc_end"))
    best = None
    i = k0
    while i < k1:
        if ins[i].cls != "LABEL":
            i += 1
            continue
        a, b = i, i + 1
        while b < k1 and ins[b].cls != "LABEL" and not ins[b].mn.startswith(("s_cbranch", "s_branch", "s_endpgm")):
            b += 1
        if b < k1 and ins[b].cls == "LABEL":
            b -= 1
        n = sum(1 for j in range(a, b + 1) if ins[j].mn)
        nw = sum(1 for j in range(a, b + 1) if ins[j].cls == "XDL")
        if nw and not (skip[0] <= a <= skip[1]) and (best is None or n > best[2]):
            best = (a, b, n)
        i = b + 1
    return best


def _derived(T, x):
    if x.cls in ("MSB", "VNOP", "CLAUSE", "DELAY"):
        return True
    if x.cls == "WAIT_ALU":
        if set(T.isa.wait_fields(x)) - {"va_vdst", "vm_vsrc"}:
            raise RuntimeError(f"s_wait_alu field not re-derived by asmkit: {x.text.strip()}")
        return True
    if x.cls == "WAIT_CNT":
        f = T.isa.cnt_fields(x)
        return bool(f) and set(f) <= DERIVED_CNT
    return False


class _RItem:
    """asmkit's Item with a __dict__ (req: the s_set_vgpr_msb fields it needs)."""

    __slots__ = ("text", "x", "kind", "vdefs", "vuses", "sdefs", "suses", "pos", "req")

    def __repr__(self):
        return f"I{self.pos}:{self.x.mn}"


def _items(T, ins, a, b):
    items = []
    for j in range(a + 1, b + 1):
        x = ins[j]
        if not x.mn or x.cls in ("DIR", "LABEL") or _derived(T, x):
            continue
        if "::" in x.mn or x.mn == "s_nop":
            raise RuntimeError(f"unsupported in the loop: {x.text.strip()}")
        n0 = T.EM.make_item(x.text)
        n = _RItem()
        for k in T.EM.Item.__slots__:
            setattr(n, k, getattr(n0, k))
        err = T.isa.coverage_errors(n.x)
        if err:
            raise RuntimeError(f"coverage {err}: {x.text.strip()}")
        n.pos = len(items)
        items.append(n)
    return items


def _order_class(n):
    if n.kind in ("WAIT_CNT", "BAR_S", "BAR_W") or n.x.mn.startswith(
        ("s_setprio", "s_sleep", "s_sendmsg", "s_trap", "s_icache", "s_wait_kmcnt")
    ):
        return "FENCE"
    if n.kind in ("DS_LD", "DS_ST", "VM_LD", "VM_ST", "TDM"):
        return n.kind
    if n.x.mn.startswith(("s_load", "s_buffer_load")):
        return "SMEM"
    return None


def _dag(T, items, lat_ds=None):
    """succ[i] = {j: latency} for every dependence i -> j a new order must keep (lat_ds: the LDS load latency the
    schedule assumes, default LAT_RAW's)."""
    LAT = dict(LAT_RAW, **({"DS_LD": lat_ds} if lat_ds is not None else {}))
    n = len(items)
    succ = [dict() for _ in range(n)]

    def edge(i, j, lat):
        if i != j and succ[i].get(j, -1) < lat:
            succ[i][j] = lat

    last_def, readers = {}, collections.defaultdict(list)
    for j, it in enumerate(items):
        ru = [("v", r) for r in it.vuses] + [("s", r) for r in it.suses]
        rd = [("v", r) for r in it.vdefs] + [("s", r) for r in it.sdefs]
        for r in ru:
            i = last_def.get(r)
            if i is not None:
                p = items[i]
                lat = LAT.get(p.kind, 1)
                if p.kind == "WMMA" and it.kind == "WMMA":
                    _, ab = T.EM.wmma_regs(it.x)
                    lat = 16 if (r[0] == "v" and r[1] in ab) else 8
                edge(i, j, lat)
        for r in rd:
            for i in readers.get(r, []):
                p = items[i]
                if p.kind == "WMMA":
                    lat = 40 if it.kind in ("DS_LD", "VM_LD") else 5
                elif p.kind in ("DS_LD", "DS_ST", "VM_LD", "VM_ST", "TDM"):
                    lat = 8
                else:
                    lat = 1
                edge(i, j, lat)
            i = last_def.get(r)
            if i is not None:
                edge(i, j, LAT.get(items[i].kind, 1) if items[i].kind in ("DS_LD", "VM_LD") else 1)
        for r in rd:
            last_def[r] = j
            readers[r] = []
        for r in ru:
            readers[r].append(j)
    oc = [_order_class(it) for it in items]
    for j in range(n):
        if oc[j] is None:
            continue
        for i in range(j):
            a, b = oc[i], oc[j]
            if a is None:
                continue
            if (
                a == "FENCE"
                or b == "FENCE"
                or ("DS_ST" in (a, b) and a.startswith("DS") and b.startswith("DS"))
                or (a in ("VM_LD", "VM_ST", "TDM") and b in ("VM_LD", "VM_ST", "TDM"))
                or (a == "SMEM" and b == "SMEM")
            ):
                edge(i, j, 1)
    for j in range(n):
        if items[j].kind in ("BAR_S", "BAR_W"):
            for i in range(j):
                edge(i, j, 1)
            for k in range(j + 1, n):
                edge(j, k, 1)
    for i in range(n - 1):
        edge(i, n - 1, 0)
    return succ


FIELDS = ("src0", "src1", "src2", "dst")


class _Machine:
    """Per-wave issue model (see the module doc); an instruction that needs another s_set_vgpr_msb mode issues >= 4
    cycles after an LDS instruction and >= 3 after a WMMA (the switch waits until they read their operands)."""

    def __init__(self, lds_gap, lds_burst):
        self.xdl = -100
        self.trans_free = 0
        self.lds_gap, self.lds_burst = lds_gap, lds_burst
        self.tok, self.tok_t = float(lds_burst), -100
        self.last_kind, self.last_t = None, -100
        self.mode = {f: 0 for f in FIELDS}

    def switch(self, n):
        return any(self.mode[f] != v for f, v in n.req.items())

    def tokens(self, t):
        return min(self.lds_burst, self.tok + (t - self.tok_t) / self.lds_gap)

    def earliest(self, n, t):
        while True:
            t0, k, dt = t, n.kind, t - self.xdl
            if k == "WMMA":
                if dt < 8:
                    t = self.xdl + 8
            elif 0 <= dt < 8 and dt not in WIN_OK:
                t += 1
                continue
            if k == "TRANS" and t < self.trans_free:
                t = self.trans_free
            if k in ("DS_LD", "DS_ST"):
                if self.last_kind in ("DS_LD", "DS_ST") and t < self.last_t + DS_GAP:
                    t = self.last_t + DS_GAP
                if self.tokens(t) < 1:
                    t = max(t, int(self.tok_t + (1 - self.tok) * self.lds_gap + 0.999))
            if n.req and self.switch(n):
                if self.last_kind in ("DS_LD", "DS_ST") and t < self.last_t + 4:
                    t = self.last_t + 4
                elif self.last_kind == "WMMA" and t < self.last_t + 3:
                    t = self.last_t + 3
            if t == t0:
                return t

    def issue(self, n, t):
        if n.kind == "WMMA":
            self.xdl = t
        if n.kind == "TRANS":
            self.trans_free = t + 2
        if n.kind in ("DS_LD", "DS_ST"):
            self.tok, self.tok_t = self.tokens(t) - 1, t
        if n.req and self.switch(n):
            self.mode.update(n.req)
        if n.kind == "BR":
            self.mode = {f: 0 for f in FIELDS}
        self.last_kind, self.last_t = n.kind, t


def _heights(succ):
    """The longest latency path from each item to the end."""
    h = [0] * len(succ)
    for i in range(len(succ) - 1, -1, -1):
        h[i] = max([lat + h[j] for j, lat in succ[i].items()], default=0)
    return h


def _schedule(items, succ, lds_gap, lds_burst, prio="ready"):
    """List schedule on the issue model: the ready item that can issue first; ties by prio: "ready" the earliest ready
    (then the original order), "height" the longest latency path to the end, an item needing no s_set_vgpr_msb switch
    preferred by one cycle of height (the lab's resched.py --prio height)."""
    n = len(items)
    h = _heights(succ) if prio == "height" else None
    pred_n = [0] * n
    for i in range(n):
        for j in succ[i]:
            pred_n[j] += 1
    ready_t = [0] * n
    avail = [i for i in range(n) if pred_n[i] == 0]
    m = _Machine(lds_gap, lds_burst)
    order, t = [], 0
    while avail:
        best = None
        for i in avail:
            e = m.earliest(items[i], max(t, ready_t[i]))
            if h is not None:
                sw = 1 if (items[i].req and m.switch(items[i])) else 0
                key = (-e, h[i] - sw, -i)
            else:
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
    return order


def sgpr_waits(items, carried_valu=()):
    """Prefix s_wait_alu depctr_va_sdst(0) / depctr_va_vcc(0) to every SALU / branch / memory item that reads an SGPR (vcc)
    whose last def in the order is a VALU (asmkit does not derive these waits). carried_valu: SGPRs whose last def is a
    VALU on entry (a loop's previous iteration)."""
    last_valu = {r: True for r in carried_valu}
    for it in items:
        if it.kind in ("SALU", "BR", "DS_LD", "DS_ST", "VM_LD", "VM_ST", "TDM", "OTHER"):
            need = sorted(r for r in it.suses if r != "exec" and last_valu.get(r))
            if need:
                f = "va_vcc" if all(r in ("vcc", "vcc_hi") for r in need) else "va_sdst"
                it.text = f"\ts_wait_alu depctr_{f}(0)\n" + it.text
                for r in list(last_valu):
                    last_valu[r] = False  # the wait completes every VALU
        for r in it.sdefs:
            last_valu[r] = it.x.cls in ("CSMACC", "DPMACC", "TRANS", "XDL")
    return items


def loop_carried_valu_sgprs(items):
    """SGPRs read before their first def in the body whose last def in the body is a VALU."""
    first_def, last_def, early = {}, {}, set()
    for k, it in enumerate(items):
        for r in it.suses:
            if r not in first_def:
                early.add(r)
        for r in it.sdefs:
            first_def.setdefault(r, k)
            last_def[r] = k
    return [r for r in early if r in last_def and items[last_def[r]].x.cls in ("CSMACC", "DPMACC", "TRANS", "XDL")]


def _fmap(T, ins):
    fmap = T.EM.learn_msb_map(ins)
    for x in ins:
        if x.mn.startswith("v_wmma_scale"):
            for p in (4, 5):
                if not fmap.get((x.mn, p)):
                    fmap[(x.mn, p)] = {"-"}  # WMMA scale operands: not remapped (v0..v255)
        if x.cls in ("FLAT", "VMEM"):  # by operand position: loads vdst dst, vaddr src0; stores vaddr src0, vdata src1
            for q in range(min(2, len(x.ops))):
                if T.isa.regs_of(T.isa.first_tok(x.ops[q])):
                    fmap[(x.mn, q)] = ({"dst"} if q == 0 else {"src0"}) if "_load" in x.mn else ({"src0"} if q == 0 else {"src1"})
        if x.cls == "DS":
            if x.mn.startswith(("ds_load", "ds_read")):
                fmap[(x.mn, 0)], fmap[(x.mn, 1)] = {"dst"}, {"src0"}
            elif x.mn.startswith(("ds_store", "ds_write")):
                fmap[(x.mn, 0)] = {"src0"}
                for p in range(1, len(x.ops)):
                    if T.isa.regs_of(T.isa.first_tok(x.ops[p])):
                        fmap[(x.mn, p)] = {"src1"} if p == 1 else {"src2"}
    return fmap


def _msb_required(T):
    EM = T.EM

    def req(x, fmap):
        out = {}
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
                if SDST_RE.match(x.mn):  # VOP3 with an SGPR dst after the VGPR dst: vdst, sdst, src0, src1, src2
                    dflt = ("dst", None, "src0", "src1", "src2")[p] if p < 5 else None
                if dflt in fs:
                    fs = {dflt}
            if len(fs) > 1:
                raise RuntimeError(f"ambiguous MSB field {sorted(fs)} for {x.mn} operand {p}: {x.text.strip()}")
            f = next(iter(fs))
            if f in out and out[f] != b:
                raise RuntimeError(f"conflicting MSB needs in {x.text.strip()}")
            out[f] = b
        return out

    return req


def _emit_loop(T, items, fmap):
    entry0 = T.EM.Flow([T.E.State(unknown=False)], [T.EM.XState(False)], [[T.EM._NOP] * 5])
    fl_in = entry0
    for _ in range(30):
        lines, fl_out = T.EM.emit_block(items, fl_in, fmap, VA_RULE)
        nxt = T.EM.Flow.join([entry0, fl_out])
        if repr(nxt.sig()) == repr(fl_in.sig()):
            return lines
        fl_in = nxt
    raise RuntimeError("no fixed point")


def _verify(T, new, objdump, tmp):
    od = Path(objdump).read_text().split("\n")
    heads = [i for i, l in enumerate(od) if re.match(r"^[0-9a-f]{16} <[\w.$]+>:", l)]
    bad = []
    for k, h in enumerate(heads):
        name = re.match(r"^[0-9a-f]{16} <([\w.$]+)>:", od[h]).group(1)
        end = heads[k + 1] if k + 1 < len(heads) else len(od)
        while end > h and (not od[end - 1].strip() or od[end - 1].strip().startswith("s_nop 0 ")):
            end -= 1
        (tmp / f"od_{name}.s").write_text("\n".join(od[:6] + od[h:end]) + "\n")
        i0 = next(i for i, l in enumerate(new) if l.startswith(name + ":"))
        i1 = next(i for i in range(i0, len(new)) if new[i].strip().startswith(".Lfunc_end"))
        (tmp / f"cc_{name}.s").write_text("\n".join(new[i0 : i1 + 1]) + "\n")
        bad += T.VO.compare(str(tmp / f"cc_{name}.s"), str(tmp / f"od_{name}.s"), log=lambda *a: None)
    return bad


def build(cxx, src, out, flags, kernels, cwd, tmp, lds_gap=6.0, lds_burst=4.0, regions=True, prio="ready", lat_ds=None):
    """Compile src (flags + no VOPD + save-temps) into out with the main loop of each kernel re-scheduled (prio, lat_ds:
    _schedule's tie rule and the assumed LDS load latency); raises on any failure (the caller falls back to the plain
    build). Returns {kernel: (items, LLVM lines, emitted lines)}."""
    T = _Tools()
    T.EM.msb_required = _msb_required(T)  # + operands not remapped by s_set_vgpr_msb (WMMA scale operands)
    tmp = Path(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    cmd = [cxx, *flags, "-mllvm", "-amdgpu-enable-vopd=0", "-save-temps=obj", "-o", str(tmp / "cc.hsaco"), str(src)]
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"compile failed:\n{r.stderr[-4000:]}")
    s_path = next(tmp.glob("*-gfx1250.s"))
    lines = s_path.read_text().rstrip("\n").split("\n")
    ins = T.isa.parse(lines)
    fmap = _fmap(T, ins)
    edits, info = [], {}
    req_of = _msb_required(T)

    def with_req(items):
        for it in items:
            it.req = {} if it.kind == "BR" else req_of(it.x, fmap)
        return items

    for kn in kernels:
        la, lb = _main_loop(T, ins, kn)
        items = with_req(_items(T, ins, la, lb))
        if items[-1].kind != "BR":
            raise RuntimeError("the loop body does not end with its branch")
        order = _schedule(items, _dag(T, items, lat_ds), lds_gap, lds_burst, prio)
        label = ins[la].label
        new_items = [items[i] for i in order]
        sgpr_waits(new_items, loop_carried_valu_sgprs(new_items))
        new_items[-1].text = new_items[-1].text.replace(label, label + "_rs")
        body = _emit_loop(T, new_items, fmap)
        info[kn] = [(len(items), len(body))]
        edits.append((la, lb, [lines[la]] + DRAIN + [label + "_rs:"] + body + DRAIN))
        if regions:  # the largest straight-line block with WMMAs (the forward: peeled diagonal tiles + epilogue)
            rg = _largest_block(T, ins, kn, (la, lb))
            if rg is not None and rg[2] >= 500:
                ra, rb, _ = rg
                items = with_req(_items(T, ins, ra, rb))
                tail = items.pop() if items and items[-1].kind == "BR" else None
                order = _schedule(items, _dag(T, items, lat_ds), lds_gap, lds_burst, prio)
                new_items = sgpr_waits([items[i] for i in order])
                entry = T.EM.Flow([T.E.State(unknown=False)], [T.EM.XState(False)], [[T.EM._NOP] * 5])
                body, _ = T.EM.emit_block(new_items, entry, fmap, VA_RULE)
                info[kn].append((len(items), len(body)))
                edits.append((ra, rb, [lines[ra]] + DRAIN + body + DRAIN + ([tail.text] if tail else [])))
    new = list(lines)
    for la, lb, nl in sorted(edits, reverse=True):
        new[la : lb + 1] = nl
    spl = tmp / "resched.s"
    spl.write_text("\n".join(new) + "\n")
    cc1as, lld = T.B.driver_steps(cmd, str(cwd))
    obj = tmp / "rs.hsaco"
    T.B.assemble(spl, obj, cc1as, lld, cwd=str(cwd))
    bad = _verify(T, new, str(tmp / "rs.objdump.s"), tmp)
    if bad:
        raise RuntimeError(f"verify_obj: {bad[:3]}")
    shutil.copyfile(obj, out)
    return info
