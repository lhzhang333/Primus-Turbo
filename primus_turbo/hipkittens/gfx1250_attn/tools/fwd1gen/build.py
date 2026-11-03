###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""build.py -- build k_attn_fwd_v1 with its item body generated (fwd1gen).

    python3 build.py --out DIR [--cxx CXX] [--va-rule RULE] [model / schedule options]

 1. compile the marker shell (the manifest's command for k_attn_fwd_v1, -DHKGEN=1): DIR/shell.s; and the reference
    (the same command with -DHKGEN=0: the HIP kernel in expert scheduling mode, DIR/ref/k_attn_fwd_v1.hsaco), whose
    operands in high VGPR blocks teach, with the shell's, the s_set_vgpr_msb operand-field map;
 2. read the markers (FRONT, STAGE_LOOP, STAGE_MAIN, STAGE_PEN, STAGE_LAST, EPI) and the registers LLVM gave their
    operands; base = 1 + the highest VGPR the shell references;
 3. generate every region's blocks (fwd1gen, scheduled by bwd1gen's slot scheduler); asmkit.emit derives
    s_set_vgpr_msb, the expert-mode waits and the hazard v_nops over the region graph (shell code between regions:
    unknown VALU / memory state, every region starts with va_vdst 0) to a fixed point of the block entry states; a
    stage region ends with vm_vsrc 0 (EPI also s_wait_xcnt 0: a tensor_load_to_lds reads its descriptor SGPRs at
    issue, as LLVM assumes); a re-walk of every block from its entry state must find no open hazard;
 4. splice each region behind its marker, the out-of-line rescale blocks behind the kernel's s_endpgm, the kernel's
    VGPR allocation 1024: DIR/k_attn_fwd_v1.s;
 5. assemble with the driver's own cc1as / lld (asmkit.build): DIR/k_attn_fwd_v1.hsaco; asmkit.verify_obj;
 6. contract: the generated code writes VGPRs only at or above base and SGPRs only in its marker's clobber list, its
    in / out operands or the SGPRs it saves and restores (STAGE_LOOP); no spills.
Writes DIR/build.json; exit status 1 on any failed check. CPU only."""

import argparse
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent
PKG = TOOLS.parent
HIP = PKG / "hip"
sys.path[:0] = [str(HERE), str(TOOLS / "bwd1gen"), str(TOOLS)]
import bwd1gen as GC  # noqa: E402
import fwd1gen as G  # noqa: E402
from asmkit import build as B  # noqa: E402
from asmkit import emit as M  # noqa: E402
from asmkit import expert as E  # noqa: E402
from asmkit import verify_obj as VO  # noqa: E402
from asmkit.isa import coverage_errors, parse, parse_line, sgpr_defs_uses  # noqa: E402

ENTRY = "k_attn_fwd_v1"
FUNC = "k_attn_fwd_v1"
MARK = re.compile(r";\s*HKGEN\s+(\w+)\s*(.*)$")
POST = ["\ts_wait_xcnt 0x0", "\ts_wait_alu depctr_vm_vsrc(0)"]
# ENTRY_VA: every region starts with s_wait_alu depctr_va_vdst(0) (the shell's VALU writes retire while the WMMA pipe
# is empty anyway); STAGE_XCNT: the stage regions end with s_wait_xcnt 0 too (only EPI has VMEM)
OPT = {"entry_va": True, "stage_xcnt": False}
ENTRY_VA = "\ts_wait_alu depctr_va_vdst(0)"


def post_of(kind):
    return POST if (kind == "EPI" or OPT["stage_xcnt"]) else POST[1:]


VOP3_FIELDS = ("dst", "src0", "src1", "src2")


def log(*a):
    print(*a, flush=True)


def load_hsaco():
    spec = importlib.util.spec_from_file_location("_gfx1250_attn_hsaco", PKG / "hsaco.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compile_variant(hsaco, cxx, out_dir, reference):
    """Compile the shell (the manifest command + -DHKGEN=1) or, reference=True, the manifest command as is, into
    out_dir; returns (argv, the compiler's .s, the code object)."""
    tmp = out_dir / "tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    obj = tmp / f"{ENTRY}.cc.hsaco"
    cmd = hsaco.compile_command(cxx, ENTRY, obj)
    # the manifest builds the shell (-DHKGEN=1); the reference is the same command with HKGEN=0
    head, tail = cmd[: cmd.index("-c")], cmd[cmd.index("-c") :]
    if reference:
        cmd = [f for f in head if not f.startswith("-DHKGEN")] + ["-DHKGEN=0"] + tail
    elif "-DHKGEN=1" not in head:
        cmd = head + ["-DHKGEN=1"] + tail
    r = subprocess.run(cmd, cwd=HIP, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"compile failed:\n{r.stderr[-4000:]}")
    return cmd, next(tmp.glob("*-gfx1250.s")), obj


def func_range(lines, name):
    """(first, last) line index of function `name` in a compiler .s: its label to its .Lfunc_end line."""
    lo = next(i for i, l in enumerate(lines) if l.startswith(f"{name}:"))
    hi = next(i for i in range(lo, len(lines)) if lines[i].startswith(".Lfunc_end"))
    return lo, hi


def items_of(texts, fmap):
    for t in texts:
        x = parse_line(0, "\t" + t)
        if x.cls in ("XDL", "CSMACC", "DPMACC", "TRANS"):
            for p in range(min(len(x.ops), 4)):
                fmap.setdefault((x.mn, p), {VOP3_FIELDS[p]})
    items = [M.make_item("\t" + t) for t in texts]
    for k, it in enumerate(items):
        it.pos = k
    return items


def unknown_flow():
    return M.Flow([E.State(unknown=True)], [M.XState(True)], [[]])


def open_hazards(lines, flow):
    ins = parse(lines)
    found = []

    def note(k, field, _):
        found.append((k, field, ins[k].text.strip()))

    M.walk_block(ins, flow.copy(), note)
    return found, [(k, e) for k, x in enumerate(ins) for e in coverage_errors(x)]


def set_vgpr_count_of(lines, name, n):
    """Raise kernel `name`'s VGPR allocation to n: .amdhsa_next_free_vgpr in its .amdhsa_kernel block, .vgpr_count
    in its metadata entry (the entry's .name precedes its .vgpr_count: keys are sorted)."""
    out, in_kd, cur, hits = [], False, None, 0
    for l in lines:
        s = l.strip()
        if s.startswith(".amdhsa_kernel "):
            in_kd = s.split()[1] == name
        elif s.startswith(".end_amdhsa_kernel"):
            in_kd = False
        m = re.match(r"(\s*\.amdhsa_next_free_vgpr\s+)(\d+)(.*)", l)
        if m and in_kd:
            l, hits = f"{m.group(1)}{n}{m.group(3)}", hits + 1
        m = re.match(r"\s*\.name:\s+(\S+)", l)
        if m:
            cur = m.group(1)
        m = re.match(r"(\s*\.vgpr_count:\s+)(\d+)(.*)", l)
        if m and cur == name:
            l, hits = f"{m.group(1)}{n}{m.group(3)}", hits + 1
        out.append(l)
    if hits != 2:
        raise RuntimeError(f"VGPR count of {name}: {hits} fields changed, want 2")
    return out


def verify_func(s_lines, objdump_path, out_dir):
    """asmkit.verify_obj.compare on function FUNC of the intended .s and of the objdump listing."""
    lo, hi = func_range(s_lines, FUNC)
    a = out_dir / f"{FUNC}.intended.s"
    a.write_text("\n".join(s_lines[lo : hi + 1]) + "\n")
    dis = Path(objdump_path).read_text().split("\n")
    st = next(i for i, l in enumerate(dis) if re.match(rf"^[0-9a-f]{{16}} <{FUNC}>:", l))
    en = next((i for i in range(st + 1, len(dis)) if re.match(r"^[0-9a-f]{16} <", dis[i])), len(dis))
    b = out_dir / f"{FUNC}.objdump.s"
    head = next(l for l in dis[:5] if l.strip().endswith("file format elf64-amdgpu"))
    b.write_text("\n".join([head, ""] + dis[st:en]) + "\n")
    return VO.compare(str(a), str(b), log=log)


REGION_KINDS = ("FRONT", "STAGE_LOOP", "STAGE_MAIN", "STAGE_PEN", "STAGE_LAST", "EPI")
# region graph: the regions that can run right after each one (shell code in between)
NEXT = {"FRONT": ["STAGE_LOOP", "STAGE_MAIN", "STAGE_PEN", "STAGE_LAST"], "STAGE_LOOP": ["STAGE_MAIN"],
        "STAGE_MAIN": ["STAGE_MAIN", "STAGE_PEN", "STAGE_LAST"], "STAGE_PEN": ["STAGE_LAST"], "STAGE_LAST": ["EPI"],
        "EPI": ["FRONT"]}  # fmt: skip


def parse_markers(lines, lo, hi):
    """{kind: (operands, line index)}: exactly one marker per region kind, each a lone inline asm statement."""
    found = {}
    for i in range(lo, hi):
        m = MARK.search(lines[i])
        if not m:
            continue
        if ";;#ASMSTART" not in lines[i - 1] or ";;#ASMEND" not in lines[i + 1]:
            raise ValueError(f"marker at line {i + 1} is not a lone inline asm statement")
        k = m.group(1)
        if k in found:
            raise ValueError(f"a second {k} marker at line {i + 1}")
        found[k] = (dict(kv.split("=", 1) for kv in m.group(2).split()), i)
    if sorted(found) != sorted(REGION_KINDS):
        raise ValueError(f"markers {sorted(found)}, want {sorted(REGION_KINDS)}")
    return found


def sbase_of(op):
    m = re.fullmatch(r"s\[(\d+):(\d+)\]|s(\d+)", op)
    return int(m.group(1) or m.group(3))


def nsg_of(op):
    m = re.fullmatch(r"s\[(\d+):(\d+)\]|s(\d+)", op)
    return int(m.group(2)) - int(m.group(1)) + 1 if m.group(1) else 1


def with_unknown(fl, regs):
    fl = fl.copy()
    for st in fl.sts:
        st.valu.append(E.Rec("UNK", [], sorted(regs), False, 0))
    for x in fl.xs:
        x.unknown = True
    return fl


def tail(fl):
    """fl after the region tail POST (s_wait_xcnt 0, vm_vsrc 0)."""
    fl = fl.copy()
    for x in fl.xs:
        x.unknown, x.regs = False, set()
    for st in fl.sts:
        E.apply_vm(st, 0)
    return fl


def generate(lines, lo, hi, markers, fmap, rec, va_rule):
    fail = rec["fail"]
    ins = parse(lines[lo : hi + 1])
    shell_hi = max([-1] + [r for x in ins if x.mn for r in x.defs + x.uses])
    base = shell_hi + 1
    L = G.Layout(base)
    rec["layout"] = {"base": base, "top": L.top}
    log(f"shell: highest VGPR v{shell_hi}; base v{base}; generated state up to v{L.top - 1}")
    if L.top > 1024:
        fail.append(f"layout top v{L.top} > 1024")
    shell_regs = set(range(base))
    regions, cyc = {}, {}
    for kind in REGION_KINDS:
        ops = markers[kind][0]
        if kind == "FRONT":
            bl, c = G.region_front(L, ops)
        elif kind == "EPI":
            bl, c = G.region_epi(L, ops)
        elif kind == "STAGE_LOOP":
            bl, c = G.region_loop(L, ops)
        else:
            bl, c = G.region_stage(L, ops, kind.split("_")[1])
        regions[kind] = bl
        cyc.update(c)
    rec["model_cycles"] = cyc
    log(f"model cycles: {cyc}")
    # blocks of every region, keyed (kind, name); "X" = the region's exit
    items, succ, preds = {}, {}, {}
    for kind, bl in regions.items():
        for name, texts, sc in bl:
            succ[(kind, name)] = [(kind, s) for s in sc]
    # label placeholders: @NAME -> .LHKF_<kind>_<NAME>
    lab = lambda kind, name: f".LHKF_{kind}_{name}"  # noqa: E731
    for kind, bl in regions.items():
        for k, (name, texts, sc) in enumerate(bl):
            texts = [re.sub(r"@(\w+)", lambda m, kind=kind: lab(kind, m.group(1)), t) for t in texts]
            bl[k] = (name, texts, sc)
            items[(kind, name)] = items_of(texts, fmap)
    keys = list(items)
    for k in keys:
        preds[k] = []
    for k in keys:
        for s in succ[k]:
            if s[1] != "X":
                preds[s].append(k)
    entry, out, exits, sig = {}, {}, {}, {}
    first = {kind: (kind, regions[kind][0][0]) for kind in regions}
    rpreds = {kind: [] for kind in regions}
    for kind, nx in NEXT.items():
        for n2 in nx:
            rpreds[n2].append(kind)
    for it in range(256):
        changed = False
        # region exits: the joined exit flows of the blocks that leave the region
        rexit = {}
        for kind in regions:
            fl = [exits[k] for k in keys if k[0] == kind and k in exits and any(s[1] == "X" for s in succ[k])]
            if fl:
                rexit[kind] = with_unknown(tail(M.Flow.join(fl)), shell_regs)
        for k in keys:
            kind, name = k
            if k == first[kind]:
                ins_fl = [rexit[p] for p in rpreds[kind] if p in rexit]
                if kind == "FRONT":
                    ins_fl.append(unknown_flow())
                ins_fl += [exits[p] for p in preds[k] if p in exits]
            else:
                ins_fl = [exits[p] for p in preds[k] if p in exits]
            if not ins_fl:
                continue
            fl_in = M.Flow.join(ins_fl + ([entry[k]] if k in entry and it >= 32 else []))
            if it >= 32 and len(fl_in.sts) > 1:
                fl_in.sts = [E.join(fl_in.sts)]
            if k == first[kind] and OPT["entry_va"]:
                fl_in = fl_in.copy()
                for st in fl_in.sts:
                    E.apply_va(st, 0)
            sg = fl_in.sig()
            if sig.get(k) == sg:
                continue
            sig[k], entry[k] = sg, fl_in
            out[k], exits[k] = M.emit_block(items[k], fl_in, fmap, va_rule=va_rule, bar_dscnt=False)
            if k == first[kind] and OPT["entry_va"]:
                out[k] = [ENTRY_VA] + out[k]
            changed = True
        if not changed:
            break
    else:
        fail.append("region entry states did not converge")
    rec["checks"]["fixed_point_iterations"] = it + 1
    spliced = {}
    stats = {}
    cold = []  # the rescale blocks (R*), out of line after the kernel's s_endpgm
    cold_of = {}
    for kind, bl in regions.items():
        lines_out = []
        for name, texts, sc in bl:
            if (kind, name) not in out:
                fail.append(f"{kind} block {name} never reached")
                continue
            blk = [f"{lab(kind, name)}:"] + out[(kind, name)]
            if name.startswith("R"):
                cold += blk
                cold_of.setdefault(kind, []).extend(blk)
            else:
                lines_out += blk
        lines_out += [f"{lab(kind, 'X')}:"] + post_of(kind)
        spliced[markers[kind][1]] = lines_out
        o = lines_out
        stats[kind] = {"lines": len(o), "wmma": sum("v_wmma" in l for l in o), "va_vdst0": sum("va_vdst(0)" in l for l in o),
                       "s_wait_alu": sum("s_wait_alu" in l for l in o), "v_nop": sum(l.strip() == "v_nop" for l in o),
                       "dscnt": sum("s_wait_dscnt" in l for l in o), "xcnt": sum("s_wait_xcnt" in l for l in o)}  # fmt: skip
        for l in o + [l for l in cold_of.get(kind, [])]:
            x = parse_line(0, l)
            if not x.mn:
                continue
            low = [r for r in x.defs if r < base]
            if low:
                fail.append(f"{kind} writes shell VGPRs {low}: {l.strip()}")
            clob = {"vcc", "vcc_lo", "vcc_hi", "scc"} if kind in G.SAVED else set(G.CLOBBERED)
            ok = (
                clob
                | G.SAVED.get(kind, set())
                | {
                    f"s{sbase_of(markers[kind][0][o]) + i}"
                    for o in G.INOUT.get(kind, ())
                    for i in range(nsg_of(markers[kind][0][o]))
                }
            )
            bad = [r for r in sgpr_defs_uses(x)[0] if r not in ok]
            if bad and not x.mn.startswith("s_wait") and x.cls != "MSB":
                fail.append(f"{kind} writes SGPRs {bad}: {l.strip()}")
    for k in keys:
        if k in out:
            found, cov = open_hazards(out[k], entry[k])
            if found or cov:
                fail.append(
                    f"{k}: {len(found)} open hazards, {len(cov)} coverage errors, e.g. {(found + cov)[:3]}"
                )
    rec["blocks"] = stats
    log(f"regions: {json.dumps(stats)}")
    ends = [i for i in range(lo, hi + 1) if lines[i].strip() == "s_endpgm"]
    if not ends:
        fail.append("no s_endpgm in the kernel")
    elif cold:
        spliced[ends[-1]] = cold
    return spliced


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--cxx", default="amdclang++", help="compiler (the manifest's toolchain)")
    ap.add_argument("--va-rule", default="class", choices=E.VA_RULES)
    ap.add_argument(
        "--no-entry-va", action="store_true", help="no s_wait_alu depctr_va_vdst(0) at the region starts"
    )
    ap.add_argument("--stage-xcnt", action="store_true", help="end the stage regions with s_wait_xcnt 0 too")
    ap.add_argument(
        "--wmma-reuse", default="", help="a / b / ab: WMMA operand-reuse hints (QK: K tile, PV: P tile)"
    )
    # the scheduler's issue model (bwd1gen): RAW latencies (cycles from issue), the TRANS rate, the LDS model
    for k, d, h in (
        ("trans-lat", None, "v_exp result latency"),
        ("trans-rate", 0, "v_exp completions: one per N cycles"),
        ("valu-lat", None, "VALU result latency"),
        ("pk-lat", 9, "v_pk_* result latency"),
        ("sdst-lat", 10, "VALU-written SGPR -> SALU / memory reader"),
        ("lds-lat", None, "LDS latency"),
        ("lds-rate", None, "LDS completions: one per N cycles"),
        ("ds-max", None, "LDS loads in flight"),
    ):
        ap.add_argument(f"--{k}", type=int, default=d, help=h)
    a = ap.parse_args()
    OPT["entry_va"], OPT["stage_xcnt"] = not a.no_entry_va, a.stage_xcnt
    # LLVM's semantics: a tensor_load_to_lds reads its descriptor SGPRs at issue (the compiler rewrites them for the
    # next stage with no s_wait_xcnt); only VMEM / FLAT sources wait for XCNT
    M.XCNT_CLASSES = ("VMEM", "FLAT")
    G.REUSE["a"], G.REUSE["b"] = "a" in a.wmma_reuse, "b" in a.wmma_reuse
    GC.set_model(
        LDS_LAT=a.lds_lat,
        LDS_RATE=a.lds_rate,
        PK_LAT=a.pk_lat,
        SDST_LAT=a.sdst_lat,
        DS_MAX=a.ds_max,
        TRANS_RATE=a.trans_rate,
    )
    if a.trans_lat is not None:
        GC.LAT["TRANS"] = a.trans_lat
    if a.valu_lat is not None:
        GC.LAT["VALU"] = a.valu_lat
    hsaco = load_hsaco()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rec = {
        "recipe": {
            "va_rule": a.va_rule,
            "args": sys.argv[1:],
            "opt": dict(OPT),
            "model": {
                k: getattr(GC, k)
                for k in ("LDS_LAT", "LDS_RATE", "PK_LAT", "SDST_LAT", "DS_MAX", "TRANS_RATE")
            }
            | {"LAT": dict(GC.LAT)},
        },
        "checks": {},
        "fail": [],
    }
    fail = rec["fail"]
    cmd, s_path, _ = compile_variant(hsaco, a.cxx, out / "shell", reference=False)
    rec["recipe"]["compile"] = cmd
    shell = s_path.read_text().rstrip("\n").split("\n")
    (out / "shell.s").write_text("\n".join(shell) + "\n")
    _, ref_s, ref_obj = compile_variant(hsaco, a.cxx, out / "ref", reference=True)
    shutil.copy2(ref_obj, out / "ref" / f"{ENTRY}.hsaco")
    shutil.copy2(ref_s, out / "ref" / f"{ENTRY}.s")
    fmap = M.merge_msb_maps(
        M.learn_msb_map(parse(shell)), M.learn_msb_map(parse(ref_s.read_text().split("\n")))
    )
    for mn in ("ds_load_b128", "ds_load_tr16_b128"):
        fmap[(mn, 0)], fmap[(mn, 1)] = {"dst"}, {"src0"}
    fmap[("ds_store_b128", 0)], fmap[("ds_store_b128", 1)] = {"src0"}, {"src1"}
    for mn in ("buffer_store_b128", "buffer_store_b32"):
        fmap[(mn, 0)], fmap[(mn, 1)] = {"dst"}, {"src0"}
    # v_div_scale_f32 vdst, sdst, src0, src1, src2 (the compiles give src0 = src1: ambiguous)
    fmap[("v_div_scale_f32", 0)] = {"dst"}
    fmap[("v_div_scale_f32", 2)], fmap[("v_div_scale_f32", 3)], fmap[("v_div_scale_f32", 4)] = (
        {"src0"},
        {"src1"},
        {"src2"},
    )
    lo, hi = func_range(shell, FUNC)
    for x in parse(shell[lo : hi + 1]):
        if x.cls in ("XDL", "TRANS"):
            fail.append(f"shell {FUNC} has a WMMA / TRANS: {x.text.strip()}")
    markers = parse_markers(shell, lo, hi)
    log(f"markers: { {k: v[1] for k, v in markers.items()} }")
    spliced = generate(shell, lo, hi, markers, fmap, rec, a.va_rule)
    new = list(shell)
    for i in sorted(spliced, reverse=True):
        new[i + 1 : i + 1] = spliced[i]
    new = set_vgpr_count_of(new, FUNC, 1024)
    spl = out / f"{ENTRY}.s"
    spl.write_text("\n".join(new) + "\n")
    cc1as, lld = B.driver_steps(cmd, str(HIP))
    meta = B.assemble(spl, out / f"{ENTRY}.hsaco", cc1as, lld, cwd=str(HIP))
    rec["hsaco"] = meta
    log(f"hsaco: {meta}")
    bad = verify_func(new, out / f"{ENTRY}.objdump.s", out)
    rec["checks"]["verify_obj"] = "ok" if not bad else [str(b) for b in bad]
    if bad:
        fail.append(f"verify_obj: {bad[:3]}")
    if (
        meta.get("vgpr_spill_count", 0)
        or meta.get("sgpr_spill_count", 0)
        or meta.get("private_segment_fixed_size", 0)
    ):
        fail.append(f"spills / scratch: {meta}")
    for f in fail:
        log("FAIL:", f)
    rec["verdict"] = "PASS" if not fail else "FAIL"
    (out / "build.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    log(f"VERDICT {rec['verdict']}  {out / (ENTRY + '.hsaco')}  sha256 {meta.get('sha256')}")
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
