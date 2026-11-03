###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""build.py -- build k_attn_bwd_v1 (prep, g4, dqcvt) with k_hk_bwd1_g4's body generated (bwd1gen).

    python3 build.py --out DIR [--cxx CXX] [--va-rule RULE]

 1. compile the marker shell (the manifest's command for k_attn_bwd_v1 plus -DHKGEN=1): DIR/shell.s; and the
    reference (the manifest's command as is: the HIP kernel, DIR/ref/k_attn_bwd_v1.hsaco), whose operands in high
    VGPR blocks teach, with the shell's, the s_set_vgpr_msb operand-field map;
 2. read the BODY marker of k_hk_bwd1_g4 and the registers LLVM gave its operands; base = 1 + the highest VGPR the
    shell's k_hk_bwd1_g4 references;
 3. generate the region's blocks (bwd1gen); asmkit.emit derives s_set_vgpr_msb, the expert-mode waits and the hazard
    v_nops (the loop blocks to a fixed point of their entry states); the region ends with s_wait_xcnt 0 + vm_vsrc 0;
    a re-walk of every block from its entry state must find no open hazard;
 4. splice the region behind the marker, k_hk_bwd1_g4's VGPR allocation 1024: DIR/k_attn_bwd_v1.s;
 5. assemble with the driver's own cc1as / lld (asmkit.build): DIR/k_attn_bwd_v1.hsaco; asmkit.verify_obj on
    k_hk_bwd1_g4;
 6. contract: the generated code writes VGPRs only at or above base and SGPRs only in the marker's clobber list; no
    spills.
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
sys.path[:0] = [str(HERE), str(TOOLS)]
import bwd1gen as G  # noqa: E402
from asmkit import build as B  # noqa: E402
from asmkit import emit as M  # noqa: E402
from asmkit import expert as E  # noqa: E402
from asmkit import verify_obj as VO  # noqa: E402
from asmkit.isa import coverage_errors, parse, parse_line, sgpr_defs_uses  # noqa: E402

ENTRY = "k_attn_bwd_v1"
FUNC = "k_hk_bwd1_g4"
MARK = re.compile(r";\s*HKGEN\s+(\w+)\s*(.*)$")
POST = ["\ts_wait_xcnt 0x0", "\ts_wait_alu depctr_vm_vsrc(0)"]
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
    if not reference:
        cmd = cmd[: cmd.index("-c")] + ["-DHKGEN=1"] + cmd[cmd.index("-c") :]
    r = subprocess.run(cmd, cwd=HIP, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"compile failed:\n{r.stderr[-4000:]}")
    return cmd, next(tmp.glob("*-gfx1250.s")), obj


def func_range(lines, name):
    """(first, last) line index of function `name` in a compiler .s: its label to its .Lfunc_end line."""
    lo = next(i for i, l in enumerate(lines) if l.startswith(f"{name}:"))
    hi = next(i for i in range(lo, len(lines)) if lines[i].startswith(".Lfunc_end"))
    return lo, hi


def parse_marker(lines, lo, hi):
    """(operands, line index) of the one BODY marker of the function."""
    found = []
    for i in range(lo, hi):
        m = MARK.search(lines[i])
        if not m:
            continue
        if ";;#ASMSTART" not in lines[i - 1] or ";;#ASMEND" not in lines[i + 1]:
            raise ValueError(f"marker at line {i + 1} is not a lone inline asm statement")
        found.append((m.group(1), dict(kv.split("=", 1) for kv in m.group(2).split()), i))
    if [f[0] for f in found] != ["BODY"]:
        raise ValueError(f"markers {[f[0] for f in found]}, want one BODY")
    return found[0][1], found[0][2]


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


def generate(lines, lo, hi, ops, mline, fmap, rec, va_rule):
    """The region's lines (the blocks, emitted from their fixed-point entry flows) or None; failed checks to
    rec["fail"]."""
    fail = rec["fail"]
    ins = parse(lines[lo : hi + 1])
    shell_hi = max([-1] + [r for x in ins if x.mn for r in x.defs + x.uses])
    base = shell_hi + 1
    for k, v in ops.items():
        m = re.fullmatch(r"v(\d+)|v\[(\d+):(\d+)\]", v)
        if m:
            base = max(base, int(m.group(3) or m.group(1)) + 1)
    L = G.Layout(base)
    rec["layout"] = {"base": base, "top": L.top}
    log(f"shell {FUNC}: highest VGPR v{shell_hi}; base v{base}; generated state up to v{L.top - 1}")
    if L.top > 1024:
        fail.append(f"layout top v{L.top} > 1024")
    lp = ".LHKG"
    blocks, cyc = G.region_body(L, ops, lp)
    rec["model_cycles"] = cyc
    log(f"model cycles per step: {cyc}")
    names = [b[0] for b in blocks]
    items = {name: items_of(texts, fmap) for name, _, texts, _ in blocks}
    preds = {n: [] for n in names}
    for name, _, _, succ in blocks:
        for t in succ:
            preds[t].append(name)
    entry, out, exits, sig = {}, {}, {}, {}
    for it in range(64):
        changed = False
        for n in names:
            if n == "I":
                fl_in = unknown_flow()
            else:
                ins_fl = [exits[p] for p in preds[n] if p in exits]
                if not ins_fl:
                    continue
                fl_in = M.Flow.join(ins_fl)
            sg = fl_in.sig()
            if sig.get(n) == sg:
                continue
            sig[n], entry[n] = sg, fl_in
            out[n], exits[n] = M.emit_block(items[n], fl_in, fmap, va_rule=va_rule, bar_dscnt=False)
            changed = True
        if not changed:
            break
    else:
        fail.append("region entry states did not converge")
    rec["checks"]["fixed_point_iterations"] = it + 1
    region = []
    for name, label, _, _ in blocks:
        region += [f"{label}:"] + out[name]
    region += POST
    for l in region:
        x = parse_line(0, l)
        if not x.mn:
            continue
        low = [r for r in x.defs if r < base]
        if low:
            fail.append(f"writes shell VGPRs {low}: {l.strip()}")
        bad = [r for r in sgpr_defs_uses(x)[0] if r not in G.SCALARS]
        if bad and not x.mn.startswith("s_wait") and x.cls != "MSB":
            fail.append(f"writes SGPRs {bad}: {l.strip()}")
    for name in names:
        found, cov = open_hazards(out[name], entry[name])
        if found or cov:
            fail.append(f"{name}: {len(found)} open hazards, {len(cov)} coverage errors, e.g. {(found + cov)[:3]}")
    stats = {}
    for name in names:
        o = out[name]
        stats[name] = {
            "lines": len(o),
            "wmma": sum("v_wmma" in l for l in o),
            "va_vdst0": sum("va_vdst(0)" in l for l in o),
            "s_wait_alu": sum("s_wait_alu" in l for l in o),
            "v_nop": sum(l.strip() == "v_nop" for l in o),
            "msb": sum("s_set_vgpr_msb" in l for l in o),
            "xcnt": sum("s_wait_xcnt" in l for l in o),
            "dscnt": sum("s_wait_dscnt" in l for l in o),
        }
    rec["blocks"] = stats
    log(f"blocks: {json.dumps(stats)}")
    return region


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--cxx", default="amdclang++", help="compiler (the manifest's toolchain)")
    ap.add_argument("--va-rule", default="llvm_war_xdl", choices=E.VA_RULES)
    ap.add_argument("--ds-lat", type=int, help="model: LDS latency (cycles)")
    ap.add_argument("--ds-rate", type=int, help="model: LDS cycles per operation of the wave")
    ap.add_argument("--ds-block", type=int, help="model: issue cycles an LDS instruction holds")
    ap.add_argument("--sig-before", type=int, help="the barrier signal before this WMMA of the step")
    ap.add_argument("--wait-before", type=int, help="the barrier wait before this WMMA of the step")
    ap.add_argument("--tw-after", type=int, help="the signal's tensorcnt wait after this WMMA")
    a = ap.parse_args()
    hsaco = load_hsaco()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    G.set_model(LDS_LAT=a.ds_lat, LDS_RATE=a.ds_rate, DS_BLOCK=a.ds_block, SIG_BEFORE=a.sig_before,
                WAIT_BEFORE=a.wait_before, TW_AFTER=a.tw_after)
    rec = {"recipe": {"va_rule": a.va_rule, "model": {k: getattr(G, k) for k in ("LDS_LAT", "LDS_RATE", "DS_BLOCK", "SIG_BEFORE", "WAIT_BEFORE", "TW_AFTER")}},
           "checks": {}, "fail": []}  # fmt: skip
    fail = rec["fail"]
    cmd, s_path, _ = compile_variant(hsaco, a.cxx, out / "shell", reference=False)
    rec["recipe"]["compile"] = cmd
    shell = s_path.read_text().rstrip("\n").split("\n")
    (out / "shell.s").write_text("\n".join(shell) + "\n")
    _, ref_s, ref_obj = compile_variant(hsaco, a.cxx, out / "ref", reference=True)
    shutil.copy2(ref_obj, out / "ref" / f"{ENTRY}.hsaco")
    shutil.copy2(ref_s, out / "ref" / f"{ENTRY}.s")
    rec["ref_sha256"] = B.sha256(out / "ref" / f"{ENTRY}.hsaco")
    fmap = M.merge_msb_maps(M.learn_msb_map(parse(shell)), M.learn_msb_map(parse(ref_s.read_text().split("\n"))))
    for mn in ("ds_load_b128", "ds_load_tr16_b128"):
        fmap[(mn, 0)], fmap[(mn, 1)] = {"dst"}, {"src0"}
    fmap[("ds_store_b128", 0)], fmap[("ds_store_b128", 1)] = {"src0"}, {"src1"}
    # VBUFFER: vdata in the dst field, voffset in src0 (as the compiler's buffer_atomic_add_f32)
    fmap[("buffer_store_b128", 0)], fmap[("buffer_store_b128", 1)] = {"dst"}, {"src0"}
    lo, hi = func_range(shell, FUNC)
    for x in parse(shell[lo : hi + 1]):
        if x.cls in ("XDL", "TRANS"):
            fail.append(f"shell {FUNC} has a WMMA / TRANS: {x.text.strip()}")
    ops, mline = parse_marker(shell, lo, hi)
    log(f"marker operands: {ops}")
    region = generate(shell, lo, hi, ops, mline, fmap, rec, a.va_rule)
    new = list(shell)
    new[mline + 1 : mline + 1] = region
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
    if meta.get("vgpr_spill_count", 0) or meta.get("private_segment_fixed_size", 0):
        fail.append(f"spills / scratch: {meta}")
    for f in fail:
        log("FAIL:", f)
    rec["verdict"] = "PASS" if not fail else "FAIL"
    (out / "build.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    log(f"VERDICT {rec['verdict']}  {out / (ENTRY + '.hsaco')}  sha256 {meta.get('sha256')}")
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
