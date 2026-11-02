#!/usr/bin/env python3
"""XDL idle inside a region of k_mla_fwd from an ATT trace: for consecutive WMMAs of one region execution, gap =
t2 - t1 - 8 (> 0), classified by what was issued between them. att_xdl_gaps.py <build dir> <ui dir> [region]"""
import sys, json, glob, collections
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import att_regions as A
B, U = sys.argv[1], sys.argv[2]
region = sys.argv[3] if len(sys.argv) > 3 else "TILE_C"
si = A.s_instructions(B + "/k_mla_fwd.s"); da = A.dis_addrs(B + "/k_mla_fwd.dis")
reg = {addr: r for (r, t, b), (addr, dt) in zip(si, da)}
code = json.load(open(U + "/code.json"))["code"]
ins = json.load(open(glob.glob(U + "/se*_wv*.json")[0]))["wave"]["instructions"]
R = [reg.get(code[r[4]][5], "?") for r in ins]

def cls(t):
    op = t.split()[0]
    if op.startswith("s_cbranch") or op.startswith("s_branch"): return "branch"
    if op.startswith("v_pk_mul"): return "pk_mul"
    if op.startswith("v_permlane"): return "perm"
    if op.startswith("v_exp"): return "exp"
    if op.startswith("ds_"): return "ds"
    if op.startswith("s_wait_dscnt"): return "wait_ds"
    if op.startswith("s_wait"): return "wait"
    if op.startswith("tensor"): return "tdm"
    if op.startswith("s_"): return "salu"
    return "valu"

gaps = collections.Counter(); ngap = collections.Counter(); execs = 0
lastw = None
for i, r in enumerate(ins):
    t = code[r[4]][0]
    if not t.startswith("v_wmma") or R[i] != region:
        if R[i] != region:
            lastw = None
        continue
    if lastw is None:
        execs += 1
    else:
        g = r[0] - ins[lastw][0] - 8
        if g > 0:
            between = [cls(code[ins[j][4]][0]) for j in range(lastw + 1, i)]
            key = "+".join(sorted(set(between))) or "(none)"
            if "branch" in between: key = "branch" + ("+pk_mul" if "pk_mul" in between else "")
            elif "pk_mul" in between: key = "pk_mul(resc)"
            gaps[key] += g; ngap[key] += 1
    lastw = i
tot = sum(gaps.values())
print(f"{region}: {execs} executions, in-region XDL idle {tot / max(execs,1):.1f} cycles / execution")
for k, v in gaps.most_common(18):
    print(f"  {v / execs:7.1f}  (n/exec {ngap[k] / execs:5.1f})  {k}")
