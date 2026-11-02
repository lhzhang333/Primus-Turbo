#!/usr/bin/env python3
"""Per-window analysis of a region from an ATT trace: a window = the instructions from one WMMA up to the next WMMA of
the same region execution. Static mix per window (from the code between the two WMMA addresses, incl. untraced
s_set_vgpr_msb), measured length; prints length distribution grouped by mix signature.
    att_windows.py <build dir> <ui dir> [region]"""
import sys, json, glob, collections, re
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import att_regions as A
B, U = sys.argv[1], sys.argv[2]
region = sys.argv[3] if len(sys.argv) > 3 else "TILE_C"
si = A.s_instructions(B + "/k_mla_fwd.s"); da = A.dis_addrs(B + "/k_mla_fwd.dis")
reg = {addr: r for (r, t, b), (addr, dt) in zip(si, da)}
txt_at = {addr: dt for addr, dt in da}
addrs = [a for a, _ in da]
pos = {a: k for k, a in enumerate(addrs)}
code = json.load(open(U + "/code.json"))["code"]
ins = json.load(open(glob.glob(U + "/se*_wv*.json")[0]))["wave"]["instructions"]

def kind(t):
    op = t.split()[0]
    if op.startswith("v_wmma"): return None
    if op.startswith("s_set_vgpr_msb"): return "X"
    if op.startswith("ds_load"): return "L"
    if op.startswith("v_exp"): return "E"
    if op.startswith("v_pk_fma"): return "P"
    if op.startswith("v_pk_add"): return "K"
    if op.startswith("v_pk_mul"): return "Q"
    if op.startswith("v_cvt"): return "C"
    if op.startswith("v_max3"): return "3"
    if op.startswith("v_perm"): return "R"
    if op.startswith("s_wait_dscnt"): return "Z"
    if op.startswith("s_wait"): return "z"
    if op.startswith("s_cbranch") or op.startswith("s_branch"): return "B"
    if op.startswith("v_"): return "V"
    return "S"

stats = collections.defaultdict(list)
lastw = None
for k, r in enumerate(ins):
    a = code[r[4]][5]
    if reg.get(a) != region:
        lastw = None
        continue
    if not code[r[4]][0].startswith("v_wmma"):
        continue
    if lastw is not None:
        a0 = code[ins[lastw][4]][5]
        # static code between the two WMMAs (only if straight-line: a0 < a and contiguous)
        if a0 < a and pos[a] - pos[a0] < 40:
            mix = collections.Counter(kind(txt_at[addrs[j]]) for j in range(pos[a0] + 1, pos[a]))
            sig = "".join(sorted(c * n for c, n in mix.items() if c))
            stats[sig].append(r[0] - ins[lastw][0])
    lastw = k
rows = sorted(stats.items(), key=lambda kv: -len(kv[1]) * (sum(kv[1]) / len(kv[1]) - 8))
tot_excess = sum(sum(v) - 8 * len(v) for v in stats.values())
n_exec = 480 if region == "TILE_C" else 16
print(f"{region}: excess over 8 cycles per window, total {tot_excess / n_exec:.0f} cycles / execution")
for sig, v in rows[:30]:
    m = sum(v) / len(v)
    print(f"  {sig:22s} n/exec {len(v) / n_exec:5.1f}  mean {m:5.2f}  excess/exec {(sum(v) - 8 * len(v)) / n_exec:6.1f}")
