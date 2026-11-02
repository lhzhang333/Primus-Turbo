#!/usr/bin/env python3
"""Branch costs from an ATT trace: per static branch, taken / not-taken counts and mean cycles from the branch's
issue to the next issued instruction. att_branches.py <ui_output dir> [--min N]"""
import argparse, collections, glob, json, os
ap = argparse.ArgumentParser(); ap.add_argument("ui"); ap.add_argument("--min", type=int, default=100)
a = ap.parse_args()
code = json.load(open(os.path.join(a.ui, "code.json")))["code"]
va = [c[5] for c in code]
order = sorted(range(len(code)), key=lambda i: va[i])
nxt_addr = {}
hit = [i for i in order if code[i][6] > 0]  # traced instructions only (s_set_vgpr_msb is not traced)
for k, i in enumerate(hit[:-1]):
    nxt_addr[i] = va[hit[k + 1]]
st = collections.defaultdict(collections.Counter)
for f in glob.glob(os.path.join(a.ui, "se*_wv*.json")):
    ins = json.load(open(f))["wave"]["instructions"]
    for k in range(len(ins) - 1):
        i = ins[k][4]
        t = code[i][0]
        if not (t.startswith("s_cbranch") or t.startswith("s_branch")):
            continue
        j = ins[k + 1][4]
        taken = va[j] != nxt_addr.get(i)
        key = (i, taken)
        st[key]["n"] += 1
        st[key]["c"] += ins[k + 1][0] - ins[k][0]
        st[key]["dist"] = va[j] - va[i]
for (i, tk), c in sorted(st.items(), key=lambda kv: (va[kv[0][0]], kv[0][1])):
    if c["n"] < a.min:
        continue
    print(f"idx {i:5d} {code[i][0][:34]:34s} {'TAKEN' if tk else 'fall '} n={c['n']:6d} mean={c['c']/c['n']:6.1f} dist={c['dist']:+7d} B")
