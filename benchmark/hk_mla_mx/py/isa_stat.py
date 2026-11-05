"""Static ISA statistics of a gfx1250 kernel's loops (amdclang++ -S output).

    python isa_stat.py build/asm/k_mla_fwd_mx3.s [--kernel k_mla_fwd_mx] [--seq] [--loop N]

For every loop (a label with a backward branch to it): instruction counts per class (WMMA, VALU, TRANS, LDS read /
write, TDM, waits, barriers, SALU, hazard nops / delays), and with --seq the class sequence of the largest loop,
compressed (W = WMMA, v = VALU, T = TRANS, L = LDS read, S = LDS write, w<cnt> = s_wait_dscnt, b = barrier, D = TDM,
. = SALU, n = s_nop / s_delay_alu).
"""

import argparse
import re
from collections import Counter

TRANS = re.compile(r"^v_(exp|log|rcp|rsq|sqrt|sin|cos)_(f32|f16|bf16)")


def classify(op):
    if op.startswith("v_wmma") or op.startswith("v_swmmac"):
        return "W"
    if TRANS.match(op):
        return "T"
    if op.startswith("v_"):
        return "v"
    if op.startswith("ds_load") or op.startswith("ds_read"):
        return "L"
    if op.startswith("ds_store") or op.startswith("ds_write"):
        return "S"
    if op.startswith("ds_"):
        return "ds"
    if op.startswith("tensor_"):
        return "D"
    if op.startswith("global_") or op.startswith("buffer_") or op.startswith("flat_"):
        return "G"
    if op.startswith("s_wait_"):
        return "w"
    if op.startswith("s_barrier"):
        return "b"
    if op in ("s_nop", "s_delay_alu"):
        return "n"
    if op.startswith("s_"):
        return "."
    return "?"


def parse(path, kernel=None):
    lines = open(path).read().splitlines()
    body, inside = [], kernel is None
    for ln in lines:
        if kernel is not None and re.match(rf"^{re.escape(kernel)}:", ln):
            inside = True
            continue
        if inside and kernel is not None and ln.startswith(".Lfunc_end"):
            break
        if inside:
            body.append(ln)
    out = []  # (kind, text): kind "label" or "ins"
    for ln in body:
        s = ln.split(";")[0].strip()
        if not s:
            continue
        if s.endswith(":") and not s.startswith("."):
            continue
        if re.match(r"^\.LBB\d+_\d+:$", s):
            out.append(("label", s[:-1]))
        elif not s.startswith("."):
            out.append(("ins", s))
    return out


def loops(items):
    pos = {t: i for i, (k, t) in enumerate(items) if k == "label"}
    found = []
    for i, (k, t) in enumerate(items):
        if k != "ins" or not t.startswith("s_cbranch") and not t.startswith("s_branch"):
            continue
        target = t.split()[-1]
        if target in pos and pos[target] < i:
            found.append((pos[target], i, target))
    return found


def stats(items, a, b):
    c = Counter()
    seq = []
    for k, t in items[a : b + 1]:
        if k != "ins":
            continue
        op = t.split()[0]
        cl = classify(op)
        c[cl] += 1
        if cl == "w":
            arg = t.split()[1] if len(t.split()) > 1 else ""
            c[op] += 1
            seq.append(f"w[{op[7:]}{arg}]")
        else:
            seq.append(cl)
    return c, seq


def compress(seq):
    out, prev, n = [], None, 0
    for s in seq + [None]:
        if s == prev:
            n += 1
            continue
        if prev is not None:
            out.append(prev if n == 1 else f"{prev}{n}")
        prev, n = s, 1
    return " ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("asm")
    ap.add_argument("--kernel", default=None)
    ap.add_argument("--seq", action="store_true")
    ap.add_argument("--loop", type=int, default=None, help="index into the loop list (default: the largest)")
    args = ap.parse_args()
    items = parse(args.asm, args.kernel)
    ls = loops(items)
    total, _ = stats(items, 0, len(items) - 1)
    print(f"{args.asm}: {sum(v for k, v in total.items() if len(k) <= 2)} instructions, "
          + ", ".join(f"{k} {v}" for k, v in sorted(total.items()) if len(k) <= 2))  # fmt: skip
    for j, (a, b, lab) in enumerate(ls):
        c, _ = stats(items, a, b)
        print(f"loop {j} {lab} [{a}..{b}]: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))
    if args.seq and ls:
        j = args.loop if args.loop is not None else max(range(len(ls)), key=lambda i: ls[i][1] - ls[i][0])
        a, b, lab = ls[j]
        _, seq = stats(items, a, b)
        print(f"\nsequence of loop {j} ({lab}):\n{compress(seq)}")


if __name__ == "__main__":
    main()
