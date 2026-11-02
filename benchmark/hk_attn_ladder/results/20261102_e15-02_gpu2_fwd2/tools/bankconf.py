#!/usr/bin/env python3
"""Static count of VALU bank conflicts in generated regions of a spliced .s (k_mla_fwd / k_dkdv96ds / k_dqds_c).

Rule (micro/wmma_mix, 2026-11-02, one wave per SIMD behind WMMAs): an instruction that reads its own destination
register(s) (accumulate form: a source operand overlaps the destination) costs one extra cycle per other VGPR
source operand whose banks (register % 4; a 64-bit pair covers two consecutive banks) intersect those of the
accumulated operand. Sources that are not the destination never conflict with each other (v_add v176, v180: free).

    bankconf.py <spliced .s> [--region NAME ...] [--top N]
Prints per HKGEN region: VALU count, accumulate-form count, conflicts, and the most frequent conflicting forms."""
import argparse
import collections
import re

MARK = re.compile(r";\s*HKGEN\s+(\w+)")
VR = re.compile(r"/\*v\[(\d+):(\d+)\]\*/|/\*v(\d+)\*/|(?<![\w/*])v\[(\d+):(\d+)\]|(?<![\w/*])v(\d+)\b")


def regs(op):
    """True VGPRs of one operand (comment numbers win)."""
    m = re.search(r"/\*v\[(\d+):(\d+)\]\*/", op)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    m = re.search(r"/\*v(\d+)\*/", op)
    if m:
        return [int(m.group(1))]
    m = re.match(r"\s*-?\|?v\[(\d+):(\d+)\]", op)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    m = re.match(r"\s*-?\|?v(\d+)\b", op)
    if m:
        return [int(m.group(1))]
    return []


def split_ops(s):
    out, cur, depth, i = [], "", 0, 0
    while i < len(s):
        if s.startswith("/*", i):
            j = s.index("*/", i)
            cur += s[i : j + 2]
            i = j + 2
            continue
        c = s[i]
        depth += (c in "[(") - (c in "])")
        if c == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += c
        i += 1
    if cur.strip():
        out.append(cur.strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("s")
    ap.add_argument("--region", nargs="*")
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()
    cur, stats = None, collections.OrderedDict()
    forms = collections.defaultdict(collections.Counter)
    for raw in open(a.s):
        m = MARK.search(raw)
        if m:
            cur = m.group(1) + (" " + raw.split("nkv=")[1].split()[0] if "nkv=" in raw else "")
            stats.setdefault(cur, collections.Counter())
            continue
        if ";;#ASMEND" in raw:
            cur = None
            continue
        if cur is None or not raw.startswith("\t"):
            continue
        t = raw.strip().split(";")[0].strip()
        if not t.startswith("v_") or t.startswith("v_wmma") or t.startswith("v_nop"):
            continue
        mn, _, rest = t.partition(" ")
        ops = split_ops(rest)
        if not ops:
            continue
        st = stats[cur]
        st["valu"] += 1
        if mn.startswith("v_cmp"):
            continue
        dst = set(regs(ops[0]))
        srcs = [regs(o) for o in ops[1:]]
        acc = [s for s in srcs if s and set(s) & dst]
        if not acc:
            continue
        st["acc"] += 1
        ab = {r % 4 for r in acc[0]}
        n = sum(1 for s in srcs if s and s is not acc[0] and not (set(s) & dst) and ({r % 4 for r in s} & ab))
        if n:
            st["conf"] += n
            st["conf_instr"] += 1
            forms[cur][mn] += n
    for name, st in stats.items():
        if a.region and name.split()[0] not in a.region:
            continue
        print(f"{name:<14s} VALU {st['valu']:5d}  accumulate-form {st['acc']:5d}  conflicting instr {st['conf_instr']:5d}"
              f"  conflict cycles {st['conf']:5d}   top: " + ", ".join(f"{k} {v}" for k, v in forms[name].most_common(a.top)))


if __name__ == "__main__":
    main()
