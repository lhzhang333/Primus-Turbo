#!/usr/bin/env python3
"""Per-region ATT breakdown of k_mla_fwd (one traced wave): maps each traced instruction to the HKGEN region it
belongs to (spliced .s markers; 'shell:<label>' outside), then per region: executions, cycles (issue to next
issue), WMMAs, XDL idle (cycles not covered by 8-cycle WMMA occupancy), and the top idle-causing instructions
(idle charged to the instruction issued right before a WMMA gap).

    att_regions.py <build dir with k_mla_fwd.s / .dis> <ui_output dir> [--top N] [--region R]"""
import argparse, collections, glob, json, os, re

MARK = re.compile(r";\s*HKGEN\s+(\w+)")


def s_instructions(path):
    """[(region, text)] for every instruction line of the spliced .s, in order."""
    out, cur, blk = [], None, "entry"
    for raw in open(path):
        line = raw.rstrip("\n")
        m = MARK.search(line)
        if m:
            cur = m.group(1)
            continue
        if ";;#ASMEND" in line:
            cur = None
            continue
        s = line.strip()
        if not s or s.startswith(";") or s.startswith("."):
            if re.match(r"^\.?L\w+:|^\.LBB\w+:", s):
                blk = s.rstrip(":")
            continue
        if re.match(r"^[\w.$]+:", s):
            blk = s.split(":")[0]
            continue
        if not line.startswith("\t"):
            continue
        out.append((cur if cur else f"shell", s.split(";")[0].strip(), blk))
    return out


def dis_addrs(path):
    out = []
    for line in open(path):
        m = re.search(r"//\s*([0-9A-Fa-f]{8,}):", line)
        if m and line.startswith("\t"):
            out.append((int(m.group(1), 16), line.split("//")[0].strip()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("build")
    ap.add_argument("ui")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--region", default=None)
    a = ap.parse_args()
    si = s_instructions(os.path.join(a.build, "k_mla_fwd.s"))
    da = dis_addrs(os.path.join(a.build, "k_mla_fwd.dis"))
    if len(si) != len(da):
        print(f"warning: .s {len(si)} instructions, .dis {len(da)}")
    reg_of = {}
    for (r, t, blk), (addr, dt) in zip(si, da):
        reg_of[addr] = (r, blk, dt)
    code = json.load(open(os.path.join(a.ui, "code.json")))["code"]
    vaddr = [c[5] for c in code]
    txt = [c[0] for c in code]
    fn = glob.glob(os.path.join(a.ui, "se*_sm*_sl*_wv*.json"))
    stats = collections.OrderedDict()
    idle_by = collections.defaultdict(collections.Counter)
    cyc_by = collections.defaultdict(collections.Counter)
    for f in fn:
        w = json.load(open(f))["wave"]
        ins, end = w["instructions"], w["end"]
        last_w_end = None  # XDL busy until
        prev_idx = None
        for k, r in enumerate(ins):
            t, idx = r[0], r[4]
            nxt = ins[k + 1][0] if k + 1 < len(ins) else end
            reg, blk, _ = reg_of.get(vaddr[idx], ("?", "?", ""))
            key = reg if reg != "shell" else f"shell"
            st = stats.setdefault(key, collections.Counter())
            st["cyc"] += nxt - t
            st["n"] += 1
            cyc_by[key][idx] += nxt - t
            if txt[idx].startswith("v_wmma"):
                st["wmma"] += 1
                if last_w_end is not None and t > last_w_end:
                    gap = t - last_w_end
                    st["idle"] += gap
                    if prev_idx is not None:
                        idle_by[key][prev_idx] += gap
                last_w_end = max(t, last_w_end or 0) + 8
            prev_idx = idx
    tot = sum(s["cyc"] for s in stats.values())
    print(f"{'region':<10s} {'cycles':>9s} {'%':>6s} {'instr':>8s} {'wmma':>7s} {'xdl idle (in gaps ending in region)':>36s}")
    for k, s in sorted(stats.items(), key=lambda kv: -kv[1]["cyc"]):
        print(f"{k:<10s} {s['cyc']:9d} {100*s['cyc']/tot:6.1f} {s['n']:8d} {s['wmma']:7d} {s['idle']:12d}")
    regs = [a.region] if a.region else [k for k, _ in sorted(stats.items(), key=lambda kv: -kv[1]["idle"])[:3]]
    for k in regs:
        print(f"\n== {k}: top XDL-gap instructions (gap charged to the instruction issued before the WMMA)")
        for idx, g in idle_by[k].most_common(a.top):
            print(f"  {g:8d}  {txt[idx][:90]}")
        print(f"== {k}: top cycle instructions")
        for idx, g in cyc_by[k].most_common(a.top):
            print(f"  {g:8d}  {txt[idx][:90]}")


if __name__ == "__main__":
    main()
