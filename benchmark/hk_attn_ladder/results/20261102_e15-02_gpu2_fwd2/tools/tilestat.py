#!/usr/bin/env python3
"""tilestat.py <spliced k_mla_fwd.s> [region]: instruction mix of a generated region: WMMAs, VALU, DS loads, DS->DS
adjacencies (consecutive LDS loads, ignoring s_set_vgpr_msb), s_set_vgpr_msb count, s_wait_dscnt count."""
import re, sys
s, region = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "TILE_C")
lines = open(s).read().split("\n")
i = next(k for k, l in enumerate(lines) if re.search(rf";\s*HKGEN {region}\b", l))
j = next(k for k in range(i, len(lines)) if ";;#ASMEND" in lines[k])
ins = [l.strip() for l in lines[i + 1 : j] if l.startswith("\t") and not l.strip().startswith(";")]
msb = sum(1 for x in ins if x.startswith("s_set_vgpr_msb"))
core = [x for x in ins if not x.startswith("s_set_vgpr_msb")]
ds = [k for k, x in enumerate(core) if x.startswith("ds_load")]
adj = sum(1 for k in ds if k + 1 < len(core) and core[k + 1].startswith("ds_load"))
w = sum(1 for x in core if x.startswith("v_wmma"))
v = sum(1 for x in core if x.startswith("v_") and not x.startswith("v_wmma"))
wait = sum(1 for x in core if x.startswith("s_wait_dscnt"))
print(f"{region}: instr {len(core)} wmma {w} valu {v} ds {len(ds)} ds->ds {adj} msb {msb} wait_dscnt {wait}")
