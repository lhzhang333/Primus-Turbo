#!/usr/bin/env python3
"""Build kernel variants: build_tags.py [--src k_mla_fwd_mx3.hip] [--x] TAG=DEF1+DEF2 ...

Writes kern/build/<stem of --src><TAG>.hsaco (load it with tag = <stem suffix after k_mla_fwd_mx><TAG>, e.g. --src
k_mla_fwd_mx3.hip --x x=  ->  k_mla_fwd_mx3x.hsaco, tag "3x"). --x adds the expert scheduling mode flags.
"""

import argparse

import mxlab as L

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="k_mla_fwd_mx.hip")
ap.add_argument("--x", action="store_true", help="-mllvm -amdgpu-expert-scheduling-mode")
ap.add_argument("--xp", action="store_true", help="--x + -mllvm -greedy-regclass-priority-trumps-globalness")
ap.add_argument("--no-default", action="store_true", help="skip building the default kernels first")
ap.add_argument("specs", nargs="*", help="TAG=DEF1+DEF2 (TAG may be empty, DEFs too)")
a = ap.parse_args()
if not a.no_default:
    L.build()
for spec in a.specs or ["="]:
    tag, _, defs = spec.partition("=")
    defines = [d for d in defs.split("+") if d]
    flags = L.EXPERT_P if a.xp else (L.EXPERT if a.x else ())
    out = L.build([a.src], defines=defines, tag=tag, flags=flags)
    print("built", out[a.src].name, defines, " ".join(flags))
