#!/usr/bin/env python3
"""Build k_mla_fwd_mx variants: build_tags.py TAG=DEF1,DEF2 ... (e.g. _abl1=ABL=1). Writes kern/build/k_mla_fwd_mx<TAG>.hsaco."""
import sys

import mxlab as L

L.build()
for spec in sys.argv[1:]:
    tag, _, defs = spec.partition("=")
    defines = [d for d in defs.split("+") if d]
    L.build(["k_mla_fwd_mx.hip"], defines=defines, tag=tag)
    print("built", tag, defines)
