#!/usr/bin/env python3
"""k_mxq_rc192 / rc128 against the separate rows / cols kernels: bitwise equal outputs (one read instead of two).

PYTHONPATH=<Primus-Turbo-hk-mx> python3 test_quant_rc.py [--b 2 --s 1024 --h 8]
"""

import argparse
import sys

import mxlab as L
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--b", type=int, default=2)
ap.add_argument("--s", type=int, default=1024)
ap.add_argument("--h", type=int, default=8)
ap.add_argument("--no-build", action="store_true")
a = ap.parse_args()
if not a.no_build:
    L.build()
dev = torch.device("cuda")
g = torch.Generator(device=dev).manual_seed(0)


def sbhd(d):
    return torch.randn(a.s, a.b, a.h, d, generator=g, device=dev).to(torch.bfloat16).transpose(0, 1)


q, k, v, o, do = sbhd(L.D_QK), sbhd(L.D_QK), sbhd(L.D_V), sbhd(L.D_V), sbhd(L.D_V)
ok = True
for recipe in ("r1", "m4"):
    qz = L.quant_inputs(q, k, v, recipe=recipe)
    bz = L.quant_bwd(qz, q, k, v, o, do)
    qz2, pre = L.quant_all(q, k, v, recipe=recipe)
    bz2 = L.quant_bwd_do(qz2, pre, o, do)
    torch.cuda.synchronize()
    for nm in ("q8", "qs", "k8", "ks", "v8t", "vs", "kbar", "qk"):
        eq = torch.equal(qz[nm], qz2[nm])
        ok &= eq
        print(recipe, nm, "equal" if eq else "DIFF")
    for nm in ("q8t", "qts", "k8t", "kts", "v8", "vsr", "do8", "dos", "delta", "do8t", "dots"):
        eq = torch.equal(bz[nm], bz2[nm])
        ok &= eq
        print(recipe, nm, "equal" if eq else "DIFF")
print("ALL EQUAL" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
