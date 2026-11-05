#!/usr/bin/env python3
"""Per-kernel time of the MXFP8 quantization kernels at the DSv3 shape (hot: back-to-back; GB/s of their traffic)."""

import argparse
import json

import mxlab as L
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--b", type=int, default=2)
ap.add_argument("--s", type=int, default=4096)
ap.add_argument("--h", type=int, default=128)
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--tag", default="")
a = ap.parse_args()
dev = torch.device("cuda")
b, s, h = a.b, a.s, a.h
g = torch.Generator(device=dev).manual_seed(0)


def sbhd(d):
    return torch.randn(s, b, h, d, generator=g, device=dev).to(torch.bfloat16).transpose(0, 1)


q, k, v = sbhd(L.D_QK), sbhd(L.D_QK), sbhd(L.D_V)
kbar = L.kbar_of(k, a.tag)
n192, n128 = b * s * h * 192, b * s * h * 128
arms = {
    "kbar": (lambda: L.kbar_of(k, a.tag), 2 * n192),
    "rows192_q": (lambda: L.quant_rows(q, 192, mode=2, kbar=kbar, tag=a.tag), 2 * n192 + n192 + n192 // 24),
    "rows192_k": (lambda: L.quant_rows(k, 192, mode=1, kbar=kbar, tag=a.tag), 2 * n192 + n192 + n192 // 24),
    "rows128_v": (lambda: L.quant_rows(v, 128, tag=a.tag), 2 * n128 + n128 + n128 // 32),
    "cols128_v": (lambda: L.quant_cols(v, 128, tag=a.tag), 2 * n128 + n128 + n128 // 32),
    "cols192_q": (lambda: L.quant_cols(q, 192, tag=a.tag), 2 * n192 + n192 + n192 // 32),
}
res = {}
for n, (f, nbytes) in arms.items():
    for _ in range(3):
        f()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(a.iters):
        f()
    e1.record()
    torch.cuda.synchronize()
    ms = e0.elapsed_time(e1) / a.iters
    res[n] = {"ms": round(ms, 4), "GBps": round(nbytes / ms / 1e6, 0)}
print("QUANT", json.dumps(res))
