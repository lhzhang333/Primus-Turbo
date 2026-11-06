#!/usr/bin/env python3
"""Time the MXFP8 MLA backward (its quantization, k_dkdv_mx, k_dq_mx) against the bf16 HK backward, same process.

    PYTHONPATH=<Primus-Turbo-hk-mx> python3 bench_bwd.py [--b 2 --s 4096 --h 128] [--rounds 6] [--iters 10] [--tags ,x]

Arms: hk_bf16 (primus_turbo.hipkittens.gfx1250_mla bwd on contiguous BSHD), quant (the backward's extra MX operands:
V / dO rows + delta, dO / Q / K cols), and per build tag: dkdv<tag>, dq<tag> (one kernel each), mx<tag> (both kernels
on pre-quantized operands). Hot / cold as bench_fwd.py.
"""

import argparse
import json
import statistics
import time

import mxlab as L
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--b", type=int, default=2)
ap.add_argument("--s", type=int, default=4096)
ap.add_argument("--h", type=int, default=128)
ap.add_argument("--rounds", type=int, default=6)
ap.add_argument("--iters", type=int, default=10)
ap.add_argument("--settle-s", type=float, default=1.0)
ap.add_argument("--flush-mb", type=int, default=512)
ap.add_argument("--tags", default="", help="comma list of k_mla_bwd_mx build tags ('' = the default build)")
ap.add_argument("--no-hk", action="store_true")
ap.add_argument("--build", action="store_true", help="build every tag (and the forward of each recipe) first")
ap.add_argument("--fwd-m4", default="4xp", help="the forward build tag that makes the m4 recipe's o / lse")
a = ap.parse_args()

dev = torch.device("cuda")
b, s, h = a.b, a.s, a.h
g = torch.Generator(device=dev).manual_seed(0)


def sbhd(d):
    return torch.randn(s, b, h, d, generator=g, device=dev).to(torch.bfloat16).transpose(0, 1)


q, k, v, do = sbhd(L.D_QK), sbhd(L.D_QK), sbhd(L.D_V), sbhd(L.D_V)
tags = a.tags.split(",")
recipes = sorted({L.recipe_of(t, "bwd") for t in tags})
fwd_tag = {"r1": "", "m4": a.fwd_m4}
if a.build:
    L.build()
    for r in recipes:
        L.build_tag("fwd", fwd_tag[r])
    for t in tags:
        L.build_tag("bwd", t)
QZ, BZ, LSE = {}, {}, {}
for r in recipes:
    QZ[r] = L.quant_inputs(q, k, v, recipe=r)
    o_, LSE[r] = L.fwd_mx(QZ[r], tag=fwd_tag[r])
    BZ[r] = L.quant_bwd(QZ[r], q, k, v, o_, do)
    if r == "r1":
        o, lse, qz, bz = o_, LSE[r], QZ[r], BZ[r]
if "r1" not in recipes:
    o, lse, qz, bz = o_, LSE[recipes[0]], QZ[recipes[0]], BZ[recipes[0]]
flops = 2.0 * b * h * (s * (s + 1) / 2) * (2 * L.D_QK + 3 * L.D_V)  # S, dP, dV, dQ, dK
flush = torch.empty(a.flush_mb * (1 << 20), dtype=torch.uint8, device=dev)
arms = {}
if not a.no_hk:
    from primus_turbo.hipkittens.gfx1250_mla import bwd as hkb

    qc, kc, vc, oc, doc = (t.contiguous() for t in (q, k, v, o, do))
    lse_true = lse + L.SCALE * qz["qk"]  # the MX forward's lse is of the centred logits

    def hk():
        return hkb.hk_attn_bwd(doc, qc, kc, vc, oc, lse_true, L.SCALE, True)

    try:
        hk()
        arms["hk_bf16"] = hk
    except Exception as e:  # noqa: BLE001
        print("hk_bf16 unavailable:", type(e).__name__, str(e)[:200])
arms["quant"] = lambda: L.quant_bwd(qz, q, k, v, o, do)
dq = torch.empty(s, b, h, L.D_QK, device=dev, dtype=torch.bfloat16).transpose(0, 1)
dk = torch.empty_like(dq)
dv = torch.empty(s, b, h, L.D_V, device=dev, dtype=torch.bfloat16).transpose(0, 1)


def one(name, tag):
    r = L.recipe_of(tag, "bwd")
    qz, bz = QZ[r], BZ[r]
    lse = LSE[r] * L.LOG2E if r == "m4" else LSE[r]
    if name in ("k_dkdv_mx", "k_dv_mx", "k_dk_mx"):
        args = [qz["q8"], qz["qs"], qz["k8"], qz["ks"], bz["v8"], bz["vsr"], bz["do8"], bz["dos"], bz["do8t"],
                bz["dots"], bz["q8t"], bz["qts"], lse, bz["delta"], dk, dv, float(L.SCALE), s, h, b, dk.stride(0),
                dk.stride(1), dv.stride(0), dv.stride(1)]  # fmt: skip
        rows = 128 if name == "k_dkdv_mx" else (192 if name == "k_dk_mx" and tag[:1] == "4" else 256)
        grid = (-(-s // rows), h, b)
        if name == "k_dkdv_mx" and tag[:1] == "6":  # b6: persistent
            grid = (L.dkdv_persistent_grid(b, s, h), 1, 1)
    else:
        args = [qz["q8"], qz["qs"], qz["k8"], qz["ks"], bz["v8"], bz["vsr"], bz["do8"], bz["dos"], bz["k8t"],
                bz["kts"], lse, bz["delta"], dq, float(L.SCALE), s, h, b, dq.stride(0), dq.stride(1)]  # fmt: skip
        grid = (s // 256, h, b)
    return lambda: L._launch(name, grid, args, tag)


for tag in a.tags.split(","):
    if tag[:1] in ("2", "4"):
        arms["dv" + tag] = one("k_dv_mx", tag)
        arms["dk" + tag] = one("k_dk_mx", tag)
    else:
        arms["dkdv" + tag] = one("k_dkdv_mx", tag)
    arms["dq" + tag] = one("k_dq_mx", tag)
    rec = L.recipe_of(tag, "bwd")
    arms["mx" + tag] = lambda t=tag, r=rec: L.bwd_mx(QZ[r], BZ[r], LSE[r], tag=t)

for f in arms.values():
    f()
torch.cuda.synchronize()
res = {n: {"hot": [], "cold": []} for n in arms}
for r in range(a.rounds):
    order = list(arms) if r % 2 == 0 else list(arms)[::-1]
    for n in order:
        f = arms[n]
        t_end = time.time() + a.settle_s
        while time.time() < t_end:
            f()
            torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(a.iters):
            f()
        e1.record()
        torch.cuda.synchronize()
        res[n]["hot"].append(e0.elapsed_time(e1) / a.iters)
        cold = []
        for _ in range(a.iters):
            flush.fill_(1)
            c0, c1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            c0.record()
            f()
            c1.record()
            torch.cuda.synchronize()
            cold.append(c0.elapsed_time(c1))
        res[n]["cold"].append(statistics.median(cold))
summary = {}
for n, d in res.items():
    hot, cold = statistics.median(d["hot"]), statistics.median(d["cold"])
    summary[n] = {"hot_ms": round(hot, 4), "cold_ms": round(cold, 4)}
    if n == "hk_bf16" or n.startswith("mx"):
        summary[n]["cold_tflops"] = round(flops / cold / 1e9, 1)
print("PERF", json.dumps({"b": b, "s": s, "h": h, **summary}))
