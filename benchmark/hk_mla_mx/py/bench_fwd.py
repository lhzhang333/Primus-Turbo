#!/usr/bin/env python3
"""Time the MXFP8 MLA forward (kernel only, and with its quantization) against the bf16 HK forward, same process.

    PYTHONPATH=<Primus-Turbo-hk-mx> python3 bench_fwd.py [--b 2 --s 4096 --h 128] [--rounds 8] [--iters 20] [--tags ,v1]

Arms: hk_bf16 (primus_turbo.hipkittens.gfx1250_mla fwd), quant (k_mxq_kbar + 2 x k_mxq_qk + k_mxq_vt), mx<tag> (the
MX forward of kern/build/k_mla_fwd_mx<tag>.hsaco on pre-quantized operands). Per round and arm: --settle-s of calls,
hot = one event pair over --iters back-to-back calls, cold = median of --iters calls each after a --flush-mb fill.
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
ap.add_argument("--rounds", type=int, default=8)
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--settle-s", type=float, default=1.0)
ap.add_argument("--flush-mb", type=int, default=512)
ap.add_argument("--tags", default="", help="comma list of k_mla_fwd_mx build tags ('' = the default build)")
ap.add_argument("--no-hk", action="store_true")
ap.add_argument("--build", action="store_true", help="build every tag first")
a = ap.parse_args()

dev = torch.device("cuda")
b, s, h = a.b, a.s, a.h
g = torch.Generator(device=dev).manual_seed(0)


def sbhd(d):
    return torch.randn(s, b, h, d, generator=g, device=dev).to(torch.bfloat16).transpose(0, 1)


q, k, v = sbhd(L.D_QK), sbhd(L.D_QK), sbhd(L.D_V)
if a.build:
    L.build()
    for t in a.tags.split(","):
        L.build_tag("fwd", t)
qzs = {r: L.quant_inputs(q, k, v, recipe=r) for r in sorted({L.recipe_of(t) for t in a.tags.split(",")})}
flops = 2.0 * b * h * (s * (s + 1) / 2) * (L.D_QK + L.D_V)
flush = torch.empty(a.flush_mb * (1 << 20), dtype=torch.uint8, device=dev)
arms = {}
if not a.no_hk:
    from primus_turbo.hipkittens.gfx1250_mla import fwd as hkf

    scale = L.SCALE

    qc, kc, vc = q.contiguous(), k.contiguous(), v.contiguous()  # the HK kernel takes contiguous [b, s, h, d]

    def hk():
        return hkf.hk_attn_fwd(qc, kc, vc, scale, True)

    try:
        hk()
        arms["hk_bf16"] = hk
    except Exception as e:  # noqa: BLE001
        print("hk_bf16 unavailable:", type(e).__name__, str(e)[:200])
arms["quant"] = lambda: L.quant_inputs(q, k, v, recipe="m4")
for tag in a.tags.split(","):
    out = torch.empty(s, b, h, L.D_V, device=dev, dtype=torch.bfloat16).transpose(0, 1)
    arms["mx" + tag] = lambda t=tag, o=out: L.fwd_mx(qzs[L.recipe_of(t)], out=o, tag=t)

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
    if n != "quant":
        summary[n]["cold_tflops"] = round(flops / cold / 1e9, 1)
print("PERF", json.dumps({"b": b, "s": s, "h": h, **summary}))
