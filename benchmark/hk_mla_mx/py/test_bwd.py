#!/usr/bin/env python3
"""Correctness of the MXFP8 MLA backward (k_dkdv_mx, k_dq_mx) on one GPU.

    PYTHONPATH=<Primus-Turbo-hk-mx> python3 test_bwd.py [--shapes toy,small,mid,prod] [--fwd-tag 3]

Per shape: forward (k_mla_fwd_mx<fwd-tag>) for o / lse, the backward's quantization, the two backward kernels;
dq / dk / dv against the fp32 emulation of the kernels' algorithm on the same quantized operands (kernel
correctness, >= 35 dB) and against fp32 autograd on the bf16 inputs (recipe quality); determinism; finite.
"""

import argparse
import sys
import time

import mxlab as L
import torch

SHAPES = {
    "toy": dict(b=1, s=256, h=2),
    "small": dict(b=1, s=1024, h=4),
    "mid": dict(b=1, s=4096, h=16),
    "prod": dict(b=2, s=4096, h=128),
}
ap = argparse.ArgumentParser()
ap.add_argument("--shapes", default="toy,small,mid")
ap.add_argument("--fwd-tag", default="")
ap.add_argument(
    "--tag", default="", help='k_mla_bwd_mx<tag>.hsaco (built here: "x" = expert scheduling mode)'
)
ap.add_argument("--no-build", action="store_true")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
dev = torch.device("cuda")
if not a.no_build:
    t0 = time.time()
    L.build()
    L.build_tag("fwd", a.fwd_tag)
    L.build_tag("bwd", a.tag)
    print(f"built in {time.time() - t0:.1f} s", flush=True)

ok_all = True
for name in a.shapes.split(","):
    sh = SHAPES[name]
    b, s, h = sh["b"], sh["s"], sh["h"]
    g = torch.Generator(device=dev).manual_seed(a.seed)

    def t(d, s=s, b=b, h=h, g=g):
        return torch.randn(s, b, h, d, generator=g, device=dev).to(torch.bfloat16).transpose(0, 1)

    q, k, v, do = t(L.D_QK), t(L.D_QK), t(L.D_V), t(L.D_V)
    qz = L.quant_inputs(q, k, v, recipe=L.recipe_of(a.fwd_tag))
    assert L.recipe_of(a.fwd_tag) == L.recipe_of(a.tag, "bwd"), (
        "forward and backward rungs of different recipes"
    )
    o, lse = L.fwd_mx(qz, tag=a.fwd_tag)
    bz = L.quant_bwd(qz, q, k, v, o, do)
    g1 = L.bwd_mx(qz, bz, lse, tag=a.tag)
    g2 = L.bwd_mx(qz, bz, lse, tag=a.tag)
    torch.cuda.synchronize()
    res = {}
    res["det"] = all(torch.equal(x, y) for x, y in zip(g1, g2))
    res["finite"] = all(bool(torch.isfinite(x.float()).all().item()) for x in g1)
    ge = L.emu_bwd(qz, bz, lse)
    for nm, x, e in zip(("dq", "dk", "dv"), g1, ge):
        res[nm + "_vs_emu"] = L.snr(x.float(), e)
    gr = L.ref_bwd(q, k, v, do)
    for nm, x, r in zip(("dq", "dk", "dv"), g1, gr):
        res[nm + "_vs_ref"] = L.snr(x.float(), r)
    ok = res["det"] and res["finite"] and all(res[n + "_vs_emu"] >= 35 for n in ("dq", "dk", "dv"))
    ok_all &= ok
    print(
        f"[{name}] b{b} s{s} h{h} {'PASS' if ok else 'FAIL'}  "
        + " ".join(f"{k_}={(round(v_, 2) if isinstance(v_, float) else v_)}" for k_, v_ in res.items()),
        flush=True,
    )
    if not ok:
        break
print("ALL PASS" if ok_all else "FAILURES")
sys.exit(0 if ok_all else 1)
