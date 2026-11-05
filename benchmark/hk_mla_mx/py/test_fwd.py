#!/usr/bin/env python3
"""Correctness of the MXFP8 quantization kernels and the MXFP8 MLA forward (one GPU).

    PYTHONPATH=<Primus-Turbo-hk-mx> python3 test_fwd.py [--shapes toy,small,mid,prod] [--no-center] [--data randn]

Per shape: quantization outputs bit-exact against the torch twins; forward o / lse against the fp32 emulation of the
kernel's algorithm on the same quantized operands (kernel correctness, should be >= 40 dB) and against the fp32
reference on the bf16 inputs (recipe quality); determinism (two calls bitwise); finite.
"""

import argparse
import sys
import time

import torch

import mxlab as L

SHAPES = {
    "toy": dict(b=1, s=256, h=2),
    "small": dict(b=1, s=1024, h=4),
    "mid": dict(b=1, s=4096, h=16),
    "prod": dict(b=2, s=4096, h=128),
}

ap = argparse.ArgumentParser()
ap.add_argument("--shapes", default="toy,small,mid")
ap.add_argument("--no-center", action="store_true")
ap.add_argument("--data", default="randn")
ap.add_argument("--no-build", action="store_true")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tag", default="", help="k_mla_fwd_mx<tag>.hsaco (e.g. 1 = rung m1)")
a = ap.parse_args()

dev = torch.device("cuda")
if not a.no_build:
    t0 = time.time()
    L.build()
    L.build(["k_mla_fwd_mx1.hip"])
    print(f"built in {time.time() - t0:.1f} s", flush=True)


def make(b, s, h, g):
    # Megatron's layout: SBHD storage, [b, s, h, d] views
    def t(d):
        x = torch.randn(s, b, h, d, generator=g, device=dev)
        if a.data == "kbias" and d == L.D_QK:
            x = x + torch.randn(1, 1, h, d, generator=g, device=dev) * 3.0
        return x.to(torch.bfloat16).transpose(0, 1)

    return t(L.D_QK), t(L.D_QK), t(L.D_V)


ok_all = True
for name in a.shapes.split(","):
    sh = SHAPES[name]
    b, s, h = sh["b"], sh["s"], sh["h"]
    g = torch.Generator(device=dev).manual_seed(a.seed)
    q, k, v = make(b, s, h, g)
    center = not a.no_center
    qz = L.quant_inputs(q, k, v, center_k=center)
    torch.cuda.synchronize()
    res = {}
    # --- quantization twins
    kbar_ref = k.float().mean(1).permute(0, 1, 2) if center else None  # b h d
    if center:
        kbar_ref = k.float()[:, ::L.KBAR_STRIDE].mean(1)  # [b, h, 192]: the sampled mean
        res["kbar_maxrel"] = ((qz["kbar"] - kbar_ref).abs().max() / kbar_ref.abs().max()).item()
    for nm, x, kb in (("q", q, None), ("k", k, qz["kbar"] if center else None)):
        codes, exps = L.ref_quant_qk(x, kb)
        got = L.deinterleave_qk(qz[nm + "8"])
        res[nm + "_codes_eq"] = torch.equal(got, codes.view(torch.uint8))
        res[nm + "_codes_mism"] = int((got != codes.view(torch.uint8)).sum().item())
        res[nm + "_exps_eq"] = torch.equal(qz[nm + "s"][..., :6].to(torch.int32) - 127, exps)
        res[nm + "_pad_ok"] = bool((qz[nm + "s"][..., 6:] == 127).all().item())
    vt_codes, vt_exps = L.mx_quant_last(v.float().permute(0, 2, 3, 1))  # b h d s
    res["v_codes_eq"] = torch.equal(L.deinterleave_kv(qz["v8t"]), vt_codes.view(torch.uint8))
    vs_ref = vt_exps.reshape(b, h, L.D_V, s // 128, 4).permute(0, 1, 3, 2, 4)
    res["v_exps_eq"] = torch.equal(qz["vs"].to(torch.int32) - 127, vs_ref)
    if center:
        qk_ref = torch.einsum("bshd,bhd->bhs", q.float(), qz["kbar"])
        res["qk_maxrel"] = ((qz["qk"] - qk_ref).abs().max() / qk_ref.abs().max().clamp(min=1e-30)).item()
    # --- forward
    o1, lse1 = L.fwd_mx(qz, tag=a.tag)
    o2, lse2 = L.fwd_mx(qz, tag=a.tag)
    torch.cuda.synchronize()
    res["det"] = torch.equal(o1, o2) and torch.equal(lse1, lse2)
    res["finite"] = bool(torch.isfinite(o1.float()).all().item() and torch.isfinite(lse1).all().item())
    qf, kf = L.deq_qk(qz["q8"], qz["qs"]), L.deq_qk(qz["k8"], qz["ks"])
    vf = L.deq_vt(qz["v8t"], qz["vs"])
    oe, lsee = L.emu_fwd(qf, kf, vf)
    res["o_vs_emu"] = L.snr(o1.float(), oe)
    res["lse_vs_emu"] = L.snr(lse1, lsee)
    orf, lser = L.ref_fwd(q, k, v)
    lse_true = lse1 + L.SCALE * qz["qk"] if center else lse1
    res["o_vs_ref"] = L.snr(o1.float(), orf)
    res["lse_vs_ref"] = L.snr(lse_true, lser)
    ok = (res["q_codes_eq"] and res["k_codes_eq"] and res["v_codes_eq"] and res["q_exps_eq"] and res["k_exps_eq"]
          and res["v_exps_eq"] and res["det"] and res["finite"] and res["o_vs_emu"] >= 40 and res["lse_vs_emu"] >= 60)
    ok_all &= ok
    print(f"[{name}] b{b} s{s} h{h} {'PASS' if ok else 'FAIL'}  " +
          " ".join(f"{k_}={(round(v_, 2) if isinstance(v_, float) else v_)}" for k_, v_ in res.items()), flush=True)
    if not ok and name != a.shapes.split(",")[-1]:
        print("stopping: a smaller shape failed")
        break
print("ALL PASS" if ok_all else "FAILURES")
sys.exit(0 if ok_all else 1)
