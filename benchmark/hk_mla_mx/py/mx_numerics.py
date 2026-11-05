#!/usr/bin/env python3
"""fp32 emulation of MXFP8 recipes for DeepSeek-V3 MLA attention (causal, d_qk 192 = 128 nope + 64 rope, d_v 128),
against the fp32 reference on the bf16 inputs. Prints SNR (dB) of o / lse / dq / dk / dv per variant and data kind.

    python3 mx_numerics.py [--b 1] [--s 4096] [--h 16] [--data randn,outlier,mla] [--variants ...]

Recipe pieces (all E4M3 data, E8M0 scales):
  q, k: MX along d, blocks of 32 (nope 4 + rope 2)                       (qk_mx)
  v: MX along kv, blocks of 32 tokens per channel                        (v_mx_col)
  P (forward PV): fixed scale 2^-7 after the online-softmax max (threshold tau nats), tile of 128 kv
  dO: MX along d_v (dP = dO V^T, with V MX along d_v) and MX along q (dV = P^T dO)
  dS: bf16 (A) or MX (B: along q for dK = dS^T Q with Q MX along q; along kv for dQ = dS K with K MX along kv)
Variants: bf16 (bf16 inputs, fp32 math, bf16 P / dS as the HK bf16 kernels), mx_fwd (forward only MX),
mx_a, mx_b, plus knobs: --tau, --p-scale-exp, --scale-rule floor|ceil, --rope plain (rope per-head pow2 scale, no MX).
"""

import argparse
import math

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--b", type=int, default=1)
ap.add_argument("--s", type=int, default=4096)
ap.add_argument("--h", type=int, default=16)
ap.add_argument("--data", default="randn,outlier,mla")
ap.add_argument("--variants", default="bf16,mx_fwd,mx_a,mx_b")
ap.add_argument("--tau", type=float, default=1.0, help="lazy rescale threshold in nats")
ap.add_argument("--p-scale-exp", type=int, default=-7)
ap.add_argument("--p-bwd-scale-exp", type=int, default=-8)
ap.add_argument("--scale-rule", default="ceil")
ap.add_argument("--rope", default="mx")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cuda")
a = ap.parse_args()

dev = torch.device(a.device)
torch.set_num_threads(min(64, torch.get_num_threads()))
SCALE = 0.1352337788608801
LOG2E = 1.4426950408889634
D_NOPE, D_ROPE, D_V = 128, 64, 128
TILE = 128
E4M3_MAX = 448.0


def snr(x, ref):
    x, ref = x.double(), ref.double()
    n = (x - ref).pow(2).sum().item()
    return 10 * math.log10(ref.pow(2).sum().item() / max(n, 1e-300))


def e4m3(x):
    return x.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float()


def mx_quant(x, dim, block=32, rule=a.scale_rule):
    """Fake-quantize x (fp32) to MXFP8 E4M3 along dim in blocks of `block`: returns the dequantized fp32 tensor."""
    x = x.float().movedim(dim, -1)
    sh = x.shape
    xb = x.reshape(*sh[:-1], sh[-1] // block, block)
    amax = xb.abs().amax(-1, keepdim=True).clamp(min=2.0**-126)
    if rule == "floor":  # OCP MX: 2^(floor(log2 amax) - emax), emax(e4m3) = 8; values above 448 saturate
        e = torch.floor(torch.log2(amax)) - 8
    else:  # no saturation: smallest power of two with amax / s <= 448
        e = torch.ceil(torch.log2(amax / E4M3_MAX))
    e = e.clamp(-127, 127)
    s = torch.exp2(e)
    q = e4m3(xb / s) * s
    return q.reshape(sh).movedim(-1, dim)


def pow2_quant(x, dims):
    """Fake-quantize with one power-of-two scale per slice over `dims` (no saturation)."""
    amax = x.abs().amax(dim=dims, keepdim=True).clamp(min=2.0**-126)
    s = torch.exp2(torch.ceil(torch.log2(amax / E4M3_MAX)))
    return e4m3(x / s) * s


def make_data(kind, g):
    b, s, h = a.b, a.s, a.h
    q = torch.randn(b, s, h, D_NOPE + D_ROPE, generator=g, device=dev)
    k = torch.randn(b, s, h, D_NOPE + D_ROPE, generator=g, device=dev)
    v = torch.randn(b, s, h, D_V, generator=g, device=dev)
    do = torch.randn(b, s, h, D_V, generator=g, device=dev)
    if kind in ("outlier", "mla"):
        # a few channel outliers (massive activations): 4 nope + 2 rope channels of q / k, 4 of v, x8..x32
        for t, chans in (
            (q, (3, 40, 77, 120, 130, 170)),
            (k, (5, 50, 90, 127, 140, 180)),
            (v, (7, 60, 99, 111)),
        ):
            for i, c in enumerate(chans):
                t[..., c] *= 8.0 * (1 + i % 4)
    if kind == "kbias":  # k with per-channel offsets (the bias-like channel means of real keys), q plain
        off = torch.randn(1, 1, h, D_NOPE + D_ROPE, generator=g, device=dev) * 3.0
        k = k + off
    if kind == "soft":  # weaker logits: q / k scaled to logit std ~1 nat
        q = q * 0.73
        k = k * 0.73
    if kind == "mla":
        k[..., D_NOPE:] = k[:, :, :1, D_NOPE:]  # the rope part of k is shared by every head (k_pe broadcast)
        q = q * 1.5  # larger logits: sharper attention rows
    bf = torch.bfloat16
    return q.to(bf), k.to(bf), v.to(bf), do.to(bf)


def ref_attn(q, k, v, do):
    """fp32 causal attention fwd + bwd via autograd (per head chunks to bound memory)."""
    qf, kf, vf = (t.float().detach().requires_grad_() for t in (q, k, v))
    b, s, h, _ = q.shape
    outs, lses = [], []
    dq = torch.zeros_like(qf)
    dk = torch.zeros_like(kf)
    dv = torch.zeros_like(vf)
    mask = torch.ones(s, s, device=dev, dtype=torch.bool).tril()
    for h0 in range(0, h, 4):
        sl = slice(h0, h0 + 4)
        qq, kk, vv = (t[:, :, sl].detach().clone().requires_grad_() for t in (qf, kf, vf))
        sc = torch.einsum("bqhd,bkhd->bhqk", qq, kk) * SCALE
        sc = sc.masked_fill(~mask, float("-inf"))
        lse = torch.logsumexp(sc, -1)
        p = torch.exp(sc - lse[..., None])
        o = torch.einsum("bhqk,bkhd->bqhd", p, vv)
        o.backward(do[:, :, sl].float())
        outs.append(o.detach())
        lses.append(lse.detach())
        dq[:, :, sl], dk[:, :, sl], dv[:, :, sl] = qq.grad, kk.grad, vv.grad
    return torch.cat(outs, 2), torch.cat(lses, 1), dq, dk, dv


def emu(q, k, v, do, variant):
    """Tile-wise emulation (128-kv tiles, online softmax with lazy rescale) of a recipe variant."""
    b, s, h, _ = q.shape
    ksm = variant.endswith("+ks")
    variant = variant.replace("+ks", "")
    qk_bf = variant.startswith("qkbf")
    if qk_bf:  # q / k bf16, PV and the backward as the rest of the name ("qkbf_" + mx_fwd / mx_a / mx_b)
        variant = variant[len("qkbf_") :]
    mx = variant != "bf16"
    qf, kf, vf, dof = q.float(), k.float(), v.float(), do.float()
    if ksm:  # subtract the per-(batch, head, channel) mean of k over tokens: softmax-invariant
        kf = kf - kf.mean(1, keepdim=True)
    if mx and qk_bf:
        qq, kq = qf, kf
        vq = mx_quant(vf, 1)
    elif mx:
        if a.rope == "mx":
            qq = torch.cat([mx_quant(qf[..., :D_NOPE], -1), mx_quant(qf[..., D_NOPE:], -1)], -1)
            kq = torch.cat([mx_quant(kf[..., :D_NOPE], -1), mx_quant(kf[..., D_NOPE:], -1)], -1)
        else:  # rope part: one power-of-two scale per (batch, head) for q, per tensor for k
            qq = torch.cat([mx_quant(qf[..., :D_NOPE], -1), pow2_quant(qf[..., D_NOPE:], (1, 3))], -1)
            kq = torch.cat([mx_quant(kf[..., :D_NOPE], -1), pow2_quant(kf[..., D_NOPE:], (0, 1, 2, 3))], -1)
        vq = mx_quant(vf, 1)  # along kv (dim 1 = s)
    else:
        qq, kq, vq = qf, kf, vf
    c2 = SCALE * LOG2E
    tau2 = a.tau * LOG2E
    pq_scale = 2.0**a.p_scale_exp
    # forward, per q tile of 128 rows x heads, kv tiles up to the diagonal
    o = torch.zeros(b, s, h, D_V, device=dev)
    lse = torch.zeros(b, h, s, device=dev)
    qi = torch.arange(s, device=dev)
    for q0 in range(0, s, TILE):
        Q = qq[:, q0 : q0 + TILE]  # b, 128, h, d
        m = torch.full((b, h, TILE), -1e30, device=dev)
        l = torch.zeros(b, h, TILE, device=dev)
        acc = torch.zeros(b, h, TILE, D_V, device=dev)
        for k0 in range(0, q0 + TILE, TILE):
            S = torch.einsum("bqhd,bkhd->bhqk", Q, kq[:, k0 : k0 + TILE])
            if k0 + TILE > q0:
                msk = qi[q0 : q0 + TILE, None] >= qi[None, k0 : k0 + TILE]
                S = S.masked_fill(~msk, float("-inf"))
            lm = S.amax(-1) * c2
            need = (lm - m) > tau2
            mn = torch.where(need, torch.maximum(m, lm), m)
            corr = torch.exp2(m - mn)
            p = torch.exp2(S * c2 - mn[..., None])
            l = l * corr + p.sum(-1)
            if mx:
                pv = e4m3(p / pq_scale) * pq_scale
            else:
                pv = p.to(torch.bfloat16).float()
            acc = acc * corr[..., None] + torch.einsum("bhqk,bkhd->bhqd", pv, vq[:, k0 : k0 + TILE])
            m = mn
        o[:, q0 : q0 + TILE] = (acc / l[..., None]).permute(0, 2, 1, 3)
        lse[:, :, q0 : q0 + TILE] = (m + torch.log2(l)) / LOG2E
    o = o.to(torch.bfloat16).float()
    if variant in ("bf16", "mx_fwd", "mx_fwdc"):
        do_r = dof
        do_c = dof
        vr = vf if variant == "bf16" else vf
        qb, kb = qf, kf
    if variant in ("mx_a", "mx_b"):
        do_r = mx_quant(dof, -1)  # along d_v for dP
        do_c = mx_quant(dof, 1)  # along q for dV
        vr = mx_quant(vf, -1)  # along d_v for dP
    if variant == "mx_b":
        qb = mx_quant(qf, 1)  # along q for dK
        kb = mx_quant(kf, 1)  # along kv for dQ
    elif variant == "mx_a":
        qb, kb = qf, kf
    if variant == "bf16":
        do_r = do_c = dof
        vr = vf
    # backward: delta from bf16 o and do; P recomputed from the same quantized q / k as the forward and lse
    delta = (o * dof).sum(-1).permute(0, 2, 1)  # b, h, s
    dq = torch.zeros(b, s, h, qf.shape[-1], device=dev)
    dk = torch.zeros_like(dq)
    dv = torch.zeros(b, s, h, D_V, device=dev)
    for q0 in range(0, s, TILE):
        qs, ks = (
            (qq, kq) if variant in ("mx_a", "mx_b", "mx_fwdc") else (qf, kf)
        )  # mx_fwd: bf16 bwd on mx lse
        S_all = torch.einsum("bqhd,bkhd->bhqk", qs[:, q0 : q0 + TILE], ks[:, : q0 + TILE])
        msk = qi[q0 : q0 + TILE, None] >= qi[None, : q0 + TILE]
        S_all = S_all.masked_fill(~msk, float("-inf"))
        P = torch.exp(S_all * SCALE - lse[:, :, q0 : q0 + TILE, None])  # fp32
        dP = torch.einsum("bqhd,bkhd->bhqk", do_r[:, q0 : q0 + TILE], vr[:, : q0 + TILE])
        dS = P * (dP - delta[:, :, q0 : q0 + TILE, None])
        if variant == "bf16":
            Pv, dSv = P.to(torch.bfloat16).float(), dS.to(torch.bfloat16).float()
            dSq = dSk = dSv
        elif variant in ("mx_fwd", "mx_fwdc"):
            Pv, dSv = P.to(torch.bfloat16).float(), dS.to(torch.bfloat16).float()
            dSq = dSk = dSv
        else:
            pb = 2.0**a.p_bwd_scale_exp  # P <= 1 in the backward (normalized by lse)
            Pv = e4m3(P / pb) * pb
            if variant == "mx_a":
                dSq = dSk = dS.to(torch.bfloat16).float()
            else:
                dSk = mx_quant(dS, 2) if True else dS  # dK = dS^T Q: reduction over q (dim 2 of bhqk)
                dSq = mx_quant(dS, 3)  # dQ = dS K: reduction over kv (dim 3)
        dv[:, : q0 + TILE] += torch.einsum("bhqk,bqhd->bkhd", Pv, do_c[:, q0 : q0 + TILE])
        dk[:, : q0 + TILE] += torch.einsum("bhqk,bqhd->bkhd", dSk, qb[:, q0 : q0 + TILE]) * SCALE
        dq[:, q0 : q0 + TILE] = torch.einsum("bhqk,bkhd->bqhd", dSq, kb[:, : q0 + TILE]) * SCALE
    return o, lse, dq, dk, dv


def s_snr(q, k):
    qf, kf = q.float()[:, :1024], k.float()[:, :1024]
    S = torch.einsum("bqhd,bkhd->bhqk", qf, kf)
    qq = torch.cat([mx_quant(qf[..., :D_NOPE], -1), mx_quant(qf[..., D_NOPE:], -1)], -1)
    kq = torch.cat([mx_quant(kf[..., :D_NOPE], -1), mx_quant(kf[..., D_NOPE:], -1)], -1)
    Sq = torch.einsum("bqhd,bkhd->bhqk", qq, kq)
    return snr(Sq, S), (S * SCALE).std().item(), ((Sq - S) * SCALE).std().item()


g = torch.Generator(device=dev).manual_seed(a.seed)
print(
    f"b {a.b} s {a.s} h {a.h}  tau {a.tau} nats  P scale 2^{a.p_scale_exp}  scale rule {a.scale_rule}  rope {a.rope}"
)
print(f"{'data':8s} {'variant':8s} {'o':>7s} {'lse':>7s} {'dq':>7s} {'dk':>7s} {'dv':>7s}")
for kind in a.data.split(","):
    q, k, v, do = make_data(kind, g)
    ss, lstd, estd = s_snr(q, k)
    print(f"{kind:8s} S (qk mx) snr {ss:.2f} dB, logit std {lstd:.2f} nats, logit error std {estd:.3f} nats")
    ref = ref_attn(q, k, v, do)
    for var in a.variants.split(","):
        out = emu(q, k, v, do, var)
        sn = [snr(x, r) for x, r in zip(out, ref)]
        print(f"{kind:8s} {var:8s} " + " ".join(f"{x:7.2f}" for x in sn), flush=True)
