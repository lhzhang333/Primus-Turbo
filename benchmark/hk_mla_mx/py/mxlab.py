"""MXFP8 MLA attention lab: build / load / launch the kernels in ../kern, bit-exact torch references of the
quantization, an fp32 emulation of the attention kernels' algorithm, and the fp32 reference.

    import mxlab as L
    L.build()                                   # kern/*.hip -> kern/build/*.hsaco (amdclang++)
    qz = L.quant_inputs(q, k, v)                # dict of the quantized operands (GPU kernels)
    o, lse = L.fwd_mx(qz, scale)                # the attention kernel
"""

import hashlib
import math
import os
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
KERN = HERE.parent / "kern"
BUILD = KERN / "build"
TURBO = HERE.parents[2]
if str(TURBO) not in sys.path:
    sys.path.insert(0, str(TURBO))
from primus_turbo.hipkittens.gfx1250_mla import hiprt  # noqa: E402

D_QK, D_NOPE, D_ROPE, D_V = 192, 128, 64, 128
SCALE = 0.1352337788608801
LOG2E = 1.4426950408889634
E4M3_MAX = 448.0
FLAGS = ["-x", "hip", "--offload-arch=gfx1250", "--cuda-device-only", "--no-gpu-bundle-output", "-O3", "-std=c++20"]

# kernel name -> (source, block, [(arg, kind)]); offsets follow natural alignment
KERNELS = {
    "k_mxq_kbar": ("k_mxq.hip", 768, [("K", "ptr"), ("kbar", "ptr"), ("S", "i32"), ("H", "i32"), ("sb", "i64"),
                                       ("ss", "i64"), ("sh", "i64")]),
    "k_mxq_qk": ("k_mxq.hip", 256, [("X", "ptr"), ("X8", "ptr"), ("XS", "ptr"), ("kbar", "ptr"), ("qk", "ptr"),
                                     ("B", "i32"), ("S", "i32"), ("H", "i32"), ("sb", "i64"), ("ss", "i64"),
                                     ("sh", "i64"), ("mode", "i32")]),
    "k_mxq_vt": ("k_mxq.hip", 256, [("V", "ptr"), ("V8T", "ptr"), ("VS", "ptr"), ("S", "i32"), ("H", "i32"),
                                     ("sb", "i64"), ("ss", "i64"), ("sh", "i64")]),
    "k_mla_fwd_mx": ("k_mla_fwd_mx.hip", 128, [("Q8", "ptr"), ("QS", "ptr"), ("K8", "ptr"), ("KS", "ptr"),
                                                ("V8T", "ptr"), ("VS", "ptr"), ("O", "ptr"), ("LSE", "ptr"),
                                                ("scale", "f32"), ("S", "i32"), ("H", "i32"), ("B", "i32"),
                                                ("o_sb", "i64"), ("o_ss", "i64")]),
}
_SIZE = {"ptr": 8, "i64": 8, "f64": 8, "i32": 4, "u32": 4, "f32": 4}


def _meta(name):
    src, block, args = KERNELS[name]
    off, out = 0, []
    for a, kind in args:
        sz = _SIZE[kind]
        off = (off + sz - 1) // sz * sz
        out.append({"name": a, "offset": off, "size": sz, "kind": kind})
        off += sz
    return {"block": block, "kernarg_size": off, "args": out}


def cxx():
    for c in (os.environ.get("MXLAB_CXX"), "amdclang++"):
        if c:
            from shutil import which

            p = which(c)
            if p:
                return p
    raise RuntimeError("amdclang++ not found")


def build(sources=None, defines=(), tag=""):
    """Compile kern sources (default: every one KERNELS uses) to BUILD/<stem><tag>.hsaco."""
    BUILD.mkdir(exist_ok=True)
    srcs = sorted({v[0] for v in KERNELS.values()}) if sources is None else sources
    out = {}
    for s in srcs:
        dst = BUILD / (Path(s).stem + tag + ".hsaco")
        cmd = [cxx(), *FLAGS, *[f"-D{d}" for d in defines], "-o", str(dst), str(KERN / s)]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=KERN)
        if r.returncode != 0:
            raise RuntimeError(f"build {s} failed:\n{r.stderr[-4000:]}")
        out[s] = dst
    return out


_MODS = {}


def kernel(name, tag=""):
    src = KERNELS[name][0]
    path = BUILD / (Path(src).stem + tag + ".hsaco")
    key = (str(path), name)
    k = _MODS.get(key)
    if k is None:
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        mod = _MODS.get(str(path))
        if mod is None:
            mod = _MODS[str(path)] = hiprt.Module(path, sha)
        k = _MODS[key] = hiprt.Kernel(mod, name, _meta(name))
    return k


def _launch(name, grid, values, tag=""):
    kernel(name, tag).call(grid, (KERNELS[name][1], 1, 1), values, torch.cuda.current_stream())


# ----------------------------------------------------------------------------------------- quantization (GPU kernels)


def quant_inputs(q, k, v, center_k=True, tag=""):
    """q, k [b, s, h, 192], v [b, s, h, 128] bf16 (d contiguous; any b / s / h strides) -> the MX operands."""
    b, s, h, _ = q.shape
    dev = q.device
    assert s % 256 == 0 and q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
    u8 = torch.uint8
    out = {"b": b, "s": s, "h": h}
    kbar = torch.zeros(b, h, D_QK, device=dev, dtype=torch.float32)
    if center_k:
        _launch("k_mxq_kbar", (b * h, 1, 1), [k, kbar, s, h, k.stride(0), k.stride(1), k.stride(2)], tag)
    qk = torch.zeros(b, h, s, device=dev, dtype=torch.float32)
    for name, x, mode in (("q", q, 2 if center_k else 0), ("k", k, 1 if center_k else 0)):
        x8 = torch.empty(b, h, s, D_QK, device=dev, dtype=u8)
        xs = torch.empty(b, h, s, 8, device=dev, dtype=u8)
        rows = b * h * s
        _launch("k_mxq_qk", ((rows + 31) // 32, 1, 1),
                [x, x8, xs, kbar, qk, b, s, h, x.stride(0), x.stride(1), x.stride(2), mode], tag)
        out[name + "8"], out[name + "s"] = x8, xs
    v8t = torch.empty(b, h, D_V, s, device=dev, dtype=u8)
    vs = torch.empty(b, h, s // 128, D_V, 4, device=dev, dtype=u8)
    _launch("k_mxq_vt", (s // 128, h, b), [v, v8t, vs, s, h, v.stride(0), v.stride(1), v.stride(2)], tag)
    out.update(v8t=v8t, vs=vs, kbar=kbar, qk=qk, center_k=center_k)
    return out


def fwd_mx(qz, scale=SCALE, out=None, tag=""):
    """The MX forward on quantize_inputs' operands -> o [b, s, h, 128] bf16 (BSHD view of SBHD storage when out is
    None: allocated [s, b, h, 128]), lse [b, h, s] (natural log of the centred logits)."""
    b, s, h = qz["b"], qz["s"], qz["h"]
    dev = qz["q8"].device
    if out is None:
        out = torch.empty(s, b, h, D_V, device=dev, dtype=torch.bfloat16).transpose(0, 1)
    lse = torch.empty(b, h, s, device=dev, dtype=torch.float32)
    _launch("k_mla_fwd_mx", (s // 256, h, b),
            [qz["q8"], qz["qs"], qz["k8"], qz["ks"], qz["v8t"], qz["vs"], out, lse, float(scale), s, h, b,
             out.stride(0), out.stride(1)], tag)
    return out, lse


# ---------------------------------------------------------------------------------- bit-exact torch references


def mx_exp(amax):
    """The kernels' E8M0 rule: smallest e with amax / 2^e <= 448; amax == 0 -> -127."""
    m, E = torch.frexp(amax.float())
    e = E.to(torch.int32) - 9 + (m > 0.875).to(torch.int32)
    e = torch.where(amax > 0, e, torch.full_like(e, -127))
    return e.clamp(-127, 127)


def e4m3(x):
    return x.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)


def mx_quant_last(x, block=32):
    """x [..., n] fp32 -> (codes e4m3 [..., n], exps int32 [..., n / block]) along the last dim."""
    sh = x.shape
    xb = x.float().reshape(*sh[:-1], sh[-1] // block, block)
    e = mx_exp(xb.abs().amax(-1))
    q = e4m3(xb / torch.ldexp(torch.ones_like(xb[..., :1]), e[..., None].float()))
    return q.reshape(sh), e


def perm_chunk(n):
    """Logical index -> stored byte of an MX-interleaved chunk of n (128 or 64) elements."""
    k = torch.arange(n)
    c, h, r = k // 16, (k % 16) // 8, k % 8
    return (n // 2) * h + 8 * c + r


def deinterleave_qk(x8):
    """X8 bytes [..., 192] (stored order) -> logical order (uint8)."""
    pn, pr = perm_chunk(128).to(x8.device), perm_chunk(64).to(x8.device)
    return torch.cat([x8[..., :128][..., pn], x8[..., 128:][..., pr]], -1)


def deinterleave_kv(v8t):
    """V8T bytes [..., s] (128-chunk interleave along s) -> logical s order."""
    s = v8t.shape[-1]
    p = perm_chunk(128).to(v8t.device)
    idx = (torch.arange(s // 128, device=v8t.device)[:, None] * 128 + p[None, :]).reshape(-1)
    return v8t[..., idx]


def deq_qk(x8, xs):
    """Dequantize X8 / XS -> fp32 [b, h, s, 192] (logical order)."""
    codes = deinterleave_qk(x8).view(torch.float8_e4m3fn).float()
    e = xs[..., :6].to(torch.int32) - 127
    sc = torch.ldexp(torch.ones_like(e, dtype=torch.float32), e)
    return (codes.reshape(*codes.shape[:-1], 6, 32) * sc[..., None]).reshape(codes.shape)


def deq_vt(v8t, vs):
    """Dequantize V8T / VS -> fp32 V [b, h, s, 128]."""
    b, h, d, s = v8t.shape
    codes = deinterleave_kv(v8t).view(torch.float8_e4m3fn).float()  # b h d s
    e = vs.to(torch.int32) - 127  # b h (s/128) d 4
    sc = torch.ldexp(torch.ones_like(e, dtype=torch.float32), e).permute(0, 1, 3, 2, 4).reshape(b, h, d, s // 32)
    return (codes.reshape(b, h, d, s // 32, 32) * sc[..., None]).reshape(b, h, d, s).transpose(2, 3)


def ref_quant_qk(x, kbar=None):
    """Torch twin of k_mxq_qk (logical order): x [b, s, h, 192] bf16 -> codes [b, h, s, 192], exps [b, h, s, 6]."""
    xf = x.float().permute(0, 2, 1, 3)
    if kbar is not None:
        xf = xf - kbar[:, :, None, :]
    return mx_quant_last(xf)


# ------------------------------------------------------------------------------------------- attention references


def ref_fwd(q, k, v, scale=SCALE, causal=True):
    """fp32 attention on bf16 (or fp32) inputs [b, s, h, d]: o [b, s, h, 128], lse [b, h, s]."""
    b, s, h, _ = q.shape
    outs, lses = [], []
    qi = torch.arange(s, device=q.device)
    for h0 in range(0, h, 8):
        sl = slice(h0, h0 + 8)
        sc = torch.einsum("bqhd,bkhd->bhqk", q[:, :, sl].float(), k[:, :, sl].float()) * scale
        if causal:
            sc = sc.masked_fill(qi[:, None] < qi[None, :], float("-inf"))
        lse = torch.logsumexp(sc, -1)
        outs.append(torch.einsum("bhqk,bkhd->bqhd", torch.exp(sc - lse[..., None]), v[:, :, sl].float()))
        lses.append(lse)
    return torch.cat(outs, 2), torch.cat(lses, 1)


def emu_fwd(qf, kf, vf, scale=SCALE, tau=1.0, p_exp=-7, bm=256, nb=128):
    """fp32 emulation of k_mla_fwd_mx's algorithm on dequantized operands qf, kf [b, h, s, 192], vf [b, h, s, 128]:
    per q16 tile online softmax over 128-kv tiles with the lazy rescale, P = e4m3(p / 2^p_exp) 2^p_exp."""
    b, h, s, _ = qf.shape
    dev = qf.device
    c2, tau2, ps = scale * LOG2E, tau * LOG2E, 2.0 ** p_exp
    o = torch.zeros(b, h, s, D_V, device=dev)
    lse = torch.zeros(b, h, s, device=dev)
    qi = torch.arange(s, device=dev)
    for q0 in range(0, s, bm):
        Q = qf[:, :, q0:q0 + bm]  # b h 256 d
        m = torch.full((b, h, bm), -1e30, device=dev)
        l = torch.zeros(b, h, bm, device=dev)
        acc = torch.zeros(b, h, bm, D_V, device=dev)
        for k0 in range(0, q0 + bm, nb):
            S = torch.einsum("bhqd,bhkd->bhqk", Q, kf[:, :, k0:k0 + nb])
            msk = qi[q0:q0 + bm, None] >= qi[None, k0:k0 + nb]
            S = S.masked_fill(~msk, float("-inf"))
            lm = S.amax(-1)
            need = (lm * c2 - m) > tau2  # b h 256
            resc = need.reshape(b, h, bm // 16, 16).any(-1, keepdim=True).expand(-1, -1, -1, 16).reshape(b, h, bm)
            mn = torch.where(resc, torch.maximum(m, lm * c2), m)
            corr = torch.exp2(m - mn)
            p = torch.exp2(S * c2 - mn[..., None])
            l = corr * l + p.sum(-1)
            pq = e4m3(p / ps).float() * ps
            acc = acc * torch.where(resc, corr, torch.ones_like(corr))[..., None] + pq @ vf[:, :, k0:k0 + nb]
            m = mn
        o[:, :, q0:q0 + bm] = acc / l[..., None]
        lse[:, :, q0:q0 + bm] = (m + torch.log2(l)) / LOG2E
    return o.permute(0, 2, 1, 3), lse


def snr(x, ref):
    x, ref = x.double(), ref.double()
    n = (x - ref).pow(2).sum().item()
    return 10 * math.log10(ref.pow(2).sum().item() / max(n, 1e-300))
