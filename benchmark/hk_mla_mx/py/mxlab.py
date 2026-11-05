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
_ROWS_ARGS = [("X", "ptr"), ("X8", "ptr"), ("XS", "ptr"), ("kbar", "ptr"), ("Y", "ptr"), ("aux", "ptr"),
              ("XD", "ptr"), ("B", "i32"), ("S", "i32"), ("H", "i32"), ("sb", "i64"), ("ss", "i64"), ("sh", "i64"),
              ("yb", "i64"), ("ys", "i64"), ("yh", "i64"), ("mode", "i32")]
_COLS_ARGS = [("X", "ptr"), ("X8T", "ptr"), ("XTS", "ptr"), ("kbar", "ptr"), ("S", "i32"), ("H", "i32"),
              ("sb", "i64"), ("ss", "i64"), ("sh", "i64"), ("mode", "i32")]
KERNELS = {
    "k_mxq_kbar": ("k_mxq.hip", 768, [("K", "ptr"), ("kbar", "ptr"), ("S", "i32"), ("H", "i32"), ("sb", "i64"),
                                       ("ss", "i64"), ("sh", "i64"), ("stride", "i32")]),
    "k_mxq_rows192": ("k_mxq.hip", 256, _ROWS_ARGS),
    "k_mxq_rows128": ("k_mxq.hip", 256, _ROWS_ARGS),
    "k_mxq_cols128": ("k_mxq.hip", 256, _COLS_ARGS),
    "k_mxq_cols192": ("k_mxq.hip", 256, _COLS_ARGS),
    "k_mla_fwd_mx": ("k_mla_fwd_mx.hip", 128, [("Q8", "ptr"), ("QS", "ptr"), ("K8", "ptr"), ("KS", "ptr"),
                                                ("V8T", "ptr"), ("VS", "ptr"), ("O", "ptr"), ("LSE", "ptr"),
                                                ("scale", "f32"), ("S", "i32"), ("H", "i32"), ("B", "i32"),
                                                ("o_sb", "i64"), ("o_ss", "i64")]),
    "k_dkdv_mx": ("k_mla_bwd_mx.hip", 128, [("Q8", "ptr"), ("QS", "ptr"), ("K8", "ptr"), ("KS", "ptr"), ("V8", "ptr"),
                                             ("VSR", "ptr"), ("DO8", "ptr"), ("DOS", "ptr"), ("DO8T", "ptr"),
                                             ("DOTS", "ptr"), ("Q8T", "ptr"), ("QTS", "ptr"), ("LSE", "ptr"),
                                             ("DELTA", "ptr"), ("DK", "ptr"), ("DV", "ptr"), ("scale", "f32"),
                                             ("S", "i32"), ("H", "i32"), ("B", "i32"), ("dk_sb", "i64"),
                                             ("dk_ss", "i64"), ("dv_sb", "i64"), ("dv_ss", "i64")]),
    "k_dq_mx": ("k_mla_bwd_mx.hip", 128, [("Q8", "ptr"), ("QS", "ptr"), ("K8", "ptr"), ("KS", "ptr"), ("V8", "ptr"),
                                           ("VSR", "ptr"), ("DO8", "ptr"), ("DOS", "ptr"), ("K8T", "ptr"),
                                           ("KTS", "ptr"), ("LSE", "ptr"), ("DELTA", "ptr"), ("DQ", "ptr"),
                                           ("scale", "f32"), ("S", "i32"), ("H", "i32"), ("B", "i32"),
                                           ("dq_sb", "i64"), ("dq_ss", "i64")]),
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


def _dummy(dev):
    t = _DUMMY.get(dev)
    if t is None:
        t = _DUMMY[dev] = torch.zeros(64, device=dev, dtype=torch.float32)
    return t


_DUMMY = {}


def quant_rows(x, d, mode=0, kbar=None, y=None, want_deq=False, tag=""):
    """k_mxq_rows{192,128}: x [b, s, h, d] -> (x8 [b, h, s, d], xs [b, h, s, 8 | 4], aux [b, h, s] | None,
    xd [b, s, h, d] bf16 | None)."""
    b, s, h, _ = x.shape
    dev = x.device
    rs = 8 if d == 192 else 4
    x8 = torch.empty(b, h, s, d, device=dev, dtype=torch.uint8)
    xs = torch.empty(b, h, s, rs, device=dev, dtype=torch.uint8)
    aux = torch.empty(b, h, s, device=dev, dtype=torch.float32) if mode & 6 else _dummy(dev)
    xd = torch.empty(b, s, h, d, device=dev, dtype=torch.bfloat16) if want_deq else _dummy(dev)
    yy = y if y is not None else x
    m = mode | (8 if want_deq else 0)
    rows_per_block = 8 * 8 * (4 if d == 192 else 8)  # 8 waves x 8 row groups x rows per group
    assert s % (rows_per_block // 8) == 0
    _launch(f"k_mxq_rows{d}", ((b * h * s + rows_per_block - 1) // rows_per_block, 1, 1),
            [x, x8, xs, kbar if kbar is not None else _dummy(dev), yy, aux, xd, b, s, h, x.stride(0), x.stride(1),
             x.stride(2), yy.stride(0), yy.stride(1), yy.stride(2), m], tag)
    return x8, xs, (aux if mode & 6 else None), (xd if want_deq else None)


def quant_cols(x, d, kbar=None, tag=""):
    """k_mxq_cols{128,192}: x [b, s, h, d] -> (x8t [b, h, d, s], xts [b, h, s / 128, d, 4])."""
    b, s, h, _ = x.shape
    dev = x.device
    x8t = torch.empty(b, h, d, s, device=dev, dtype=torch.uint8)
    xts = torch.empty(b, h, s // 128, d, 4, device=dev, dtype=torch.uint8)
    _launch(f"k_mxq_cols{d}", (s // 128, h, b),
            [x, x8t, xts, kbar if kbar is not None else _dummy(dev), s, h, x.stride(0), x.stride(1), x.stride(2),
             1 if kbar is not None else 0], tag)
    return x8t, xts


KBAR_STRIDE = 16  # k_mxq_kbar samples every 16th row (any constant shift of k is softmax-invariant)


def kbar_of(k, tag="", stride=KBAR_STRIDE):
    b, s, h, _ = k.shape
    kbar = torch.empty(b, h, D_QK, device=k.device, dtype=torch.float32)
    _launch("k_mxq_kbar", (b * h, 1, 1), [k, kbar, s, h, k.stride(0), k.stride(1), k.stride(2), stride], tag)
    return kbar


def quant_inputs(q, k, v, center_k=True, want_deq=False, tag=""):
    """q, k [b, s, h, 192], v [b, s, h, 128] bf16 (d contiguous; any b / s / h strides) -> the forward's MX operands
    (+ the dequantized q / k as bf16 [b, s, h, 192] when want_deq: the bf16 backward's consistent inputs)."""
    b, s, h, _ = q.shape
    assert s % 256 == 0 and q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
    out = {"b": b, "s": s, "h": h, "center_k": center_k}
    kbar = kbar_of(k, tag) if center_k else None
    q8, qs, qk, qd = quant_rows(q, D_QK, mode=2 if center_k else 0, kbar=kbar, want_deq=want_deq, tag=tag)
    k8, ks, _, kd = quant_rows(k, D_QK, mode=1 if center_k else 0, kbar=kbar, want_deq=want_deq, tag=tag)
    v8t, vs = quant_cols(v, D_V, tag=tag)
    out.update(q8=q8, qs=qs, k8=k8, ks=ks, v8t=v8t, vs=vs, kbar=kbar, qd=qd, kd=kd,
               qk=qk if qk is not None else torch.zeros(b, h, s, device=q.device))
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


# ------------------------------------------------------------------------------------------------ backward (GPU)


def quant_bwd(qz, q, k, v, o, do, tag=""):
    """The backward's extra MX operands: V and dO rows (+ delta = rowsum(dO * O)), dO cols, Q cols, K cols (centred by
    the forward's kbar). q, k, v, o, do [b, s, h, d] bf16."""
    v8, vsr, _, _ = quant_rows(v, D_V, tag=tag)
    do8, dos, delta, _ = quant_rows(do, D_V, mode=4, y=o, tag=tag)
    do8t, dots = quant_cols(do, D_V, tag=tag)
    q8t, qts = quant_cols(q, D_QK, tag=tag)
    k8t, kts = quant_cols(k, D_QK, kbar=qz["kbar"], tag=tag)
    return dict(v8=v8, vsr=vsr, do8=do8, dos=dos, delta=delta, do8t=do8t, dots=dots, q8t=q8t, qts=qts, k8t=k8t,
                kts=kts)


def bwd_mx(qz, bz, lse, scale=SCALE, tag=""):
    """dq, dk [b, s, h, 192], dv [b, s, h, 128] bf16 (BSHD views of SBHD storage) from the MX operands."""
    b, s, h = qz["b"], qz["s"], qz["h"]
    dev = qz["q8"].device
    dq = torch.empty(s, b, h, D_QK, device=dev, dtype=torch.bfloat16).transpose(0, 1)
    dk = torch.empty(s, b, h, D_QK, device=dev, dtype=torch.bfloat16).transpose(0, 1)
    dv = torch.empty(s, b, h, D_V, device=dev, dtype=torch.bfloat16).transpose(0, 1)
    _launch("k_dkdv_mx", (s // 128, h, b),
            [qz["q8"], qz["qs"], qz["k8"], qz["ks"], bz["v8"], bz["vsr"], bz["do8"], bz["dos"], bz["do8t"],
             bz["dots"], bz["q8t"], bz["qts"], lse, bz["delta"], dk, dv, float(scale), s, h, b, dk.stride(0),
             dk.stride(1), dv.stride(0), dv.stride(1)], tag)
    _launch("k_dq_mx", (s // 256, h, b),
            [qz["q8"], qz["qs"], qz["k8"], qz["ks"], bz["v8"], bz["vsr"], bz["do8"], bz["dos"], bz["k8t"], bz["kts"],
             lse, bz["delta"], dq, float(scale), s, h, b, dq.stride(0), dq.stride(1)], tag)
    return dq, dk, dv


# ------------------------------------------------------------------------------------- backward references


def deq_rows(x8, xs, d):
    """Rows dequantization for d = 128 (one interleaved chunk) -> fp32 [b, h, s, 128]."""
    p = perm_chunk(128).to(x8.device)
    codes = x8[..., p].view(torch.float8_e4m3fn).float()
    e = xs.to(torch.int32) - 127
    sc = torch.ldexp(torch.ones_like(e, dtype=torch.float32), e)
    return (codes.reshape(*codes.shape[:-1], d // 32, 32) * sc[..., None]).reshape(codes.shape)


def deq_cols(x8t, xts):
    """Cols dequantization: X8T [b, h, d, s] / XTS [b, h, s / 128, d, 4] -> fp32 [b, h, s, d]."""
    b, h, d, s = x8t.shape
    codes = deinterleave_kv(x8t).view(torch.float8_e4m3fn).float()  # b h d s
    e = xts.to(torch.int32) - 127
    sc = torch.ldexp(torch.ones_like(e, dtype=torch.float32), e).permute(0, 1, 3, 2, 4).reshape(b, h, d, s // 32)
    return (codes.reshape(b, h, d, s // 32, 32) * sc[..., None]).reshape(b, h, d, s).transpose(2, 3)


def mxq(x, dim):
    """Fake MX quantization of fp32 x along dim (blocks of 32, the kernels' scale rule)."""
    xm = x.movedim(dim, -1)
    codes, e = mx_quant_last(xm)
    sc = torch.ldexp(torch.ones_like(e, dtype=torch.float32), e)
    deq = (codes.float().reshape(*codes.shape[:-1], codes.shape[-1] // 32, 32) * sc[..., None]).reshape(codes.shape)
    return deq.movedim(-1, dim)


def emu_bwd(qz, bz, lse, scale=SCALE, chunk=256):
    """fp32 emulation of k_dkdv_mx / k_dq_mx on the quantized operands: dq, dk [b, s, h, 192], dv [b, s, h, 128]."""
    qf = deq_qk(qz["q8"], qz["qs"])
    kf = deq_qk(qz["k8"], qz["ks"])
    vr = deq_rows(bz["v8"], bz["vsr"], D_V)
    dor = deq_rows(bz["do8"], bz["dos"], D_V)
    doc = deq_cols(bz["do8t"], bz["dots"])
    qt = deq_cols(bz["q8t"], bz["qts"])
    kt = deq_cols(bz["k8t"], bz["kts"])
    b, h, s, _ = qf.shape
    dev = qf.device
    delta = bz["delta"]
    qi = torch.arange(s, device=dev)
    dq = torch.zeros(b, h, s, D_QK, device=dev)
    dk = torch.zeros(b, h, s, D_QK, device=dev)
    dv = torch.zeros(b, h, s, D_V, device=dev)
    for h0 in range(0, h, 8):
        sl = slice(h0, h0 + 8)
        for q0 in range(0, s, chunk):
            qs_ = slice(q0, q0 + chunk)
            S = torch.einsum("bhqd,bhkd->bhqk", qf[:, sl, qs_], kf[:, sl, : q0 + chunk])
            P = torch.exp2(S * (scale * LOG2E) - lse[:, sl, qs_, None] * LOG2E)
            P = P.masked_fill(qi[qs_, None] < qi[None, : q0 + chunk], 0.0)
            dP = torch.einsum("bhqd,bhkd->bhqk", dor[:, sl, qs_], vr[:, sl, : q0 + chunk])
            dS = P * (dP - delta[:, sl, qs_, None])
            Pq = e4m3(P * 256.0).float() / 256.0
            dv[:, sl, : q0 + chunk] += torch.einsum("bhqk,bhqd->bhkd", Pq, doc[:, sl, qs_])
            dk[:, sl, : q0 + chunk] += torch.einsum("bhqk,bhqd->bhkd", mxq(dS, 2), qt[:, sl, qs_]) * scale
            dq[:, sl, qs_] = torch.einsum("bhqk,bhkd->bhqd", mxq(dS, 3), kt[:, sl, : q0 + chunk]) * scale
    t = lambda x: x.permute(0, 2, 1, 3)  # noqa: E731
    return t(dq), t(dk), t(dv)


def ref_bwd(q, k, v, do, scale=SCALE):
    """fp32 autograd attention backward on the (bf16) inputs: dq, dk, dv [b, s, h, d]."""
    b, s, h, _ = q.shape
    dq = torch.zeros(b, s, h, D_QK, device=q.device)
    dk = torch.zeros_like(dq)
    dv = torch.zeros(b, s, h, D_V, device=q.device)
    mask = torch.ones(s, s, device=q.device, dtype=torch.bool).tril()
    for h0 in range(0, h, 8):
        sl = slice(h0, h0 + 8)
        qq, kk, vv = (t[:, :, sl].float().detach().clone().requires_grad_() for t in (q, k, v))
        sc = torch.einsum("bqhd,bkhd->bhqk", qq, kk) * scale
        sc = sc.masked_fill(~mask, float("-inf"))
        p = torch.softmax(sc, -1)
        o = torch.einsum("bhqk,bkhd->bqhd", p, vv)
        o.backward(do[:, :, sl].float())
        dq[:, :, sl], dk[:, :, sl], dv[:, :, sl] = qq.grad, kk.grad, vv.grad
    return dq, dk, dv


# ------------------------------------------------------------- torch twins of the quantization kernels (any device)


def _interleave_last(codes_u8, n):
    """Logical-order bytes [..., n] -> MX-interleaved stored order (n = 128 or 64 per chunk)."""
    out = torch.empty_like(codes_u8)
    out[..., perm_chunk(n).to(codes_u8.device)] = codes_u8
    return out


def twin_rows(x, d, kbar=None):
    """k_mxq_rows{192,128} twin: x [b, s, h, d] -> x8 [b, h, s, d] (stored order), xs [b, h, s, 8 | 4]."""
    xf = x.float().permute(0, 2, 1, 3)
    if kbar is not None:
        xf = xf - kbar[:, :, None, :]
    codes, e = mx_quant_last(xf)
    cu = codes.view(torch.uint8)
    if d == 192:
        x8 = torch.cat([_interleave_last(cu[..., :128], 128), _interleave_last(cu[..., 128:], 64)], -1)
        xs = torch.cat([(e + 127).to(torch.uint8), torch.full_like(e[..., :2], 127, dtype=torch.uint8)], -1)
    else:
        x8 = _interleave_last(cu, 128)
        xs = (e + 127).to(torch.uint8)
    return x8, xs


def twin_cols(x, d, kbar=None):
    """k_mxq_cols{128,192} twin: x [b, s, h, d] -> x8t [b, h, d, s] (s interleaved per 128), xts [b, h, s/128, d, 4]."""
    b, s, h, _ = x.shape
    xf = x.float().permute(0, 2, 3, 1)  # b h d s
    if kbar is not None:
        xf = xf - kbar[:, :, :, None]
    codes, e = mx_quant_last(xf)  # e: b h d s/32
    cu = codes.view(torch.uint8).reshape(b, h, d, s // 128, 128)
    x8t = _interleave_last(cu, 128).reshape(b, h, d, s)
    xts = (e + 127).to(torch.uint8).reshape(b, h, d, s // 128, 4).permute(0, 1, 3, 2, 4).contiguous()
    return x8t, xts


def twin_inputs(q, k, v, kbar_stride=KBAR_STRIDE):
    b, s, h, _ = q.shape
    kbar = k.float()[:, ::kbar_stride].mean(1)  # [b, h, 192]
    q8, qs = twin_rows(q, 192)
    k8, ks = twin_rows(k, 192, kbar)
    v8t, vs = twin_cols(v, 128)
    qk = torch.einsum("bshd,bhd->bhs", q.float(), kbar)
    return dict(b=b, s=s, h=h, center_k=True, q8=q8, qs=qs, k8=k8, ks=ks, v8t=v8t, vs=vs, kbar=kbar, qk=qk)


def twin_bwd(qz, q, k, v, o, do):
    v8, vsr = twin_rows(v, 128)
    do8, dos = twin_rows(do, 128)
    delta = (do.float() * o.float()).sum(-1).permute(0, 2, 1).contiguous()
    do8t, dots = twin_cols(do, 128)
    q8t, qts = twin_cols(q, 192)
    k8t, kts = twin_cols(k, 192, qz["kbar"])
    return dict(v8=v8, vsr=vsr, do8=do8, dos=dos, delta=delta, do8t=do8t, dots=dots, q8t=q8t, qts=qts, k8t=k8t,
                kts=kts)
