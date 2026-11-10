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
LN2 = 0.6931471805599453
E4M3_MAX = 448.0
FLAGS = [
    "-x",
    "hip",
    "--offload-arch=gfx1250",
    "--cuda-device-only",
    "--no-gpu-bundle-output",
    "-O3",
    "-std=c++20",
]
EXPERT = [
    "-mllvm",
    "-amdgpu-expert-scheduling-mode",
]  # the compiler places the dependency waits (bf16 HK: required)
# + register classes before globalness in the greedy allocator: DS addresses and WMMA scale operands live in v0..v255
# (no VGPR-MSB remap), and without it big kernels spill them
EXPERT_P = EXPERT + ["-mllvm", "-greedy-regclass-priority-trumps-globalness"]
# xi = xp + the GCN max-ILP machine scheduler (the package's flags since 11-06: dq -3.7%, forward -2%)
EXPERT_I = EXPERT_P + ["-mllvm", "-misched=gcn-max-ilp"]


def recipe_of(tag, kind="fwd"):
    """The operand recipe a build tag expects: "m4" (Q8 pre-scaled to log2 units, scale layout R2) for forward rungs m4 /
    m5 (tags 4*, 5*, 6*) and backward rungs b1 / b2 (tags 1*, 2*), else "r1"."""
    return "m4" if tag[:1] in (("4", "5", "6", "7") if kind == "fwd" else ("1", "2", "3", "4", "5", "6")) else "r1"


def split_tag(tag):
    """A build tag = <rung digits><flags>: flags "" (plain), "x" (EXPERT), "xp" (EXPERT_P)."""
    i = len(tag) - len(tag.lstrip("0123456789"))
    return tag[:i], tag[i:]


def build_tag(kind, tag, defines=()):
    """Build kern/k_mla_<kind>_mx<rung>.hip with the tag's flags into BUILD/k_mla_<kind>_mx<tag>.hsaco (kind: fwd /
    bwd), the path kernel(name, tag) loads."""
    rung, fl = split_tag(tag)
    flags = {"": (), "x": EXPERT, "xp": EXPERT_P, "xi": EXPERT_I}[fl]
    return build([f"k_mla_{kind}_mx{rung}.hip"], defines=defines, tag=fl, flags=flags)


# kernel name -> (source, block, [(arg, kind)]); offsets follow natural alignment
_ROWS_ARGS = [
    ("X", "ptr"),
    ("X8", "ptr"),
    ("XS", "ptr"),
    ("kbar", "ptr"),
    ("Y", "ptr"),
    ("aux", "ptr"),
    ("XD", "ptr"),
    ("B", "i32"),
    ("S", "i32"),
    ("H", "i32"),
    ("sb", "i64"),
    ("ss", "i64"),
    ("sh", "i64"),
    ("yb", "i64"),
    ("ys", "i64"),
    ("yh", "i64"),
    ("mode", "i32"),
    ("pre", "f32"),
]
_COLS_ARGS = [
    ("X", "ptr"),
    ("X8T", "ptr"),
    ("XTS", "ptr"),
    ("kbar", "ptr"),
    ("S", "i32"),
    ("H", "i32"),
    ("sb", "i64"),
    ("ss", "i64"),
    ("sh", "i64"),
    ("mode", "i32"),
]
_RC_ARGS = [("X", "ptr"), ("X8", "ptr"), ("XS", "ptr"), ("X8T", "ptr"), ("XTS", "ptr"), ("kbar", "ptr"), ("Y", "ptr"),
            ("aux", "ptr"), ("S", "i32"), ("H", "i32"), ("sb", "i64"), ("ss", "i64"), ("sh", "i64"), ("yb", "i64"),
            ("ys", "i64"), ("yh", "i64"), ("mode", "i32"), ("pre", "f32")]  # fmt: skip
_RCP_ARGS = _RC_ARGS[:8] + [("B", "i32")] + _RC_ARGS[8:]  # persistent: B before S
_RCTR_ARGS = [("X", "ptr"), ("X8", "ptr"), ("XS", "ptr"), ("X8T", "ptr"), ("XTS", "ptr"), ("kbar", "ptr"), ("aux", "ptr"),
              ("S", "i32"), ("H", "i32"), ("sb", "i64"), ("ss", "i64"), ("sh", "i64"), ("PE", "ptr"), ("pb", "i64"),
              ("ps", "i64"), ("ph", "i64"), ("COS", "ptr"), ("SIN", "ptr"), ("mode", "i32"), ("pre", "f32")]  # fmt: skip
_KBR_ARGS = [("X", "ptr"), ("kbar", "ptr"), ("S", "i32"), ("H", "i32"), ("sb", "i64"), ("ss", "i64"), ("sh", "i64"),
             ("PE", "ptr"), ("pb", "i64"), ("ps", "i64"), ("ph", "i64"), ("COS", "ptr"), ("SIN", "ptr"),
             ("stride", "i32")]  # fmt: skip
KERNELS = {
    "k_mxq_kbar": (
        "k_mxq.hip",
        768,
        [
            ("K", "ptr"),
            ("kbar", "ptr"),
            ("S", "i32"),
            ("H", "i32"),
            ("sb", "i64"),
            ("ss", "i64"),
            ("sh", "i64"),
            ("stride", "i32"),
        ],
    ),
    "k_mxq_rows192": ("k_mxq.hip", 256, _ROWS_ARGS),
    "k_mxq_rows128": ("k_mxq.hip", 256, _ROWS_ARGS),
    "k_mxq_cols128": ("k_mxq.hip", 256, _COLS_ARGS),
    "k_mxq_cols192": ("k_mxq.hip", 256, _COLS_ARGS),
    "k_mxq_rc192": ("k_mxq.hip", 256, _RC_ARGS),
    "k_mxq_rc128": ("k_mxq.hip", 256, _RC_ARGS),
    "k_mxq_rct192": ("k_mxq.hip", 256, _RC_ARGS),
    "k_mxq_rct128": ("k_mxq.hip", 256, _RC_ARGS),
    "k_mxq_rctr192": ("k_mxq.hip", 256, _RCTR_ARGS),
    "k_mxq_kbar_rope": ("k_mxq.hip", 768, _KBR_ARGS),
    "k_mxq_rcp192": ("k_mxq.hip", 512, _RCP_ARGS),
    "k_mxq_rcp128": ("k_mxq.hip", 512, _RCP_ARGS),
    "k_mla_fwd_mx": (
        "k_mla_fwd_mx.hip",
        128,
        [
            ("Q8", "ptr"),
            ("QS", "ptr"),
            ("K8", "ptr"),
            ("KS", "ptr"),
            ("V8T", "ptr"),
            ("VS", "ptr"),
            ("O", "ptr"),
            ("LSE", "ptr"),
            ("scale", "f32"),
            ("S", "i32"),
            ("H", "i32"),
            ("B", "i32"),
            ("o_sb", "i64"),
            ("o_ss", "i64"),
        ],
    ),
    "k_dkdv_mx": (
        "k_mla_bwd_mx.hip",
        128,
        [
            ("Q8", "ptr"),
            ("QS", "ptr"),
            ("K8", "ptr"),
            ("KS", "ptr"),
            ("V8", "ptr"),
            ("VSR", "ptr"),
            ("DO8", "ptr"),
            ("DOS", "ptr"),
            ("DO8T", "ptr"),
            ("DOTS", "ptr"),
            ("Q8T", "ptr"),
            ("QTS", "ptr"),
            ("LSE", "ptr"),
            ("DELTA", "ptr"),
            ("DK", "ptr"),
            ("DV", "ptr"),
            ("scale", "f32"),
            ("S", "i32"),
            ("H", "i32"),
            ("B", "i32"),
            ("dk_sb", "i64"),
            ("dk_ss", "i64"),
            ("dv_sb", "i64"),
            ("dv_ss", "i64"),
        ],
    ),
    "k_dkdv_mxr": ("k_mla_bwd_mx.hip", 128, None),  # = k_dkdv_mx's args + DKP, dkp_sb, dkp_ss (filled below)
    "k_rope_bwd_q": ("k_mxq.hip", 256, [("G", "ptr"), ("B", "i32"), ("S", "i32"), ("H", "i32"), ("gb", "i64"),
                                         ("gs", "i64"), ("gh", "i64"), ("COS", "ptr"), ("SIN", "ptr")]),
    "k_rope_bwd_kpe": ("k_mxq.hip", 256, [("GH", "ptr"), ("GP", "ptr"), ("B", "i32"), ("S", "i32"), ("H", "i32"),
                                           ("hb", "i64"), ("hs", "i64"), ("hh", "i64"), ("pb", "i64"), ("ps", "i64"),
                                           ("COS", "ptr"), ("SIN", "ptr")]),
    "k_dq_mx": (
        "k_mla_bwd_mx.hip",
        128,
        [
            ("Q8", "ptr"),
            ("QS", "ptr"),
            ("K8", "ptr"),
            ("KS", "ptr"),
            ("V8", "ptr"),
            ("VSR", "ptr"),
            ("DO8", "ptr"),
            ("DOS", "ptr"),
            ("K8T", "ptr"),
            ("KTS", "ptr"),
            ("LSE", "ptr"),
            ("DELTA", "ptr"),
            ("DQ", "ptr"),
            ("scale", "f32"),
            ("S", "i32"),
            ("H", "i32"),
            ("B", "i32"),
            ("dq_sb", "i64"),
            ("dq_ss", "i64"),
        ],
    ),
}
KERNELS["k_dkdv_mxr"] = (KERNELS["k_dkdv_mx"][0], 128,
                         KERNELS["k_dkdv_mx"][2] + [("DKP", "ptr"), ("dkp_sb", "i64"), ("dkp_ss", "i64")])
# the fused-RoPE path's dQ kernel (rung b5): k_dq_mx's args + COS, SIN [S, 64] bf16
KERNELS["k_dq_mxr"] = (KERNELS["k_dq_mx"][0], 128, KERNELS["k_dq_mx"][2] + [("COS", "ptr"), ("SIN", "ptr")])
# rung b2's dV / dK kernels (k_mla_bwd_mx2.hip) take k_dkdv_mx's arguments
KERNELS["k_dv_mx"] = KERNELS["k_dk_mx"] = KERNELS["k_dkdv_mx"]
# rung b7 (k_mla_bwd_mx5.hip, build without SLP): k_dkdv_mx(r) + the bf16 dS tiles (DSB), dQ as a GEMM over them
KERNELS["k_dkdv_mxd"] = (KERNELS["k_dkdv_mx"][0], 128, KERNELS["k_dkdv_mx"][2] + [("DSB", "ptr")])
KERNELS["k_dkdv_mxrd"] = (KERNELS["k_dkdv_mx"][0], 128, KERNELS["k_dkdv_mxr"][2] + [("DSB", "ptr")])
_DQG_ARGS = [("DSB", "ptr"), ("K8T", "ptr"), ("KTS", "ptr"), ("DQ", "ptr"), ("scale", "f32"), ("S", "i32"), ("H", "i32"),
             ("B", "i32"), ("dq_sb", "i64"), ("dq_ss", "i64")]
KERNELS["k_dqg_mx"] = (KERNELS["k_dq_mx"][0], 128, _DQG_ARGS)
KERNELS["k_dqg_mxr"] = (KERNELS["k_dq_mx"][0], 128, _DQG_ARGS + [("COS", "ptr"), ("SIN", "ptr")])
# rung b8: dK's dS operand (32 q x 32 kv block scales) stored as is (fp8 + scales), dQ straight from it
KERNELS["k_dkdv_mx8"] = KERNELS["k_dkdv_mxd"]
KERNELS["k_dkdv_mxr8"] = KERNELS["k_dkdv_mxrd"]
KERNELS["k_dqg_mx8"] = KERNELS["k_dqg_mx"]
KERNELS["k_dqg_mxr8"] = KERNELS["k_dqg_mxr"]
KERNELS["k_dqg_mx8x"] = KERNELS["k_dqg_mx"]  # DQG_XPF: the next tile's loads under this tile's WMMAs (no RoPE)
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


def build(sources=None, defines=(), tag="", flags=()):
    """Compile kern sources (default: every one KERNELS uses) to BUILD/<stem><tag>.hsaco (extra compiler flags:
    e.g. EXPERT)."""
    BUILD.mkdir(exist_ok=True)
    srcs = sorted({v[0] for v in KERNELS.values()}) if sources is None else sources
    out = {}
    for s in srcs:
        dst = BUILD / (Path(s).stem + tag + ".hsaco")
        cmd = [cxx(), *FLAGS, *flags, *[f"-D{d}" for d in defines], "-o", str(dst), str(KERN / s)]
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


def quant_rows(x, d, mode=0, kbar=None, y=None, want_deq=False, tag="", pre=1.0):
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
    _launch(
        f"k_mxq_rows{d}",
        ((b * h * s + rows_per_block - 1) // rows_per_block, 1, 1),
        [
            x,
            x8,
            xs,
            kbar if kbar is not None else _dummy(dev),
            yy,
            aux,
            xd,
            b,
            s,
            h,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            yy.stride(0),
            yy.stride(1),
            yy.stride(2),
            m,
            float(pre),
        ],
        tag,
    )
    return x8, xs, (aux if mode & 6 else None), (xd if want_deq else None)


def quant_cols(x, d, kbar=None, tag=""):
    """k_mxq_cols{128,192}: x [b, s, h, d] -> (x8t [b, h, d, s], xts [b, h, s / 128, d, 4])."""
    b, s, h, _ = x.shape
    dev = x.device
    x8t = torch.empty(b, h, d, s, device=dev, dtype=torch.uint8)
    xts = torch.empty(b, h, s // 128, d, 4, device=dev, dtype=torch.uint8)
    _launch(
        f"k_mxq_cols{d}",
        (s // 128, h, b),
        [
            x,
            x8t,
            xts,
            kbar if kbar is not None else _dummy(dev),
            s,
            h,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            1 if kbar is not None else 0,
        ],
        tag,
    )
    return x8t, xts


KBAR_STRIDE = 16  # k_mxq_kbar samples every 16th row (any constant shift of k is softmax-invariant)


RCP_WGS = 256  # k_mxq_rcp workgroups (one per CU)


def quant_rc(x, d, mode=0, kbar=None, y=None, pre=1.0, tag="", kname="rc"):
    """k_mxq_rc{192,128}: rows and cols of x [b, s, h, d] in one pass -> (x8, xs, aux | None, x8t, xts), the layouts of
    quant_rows / quant_cols (cols: kbar subtracted when mode & 1, never pre-scaled)."""
    b, s, h, _ = x.shape
    dev = x.device
    x8 = torch.empty(b, h, s, d, device=dev, dtype=torch.uint8)
    xs = torch.empty(b, h, s, 8 if d == 192 else 4, device=dev, dtype=torch.uint8)
    x8t = torch.empty(b, h, d, s, device=dev, dtype=torch.uint8)
    xts = torch.empty(b, h, s // 128, d, 4, device=dev, dtype=torch.uint8)
    aux = torch.empty(b, h, s, device=dev, dtype=torch.float32) if mode & 6 else _dummy(dev)
    yy = y if y is not None else x
    if kname == "rcp":  # persistent: one workgroup per CU
        _launch(f"k_mxq_rcp{d}", (min(RCP_WGS, (s // 128) * h * b), 1, 1),
                [x, x8, xs, x8t, xts, kbar if kbar is not None else _dummy(dev), yy, aux, b, s, h, x.stride(0),
                 x.stride(1), x.stride(2), yy.stride(0), yy.stride(1), yy.stride(2), mode, float(pre)], tag)  # fmt: skip
        return x8, xs, (aux if mode & 6 else None), x8t, xts
    _launch(f"k_mxq_{kname}{d}", (s // 128, h, b),
            [x, x8, xs, x8t, xts, kbar if kbar is not None else _dummy(dev), yy, aux, s, h, x.stride(0), x.stride(1),
             x.stride(2), yy.stride(0), yy.stride(1), yy.stride(2), mode, float(pre)], tag)  # fmt: skip
    return x8, xs, (aux if mode & 6 else None), x8t, xts


def kbar_rope(x, pe, cos, sin, tag="", stride=KBAR_STRIDE):
    """kbar of k = [x[..., :128] | rope(pe)] (x [b, s, h, >= 128], pe [b, s, 1 or h, 64], cos / sin [s, 64] bf16)."""
    b, s, h, _ = x.shape
    kbar = torch.empty(b, h, D_QK, device=x.device, dtype=torch.float32)
    _launch("k_mxq_kbar_rope", (b * h, 1, 1),
            [x, kbar, s, h, x.stride(0), x.stride(1), x.stride(2), pe, pe.stride(0), pe.stride(1),
             pe.stride(2) if pe.shape[2] > 1 else 0, cos, sin, stride], tag)  # fmt: skip
    return kbar


def quant_rope(x, pe, cos, sin, mode=0, kbar=None, pre=1.0, tag=""):
    """k_mxq_rctr192: rows + cols of [x[..., :128] | rope(pe)] (q: pe = x[..., 128:]; k: x = kv, pe = k_pe [b, s, 1, 64])
    -> (x8, xs, aux | None, x8t, xts) as quant_rc(.., 192, ..)."""
    b, s, h, _ = x.shape
    dev = x.device
    x8 = torch.empty(b, h, s, 192, device=dev, dtype=torch.uint8)
    xs = torch.empty(b, h, s, 8, device=dev, dtype=torch.uint8)
    x8t = torch.empty(b, h, 192, s, device=dev, dtype=torch.uint8)
    xts = torch.empty(b, h, s // 128, 192, 4, device=dev, dtype=torch.uint8)
    aux = torch.empty(b, h, s, device=dev, dtype=torch.float32) if mode & 6 else _dummy(dev)
    _launch("k_mxq_rctr192", (s // 128, h, b),
            [x, x8, xs, x8t, xts, kbar if kbar is not None else _dummy(dev), aux, s, h, x.stride(0), x.stride(1),
             x.stride(2), pe, pe.stride(0), pe.stride(1), pe.stride(2) if pe.shape[2] > 1 else 0, cos, sin, mode,
             float(pre)], tag)  # fmt: skip
    return x8, xs, (aux if mode & 6 else None), x8t, xts


def rope_ref(pe, cos, sin):
    """Megatron's MLA rotary on the last dim (64, interleaved pairs in, halves out), fp32 per op, bf16 out; cos / sin
    broadcast over the leading dims of pe as given."""
    x = pe.float()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, sn = cos.float(), sin.float()
    left = x1 * c[..., :32] - x2 * sn[..., :32]
    right = x2 * c[..., 32:] + x1 * sn[..., 32:]
    return torch.cat([left, right], dim=-1).to(torch.bfloat16)


def quant_all(q, k, v, recipe="m4", scale=SCALE, tag=""):
    """Every MX operand of both passes from one read of q / k / v (k_mxq_kbar + 3 x k_mxq_rc): (qz as quant_inputs,
    the backward's q8t / qts / k8t / kts / v8 / vsr)."""
    b, s, h, _ = q.shape
    m4 = recipe == "m4"
    kbar = kbar_of(k, tag)
    q8, qs, qk, q8t, qts = quant_rc(q, D_QK, mode=2 | (48 if m4 else 0), kbar=kbar, pre=scale * LOG2E if m4 else 1.0,
                                    tag=tag)  # fmt: skip
    k8, ks, _, k8t, kts = quant_rc(k, D_QK, mode=1 | (16 if m4 else 0), kbar=kbar, tag=tag)
    v8, vsr, _, v8t, vs = quant_rc(v, D_V, tag=tag)
    qz = {"b": b, "s": s, "h": h, "center_k": True, "recipe": recipe, "q8": q8, "qs": qs, "k8": k8, "ks": ks,
          "v8t": v8t, "vs": vs, "kbar": kbar, "qd": None, "kd": None, "qk": qk}  # fmt: skip
    return qz, {"q8t": q8t, "qts": qts, "k8t": k8t, "kts": kts, "v8": v8, "vsr": vsr}


def quant_bwd_do(qz, pre, o, do, tag=""):
    """The backward's remaining operands after quant_all: dO rows (+ delta = rowsum(dO * O)) and cols in one pass."""
    do8, dos, delta, do8t, dots = quant_rc(do, D_V, mode=4, y=o, tag=tag)
    return dict(pre, do8=do8, dos=dos, delta=delta, do8t=do8t, dots=dots)


def kbar_of(k, tag="", stride=KBAR_STRIDE):
    b, s, h, _ = k.shape
    kbar = torch.empty(b, h, D_QK, device=k.device, dtype=torch.float32)
    _launch("k_mxq_kbar", (b * h, 1, 1), [k, kbar, s, h, k.stride(0), k.stride(1), k.stride(2), stride], tag)
    return kbar


def quant_inputs(q, k, v, center_k=True, want_deq=False, tag="", recipe="r1", scale=SCALE):
    """q, k [b, s, h, 192], v [b, s, h, 128] bf16 (d contiguous; any b / s / h strides) -> the forward's MX operands
    (+ the dequantized q / k as bf16 [b, s, h, 192] when want_deq: the bf16 backward's consistent inputs)."""
    b, s, h, _ = q.shape
    assert s % 256 == 0 and q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
    out = {"b": b, "s": s, "h": h, "center_k": center_k, "recipe": recipe}
    kbar = kbar_of(k, tag) if center_k else None
    m4 = recipe == "m4"  # Q8 of q * scale * log2(e) (S in log2 units), scale layout R2 for Q and K
    q8, qs, qk, qd = quant_rows(
        q,
        D_QK,
        mode=(2 if center_k else 0) | (48 if m4 else 0),
        kbar=kbar,
        want_deq=want_deq,
        tag=tag,
        pre=scale * LOG2E if m4 else 1.0,
    )
    k8, ks, _, kd = quant_rows(
        k, D_QK, mode=(1 if center_k else 0) | (16 if m4 else 0), kbar=kbar, want_deq=want_deq, tag=tag
    )
    v8t, vs = quant_cols(v, D_V, tag=tag)
    out.update(
        q8=q8,
        qs=qs,
        k8=k8,
        ks=ks,
        v8t=v8t,
        vs=vs,
        kbar=kbar,
        qd=qd,
        kd=kd,
        qk=qk if qk is not None else torch.zeros(b, h, s, device=q.device),
    )
    return out


def fwd_mx(qz, scale=SCALE, out=None, tag=""):
    """The MX forward on quantize_inputs' operands -> o [b, s, h, 128] bf16 (BSHD view of SBHD storage when out is
    None: allocated [s, b, h, 128]), lse [b, h, s] (natural log of the centred logits)."""
    b, s, h = qz["b"], qz["s"], qz["h"]
    dev = qz["q8"].device
    if out is None:
        out = torch.empty(s, b, h, D_V, device=dev, dtype=torch.bfloat16).transpose(0, 1)
    lse = torch.empty(b, h, s, device=dev, dtype=torch.float32)
    nqb = s // 256
    paired = tag[:1] == "7" and nqb % 2 == 0 and (b * h) % 8 == 0  # rung m7: two q blocks per workgroup
    _launch(
        "k_mla_fwd_mx",
        (nqb // 2 if paired else nqb, h, b),
        [
            qz["q8"],
            qz["qs"],
            qz["k8"],
            qz["ks"],
            qz["v8t"],
            qz["vs"],
            out,
            lse,
            float(scale),
            s,
            h,
            b,
            out.stride(0),
            out.stride(1),
        ],
        tag,
    )
    if tag[:1] == "7":  # rung m7 writes log2 units (11-06): natural as the other rungs
        lse = lse * LN2
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


R2_IDX = [0, 1, 2, 3, 6, 7]  # the 6 real scale bytes of layout R2 ([s0..s3, 0, 0, s4, s5])


def deq_qk(x8, xs, r2=False):
    """Dequantize X8 / XS (scale layout R1, or R2) -> fp32 [b, h, s, 192] (logical order)."""
    codes = deinterleave_qk(x8).view(torch.float8_e4m3fn).float()
    e = (xs[..., R2_IDX] if r2 else xs[..., :6]).to(torch.int32) - 127
    sc = torch.ldexp(torch.ones_like(e, dtype=torch.float32), e)
    return (codes.reshape(*codes.shape[:-1], 6, 32) * sc[..., None]).reshape(codes.shape)


def deq_vt(v8t, vs):
    """Dequantize V8T / VS -> fp32 V [b, h, s, 128]."""
    b, h, d, s = v8t.shape
    codes = deinterleave_kv(v8t).view(torch.float8_e4m3fn).float()  # b h d s
    e = vs.to(torch.int32) - 127  # b h (s/128) d 4
    sc = (
        torch.ldexp(torch.ones_like(e, dtype=torch.float32), e)
        .permute(0, 1, 3, 2, 4)
        .reshape(b, h, d, s // 32)
    )
    return (codes.reshape(b, h, d, s // 32, 32) * sc[..., None]).reshape(b, h, d, s).transpose(2, 3)


def ref_quant_qk(x, kbar=None, pre=None):
    """Torch twin of k_mxq_qk (logical order): x [b, s, h, 192] bf16 -> codes [b, h, s, 192], exps [b, h, s, 6]
    (pre: quantize x * pre, the kernels' mode 32)."""
    xf = x.float().permute(0, 2, 1, 3)
    if kbar is not None:
        xf = xf - kbar[:, :, None, :]
    if pre is not None:
        xf = xf * torch.tensor(pre, dtype=torch.float32)
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
    c2, tau2, ps = scale * LOG2E, tau * LOG2E, 2.0**p_exp
    o = torch.zeros(b, h, s, D_V, device=dev)
    lse = torch.zeros(b, h, s, device=dev)
    qi = torch.arange(s, device=dev)
    for q0 in range(0, s, bm):
        Q = qf[:, :, q0 : q0 + bm]  # b h 256 d
        m = torch.full((b, h, bm), -1e30, device=dev)
        l = torch.zeros(b, h, bm, device=dev)
        acc = torch.zeros(b, h, bm, D_V, device=dev)
        for k0 in range(0, q0 + bm, nb):
            S = torch.einsum("bhqd,bhkd->bhqk", Q, kf[:, :, k0 : k0 + nb])
            msk = qi[q0 : q0 + bm, None] >= qi[None, k0 : k0 + nb]
            S = S.masked_fill(~msk, float("-inf"))
            lm = S.amax(-1)
            need = (lm * c2 - m) > tau2  # b h 256
            resc = (
                need.reshape(b, h, bm // 16, 16)
                .any(-1, keepdim=True)
                .expand(-1, -1, -1, 16)
                .reshape(b, h, bm)
            )
            mn = torch.where(resc, torch.maximum(m, lm * c2), m)
            corr = torch.exp2(m - mn)
            p = torch.exp2(S * c2 - mn[..., None])
            l = corr * l + p.sum(-1)
            pq = e4m3(p / ps).float() * ps
            acc = (
                acc * torch.where(resc, corr, torch.ones_like(corr))[..., None] + pq @ vf[:, :, k0 : k0 + nb]
            )
            m = mn
        o[:, :, q0 : q0 + bm] = acc / l[..., None]
        lse[:, :, q0 : q0 + bm] = (m + torch.log2(l)) / LOG2E
    return o.permute(0, 2, 1, 3), lse


def emu_fwd4(qf, kf, vf, tau2=1.75, p_exp=-7, bm=256, nb=128):
    """fp32 emulation of rung m4: qf [b, h, s, 192] dequantizes Q8 of q * scale * log2(e) (S' in log2 units). Per q16
    tile and 128-kv tile: S'' = S' - m (m from 0), lm = max S''; when any row of the q16 tile has lm > tau2 (or on
    the first tile) every row takes d = max(lm, 0) (first: lm): S'' -= d, acc and l *= 2^-d (first: 0), m += d. P =
    e4m3(2^S'' / 2^p_exp) 2^p_exp; l accumulates the quantized P (the all-ones PV WMMA)."""
    b, h, s, _ = qf.shape
    dev = qf.device
    ps = 2.0**p_exp
    o = torch.zeros(b, h, s, D_V, device=dev)
    lse = torch.zeros(b, h, s, device=dev)
    qi = torch.arange(s, device=dev)
    for q0 in range(0, s, bm):
        Q = qf[:, :, q0 : q0 + bm]
        m = torch.zeros(b, h, bm, device=dev)
        l = torch.zeros(b, h, bm, device=dev)
        acc = torch.zeros(b, h, bm, D_V, device=dev)
        for t, k0 in enumerate(range(0, q0 + bm, nb)):
            S = torch.einsum("bhqd,bhkd->bhqk", Q, kf[:, :, k0 : k0 + nb]) - m[..., None]
            msk = qi[q0 : q0 + bm, None] >= qi[None, k0 : k0 + nb]
            S = S.masked_fill(~msk, float("-inf"))
            lm = S.amax(-1)
            if t == 0:
                d = lm
                corr = torch.zeros_like(lm)
            else:
                need = (lm > tau2).reshape(b, h, bm // 16, 16).any(-1, keepdim=True).expand(-1, -1, -1, 16)
                need = need.reshape(b, h, bm)
                d = torch.where(need, torch.clamp(lm, min=0.0), torch.zeros_like(lm))
                corr = torch.where(need, torch.exp2(-d), torch.ones_like(lm))
            S = S - d[..., None]
            pq = e4m3(torch.exp2(S) / ps).float() * ps
            l = corr * l + pq.sum(-1)
            acc = acc * corr[..., None] + pq @ vf[:, :, k0 : k0 + nb]
            m = m + d
        o[:, :, q0 : q0 + bm] = acc / l[..., None]
        lse[:, :, q0 : q0 + bm] = (m + torch.log2(l)) / LOG2E
    return o.permute(0, 2, 1, 3), lse


def emu_fwd5(qf, kf, vf, tau2=3.75, p_exp=-5, bm=256, nb=128):
    """fp32 emulation of rung m5: as emu_fwd4, but P = e4m3(p / 2^p_exp) 2^p_exp with p_exp -5 (headroom 14: tau2
    3.75) and, after the first tile, an integer shift d = ceil(max(lm, 0)) applied through the P conversion's scale
    (S'' itself is not adjusted: exp2(S'') 2^-d)."""
    b, h, s, _ = qf.shape
    dev = qf.device
    ps = 2.0**p_exp
    o = torch.zeros(b, h, s, D_V, device=dev)
    lse = torch.zeros(b, h, s, device=dev)
    qi = torch.arange(s, device=dev)
    for q0 in range(0, s, bm):
        Q = qf[:, :, q0 : q0 + bm]
        m = torch.zeros(b, h, bm, device=dev)
        l = torch.zeros(b, h, bm, device=dev)
        acc = torch.zeros(b, h, bm, D_V, device=dev)
        for t, k0 in enumerate(range(0, q0 + bm, nb)):
            S = torch.einsum("bhqd,bhkd->bhqk", Q, kf[:, :, k0 : k0 + nb]) - m[..., None]
            msk = qi[q0 : q0 + bm, None] >= qi[None, k0 : k0 + nb]
            S = S.masked_fill(~msk, float("-inf"))
            lm = S.amax(-1)
            if t == 0:
                d, corr = lm, torch.zeros_like(lm)
                p = torch.exp2(S - d[..., None])
            else:
                need = (lm > tau2).reshape(b, h, bm // 16, 16).any(-1, keepdim=True).expand(-1, -1, -1, 16)
                need = need.reshape(b, h, bm)
                d = torch.where(need, torch.ceil(torch.clamp(lm, min=0.0)), torch.zeros_like(lm))
                corr = torch.exp2(-d)
                p = torch.exp2(S) * corr[..., None]
            pq = e4m3(p / ps).float() * ps
            l = corr * l + pq.sum(-1)
            acc = acc * corr[..., None] + pq @ vf[:, :, k0 : k0 + nb]
            m = m + d
        o[:, :, q0 : q0 + bm] = acc / l[..., None]
        lse[:, :, q0 : q0 + bm] = (m + torch.log2(l)) / LOG2E
    return o.permute(0, 2, 1, 3), lse


def emu_fwd6(qf, kf, vf, tau2=3.75, pe=5, bm=256, nb=128):
    """fp32 emulation of rung m6 (per q row, no wave-uniform decision): S'' = S' - m (m from 0); lm = max S''. First
    tile: d = ceil(lm), S'' -= d before the exp2 (exact), the reference mref = d. Later tiles: d = ceil(lm) if lm > tau2
    else 0; p = exp2(S'') unshifted and P = e4m3(p / 2^(d - pe)) 2^-pe (the shift goes into the conversion scale).
    O and the row sum accumulate relative to mref: weight 2^D, D = m - mref (the PV WMMA's per-lane E8M0 B scale), so
    O is never rescaled. lse = mref + log2(l)."""
    b, h, s, _ = qf.shape
    dev = qf.device
    ps = 2.0**-pe
    o = torch.zeros(b, h, s, D_V, device=dev)
    lse = torch.zeros(b, h, s, device=dev)
    qi = torch.arange(s, device=dev)
    for q0 in range(0, s, bm):
        Q = qf[:, :, q0 : q0 + bm]
        m = torch.zeros(b, h, bm, device=dev)
        D = torch.zeros(b, h, bm, device=dev)
        mref = torch.zeros(b, h, bm, device=dev)
        l = torch.zeros(b, h, bm, device=dev)
        acc = torch.zeros(b, h, bm, D_V, device=dev)
        for t, k0 in enumerate(range(0, q0 + bm, nb)):
            S = torch.einsum("bhqd,bhkd->bhqk", Q, kf[:, :, k0 : k0 + nb]) - m[..., None]
            msk = qi[q0 : q0 + bm, None] >= qi[None, k0 : k0 + nb]
            S = S.masked_fill(~msk, float("-inf"))
            lm = S.amax(-1)
            if t == 0:
                d = torch.ceil(lm)
                mref = d.clone()
                P = e4m3(torch.exp2(S - d[..., None]) / ps).float() * ps
                w = torch.ones_like(d)
            else:
                d = torch.where(lm > tau2, torch.ceil(lm), torch.zeros_like(lm))
                P = e4m3(torch.exp2(S) / torch.exp2(d - pe)[..., None]).float() * ps
                D = D + d
                w = torch.exp2(D)
            l = l + w * P.sum(-1)
            acc = acc + w[..., None] * (P @ vf[:, :, k0 : k0 + nb])
            m = m + d
        o[:, :, q0 : q0 + bm] = acc / l[..., None]
        lse[:, :, q0 : q0 + bm] = (mref + torch.log2(l)) / LOG2E
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
    return dict(
        v8=v8,
        vsr=vsr,
        do8=do8,
        dos=dos,
        delta=delta,
        do8t=do8t,
        dots=dots,
        q8t=q8t,
        qts=qts,
        k8t=k8t,
        kts=kts,
    )


def dkdv_persistent_grid(b, s, h, cus=256):
    """b6's k_dkdv_mx grid: 8 x (workgroups per XCD), one per CU at most, over (B H / 8) x (S / 256) items per XCD."""
    assert (s // 128) % 2 == 0 and (b * h) % 8 == 0, "b6 needs S / 128 even and B H % 8 == 0"
    items = (b * h // 8) * (s // 256)
    return 8 * min(cus // 8, items)


def bwd_mx(qz, bz, lse, scale=SCALE, tag=""):
    """dq, dk [b, s, h, 192], dv [b, s, h, 128] bf16 (BSHD views of SBHD storage) from the MX operands. Rungs b1 / b2
    (tags 1*, 2*) take the m4 recipe's operands and the lse in log2 units; b2 runs dV and dK as two kernels
    (k_dv_mx, k_dk_mx: 4 kv16 tiles per wave, 256 kv rows per workgroup) instead of k_dkdv_mx (128 rows)."""
    b, s, h = qz["b"], qz["s"], qz["h"]
    dev = qz["q8"].device
    if recipe_of(tag, "bwd") == "m4":
        assert qz.get("recipe") == "m4", "rungs b1 / b2 need the m4 recipe's operands"
        lse = lse * LOG2E
    dq = torch.empty(s, b, h, D_QK, device=dev, dtype=torch.bfloat16).transpose(0, 1)
    dk = torch.empty(s, b, h, D_QK, device=dev, dtype=torch.bfloat16).transpose(0, 1)
    dv = torch.empty(s, b, h, D_V, device=dev, dtype=torch.bfloat16).transpose(0, 1)
    args = [qz["q8"], qz["qs"], qz["k8"], qz["ks"], bz["v8"], bz["vsr"], bz["do8"], bz["dos"], bz["do8t"], bz["dots"],
            bz["q8t"], bz["qts"], lse, bz["delta"], dk, dv, float(scale), s, h, b, dk.stride(0), dk.stride(1),
            dv.stride(0), dv.stride(1)]  # fmt: skip
    if tag[:1] in ("2", "4"):  # b2 / b4: dV and dK as two kernels (kv rows per workgroup: 64 NJ_DV, 64 NJ_DK)
        _launch("k_dv_mx", (-(-s // 256), h, b), args, tag)
        _launch("k_dk_mx", (-(-s // (192 if tag[:1] == "4" else 256)), h, b), args, tag)
    elif tag[:1] == "6":  # b6: persistent k_dkdv_mx (S / 128 even, B H % 8 == 0)
        _launch("k_dkdv_mx", (dkdv_persistent_grid(b, s, h), 1, 1), args, tag)
    else:
        _launch("k_dkdv_mx", (s // 128, h, b), args, tag)
    _launch("k_dq_mx", (s // 256, h, b),
            [qz["q8"], qz["qs"], qz["k8"], qz["ks"], bz["v8"], bz["vsr"], bz["do8"], bz["dos"], bz["k8t"], bz["kts"],
             lse, bz["delta"], dq, float(scale), s, h, b, dq.stride(0), dq.stride(1)], tag)  # fmt: skip
    return dq, dk, dv


DS_TILE = 128 * 128 * 2  # rung b7's dS tile bytes (k_mla_bwd_mx5.hip ds_tile_index layout)


def ds_bytes(b, h, s):
    """Bytes of rung b7's materialized dS (bf16 causal 128 x 128 tiles per (b, h))."""
    nb = s // 128
    return b * h * (nb * (nb + 1) // 2) * DS_TILE


def bwd_mx7(qz, bz, lse, scale=SCALE, tag="5ni", keep_ds=False, dsb=None):
    """Rung b7: k_dkdv_mxd (dK, dV and the bf16 dS tiles) then k_dqg_mx (dQ = scale dS K). lse natural (as bwd_mx).
    Returns dq, dk, dv (+ the dS buffer with keep_ds)."""
    b, s, h = qz["b"], qz["s"], qz["h"]
    assert s % 256 == 0
    dev = qz["q8"].device
    lse2 = lse * LOG2E
    dq = torch.empty(s, b, h, D_QK, device=dev, dtype=torch.bfloat16).transpose(0, 1)
    dk = torch.empty(s, b, h, D_QK, device=dev, dtype=torch.bfloat16).transpose(0, 1)
    dv = torch.empty(s, b, h, D_V, device=dev, dtype=torch.bfloat16).transpose(0, 1)
    if dsb is None:
        dsb = torch.empty(ds_bytes(b, h, s), device=dev, dtype=torch.uint8)
    _launch("k_dkdv_mxd", (s // 128, h, b),
            [qz["q8"], qz["qs"], qz["k8"], qz["ks"], bz["v8"], bz["vsr"], bz["do8"], bz["dos"], bz["do8t"], bz["dots"],
             bz["q8t"], bz["qts"], lse2, bz["delta"], dk, dv, float(scale), s, h, b, dk.stride(0), dk.stride(1),
             dv.stride(0), dv.stride(1), dsb], tag)  # fmt: skip
    _launch("k_dqg_mx", dqg_grid(s, h, b, "P" in tag),
            [dsb, bz["k8t"], bz["kts"], dq, float(scale), s, h, b, dq.stride(0), dq.stride(1)], tag)  # fmt: skip
    return (dq, dk, dv, dsb) if keep_ds else (dq, dk, dv)


DS8_TILE = 128 * 128 + 128  # rung b8's dS tile record: 16 KB of E4M3 + the 4 waves' scale words (+ pad)


def ds8_bytes(b, h, s):
    """Bytes of rung b8's dS records (causal 128 x 128 tiles per (b, h))."""
    nb = s // 128
    return b * h * (nb * (nb + 1) // 2) * DS8_TILE


def bwd_mx8(qz, bz, lse, scale=SCALE, tag="5ni", keep_ds=False, dsb=None):
    """Rung b8: k_dkdv_mx8 (dK, dV and the fp8 dS records) then k_dqg_mx8 (dQ from them, paired grid with "P" in the
    tag). Returns dq, dk, dv (+ the dS buffer with keep_ds)."""
    b, s, h = qz["b"], qz["s"], qz["h"]
    assert s % 256 == 0
    dev = qz["q8"].device
    lse2 = lse * LOG2E
    dq = torch.empty(s, b, h, D_QK, device=dev, dtype=torch.bfloat16).transpose(0, 1)
    dk = torch.empty(s, b, h, D_QK, device=dev, dtype=torch.bfloat16).transpose(0, 1)
    dv = torch.empty(s, b, h, D_V, device=dev, dtype=torch.bfloat16).transpose(0, 1)
    if dsb is None:
        dsb = torch.empty(ds8_bytes(b, h, s), device=dev, dtype=torch.uint8)
    _launch("k_dkdv_mx8", dkdv_grid(s, h, b, "W" in tag),
            [qz["q8"], qz["qs"], qz["k8"], qz["ks"], bz["v8"], bz["vsr"], bz["do8"], bz["dos"], bz["do8t"], bz["dots"],
             bz["q8t"], bz["qts"], lse2, bz["delta"], dk, dv, float(scale), s, h, b, dk.stride(0), dk.stride(1),
             dv.stride(0), dv.stride(1), dsb], tag)  # fmt: skip
    _launch("k_dqg_mx8x" if "X" in tag else "k_dqg_mx8", dqg_grid(s, h, b, "P" in tag),
            [dsb, bz["k8t"], bz["kts"], dq, float(scale), s, h, b, dq.stride(0), dq.stride(1)], tag)  # fmt: skip
    return (dq, dk, dv, dsb) if keep_ds else (dq, dk, dv)


def ds8_unpack(dsb, b, h, s):
    """rung b8's dS records -> (dequantized dense fp32 [b, h, s (q), s (kv)], scale bytes [b, h, tiles, 4 w, 4 m])."""
    nb = s // 128
    nt = nb * (nb + 1) // 2
    rec = dsb.view(b, h, nt, DS8_TILE)
    codes = rec[..., :16384].contiguous().view(torch.float8_e4m3fn).float().view(b, h, nt, 4, 4, 2, 2, 16, 2, 8)
    scb = rec[..., 16384:16400].contiguous().view(b, h, nt, 4, 4)  # w, m
    sc = torch.ldexp(torch.ones_like(scb, dtype=torch.float32), scb.to(torch.int32) - 127)
    # codes dims: w m j lh lr s r; q = 32 m + 16 s + 8 lh + r, kv = 32 w + 16 j + lr
    deq = codes * sc[..., None, None, None, None, None]
    out = torch.zeros(b, h, s, s, device=dsb.device)
    for qb in range(nb):
        for kb in range(qb + 1):
            x = deq[:, :, qb * (qb + 1) // 2 + kb]  # b h w m j lh lr s r
            blk = x.permute(0, 1, 3, 7, 5, 8, 2, 4, 6).reshape(b, h, 128, 128)  # (m s lh r) x (w j lr)
            out[:, :, qb * 128:(qb + 1) * 128, kb * 128:(kb + 1) * 128] = blk
    return out, scb


def dkdv_grid(s, h, b, paired):
    """k_dkdv_mx(8)'s grid: kv blocks of 128; paired builds (DKDV_PAIR, lab tags with a "W"): two blocks per workgroup
    when nkb is even"""
    nkb = s // 128
    return (nkb // 2, h, b) if paired and nkb % 2 == 0 else (nkb, h, b)


def dqg_grid(s, h, b, paired):
    """k_dqg_mx's grid: q blocks of 256 (paired builds, DQG_PAIR: two blocks per workgroup when nqb % 2 == 0 and
    b h % 8 == 0; lab tags with a "P")"""
    nqb = s // 256
    if paired and nqb % 2 == 0 and (b * h) % 8 == 0:
        return (nqb // 2, h, b)
    return (nqb, h, b)


def ds_unpack(dsb, b, h, s):
    """rung b7's dS buffer -> dense bf16 [b, h, s (q), s (kv)] (tiles above the diagonal block row: 0)."""
    nb = s // 128
    nt = nb * (nb + 1) // 2
    t = dsb.view(torch.bfloat16).view(b, h, nt, 4, 8, 2, 2, 16, 8)  # tile, w, qt, j, qh, kvl, r
    out = torch.zeros(b, h, s, s, device=dsb.device, dtype=torch.bfloat16)
    for qb in range(nb):
        for kb in range(qb + 1):
            x = t[:, :, qb * (qb + 1) // 2 + kb]  # b h w qt j qh kvl r
            # q = 16 qt + 8 qh + r, kv = 32 w + 16 j + kvl
            blk = x.permute(0, 1, 3, 5, 7, 2, 4, 6).reshape(b, h, 128, 128)
            out[:, :, qb * 128:(qb + 1) * 128, kb * 128:(kb + 1) * 128] = blk
    return out


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
    sc = (
        torch.ldexp(torch.ones_like(e, dtype=torch.float32), e)
        .permute(0, 1, 3, 2, 4)
        .reshape(b, h, d, s // 32)
    )
    return (codes.reshape(b, h, d, s // 32, 32) * sc[..., None]).reshape(b, h, d, s).transpose(2, 3)


def mxq(x, dim):
    """Fake MX quantization of fp32 x along dim (blocks of 32, the kernels' scale rule)."""
    xm = x.movedim(dim, -1)
    codes, e = mx_quant_last(xm)
    sc = torch.ldexp(torch.ones_like(e, dtype=torch.float32), e)
    deq = (codes.float().reshape(*codes.shape[:-1], codes.shape[-1] // 32, 32) * sc[..., None]).reshape(
        codes.shape
    )
    return deq.movedim(-1, dim)


def mxq_block(x, bq=32, bk=32):
    """Fake MX quantization of fp32 x [..., Q, KV] with one scale per aligned bq x bk block (the kernels' rule)."""
    *lead, nq, nk = x.shape
    xb = x.reshape(*lead, nq // bq, bq, nk // bk, bk).movedim(-3, -2).reshape(*lead, nq // bq, nk // bk, bq * bk)
    codes, e = mx_quant_last(xb, block=bq * bk)
    deq = codes.float() * torch.ldexp(torch.ones_like(e, dtype=torch.float32), e)  # e [..., 1]
    return deq.reshape(*lead, nq // bq, nk // bk, bq, bk).movedim(-2, -3).reshape(x.shape)


def emu_bwd(qz, bz, lse, scale=SCALE, chunk=256, dq_bf16=False, ds_out=None, ds_block=False):
    """fp32 emulation of k_dkdv_mx / k_dq_mx on the quantized operands: dq, dk [b, s, h, 192], dv [b, s, h, 128]
    (recipe m4: Q8 already in log2 units). dq_bf16: rung b7's dQ (MX of bf16(dS)); ds_block: rung b8 (one fp8 dS with
    32 q x 32 kv block scales for dK and dQ); ds_out: a [b, h, s, s] tensor that receives dS (fp32, masked)."""
    m4 = qz.get("recipe") == "m4"
    qf = deq_qk(qz["q8"], qz["qs"], r2=m4)
    kf = deq_qk(qz["k8"], qz["ks"], r2=m4)
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
            P = torch.exp2((S if m4 else S * (scale * LOG2E)) - lse[:, sl, qs_, None] * LOG2E)
            P = P.masked_fill(qi[qs_, None] < qi[None, : q0 + chunk], 0.0)
            dP = torch.einsum("bhqd,bhkd->bhqk", dor[:, sl, qs_], vr[:, sl, : q0 + chunk])
            dS = P * (dP - delta[:, sl, qs_, None])
            if ds_out is not None:
                ds_out[:, sl, qs_, : q0 + chunk] = dS
            Pq = e4m3(P * 256.0).float() / 256.0
            dv[:, sl, : q0 + chunk] += torch.einsum("bhqk,bhqd->bhkd", Pq, doc[:, sl, qs_])
            dsb8 = mxq_block(dS) if ds_block else None
            dk[:, sl, : q0 + chunk] += torch.einsum("bhqk,bhqd->bhkd", dsb8 if ds_block else mxq(dS, 2), qt[:, sl, qs_]) * scale
            dsq = dsb8 if ds_block else (mxq(dS.bfloat16().float(), 3) if dq_bf16 else mxq(dS, 3))
            dq[:, sl, qs_] = torch.einsum("bhqk,bhkd->bhqd", dsq, kt[:, sl, : q0 + chunk]) * scale
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
    return dict(
        v8=v8,
        vsr=vsr,
        do8=do8,
        dos=dos,
        delta=delta,
        do8t=do8t,
        dots=dots,
        q8t=q8t,
        qts=qts,
        k8t=k8t,
        kts=kts,
    )
