###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""MXFP8 DeepSeek-V3 MLA attention on gfx1250: build, load and launch the HIP kernels of ``hip/``.

    out, lse, *mx_saved = attn_fwd(q, k, v, softmax_scale)    # q / k [b, s, h, 192], v [b, s, h, 128] bf16
    dq, dk, dv = attn_bwd(dout, q, k, v, out, *mx_saved, softmax_scale)

Recipe (benchmark/hk_mla_mx/PLAN.md): E4M3 data with E8M0 scales per 32 elements along every GEMM's reduction dim;
k centred by kbar (the mean of every 16th row: any constant shift of k is softmax-invariant); P quantized in the
kernels with fixed scales; dS quantized in the kernels along the reduction dim of dK (q) and of dQ (kv). The forward
returns the lse of the true logits; ``mx_saved`` = (lse of the centred logits, q8, qs, k8, ks, kbar) is what the
backward kernels take (``prepare_bwd`` makes it from a bf16 forward's q / k / lse).

Tensors are [b, s, h, d] with any b / s / h strides and a contiguous, 16-byte aligned d; outputs and gradients are
[b, s, h, d] views of [s, b, h, d] storage (Megatron's SBHD). Development package: the code objects are compiled at
first use with amdclang++ (``hsaco.find_compiler``) into ~/.cache/primus_turbo/gfx1250_mla_mx/<sha of hip/ and the
flags>/, one process at a time; no sha pins yet.
"""

import fcntl
import functools
import hashlib
import os
import subprocess
import tempfile
import threading
from pathlib import Path

import torch

from primus_turbo.hipkittens.gfx1250_mla import hiprt

D_QK, D_V = 192, 128
KBAR_STRIDE = 16
HIP = Path(__file__).resolve().parent / "hip"
SOURCES = ("k_mxq.hip", "k_mla_fwd_mx.hip", "k_mla_bwd_mx.hip")
FLAGS = (
    "-x",
    "hip",
    "--offload-arch=gfx1250",
    "--cuda-device-only",
    "--no-gpu-bundle-output",
    "-O3",
    "-std=c++20",
)

_ROWS_ARGS = (("X", "ptr"), ("X8", "ptr"), ("XS", "ptr"), ("kbar", "ptr"), ("Y", "ptr"), ("aux", "ptr"),
              ("XD", "ptr"), ("B", "i32"), ("S", "i32"), ("H", "i32"), ("sb", "i64"), ("ss", "i64"), ("sh", "i64"),
              ("yb", "i64"), ("ys", "i64"), ("yh", "i64"), ("mode", "i32"))  # fmt: skip
_COLS_ARGS = (("X", "ptr"), ("X8T", "ptr"), ("XTS", "ptr"), ("kbar", "ptr"), ("S", "i32"), ("H", "i32"),
              ("sb", "i64"), ("ss", "i64"), ("sh", "i64"), ("mode", "i32"))  # fmt: skip
# kernel -> (source, workgroup size, arguments in order)
KERNELS = {
    "k_mxq_kbar": (
        "k_mxq.hip",
        768,
        (
            ("K", "ptr"),
            ("kbar", "ptr"),
            ("S", "i32"),
            ("H", "i32"),
            ("sb", "i64"),
            ("ss", "i64"),
            ("sh", "i64"),
            ("stride", "i32"),
        ),
    ),  # fmt: skip
    "k_mxq_rows192": ("k_mxq.hip", 256, _ROWS_ARGS),
    "k_mxq_rows128": ("k_mxq.hip", 256, _ROWS_ARGS),
    "k_mxq_cols128": ("k_mxq.hip", 256, _COLS_ARGS),
    "k_mxq_cols192": ("k_mxq.hip", 256, _COLS_ARGS),
    "k_mla_fwd_mx": (
        "k_mla_fwd_mx.hip",
        128,
        (
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
        ),
    ),  # fmt: skip
    "k_dkdv_mx": (
        "k_mla_bwd_mx.hip",
        128,
        (
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
        ),
    ),  # fmt: skip
    "k_dq_mx": (
        "k_mla_bwd_mx.hip",
        128,
        (
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
        ),
    ),  # fmt: skip
}
_SIZE = {"ptr": 8, "i64": 8, "i32": 4, "f32": 4}
_LOCK = threading.Lock()
_MODULES = {}
_KERNELS = {}
_DUMMY = {}


def _meta(name):
    _, block, args = KERNELS[name]
    off, out = 0, []
    for a, kind in args:
        sz = _SIZE[kind]
        off = (off + sz - 1) // sz * sz
        out.append({"name": a, "offset": off, "size": sz, "kind": kind})
        off += sz
    return {"block": block, "kernarg_size": off, "args": out}


@functools.lru_cache(maxsize=None)
def _build_dir():
    h = hashlib.sha256()
    for f in sorted(p.name for p in HIP.iterdir()):
        h.update(f.encode())
        h.update((HIP / f).read_bytes())
    h.update(" ".join(FLAGS).encode())
    for root in (
        Path(os.path.expanduser("~")) / ".cache",
        Path(tempfile.gettempdir()) / f"primus_turbo_{os.getuid()}",
    ):
        d = root / "primus_turbo" / "gfx1250_mla_mx" / h.hexdigest()[:16]
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        if os.access(d, os.W_OK):
            return d
    raise RuntimeError("gfx1250_mla_mx: no writable cache directory for the code objects")


def _code_object(src):
    d = _build_dir()
    out = d / (Path(src).stem + ".hsaco")
    if out.is_file():
        return out
    from primus_turbo.hipkittens.gfx1250_mla import hsaco

    cxx = hsaco.find_compiler()
    if cxx is None:
        raise RuntimeError("gfx1250_mla_mx: no amdclang++ to build the MXFP8 MLA kernels")
    with open(d / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # one builder per cache directory (e.g. the ranks of a node)
        if not out.is_file():
            tmp = out.with_suffix(f".{os.getpid()}.tmp")
            r = subprocess.run(
                [cxx, *FLAGS, "-o", str(tmp), str(HIP / src)], capture_output=True, text=True, cwd=HIP
            )
            if r.returncode != 0:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"gfx1250_mla_mx: building {src} failed:\n{r.stderr[-4000:]}")
            os.replace(tmp, out)
    return out


@functools.lru_cache(maxsize=None)
def unavailable_reason():
    """None when every code object is built (building the missing ones now), else why not."""
    try:
        for src in SOURCES:
            _code_object(src)
    except (OSError, RuntimeError) as e:
        return str(e)
    return None


def kernel(name):
    k = _KERNELS.get(name)
    if k is None:
        with _LOCK:
            k = _KERNELS.get(name)
            if k is None:
                path = _code_object(KERNELS[name][0])
                mod = _MODULES.get(path)
                if mod is None:
                    mod = _MODULES[path] = hiprt.Module(path, hashlib.sha256(path.read_bytes()).hexdigest())
                k = _KERNELS[name] = hiprt.Kernel(mod, name, _meta(name))
    return k


def _launch(name, grid, values):
    kernel(name).call(grid, (KERNELS[name][1], 1, 1), values, torch.cuda.current_stream())


def _dummy(dev):
    t = _DUMMY.get(dev)
    if t is None:
        t = _DUMMY[dev] = torch.zeros(64, device=dev, dtype=torch.float32)
    return t


# ------------------------------------------------------------------------------------------------- quantization


def kbar_of(k, stride=KBAR_STRIDE):
    b, s, h, _ = k.shape
    kbar = torch.empty(b, h, D_QK, device=k.device, dtype=torch.float32)
    _launch("k_mxq_kbar", (b * h, 1, 1), [k, kbar, s, h, k.stride(0), k.stride(1), k.stride(2), stride])
    return kbar


def quant_rows(x, d, mode=0, kbar=None, y=None):
    """x [b, s, h, d] -> x8 [b, h, s, d] (MX-interleaved), xs [b, h, s, 8 | 4], aux [b, h, s] (mode & 6) or None."""
    b, s, h, _ = x.shape
    dev = x.device
    x8 = torch.empty(b, h, s, d, device=dev, dtype=torch.uint8)
    xs = torch.empty(b, h, s, 8 if d == 192 else 4, device=dev, dtype=torch.uint8)
    aux = torch.empty(b, h, s, device=dev, dtype=torch.float32) if mode & 6 else None
    yy = y if y is not None else x
    rows_per_block = 8 * 8 * (4 if d == 192 else 8)
    _launch(f"k_mxq_rows{d}", ((b * h * s + rows_per_block - 1) // rows_per_block, 1, 1),
            [x, x8, xs, kbar if kbar is not None else _dummy(dev), yy, aux if aux is not None else _dummy(dev),
             _dummy(dev), b, s, h, x.stride(0), x.stride(1), x.stride(2), yy.stride(0), yy.stride(1), yy.stride(2),
             mode])  # fmt: skip
    return x8, xs, aux


def quant_cols(x, d, kbar=None):
    """x [b, s, h, d] -> x8t [b, h, d, s] (s MX-interleaved per 128), xts [b, h, s / 128, d, 4]."""
    b, s, h, _ = x.shape
    dev = x.device
    x8t = torch.empty(b, h, d, s, device=dev, dtype=torch.uint8)
    xts = torch.empty(b, h, s // 128, d, 4, device=dev, dtype=torch.uint8)
    _launch(f"k_mxq_cols{d}", (s // 128, h, b),
            [x, x8t, xts, kbar if kbar is not None else _dummy(dev), s, h, x.stride(0), x.stride(1), x.stride(2),
             1 if kbar is not None else 0])  # fmt: skip
    return x8t, xts


def unsupported_reason(q, k, v, causal):
    if not causal:
        return "causal only"
    b, s, h, _ = q.shape
    if k.shape[1] != s:
        return "seqlen_q == seqlen_kv only"
    if s % 256 != 0:
        return f"seqlen {s} must be a multiple of 256"
    if any(t.stride(-1) != 1 or t.data_ptr() % 16 for t in (q, k, v)):
        return "q / k / v need a contiguous, 16-byte aligned head dim"
    return None


# ------------------------------------------------------------------------------------------------------ passes


def attn_fwd(q, k, v, softmax_scale):
    """q / k [b, s, h, 192], v [b, s, h, 128] bf16 -> (out [b, s, h, 128] bf16, lse [b, h, s] fp32 (natural log of
    the true logits), then the backward's operands: lse_c (of the centred logits), q8, qs, k8, ks, kbar)."""
    b, s, h, _ = q.shape
    dev = q.device
    kbar = kbar_of(k)
    q8, qs, qk = quant_rows(q, D_QK, mode=2, kbar=kbar)
    k8, ks, _ = quant_rows(k, D_QK, mode=1, kbar=kbar)
    v8t, vs = quant_cols(v, D_V)
    out = torch.empty(s, b, h, D_V, device=dev, dtype=torch.bfloat16).transpose(0, 1)
    lse_c = torch.empty(b, h, s, device=dev, dtype=torch.float32)
    _launch(
        "k_mla_fwd_mx",
        (s // 256, h, b),
        [q8, qs, k8, ks, v8t, vs, out, lse_c, float(softmax_scale), s, h, b, out.stride(0), out.stride(1)],
    )
    lse = torch.add(
        lse_c, qk, alpha=float(softmax_scale)
    )  # the true logits: q . k = q . (k - kbar) + q . kbar
    return out, lse, lse_c, q8, qs, k8, ks, kbar


def prepare_bwd(q, k, lse, softmax_scale):
    """A bf16 forward's q / k / lse -> the backward's (lse_c, q8, qs, k8, ks, kbar)."""
    kbar = kbar_of(k)
    q8, qs, qk = quant_rows(q, D_QK, mode=2, kbar=kbar)
    k8, ks, _ = quant_rows(k, D_QK, mode=1, kbar=kbar)
    lse_c = torch.add(lse.float(), qk.view(lse.shape), alpha=-float(softmax_scale))
    return lse_c, q8, qs, k8, ks, kbar


def attn_bwd(dout, q, k, v, out, lse_c, q8, qs, k8, ks, kbar, softmax_scale):
    """dq, dk [b, s, h, 192], dv [b, s, h, 128] bf16 from dout and the forward's tensors."""
    b, s, h, _ = q.shape
    dev = q.device
    if dout.stride(-1) != 1 or dout.data_ptr() % 16:
        dout = dout.contiguous()
    v8, vsr, _ = quant_rows(v, D_V)
    do8, dos, delta = quant_rows(dout, D_V, mode=4, y=out)
    do8t, dots = quant_cols(dout, D_V)
    q8t, qts = quant_cols(q, D_QK)
    k8t, kts = quant_cols(k, D_QK, kbar=kbar)
    dq = torch.empty(s, b, h, D_QK, device=dev, dtype=torch.bfloat16).transpose(0, 1)
    dk = torch.empty(s, b, h, D_QK, device=dev, dtype=torch.bfloat16).transpose(0, 1)
    dv = torch.empty(s, b, h, D_V, device=dev, dtype=torch.bfloat16).transpose(0, 1)
    sc = float(softmax_scale)
    _launch("k_dkdv_mx", (s // 128, h, b),
            [q8, qs, k8, ks, v8, vsr, do8, dos, do8t, dots, q8t, qts, lse_c, delta, dk, dv, sc, s, h, b,
             dk.stride(0), dk.stride(1), dv.stride(0), dv.stride(1)])  # fmt: skip
    _launch("k_dq_mx", (s // 256, h, b),
            [q8, qs, k8, ks, v8, vsr, do8, dos, k8t, kts, lse_c, delta, dq, sc, s, h, b, dq.stride(0),
             dq.stride(1)])  # fmt: skip
    return dq, dk, dv
