###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""MXFP8 DeepSeek-V3 MLA attention on gfx1250: build, load and launch the HIP kernels of ``hip/``.

    out, lse, *saved = attn_fwd(q, k, v, softmax_scale, for_mx_bwd=True)  # q / k [b, s, h, 192], v [b, s, h, 128]
    dq, dk, dv = attn_bwd(dout, out, *saved, softmax_scale)

Recipe "m4" (benchmark/hk_mla_mx/PLAN.md, LOG.md): E4M3 data with E8M0 scales per 32 elements along every GEMM's
reduction dim. k is centred by kbar (the mean of every 16th row: softmax-invariant). Q8 rows are quantized from
q * softmax_scale * log2(e), so the kernels' logits are in log2 units. Q / K row scales use layout R2
([s0..s3, 0, 0, s4, s5]): the kernels run rope as a WMMA over d 64..191 with d 64..127 silenced. P is quantized in the
kernels with fixed scales; dS along the reduction dim of dK (q) and of dQ (kv).

Kernels: forward rung m7 (``k_mla_fwd_mx7.hip``: m6 with two q blocks per workgroup; branch-free softmax with per-lane integer shifts, O accumulated against
a fixed per-row reference through the PV WMMAs' E8M0 B scale; the first tile's exp2 runs unshifted, so the centred
logits of a row's first 128 keys must stay under 126 in log2 units, i.e. 87 nats), backward rung b3
(``k_mla_bwd_mx3.hip``: k_dkdv_mx, k_dq_mx); all in XCD-local workgroup order with the TDM ops spread over the waves
and the outputs written through LDS by TDM stores (benchmark/hk_mla_mx/LOG.md, 2026-11-05 (b)),
quantization ``k_mxq.hip`` (kbar; rows + cols of a tensor in one read). With ``for_mx_bwd`` the forward quantizes
q / k / v once for both passes: rows for itself, cols for the backward. ``saved`` = (lse of the centred logits, q8, qs,
k8, ks, kbar, q8t, qts, k8t, kts, v8, vsr), and the backward quantizes only dout (rows + cols + delta). Without it
the forward makes only its own operands and returns ``saved = ()``. ``prepare_bwd`` makes ``saved`` from a bf16
forward's q / k / v / lse.

Tensors are [b, s, h, d] with any b / s / h strides and a contiguous, 16-byte aligned d. Outputs and gradients are
[b, s, h, d] views of [s, b, h, d] storage (Megatron's SBHD). Development package: the code objects are compiled at
first use with amdclang++ (``hsaco.find_compiler``) into ~/.cache/primus_turbo/gfx1250_mla_mx/<sha of hip/ and the
flags>/, one process at a time; no sha pins yet.
"""

import fcntl
import functools
import hashlib
import math
import os
import subprocess
import tempfile
import threading
from pathlib import Path

import torch

from primus_turbo.hipkittens.gfx1250_mla import hiprt

D_QK, D_V = 192, 128
KBAR_STRIDE = 16
LOG2E = 1.4426950408889634
HIP = Path(__file__).resolve().parent / "hip"
SOURCES = ("k_mxq.hip", "k_mla_fwd_mx7.hip", "k_mla_bwd_mx3.hip")
# expert scheduling mode (the compiler places the dependency waits, as every bf16 HK kernel) and register classes
# before globalness (DS addresses and WMMA scale operands must be in v0..v255)
FLAGS = ("-x", "hip", "--offload-arch=gfx1250", "--cuda-device-only", "--no-gpu-bundle-output", "-O3", "-std=c++20",
         "-mllvm", "-amdgpu-expert-scheduling-mode", "-mllvm", "-greedy-regclass-priority-trumps-globalness")  # fmt: skip
DKDV_KV_ROWS = 128  # kv rows per k_dkdv_mx workgroup (4 waves x 2 kv16 tiles)

_ROWS_ARGS = (("X", "ptr"), ("X8", "ptr"), ("XS", "ptr"), ("kbar", "ptr"), ("Y", "ptr"), ("aux", "ptr"),
              ("XD", "ptr"), ("B", "i32"), ("S", "i32"), ("H", "i32"), ("sb", "i64"), ("ss", "i64"), ("sh", "i64"),
              ("yb", "i64"), ("ys", "i64"), ("yh", "i64"), ("mode", "i32"), ("pre", "f32"))  # fmt: skip
_COLS_ARGS = (("X", "ptr"), ("X8T", "ptr"), ("XTS", "ptr"), ("kbar", "ptr"), ("S", "i32"), ("H", "i32"),
              ("sb", "i64"), ("ss", "i64"), ("sh", "i64"), ("mode", "i32"))  # fmt: skip
_RC_ARGS = (("X", "ptr"), ("X8", "ptr"), ("XS", "ptr"), ("X8T", "ptr"), ("XTS", "ptr"), ("kbar", "ptr"), ("Y", "ptr"),
            ("aux", "ptr"), ("S", "i32"), ("H", "i32"), ("sb", "i64"), ("ss", "i64"), ("sh", "i64"), ("yb", "i64"),
            ("ys", "i64"), ("yh", "i64"), ("mode", "i32"), ("pre", "f32"))  # fmt: skip
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
    "k_mxq_rc192": ("k_mxq.hip", 256, _RC_ARGS),
    "k_mxq_rc128": ("k_mxq.hip", 256, _RC_ARGS),
    "k_mla_fwd_mx": (
        "k_mla_fwd_mx7.hip",
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
        "k_mla_bwd_mx3.hip",
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
            ("LSE2", "ptr"),
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
        "k_mla_bwd_mx3.hip",
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
            ("LSE2", "ptr"),
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
# k_mxq modes (k_mxq.hip): 1 subtract kbar, 2 aux = x . kbar, 4 aux = x . Y, 16 scale layout R2, 32 quantize x * pre
MODE_Q, MODE_K, MODE_DO = 2 | 16 | 32, 1 | 16, 4
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


def quant_rows(x, d, mode=0, kbar=None, y=None, pre=1.0):
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
             mode, float(pre)])  # fmt: skip
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


def quant_rc(x, d, mode=0, kbar=None, y=None, pre=1.0):
    """Rows and cols of x [b, s, h, d] in one read: (x8, xs, aux | None, x8t, xts) as quant_rows / quant_cols (cols:
    kbar subtracted when mode & 1, never pre-scaled)."""
    b, s, h, _ = x.shape
    dev = x.device
    x8 = torch.empty(b, h, s, d, device=dev, dtype=torch.uint8)
    xs = torch.empty(b, h, s, 8 if d == 192 else 4, device=dev, dtype=torch.uint8)
    x8t = torch.empty(b, h, d, s, device=dev, dtype=torch.uint8)
    xts = torch.empty(b, h, s // 128, d, 4, device=dev, dtype=torch.uint8)
    aux = torch.empty(b, h, s, device=dev, dtype=torch.float32) if mode & 6 else None
    yy = y if y is not None else x
    _launch(f"k_mxq_rc{d}", (s // 128, h, b),
            [x, x8, xs, x8t, xts, kbar if kbar is not None else _dummy(dev), yy,
             aux if aux is not None else _dummy(dev), s, h, x.stride(0), x.stride(1), x.stride(2), yy.stride(0),
             yy.stride(1), yy.stride(2), mode, float(pre)])  # fmt: skip
    return x8, xs, aux, x8t, xts


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


def _quant_qkv(q, k, v, softmax_scale, both):
    """The forward's operands (and with ``both`` the backward's: one read per tensor) -> (q8, qs, qk, k8, ks, v8t, vs,
    kbar, q8t, qts, k8t, kts, v8, vsr) (the last six None without ``both``)."""
    pre = float(softmax_scale) * LOG2E
    kbar = kbar_of(k)
    if both:
        q8, qs, qk, q8t, qts = quant_rc(q, D_QK, mode=MODE_Q, kbar=kbar, pre=pre)
        k8, ks, _, k8t, kts = quant_rc(k, D_QK, mode=MODE_K, kbar=kbar)
        v8, vsr, _, v8t, vs = quant_rc(v, D_V)
        return q8, qs, qk, k8, ks, v8t, vs, kbar, q8t, qts, k8t, kts, v8, vsr
    q8, qs, qk = quant_rows(q, D_QK, mode=MODE_Q, kbar=kbar, pre=pre)
    k8, ks, _ = quant_rows(k, D_QK, mode=MODE_K, kbar=kbar)
    v8t, vs = quant_cols(v, D_V)
    return q8, qs, qk, k8, ks, v8t, vs, kbar, None, None, None, None, None, None


def attn_fwd(q, k, v, softmax_scale, for_mx_bwd=True):
    """q / k [b, s, h, 192], v [b, s, h, 128] bf16 -> (out [b, s, h, 128] bf16, lse [b, h, s] fp32 (natural log of the
    true logits), *saved): saved = the backward's operands (module docstring) with ``for_mx_bwd``, else ()."""
    b, s, h, _ = q.shape
    q8, qs, qk, k8, ks, v8t, vs, kbar, q8t, qts, k8t, kts, v8, vsr = _quant_qkv(
        q, k, v, softmax_scale, for_mx_bwd
    )
    out = torch.empty(s, b, h, D_V, device=q.device, dtype=torch.bfloat16).transpose(0, 1)
    lse_c = torch.empty(b, h, s, device=q.device, dtype=torch.float32)
    nqb = s // 256
    paired = nqb % 2 == 0 and (b * h) % 8 == 0  # the kernel's pair_mode(): q blocks nqb - 1 - p and p per workgroup
    _launch(
        "k_mla_fwd_mx",
        (nqb // 2 if paired else nqb, h, b),
        [q8, qs, k8, ks, v8t, vs, out, lse_c, float(softmax_scale), s, h, b, out.stride(0), out.stride(1)],
    )
    lse = torch.add(
        lse_c, qk, alpha=float(softmax_scale)
    )  # the true logits: q . k = q . (k - kbar) + q . kbar
    if not for_mx_bwd:
        return (out, lse)
    return (out, lse, lse_c, q8, qs, k8, ks, kbar, q8t, qts, k8t, kts, v8, vsr)


def prepare_bwd(q, k, v, lse, softmax_scale):
    """A bf16 forward's q / k / v / lse -> the backward's saved operands (as attn_fwd's)."""
    q8, qs, qk, k8, ks, _, _, kbar, q8t, qts, k8t, kts, v8, vsr = _quant_qkv(q, k, v, softmax_scale, True)
    lse_c = torch.add(lse.float(), qk.view(lse.shape), alpha=-float(softmax_scale))
    return (lse_c, q8, qs, k8, ks, kbar, q8t, qts, k8t, kts, v8, vsr)


def attn_bwd(dout, out, lse_c, q8, qs, k8, ks, kbar, q8t, qts, k8t, kts, v8, vsr, softmax_scale):
    """dq, dk [b, s, h, 192], dv [b, s, h, 128] bf16 from dout [b, s, h, 128], the forward's out and saved operands."""
    b, h, s, _ = q8.shape
    dev = q8.device
    if dout.stride(-1) != 1 or dout.data_ptr() % 16:
        dout = dout.contiguous()
    do8, dos, delta, do8t, dots = quant_rc(dout, D_V, mode=MODE_DO, y=out)
    lse2 = lse_c * LOG2E
    dq = torch.empty(s, b, h, D_QK, device=dev, dtype=torch.bfloat16).transpose(0, 1)
    dk = torch.empty(s, b, h, D_QK, device=dev, dtype=torch.bfloat16).transpose(0, 1)
    dv = torch.empty(s, b, h, D_V, device=dev, dtype=torch.bfloat16).transpose(0, 1)
    sc = float(softmax_scale)
    _launch("k_dkdv_mx", (math.ceil(s / DKDV_KV_ROWS), h, b),
            [q8, qs, k8, ks, v8, vsr, do8, dos, do8t, dots, q8t, qts, lse2, delta, dk, dv, sc, s, h, b,
             dk.stride(0), dk.stride(1), dv.stride(0), dv.stride(1)])  # fmt: skip
    _launch("k_dq_mx", (s // 256, h, b),
            [q8, qs, k8, ks, v8, vsr, do8, dos, k8t, kts, lse2, delta, dq, sc, s, h, b, dq.stride(0),
             dq.stride(1)])  # fmt: skip
    return dq, dk, dv
