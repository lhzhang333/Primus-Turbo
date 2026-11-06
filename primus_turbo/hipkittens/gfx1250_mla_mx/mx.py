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
logits of a row's first 128 keys must stay under 126 in log2 units, i.e. 87 nats), backward rung b5
(``k_mla_bwd_mx5.hip``: k_dkdv_mx software-pipelined across steps, k_dq_mx); all in XCD-local workgroup order with the
TDM ops spread over the waves
and the outputs written through LDS by TDM stores (benchmark/hk_mla_mx/LOG.md, 2026-11-05 (b)),
quantization ``k_mxq.hip`` (kbar; rows + cols of a tensor in one read: ``k_mxq_rct``, the tile in and the four
outputs out by TDM). With ``for_mx_bwd`` the forward quantizes
q / k / v once for both passes: rows for itself, cols for the backward. ``saved`` = (lse2: the log2 of the centred
logits' sums (the kernels' units), q8, qs, k8, ks, kbar, q8t, qts, k8t, kts, v8, vsr), and the backward quantizes only
dout (rows + cols + delta). Without it
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
import struct
import subprocess
import tempfile
import threading
import warnings
from pathlib import Path

import torch

from primus_turbo.common import envs
from primus_turbo.hipkittens.gfx1250_mla import hiprt
from primus_turbo.hipkittens.gfx1250_mla_mx import asm_resched

D_QK, D_V = 192, 128
D_NOPE = 128  # q / k columns before the rope part
KBAR_STRIDE = 16
LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453
HIP = Path(__file__).resolve().parent / "hip"
SOURCES = ("k_mxq.hip", "k_mla_fwd_mx7.hip", "k_mla_bwd_mx5.hip")
# expert scheduling mode (the compiler places the dependency waits, as every bf16 HK kernel), register classes
# before globalness (DS addresses and WMMA scale operands must be in v0..v255), the GCN max-ILP machine scheduler
# (dq -3.7%, forward -2%, the rest neutral)
FLAGS = ("-x", "hip", "--offload-arch=gfx1250", "--cuda-device-only", "--no-gpu-bundle-output", "-O3", "-std=c++20",
         "-mllvm", "-amdgpu-expert-scheduling-mode", "-mllvm", "-greedy-regclass-priority-trumps-globalness",
         "-mllvm", "-misched=gcn-max-ilp")  # fmt: skip
# per-source extra flags: the quantization kernels without VOPD (LLVM re-materialized a VOPD constant into an SGPR
# before every dual op: k -7%, the others -1..2%)
SRC_FLAGS = {"k_mxq.hip": ("-mllvm", "-amdgpu-enable-vopd=0", "-fno-slp-vectorize")}  # no SLP: fused-RoPE q / dout -3%
# main loops re-scheduled after register allocation (asm_resched: a list schedule of LLVM's loop body, emitted through
# gfx1250_mla's asmkit, verified against the code object): the forward's tile loop, -7% (bit-exact). The dK / dV loop
# is LDS-bound and slower re-scheduled: LLVM's order. PRIMUS_TURBO_MLA_MX_RESCHED=0: plain builds; on any failure
# the source is built plainly (a warning)
RESCHED = {"k_mla_fwd_mx7.hip": ("k_mla_fwd_mx",)}
DKDV_KV_ROWS = 128  # kv rows per k_dkdv_mx workgroup (4 waves x 2 kv16 tiles)

_ROWS_ARGS = (("X", "ptr"), ("X8", "ptr"), ("XS", "ptr"), ("kbar", "ptr"), ("Y", "ptr"), ("aux", "ptr"),
              ("XD", "ptr"), ("B", "i32"), ("S", "i32"), ("H", "i32"), ("sb", "i64"), ("ss", "i64"), ("sh", "i64"),
              ("yb", "i64"), ("ys", "i64"), ("yh", "i64"), ("mode", "i32"), ("pre", "f32"))  # fmt: skip
_COLS_ARGS = (("X", "ptr"), ("X8T", "ptr"), ("XTS", "ptr"), ("kbar", "ptr"), ("S", "i32"), ("H", "i32"),
              ("sb", "i64"), ("ss", "i64"), ("sh", "i64"), ("mode", "i32"))  # fmt: skip
_RC_ARGS = (("X", "ptr"), ("X8", "ptr"), ("XS", "ptr"), ("X8T", "ptr"), ("XTS", "ptr"), ("kbar", "ptr"), ("Y", "ptr"),
            ("aux", "ptr"), ("S", "i32"), ("H", "i32"), ("sb", "i64"), ("ss", "i64"), ("sh", "i64"), ("yb", "i64"),
            ("ys", "i64"), ("yh", "i64"), ("mode", "i32"), ("pre", "f32"))  # fmt: skip
_RCTR_ARGS = (("X", "ptr"), ("X8", "ptr"), ("XS", "ptr"), ("X8T", "ptr"), ("XTS", "ptr"), ("kbar", "ptr"),
              ("aux", "ptr"), ("S", "i32"), ("H", "i32"), ("sb", "i64"), ("ss", "i64"), ("sh", "i64"), ("PE", "ptr"),
              ("pb", "i64"), ("ps", "i64"), ("ph", "i64"), ("COS", "ptr"), ("SIN", "ptr"), ("mode", "i32"),
              ("pre", "f32"))  # fmt: skip
_KBR_ARGS = (("X", "ptr"), ("kbar", "ptr"), ("S", "i32"), ("H", "i32"), ("sb", "i64"), ("ss", "i64"), ("sh", "i64"),
             ("PE", "ptr"), ("pb", "i64"), ("ps", "i64"), ("ph", "i64"), ("COS", "ptr"), ("SIN", "ptr"),
             ("stride", "i32"))  # fmt: skip
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
    "k_mxq_rct192": ("k_mxq.hip", 256, _RC_ARGS),
    "k_mxq_rct128": ("k_mxq.hip", 256, _RC_ARGS),
    "k_mxq_rctr192": ("k_mxq.hip", 256, _RCTR_ARGS),
    "k_mxq_kbar_rope": ("k_mxq.hip", 768, _KBR_ARGS),
    "k_rope_bwd_kpe": ("k_mxq.hip", 256, (("GH", "ptr"), ("GP", "ptr"), ("B", "i32"), ("S", "i32"), ("H", "i32"),
                                          ("hb", "i64"), ("hs", "i64"), ("hh", "i64"), ("pb", "i64"), ("ps", "i64"),
                                          ("COS", "ptr"), ("SIN", "ptr"))),
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
        "k_mla_bwd_mx5.hip",
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
        "k_mla_bwd_mx5.hip",
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
KERNELS["k_dkdv_mxr"] = (KERNELS["k_dkdv_mx"][0], 128,
                         KERNELS["k_dkdv_mx"][2] + (("DKP", "ptr"), ("dkp_sb", "i64"), ("dkp_ss", "i64")))
# the fused-RoPE path's dQ kernel: dQ's rope columns through the inverse rotation (COS / SIN [S, 64] bf16)
KERNELS["k_dq_mxr"] = (KERNELS["k_dq_mx"][0], 128, KERNELS["k_dq_mx"][2] + (("COS", "ptr"), ("SIN", "ptr")))
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


def _resched_on():
    return bool(envs.PRIMUS_TURBO_MLA_MX_RESCHED) and asm_resched.available()


@functools.lru_cache(maxsize=None)
def _build_dir():
    h = hashlib.sha256()
    for f in sorted(p.name for p in HIP.iterdir()):
        h.update(f.encode())
        h.update((HIP / f).read_bytes())
    h.update(" ".join(FLAGS).encode())
    h.update(repr(sorted(SRC_FLAGS.items())).encode())
    if _resched_on():
        h.update(repr(sorted(RESCHED.items())).encode())
        for f in (Path(__file__).with_name("asm_resched.py"), *sorted(asm_resched._ASMKIT.glob("*.py"))):
            if f.is_file():
                h.update(f.read_bytes())
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


# kernels whose start prefetches their own code (PF_CODE in the source): the bytes from the entry the prefetches may
# reach (s_prefetch_inst past the code object's end page-faults); checked after the build, else rebuilt with PF_CODE 0
PREFETCH = {"k_mla_bwd_mx5.hip": {"k_dq_mx": 45 * 1024, "k_dq_mxr": 45 * 1024}}


def _elf_funcs(path):
    """(the end address of the .text section, {function symbol: address}) of an ELF64 code object."""
    b = Path(path).read_bytes()
    (shoff,) = struct.unpack_from("<Q", b, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", b, 0x3A)
    secs = [struct.unpack_from("<IIQQQQIIQQ", b, shoff + i * shentsize) for i in range(shnum)]

    def name(off, strsec):
        o = strsec[4] + off
        return b[o : b.index(b"\0", o)].decode()

    text_end = max((sh[3] + sh[5] for sh in secs if name(sh[0], secs[shstrndx]) == ".text"), default=0)
    funcs = {}
    for sh in secs:
        if sh[1] == 2:  # SHT_SYMTAB
            for k in range(sh[5] // 24):
                st_name, st_info, _, _, st_value, _ = struct.unpack_from("<IBBHQQ", b, sh[4] + 24 * k)
                if st_info & 0xF == 2:  # STT_FUNC
                    funcs[name(st_name, secs[sh[6]])] = st_value
    return text_end, funcs


def _prefetch_ok(src, path):
    need = PREFETCH.get(src)
    if not need:
        return True
    text_end, funcs = _elf_funcs(path)
    return all(k in funcs and funcs[k] + n <= text_end for k, n in need.items())


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
        if not out.is_file() and src in RESCHED and _resched_on():
            tmp = out.with_suffix(f".{os.getpid()}.tmp")
            try:
                with tempfile.TemporaryDirectory(prefix="mx_resched_") as td:
                    asm_resched.build(cxx, HIP / src, tmp, [*FLAGS, *SRC_FLAGS.get(src, ())], RESCHED[src], HIP, td)
                os.replace(tmp, out)
            except Exception as e:  # noqa: BLE001 - any failure: the plain build below
                tmp.unlink(missing_ok=True)
                warnings.warn(f"gfx1250_mla_mx: re-scheduling {src} failed ({str(e)[:300]}); plain build")
        for extra in ((), ("-DPF_CODE=0",)):
            if out.is_file():
                break
            tmp = out.with_suffix(f".{os.getpid()}.tmp")
            r = subprocess.run(
                [cxx, *FLAGS, *SRC_FLAGS.get(src, ()), *extra, "-o", str(tmp), str(HIP / src)], capture_output=True,
                text=True, cwd=HIP,
            )
            if r.returncode != 0:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"gfx1250_mla_mx: building {src} failed:\n{r.stderr[-4000:]}")
            if not extra and not _prefetch_ok(src, tmp):
                tmp.unlink(missing_ok=True)
                warnings.warn(f"gfx1250_mla_mx: {src}: a code prefetch would pass the code object's end; PF_CODE 0")
                continue
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
    _launch(f"k_mxq_rct{d}", (s // 128, h, b),
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
    lse2 = torch.empty(b, h, s, device=q.device, dtype=torch.float32)  # log2 units of the centred logits (the kernel's)
    nqb = s // 256
    paired = nqb % 2 == 0 and (b * h) % 8 == 0  # the kernel's pair_mode(): q blocks nqb - 1 - p and p per workgroup
    _launch(
        "k_mla_fwd_mx",
        (nqb // 2 if paired else nqb, h, b),
        [q8, qs, k8, ks, v8t, vs, out, lse2, float(softmax_scale), s, h, b, out.stride(0), out.stride(1)],
    )
    # natural log of the true logits: q . k = q . (k - kbar) + q . kbar
    lse = torch.add(lse2 * LN2, qk, alpha=float(softmax_scale))
    if not for_mx_bwd:
        return (out, lse)
    return (out, lse, lse2, q8, qs, k8, ks, kbar, q8t, qts, k8t, kts, v8, vsr)


def prepare_bwd(q, k, v, lse, softmax_scale):
    """A bf16 forward's q / k / v / lse -> the backward's saved operands (as attn_fwd's)."""
    q8, qs, qk, k8, ks, _, _, kbar, q8t, qts, k8t, kts, v8, vsr = _quant_qkv(q, k, v, softmax_scale, True)
    lse2 = torch.add(lse.float(), qk.view(lse.shape), alpha=-float(softmax_scale)) * LOG2E
    return (lse2, q8, qs, k8, ks, kbar, q8t, qts, k8t, kts, v8, vsr)


def attn_bwd(dout, out, lse2, q8, qs, k8, ks, kbar, q8t, qts, k8t, kts, v8, vsr, softmax_scale):
    """dq, dk [b, s, h, 192], dv [b, s, h, 128] bf16 from dout [b, s, h, 128], the forward's out and saved operands."""
    b, h, s, _ = q8.shape
    dev = q8.device
    if dout.stride(-1) != 1 or dout.data_ptr() % 16:
        dout = dout.contiguous()
    do8, dos, delta, do8t, dots = quant_rc(dout, D_V, mode=MODE_DO, y=out)
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


# ------------------------------------------------------------------------------------- the fused-RoPE path
# MLA's rotary (Megatron fused_apply_mla_rope_for_q / _for_kv: interleaved pairs of d 128..191 in, halves out) folded
# into the quantization and the backward's epilogues: no bf16 query / key / value are materialized.


def rope_unsupported_reason(q, kv, kpe, cos, sin, causal):
    """Why the fused-RoPE path cannot take q [b, s, h, 192], kv [b, s, h, 256], kpe [b, s, 1, 64], cos / sin [>= s, 64]
    (BSHD views), or None."""
    if not causal:
        return "causal only"
    b, s, h, dq = q.shape
    if dq != D_QK or kv.shape != (b, s, h, 256) or kpe.shape[-1] != 64 or kpe.shape[:2] != (b, s):
        return "MLA shapes only: q [.., 192], kv [.., 256], k_pe [.., 64]"
    if s % 256 != 0:
        return f"seqlen {s} must be a multiple of 256"
    if cos.dtype != torch.bfloat16 or sin.dtype != torch.bfloat16 or cos.numel() < s * 64 or sin.numel() < s * 64:
        return "cos / sin: bf16, >= seqlen rows of 64"
    if not (cos.is_contiguous() and sin.is_contiguous()) or cos.data_ptr() % 16 or sin.data_ptr() % 16:
        return "cos / sin must be contiguous and 16-byte aligned"
    for t in (q, kv, kpe):
        if t.stride(-1) != 1 or t.data_ptr() % 16 or any(x % 8 for x in t.stride()[:-1]):
            return "q / kv / k_pe need a contiguous last dim and 16-byte aligned rows"
    if (h != 1 and kv.stride(2) != 256) or (h != 1 and q.stride(2) != D_QK):
        return "q / kv heads must be packed (head strides 192 / 256)"
    return None


def kbar_rope(kv, kpe, cos, sin, stride=KBAR_STRIDE):
    b, s, h, _ = kv.shape
    kbar = torch.empty(b, h, D_QK, device=kv.device, dtype=torch.float32)
    _launch("k_mxq_kbar_rope", (b * h, 1, 1),
            [kv, kbar, s, h, kv.stride(0), kv.stride(1), kv.stride(2), kpe, kpe.stride(0), kpe.stride(1), 0, cos, sin,
             stride])  # fmt: skip
    return kbar


def quant_rope(x, pe, pe_head_stride, cos, sin, mode=0, kbar=None, pre=1.0):
    """k_mxq_rctr192: rows + cols of [x[..., :128] | rope(pe)] -> (x8, xs, aux | None, x8t, xts) as quant_rc(.., 192)."""
    b, s, h, _ = x.shape
    dev = x.device
    x8 = torch.empty(b, h, s, D_QK, device=dev, dtype=torch.uint8)
    xs = torch.empty(b, h, s, 8, device=dev, dtype=torch.uint8)
    x8t = torch.empty(b, h, D_QK, s, device=dev, dtype=torch.uint8)
    xts = torch.empty(b, h, s // 128, D_QK, 4, device=dev, dtype=torch.uint8)
    aux = torch.empty(b, h, s, device=dev, dtype=torch.float32) if mode & 6 else None
    _launch("k_mxq_rctr192", (s // 128, h, b),
            [x, x8, xs, x8t, xts, kbar if kbar is not None else _dummy(dev), aux if aux is not None else _dummy(dev), s,
             h, x.stride(0), x.stride(1), x.stride(2), pe, pe.stride(0), pe.stride(1), pe_head_stride, cos, sin, mode,
             float(pre)])  # fmt: skip
    return x8, xs, aux, x8t, xts


def attn_fwd_rope(q, kv, kpe, cos, sin, softmax_scale, need_lse=True):
    """q [b, s, h, 192] (d 128..191 before RoPE), kv [b, s, h, 256] ([k nope | v]), kpe [b, s, 1, 64] (before RoPE,
    shared by the heads), cos / sin [>= s, 64] bf16 -> (out [b, s, h, 128] bf16 (BSHD view of SBHD storage), lse [b, h,
    s], *saved for attn_bwd_rope)."""
    b, s, h, _ = q.shape
    pre = float(softmax_scale) * LOG2E
    kbar = kbar_rope(kv, kpe, cos, sin)
    # q . kbar (the aux of mode 2) only forms the API lse: without need_lse the q quantization skips the row dots
    mode_q = MODE_Q if need_lse else MODE_Q & ~2
    q8, qs, qk, q8t, qts = quant_rope(q, q[..., D_NOPE:], q.stride(2), cos, sin, mode=mode_q, kbar=kbar, pre=pre)
    k8, ks, _, k8t, kts = quant_rope(kv, kpe, 0, cos, sin, mode=MODE_K, kbar=kbar)
    v8, vsr, _, v8t, vs = quant_rc(kv[..., D_NOPE:], D_V)
    out = torch.empty(s, b, h, D_V, device=q.device, dtype=torch.bfloat16).transpose(0, 1)
    lse2 = torch.empty(b, h, s, device=q.device, dtype=torch.float32)  # log2 units of the centred logits
    nqb = s // 256
    paired = nqb % 2 == 0 and (b * h) % 8 == 0
    _launch("k_mla_fwd_mx", (nqb // 2 if paired else nqb, h, b),
            [q8, qs, k8, ks, v8t, vs, out, lse2, float(softmax_scale), s, h, b, out.stride(0), out.stride(1)])
    lse = torch.add(lse2 * LN2, qk, alpha=float(softmax_scale)) if need_lse else None
    return (out, lse, lse2, q8, qs, k8, ks, kbar, q8t, qts, k8t, kts, v8, vsr)


def attn_bwd_rope(dout, out, lse2, q8, qs, k8, ks, kbar, q8t, qts, k8t, kts, v8, vsr, cos, sin, softmax_scale):
    """-> (dq [b, s, h, 192] (the gradient of q before RoPE), dkv [b, s, h, 256], dkpe [b, s, 64]), BSHD views of SBHD
    storage."""
    b, h, s, _ = q8.shape
    dev = q8.device
    if dout.stride(-1) != 1 or dout.data_ptr() % 16:
        dout = dout.contiguous()
    do8, dos, delta, do8t, dots = quant_rc(dout, D_V, mode=MODE_DO, y=out)
    dq = torch.empty(s, b, h, D_QK, device=dev, dtype=torch.bfloat16).transpose(0, 1)
    dkv = torch.empty(s, b, h, 256, device=dev, dtype=torch.bfloat16).transpose(0, 1)
    dkph = torch.empty(s, b, h, 64, device=dev, dtype=torch.bfloat16).transpose(0, 1)
    dk, dv = dkv[..., :D_NOPE], dkv[..., D_NOPE:]
    sc = float(softmax_scale)
    _launch("k_dkdv_mxr", (math.ceil(s / DKDV_KV_ROWS), h, b),
            [q8, qs, k8, ks, v8, vsr, do8, dos, do8t, dots, q8t, qts, lse2, delta, dk, dv, sc, s, h, b, dk.stride(0),
             dk.stride(1), dv.stride(0), dv.stride(1), dkph, dkph.stride(0), dkph.stride(1)])  # fmt: skip
    _launch("k_dq_mxr", (s // 256, h, b),
            [q8, qs, k8, ks, v8, vsr, do8, dos, k8t, kts, lse2, delta, dq, sc, s, h, b, dq.stride(0), dq.stride(1),
             cos, sin])  # fmt: skip
    dkpe = torch.empty(s, b, 64, device=dev, dtype=torch.bfloat16).transpose(0, 1)
    _launch("k_rope_bwd_kpe", (b * s, 1, 1),
            [dkph, dkpe, b, s, h, dkph.stride(0), dkph.stride(1), dkph.stride(2), dkpe.stride(0), dkpe.stride(1), cos,
             sin])  # fmt: skip
    return dq, dkv, dkpe


class _MLARopeAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, kpe, cos, sin, softmax_scale):
        qb, kvb, kpeb = q.transpose(0, 1), kv.transpose(0, 1), kpe.transpose(0, 1)
        cos2, sin2 = cos.reshape(-1, 64), sin.reshape(-1, 64)
        res = attn_fwd_rope(qb, kvb, kpeb.reshape(kpeb.shape[0], kpeb.shape[1], 1, 64), cos2, sin2, softmax_scale,
                            need_lse=False)
        out = res[0]
        ctx.save_for_backward(out, cos2, sin2, *res[2:])
        ctx.softmax_scale = softmax_scale
        ctx.kpe_shape = kpe.shape
        return out.transpose(0, 1)

    @staticmethod
    def backward(ctx, dout):
        out, cos2, sin2, *saved = ctx.saved_tensors
        dq, dkv, dkpe = attn_bwd_rope(dout.transpose(0, 1), out, *saved, cos2, sin2, ctx.softmax_scale)
        return dq.transpose(0, 1), dkv.transpose(0, 1), dkpe.transpose(0, 1).reshape(ctx.kpe_shape), None, None, None


def mla_rope_attention(q, kv, kpe, cos, sin, softmax_scale):
    """Causal MLA attention on the MXFP8 kernels with Megatron's MLA RoPE fused in (apply_rope_fusion layout), SBHD:
    q [s, b, h, 192] (the q up-projection, d 128..191 before RoPE), kv [s, b, h, 256] (the kv up-projection: [k nope |
    v]), kpe [s, b, 1, 64] or [s, b, 64] (k_pos_emb before RoPE), cos / sin [>= s, 1, 1, 64] bf16 (RotaryEmbedding's
    cached cos / sin) -> out [s, b, h, 128] bf16 (contiguous). Gradients flow to q, kv and kpe."""
    return _MLARopeAttention.apply(q, kv, kpe, cos, sin, softmax_scale)
