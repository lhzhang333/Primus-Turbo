###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""HipKittens flash attention for gfx1250 (MI455X), head dim 128: what the HIPKITTENS backend runs there.

The kernels are code objects built from ``csrc/kernels/attention/hipkittens/gfx1250`` into
``primus_turbo/lib/hk_gfx1250`` (see the README there) and launched through the HIP runtime torch loaded:
  k_attn_fwd       O and LSE (natural log, fp32 [B, Hq, Sq]); G = Hq / Hkv in (4, 8, 16), persistent grid
  k_attn_bwd_prep  the per-row softmax terms of the backward, and a zeroed fp32 dQ workspace
  k_attn_bwd_main  dK, dV and the dQ workspace (fp32 atomics: dQ is not bitwise reproducible); G = 4
  k_attn_bwd_dq    dQ = bf16(scale * workspace)
They take bf16, bottom-right causal attention with sequence lengths a multiple of 64 (Skv of 128 for the
backward), the same LSE convention as the FlyDSL gfx1250 kernels (``attention_flydsl_gfx1250_impl``), and every
[b, s, h, d] tensor either contiguous ("bshd") or as a view of contiguous [s, b, h, d] storage ("sbhd",
Megatron's layout), each tensor independently. Outputs come in the layouts of the inputs they belong to (out and
dq in q's, dk in k's, dv in v's), so an SBHD caller makes no layout copy.

The HIPKITTENS backend takes every call the FlyDSL gfx1250 head-dim-128 kernels take; each direction of a call
the HipKittens kernels cannot take (``hk_gfx1250_unsupported_reason``: shapes, a deterministic backward, missing
or unpinned code objects) runs on FlyDSL instead, with one warning per reason.
"""

import ctypes
import functools
import hashlib
import math
import os
import struct
import threading
from pathlib import Path
from typing import Optional, Tuple

import torch

from primus_turbo.common.logger import logger
from primus_turbo.pytorch.kernels.attention.attention_flydsl_gfx1250_impl import (
    _interface as _flydsl_interface,
)
from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    _eager_custom_op,
)

__all__ = [
    "flash_attn_hipkittens_gfx1250_backward_impl",
    "flash_attn_hipkittens_gfx1250_forward_impl",
    "hk_gfx1250_operands",
    "hk_gfx1250_unsupported_reason",
]

D = 128
_I32 = 1 << 31
_FWD_G = (4, 8, 16)  # the forward's kv head is a shift by log2(G)
_BWD_G = 4
# code object: (sha256 pin, {kernel: (workgroup size, packed arguments)}); the pins and the kernel signatures of
# csrc/kernels/attention/hipkittens/gfx1250 (a test keeps them equal to its manifest.json and sources)
_CODE_OBJECTS = {
    "k_attn_fwd": (
        "865fc5939656367233efcac721d8a98cfd0d2dd807ab674fce647caa81571aec",
        # Q K V O LSE, o_bytes lse_bytes, B Sq Skv Hq Hkv G nqb nhq4 nkv_st cst causal, c2 c2inv, W mode,
        # (magic, shift) of W, max(nqb / 2, 1), nhq4, B nhq4, element strides (batch, seq) of Q K V O
        {"k_attn_fwd": (128, "<5Q2q11i2f2iIiIiIiIi4x8q")},
    ),
    "k_attn_bwd": (
        "9ffc772e61727151aadaf348d9afde7e3c723f8aa648c3a9003e4fa946c19c38",
        {
            # DO O LSE LD DQA, Sq Hq B, strides of DO O
            "k_attn_bwd_prep": (256, "<5Q3i4x4q"),
            # Q K V DO LD DK DV DQA, scale, Sq Skv Hq Hkv B causal, strides of Q K V DO DK DV
            "k_attn_bwd_main": (128, "<8Qf6i4x12q"),
            # DQA DQ, scale, Sq Hq B, strides of DQ
            "k_attn_bwd_dq": (256, "<2Qf3i2q"),
        },
    ),
}
_KERNEL_DIR = Path(__file__).resolve().parents[3] / "lib" / "hk_gfx1250"


# ---------------------------------------------------------------- HIP runtime


@functools.lru_cache(maxsize=None)
def _hip_runtime() -> ctypes.CDLL:
    """The libamdhip64 torch loaded, typed for the module API: launches through another copy would run in another
    HIP runtime, outside torch's contexts and streams."""
    with open("/proc/self/maps") as f:
        paths = {os.path.realpath(line[line.index("/") :].strip()) for line in f if "/libamdhip64.so" in line}
    found = paths
    if len(found) > 1:  # e.g. the copy a torch wheel bundles next to a system one: torch's own
        torch_lib = os.path.realpath(os.path.join(os.path.dirname(torch.__file__), "lib")) + os.sep
        found = {p for p in paths if p.startswith(torch_lib)}
    if len(found) != 1:
        raise RuntimeError(f"cannot tell which libamdhip64 torch loaded: {sorted(paths)}")
    hip = ctypes.CDLL(found.pop(), mode=os.RTLD_NOLOAD | os.RTLD_LAZY)
    p, u = ctypes.c_void_p, ctypes.c_uint
    hip.hipModuleLoadData.argtypes = [ctypes.POINTER(p), p]
    hip.hipModuleGetFunction.argtypes = [ctypes.POINTER(p), p, ctypes.c_char_p]
    hip.hipModuleLaunchKernel.argtypes = [p, u, u, u, u, u, u, u, p, ctypes.POINTER(p), ctypes.POINTER(p)]
    hip.hipGetErrorString.argtypes = [ctypes.c_int]
    hip.hipGetErrorString.restype = ctypes.c_char_p
    return hip


def _hip_check(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{what} failed: {_hip_runtime().hipGetErrorString(rc).decode()} ({rc})")


class _CodeObject:
    """A pinned code object of _KERNEL_DIR, loaded once per device; its kernels launch on a torch stream."""

    _lock = threading.Lock()

    def __init__(self, name: str):
        self.name = name
        pin, self.kernels = _CODE_OBJECTS[name]
        path = _KERNEL_DIR / f"{name}.hsaco"
        if not path.is_file():
            raise RuntimeError(
                f"{path} is not built (GPU_ARCHS=gfx1250 and the toolchain manifest.json pins)"
            )
        image = path.read_bytes()
        if hashlib.sha256(image).hexdigest() != pin:
            raise RuntimeError(f"{path} is not the pinned code object: stale or foreign, rebuild it")
        self.args = {k: struct.Struct(layout) for k, (_, layout) in self.kernels.items()}
        self.image = ctypes.create_string_buffer(image)
        self.functions = {}  # device index -> {kernel: hipFunction_t}

    def load(self, device: int) -> dict:
        with self._lock:
            if device not in self.functions:
                module = ctypes.c_void_p()
                hip = _hip_runtime()
                with torch.cuda.device(device):  # hipModuleLoadData loads into the current device
                    _hip_check(
                        hip.hipModuleLoadData(ctypes.byref(module), self.image), f"loading {self.name}"
                    )
                fns = {}
                for kernel in self.kernels:
                    fn = ctypes.c_void_p()
                    _hip_check(hip.hipModuleGetFunction(ctypes.byref(fn), module, kernel.encode()), kernel)
                    fns[kernel] = fn
                self.functions[device] = fns
            return self.functions[device]

    def launch(
        self, kernel: str, grid: Tuple[int, int, int], values: tuple, stream: torch.cuda.Stream
    ) -> None:
        fns = self.functions.get(stream.device_index)
        if fns is None:
            fns = self.load(stream.device_index)
        args = self.args[kernel].pack(*(x.data_ptr() if isinstance(x, torch.Tensor) else x for x in values))
        buf, size = ctypes.create_string_buffer(args, len(args)), ctypes.c_size_t(len(args))
        # HIP_LAUNCH_PARAM_BUFFER_POINTER, buf, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END
        extra = (ctypes.c_void_p * 5)(1, ctypes.addressof(buf), 2, ctypes.addressof(size), 3)
        rc = _hip_runtime().hipModuleLaunchKernel(
            fns[kernel], *grid, self.kernels[kernel][0], 1, 1, 0, stream.cuda_stream, None, extra
        )
        _hip_check(rc, f"launching {kernel}")


_LOADED = {}  # code object name -> _CodeObject


def _code_object(name: str) -> _CodeObject:
    co = _LOADED.get(name)
    if co is None:
        co = _LOADED[name] = _CodeObject(name)
    return co


_UNAVAILABLE = {}  # device index -> why the kernels cannot run there, or None


@torch.compiler.assume_constant_result  # asked in compiled regions; the answer is fixed per device
def hk_gfx1250_unavailable_reason() -> Optional[str]:
    """None when the code objects are the pinned ones and load on the current device, else why not."""
    try:
        device = torch.cuda.current_device()
    except RuntimeError as e:  # no usable GPU
        return f"the gfx1250 HipKittens attention kernels are unavailable: {e}"
    if device not in _UNAVAILABLE:
        reason = None
        try:
            for name in _CODE_OBJECTS:
                _code_object(name).load(device)
        except (OSError, RuntimeError) as e:
            reason = f"the gfx1250 HipKittens attention kernels are unavailable: {e}"
        _UNAVAILABLE[device] = reason
    return _UNAVAILABLE[device]


# ---------------------------------------------------------------- layouts


def tensor_layout(t: torch.Tensor) -> Optional[str]:
    """ "bshd" for a contiguous [b, s, h, d] tensor, "sbhd" for a [b, s, h, d] view of contiguous [s, b, h, d]
    storage (``x.permute(1, 0, 2, 3)``), None otherwise. A tensor that is both (b == 1 or s == 1) is "bshd"."""
    if t.dim() != 4:
        return None
    if t.is_contiguous():
        return "bshd"
    if t.permute(1, 0, 2, 3).is_contiguous():
        return "sbhd"
    return None


def hk_gfx1250_operands(*tensors: torch.Tensor) -> tuple:
    """The tensors to hand the kernels: each "bshd" / "sbhd" one as it is, any other made contiguous."""
    # a list comprehension: Dynamo traces it (a generator expression calling tensor_layout it does not)
    return tuple([t if tensor_layout(t) is not None else t.contiguous() for t in tensors])


def _strides(t: torch.Tensor) -> Tuple[int, int]:
    """(batch, seq) element strides of a "bshd" / "sbhd" tensor (those of a size-1 dimension address nothing:
    the layout's own are used)."""
    b, s, h, d = t.shape
    return (s * h * d, h * d) if tensor_layout(t) == "bshd" else (h * d, b * h * d)


def _empty(like: torch.Tensor, shape, layout: str) -> torch.Tensor:
    """Uninitialised [b, s, h, d] ``shape`` on like's device / dtype, contiguous or in SBHD storage."""
    b, s, h, d = shape
    if layout == "sbhd":
        return like.new_empty((s, b, h, d)).permute(1, 0, 2, 3)
    return like.new_empty((b, s, h, d))


def _layout_of_output(t: torch.Tensor) -> str:
    """The layout of the output or gradient that belongs to input ``t``."""
    return "sbhd" if tensor_layout(t) == "sbhd" else "bshd"


def _in_layout(t: torch.Tensor, layout: str) -> torch.Tensor:
    """``t`` when it is in ``layout`` already, else a copy that is."""
    return t if tensor_layout(t) == layout else _empty(t, t.shape, layout).copy_(t)


# ---------------------------------------------------------------- gate


def _shape_reason(direction: str, b: int, sq: int, skv: int, hq: int, hkv: int, d: int, causal: bool):
    """Why the kernels of ``direction`` cannot take this problem (shapes only), or None."""
    if d != D:
        return f"head dim must be {D}, got {d}"
    if min(b, sq, skv, hq, hkv) < 1:
        return f"empty problem: b {b}, sq {sq}, skv {skv}, hq {hq}, hkv {hkv}"
    if not causal:
        return "causal attention only"
    if sq > skv:
        return f"seqlen_q {sq} > seqlen_kv {skv}"
    if hq % hkv:
        return f"heads_q {hq} not a multiple of heads_kv {hkv}"
    g = hq // hkv
    if direction == "fwd":
        if g not in _FWD_G:
            return f"heads_q / heads_kv must be one of {_FWD_G}, got {g}"
        if sq % 64 or skv % 64:
            return f"seqlen_q / seqlen_kv must be multiples of 64, got {sq} / {skv}"
    else:
        if g != _BWD_G:
            return f"heads_q / heads_kv must be {_BWD_G}, got {g}"
        if sq % 64 or skv % 128:
            return f"seqlen_q / seqlen_kv must be multiples of 64 / 128, got {sq} / {skv}"
    # the kernels' byte offsets are int32: q / o / dq and k / v / dk / dv (every other offset is smaller), and the
    # backward's fp32 dQ workspace
    extents = [2 * b * sq * hq * D, 2 * b * skv * hkv * D]
    if direction == "bwd":
        extents.append(4 * b * hq * sq * D)
    if max(extents) >= _I32:
        return "a tensor exceeds the kernels' int32 byte offsets"
    return None


def hk_gfx1250_unsupported_reason(
    direction: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    deterministic: bool = False,
) -> Optional[str]:
    """Why the HipKittens kernels cannot run ``direction`` ("fwd" / "bwd") of this call, or None; never raises."""
    if direction == "bwd" and deterministic:
        return "deterministic=True (dQ is summed with fp32 atomics)"
    if any(t.dim() != 4 for t in (q, k, v)):
        return "q / k / v must be 4-D [b, s, h, d]"
    if any(t.dtype != torch.bfloat16 or t.device != q.device for t in (q, k, v)):
        return "q / k / v must be bfloat16 on one device"
    if tuple(k.shape) != tuple(v.shape) or k.shape[0] != q.shape[0]:
        return f"k {tuple(k.shape)} and v {tuple(v.shape)} must match, with q's batch"
    b, sq, hq, d = q.shape
    _, skv, hkv, _ = k.shape
    reason = _shape_reason(direction, b, sq, skv, hq, hkv, d, bool(causal))
    return reason if reason is not None else hk_gfx1250_unavailable_reason()


def _misaligned(*tensors: torch.Tensor) -> bool:
    return any(t.data_ptr() % 16 for t in tensors)


# ---------------------------------------------------------------- launches


def _fdiv(d: int) -> Tuple[int, int]:
    """(m, sh) with n // d == (n * m) >> sh for every 0 <= n < 2^31 (fwd_idx.h Fdiv)."""
    lg = max(d - 1, 0).bit_length()
    return ((1 << (31 + lg)) + d - 1) // d, 31 + lg


def _fwd_plan(q, k, v, scale: float, causal: bool, cus: int):
    """((kernel, grid, argument values), (o in q's layout, lse)) of k_attn_fwd for "bshd" / "sbhd" q / k / v
    the kernel takes; works on meta tensors."""
    b, sq, hq, _ = q.shape
    skv, hkv = k.shape[1], k.shape[2]
    nqb, nhq4, nkv_st = sq // 64, hq // 4, skv // 64
    pairs = b * nhq4 * ((nqb + 1) // 2)
    # persistent grid: zig-zag pairs of q blocks over min(#CUs, #pairs) workgroups, else one item per workgroup
    mode, w = (1, min(cus, pairs)) if pairs >= cus else (0, b * nhq4 * nqb)
    o = _empty(q, q.shape, _layout_of_output(q))
    lse = q.new_empty((b, hq, sq), dtype=torch.float32)
    c2 = scale * math.log2(math.e)  # packed as fp32, like 1 / c2
    shape = (b, sq, skv, hq, hkv, hq // hkv, nqb, nhq4, nkv_st, (skv - sq) // 64, int(causal))
    values = (q, k, v, o, lse, 2 * o.numel(), 4 * lse.numel(), *shape, c2, 1.0 / c2, w, mode)
    values += (*_fdiv(w), *_fdiv(max(nqb // 2, 1)), *_fdiv(nhq4), *_fdiv(b * nhq4))
    values += (*_strides(q), *_strides(k), *_strides(v), *_strides(o))
    return ("k_attn_fwd", (w, 1, 1), values), (o, lse)


def _bwd_plan(do, q, k, v, o, lse, scale: float, causal: bool):
    """([(kernel, grid, argument values)], (dq, dk, dv) in the layouts of q / k / v) of the backward kernels;
    works on meta tensors."""
    b, sq, hq, _ = q.shape
    skv, hkv = k.shape[1], k.shape[2]
    ld = lse.new_empty((b, hq, sq // 4, 8))  # {NL, delta} records
    dqa = lse.new_empty((b * hq * (sq // 32) * 4096,))  # fp32 dQ workspace, zeroed by k_attn_bwd_prep
    dq, dk, dv = (_empty(t, t.shape, _layout_of_output(t)) for t in (q, k, v))
    main = (q, k, v, do, ld, dk, dv, dqa, scale, sq, skv, hq, hkv, b, int(causal))
    main += tuple(x for t in (q, k, v, do, dk, dv) for x in _strides(t))
    launches = [
        ("k_attn_bwd_prep", (sq // 64, hq, b), (do, o, lse, ld, dqa, sq, hq, b, *_strides(do), *_strides(o))),
        ("k_attn_bwd_main", (hkv, skv // 128, b), main),
        ("k_attn_bwd_dq", (sq // 32, hq, b), (dqa, dq, scale, sq, hq, b, *_strides(dq))),
    ]
    return launches, (dq, dk, dv)


def _hk_forward(q, k, v, scale: float, causal: bool):
    cus = torch.cuda.get_device_properties(q.device).multi_processor_count
    (kernel, grid, values), outputs = _fwd_plan(q, k, v, scale, causal, cus)
    _code_object("k_attn_fwd").launch(kernel, grid, values, torch.cuda.current_stream(q.device))
    return outputs


def _hk_backward(do, q, k, v, o, lse, scale: float, causal: bool):
    launches, grads = _bwd_plan(do, q, k, v, o, lse, scale, causal)
    co, stream = _code_object("k_attn_bwd"), torch.cuda.current_stream(q.device)
    for kernel, grid, values in launches:
        co.launch(kernel, grid, values, stream)
    return grads


# ---------------------------------------------------------------- the ops


def _fallback(direction: str, reason: str) -> None:
    name = "forward" if direction == "fwd" else "backward"
    logger.warning(f"HIPKITTENS attention on gfx1250: this {name} runs on FlyDSL ({reason})", once=True)


@_eager_custom_op("primus_turbo::flash_attn_hipkittens_gfx1250_forward")
def _forward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float, causal: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    # as the caller prepared them, unless a compiled graph changed a layout
    q, k, v = hk_gfx1250_operands(q, k, v)
    reason = hk_gfx1250_unsupported_reason("fwd", q, k, v, causal)
    if reason is None and _misaligned(q, k, v):
        reason = "q / k / v data_ptr not 16-byte aligned"
    if reason is None:
        return _hk_forward(q, k, v, softmax_scale, causal)
    _fallback("fwd", reason)
    out, lse = _flydsl_interface().flash_attn_fwd(*(t.contiguous() for t in (q, k, v)), softmax_scale, causal)
    return _in_layout(out, _layout_of_output(q)), lse


@_forward.register_fake
def _(q, k, v, softmax_scale, causal):
    b, sq, hq, _ = q.shape
    return _empty(q, q.shape, _layout_of_output(q)), q.new_empty((b, hq, sq), dtype=torch.float32)


@_eager_custom_op("primus_turbo::flash_attn_hipkittens_gfx1250_backward")
def _backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    deterministic: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    layouts = [_layout_of_output(t) for t in (q, k, v)]  # what the fake promised
    dout, q, k, v, out = hk_gfx1250_operands(dout, q, k, v, out)
    lse = lse.contiguous()
    reason = hk_gfx1250_unsupported_reason("bwd", q, k, v, causal, deterministic)
    if reason is None and _misaligned(dout, q, k, v, out, lse):
        reason = "a tensor's data_ptr is not 16-byte aligned"
    if reason is None:
        return _hk_backward(dout, q, k, v, out, lse, softmax_scale, causal)
    _fallback("bwd", reason)
    dout, q_c, k_c, v_c, out = (t.contiguous() for t in (dout, q, k, v, out))
    grads = _flydsl_interface().flash_attn_bwd(
        dout, q_c, k_c, v_c, out, lse, softmax_scale, causal, deterministic
    )
    return tuple(_in_layout(g, layout) for g, layout in zip(grads, layouts))


@_backward.register_fake
def _(dout, q, k, v, out, lse, softmax_scale, causal, deterministic):
    return tuple(_empty(t, t.shape, _layout_of_output(t)) for t in (q, k, v))


def _scale(softmax_scale: Optional[float], head_dim: int) -> float:
    return float(head_dim**-0.5 if softmax_scale is None else softmax_scale)


def flash_attn_hipkittens_gfx1250_forward_impl(q, k, v, softmax_scale=None, causal=True):
    """(out [b, sq, hq, d] in q's layout, lse [b, hq, sq] fp32 natural log).

    q / k / v: [b, s, h, d] in any byte order; "bshd" / "sbhd" tensors are used as they are (a caller that keeps
    them for the backward prepares them with ``hk_gfx1250_operands`` first), any other is made contiguous."""
    q, k, v = hk_gfx1250_operands(q, k, v)
    return _forward(q, k, v, _scale(softmax_scale, q.shape[-1]), bool(causal))


def flash_attn_hipkittens_gfx1250_backward_impl(
    dout, q, k, v, out, lse, softmax_scale=None, causal=True, deterministic=False
):
    """(dq, dk, dv) in the layouts of q / k / v; dk / dv have k's heads.

    q / k / v / out / lse are what the forward saw and returned; dout may come in any byte order."""
    dout, q, k, v, out = hk_gfx1250_operands(dout, q, k, v, out)
    return _backward(
        dout, q, k, v, out, lse, _scale(softmax_scale, q.shape[-1]), bool(causal), bool(deterministic)
    )
