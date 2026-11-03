###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""bwd_v1 -- host side of the HK gfx1250 d128 GQA attention backward v1.

    flash_attn_bwd(do, q, k, v, o, lse, softmax_scale=None, causal=True) -> (dq, dk, dv)

Same contract, shapes, gate and numerics as bwd_v0 (q / o / do [B, Sq, Hq, 128], k / v [B, Skv, Hkv, 128] bf16
contiguous BSHD, lse fp32 [B, Hq, Sq] natural log, bottom-right causal). Kernels (one code object,
hip/src/k_attn_bwd_v1.hip; design hk/notes/bwd_v1_design.md, implementation note hk/notes/bwd_v1_impl.md), all
on torch.cuda.current_stream(), in this order:

    k_hk_bwd1_prep_z  LD records {NL, delta} + dQ_acc zeroed (HK_BWD1_ZERO=memset: torch memset + k_hk_bwd1_prep)
                                                                               grid (Sq/64, Hq, B), block 256
    k_hk_bwd1_g4      dK, dV, fp32 dQ_acc atomics (G = 4; re-timed expert-mode step, head rotation)
                                                                               grid (Hkv, Skv/128, B), block 128
    k_hk_bwd1_dqcvt   dq = bf16(scale dQ_acc)                                  grid (Sq/32, Hq, B), block 256

The compile-time variant (BWD1_SCHED / ATOM / NWIN / HROT, expert or default scheduling) is a property of the
code-object directory; every variant has the same kernels, arguments and grids.

Lab contract (hk/lab/lab.py): arm `name=hk:<this package dir>#bwd_v1@<hsaco dir>`. Every launch, the memset
included, goes through the module-level _launch(name, fn, args); the code object <dir>/k_attn_bwd_v1.hsaco is
resolved at the first launch from HSACO_DIR (lab / set_hsaco_dir), else env HK_HSACO_DIR, else DEFAULT_HSACO_DIR.
Validation aid: HK_BWD1_POISON=1 NaN-fills ld / dq / dk / dv first.

Production (impl.py, the package default "v1"; manifest entry k_attn_bwd_v1): impl points the module at the package's
pinned bin/ with set_hsaco_dir before its first launch, so DEFAULT_HSACO_DIR (the lab's default-variant directory) is
never used there.

Strided layouts (``SUPPORTS_STRIDES``; bwd1/idx.h HK_STRIDES). A code object built with -DHK_STRIDES=1 takes, after
each kernel's v1 arguments, the int64 (batch, seq) element strides of the BSHD-family tensors it addresses
(``STRIDE_TENSORS``: prep / prep_z do, o; g4 q, k, v, do, dk, dv; dqcvt dq) and then accepts each of do, q, k, v, o
independently as contiguous BSHD or as a [B, S, H, D] view of [S, B, H, D] storage (``layout_of``, Megatron's layout);
dq / dk / dv are allocated in the layouts of q / k / v. lse stays [B, Hq, Sq] fp32 contiguous; the bits of dq / dk / dv
do not depend on the layouts. Whether the code object is such a build is read from its own kernel metadata (every
kernel's argument count is the default one or the strided one, all four the same; a cc.sh build's recorded
-DHK_STRIDES flag must agree), never guessed.
"""

import json
import math
import os
import threading

import torch

from . import bwd_v0 as _v0
from . import codeobj, hiprt

__all__ = [
    "flash_attn_bwd",
    "plan",
    "host_gate",
    "unsupported_reason",
    "set_hsaco_dir",
    "SPECS",
    "STRIDE_TENSORS",
    "layout_of",
    "supports_strides",
]

DEFAULT_HSACO_DIR = "/home/lihuzhan/projects/2026_1010__llama3.1_attn/hk/build/bwd_v1/default"
HSACO_NAME = "k_attn_bwd_v1.hsaco"
HSACO_DIR = None

D, G, BKV, QS, PREP_ROWS = _v0.D, _v0.G, _v0.BKV, _v0.QS, _v0.PREP_ROWS
LIM = _v0.LIM
Ref = _v0.Ref

_P, _I, _F = "ptr", "i32", "f32"
SPECS = {
    "k_hk_bwd1_prep": {"block": 256, "args": _v0._PREP_ARGS, "lds": 256},
    "k_hk_bwd1_prep_z": {"block": 256, "args": _v0._PREP_ARGS, "lds": 256},
    "k_hk_bwd1_g4": {"block": 128, "args": _v0.SPECS["k_hk_bwd_g4"]["args"], "lds": 266240},
    "k_hk_bwd1_dqcvt": {"block": 256, "args": _v0.SPECS["k_hk_bwd_dqcvt"]["args"], "lds": 8704},
}


# -DHK_STRIDES=1 builds: the tensors whose (batch, seq) element strides follow each kernel's v1 arguments, in order
# (k_attn_bwd_v1.hip); the head stride stays D, the element stride 1
STRIDE_TENSORS = {
    "k_hk_bwd1_prep": ("do", "o"),
    "k_hk_bwd1_prep_z": ("do", "o"),
    "k_hk_bwd1_g4": ("q", "k", "v", "do", "dk", "dv"),
    "k_hk_bwd1_dqcvt": ("dq",),
}
LAYOUTS = ("bshd", "sbhd")
TDM_STRIDE_LIMIT = (1 << 48) - 1  # b1::TDM_STRIDE_MAX: the TDM descriptor's row-stride field (elements)
I32_LIMIT = 1 << 31
ROW_B = D * 2


def stride_args(name):
    """The kernel's extra (name, kind) arguments of a -DHK_STRIDES=1 build."""
    return [(f"{t}_{x}", "i64") for t in STRIDE_TENSORS[name] for x in ("sb", "ss")]


def specs(strided):
    """SPECS with the stride arguments appended to every kernel when ``strided``."""
    return {n: {"block": s["block"], "args": list(s["args"]) + (stride_args(n) if strided else []), "lds": s["lds"]}
            for n, s in SPECS.items()}  # fmt: skip


def layout_of(t):
    """The layout of a 4-D [B, S, H, D] tensor: "bshd" (contiguous), "sbhd" ([S, B, H, D] contiguous storage viewed
    through permute(1, 0, 2, 3)) or None. Strides of size-1 dims are ignored; a tensor that is both (B == 1 or S == 1:
    the same memory) is "bshd". (Same rule as fwd_v1.layout_of, kept local: the modules are edited independently.)"""
    if not isinstance(t, torch.Tensor) or t.dim() != 4:
        return None
    b, s, h, d = t.shape

    def match(want):
        return all(n == 1 or x == y for n, x, y in zip(t.shape, t.stride(), want))

    if match((s * h * d, h * d, d, 1)):
        return "bshd"
    if match((h * d, b * h * d, d, 1)):
        return "sbhd"
    return None


def layout_strides(shape, layout):
    """(batch, seq) element strides of a [B, S, H, D] tensor in ``layout`` (the kernel's x_sb, x_ss)."""
    b, s, h, d = shape
    if layout == "bshd":
        return s * h * d, h * d
    if layout == "sbhd":
        return h * d, b * h * d
    raise ValueError(f"layout {layout!r} not in {LAYOUTS}")


def strides_reason(name, shape, layout):
    """None when the kernels can address a [B, S, H, D] tensor in ``layout`` (b1::strides_ok; dk / dv also
    b1::slab_ok), else the reason."""
    b, s, h, _ = shape
    sb, ss = layout_strides(shape, layout)
    if ss > TDM_STRIDE_LIMIT or 2 * ss >= I32_LIMIT or b * s * h * ROW_B >= I32_LIMIT:
        return f"{name} {tuple(shape)} {layout}: strides ({sb}, {ss}) or extent beyond the kernels' int32 / TDM fields"
    if name in ("dk", "dv") and (BKV - 1) * ss * 2 + ROW_B >= I32_LIMIT:
        return f"{name} {tuple(shape)} {layout}: the 128-row slab exceeds the int32 voffset range"
    return None


def _launch(name, fn, args):
    """Every kernel launch and memset of this module goes through here (lab --seg wraps it)."""
    return fn(*args)


# ---------------------------------------------------------------- shape rules (== v0: same kernels' gate)

GATE_OK, GATE_NAMES = _v0.GATE_OK, _v0.GATE_NAMES


def host_gate(B, Sq, Skv, Hq, Hkv, causal, g=G):
    """Mirror of b0::host_gate (bwd/idx.h; v1 adds no rule: n and nf are multiples of G = 4 for every admitted
    shape, which the x4-unrolled loop needs -- proved by hk/lab/bounds/bwd_v1_check.cpp)."""
    return _v0.host_gate(B, Sq, Skv, Hq, Hkv, causal, g)


def unsupported_reason(q_shape, k_shape, v_shape, dtype, causal):
    """None when bwd_v1 takes the problem (the bwd_v0 rules), else the reason."""
    r = _v0.unsupported_reason(q_shape, k_shape, v_shape, dtype, causal)
    return None if r is None else r.replace("bwd_v0", "bwd_v1")


def _check(do, q, k, v, o, lse, causal, strided=False):
    """bwd_v0's contract (contiguous BSHD). ``strided``: the code object takes strides, so each of do, q, k, v, o may
    be "bshd" or "sbhd" (``layout_of``, with the kernels' stride limits); everything else is v0's rule."""
    if not strided:
        try:
            _v0._check(do, q, k, v, o, lse, causal)
        except ValueError as e:
            raise ValueError(str(e).replace("bwd_v0", "bwd_v1")) from None
        return
    named = {"do": do, "q": q, "k": k, "v": v, "o": o}
    for n, t in list(named.items()) + [("lse", lse)]:
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"{n} must be a tensor")
        if t.device != q.device:
            raise ValueError(f"all tensors must be on one device: q on {q.device}, {n} on {t.device}")
    if not lse.is_contiguous():
        raise ValueError("lse must be contiguous")
    for n, t in named.items():
        if t.dtype != torch.bfloat16:
            raise ValueError(f"{n} must be bfloat16, got {t.dtype}")
    reason = unsupported_reason(q.shape, k.shape, v.shape, q.dtype, causal)
    if reason is not None:
        raise ValueError(reason)
    if do.shape != q.shape or o.shape != q.shape:
        raise ValueError(
            f"do / o must have q's shape {tuple(q.shape)}, got {tuple(do.shape)} / {tuple(o.shape)}"
        )
    for n, t in named.items():
        lay = layout_of(t)
        if lay is None:
            raise ValueError(
                f"{n} must be contiguous BSHD or a [B, S, H, D] view of SBHD storage, strides {t.stride()}"
            )
        r = strides_reason(n, t.shape, lay)
        if r is not None:
            raise ValueError(f"bwd_v1: {r}")
    for n, t in (("dk", k), ("dv", v)):
        r = strides_reason(n, t.shape, layout_of(t))
        if r is not None:
            raise ValueError(f"bwd_v1: {r}")
    b, sq, hq, _ = q.shape
    if lse.dtype != torch.float32 or tuple(lse.shape) != (b, hq, sq):
        raise ValueError(f"lse must be float32 of shape {(b, hq, sq)}, got {lse.dtype} {tuple(lse.shape)}")


def zero_mode():
    """'prep' (default: k_hk_bwd1_prep_z zeroes dQ_acc) or 'memset' (env HK_BWD1_ZERO=memset)."""
    m = os.environ.get("HK_BWD1_ZERO", "prep")
    if m not in ("prep", "memset"):
        raise ValueError(f"HK_BWD1_ZERO must be prep or memset, got {m!r}")
    return m


# ---------------------------------------------------------------- launch plan (shapes only; CPU-testable)


def plan(B, Sq, Skv, Hq, Hkv, causal, scale, prep_zero=True, strided=False, layouts=None):
    """{"work": {name: (shape, dtype)}, "launches": [(name, grid, block, args)]} of one call; grid None = a torch
    memset (args = the tensor). ValueError for every shape the gate rejects: no launch is ever planned for it.
    ``strided``: a -DHK_STRIDES=1 code object; ``layouts`` {do, q, k, v, o: "bshd" | "sbhd"} (default all "bshd"),
    the outputs dq / dk / dv in the layouts of q / k / v; each kernel's stride values follow its v1 arguments."""
    lay = {n: "bshd" for n in ("do", "q", "k", "v", "o")}
    lay.update(layouts or {})
    lay.update(dq=lay["q"], dk=lay["k"], dv=lay["v"])
    if not strided and any(x != "bshd" for x in lay.values()):
        raise ValueError(f"bwd_v1: non-BSHD layouts {lay} need a -DHK_STRIDES=1 code object")
    shp = {n: (B, Sq, Hq, D) for n in ("do", "q", "o", "dq")}
    shp.update({n: (B, Skv, Hkv, D) for n in ("k", "v", "dk", "dv")})
    rc = host_gate(B, Sq, Skv, Hq, Hkv, int(bool(causal)))
    if rc != GATE_OK:
        raise ValueError(f"bwd_v1 host gate: {GATE_NAMES[rc]}")
    nqp = Sq // QS
    work = {"ld": ((B, Hq, Sq // 4, 8), torch.float32), "dqa": ((B * Hq * nqp * 4096,), torch.float32)}
    L = []
    if not prep_zero:
        L.append(("zero_dqa", None, None, (Ref("dqa"),)))
    pre = (Ref("do"), Ref("o"), Ref("lse"), Ref("ld"), Ref("dqa"), Sq, Hq, B)
    L.append(("k_hk_bwd1_prep_z" if prep_zero else "k_hk_bwd1_prep", (Sq // PREP_ROWS, Hq, B), 256, pre))
    L.append(("k_hk_bwd1_g4", (Hkv, Skv // BKV, B), 128,
              (Ref("q"), Ref("k"), Ref("v"), Ref("do"), Ref("ld"), Ref("dk"), Ref("dv"), Ref("dqa"), float(scale),
               Sq, Skv, Hq, Hkv, B, int(bool(causal)))))  # fmt: skip
    L.append(("k_hk_bwd1_dqcvt", (Sq // QS, Hq, B), 256, (Ref("dqa"), Ref("dq"), float(scale), Sq, Hq, B)))
    if strided:
        for n in shp:
            r = strides_reason(n, shp[n], lay[n])
            if r is not None:
                raise ValueError(f"bwd_v1: {r}")
        L = [(name, grid, block, args if grid is None else
              tuple(args) + tuple(x for t in STRIDE_TENSORS[name] for x in layout_strides(shp[t], lay[t])))
             for name, grid, block, args in L]  # fmt: skip
    sp = specs(strided)
    for name, grid, block, args in L:
        if grid is None:
            continue
        spec = sp[name]
        assert block == spec["block"] and len(args) == len(spec["args"]), name
        for a, (an, kind) in zip(args, spec["args"]):
            assert (kind == _P) == isinstance(a, Ref), (name, an)
            if kind == _I:
                assert isinstance(a, int) and 0 <= a < LIM, (name, an, a)
            if kind == "i64":
                assert isinstance(a, int) and D <= a < I32_LIMIT, (name, an, a)
        assert all(1 <= x < (1 << 31) for x in grid), (name, grid)
    return {"work": work, "launches": L, "layouts": lay, "strided": bool(strided)}


# ---------------------------------------------------------------- code object (lazy)

_KERNELS = {}
_LOCK = threading.Lock()


def set_hsaco_dir(path):
    """lab hook: the code-object directory of later launches (resolved at the next launch)."""
    global HSACO_DIR
    HSACO_DIR = None if path is None else os.path.abspath(path)


def hsaco_path():
    d = HSACO_DIR or os.environ.get("HK_HSACO_DIR") or DEFAULT_HSACO_DIR
    return os.path.join(d, HSACO_NAME)


_STRIDED = {}  # code object path -> ((mtime_ns, size), strided)


def _flag_strides(flags):
    """The HK_STRIDES value a recorded compile command defines (0 when it defines none)."""
    val = 0
    for f in flags:
        if isinstance(f, str) and (f == "-DHK_STRIDES" or f.startswith("-DHK_STRIDES=")):
            v = f.partition("=")[2] or "1"
            if v not in ("0", "1"):
                raise hiprt.HipError(f"compile flag {f!r}: HK_STRIDES must be 0 or 1")
            val = int(v)
    return val


def _check_code_object(path):
    """True / False: the code object is a -DHK_STRIDES=1 / default build, read from its kernel metadata (every kernel
    present, its resources as SPECS says, and its argument count exactly the default or the strided one -- the same
    for all four kernels -- with 8-byte stride arguments after the v1 ones); a cc.sh meta.json must agree."""
    if not os.path.isfile(path):
        raise hiprt.HipError(f"bwd_v1: code object {path} not found (build it with hk/lab/cc.sh)")
    table = codeobj.kernels(path)
    kinds = set()
    for name, spec in SPECS.items():
        t = table.get(name)
        if t is None:
            raise hiprt.HipError(f"bwd_v1: {name} not in {path}")
        if t["lds"] != spec["lds"] or t["scratch"] != 0 or t["vgpr_spill"] or t["sgpr_spill"]:
            raise hiprt.HipError(f"bwd_v1: {name} in {path}: lds {t['lds']} (want {spec['lds']}), scratch "
                                 f"{t['scratch']}, spill v{t['vgpr_spill']} s{t['sgpr_spill']}")  # fmt: skip
        n0, ns = len(spec["args"]), len(spec["args"]) + len(stride_args(name))
        if len(t["args"]) == n0:
            kinds.add(False)
        elif len(t["args"]) == ns and all(a["size"] == 8 and a["offset"] % 8 == 0 for a in t["args"][n0:]):
            kinds.add(True)
        else:
            raise hiprt.HipError(f"bwd_v1: {name} in {path} has {len(t['args'])} arguments: neither the v1 ones ({n0}) "
                                 f"nor v1 + strides ({ns})")  # fmt: skip
    if len(kinds) != 1:
        raise hiprt.HipError(f"bwd_v1: {path} mixes strided and default kernels")
    strided = kinds.pop()
    mj = os.path.splitext(path)[0] + ".meta.json"
    if os.path.isfile(mj):
        with open(mj) as f:
            m = json.load(f)
        # a generator rung (hkl.build "gen") records the generator command, not the compile flags: nothing to compare
        if "flags" in m and "gen/build.py" not in m["flags"] and _flag_strides(m["flags"]) != int(strided):
            raise hiprt.HipError(f"bwd_v1: {path}: the argument layout says HK_STRIDES={int(strided)}, {mj} records "
                                 f"{[f for f in m['flags'] if 'HK_STRIDES' in str(f)]}")  # fmt: skip
    return strided


def supports_strides():
    """Whether the code object the next launch loads was built with -DHK_STRIDES=1 (``_check_code_object``). A missing
    code object raises. Also the module attribute ``SUPPORTS_STRIDES``."""
    path = hsaco_path()
    k = _KERNELS.get(path)
    if k is not None:
        return k[1]
    if not os.path.isfile(path):
        raise hiprt.HipError(f"bwd_v1: code object {path} not found: cannot tell whether it takes strides")
    st = os.stat(path)
    key = (st.st_mtime_ns, st.st_size)
    hit = _STRIDED.get(path)
    if hit is not None and hit[0] == key:
        return hit[1]
    strided = bool(_check_code_object(path))
    _STRIDED[path] = (key, strided)
    return strided


def __getattr__(name):
    # SUPPORTS_STRIDES is computed on access (PEP 562): it always describes the code object this module would load
    if name == "SUPPORTS_STRIDES":
        return supports_strides()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _kernels():
    """({name: kernel}, strided) of the resolved code object (loaded at the first launch)."""
    path = hsaco_path()
    k = _KERNELS.get(path)
    if k is not None:
        return k
    with _LOCK:
        k = _KERNELS.get(path)
        if k is None:
            strided = _check_code_object(path)
            kern = hiprt.load_hsaco(
                path, {n: {"block": s["block"], "args": s["args"]} for n, s in specs(strided).items()}
            )
            k = (kern, strided)
            _KERNELS[path] = k
    return k


# ---------------------------------------------------------------- entry point


def flash_attn_bwd(do, q, k, v, o, lse, softmax_scale=None, causal=True, fallback=None):
    """(dq [B,Sq,Hq,128] bf16, dk [B,Skv,Hkv,128] bf16, dv) of the attention backward; see the module doc.
    fallback: optional callable with flash_attn_bwd's signature for problems outside the gate (else ValueError)."""
    if fallback is not None and (not all(isinstance(t, torch.Tensor) for t in (q, k, v))
                                 or unsupported_reason(q.shape, k.shape, v.shape, q.dtype, causal) is not None):  # fmt: skip
        return fallback(do, q, k, v, o, lse, softmax_scale, causal)
    kern, strided = _kernels()
    _check(do, q, k, v, o, lse, causal, strided)
    if q.device.type != "cuda":
        raise ValueError(f"bwd_v1 runs on a gfx1250 GPU, got tensors on {q.device}")
    B, Sq, Hq, d = q.shape
    Skv, Hkv = k.shape[1], k.shape[2]
    scale = float(1.0 / math.sqrt(d) if softmax_scale is None else softmax_scale)
    layouts = (
        {n: layout_of(t) for n, t in (("do", do), ("q", q), ("k", k), ("v", v), ("o", o))}
        if strided
        else None
    )
    pl = plan(
        B, Sq, Skv, Hq, Hkv, causal, scale, prep_zero=zero_mode() == "prep", strided=strided, layouts=layouts
    )
    return _run(
        pl,
        {"do": do, "q": q, "k": k, "v": v, "o": o, "lse": lse},
        kern,
        torch.cuda.current_stream(),
        poison=os.environ.get("HK_BWD1_POISON") == "1",
    )


def _run(pl, t, kern, stream, poison=False):
    """Allocate the workspace / outputs on the inputs' device and issue the plan's launches in order through
    _launch (kern: {name: object with .call(grid, block, values, stream)}). Returns (dq, dk, dv)."""
    dev = t["q"].device
    for name, (shape, dtype) in pl["work"].items():
        t[name] = torch.empty(shape, device=dev, dtype=dtype)
    lay = pl.get("layouts") or {}
    for out, src in (("dq", "q"), ("dk", "k"), ("dv", "v")):
        b, s_, h, d = t[src].shape
        if lay.get(out, "bshd") == "sbhd":  # [S, B, H, D] storage seen as [B, S, H, D] (the input's layout)
            t[out] = torch.empty((s_, b, h, d), device=dev, dtype=t[src].dtype).permute(1, 0, 2, 3)
        else:
            t[out] = torch.empty((b, s_, h, d), device=dev, dtype=t[src].dtype)
    if poison:
        for n in ("ld", "dq", "dk", "dv"):
            _launch("poison_" + n, torch.Tensor.fill_, (t[n], float("nan")))
    for name, grid, block, args in pl["launches"]:
        vals = [t[a.name] if isinstance(a, Ref) else a for a in args]
        if grid is None:
            _launch(name, torch.Tensor.zero_, (vals[0],))
        else:
            _launch(name, kern[name].call, (grid, (block, 1, 1), vals, stream))
    return t["dq"], t["dk"], t["dv"]
