###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""HipKittens flash-attention forward/backward for gfx1250 (MI455X), head dim 128, dense [b, s, h, d]
tensors: what the HIPKITTENS backend runs on gfx1250.

The HIP kernels (``primus_turbo/hipkittens/gfx1250_attn``, ``impl.hk_attn_fwd`` / ``hk_attn_bwd``) take a
subset of what the FlyDSL gfx1250 head-dim-128 kernels take (bf16, causal, GQA groups of 4, sequence
multiples of 64 / 128, ...), with the same lse [B, Hq, Sq] fp32 natural log. The backend therefore accepts
exactly the FlyDSL gfx1250 head-dim-128 calls, and each custom op below runs its direction on HK when
``hk_unsupported_reason`` allows and on the FlyDSL gfx1250 kernels otherwise, logging one warning per reason.
The decision is made inside the op, at run time, so a compiled graph holds one op per direction and never
traces the gate. Cases that run FlyDSL: deterministic=True backward (HK adds dQ with atomics; FlyDSL's
deterministic two-kernel backward runs), shapes outside HK's rules, code objects that are not built / not
pinned by ``manifest.json``, a kernel module that is missing or fails to import, and the mixed-mode settings
``PRIMUS_TURBO_HK_FWD=flydsl`` / ``PRIMUS_TURBO_HK_BWD=flydsl`` (read at import, like the forward module
choice ``v1`` / ``v0`` / ``pp``; the lse convention is shared, so the directions can mix).

Layouts. Every [b, s, h, d] tensor is "bshd" (contiguous), "sbhd" (a [b, s, h, d] view of contiguous
[s, b, h, d] storage: what Megatron hands over) or neither (``tensor_layout``). When the HK kernels that run a
direction take strides (``hk_gfx1250_supports_strides``: impl's ``hk_supports_strides``, code objects built
with -DHK_STRIDES=1), ``hk_gfx1250_operands`` hands that direction's op each "bshd" / "sbhd" tensor as it is,
per tensor, and the kernels read and write them in place; otherwise, and for a tensor of neither layout, it
makes the tensor contiguous (the copy this path always made; none for a contiguous tensor). Each op returns
out in q's layout, dq in q's, dk in k's and dv in v's ("sbhd" when that input is "sbhd", else contiguous;
lse contiguous), whichever kernels ran: its fake (torch.compile) says so from the input layouts alone, and a
run that cannot write a layout in place (FlyDSL, or HK without strides) copies its result into it. With
contiguous inputs nothing changes: contiguous outputs, no copy.

Importing this module imports neither the HK package nor the FlyDSL kernels; the first call does.
"""

from typing import Optional

import torch

from primus_turbo.common.logger import logger
from primus_turbo.pytorch.kernels.attention.attention_flydsl_gfx1250_impl import (
    _interface as _flydsl_interface,
)
from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    _eager_custom_op,
)

__all__ = [
    "flash_attn_hk_gfx1250_backward_impl",
    "flash_attn_hk_gfx1250_forward_impl",
    "hk_gfx1250_operands",
    "hk_gfx1250_supports_strides",
    "hk_gfx1250_unsupported_reason",
    "tensor_layout",
]

_DIRECTION_NAME = {"fwd": "forward", "bwd": "backward"}

# The HK package's impl module once imported or, when it cannot be, the reason (a str); None until
# the first call. Resolved once: Python does not cache a failed import.
_IMPL = None


def _import_impl():
    from primus_turbo.hipkittens.gfx1250_attn import impl

    return impl


def _load_impl():
    """The HK impl module, or why it cannot be used (a str); never raises."""
    global _IMPL
    if _IMPL is None:
        try:
            _IMPL = _import_impl()
        except Exception as exc:  # noqa: BLE001 -- an HK package that cannot load is a fallback
            _IMPL = f"the gfx1250 HipKittens attention package failed to import: {type(exc).__name__}: {exc}"
    return _IMPL


def hk_gfx1250_unsupported_reason(
    direction: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    deterministic: bool = False,
    softmax_scale: Optional[float] = None,
) -> Optional[str]:
    """Why HK cannot run ``direction`` ("fwd" / "bwd") of this call, or None; never raises. A call it
    declines runs on the FlyDSL gfx1250 kernels, which the backend's gate already accepted."""
    impl = _load_impl()
    if isinstance(impl, str):
        return impl
    return impl.hk_unsupported_reason(direction, q, k, v, causal, deterministic, softmax_scale=softmax_scale)


# ---------------------------------------------------------------- layouts


def tensor_layout(t: torch.Tensor) -> Optional[str]:
    """ "bshd" for a contiguous [b, s, h, d] tensor, "sbhd" for a [b, s, h, d] view of contiguous [s, b, h, d]
    storage (``x.permute(1, 0, 2, 3)``), None otherwise. A tensor that is both (b == 1 or s == 1: the same
    bytes) is "bshd". Stride tests only, so torch.compile traces it (guards on the layout)."""
    if t.dim() != 4:
        return None
    if t.is_contiguous():
        return "bshd"
    if t.permute(1, 0, 2, 3).is_contiguous():
        return "sbhd"
    return None


def _out_layout(t: torch.Tensor) -> str:
    """The layout of the gradient / output that belongs to input ``t``: "sbhd" for "sbhd", else "bshd"."""
    return "sbhd" if tensor_layout(t) == "sbhd" else "bshd"


def _empty(like: torch.Tensor, shape, layout: str) -> torch.Tensor:
    """Uninitialised [b, s, h, d] ``shape`` on like's device / dtype, contiguous ("bshd") or SBHD storage."""
    b, s, h, d = shape
    if layout == "sbhd":
        return like.new_empty((s, b, h, d)).permute(1, 0, 2, 3)
    return like.new_empty((b, s, h, d))


def _in_layout(t: torch.Tensor, layout: str) -> torch.Tensor:
    """``t`` when it already is in ``layout`` (what the op's fake promised), else a copy that is."""
    if tensor_layout(t) == layout:
        return t
    return _empty(t, t.shape, layout).copy_(t)


@torch.compiler.assume_constant_result
def hk_gfx1250_supports_strides(direction: str) -> bool:
    """Whether the HK kernels that run ``direction`` take [b, s, h, d] views of SBHD storage as well as
    contiguous BSHD (impl's ``hk_supports_strides``); False when HK cannot run the direction or the package
    cannot load. Never raises.

    Marked constant for torch.compile, whose Dynamo calls it instead of tracing it (the glue asks it inside
    compiled regions): the answer is a property of the code objects the process loads, and the import and
    module state behind it are not traceable."""
    impl = _load_impl()
    probe = None if isinstance(impl, str) else getattr(impl, "hk_supports_strides", None)
    try:
        return bool(probe(direction)) if callable(probe) else False
    except Exception:  # noqa: BLE001 -- a question that cannot be answered is a no
        return False


def hk_gfx1250_operands(direction: str, *tensors: torch.Tensor, strided: bool = True):
    """The tensors to hand ``direction``'s op: each one as it is when the HK kernels of the direction take
    strides (``hk_gfx1250_supports_strides``; ``strided=False`` says they will not run, e.g. a deterministic
    backward) and it is "bshd" or "sbhd"; else ``.contiguous()``, the copy this path always made (none for a
    contiguous tensor)."""
    keep = strided and hk_gfx1250_supports_strides(direction)
    # a list comprehension: Dynamo traces it (a generator expression calling tensor_layout it does not)
    return tuple([t if keep and tensor_layout(t) is not None else t.contiguous() for t in tensors])


def _hk_operands(direction, *tensors):
    """Inside an op, right before HK runs: the tensors as they came when the selected kernels take strides now
    (asked again: the glue's answer may predate a re-pointed module), else contiguous (a copy only for an
    "sbhd" tensor, which then reaches a module without strides)."""
    probe = getattr(_IMPL, "hk_supports_strides", None)
    keep = callable(probe) and probe(direction)
    return tuple(t if keep and tensor_layout(t) is not None else t.contiguous() for t in tensors)


# ---------------------------------------------------------------- the ops


def _fallback_warning(direction, reason) -> None:
    name = _DIRECTION_NAME[direction]
    logger.warning(f"HIPKITTENS attention on gfx1250: {name} runs on FlyDSL ({reason})", once=True)


def _decide(direction, q, k, v, causal, deterministic=False, softmax_scale=None) -> bool:
    """True to run HK; else logs (once per reason) that FlyDSL runs this direction."""
    reason = hk_gfx1250_unsupported_reason(direction, q, k, v, causal, deterministic, softmax_scale)
    if reason is not None:
        _fallback_warning(direction, reason)
        return False
    logger.info(f"HIPKITTENS attention on gfx1250: {_IMPL.describe(direction)}", once=True)
    return True


# hk_attn_fwd / hk_attn_bwd raise NotImplementedError only before anything is launched (what the gate
# cannot see from q / k / v: do / o / lse alignment, a pinned code object replaced since the gate's
# check), so that call still runs on FlyDSL. FlyDSL reads contiguous BSHD: an "sbhd" input is copied for it
# and its results are copied into the layouts the op returns.


def _run_forward(q, k, v, softmax_scale, causal):
    layout = _out_layout(q)
    if _decide("fwd", q, k, v, causal, softmax_scale=softmax_scale):
        try:
            out, lse = _IMPL.hk_attn_fwd(*_hk_operands("fwd", q, k, v), softmax_scale, causal)
            return _in_layout(out, layout), lse
        except NotImplementedError as exc:
            _fallback_warning("fwd", exc)
    q, k, v = (t.contiguous() for t in (q, k, v))
    out, lse = _flydsl_interface().flash_attn_fwd(q, k, v, softmax_scale, causal)
    return _in_layout(out, layout), lse


def _run_backward(dout, q, k, v, out, lse, softmax_scale, causal, deterministic):
    layouts = tuple(_out_layout(t) for t in (q, k, v))
    if _decide("bwd", q, k, v, causal, deterministic, softmax_scale):
        try:
            grads = _IMPL.hk_attn_bwd(
                *_hk_operands("bwd", dout, q, k, v, out), lse, softmax_scale, causal, False
            )
            return tuple(_in_layout(g, layout) for g, layout in zip(grads, layouts))
        except NotImplementedError as exc:
            _fallback_warning("bwd", exc)
    dout, q, k, v, out = (t.contiguous() for t in (dout, q, k, v, out))
    grads = _flydsl_interface().flash_attn_bwd(dout, q, k, v, out, lse, softmax_scale, causal, deterministic)
    return tuple(_in_layout(g, layout) for g, layout in zip(grads, layouts))


@_eager_custom_op("primus_turbo::flash_attn_hk_gfx1250_forward")
def _forward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float, causal: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    return _run_forward(q, k, v, softmax_scale, causal)


@_forward.register_fake
def _(q, k, v, softmax_scale, causal):
    b, sq, hq, _ = q.shape
    out = _empty(q, (b, sq, hq, v.shape[-1]), _out_layout(q))
    return out, q.new_empty((b, hq, sq), dtype=torch.float32)


@_eager_custom_op("primus_turbo::flash_attn_hk_gfx1250_backward")
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _run_backward(dout, q, k, v, out, lse, softmax_scale, causal, deterministic)


@_backward.register_fake
def _(dout, q, k, v, out, lse, softmax_scale, causal, deterministic):
    return tuple(_empty(t, t.shape, _out_layout(t)) for t in (q, k, v))


def _scale(softmax_scale: Optional[float], head_dim: int) -> float:
    return float(head_dim**-0.5 if softmax_scale is None else softmax_scale)


def flash_attn_hk_gfx1250_forward_impl(q, k, v, softmax_scale=None, causal=True):
    """Returns (out [b, sq, hq, d], lse [b, hq, sq] fp32 natural log), from HK or FlyDSL.

    q / k / v: [b, s, h, d] in any byte order, handed over as ``hk_gfx1250_operands`` says (a caller that
    keeps the tensors for the backward prepares them with it first; then nothing is copied here). out comes
    in q's layout as handed over: "sbhd" only when the kernels take strides and q is SBHD storage."""
    q, k, v = hk_gfx1250_operands("fwd", q, k, v)
    return _forward(q, k, v, _scale(softmax_scale, q.shape[-1]), bool(causal))


def flash_attn_hk_gfx1250_backward_impl(
    dout, q, k, v, out, lse, softmax_scale=None, causal=True, deterministic=False
):
    """Returns (dq, dk, dv), [b, s, h, d] in the layouts of q / k / v as handed over; dk/dv have k's heads.

    q/k/v/out/lse are what the forward saw and returned (either family's); dout may arrive in any byte
    order. dout / q / k / v / out go to the op as ``hk_gfx1250_operands`` says (deterministic=True runs
    FlyDSL's deterministic backward, which reads contiguous BSHD: all of them contiguous then)."""
    dout, q, k, v, out = hk_gfx1250_operands("bwd", dout, q, k, v, out, strided=not deterministic)
    return _backward(
        dout,
        q,
        k,
        v,
        out,
        lse,
        _scale(softmax_scale, q.shape[-1]),
        bool(causal),
        bool(deterministic),
    )
