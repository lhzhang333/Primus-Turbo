###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Flash attention for gfx1250 (MI455X) at the DeepSeek-V3 MLA head dims, on FlyDSL or HIP kernels.

q/k head dim 192, v head dim 128, bf16, MHA, bottom-right causal, any softmax scale; no dropout,
bias, alibi, sink or sliding window. The kernels take contiguous [b, s, h, d] tensors. b > 1 SBHD
storage (Megatron's layout) is handed over as [1, s, b*h, d], the same bytes with the batch folded
into the heads, so it is not copied. Dispatch and autograd wiring belong to the caller.

Implementations (``PRIMUS_TURBO_MLA_FWD_IMPL`` / ``PRIMUS_TURBO_MLA_BWD_IMPL``, read at import): ``flydsl``
(default) or ``hk``, the HIP kernels of ``primus_turbo/hipkittens/gfx1250_mla`` (same layouts; the ``hk`` forward is
bitwise equal to the FlyDSL one, so either backward takes either forward's out / lse). A pass ``hk`` cannot take
(``hk_mla_fwd_unsupported_reason`` / ``hk_mla_bwd_unsupported_reason``) runs on FlyDSL, with one warning per reason.

``hk_mx`` (development): the MXFP8 HIP kernels of ``primus_turbo/hipkittens/gfx1250_mla_mx`` (E4M3 operands with
E8M0 scales per 32 along each GEMM's reduction dim, quantized from the bf16 inputs on the fly; k centred). They take
[b, s, h, d] tensors with any b / s / h strides (no fold, no copy). An ``hk_mx`` forward saves its MX operands for the
backward; any backward takes any forward (a bf16 backward reads its q / k / v / out / lse, an ``hk_mx`` backward
after a bf16 forward quantizes q / k itself). A pass ``hk_mx`` cannot take (``hk_mx_mla_unsupported_reason``) runs
as ``hk`` would.
"""

from typing import Optional, Tuple

import torch

import primus_turbo.common.envs as envs
from primus_turbo.common.logger import logger
from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    _eager_custom_op,
)

D_QK = 192
D_V = 128

MLA_IMPLS = ("flydsl", "hk", "hk_mx")
MLA_FWD_IMPL_ENV = "PRIMUS_TURBO_MLA_FWD_IMPL"
MLA_BWD_IMPL_ENV = "PRIMUS_TURBO_MLA_BWD_IMPL"
# Read once at import, like the kernels' other switches: the passes route on module constants, which
# torch.compile traces without a guard on the environment.
MLA_FWD_IMPL_SETTING = envs.PRIMUS_TURBO_MLA_FWD_IMPL
MLA_BWD_IMPL_SETTING = envs.PRIMUS_TURBO_MLA_BWD_IMPL


def _impl(env: str, setting: str) -> str:
    impl = setting.strip().lower() or "flydsl"
    if impl not in MLA_IMPLS:
        raise ValueError(f"{env}={setting!r}: expected one of {MLA_IMPLS}")
    return impl


def mla_fwd_impl() -> str:
    """The MLA forward implementation: "flydsl" (default, also for an empty setting), "hk" or "hk_mx"."""
    return _impl(MLA_FWD_IMPL_ENV, MLA_FWD_IMPL_SETTING)


def mla_bwd_impl() -> str:
    """The MLA backward implementation: "flydsl" (default, also for an empty setting), "hk" or "hk_mx"."""
    return _impl(MLA_BWD_IMPL_ENV, MLA_BWD_IMPL_SETTING)


def _fwd():
    # Imported on first use: gfx1250-only kernels, never loaded on other archs.
    from primus_turbo.flydsl.attention.gfx1250_mla import fwd

    return fwd


def _bwd():
    from primus_turbo.flydsl.attention.gfx1250_mla import bwd

    return bwd


def is_mla_head_dims(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    """Whether q/k/v carry the DeepSeek-V3 MLA head dims (qk 192, v 128)."""
    return q.shape[-1] == D_QK and k.shape[-1] == D_QK and v.shape[-1] == D_V


def flydsl_gfx1250_mla_unsupported_reason(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool
) -> Optional[str]:
    """Why the gfx1250 MLA kernels cannot take these [b, s, h, d]-shaped tensors, or None.

    Covers the backward too: a training step needs both passes.
    """
    if k.device != q.device or v.device != q.device:
        return f"q/k/v must be on one device, got {q.device}/{k.device}/{v.device}"
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        return f"dtype must be bfloat16, got {q.dtype}/{k.dtype}/{v.dtype}"
    if any(t.dim() != 4 for t in (q, k, v)):
        return "q/k/v must be 4-D [b, s, h, d]"
    if not is_mla_head_dims(q, k, v):
        return f"head dims must be qk {D_QK} / v {D_V}, got {q.shape[-1]}/{k.shape[-1]}/{v.shape[-1]}"
    b, sq, h, _ = q.shape
    _, skv, hkv, _ = k.shape
    if k.shape[0] != b or v.shape[:3] != k.shape[:3]:
        return f"k {tuple(k.shape)} / v {tuple(v.shape)} must match each other and q's batch {b}"
    if h != hkv:
        return f"MHA only (heads_q == heads_kv), got {h} / {hkv}"
    if not causal:
        return "causal only"
    return _bwd().shape_unsupported_reason(b, sq, skv, h)


@torch.compiler.assume_constant_result
def _hk_unavailable_reason(pass_: str) -> Optional[str]:
    """None when the HIP kernels of ``pass_`` ("fwd" / "bwd") are built, byte-identical to the ones
    ``manifest.json`` pins, and the HIP runtime torch loaded can launch them; else why not, said once in the log.
    Marked constant for torch.compile, whose Dynamo then calls it instead of tracing it (the gate runs inside
    compiled regions): the answer cannot change within a process, and the file reads and the log line are not
    traceable."""
    from primus_turbo.hipkittens.gfx1250_mla import hiprt, hsaco

    reason = hsaco.unavailable_reason(hsaco.FWD_KERNELS if pass_ == "fwd" else hsaco.BWD_KERNELS)
    reason = reason or hiprt.runtime_unavailable_reason()
    if reason is not None:
        logger.warning(f"the gfx1250 HIP MLA {pass_} kernels are unavailable: {reason}", once=True)
    return reason


def _hk_unsupported_reason(q, k, v, causal, pass_, softmax_scale=None):
    """The FlyDSL MLA rules (a training step needs both passes, on either implementation), then the HIP kernels'
    shape (and, backward, scale) rules, then their code objects."""
    reason = flydsl_gfx1250_mla_unsupported_reason(q, k, v, causal)
    if reason is not None:
        return reason
    from primus_turbo.hipkittens.gfx1250_mla import limits

    b, sq, h, _ = q.shape
    skv, hkv = k.shape[1], k.shape[2]
    if pass_ == "fwd":
        reason = limits.fwd_shape_unsupported_reason(b, sq, skv, h, hkv, causal)
    else:
        reason = limits.bwd_shape_unsupported_reason(b, sq, skv, h, hkv, causal)
        reason = reason or limits.bwd_scale_unsupported_reason(_scale(softmax_scale))
    return reason if reason is not None else _hk_unavailable_reason(pass_)


def hk_mla_fwd_unsupported_reason(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool
) -> Optional[str]:
    """Why the HIP MLA forward (``hk``) cannot take these [b, s, h, d]-shaped tensors, or None."""
    return _hk_unsupported_reason(q, k, v, causal, "fwd")


def hk_mla_bwd_unsupported_reason(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool, softmax_scale: Optional[float] = None
) -> Optional[str]:
    """Why the HIP MLA backward (``hk``) cannot take these [b, s, h, d]-shaped tensors and scale, or None."""
    return _hk_unsupported_reason(q, k, v, causal, "bwd", softmax_scale)


def _warn_hk_fallback(
    env: str, pass_name: str, reason: str, impl: str = "hk", runs_on: str = "FlyDSL"
) -> None:
    """Say once that ``impl`` could not take a pass. Skipped under torch.compile: Dynamo cannot trace a
    logging.Logger call (unavailable kernels are still logged, by _hk_unavailable_reason)."""
    if not torch.compiler.is_compiling():
        logger.warning(f"{env}={impl}: this {pass_name} runs on {runs_on} ({reason})", once=True)


@torch.compiler.assume_constant_result
def _hk_mx_unavailable_reason() -> Optional[str]:
    """None when the MXFP8 HIP kernels are built (built now at first use) and the HIP runtime torch loaded can
    launch them; else why not, said once in the log. Constant for torch.compile, as _hk_unavailable_reason."""
    from primus_turbo.hipkittens.gfx1250_mla import hiprt
    from primus_turbo.hipkittens.gfx1250_mla_mx import mx

    reason = hiprt.runtime_unavailable_reason() or mx.unavailable_reason()
    if reason is not None:
        logger.warning(f"the gfx1250 MXFP8 HIP MLA kernels are unavailable: {reason}", once=True)
    return reason


def hk_mx_mla_unsupported_reason(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool
) -> Optional[str]:
    """Why the MXFP8 HIP MLA kernels (``hk_mx``, both passes) cannot take these [b, s, h, d]-shaped tensors, or
    None: the FlyDSL MLA rules, the MX kernels' shape rules, then their code objects."""
    reason = flydsl_gfx1250_mla_unsupported_reason(q, k, v, causal)
    if reason is not None:
        return reason
    from primus_turbo.hipkittens.gfx1250_mla_mx import mx

    reason = mx.unsupported_reason(q, k, v, causal)
    return reason if reason is not None else _hk_mx_unavailable_reason()


def _hk_fwd():
    from primus_turbo.hipkittens.gfx1250_mla import fwd

    return fwd


def _hk_bwd():
    from primus_turbo.hipkittens.gfx1250_mla import bwd

    return bwd


def _hk_mx():
    from primus_turbo.hipkittens.gfx1250_mla_mx import mx

    return mx


@_eager_custom_op("primus_turbo::flash_attn_flydsl_gfx1250_mla_forward")
def _forward_op(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float, causal: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _fwd().flash_attn_mla_fwd(q, k, v, softmax_scale, causal)


@_forward_op.register_fake
def _forward_op_fake(q, k, v, softmax_scale, causal):
    b, sq, h, _ = q.shape
    return q.new_empty((b, sq, h, D_V)), q.new_empty((b, h, sq), dtype=torch.float32)


@_eager_custom_op("primus_turbo::flash_attn_hk_gfx1250_mla_forward")
def _forward_hk_op(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float, causal: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _hk_fwd().hk_attn_fwd(q, k, v, softmax_scale, causal)


@_forward_hk_op.register_fake
def _forward_hk_op_fake(q, k, v, softmax_scale, causal):
    return _forward_op_fake(q, k, v, softmax_scale, causal)


# the MX backward's operands (gfx1250_mla_mx.mx): lse of the centred logits, q8, qs, k8, ks, kbar, q8t, qts, k8t, kts,
# v8, vsr
_N_MX_SAVED = 12
_MX_SAVED = Tuple[(torch.Tensor,) * _N_MX_SAVED]


@_eager_custom_op("primus_turbo::flash_attn_hk_mx_gfx1250_mla_forward")
def _forward_hk_mx_op(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _hk_mx().attn_fwd(q, k, v, softmax_scale, for_mx_bwd=False)


@_eager_custom_op("primus_turbo::flash_attn_hk_mx_gfx1250_mla_forward_train")
def _forward_hk_mx_train_op(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float
) -> Tuple[(torch.Tensor,) * (2 + _N_MX_SAVED)]:
    return _hk_mx().attn_fwd(q, k, v, softmax_scale, for_mx_bwd=True)


def _mx_saved_fake(q):
    b, sq, h, _ = q.shape
    u8, f32 = torch.uint8, torch.float32
    rows = lambda d: q.new_empty((b, h, sq, d), dtype=u8)  # noqa: E731
    cols = lambda d: (q.new_empty((b, h, d, sq), dtype=u8), q.new_empty((b, h, sq // 128, d, 4), dtype=u8))  # noqa: E731
    return (q.new_empty((b, h, sq), dtype=f32), rows(D_QK), rows(8), rows(D_QK), rows(8),
            q.new_empty((b, h, D_QK), dtype=f32), *cols(D_QK), *cols(D_QK), rows(D_V), rows(4))  # fmt: skip


def _mx_out_fake(q):
    b, sq, h, _ = q.shape
    return q.new_empty((sq, b, h, D_V)).transpose(0, 1), q.new_empty((b, h, sq), dtype=torch.float32)


@_forward_hk_mx_op.register_fake
def _forward_hk_mx_op_fake(q, k, v, softmax_scale):
    return _mx_out_fake(q)


@_forward_hk_mx_train_op.register_fake
def _forward_hk_mx_train_op_fake(q, k, v, softmax_scale):
    return (*_mx_out_fake(q), *_mx_saved_fake(q))


@_eager_custom_op("primus_turbo::flash_attn_hk_mx_gfx1250_mla_prepare_backward")
def _prepare_bwd_hk_mx_op(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, lse: torch.Tensor, softmax_scale: float
) -> _MX_SAVED:
    return _hk_mx().prepare_bwd(q, k, v, lse, softmax_scale)


@_prepare_bwd_hk_mx_op.register_fake
def _prepare_bwd_hk_mx_op_fake(q, k, v, lse, softmax_scale):
    return _mx_saved_fake(q)


@_eager_custom_op("primus_turbo::flash_attn_hk_mx_gfx1250_mla_backward")
def _backward_hk_mx_op(
    dout: torch.Tensor,
    out: torch.Tensor,
    lse_c: torch.Tensor,
    q8: torch.Tensor,
    qs: torch.Tensor,
    k8: torch.Tensor,
    ks: torch.Tensor,
    kbar: torch.Tensor,
    q8t: torch.Tensor,
    qts: torch.Tensor,
    k8t: torch.Tensor,
    kts: torch.Tensor,
    v8: torch.Tensor,
    vsr: torch.Tensor,
    softmax_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _hk_mx().attn_bwd(
        dout, out, lse_c, q8, qs, k8, ks, kbar, q8t, qts, k8t, kts, v8, vsr, softmax_scale
    )


@_backward_hk_mx_op.register_fake
def _backward_hk_mx_op_fake(
    dout, out, lse_c, q8, qs, k8, ks, kbar, q8t, qts, k8t, kts, v8, vsr, softmax_scale
):
    b, h, sq, _ = q8.shape
    return tuple(q8.new_empty((sq, b, h, d), dtype=torch.bfloat16).transpose(0, 1) for d in (D_QK, D_QK, D_V))


@_eager_custom_op("primus_turbo::flash_attn_flydsl_gfx1250_mla_backward")
def _backward_op(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _bwd().flash_attn_mla_bwd(dout, q, k, v, out, lse, softmax_scale, causal)


@_backward_op.register_fake
def _backward_op_fake(dout, q, k, v, out, lse, softmax_scale, causal):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


@_eager_custom_op("primus_turbo::flash_attn_hk_gfx1250_mla_backward")
def _backward_hk_op(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _hk_bwd().hk_attn_bwd(dout, q, k, v, out, lse, softmax_scale, causal)


@_backward_hk_op.register_fake
def _backward_hk_op_fake(dout, q, k, v, out, lse, softmax_scale, causal):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _scale(softmax_scale: Optional[float]) -> float:
    return float(D_QK**-0.5 if softmax_scale is None else softmax_scale)


def _to_kernel_layout(t: torch.Tensor, fold: bool) -> torch.Tensor:
    """[b, s, h, d] in any byte order -> contiguous [b, s, h, d], or [1, s, b*h, d] when folding."""
    if fold:
        b, s, h, d = t.shape
        return t.transpose(0, 1).contiguous().view(1, s, b * h, d)
    return t.contiguous()


def _from_kernel_layout(x: torch.Tensor, batch: int) -> torch.Tensor:
    """Inverse of _to_kernel_layout; a fold comes back as a [b, s, h, d] view of SBHD storage."""
    if x.shape[0] == batch:
        return x
    _, s, bh, d = x.shape
    return x.view(s, batch, bh // batch, d).transpose(0, 1)


def flash_attn_flydsl_gfx1250_mla_forward_impl(q, k, v, softmax_scale=None, causal=True):
    """Returns (out [b, sq, h, 128], lse [b, h, sq] fp32 natural log, saved).

    ``saved`` is what the backward impl takes after dout: the kernel-layout (q, k, v, out, lse), out and lse
    views of its last two. An ``hk_mx`` forward for an ``hk_mx`` backward saves (None, None, None, out, lse, *the MX
    operands of both passes, quantized once): no bf16 q / k / v. ``mla_fwd_impl()`` picks the kernels.
    """
    b, sq, h, _ = q.shape
    impl = mla_fwd_impl()
    fold = b > 1 and q.transpose(0, 1).is_contiguous()
    if impl == "hk_mx":
        reason = hk_mx_mla_unsupported_reason(q, k, v, bool(causal))
        if reason is None:
            scale = _scale(softmax_scale)
            if mla_bwd_impl() == "hk_mx":
                out, lse, *mx_saved = _forward_hk_mx_train_op(q, k, v, scale)
                return out, lse, (None, None, None, out, lse, *mx_saved)
            out, lse = _forward_hk_mx_op(q, k, v, scale)
            # a bf16 backward takes what a bf16 forward would have saved: the kernel-layout tensors
            qk, kk, vk, ok = (_to_kernel_layout(t, fold) for t in (q, k, v, out))
            return out, lse, (qk, kk, vk, ok, lse.reshape(qk.shape[0], qk.shape[2], sq))
        _warn_hk_fallback(MLA_FWD_IMPL_ENV, "forward", reason, "hk_mx", "hk")
        impl = "hk"
    qk, kk, vk = (_to_kernel_layout(t, fold) for t in (q, k, v))
    forward = _forward_op
    if impl == "hk":
        reason = hk_mla_fwd_unsupported_reason(qk, kk, vk, bool(causal))
        if reason is None:
            forward = _forward_hk_op
        else:
            _warn_hk_fallback(MLA_FWD_IMPL_ENV, "forward", reason)
    out, lse = forward(qk, kk, vk, _scale(softmax_scale), bool(causal))
    return _from_kernel_layout(out, b), lse.view(b, h, sq), (qk, kk, vk, out, lse)


def flash_attn_flydsl_gfx1250_mla_backward_impl(
    dout, q, k, v, out, lse, *mx_saved, softmax_scale=None, causal=True
):
    """q/k/v/out/lse(/mx_saved) are the forward impl's ``saved``; dout is [b, sq, h, 128] in any byte order.

    Returns (dq, dk, dv) shaped like the forward's q/k/v. ``mla_bwd_impl()`` picks the kernels.
    """
    batch = dout.shape[0]
    scale = _scale(softmax_scale)
    if mx_saved:  # an hk_mx forward made the operands (and checked the shapes)
        return _backward_hk_mx_op(dout, out, *mx_saved, scale)
    impl = mla_bwd_impl()
    if impl == "hk_mx":
        reason = hk_mx_mla_unsupported_reason(q, k, v, bool(causal))
        if reason is None:  # after a bf16 forward: quantize its kernel-layout q / k / v
            fold = q.shape[0] != batch
            mx_saved = _prepare_bwd_hk_mx_op(q, k, v, lse, scale)
            grads = _backward_hk_mx_op(_to_kernel_layout(dout, True) if fold else dout, out, *mx_saved, scale)
            return tuple(_from_kernel_layout(g, batch) for g in grads)
        _warn_hk_fallback(MLA_BWD_IMPL_ENV, "backward", reason, "hk_mx", "hk")
        impl = "hk"
    fold = q.shape[0] != batch  # the forward folded the batch into the heads
    backward = _backward_op
    if impl == "hk":
        reason = hk_mla_bwd_unsupported_reason(q, k, v, bool(causal), softmax_scale)
        if reason is None:
            backward = _backward_hk_op
        else:
            _warn_hk_fallback(MLA_BWD_IMPL_ENV, "backward", reason)
    grads = backward(_to_kernel_layout(dout, fold), q, k, v, out, lse, _scale(softmax_scale), bool(causal))
    return tuple(_from_kernel_layout(g, batch) for g in grads)
