###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MXFP8 HIP kernels of the gfx1250 (MI455X) MLA attention (``hk_mx``): primus_turbo/hipkittens/gfx1250_mla_mx.

Settings, shape rules and routing use meta / CPU tensors and stand-in kernels. The numeric tests need a gfx1250 card
and amdclang++ (the code objects are built at first use); their fp32 reference runs on the CPU. E4M3 operands bound
the accuracy: the floors are the recipe's (benchmark/hk_mla_mx/LOG.md: o ~23.5 dB, grads ~22 dB on randn), far under
bf16's.
"""

import pytest
import torch

from primus_turbo.pytorch.core.utils import is_gfx1250
from primus_turbo.pytorch.ops import flash_attn_func
from tests.pytorch.ops.test_attention_flydsl_gfx1250_mla import (
    D_QK,
    D_V,
    MLA_SCALE,
    _inputs,
    _meta,
    _on_gfx1250,
    _pinned_flydsl,
    _reference,
    _run,
    mla_impl,
)
from tests.pytorch.ops.test_attention_hipkittens_gfx1250_mla import (
    _settings,
    _stub_kernels,
    needs_hk,
)
from tests.pytorch.test_utils import compute_snr


def _mx():
    from primus_turbo.hipkittens.gfx1250_mla_mx import mx

    return mx


def _hk_mx_reason():
    if not (torch.cuda.is_available() and is_gfx1250()):
        return "needs a gfx1250 (MI455X) GPU"
    return _mx().unavailable_reason()  # builds the code objects (a few seconds, once per cache directory)


needs_hk_mx = pytest.mark.skipif(
    _hk_mx_reason() is not None,
    reason="needs a gfx1250 (MI455X) GPU and amdclang++ for the MXFP8 HIP MLA kernels",
)


# ---------------------------------------------------------------------------
# Settings, shape rules, routing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pass_", ["fwd", "bwd"])
@pytest.mark.parametrize("setting", ["hk_mx", " HK_MX "])
def test_hk_mx_setting(monkeypatch, pass_, setting):
    monkeypatch.setattr(mla_impl, f"MLA_{pass_.upper()}_IMPL_SETTING", setting)
    assert getattr(mla_impl, f"mla_{pass_}_impl")() == "hk_mx"


@pytest.mark.parametrize(
    ("dims", "causal", "match"),
    [
        ((1, 256, 256, 2), True, None),
        ((2, 4096, 4096, 128), True, None),
        ((1, 384, 384, 2), True, "multiple of 256"),
        ((1, 256, 512, 2), True, "seqlen_q == seqlen_kv"),
        ((1, 256, 256, 2), False, "causal only"),
    ],
)
def test_mx_shape_rules(dims, causal, match):
    q, k, v = _meta(*dims, layout="sbhd")
    reason = _mx().unsupported_reason(q, k, v, causal)
    assert reason is None if match is None else match in reason


def _stub_mx_kernels(monkeypatch, calls):
    """Stand-ins for the MX ops on CPU tensors: the fp32 reference; the saved MX operands carry q / k / v in the q8 / k8
    / v8 slots (placeholders elsewhere) so the backward stand-in can compute the reference gradients."""

    def placeholders(q, k, v, lse):
        z = torch.zeros(1)
        return (lse.clone(), q, z, k, z, z, z, z, z, z, v, z)

    def ref_out(q, k, v, softmax_scale):
        out, lse, *_ = _reference(q, k, v, torch.zeros(*q.shape[:3], D_V), softmax_scale)
        return out.to(q.dtype), lse

    def fwd(q, k, v, softmax_scale):
        calls.append(("hk_mx_fwd", tuple(q.shape)))
        return ref_out(q, k, v, softmax_scale)

    def fwd_train(q, k, v, softmax_scale):
        calls.append(("hk_mx_fwd_train", tuple(q.shape)))
        out, lse = ref_out(q, k, v, softmax_scale)
        return (out, lse, *placeholders(q, k, v, lse))

    def prepare(q, k, v, lse, softmax_scale):
        calls.append(("hk_mx_prepare", tuple(q.shape)))
        return placeholders(q, k, v, lse)

    def bwd(dout, out, lse_c, q8, qs, k8, ks, kbar, q8t, qts, k8t, kts, v8, vsr, softmax_scale):
        calls.append(("hk_mx_bwd", tuple(q8.shape)))
        return tuple(g.to(q8.dtype) for g in _reference(q8, k8, v8, dout, softmax_scale)[2:])

    monkeypatch.setattr(mla_impl, "_hk_mx_unavailable_reason", lambda: None)
    monkeypatch.setattr(mla_impl, "_forward_hk_mx_op", fwd)
    monkeypatch.setattr(mla_impl, "_forward_hk_mx_train_op", fwd_train)
    monkeypatch.setattr(mla_impl, "_prepare_bwd_hk_mx_op", prepare)
    monkeypatch.setattr(mla_impl, "_backward_hk_mx_op", bwd)


_CALLER = (2, 256, 2, D_QK)  # hk_mx takes the caller's [b, s, h, d] tensors
_FOLDED = (1, 256, 4, D_QK)  # the bf16 kernels: b > 1 SBHD folded into the heads


@pytest.mark.parametrize(
    ("seqlen", "fwd", "bwd", "expected"),
    [
        (256, "hk_mx", "hk_mx", [("hk_mx_fwd_train", _CALLER), ("hk_mx_bwd", _CALLER)]),
        (256, "hk_mx", "hk", [("hk_mx_fwd", _CALLER), ("hk_bwd", _FOLDED)]),
        (256, "hk_mx", "flydsl", [("hk_mx_fwd", _CALLER), ("flydsl_bwd", _FOLDED)]),
        (256, "hk", "hk_mx", [("hk_fwd", _FOLDED), ("hk_mx_prepare", _FOLDED), ("hk_mx_bwd", _FOLDED)]),
        (
            256,
            "flydsl",
            "hk_mx",
            [("flydsl_fwd", _FOLDED), ("hk_mx_prepare", _FOLDED), ("hk_mx_bwd", _FOLDED)],
        ),
        # seqlen 384 is no multiple of 256: both passes run as hk would
        (384, "hk_mx", "hk_mx", [("hk_fwd", (1, 384, 4, D_QK)), ("hk_bwd", (1, 384, 4, D_QK))]),
    ],
)
def test_routes_and_falls_back(monkeypatch, seqlen, fwd, bwd, expected):
    """Each pass takes its own setting. An hk_mx forward for an hk_mx backward saves the MX operands of both passes
    (no bf16 q / k / v); for a bf16 backward it saves what a bf16 forward would (kernel layout). An hk_mx backward after
    a bf16 forward quantizes the forward's q / k / v first. A pass hk_mx cannot take runs as hk would."""
    _on_gfx1250(monkeypatch)
    _settings(monkeypatch, fwd, bwd)
    calls = []
    _stub_kernels(monkeypatch, calls)
    _stub_mx_kernels(monkeypatch, calls)
    b, h = 2, 2
    q, k, v, dout = _inputs(b, seqlen, seqlen, h, layout="sbhd")
    qg, kg, vg = (t.clone().requires_grad_() for t in (q, k, v))
    with _pinned_flydsl():
        out = flash_attn_func(qg, kg, vg, softmax_scale=MLA_SCALE, causal=True)
    out.backward(dout)
    assert calls == expected
    ref = _reference(q, k, v, dout, MLA_SCALE)
    for name, x, r in zip(("out", "dq", "dk", "dv"), (out, qg.grad, kg.grad, vg.grad), ref[:1] + ref[2:]):
        assert compute_snr(r, x.detach().float()) > 40, name


# ---------------------------------------------------------------------------
# Numerics (gfx1250)
# ---------------------------------------------------------------------------

# dB against the fp32 reference on the bf16 inputs (randn). A bf16 pass keeps bf16 HK's floors for what it alone
# computes; an MX forward's out / lse error reaches every gradient.
_MX_FLOORS = {"out": 20.0, "lse": 38.0, "dq": 18.0, "dk": 18.0, "dv": 19.0}
_BF16_FLOORS = {"out": 48.0, "lse": 100.0}


def _check_mx(got, ref, fwd):
    for name, x, r in zip(("out", "lse", "dq", "dk", "dv"), got, ref):
        x = x.detach().float().cpu()
        assert x.shape == r.shape and torch.isfinite(x).all(), name
        floor = _MX_FLOORS[name] if fwd == "hk_mx" or name not in _BF16_FLOORS else _BF16_FLOORS[name]
        snr = compute_snr(r, x)
        assert snr >= floor, f"{name}: {snr:.1f} dB < {floor} dB"


_NUMERIC_CASES = [
    pytest.param((2, 512, 512, 4), "sbhd", id="b2_sbhd"),
    pytest.param((1, 1024, 1024, 2), "bshd", id="b1_bshd_s1024"),
    pytest.param((2, 256, 256, 2), "bhsd", id="b2_bhsd_strided"),
]


@needs_hk_mx
@pytest.mark.parametrize(("fwd", "bwd"), [("hk_mx", "hk_mx"), ("hk_mx", "flydsl"), ("flydsl", "hk_mx")])
@pytest.mark.parametrize(("dims", "layout"), _NUMERIC_CASES)
def test_matches_reference(monkeypatch, fwd, bwd, dims, layout):
    _settings(monkeypatch, fwd, bwd)
    q, k, v, dout = _inputs(*dims, layout=layout)
    _check_mx(_run(q, k, v, dout), _reference(q, k, v, dout, MLA_SCALE), fwd)


@needs_hk_mx
@needs_hk
@pytest.mark.parametrize(("fwd", "bwd"), [("hk_mx", "hk"), ("hk", "hk_mx")])
def test_mixes_with_bf16_hk(monkeypatch, fwd, bwd):
    _settings(monkeypatch, fwd, bwd)
    q, k, v, dout = _inputs(2, 768, 768, 2, layout="sbhd")
    _check_mx(_run(q, k, v, dout), _reference(q, k, v, dout, MLA_SCALE), fwd)


@needs_hk_mx
def test_deterministic_and_layout_independent(monkeypatch):
    """A repeated call is bitwise identical, and the byte order of the inputs does not change the arithmetic."""
    _settings(monkeypatch, "hk_mx", "hk_mx")
    q, k, v, dout = _inputs(2, 512, 512, 2, layout="sbhd")
    first = _run(q, k, v, dout)
    again = _run(q, k, v, dout)
    bshd = _run(*(t.contiguous() for t in (q, k, v, dout)))
    for a, b, c in zip(first, again, bshd):
        assert torch.equal(a, b) and torch.equal(a, c)


@needs_hk_mx
def test_production_shape_head_slice(monkeypatch):
    """The DeepSeek-V3 sequence length (4096) on a few heads: every q block, kv tile and the longest-first order."""
    from tests.pytorch.ops.test_attention_flydsl_gfx1250_mla import _reference_by_heads

    _settings(monkeypatch, "hk_mx", "hk_mx")
    q, k, v, dout = _inputs(1, 4096, 4096, 4, layout="sbhd")
    _check_mx(_run(q, k, v, dout), _reference_by_heads(q, k, v, dout, MLA_SCALE), "hk_mx")


# ---------------------------------------------------------------------------
# The fused-RoPE path (mx.mla_rope_attention)
# ---------------------------------------------------------------------------


def _rope_ref(pe, cos, sin):
    """Megatron's MLA rotary (interleaved pairs in, halves out), fp32 per op, bf16 out."""
    x = pe.float()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos.float(), sin.float()
    return torch.cat([x1 * c[..., :32] - x2 * s[..., :32], x2 * c[..., 32:] + x1 * s[..., 32:]], -1).to(torch.bfloat16)


@needs_hk_mx
@pytest.mark.parametrize(("b", "s", "h"), [(1, 256, 2), (2, 512, 4)])
def test_rope_fused_matches_unfused(b, s, h):
    """mla_rope_attention == RoPE (torch) followed by the hk_mx attention: out bitwise, dq within 1 bf16 ulp, dkv bitwise,
    dk_pe within the reference's own bf16 rounding of the head sum."""
    mx = _mx()
    g = torch.Generator(device="cuda").manual_seed(11)
    q = torch.randn(s, b, h, 192, generator=g, device="cuda").to(torch.bfloat16)
    kv = torch.randn(s, b, h, 256, generator=g, device="cuda").to(torch.bfloat16)
    kvc = torch.randn(s, b, 576, generator=g, device="cuda").to(torch.bfloat16)
    ang = torch.rand(s, 64, generator=g, device="cuda") * 6.0
    cos = ang.cos().to(torch.bfloat16).reshape(s, 1, 1, 64).contiguous()
    sin = ang.sin().to(torch.bfloat16).reshape(s, 1, 1, 64).contiguous()
    dout = torch.randn(s, b, h, 128, generator=g, device="cuda").to(torch.bfloat16)

    q1, kv1, kvc1 = (t.clone().requires_grad_() for t in (q, kv, kvc))
    out1 = mx.mla_rope_attention(q1, kv1, kvc1[..., 512:].unsqueeze(2), cos, sin, MLA_SCALE)
    out1.backward(dout)

    class _Unfused(torch.autograd.Function):
        @staticmethod
        def forward(ctx, qq, kk, vv):
            r = mx.attn_fwd(qq.transpose(0, 1), kk.transpose(0, 1), vv.transpose(0, 1), MLA_SCALE, True)
            ctx.save_for_backward(r[0], *r[2:])
            return r[0].transpose(0, 1)

        @staticmethod
        def backward(ctx, do):
            o, *saved = ctx.saved_tensors
            dq, dk, dv = mx.attn_bwd(do.transpose(0, 1), o, *saved, MLA_SCALE)
            return dq.transpose(0, 1), dk.transpose(0, 1), dv.transpose(0, 1)

    q2, kv2, kvc2 = (t.clone().requires_grad_() for t in (q, kv, kvc))
    kpe2 = kvc2[..., 512:].unsqueeze(2)
    query = torch.cat([q2[..., :128], _rope_ref(q2[..., 128:], cos, sin)], -1)
    key = torch.cat([kv2[..., :128], _rope_ref(kpe2, cos, sin).expand(s, b, h, 64)], -1)
    out2 = _Unfused.apply(query.contiguous(), key.contiguous(), kv2[..., 128:].contiguous())
    out2.backward(dout)

    assert torch.equal(out1, out2)
    assert torch.equal(kv1.grad, kv2.grad)
    dq_diff = (q1.grad.float() - q2.grad.float()).abs()
    assert bool((dq_diff <= q2.grad.float().abs() * 2.0**-7).all())
    assert compute_snr(kvc2.grad[..., 512:].float(), kvc1.grad[..., 512:].float()) > 45
    assert kvc1.grad[..., :512].abs().max() == 0


# ---------------------------------------------------------------------------
# The backward's dQ: a GEMM over the dS that the dK / dV kernel stores (default) or the recompute kernel
# ---------------------------------------------------------------------------


def _setenv(monkeypatch, name, value):
    from primus_turbo.common import envs

    monkeypatch.setenv(name, value)
    envs.reset_cache(name)  # envs caches a value at its first read


def _mx_fwd_bwd(b, s, h, seed=7):
    mx = _mx()
    g = torch.Generator(device="cuda").manual_seed(seed)
    q, k, v, dout = (torch.randn(b, s, h, d, generator=g, device="cuda").to(torch.bfloat16) for d in (D_QK, D_QK, D_V, D_V))
    res = mx.attn_fwd(q, k, v, MLA_SCALE, True)
    return mx, (q, k, v, dout), res


@needs_hk_mx
@pytest.mark.parametrize(("b", "s", "h"), [(2, 512, 8), (1, 768, 4), (1, 1024, 2)])  # paired dQ grid, odd q blocks, b h % 8
def test_dq_gemm_matches_recompute(monkeypatch, b, s, h):
    """The three dQ paths against each other: recompute (DQ_GEMM=0), rung b7 (bf16 dS, DS_FP8=0) and rung b8 (fp8 dS on
    32 x 32 block scales, the default). dV is bitwise the same in all three; dK is bitwise between recompute and b7 and
    within the block-scale rounding for b8 (dK's dS operand); dQ differs by the dS quantization only."""
    mx, (q, k, v, dout), res = _mx_fwd_bwd(b, s, h)
    out = {}
    for name, gemm, fp8 in (("b8", "1", "1"), ("b7", "1", "0"), ("rc", "0", "1")):
        _setenv(monkeypatch, "PRIMUS_TURBO_MLA_MX_DQ_GEMM", gemm)
        _setenv(monkeypatch, "PRIMUS_TURBO_MLA_MX_DS_FP8", fp8)
        out[name] = mx.attn_bwd(dout, res[0], *res[2:], MLA_SCALE)
    (dq8, dk8, dv8), (dq7, dk7, dv7), (dq0, dk0, dv0) = out["b8"], out["b7"], out["rc"]
    assert torch.equal(dv8, dv0) and torch.equal(dv7, dv0)
    assert torch.equal(dk7, dk0)
    assert not torch.equal(dq7, dq0) and not torch.equal(dq8, dq7)  # the three paths ran
    assert compute_snr(dk0.float(), dk8.float()) > 50
    assert compute_snr(dq0.float(), dq7.float()) > 30 and compute_snr(dq0.float(), dq8.float()) > 30
    ref = _reference(q, k, v, dout, MLA_SCALE)
    for x in (dq0, dq7, dq8):
        assert compute_snr(ref[2].float().cpu(), x.float().cpu()) >= _MX_FLOORS["dq"]
    assert compute_snr(ref[3].float().cpu(), dk8.float().cpu()) >= _MX_FLOORS["dk"]


@needs_hk_mx
def test_dq_gemm_workspace_cap(monkeypatch):
    """A dS workspace over PRIMUS_TURBO_MLA_MX_DS_MAX_BYTES runs the recompute kernel (bitwise as with the GEMM off)."""
    mx, (q, k, v, dout), res = _mx_fwd_bwd(2, 512, 8)
    _setenv(monkeypatch, "PRIMUS_TURBO_MLA_MX_DQ_GEMM", "0")
    ref = mx.attn_bwd(dout, res[0], *res[2:], MLA_SCALE)
    _setenv(monkeypatch, "PRIMUS_TURBO_MLA_MX_DQ_GEMM", "1")
    gemm = mx.attn_bwd(dout, res[0], *res[2:], MLA_SCALE)
    _setenv(monkeypatch, "PRIMUS_TURBO_MLA_MX_DS_MAX_BYTES", "1")
    capped = mx.attn_bwd(dout, res[0], *res[2:], MLA_SCALE)
    assert not torch.equal(gemm[0], ref[0])  # the cap is what switches the path back
    assert all(torch.equal(a, c) for a, c in zip(ref, capped))
