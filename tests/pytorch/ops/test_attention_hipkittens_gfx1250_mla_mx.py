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
    """Stand-ins for the MX ops on CPU tensors: the fp32 reference, and the MX operands as placeholders."""

    def fwd(q, k, v, softmax_scale):
        calls.append(("hk_mx_fwd", tuple(q.shape)))
        out, lse, *_ = _reference(q, k, v, torch.zeros(*q.shape[:3], D_V), softmax_scale)
        b, _, h, _ = q.shape
        return (
            out.to(q.dtype),
            lse,
            lse.clone(),
            *(torch.zeros(1) for _ in range(4)),
            torch.zeros(b, h, D_QK),
        )

    def prepare(q, k, lse, softmax_scale):
        calls.append(("hk_mx_prepare", tuple(q.shape)))
        return (lse.clone(), *(torch.zeros(1) for _ in range(5)))

    def bwd(dout, q, k, v, out, lse_c, q8, qs, k8, ks, kbar, softmax_scale):
        calls.append(("hk_mx_bwd", tuple(q.shape)))
        return tuple(g.to(q.dtype) for g in _reference(q, k, v, dout, softmax_scale)[2:])

    monkeypatch.setattr(mla_impl, "_hk_mx_unavailable_reason", lambda: None)
    monkeypatch.setattr(mla_impl, "_forward_hk_mx_op", fwd)
    monkeypatch.setattr(mla_impl, "_prepare_bwd_hk_mx_op", prepare)
    monkeypatch.setattr(mla_impl, "_backward_hk_mx_op", bwd)


_CALLER = (2, 256, 2, D_QK)  # hk_mx takes the caller's [b, s, h, d] tensors
_FOLDED = (1, 256, 4, D_QK)  # the bf16 kernels: b > 1 SBHD folded into the heads


@pytest.mark.parametrize(
    ("seqlen", "fwd", "bwd", "expected"),
    [
        (256, "hk_mx", "hk_mx", [("hk_mx_fwd", _CALLER), ("hk_mx_bwd", _CALLER)]),
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
    """Each pass takes its own setting; an hk_mx forward hands any backward its tensors (in the bf16 kernels'
    layout for those), an hk_mx backward after a bf16 forward quantizes the forward's q / k first, and a pass hk_mx
    cannot take runs as hk would."""
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
