###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MXFP8 routed experts on gfx1250, strict mode (a fallback raises): grouped_gemm_fp8, and grouped_mlp_fp8 with its
GLU / dGLU in two phases (the grouped MX GEMM, then the FlyDSL GLU / dGLU + dual-quant kernel), against BF16, at a
toy shape and the DSv3 expert shape (32 experts x 2048 tokens, H 7168, I 2048), balanced and skewed (empty experts).
The grouped GEMMs run on gfx1250's FlyDSL MX kernels ("default"; skipped while they are missing) or on Triton, pinned
(the fallback)."""

import pytest
import torch
import torch.nn.functional as F

import primus_turbo.common.envs as envs
from primus_turbo.pytorch.core.backend import (
    BackendType,
    GlobalBackendManager,
    PrecisionType,
)
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScaleDtype,
    ScalingGranularity,
)
from primus_turbo.pytorch.core.utils import is_gfx1250
from primus_turbo.pytorch.ops import grouped_gemm_fp8
from primus_turbo.pytorch.ops.grouped_mlp_fp8 import grouped_mlp_fp8
from tests.pytorch.ref.mxfp8_ref import snr_db

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not is_gfx1250(), reason="gfx1250 MXFP8")

MX = Float8QuantConfig(
    format=Format.E4M3,
    granularity=ScalingGranularity.MX_BLOCKWISE,
    scale_dtype=ScaleDtype.E8M0,
    block_size=32,
)
# (G, tokens, H, I): toy, DSv3 routed experts at T = 8192 (top-8)
SHAPES = [(4, 1000, 512, 256), (32, 65536, 7168, 2048)]


@pytest.fixture(params=["default", "triton"])
def backend(request, monkeypatch):
    monkeypatch.setenv("PRIMUS_TURBO_MX_STRICT", "2")
    monkeypatch.delenv("PRIMUS_TURBO_MXFP8_SCALE_RULE", raising=False)
    envs.reset_cache()
    GlobalBackendManager.reset()
    if request.param == "triton":
        GlobalBackendManager.set_grouped_gemm_backend(BackendType.TRITON, PrecisionType.FP8)
    else:
        from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import mx1250_grouped

        if None in mx1250_grouped():
            pytest.skip("gemm_mxfp8_kernel_gfx1250 has no grouped mode yet")
    yield request.param
    GlobalBackendManager.reset()


def _group_lens(G, M, skew):
    if not skew:
        return torch.full((G,), M // G, dtype=torch.int64, device="cuda")
    w = torch.tensor(
        [(7 * g) % 11 for g in range(G)], dtype=torch.float64
    )  # 0..10: empty experts, ragged sizes
    lens = (w / w.sum() * M).floor().long()
    lens[int(w.argmax())] += M - int(lens.sum())
    return lens.cuda()


def _randn(*shape, scale=1.0, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(*shape, device="cuda", generator=g) * scale).bfloat16()


def _check(name, got, ref, snr_min):
    s = snr_db(got, ref)
    cos = F.cosine_similarity(got.double().flatten(), ref.double().flatten(), dim=0).item()
    assert s >= snr_min and cos >= 0.995, f"{name}: SNR {s:.2f} dB, cos {cos:.5f}"


@pytest.mark.parametrize("skew", [False, True])
@pytest.mark.parametrize("G, M, H, I", SHAPES)
def test_grouped_gemm(backend, G, M, H, I, skew):
    """fc1-shaped grouped_gemm_fp8: y / dx / dw vs per-expert BF16 GEMMs (SNR >= 25 dB)."""
    lens = _group_lens(G, M, skew)
    x, w, dy = _randn(M, H, seed=1), _randn(G, 2 * I, H, scale=H**-0.5, seed=2), _randn(M, 2 * I, seed=3)
    xg, wg = x.clone().requires_grad_(), w.clone().requires_grad_()
    out = grouped_gemm_fp8(xg, wg, lens, trans_b=True, config=MX)
    out.backward(dy)
    ref_y, ref_dx, ref_dw = torch.empty_like(out), torch.empty_like(x), torch.zeros_like(w)
    o = 0
    for g, n in enumerate(lens.tolist()):
        ref_y[o : o + n] = x[o : o + n] @ w[g].t()
        ref_dx[o : o + n] = dy[o : o + n] @ w[g]
        ref_dw[g] = dy[o : o + n].t() @ x[o : o + n]
        o += n
    for name, got, ref in (("y", out, ref_y), ("dx", xg.grad, ref_dx), ("dw", wg.grad, ref_dw)):
        _check(name, got, ref, 25)


def _mlp_bf16(x, w1, w2, lens, probs):
    """The BF16 experts: per expert fc1 -> silu(gate) * up * prob -> fc2."""
    out, o = [], 0
    for g, n in enumerate(lens.tolist()):
        gate, up = (x[o : o + n] @ w1[g].t()).chunk(2, dim=-1)
        act = (F.silu(gate.float()) * up.float() * probs[o : o + n, None]).bfloat16()
        out.append(act @ w2[g].t())
        o += n
    return torch.cat(out)


@pytest.mark.parametrize("G, M, H, I", SHAPES)
def test_grouped_chain(backend, G, M, H, I):
    """The unfused experts (``turbo_fused_grouped_gemm`` off): fc1 and fc2 by grouped_gemm_fp8 around a torch SwiGLU
    with probs, plain bf16 leaves (no cached weights): every leaf's gradient vs BF16 (SNR >= 20 dB)."""
    lens = _group_lens(G, M, True)
    x, dy = _randn(M, H, seed=1), _randn(M, H, seed=4)
    w1, w2 = _randn(G, 2 * I, H, scale=H**-0.5, seed=2), _randn(G, H, I, scale=I**-0.5, seed=3)
    probs = torch.rand(M, device="cuda", generator=torch.Generator(device="cuda").manual_seed(5))
    leaves = [t.clone().requires_grad_() for t in (x, w1, w2, probs)]
    gate, up = grouped_gemm_fp8(leaves[0], leaves[1], lens, trans_b=True, config=MX).chunk(2, dim=-1)
    act = (F.silu(gate.float()) * up.float() * leaves[3][:, None]).bfloat16()
    grouped_gemm_fp8(act, leaves[2], lens, trans_b=True, config=MX).backward(dy)
    refs = [t.clone().requires_grad_() for t in (x, w1, w2, probs)]
    _mlp_bf16(*refs[:3], lens, refs[3]).backward(dy)
    for name, got, ref in zip(("dx", "dw1", "dw2", "dprobs"), leaves, refs):
        assert got.grad is not None, name
        _check(name, got.grad, ref.grad, 20)


@pytest.mark.parametrize("skew", [False, True])
@pytest.mark.parametrize("G, M, H, I", SHAPES)
def test_grouped_mlp(backend, G, M, H, I, skew):
    """grouped_mlp_fp8 (fused GLU with probs): y / dx / dprobs / dw1 / dw2 vs BF16 (SNR >= 20 dB, cos >= 0.995)."""
    lens = _group_lens(G, M, skew)
    x, dy = _randn(M, H, seed=1), _randn(M, H, seed=4)
    w1, w2 = _randn(G, 2 * I, H, scale=H**-0.5, seed=2), _randn(G, H, I, scale=I**-0.5, seed=3)
    probs = torch.rand(M, device="cuda", generator=torch.Generator(device="cuda").manual_seed(5))
    leaves = [t.clone().requires_grad_() for t in (x, w1, w2, probs)]
    y = grouped_mlp_fp8(leaves[0], leaves[1], leaves[2], lens, leaves[3], trans_w1=True, trans_w2=True, config=MX,
                        activation="silu")  # fmt: skip
    y.backward(dy)
    refs = [t.clone().requires_grad_() for t in (x, w1, w2, probs)]
    ref_y = _mlp_bf16(refs[0], refs[1], refs[2], lens, refs[3])
    ref_y.backward(dy)
    _check("y", y, ref_y, 20)
    for name, got, ref in zip(("dx", "dw1", "dw2", "dprobs"), leaves, refs):
        _check(name, got.grad, ref.grad, 20)
