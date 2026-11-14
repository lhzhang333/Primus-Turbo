###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MXFP8 routed experts on gfx1250, strict mode (a fallback raises): grouped_gemm_fp8, and grouped_mlp_fp8 with its
GLU / dGLU in two phases (the grouped MX GEMM, then the FlyDSL GLU / dGLU + dual-quant kernel), against BF16, at a
toy shape (and once per test at the DSv3 expert shape: 32 experts x 2048 tokens, H 7168, I 2048), balanced and skewed
(empty experts), and as Primus runs them (cached weights, wgrads fused into main_grad).
The grouped GEMMs run on gfx1250's FlyDSL MX kernels ("default") or on Triton, pinned (the fallback)."""

import itertools

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
    float8_e4m3,
)
from primus_turbo.pytorch.core.quantized_tensor import (
    QuantizedTensorPair,
    create_quantized_weight,
)
from primus_turbo.pytorch.core.utils import is_gfx1250
from primus_turbo.pytorch.ops import (
    grouped_gemm_fp8,
    mx_weight_layout_recipe,
    mx_wgrad_accum_supported,
)
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
TOY, DSV3 = (4, 1000, 512, 256), (32, 65536, 7168, 2048)


def _cases(*axes, dsv3, toy=TOY, big=DSV3):
    """(backend, shape, *axes): both backends x every value of ``axes`` at the toy shape, plus one case at DSv3 scale
    (FlyDSL, the ``dsv3`` values): those hold about 20 GB and run per-expert BF16 references."""

    def case(b, shape, rest, name):
        ids = [b, name, *(str(v).replace("torch.", "") for v in rest)]
        return pytest.param(b, shape, *rest, id="-".join(ids))

    toys = [case(b, toy, rest, "toy") for b in ("default", "triton") for rest in itertools.product(*axes)]
    return toys + [case("default", big, dsv3, "dsv3")]


@pytest.fixture
def backend(request, monkeypatch):
    monkeypatch.setenv("PRIMUS_TURBO_MX_STRICT", "2")
    monkeypatch.delenv("PRIMUS_TURBO_MXFP8_SCALE_RULE", raising=False)
    envs.reset_cache()
    GlobalBackendManager.reset()
    if request.param == "triton":
        GlobalBackendManager.set_grouped_gemm_backend(BackendType.TRITON, PrecisionType.FP8)
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


@pytest.mark.parametrize("backend, shape, skew", _cases([False, True], dsv3=(True,)), indirect=["backend"])
def test_grouped_gemm(backend, shape, skew):
    """fc1-shaped grouped_gemm_fp8: y / dx / dw vs per-expert BF16 GEMMs (SNR >= 25 dB)."""
    G, M, H, I = shape
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


@pytest.mark.parametrize("backend, shape", _cases(dsv3=()), indirect=["backend"])
def test_grouped_chain(backend, shape):
    """The unfused experts (``turbo_fused_grouped_gemm`` off): fc1 and fc2 by grouped_gemm_fp8 around a torch SwiGLU
    with probs, plain bf16 leaves (no cached weights): every leaf's gradient vs BF16 (SNR >= 20 dB)."""
    G, M, H, I = shape
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


@pytest.mark.parametrize("backend, shape, skew", _cases([False, True], dsv3=(True,)), indirect=["backend"])
def test_grouped_mlp(backend, shape, skew):
    """grouped_mlp_fp8 (fused GLU with probs): y / dx / dprobs / dw1 / dw2 vs BF16 (SNR >= 20 dB, cos >= 0.995)."""
    G, M, H, I = shape
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


@pytest.mark.parametrize(
    "backend, shape, main_grad_dtype",
    _cases([torch.bfloat16, torch.float32], dsv3=(torch.float32,)),
    indirect=["backend"],
)
def test_grouped_mlp_primus(backend, shape, main_grad_dtype):
    """Primus' routed experts (fused grouped MLP, several micro-batches): w1 / w2 cached by create_quantized_weight
    in its default recipe (gfx1250's GEMM layout; raw with Triton pinned),
    wgrads accumulated into main_grad (beta=1): y / dx / dprobs bitwise those of plain bf16 weights (the same
    operands), each main_grad increment vs the plain run's dw (>= 40 dB: one rounding against two) and BF16."""
    if not mx_wgrad_accum_supported(True, main_grad_dtype):
        pytest.skip(f"no grouped MXFP8 beta=1 wgrad into {main_grad_dtype} here")
    G, M, H, I = shape
    lens = _group_lens(G, M, True)
    x, dy = _randn(M, H, seed=1), _randn(M, H, seed=4)
    w1, w2 = _randn(G, 2 * I, H, scale=H**-0.5, seed=2), _randn(G, H, I, scale=I**-0.5, seed=3)
    probs = torch.rand(M, device="cuda", generator=torch.Generator(device="cuda").manual_seed(5))

    def run(wa, wb, fuse):
        xl, pl = x.clone().requires_grad_(), probs.clone().requires_grad_()
        y = grouped_mlp_fp8(xl, wa, wb, lens, pl, trans_w1=True, trans_w2=True, config=MX, activation="silu",
                            fuse_wgrad_accum_pattern="megatron" if fuse else None)  # fmt: skip
        y.backward(dy)
        return y, xl.grad, pl.grad

    plain = [w.clone().requires_grad_() for w in (w1, w2)]
    ref = run(*plain, False)
    cached = []
    for w in (w1, w2):
        q, q_t = create_quantized_weight(w, float8_e4m3, MX, need_weight_transpose_cache=True)
        assert q.scaling_recipe == mx_weight_layout_recipe(True, w.shape)
        q.main_grad = (_randn(*w.shape, scale=1e-3, seed=6)).to(main_grad_dtype)
        q.grad_added_to_main_grad = False
        cached.append(QuantizedTensorPair(q, q_t))
    main_grad0 = [p.data.main_grad.clone() for p in cached]
    got = run(*cached, True)
    for name, a, b in zip(("y", "dx", "dprobs"), got, ref):
        assert torch.equal(a, b), name
    bf = [t.clone().requires_grad_() for t in (x, w1, w2, probs)]
    _mlp_bf16(bf[0], bf[1], bf[2], lens, bf[3]).backward(dy)
    for name, p, g0, dw, dw_bf in zip(("dw1", "dw2"), cached, main_grad0, (plain[0].grad, plain[1].grad),
                                      (bf[1].grad, bf[2].grad)):  # fmt: skip
        assert p.data.grad_added_to_main_grad, name
        inc = p.data.main_grad.double() - g0.double()
        s = snr_db(inc, dw)
        assert s >= 40, f"{name}: main_grad increment vs the unfused wgrad {s:.2f} dB"
        _check(name, inc, dw_bf, 20)


class _Permute(torch.autograd.Function):
    """Stand-in for Primus' permute (moe_permute_with_probs): ``lazy`` returns the placeholder of
    ``_LazyPermuteFn`` (no values; the caller registers the gather metadata), else the materialized rows; backward =
    the permute's backward (each token's rows summed in a fixed order)."""

    @staticmethod
    def forward(ctx, x, d2s, order, k, lazy):
        ctx.order, ctx.k = order, k
        if lazy:
            return torch.empty(1, x.shape[1], dtype=x.dtype, device=x.device).expand(d2s.numel(), x.shape[1])
        return x.index_select(0, d2s.long())

    @staticmethod
    def backward(ctx, grad):
        return _combine(grad, ctx.order, ctx.k), None, None, None, None


class _Unpermute(torch.autograd.Function):
    """Stand-in for Primus' moe_unpermute (no merging probs): claims the fused MLP's output, and its backward
    defers the gather of dOut to the quantizer (ops/moe_gather.py) instead of writing the permuted bf16 copy."""

    @staticmethod
    def forward(ctx, y, d2s, row_id_map, order, k, T, E):
        from primus_turbo.pytorch.ops import moe_gather

        ctx.plan = moe_gather.claim_backward_gather_output(y, row_id_map, None, None, T, E, y.shape[1])
        ctx.d2s = d2s
        return _combine(y, order, k)

    @staticmethod
    def backward(ctx, dout):
        from primus_turbo.pytorch.ops import moe_gather

        lazy = moe_gather.make_backward_gather(dout.contiguous(), ctx.plan) if ctx.plan is not None else None
        grad = lazy if lazy is not None else dout.index_select(0, ctx.d2s.long())
        return grad, None, None, None, None, None, None


def _combine(rows, order, k):
    """Each token's k permuted rows summed in a fixed order (fp32): a deterministic unpermute / permute backward."""
    return rows.float().index_select(0, order).view(-1, k, rows.shape[1]).sum(1).to(rows.dtype)


@pytest.mark.parametrize(
    "backend, shape, skew",
    _cases([False, True], dsv3=(True,), toy=(257, 512, 256), big=(8192, 7168, 2048)),
    indirect=["backend"],
)
def test_grouped_mlp_permute_fusion(backend, shape, skew, monkeypatch):
    """Primus' MoE permute -> fused grouped MLP -> unpermute with moe_permute_quant_fusion and its backward twin
    (32 experts, top-8): x and dOut are quantized straight from the token rows through dest2src (no bf16 permuted
    copy), and y / dx / dw1 / dw2 / dprobs are bitwise those of the materialized permute."""
    from primus_turbo.pytorch.ops import grouped_mlp_fp8 as op
    from primus_turbo.pytorch.ops import moe_gather

    assert op.CONSUMES_PERMUTE_SEAM_MXFP8  # what Primus probes before it hands over a placeholder
    (T, H, I), E, k = shape, 32, 8
    g = torch.Generator(device="cuda").manual_seed(T + H)
    if skew:  # hot experts, empty ones
        w = torch.rand(E, generator=g, device="cuda") ** 4
        w[3:9] = 0
        experts = torch.multinomial(w.expand(T, E), k, generator=g)
    else:
        experts = (torch.arange(T, device="cuda")[:, None] + 4 * torch.arange(k, device="cuda")) % E
    rmap = torch.zeros(T, E, dtype=torch.bool, device="cuda").scatter_(1, experts, True)
    probs = (0.1 + torch.rand(T, E, generator=g, device="cuda")) * rmap
    e_idx, t_idx = rmap.T.nonzero(as_tuple=True)  # the permuted rows: expert-major, tokens in order
    d2s, pprobs = t_idx.to(torch.int32), probs[t_idx, e_idx]
    order = torch.argsort(t_idx, stable=True)  # each token's permuted rows, in expert order
    lens = rmap.sum(0)
    row_id_map = torch.zeros(T, 2 * E + 1, dtype=torch.int32, device="cuda")
    x, dout = _randn(T, H, seed=1), _randn(T, H, seed=4)
    w1, w2 = _randn(E, 2 * I, H, scale=H**-0.5, seed=2), _randn(E, H, I, scale=I**-0.5, seed=3)

    calls = []
    quant = op.grouped_quantize_fp8_with_trans
    monkeypatch.setattr(op, "grouped_quantize_fp8_with_trans",
                        lambda *a, **kw: calls.append(kw.get("dest2src") is not None) or quant(*a, **kw))  # fmt: skip

    def run(fused):
        leaves = [t.clone().requires_grad_() for t in (x, w1, w2, pprobs)]
        xp = _Permute.apply(leaves[0], d2s, order, k, fused)
        if fused:  # as moe_permute_with_probs registers the output of _LazyPermuteFn
            moe_gather.register_permuted_activation_seam(xp, leaves[0].detach(), d2s, pprobs, row_id_map,
                                                         fuse_backward_permute_quant=True)  # fmt: skip
        y = grouped_mlp_fp8(xp, leaves[1], leaves[2], lens, leaves[3], trans_w1=True, trans_w2=True, config=MX,
                            activation="silu")  # fmt: skip
        out = _Unpermute.apply(y, d2s, row_id_map, order, k, T, E) if fused else _combine(y, order, k)
        out.backward(dout)
        return [out] + [t.grad for t in leaves]

    ref = run(False)
    assert calls == [False, False]
    calls.clear()
    got = run(True)
    assert calls == [True, True], f"the permute fusion did not run (dest2src per quantization: {calls})"
    assert not moe_gather._PERMUTED_ACTIVATION_SEAM_TABLE and not moe_gather._BACKWARD_GATHER_OUTPUTS
    for name, a, b in zip(("y", "dx", "dw1", "dw2", "dprobs"), got, ref):
        assert torch.equal(a, b), name
