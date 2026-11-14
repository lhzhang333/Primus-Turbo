###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""gemm_fp8 (MX_BLOCKWISE) on gfx1250: FP8GemmMXFunction on the FlyDSL MX GEMM, in strict mode (a fallback raises).
fwd / dgrad / wgrad against BF16 and against the fp32 emulation of the recipe (``tests/pytorch/ref/mxfp8_ref.py``:
the same quantization, so only the GEMM's accumulation order and the bf16 output rounding differ)."""

import importlib

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
    ScalingRecipe,
    float8_e4m3,
)
from primus_turbo.pytorch.core.quantized_tensor import (
    QuantizedTensorPair,
    create_quantized_weight,
)
from primus_turbo.pytorch.core.utils import is_gfx1250
from primus_turbo.pytorch.ops import gemm_fp8, grouped_gemm_fp8, mlp_fp8
from primus_turbo.pytorch.ops.normalization import rmsnorm_mxfp8
from primus_turbo.pytorch.ops.utils import MX_A, MX_W
from tests.pytorch.ref.mxfp8_ref import dequant, quant_2d, quant_dual, snr_db

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not is_gfx1250(), reason="gfx1250 MX GEMM")
# the module (``primus_turbo.pytorch.ops`` re-exports the function under the same name)
gemm_fp8_mod = importlib.import_module("primus_turbo.pytorch.ops.gemm_fp8")

MX = Float8QuantConfig(
    format=Format.E4M3,
    granularity=ScalingGranularity.MX_BLOCKWISE,
    scale_dtype=ScaleDtype.E8M0,
    block_size=32,
)


@pytest.fixture(autouse=True)
def strict(monkeypatch):
    """Strict mode on, the default scale rule, no pinned backend."""

    def set_strict(level: int):
        monkeypatch.setenv("PRIMUS_TURBO_MX_STRICT", str(level))
        envs.reset_cache()

    monkeypatch.delenv("PRIMUS_TURBO_MXFP8_SCALE_RULE", raising=False)
    set_strict(2)
    GlobalBackendManager.reset()
    yield set_strict
    GlobalBackendManager.reset()


def _emulated(x, w, dy):
    """(out, dx, dw) of the recipe in fp32: dequantized reference operands (activation / gradient 1x32 rows and
    columns, weight 32x32 blocks), multiplied exactly."""
    xr, sxr, xc, sxc = quant_dual(x)
    wq, sw, wt, swt = quant_2d(w)
    gr, sgr, gc, sgc = quant_dual(dy)
    mm = lambda a, sa, b, sb: dequant(a, sa) @ dequant(b, sb).t()  # noqa: E731
    return mm(xr, sxr, wq, sw), mm(gr, sgr, wt, swt), mm(gc, sgc, xc, sxc)


def _data(M, N, K, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(M, K, device="cuda", generator=g).bfloat16()
    w = (torch.randn(N, K, device="cuda", generator=g) / K**0.5).bfloat16()
    return x, w, torch.randn(M, N, device="cuda", generator=g).bfloat16()


# toy, M / N / K tails (K padded to 128 by the quantizer), DSv3 shapes at T = 1024: kv_down, shared fc2, q_down
@pytest.mark.parametrize("M, N, K", [(256, 512, 384), (200, 136, 320), (1024, 576, 7168), (1024, 7168, 2048),
                                     (1024, 1536, 7168)])  # fmt: skip
def test_linear(M, N, K):
    x, w, dy = _data(M, N, K)
    xg, wg = x.clone().requires_grad_(), w.clone().requires_grad_()
    out = gemm_fp8(xg, wg, trans_b=True, out_dtype=torch.bfloat16, config=MX)
    out.backward(dy)
    xf, wf, dyf = x.float(), w.float(), dy.float()
    for name, got, bf, emu in zip(("out", "dx", "dw"), (out, xg.grad, wg.grad), (xf @ wf.t(), dyf @ wf, dyf.t() @ xf),
                                  _emulated(x, w, dy)):  # fmt: skip
        assert got.dtype == torch.bfloat16 and got.shape == bf.shape, name
        s_bf, s_emu = snr_db(got, bf), snr_db(got, emu)
        assert s_bf >= 25 and s_emu >= 50, (
            f"{name}: SNR vs bf16 {s_bf:.2f} dB, vs the emulated recipe {s_emu:.2f} dB"
        )


@pytest.mark.parametrize("gemm_layout", [True, False])
def test_cached_weight(gemm_layout):
    """A weight quantized ahead (``create_quantized_weight``, Primus' first-micro-batch cache), in gfx1250's MX GEMM
    layout (``MX_W``, its default here: what the op quantizes a plain weight into, so bitwise the same) or the raw
    one (another kernel config, the same values: >= 50 dB)."""
    x, w, dy = _data(512, 1536, 1024)
    recipe = MX_W if gemm_layout else ScalingRecipe(use_2d_block=True)
    q, q_t = create_quantized_weight(
        w, float8_e4m3, MX, need_weight_transpose_cache=True, scaling_recipe=recipe
    )
    assert create_quantized_weight(w, float8_e4m3, MX)[0].scaling_recipe == MX_W  # the default here
    res = []
    for b in (w, QuantizedTensorPair(q, q_t)):
        xg = x.clone().requires_grad_()
        out = gemm_fp8(xg, b, trans_b=True, out_dtype=torch.bfloat16, config=MX)
        out.backward(dy)
        res.append((out, xg.grad))
    for got, ref in zip(*res):
        assert torch.equal(got, ref) if gemm_layout else snr_db(got, ref) >= 50


def test_chain():
    """Two linears around a GELU on plain bf16 leaves: every leaf's gradient vs BF16 (SNR >= 20 dB)."""
    x, w1, _ = _data(512, 1024, 768)
    _, w2, dy = _data(512, 768, 1024, seed=1)

    def grads(mm):
        leaves = [t.clone().requires_grad_() for t in (x, w1, w2)]
        mm(F.gelu(mm(leaves[0], leaves[1]).float()).bfloat16(), leaves[2]).backward(dy)
        return [t.grad for t in leaves]

    mx = grads(lambda a, b: gemm_fp8(a, b, trans_b=True, out_dtype=torch.bfloat16, config=MX))
    for name, got, ref in zip(("dx", "dw1", "dw2"), mx, grads(lambda a, b: a @ b.t())):
        assert got is not None and snr_db(got, ref) >= 20, name


@pytest.mark.parametrize("fmt, dtype", [(Format.HYBRID, torch.bfloat16), (Format.E4M3, torch.float16)])
@pytest.mark.parametrize("grouped", [False, True])
def test_other_formats_take_the_raw_layouts(strict, fmt, dtype, grouped):
    """What gfx1250's quantizers cannot write in its GEMM layouts (E5M2 gradients, fp16) runs on the raw layouts with
    the HIP quantizers and the Triton MX GEMM (strict mode off: those are fallbacks) instead of raising; a weight
    cached in the GEMM layout is refused up front."""
    strict(0)
    mx = ScalingGranularity.MX_BLOCKWISE
    cfg = Float8QuantConfig(format=fmt, granularity=mx, scale_dtype=ScaleDtype.E8M0, block_size=32)
    x, w, dy = (t.to(dtype) for t in _data(256, 512, 384))
    lens = torch.tensor([96, 160], device="cuda", dtype=torch.int64)
    wg = w.view(1, 512, 384).expand(2, 512, 384).contiguous() if grouped else w
    xg, wl = x.clone().requires_grad_(), wg.clone().requires_grad_()
    if grouped:
        out = grouped_gemm_fp8(xg, wl, lens, trans_b=True, config=cfg)
    else:
        out = gemm_fp8(xg, wl, trans_b=True, out_dtype=dtype, config=cfg)
    out.backward(dy)
    xf, wf, dyf = x.float(), w.float(), dy.float()
    dw = dyf.t() @ xf
    if grouped:
        dw = torch.stack([dyf[:96].t() @ xf[:96], dyf[96:].t() @ xf[96:]])
    bwd_min = 18 if fmt == Format.HYBRID else 20  # E5M2 gradients: 2 mantissa bits
    for name, got, ref, snr_min in (
        ("out", out, xf @ wf.t(), 20),
        ("dx", xg.grad, dyf @ wf, bwd_min),
        ("dw", wl.grad, dw, bwd_min),
    ):
        s = snr_db(got, ref)
        assert got.dtype == dtype and s >= snr_min, f"{name}: SNR {s:.2f} dB"
    if not grouped:  # a weight cached in the GEMM layout (from bf16, E4M3) forces it: refused here
        q = create_quantized_weight(
            w.bfloat16(), float8_e4m3, MX, need_weight_transpose_cache=True, scaling_recipe=MX_W
        )
        with pytest.raises(ValueError, match="cached weight is in gfx1250's MX GEMM layout"):
            gemm_fp8(x, QuantizedTensorPair(*q), trans_b=True, out_dtype=dtype, config=cfg)


def test_fallback_refused(strict):
    """N % 8 != 0 is not the FlyDSL kernel's: strict mode refuses the Triton fallback; without it, Triton's result."""
    x, w, _ = _data(256, 100, 384)
    with pytest.raises(RuntimeError, match="would fall back to TRITON"):
        gemm_fp8(x, w, trans_b=True, out_dtype=torch.bfloat16, config=MX)
    strict(0)
    out = gemm_fp8(x, w, trans_b=True, out_dtype=torch.bfloat16, config=MX)
    assert snr_db(out, x.float() @ w.float().t()) >= 25


def test_prequantized_recipe_mismatch():
    """A pre-quantized activation pair (``rmsnorm_mxfp8``) in another layout than the GEMM reads is refused with a
    clear error: the GEMM-layout pair against a weight cached raw, a raw column copy next to a GEMM-layout row."""
    x, w, _ = _data(256, 512, 1024)
    gamma = torch.ones(1024, device="cuda", dtype=torch.bfloat16)
    _, pair = rmsnorm_mxfp8(x, gamma)  # gfx1250 default: MX_A rows, MX_B columns
    raw = ScalingRecipe(use_2d_block=True)
    raw_w = QuantizedTensorPair(
        *create_quantized_weight(w, float8_e4m3, MX, need_weight_transpose_cache=True, scaling_recipe=raw)
    )
    with pytest.raises(ValueError, match="a was quantized with .* raw MX layout"):
        gemm_fp8(pair, raw_w, trans_b=True, out_dtype=torch.bfloat16, config=MX)
    _, pair = rmsnorm_mxfp8(x, gamma, row_recipe=MX_A, col_recipe=ScalingRecipe())
    with pytest.raises(ValueError, match="a_t .* gfx1250 MX GEMM layout"):
        gemm_fp8(pair, w, trans_b=True, out_dtype=torch.bfloat16, config=MX)


def test_direct_path(monkeypatch):
    """Eager gfx1250 MX linears call the FlyDSL GEMM directly (``mx1250_gemm_direct``: no torch.library custom op,
    no dispatcher), bitwise the custom-op path's results; a pinned FP8 GEMM backend goes through the dispatcher."""
    x, w, dy = _data(512, 1536, 1024)

    def run():
        xg, wg = x.clone().requires_grad_(), w.clone().requires_grad_()
        out = gemm_fp8(xg, wg, trans_b=True, out_dtype=torch.bfloat16, config=MX)
        out.backward(dy)
        return out, xg.grad, wg.grad

    calls, real = [], gemm_fp8_mod.gemm_fp8_impl
    monkeypatch.setattr(gemm_fp8_mod, "gemm_fp8_impl", lambda *a, **k: calls.append(1) or real(*a, **k))
    fast = run()
    assert not calls, "the eager MX linear went through the custom op"
    GlobalBackendManager.set_gemm_backend(BackendType.FLYDSL, PrecisionType.FP8)
    pinned = run()
    assert len(calls) == 3, "a pinned backend must go through the dispatcher (fwd, dgrad, wgrad)"
    GlobalBackendManager.reset()
    monkeypatch.setattr(gemm_fp8_mod, "mx1250_gemm_direct", lambda *a, **k: None)
    slow = run()
    for f, p, s in zip(fast, pinned, slow):
        assert torch.equal(f, s) and torch.equal(f, p)


class _KernelSwiGLU(torch.autograd.Function):
    """bf16 SwiGLU with the fused GLU / dGLU kernels' own values (their bf16 test outputs): the unfused chain
    ``gemm_fp8 -> this -> gemm_fp8`` then quantizes exactly what ``mlp_fp8`` quantizes inside the kernels."""

    @staticmethod
    def forward(ctx, l1):
        from primus_turbo.flydsl.quantization import mxfp8_quant_gfx1250 as Q

        act = torch.empty(l1.shape[0], l1.shape[1] // 2, dtype=l1.dtype, device=l1.device)
        Q.glu_quant_mxfp8(l1, pad=True, act_out=act)
        ctx.save_for_backward(l1)
        return act

    @staticmethod
    def backward(ctx, d_act):
        from primus_turbo.flydsl.quantization import mxfp8_quant_gfx1250 as Q

        (l1,) = ctx.saved_tensors
        grad = torch.empty_like(l1)
        Q.dglu_quant_mxfp8(d_act.contiguous(), l1, pad=True, grad_out=grad)
        return grad


# M / H tails (H % 128 = 64) in every mode; the DSv3 shared expert at T = 8192 (H 7168, I 2048) as Primus runs it
@pytest.mark.parametrize(
    "M, H, I, mode",
    [(1000, 576, 256, mode) for mode in ("plain", "cached_bf16", "cached_fp32")]
    + [(8192, 7168, 2048, "cached_fp32")],
)
def test_mlp(M, H, I, mode):
    """mlp_fp8 (one Function: fc1, fused GLU + dual cast, fc2; fused dGLU in the backward) bitwise equal to the
    unfused gemm_fp8 chain with the same recipes: out, dx and the weight gradients -- plain bf16 weights, or
    weight caches (``create_quantized_weight``'s default recipe) with the wgrads accumulated into a bf16 / fp32
    main_grad (two micro-batches)."""
    g = torch.Generator(device="cuda").manual_seed(M + H)
    w1 = (torch.randn(2 * I, H, device="cuda", generator=g) / H**0.5).bfloat16()
    w2 = (torch.randn(H, I, device="cuda", generator=g) / I**0.5).bfloat16()
    xs = [torch.randn(M, H, device="cuda", generator=g).bfloat16() for _ in range(2)]
    dys = [torch.randn(M, H, device="cuda", generator=g).bfloat16() for _ in range(2)]
    main_grads = [(torch.randn(w.shape, device="cuda", generator=g) * 0.01) for w in (w1, w2)]

    def chain(x, a, b, fuse):
        kw = dict(trans_b=True, out_dtype=torch.bfloat16, config=MX, fuse_bgrad_accum_pattern=fuse)
        return gemm_fp8(_KernelSwiGLU.apply(gemm_fp8(x, a, **kw)), b, **kw)

    def fused(x, a, b, fuse):
        return mlp_fp8(x, a, b, config=MX, fuse_wgrad_accum_pattern=fuse)

    def run(op):
        if mode == "plain":
            ws = [w.clone().requires_grad_() for w in (w1, w2)]
            x = xs[0].clone().requires_grad_()
            out = op(x, *ws, None)
            out.backward(dys[0])
            return [out, x.grad, ws[0].grad, ws[1].grad]
        dtype = torch.bfloat16 if mode == "cached_bf16" else torch.float32
        ws = []
        for w, mg in zip((w1, w2), main_grads):
            q, q_t = create_quantized_weight(w, float8_e4m3, MX, need_weight_transpose_cache=True)
            q.main_grad, q.grad_added_to_main_grad = mg.to(dtype), False
            ws.append(QuantizedTensorPair(q, q_t))
        res = []
        for x0, dy in zip(xs, dys):
            x = x0.clone().requires_grad_()
            out = op(x, *ws, "megatron")
            out.backward(dy)
            res += [out, x.grad]
        return res + [p.data.main_grad for p in ws]

    for i, (got, ref) in enumerate(zip(run(fused), run(chain))):
        assert got.dtype == ref.dtype and torch.equal(got, ref), f"result {i}: max |diff| {(got - ref).abs().max()}"
