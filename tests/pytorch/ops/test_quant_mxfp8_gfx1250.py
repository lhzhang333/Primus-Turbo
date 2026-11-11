###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""gfx1250 MXFP8 quantization kernels (primus_turbo/flydsl/quantization/mxfp8_quant_gfx1250.py)
against the recipe reference tests/pytorch/ref/mxfp8_ref.py, bit for bit (reference on the CPU: IEEE
denormals, no device pow)."""

import os
import sys

import pytest
import torch

from primus_turbo.pytorch.core.utils import is_gfx1250

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ref"))
import mxfp8_ref as R  # noqa: E402

pytestmark = pytest.mark.skipif(not is_gfx1250(), reason="gfx1250 kernels")
RULES = ("nosat", "even")


def _x(*shape, seed=0, special=True):
    """bf16 data over a wide range, per-row scaled, with all-zero blocks, denormal blocks, the scale rules'
    boundary values (+-448 * 2^k, mantissas 1.75 / 1.7578125 / 1.875 / 1.9375) and +-Inf / NaN."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows = shape[-2]
    x = torch.randn(*shape, generator=g, device="cuda")
    x = x * torch.exp2(torch.randint(-24, 24, (*shape[:-1], 1), generator=g, device="cuda").float())
    if special and x.numel() >= 64:
        x[..., ::7, :32] = 0
        x[..., 1::5, :] *= 2.0**-128  # rows of denormals / tiny normals
        f = x.view(-1)
        n = f.numel()

        def at(k):
            return torch.randint(0, n, (k,), generator=g, device="cuda")

        k = max(1, n // 64)
        mant = torch.tensor([1.75, 1.7578125, 1.875, 1.9375, 1.0, 1.9921875], device="cuda")
        v = mant[torch.randint(0, 6, (k,), generator=g, device="cuda")] * 256
        e = torch.randint(-40, 40, (k,), generator=g, device="cuda").float()
        s = torch.randint(0, 2, (k,), generator=g, device="cuda").float() * 2 - 1
        f[at(k)] = v * torch.exp2(e) * s
        if rows > 2:
            f[at(3)] = torch.tensor([float("inf"), -float("inf"), float("nan")], device="cuda")
    return x.to(torch.bfloat16)


def _eq(a, b, what):
    a, b = a.contiguous().view(torch.uint8).cpu(), b.contiguous().view(torch.uint8).cpu()
    assert a.shape == b.shape, f"{what}: shape {tuple(a.shape)} vs {tuple(b.shape)}"
    bad = (a != b).nonzero()
    assert bad.numel() == 0, f"{what}: {bad.shape[0]} bytes differ, first at {bad[0].tolist()}: {a[tuple(bad[0])]} vs {b[tuple(bad[0])]}"


@pytest.fixture(scope="module")
def Q():
    from primus_turbo.flydsl.quantization import mxfp8_quant_gfx1250

    return mxfp8_quant_gfx1250


SHAPES = [(1, 1), (1, 32), (3, 200), (33, 64), (100, 33), (128, 128), (130, 300), (257, 576), (300, 7168), (64, 1000)]


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("M,K", SHAPES)
def test_dual(Q, M, K, rule):
    x = _x(M, K, seed=M * 7 + K)
    out = Q.quant_mxfp8_raw(x, torch.float8_e4m3fn, rule=rule)
    for got, ref, what in zip(out, R.quant_dual(x.cpu(), rule), ("q_row", "s_row", "q_col", "s_col")):
        _eq(got, ref, what)
    q, s = Q.quant_mxfp8_rows(x, rule=rule)
    _eq(q, out[0], "rows-only q")
    _eq(s, out[1], "rows-only s")


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("B,N,K", [(1, 1, 1), (1, 70, 130), (3, 64, 96), (2, 129, 257), (4, 256, 2048), (1, 300, 7168)])
def test_2d(Q, B, N, K, rule):
    w = _x(B, N, K, seed=B + N + K)
    out = Q.quant_mxfp8_raw_batched(w, torch.float8_e4m3fn, row_2d=True, col_2d=True, rule=rule)
    for b in range(B):
        for got, ref, what in zip(out, R.quant_2d(w[b].cpu(), rule), ("q", "s_row", "q_t", "s_t")):
            _eq(got[b], ref, f"{what}[{b}]")
    if B == 1:
        for got, ref in zip(Q.quant_mxfp8_raw(w[0], torch.float8_e4m3fn, True, True, rule=rule), out):
            _eq(got, ref[0], "2-D entry")


def _lens(case, seed):
    if case in ("random", "many"):  # many: 70 experts, the lookup's multi-chunk path
        g = torch.Generator().manual_seed(seed)
        n, hi = (13, 300) if case == "random" else (70, 60)
        lens = torch.randint(0, hi, (n,), generator=g)
        lens[torch.randint(0, n, (n // 4,), generator=g)] = 0
        return lens
    return torch.tensor(case)


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("K", [64, 200, 2048])
@pytest.mark.parametrize(
    "case", [[1], [0, 5, 0], [31, 32, 33, 1, 0, 127, 128, 129], [256, 256, 256, 256], [0, 0, 300], "random", "many"]
)
def test_grouped(Q, case, K, rule):
    lens = _lens(case, K)
    x = _x(int(lens.sum()), K, seed=K + int(lens.sum()), special=int(lens.sum()) > 0)
    lens_d = lens.cuda()
    offs = torch.nn.functional.pad(torch.cumsum(lens_d, 0), (1, 0))
    if x.shape[0] == 0:
        return
    out = Q.grouped_quant_mxfp8_raw(x, lens_d, offs, torch.float8_e4m3fn, rule=rule)
    q_row, s_row, q_col, s_col, lens_c, offs_c = R.quant_grouped(x.cpu(), lens, rule)
    _eq(out[0], q_row, "q_row")
    _eq(out[1], s_row, "s_row")
    assert out[4] is lens_d and out[5] is offs, "row copy is unpadded: lens / offs are the inputs"
    assert torch.equal(out[6].cpu(), lens_c.cpu()) and torch.equal(out[7].cpu(), offs_c.cpu()), "col lens / offs"
    Mc = int(offs_c[-1])
    assert out[2].shape[1] >= Mc and out[3].shape[1] * 32 == out[2].shape[1]
    _eq(out[2][:, :Mc], q_col, "q_col")
    _eq(out[3][:, : Mc // 32], s_col, "s_col")


def test_special_blocks(Q):
    """Hand-made blocks: zero, denormal, NaN, +-Inf, exact powers, the rules' boundaries."""
    vals = [0.0, 2.0**-133, -(2.0**-130), 2.0**-126, 448.0, -448.0, 449.0, 1.9375 * 2**20, -1.875 * 2**-90,
            float("inf"), -float("inf"), float("nan"), 3.0e38, -3.0e38, 1.0, -0.0]
    x = torch.zeros(64, 64)
    for i, v in enumerate(vals):
        x[i, :32] = v * torch.linspace(-1, 1, 32) if i % 2 else v  # whole-block value or a ramp ending in it
        x[i, 32 + i] = v
        x[16 + i, 7] = v
    x = x.to(torch.bfloat16).cuda()
    for rule in RULES:
        for got, ref, what in zip(Q.quant_mxfp8_raw(x, rule=rule), R.quant_dual(x.cpu(), rule), ("qr", "sr", "qc", "sc")):
            _eq(got, ref, f"{rule} {what}")


# ---- fused producers: GLU / dGLU (gfx950 phase-2 math) and gather (permute), MoE dual cast
def _ulp_close(got, ref, what):
    """bf16 within one ulp of the torch math (the kernels use the hardware exp2 / reciprocal), and
    nearly all exactly equal."""
    a, b = got.cpu().view(torch.int16).int(), ref.cpu().view(torch.int16).int()
    near = ((a - b).abs() <= 1) | ((got.cpu().float() - ref.cpu().float()).abs() <= 1e-35)
    assert near.all(), f"{what}: {int((~near).sum())} values more than an ulp off"
    assert (a == b).float().mean() > 0.995, f"{what}: only {float((a == b).float().mean()):.4f} exact"


def _moe_lens(case, seed):
    if case == "skew":  # one hot expert, empties, a single token
        return torch.tensor([1500, 0, 1, 300, 0, 0, 77, 640, 0, 2, 129, 0, 31, 0, 0, 256])
    return _lens(case, seed)


def _check_grouped(out, ref_vals, lens, rule, M):
    q_row, s_row, q_col, s_col, lens_c, offs_c = R.quant_grouped(ref_vals.cpu(), lens, rule)
    _eq(out[0], q_row, "q_row")
    _eq(out[1], s_row, "s_row")
    assert torch.equal(out[6].cpu(), lens_c) and torch.equal(out[7].cpu(), offs_c), "col lens / offs"
    Mc = int(offs_c[-1])
    _eq(out[2][:, :Mc], q_col, "q_col")
    _eq(out[3][:, : Mc // 32], s_col, "s_col")


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("act,clamp", [("silu", None), ("silu", 7.0), ("gelu", None), ("gelu", 7.0)])
@pytest.mark.parametrize("I", [96, 2048])
@pytest.mark.parametrize("case", [[0, 5, 0], [31, 32, 33, 1, 0, 127, 128, 129], "random", "skew"])
def test_glu(Q, case, I, act, clamp, rule):
    lens = _moe_lens(case, I)
    M = int(lens.sum())
    g = torch.Generator(device="cuda").manual_seed(M + I)
    l1 = (torch.randn(M, 2 * I, generator=g, device="cuda") * 3).to(torch.bfloat16)
    probs = torch.rand(M, generator=g, device="cuda")
    probs[::17] = 0
    lens_d = lens.cuda()
    offs = torch.nn.functional.pad(torch.cumsum(lens_d, 0), (1, 0))
    act_v = torch.full((M, I), float("nan"), dtype=torch.bfloat16, device="cuda")
    out = Q.grouped_glu_quant_mxfp8(l1, probs, lens_d, offs, act, clamp, rule=rule, act_out=act_v)
    _ulp_close(act_v, R.glu_ref(l1.cpu(), probs.cpu(), act, clamp), "act")
    _check_grouped(out, act_v, lens, rule, M)


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("act,clamp", [("silu", None), ("silu", 7.0), ("gelu", None), ("gelu", 7.0)])
@pytest.mark.parametrize("I", [96, 2048])
@pytest.mark.parametrize("case", [[0, 5, 0], [31, 32, 33, 1, 0, 127, 128, 129], "random", "skew"])
def test_dglu(Q, case, I, act, clamp, rule):
    lens = _moe_lens(case, I + 1)
    M = int(lens.sum())
    g = torch.Generator(device="cuda").manual_seed(M + 3 * I)
    l1 = (torch.randn(M, 2 * I, generator=g, device="cuda") * 3).to(torch.bfloat16)
    dact = torch.randn(M, I, generator=g, device="cuda").to(torch.bfloat16)
    probs = torch.rand(M, generator=g, device="cuda")
    lens_d = lens.cuda()
    offs = torch.nn.functional.pad(torch.cumsum(lens_d, 0), (1, 0))
    grad_v = torch.full((M, 2 * I), float("nan"), dtype=torch.bfloat16, device="cuda")
    gp, *out = Q.grouped_dglu_quant_mxfp8(dact, l1, probs, lens_d, offs, act, clamp, rule=rule, grad_out=grad_v)
    ref_grad, ref_gp = R.dglu_ref(dact.cpu(), l1.cpu(), probs.cpu(), act, clamp)
    _ulp_close(grad_v, ref_grad, "grad_l1")
    torch.testing.assert_close(gp.cpu(), ref_gp, rtol=1e-4, atol=1e-4 * float(ref_gp.abs().max() + 1))
    _check_grouped(out, grad_v, lens, rule, M)


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("H", [64, 200, 7168])
@pytest.mark.parametrize("case", [[0, 5, 0], [31, 32, 33, 1, 0, 127, 128, 129], "random", "skew"])
@pytest.mark.parametrize("with_probs", [True, False])
def test_gather(Q, case, H, with_probs, rule):
    lens = _moe_lens(case, H)
    M = int(lens.sum())
    T = max(1, M // 3)
    g = torch.Generator(device="cuda").manual_seed(M + H)
    x = _x(T, H, seed=H)
    d2s = torch.randint(0, T, (M,), generator=g, device="cuda", dtype=torch.int32)
    if M > 4:
        d2s[1::9], d2s[3] = -1, T + 5  # rows of no token
    pp = torch.rand(M, generator=g, device="cuda") if with_probs else None
    if pp is not None:
        pp[::7] = 0
    lens_d = lens.cuda()
    offs = torch.nn.functional.pad(torch.cumsum(lens_d, 0), (1, 0))
    out = Q.grouped_gather_quant_mxfp8(x, d2s, lens_d, offs, pp, rule=rule)
    _check_grouped(out, R.gather_rows(x.cpu(), d2s.cpu(), None if pp is None else pp.cpu()), lens, rule, M)


# ---- wiring: quantization_impl on gfx1250 (HIP layouts: K / M padded to 128, zero codes, byte-0 scales)
def _padded(t, shape):
    out = torch.zeros(shape, dtype=torch.uint8)
    out[tuple(slice(0, d) for d in t.shape)] = t.cpu().view(torch.uint8)
    return out


@pytest.mark.parametrize("rule", RULES)
def test_dispatch(rule):
    import primus_turbo.common.envs as envs
    from primus_turbo.pytorch.core.low_precision import (
        Float8QuantConfig,
        Format,
        ScaleDtype,
        ScalingGranularity,
        ScalingRecipe,
    )
    from primus_turbo.pytorch.core.quantized_tensor import (
        QuantizedTensor,
        create_quantized_weight,
    )
    from primus_turbo.pytorch.ops.quantization import (
        grouped_quantize_fp8_with_trans,
        quantize_fp8,
        quantize_fp8_with_trans,
    )

    MX, E4 = ScalingGranularity.MX_BLOCKWISE, torch.float8_e4m3fn
    envs.set_env("PRIMUS_TURBO_MXFP8_SCALE_RULE", rule)
    try:
        x = _x(200, 330, seed=5)
        Mp, Kp = 256, 384
        qr, sr, qc, sc = quantize_fp8_with_trans(x, E4, MX, block_size=32)
        rq, rs, rqc, rsc = R.quant_dual(x.cpu(), rule)
        for got, ref, shape, what in ((qr, rq, (200, Kp), "q_row"), (sr, rs, (200, Kp // 32), "s_row"),
                                      (qc, rqc, (330, Mp), "q_col"), (sc, rsc, (330, Mp // 32), "s_col")):
            _eq(got, _padded(ref, shape), what)
        for axis, ref, what in ((1, (rq, rs), "rows"), (0, (rqc, rsc), "cols")):
            got = quantize_fp8(x, E4, MX, block_size=32, axis=axis, scaling_recipe=ScalingRecipe())
            _eq(got[0], _padded(ref[0], got[0].shape), f"{what} q")
            _eq(got[1], _padded(ref[1], got[1].shape), f"{what} s")
        w = _x(3, 96, 200, seed=6)
        r2 = ScalingRecipe(use_2d_block=True)
        out = quantize_fp8_with_trans(w, E4, MX, block_size=32, scaling_recipe=r2, scaling_recipe_for_trans=r2)
        for b in range(3):
            for got, ref, what in zip(out, R.quant_2d(w[b].cpu(), rule), ("q", "s_row", "q_t", "s_t")):
                _eq(got[b], _padded(ref, got[b].shape), f"2d {what}[{b}]")
        cfg = Float8QuantConfig(format=Format.E4M3, granularity=MX, scale_dtype=ScaleDtype.E8M0, block_size=32)
        qw, qw_t = create_quantized_weight(w, E4, cfg, need_weight_transpose_cache=True)
        for qt_, axis in ((qw, -1), (qw_t, -2)):
            two = QuantizedTensor.quantize(w, E4, MX, block_size=32, scaling_recipe=r2, axis=axis)
            _eq(qt_.qdata, two.qdata, f"one-pass weight {axis}")
            _eq(qt_.scale_inv, two.scale_inv, f"one-pass weight scale {axis}")
        lens = torch.tensor([0, 37, 1, 200, 0, 129])
        lens_d = lens.cuda()
        offs = torch.nn.functional.pad(torch.cumsum(lens_d, 0), (1, 0))
        xg = _x(int(lens.sum()), 330, seed=7)
        out = grouped_quantize_fp8_with_trans(xg, E4, MX, lens_d, offs, block_size=32)
        ref = R.quant_grouped(xg.cpu(), lens, rule)
        _eq(out[0], _padded(ref[0], out[0].shape), "grouped q_row")
        _eq(out[1], _padded(ref[1], out[1].shape), "grouped s_row")
        Mc = int(ref[5][-1])
        _eq(out[2][:, :Mc], ref[2], "grouped q_col")
        assert torch.equal(out[5].cpu(), offs.cpu()) and torch.equal(out[7].cpu(), ref[5]), "grouped offs"
        src = _x(100, 330, seed=8)
        d2s = torch.randint(0, 100, (int(lens.sum()),), device="cuda", dtype=torch.int32)
        pp = torch.rand(d2s.numel(), device="cuda")
        pp[::5] = 0
        out = grouped_quantize_fp8_with_trans(src, E4, MX, lens_d, offs, block_size=32, dest2src=d2s,
                                              permuted_probs=pp, total_m=d2s.numel())
        ref = R.quant_grouped(R.gather_rows(src.cpu(), d2s.cpu(), pp.cpu()), lens, rule)
        _eq(out[0], _padded(ref[0], out[0].shape), "gathered q_row")
        _eq(out[2][:, :Mc], ref[2], "gathered q_col")
    finally:
        envs.set_env("PRIMUS_TURBO_MXFP8_SCALE_RULE", None)
