###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""The op layer on gfx1250's MXFP8 quantizers (primus_turbo/flydsl/quantization/mxfp8_quant_gfx1250.py), bit for
bit against tests/pytorch/ref/mxfp8_ref.py: the quantization ops' dispatch (HIP layouts, and the GEMM layouts the
recipes' shuffle flags select), the strict mode's HIP fallback, the GEMM operands' storage slack, the permute ->
grouped MXFP8 handoffs (ops.moe_gather), and ``ops.normalization.rmsnorm_mxfp8`` feeding ``gemm_fp8``."""

import pytest
import torch

from primus_turbo.pytorch.core.utils import is_gfx1250
from tests.pytorch.ref import mxfp8_ref as R
from tests.pytorch.ref.mxfp8_ref import assert_bytes_equal as _eq
from tests.pytorch.ref.mxfp8_ref import mx_input as _x
from tests.pytorch.ref.mxfp8_ref import pack_scales as _pack
from tests.pytorch.ref.mxfp8_ref import padded as _padded

pytestmark = pytest.mark.skipif(not is_gfx1250(), reason="gfx1250 kernels")
if is_gfx1250():
    from primus_turbo.flydsl.quantization import mxfp8_quant_gfx1250 as Q
    from primus_turbo.flydsl.quantization.mxfp8_quant_gfx1250 import (
        PACK_COL,
        PACK_ROW,
        SHUF_COL,
        SHUF_ROW,
    )


def _flags(lay):
    return dict(
        pack_row=bool(lay & PACK_ROW),
        shuf_row=bool(lay & SHUF_ROW),
        pack_col=bool(lay & PACK_COL),
        shuf_col=bool(lay & SHUF_COL),
    )


def _lay(ref, lay):
    return R.to_layout(ref, **_flags(lay))


def _check_grouped(out, vals, lens, rule, lay=0):
    R.check_grouped(out, vals, lens, rule, **_flags(lay))


@pytest.mark.parametrize("H", [384, 7168])
@pytest.mark.parametrize("routing", ["balanced", "skew"])
def test_moe_gather_handoff(H, routing):
    """The permute -> grouped MXFP8 handoffs as Turbo's fused grouped MLP takes them (ops.moe_gather), at the DSv3
    routing (32 experts, top-8): forward, the lazy permute placeholder quantized straight from x (with the routing
    probs' zero rule, the activation's layouts); backward, the deferred unpermute gradient resolved to dOut plus
    dest2src and quantized without probs (dY: both copies packed). Each = the grouped dual cast of the materialized
    gather, bit for bit, through quantization.grouped_quantize_fp8_with_trans."""
    from primus_turbo.pytorch.core.low_precision import ScalingGranularity, float8_e4m3
    from primus_turbo.pytorch.ops import moe_gather as MG
    from primus_turbo.pytorch.ops.quantization import grouped_quantize_fp8_with_trans
    from primus_turbo.pytorch.ops.utils import MX_A, MX_B

    T, E, k = 257, 32, 8
    g = torch.Generator(device="cuda").manual_seed(H + len(routing))
    if routing == "balanced":
        experts = (torch.arange(T, device="cuda")[:, None] + 4 * torch.arange(k, device="cuda")) % E
    else:  # hot experts, several empty ones
        w = torch.rand(E, generator=g, device="cuda") ** 4
        w[3:9] = 0
        experts = torch.multinomial(w.expand(T, E), k, generator=g)
    rmap = torch.zeros(T, E, dtype=torch.bool, device="cuda").scatter_(1, experts, True)
    probs = torch.rand(T, E, generator=g, device="cuda") * rmap
    probs[::11] = 0  # exact zeros: the permute's padding rule zeroes those rows
    e_idx, t_idx = rmap.T.nonzero(as_tuple=True)  # the permuted row order: expert-major, tokens in order
    d2s, pprobs = t_idx.to(torch.int32), probs[t_idx, e_idx]
    lens = rmap.sum(0)
    offs = torch.nn.functional.pad(torch.cumsum(lens, 0), (1, 0))
    M, row_id_map = int(d2s.numel()), torch.zeros(T, 2 * E + 1, dtype=torch.int32, device="cuda")
    x, dout = _x(T, H, seed=H, special=False), _x(T, H, seed=H + 1, special=False)

    def quant(src, rec, rec_t, **gather):
        return grouped_quantize_fp8_with_trans(
            src,
            float8_e4m3,
            ScalingGranularity.MX_BLOCKWISE,
            lens,
            offs,
            block_size=32,
            scaling_recipe=rec,
            scaling_recipe_for_trans=rec_t,
            **gather,
        )

    # forward: the placeholder resolves (with the backward plan) to x + dest2src + permuted probs
    placeholder = torch.empty(1, H, dtype=torch.bfloat16, device="cuda").expand(M, H)
    MG.register_permuted_activation_seam(
        placeholder, x, d2s, pprobs, row_id_map, fuse_backward_permute_quant=True
    )
    src, dest2src, pp, plan = MG.lookup_permuted_activation_seam(placeholder, include_plan=True)
    assert plan is not None and plan.experts == E and plan.shape == (M, H), (
        "no backward plan at the DSv3 shape"
    )
    act = R.gather_rows(x.cpu(), d2s.cpu(), pprobs.cpu())
    _check_grouped(
        quant(src, MX_A, MX_B, dest2src=dest2src, permuted_probs=pp, total_m=M),
        act,
        lens.cpu(),
        "nosat",
        PACK_ROW | PACK_COL | SHUF_COL,
    )
    MG.register_permuted_activation_seam(
        placeholder, x, d2s, pprobs, row_id_map
    )  # the dispatcher's own lookup
    _check_grouped(quant(placeholder, MX_A, MX_B), act, lens.cpu(), "nosat", PACK_ROW | PACK_COL | SHUF_COL)

    # backward: the fused MLP enrolls its output, the unpermute claims it and defers its no-probs gather of dOut
    owner, y = type("Owner", (), {})(), torch.empty(M, H, dtype=torch.bfloat16, device="cuda")
    MG.enroll_backward_gather_output(y, plan, owner)
    assert MG.claim_backward_gather_output(y, row_id_map, None, None, T, E, H) is plan
    with torch.no_grad():  # (autograd's backward)
        lazy = MG.make_backward_gather(dout, plan)
        grad, gather = MG.resolve_backward_gather(lazy, plan)
    assert (
        grad is dout
        and gather["dest2src"] is d2s
        and gather["permuted_probs"] is None
        and gather["total_m"] == M
    )
    _check_grouped(
        quant(grad, MX_A, MX_A, **gather),
        R.gather_rows(dout.cpu(), d2s.cpu()),
        lens.cpu(),
        "nosat",
        PACK_ROW | PACK_COL,
    )


def _dispatch_names():
    from primus_turbo.pytorch.core.low_precision import (
        Float8QuantConfig,
        Format,
        ScaleDtype,
        ScalingGranularity,
        ScalingRecipe,
    )

    cfg = Float8QuantConfig(
        format=Format.E4M3,
        granularity=ScalingGranularity.MX_BLOCKWISE,
        scale_dtype=ScaleDtype.E8M0,
        block_size=32,
    )
    return ScalingGranularity.MX_BLOCKWISE, ScalingRecipe, cfg


@pytest.mark.parametrize("rule", ["nosat", "even"])
def test_dispatch(rule):
    import primus_turbo.common.envs as envs
    from primus_turbo.pytorch.core.quantized_tensor import (
        QuantizedTensor,
        create_quantized_weight,
    )
    from primus_turbo.pytorch.ops.quantization import (
        grouped_quantize_fp8,
        grouped_quantize_fp8_with_trans,
        quantize_fp8,
        quantize_fp8_with_trans,
    )

    (MX, Recipe, cfg), E4 = _dispatch_names(), torch.float8_e4m3fn
    envs.set_env("PRIMUS_TURBO_MXFP8_SCALE_RULE", rule)
    try:
        x = _x(200, 330, seed=5, special=False)  # (special: a NaN block's codes in the K pad are 0xFF, not 0)
        Mp, Kp = 256, 384
        qr, sr, qc, sc = quantize_fp8_with_trans(x, E4, MX, block_size=32)
        rq, rs, rqc, rsc = R.quant_dual(x.cpu(), rule)
        for got, ref, shape, what in (
            (qr, rq, (200, Kp), "q_row"),
            (sr, rs, (200, Kp // 32), "s_row"),
            (qc, rqc, (330, Mp), "q_col"),
            (sc, rsc, (330, Mp // 32), "s_col"),
        ):
            _eq(got, _padded(ref, shape), what)
        for axis, ref, what in ((1, (rq, rs), "rows"), (0, (rqc, rsc), "cols")):
            got = quantize_fp8(x, E4, MX, block_size=32, axis=axis, scaling_recipe=Recipe())
            _eq(got[0], _padded(ref[0], got[0].shape), f"{what} q")
            _eq(got[1], _padded(ref[1], got[1].shape), f"{what} s")
        w = _x(3, 96, 200, seed=6, special=False)
        r2 = Recipe(use_2d_block=True)
        out = quantize_fp8_with_trans(
            w, E4, MX, block_size=32, scaling_recipe=r2, scaling_recipe_for_trans=r2
        )
        for b in range(3):
            for got, ref, what in zip(out, R.quant_2d(w[b].cpu(), rule), ("q", "s_row", "q_t", "s_t")):
                _eq(got[b], _padded(ref, got[b].shape), f"2d {what}[{b}]")
        qw, qw_t = create_quantized_weight(w, E4, cfg, need_weight_transpose_cache=True)
        for qt_, axis in ((qw, -1), (qw_t, -2)):
            two = QuantizedTensor.quantize(w, E4, MX, block_size=32, scaling_recipe=r2, axis=axis)
            _eq(qt_.qdata, two.qdata, f"one-pass weight {axis}")
            _eq(qt_.scale_inv, two.scale_inv, f"one-pass weight scale {axis}")
        lens = torch.tensor([0, 37, 1, 200, 0, 129])
        lens_d = lens.cuda()
        offs = torch.nn.functional.pad(torch.cumsum(lens_d, 0), (1, 0))
        xg = _x(int(lens.sum()), 330, seed=7, special=False)
        out = grouped_quantize_fp8_with_trans(xg, E4, MX, lens_d, offs, block_size=32)
        ref = R.quant_grouped(xg.cpu(), lens, rule)
        Mr, Mc = int(ref[7][-1]), int(ref[5][-1])
        _eq(out[0][:Mr], _padded(ref[0], (Mr, Kp)), "grouped q_row")
        _eq(out[1][:Mr], _padded(ref[1], (Mr, Kp // 32)), "grouped s_row")
        _eq(out[2][:, :Mc], ref[2], "grouped q_col")
        _eq(out[3][:, : Mc // 32], ref[3], "grouped s_col")
        for got, r in zip(out[4:], (ref[6], ref[7], ref[4], ref[5])):
            assert torch.equal(got.cpu(), r), "grouped lens / offs"
        for axis, idx, n in ((1, (0, 1, 6, 7), Mr), (0, (2, 3, 4, 5), Mc)):  # one copy: (q, s, lens, offs)
            one = grouped_quantize_fp8(
                xg, E4, MX, lens_d, offs, block_size=32, axis=axis, scaling_recipe=Recipe()
            )
            _eq(
                one[0][:n] if axis else one[0][:, :n],
                out[idx[0]][:n] if axis else out[idx[0]][:, :n],
                f"single {axis}",
            )
            assert torch.equal(one[3].cpu(), ref[idx[3]]), f"single {axis} offs"
        src = _x(100, 330, seed=8, special=False)
        d2s = torch.randint(0, 100, (int(lens.sum()),), device="cuda", dtype=torch.int32)
        pp = torch.rand(d2s.numel(), device="cuda")
        pp[::5] = 0
        out = grouped_quantize_fp8_with_trans(
            src, E4, MX, lens_d, offs, block_size=32, dest2src=d2s, permuted_probs=pp, total_m=d2s.numel()
        )
        ref = R.quant_grouped(R.gather_rows(src.cpu(), d2s.cpu(), pp.cpu()), lens, rule)
        _eq(out[0][:Mr], _padded(ref[0], (Mr, Kp)), "gathered q_row")
        _eq(out[2][:, :Mc], ref[2], "gathered q_col")
    finally:
        envs.set_env("PRIMUS_TURBO_MXFP8_SCALE_RULE", None)


def test_dispatch_layouts():
    """gfx1250 recipes' shuffle flags (shuffle_scale: packed scales, shuffle_out: 16x16 codes; row recipe = row
    copy, trans recipe = column copy) through the op entries, create_quantized_weight and QuantizedTensor;
    dequantize undoes them; what the kernels cannot write raises."""
    from primus_turbo.pytorch.core.quantized_tensor import (
        QuantizedTensor,
        create_quantized_weight,
    )
    from primus_turbo.pytorch.ops.quantization import (
        grouped_quantize_fp8_with_trans,
        quantize_fp8,
        quantize_fp8_with_trans,
    )

    (MX, Recipe, cfg), E4 = _dispatch_names(), torch.float8_e4m3fn
    raw2, shuf2 = Recipe(use_2d_block=True), Recipe(use_2d_block=True, shuffle_scale=True, shuffle_out=True)
    rr, rc = (
        Recipe(shuffle_scale=True),
        Recipe(shuffle_scale=True, shuffle_out=True),
    )  # A rows, B-type columns
    for w in (_x(3, 96, 320, seed=6, special=False), _x(96, 320, seed=7, special=False)):  # weights: W, W^T
        pair = create_quantized_weight(w, E4, cfg, need_weight_transpose_cache=True, scaling_recipe=shuf2)
        single = create_quantized_weight(w, E4, cfg, scaling_recipe=shuf2)[0]
        for qt_, axis in zip(pair, (-1, -2)):
            raw = QuantizedTensor.quantize(w, E4, MX, block_size=32, scaling_recipe=raw2, axis=axis)
            assert qt_.scaling_recipe == shuf2
            _eq(qt_.qdata, R.preshuffle_b16(raw.qdata.cpu()), f"weight {w.ndim}-D {axis}")
            _eq(qt_.scale_inv, _pack(raw.scale_inv), f"weight {w.ndim}-D {axis} scales")
            if w.ndim == 2:
                _eq(qt_.dequantize(), raw.dequantize(), f"weight {axis} dequantize")
        _eq(single.qdata, pair[0].qdata, "weight without the transpose cache")
        _eq(single.scale_inv, pair[0].scale_inv, "weight scales without the transpose cache")
    with pytest.raises(NotImplementedError):  # the GEMM layouts have no view to another shape
        pair[0].view(48, 640)
    x = _x(200, 320, seed=5, special=False)
    raw = quantize_fp8_with_trans(x, E4, MX, block_size=32)
    got = quantize_fp8_with_trans(x, E4, MX, block_size=32, scaling_recipe=rr, scaling_recipe_for_trans=rc)
    for g_, e_, what in zip(
        got, _lay(raw, PACK_ROW | PACK_COL | SHUF_COL), ("q_row", "s_row", "q_col", "s_col")
    ):
        _eq(g_, e_, f"activation {what}")
    lens = torch.tensor([0, 37, 1, 200, 0, 129])
    lens_d = lens.cuda()
    offs = torch.nn.functional.pad(torch.cumsum(lens_d, 0), (1, 0))
    xg = _x(int(lens.sum()), 384, seed=7, special=False)
    out = grouped_quantize_fp8_with_trans(
        xg, E4, MX, lens_d, offs, block_size=32, scaling_recipe=rr, scaling_recipe_for_trans=rc
    )
    _check_grouped(out, xg, lens, "nosat", PACK_ROW | PACK_COL | SHUF_COL)
    for axis, rec in ((-1, rr), (-2, rc)):  # dequantize: dense and grouped, each copy
        for kw, t in (({}, x), ({"group_lens": lens_d}, xg)):
            a = QuantizedTensor.quantize(t, E4, MX, block_size=32, scaling_recipe=rec, axis=axis, **kw)
            b = QuantizedTensor.quantize(t, E4, MX, block_size=32, scaling_recipe=Recipe(), axis=axis, **kw)
            _eq(a.dequantize(), b.dequantize(), f"dequantize {axis} {list(kw)}")
    with pytest.raises(
        NotImplementedError
    ):  # fp16 / E5M2 / mixed block shapes: not these kernels, no layouts
        quantize_fp8_with_trans(
            x.half(), E4, MX, block_size=32, scaling_recipe=rr, scaling_recipe_for_trans=rc
        )
    with pytest.raises(NotImplementedError):
        quantize_fp8(x, torch.float8_e5m2, MX, block_size=32, axis=1, scaling_recipe=rr)
    with pytest.raises(NotImplementedError):
        quantize_fp8_with_trans(x, E4, MX, block_size=32, scaling_recipe=shuf2, scaling_recipe_for_trans=rc)
    with pytest.raises(AssertionError):  # 16x16 codes need rows % 16 == 0
        quantize_fp8(
            _x(100, 320, special=False),
            E4,
            MX,
            block_size=32,
            axis=1,
            scaling_recipe=Recipe(shuffle_out=True),
        )


def test_strict_hip_fallback(monkeypatch):
    """PRIMUS_TURBO_MX_STRICT=2: an MXFP8 quantization the gfx1250 kernels do not take (and HIP would) raises."""
    import primus_turbo.common.envs as envs
    from primus_turbo.pytorch.ops.quantization import (
        grouped_quantize_fp8_with_trans,
        quantize_fp8_with_trans,
    )

    (MX, Recipe, _), E4 = _dispatch_names(), torch.float8_e4m3fn
    monkeypatch.setenv("PRIMUS_TURBO_MX_STRICT", "2")
    envs.reset_cache()
    x = _x(64, 128, special=False)
    lens = torch.tensor([10, 54], device="cuda")
    offs = torch.nn.functional.pad(torch.cumsum(lens, 0), (1, 0))
    quantize_fp8_with_trans(x, E4, MX, block_size=32)  # the FlyDSL kernels: no fallback
    grouped_quantize_fp8_with_trans(x, E4, MX, lens, offs, block_size=32)
    r2, r1 = Recipe(use_2d_block=True), Recipe()
    for call in (
        lambda: quantize_fp8_with_trans(x.half(), E4, MX, block_size=32),
        lambda: quantize_fp8_with_trans(
            x, E4, MX, block_size=32, scaling_recipe=r2, scaling_recipe_for_trans=r1
        ),
        lambda: grouped_quantize_fp8_with_trans(x, E4, MX, lens, offs, block_size=32, scaling_recipe=r2),
    ):
        with pytest.raises(RuntimeError, match="PRIMUS_TURBO_MX_STRICT=2"):
            call()


def test_operand_slack():
    """Every GEMM operand the quantizers allocate (codes and scales, both copies, any layout and producer, the cached
    weights) has >= 3840 bytes of storage past its last byte: the gfx1250 MX GEMM's partial last K step addresses up
    to that far past a 16x16-preshuffled B (the quantizers leave GEMM_SLACK = 4 KiB)."""
    from primus_turbo.pytorch.core.quantized_tensor import create_quantized_weight
    from primus_turbo.pytorch.ops.quantization import quantize_fp8_with_trans

    (MX, Recipe, cfg), E4 = _dispatch_names(), torch.float8_e4m3fn

    def slack(t):
        return t.untyped_storage().nbytes() - (t.storage_offset() + t.numel()) * t.element_size()

    x = _x(208, 256, special=False)
    lens = torch.tensor([0, 37, 1, 170], device="cuda")
    offs = torch.nn.functional.pad(torch.cumsum(lens, 0), (1, 0))
    l1 = _x(208, 512, special=False)
    probs = torch.rand(208, device="cuda")
    d2s = torch.randint(0, 208, (208,), device="cuda", dtype=torch.int32)
    outs = {
        "dual": Q.quant_mxfp8_raw(x),
        "dual 16x16": Q.quant_mxfp8_raw(x, pad=True, layout=15)[0:4],
        "rows": Q.quant_mxfp8_rows(x),
        "cols": Q.quant_mxfp8_cols(x, layout=PACK_COL),
        "2d": Q.quant_mxfp8_raw_batched(_x(3, 64, 96, special=False), row_2d=True, col_2d=True, layout=15),
        "grouped": Q.grouped_quant_mxfp8_raw(x, lens, offs)[:4],
        "glu": Q.grouped_glu_quant_mxfp8(l1, probs, lens, offs, layout=PACK_ROW | PACK_COL | SHUF_COL)[:4],
        "dglu": Q.grouped_dglu_quant_mxfp8(x, l1, probs, lens, offs)[1:5],
        "gather": Q.grouped_gather_quant_mxfp8(x, d2s, lens, offs, probs)[:4],
        "op": quantize_fp8_with_trans(x, E4, MX, block_size=32),
    }
    shuf2 = Recipe(use_2d_block=True, shuffle_scale=True, shuffle_out=True)
    w = create_quantized_weight(
        _x(64, 96, special=False), E4, cfg, need_weight_transpose_cache=True, scaling_recipe=shuf2
    )
    outs["weight"] = [t for qt_ in w for t in (qt_.qdata, qt_.scale_inv)]
    for name, ts in outs.items():
        for i, t in enumerate(ts):
            assert t is not None and slack(t) >= 3840, f"{name}[{i}]: only {slack(t)} bytes past the end"


def _rms_inputs(M, H, stride_pad=0, seed=0):
    """x [M, H] bf16 (a column slice of [M, H + stride_pad] when stride_pad), rows scaled over 2^+-20, a zero
    row, a row whose squares overflow fp32 (rstd 0), a constant row; gamma [H] bf16 around 1."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    full = torch.randn(M, H + stride_pad, generator=g, device="cuda")
    full = full * torch.exp2(torch.randint(-20, 20, (M, 1), generator=g, device="cuda").float())
    if M > 3:
        full[1], full[2], full[3] = 0.0, 3.0e19, -0.75
    return full.to(torch.bfloat16)[:, :H], (1 + 0.25 * torch.randn(H, generator=g, device="cuda")).to(
        torch.bfloat16
    )


def _mx_config():
    from primus_turbo.pytorch.core.low_precision import (
        Float8QuantConfig,
        Format,
        ScaleDtype,
        ScalingGranularity,
    )

    return Float8QuantConfig(
        format=Format.E4M3,
        granularity=ScalingGranularity.MX_BLOCKWISE,
        scale_dtype=ScaleDtype.E8M0,
        block_size=32,
    )


@pytest.mark.parametrize(
    "M,H,N_out,stride_pad", [(256, 512, 1024, 64), (300, 1536, 512, 0), (512, 7168, 576, 0)]
)
def test_rmsnorm_mxfp8_gemm(M, H, N_out, stride_pad, monkeypatch):
    """``rmsnorm_mxfp8`` feeding ``gemm_fp8`` with its pair (the recipes the op layer picks for the weight): no
    re-quantization (strict mode), and y, the GEMM output and the gradients (through y and through the pair) bitwise
    those of Turbo's RMSNorm followed by ``gemm_fp8`` (the same rstd and y, the same codes, the same two gradients
    summed)."""
    import primus_turbo.common.envs as envs
    from primus_turbo.pytorch.core.low_precision import ScalingRecipe
    from primus_turbo.pytorch.ops.gemm_fp8 import gemm_fp8
    from primus_turbo.pytorch.ops.normalization import rmsnorm, rmsnorm_mxfp8
    from primus_turbo.pytorch.ops.utils import MX_A, MX_B, mx_gemm_layout

    cfg, eps = _mx_config(), 1e-6
    g = torch.Generator(device="cuda").manual_seed(M)
    w = (torch.randn(N_out, H, generator=g, device="cuda") / H**0.5).to(torch.bfloat16)
    rr, rc = (MX_A, MX_B) if mx_gemm_layout(w) else (ScalingRecipe(), ScalingRecipe())
    x0, gamma0 = _rms_inputs(M, H, stride_pad, seed=7)
    v = torch.randn(M, H, generator=g, device="cuda")
    monkeypatch.setenv("PRIMUS_TURBO_MX_STRICT", "1")
    envs.reset_cache()

    def run(fused):
        x, gamma = x0.detach().clone().requires_grad_(), gamma0.detach().clone().requires_grad_()
        if fused:
            y, pair = rmsnorm_mxfp8(x, gamma, eps, row_recipe=rr, col_recipe=rc)
            out = gemm_fp8(pair, w, trans_b=True, config=cfg)
        else:
            y = rmsnorm(x, gamma, eps)
            out = gemm_fp8(y, w, trans_b=True, config=cfg)
        ((out.float() ** 2).sum() + (y.float() * v).sum()).backward()
        return y.detach(), out.detach(), x.grad, gamma.grad

    y, out, dx, dg = run(True)
    assert torch.equal(out, gemm_fp8(y, w, trans_b=True, config=cfg)), "the pair is the MXFP8 cast of y"
    y2, out2, dx2, dg2 = run(False)
    for a, b, what in ((y, y2, "y"), (out, out2, "out"), (dx, dx2, "dx"), (dg, dg2, "dgamma")):
        assert torch.equal(a, b), f"{what}: not bitwise the unfused path's"


def test_rmsnorm_mxfp8_fp16_falls_back():
    """fp16 x, which gfx1250's fused kernels and quantizers do not take: Turbo's RMSNorm, then the MXFP8 dual
    quantization of its output in raw layouts (the default recipes follow x's dtype), instead of raising."""
    from primus_turbo.pytorch.core.low_precision import ScalingGranularity, ScalingRecipe, float8_e4m3
    from primus_turbo.pytorch.ops.normalization import rmsnorm, rmsnorm_mxfp8
    from primus_turbo.pytorch.ops.quantization import quantize_fp8_with_trans

    g = torch.Generator(device="cuda").manual_seed(9)
    x0 = torch.randn(256, 1536, generator=g, device="cuda").half()
    gamma0 = (1 + 0.25 * torch.randn(1536, generator=g, device="cuda")).half()
    x, gamma = x0.clone().requires_grad_(), gamma0.clone().requires_grad_()
    y, pair = rmsnorm_mxfp8(x, gamma, 1e-6)
    assert pair.data.scaling_recipe == pair.data_t.scaling_recipe == ScalingRecipe()
    y_ref = rmsnorm(x0, gamma0, 1e-6)
    _eq(y, y_ref, "y")
    ref = quantize_fp8_with_trans(y_ref, float8_e4m3, ScalingGranularity.MX_BLOCKWISE, block_size=32)
    for got, exp, what in zip((pair.data.qdata, pair.data.scale_inv, pair.data_t.qdata, pair.data_t.scale_inv), ref,
                              ("q_row", "s_row", "q_col", "s_col")):  # fmt: skip
        _eq(got, exp, what)
    y.float().sum().backward()
    assert x.grad is not None and gamma.grad is not None
