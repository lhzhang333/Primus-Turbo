###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""The op layer on gfx1250's MXFP8 quantizers (primus_turbo/flydsl/quantization/mxfp8_quant_gfx1250.py), bit for
bit against tests/pytorch/ref/mxfp8_ref.py: the quantization ops' dispatch (HIP layouts, and the GEMM layouts the
recipes' shuffle flags select), the strict mode's HIP fallback, the GEMM operands' storage slack."""

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
