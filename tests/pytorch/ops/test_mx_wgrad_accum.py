###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MXFP8 wgrad accumulated into main_grad (fuse_bgrad_accum_pattern="megatron", beta=1) against the unfused wgrad
added by the framework, for gemm_fp8 and grouped_gemm_fp8, over two micro-batches on top of an existing gradient;
which backends / main_grad dtypes ``mx_wgrad_accum_supported`` reports; and which MX layouts the op layer picks per
arch, format and dtype (``mx_weight_layout_recipe`` for weight caches, ``mx_gemm_layout`` in the ops; host-only, on a
faked arch)."""

import pytest
import torch

import primus_turbo.pytorch.ops.utils as ops_utils
from primus_turbo.pytorch.core.backend import (
    BackendChoice,
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
    check_mxfp8_support,
    float8_e4m3,
)
from primus_turbo.pytorch.core.quantized_tensor import QuantizedTensor, QuantizedTensorPair
from primus_turbo.pytorch.ops import (
    gemm_fp8,
    grouped_gemm_fp8,
    mx_weight_layout_recipe,
    mx_wgrad_accum_backend,
    mx_wgrad_accum_supported,
)

MX = Float8QuantConfig(
    format=Format.E4M3,
    granularity=ScalingGranularity.MX_BLOCKWISE,
    scale_dtype=ScaleDtype.E8M0,
    block_size=32,
)
requires_mxfp8 = pytest.mark.skipif(
    not torch.cuda.is_available() or not check_mxfp8_support()[0], reason="needs MXFP8"
)


@pytest.fixture(autouse=True)
def _backend_state():
    GlobalBackendManager.reset()
    yield
    GlobalBackendManager.reset()


@pytest.mark.parametrize(
    "grouped, pinned, arch, dtype, want",
    [
        (False, None, "gfx950", torch.bfloat16, True),  # hipBLASLt
        (False, None, "gfx950", torch.float32, True),
        (False, None, "gfx1250", torch.bfloat16, True),  # FlyDSL
        (False, None, "gfx1250", torch.float32, True),
        (False, None, "gfx1250", torch.float16, False),
        (False, BackendType.HIPBLASLT, "gfx1250", torch.bfloat16, False),  # no hipBLASLt MX GEMM there
        (False, BackendType.TRITON, "gfx950", torch.bfloat16, False),  # the dense Triton MX path is gfx1250's
        (False, BackendType.TRITON, "gfx1250", torch.float32, True),
        (True, None, "gfx1250", torch.float32, True),
        (True, BackendType.TRITON, "gfx1250", torch.float32, True),
        (True, BackendType.FLYDSL, "gfx950", torch.bfloat16, True),  # gfx950 FlyDSL: 16-bit only
        (True, BackendType.FLYDSL, "gfx950", torch.float32, False),
        (True, BackendType.CK, "gfx950", torch.bfloat16, False),  # no MX beta=1 epilogue
    ],
)
def test_supported(monkeypatch, grouped, pinned, arch, dtype, want):
    monkeypatch.setattr(ops_utils, "is_gfx942", lambda: False)
    monkeypatch.setattr(ops_utils, "is_gfx950", lambda: arch == "gfx950")
    monkeypatch.setattr(ops_utils, "is_gfx1250", lambda: arch == "gfx1250")
    choice = BackendChoice(backend=pinned) if pinned is not None else None
    getter = "get_grouped_gemm_backend" if grouped else "get_gemm_backend"
    monkeypatch.setattr(GlobalBackendManager, getter, classmethod(lambda cls, precision: choice))
    assert mx_wgrad_accum_supported(grouped, dtype) is want


@pytest.mark.parametrize(
    "arch, grouped, pinned, shape, want",
    [
        ("gfx1250", False, None, (64, 64), True),  # the FlyDSL MX GEMMs read it
        ("gfx1250", False, None, (48, 80), True),
        ("gfx1250", True, None, (4, 64, 64), True),
        ("gfx1250", True, None, (4, 48, 64), False),  # grouped packed scales: N and K % 32
        ("gfx1250", False, BackendType.TRITON, (64, 64), False),  # another FP8 backend pinned
        ("gfx1250", True, BackendType.TRITON, (4, 64, 64), False),
        ("gfx1250", False, None, (64, 72), False),  # no 16x16 blocks
        ("gfx950", False, None, (64, 64), False),
    ],
)
def test_weight_layout_recipe(monkeypatch, arch, grouped, pinned, shape, want):
    monkeypatch.setattr(ops_utils, "is_gfx1250", lambda: arch == "gfx1250")
    choice = BackendChoice(backend=pinned) if pinned is not None else None
    getter = "get_grouped_gemm_backend" if grouped else "get_gemm_backend"
    monkeypatch.setattr(GlobalBackendManager, getter, classmethod(lambda cls, precision: choice))
    assert mx_weight_layout_recipe(grouped, shape) == (
        ops_utils.MX_W if want else ScalingRecipe(use_2d_block=True)
    )
    w = torch.empty(shape, dtype=torch.bfloat16)
    assert ops_utils.mx_gemm_layout(w, grouped, config=MX, out_dtype=torch.bfloat16) is want  # the op agrees


def _fake_arch(monkeypatch, arch):
    for name in ("gfx942", "gfx950", "gfx1250"):
        monkeypatch.setattr(ops_utils, f"is_{name}", lambda name=name: arch == name)


def _cfg(fmt):
    return Float8QuantConfig(
        format=fmt, granularity=ScalingGranularity.MX_BLOCKWISE, scale_dtype=ScaleDtype.E8M0, block_size=32
    )


def _qt(shape, recipe, axis=-1):
    """A host-side MXFP8 QuantizedTensor carrying ``recipe`` (the layout decisions read its recipe only)."""
    scale = torch.zeros(*shape[:-1], shape[-1] // 32, dtype=torch.uint8)
    return QuantizedTensor(
        torch.zeros(shape, dtype=float8_e4m3),
        scale,
        shape=torch.Size(shape),
        orig_dtype=torch.bfloat16,
        dest_dtype=float8_e4m3,
        granularity=ScalingGranularity.MX_BLOCKWISE,
        block_size=32,
        scaling_recipe=recipe,
        quantized_axis=axis,
    )


@pytest.mark.parametrize(
    "fmt, w_dtype, a_dtype, out_dtype, want",
    [
        (Format.E4M3, torch.bfloat16, torch.bfloat16, torch.bfloat16, True),
        (Format.E4M3, torch.float16, torch.float16, torch.float16, False),  # fp16: raw layouts
        (Format.E4M3, torch.bfloat16, torch.float16, torch.bfloat16, False),
        (Format.E4M3, torch.bfloat16, torch.bfloat16, torch.float16, False),
        (Format.HYBRID, torch.bfloat16, torch.bfloat16, torch.bfloat16, False),  # E5M2 gradients
        (Format.E5M2, torch.bfloat16, torch.bfloat16, torch.bfloat16, False),
    ],
)
def test_gemm_layout_dtypes(monkeypatch, fmt, w_dtype, a_dtype, out_dtype, want):
    """gfx1250 lays out only what its quantizers write (bf16 -> E4M3 both ways, bf16 out); else the raw layouts."""
    _fake_arch(monkeypatch, "gfx1250")
    cfg, w, a = _cfg(fmt), torch.empty(64, 64, dtype=w_dtype), torch.empty(32, 64, dtype=a_dtype)
    for grouped in (False, True):
        wg = w.expand(2, 64, 64) if grouped else w
        assert ops_utils.mx_gemm_layout(wg, grouped, a=a, config=cfg, out_dtype=out_dtype) is want
        cached = mx_weight_layout_recipe(grouped, wg.shape, config=cfg, dtype=w_dtype) == ops_utils.MX_W
        assert cached is (fmt == Format.E4M3 and w_dtype == torch.bfloat16)


def test_gemm_layout_forced(monkeypatch):
    """A cached MX_W weight or a quantized MX_A input forces the layout, and raises where it cannot run; a raw
    cache keeps the raw layouts."""
    _fake_arch(monkeypatch, "gfx1250")
    bf16, w = torch.bfloat16, _qt((64, 64), ops_utils.MX_W)
    assert ops_utils.mx_gemm_layout(w, a=torch.empty(32, 64, dtype=bf16), config=MX, out_dtype=bf16)
    raw = _qt((64, 64), ScalingRecipe(use_2d_block=True))
    assert not ops_utils.mx_gemm_layout(raw, a=torch.empty(32, 64, dtype=bf16), config=MX, out_dtype=bf16)
    hybrid, fp16 = _cfg(Format.HYBRID), torch.float16
    for cfg, a_dtype, out_dtype in ((hybrid, bf16, bf16), (MX, fp16, bf16), (MX, bf16, fp16)):
        with pytest.raises(ValueError, match="cached weight is in gfx1250's MX GEMM layout"):
            ops_utils.mx_gemm_layout(w, a=torch.empty(32, 64, dtype=a_dtype), config=cfg, out_dtype=out_dtype)
    with pytest.raises(ValueError, match="quantized input is in gfx1250's MX GEMM layout"):
        ops_utils.mx_gemm_layout(
            torch.empty(64, 64, dtype=bf16), a=_qt((32, 64), ops_utils.MX_A), config=_cfg(Format.HYBRID)
        )
    choice = BackendChoice(backend=BackendType.TRITON)
    monkeypatch.setattr(GlobalBackendManager, "get_gemm_backend", classmethod(lambda cls, precision: choice))
    with pytest.raises(ValueError, match="no other FP8 GEMM backend pinned"):
        ops_utils.mx_gemm_layout(w, config=MX, out_dtype=bf16)


def test_gemm_layout_gfx950(monkeypatch):
    """gfx950 reads the shuffle flags as the CDNA4 layouts: a cache whose recipe equals MX_W does not switch the op
    to gfx1250's layouts, and gemm_fp8 refuses it as the raw-recipe check always did."""
    import importlib

    gemm_fp8_mod = importlib.import_module("primus_turbo.pytorch.ops.gemm_fp8")

    _fake_arch(monkeypatch, "gfx950")
    w, w_t = _qt((64, 64), ops_utils.MX_W), _qt((64, 64), ops_utils.MX_W, axis=-2)
    assert not ops_utils.mx_gemm_layout(w, a=torch.empty(32, 64, dtype=torch.bfloat16), config=MX)
    assert not ops_utils.mx_gemm_layout(torch.empty(64, 64, dtype=torch.bfloat16))
    assert mx_weight_layout_recipe(False, (64, 64)) == ScalingRecipe(use_2d_block=True)
    monkeypatch.setattr(gemm_fp8_mod, "check_mxfp8_support", lambda: (True, ""))
    a = QuantizedTensorPair(_qt((32, 64), ScalingRecipe()), _qt((32, 64), ScalingRecipe(), axis=-2))
    with pytest.raises(AssertionError, match="scaling_recipe"):
        gemm_fp8(a, QuantizedTensorPair(w, w_t), trans_b=True, out_dtype=torch.bfloat16, config=MX)


def test_operand_recipe_checks(monkeypatch):
    """gfx1250: every pre-quantized copy (a, a_t, b, b_t) must carry the recipe its GEMM reads (ValueError);
    elsewhere main's checks: a and b asserted, the transposed copies taken as given."""
    import importlib

    gemm_fp8_mod = importlib.import_module("primus_turbo.pytorch.ops.gemm_fp8")
    reached = []

    def gemm(*args, **kw):
        reached.append(1)
        return torch.empty(32, 64)

    monkeypatch.setattr(gemm_fp8_mod, "check_mxfp8_support", lambda: (True, ""))
    monkeypatch.setattr(gemm_fp8_mod, "mx1250_gemm_direct", lambda *args, **kw: None)
    monkeypatch.setattr(gemm_fp8_mod, "gemm_fp8_impl", gemm)
    raw_a, raw_w, mx_w = ScalingRecipe(), ScalingRecipe(use_2d_block=True), ops_utils.MX_W
    a = QuantizedTensorPair(_qt((32, 64), raw_a), _qt((32, 64), raw_w, axis=-2))  # a_t: 2-D blocks
    w = QuantizedTensorPair(_qt((64, 64), raw_w), _qt((64, 64), mx_w, axis=-2))  # b_t: another layout
    kw = dict(trans_b=True, out_dtype=torch.bfloat16, config=MX)
    _fake_arch(monkeypatch, "gfx950")
    gemm_fp8(a, w, **kw)
    assert reached
    _fake_arch(monkeypatch, "gfx1250")
    with pytest.raises(ValueError, match="a_t was quantized with .* raw MX layout"):
        gemm_fp8(a, w, **kw)
    a = QuantizedTensorPair(a.data, _qt((32, 64), raw_a, axis=-2))
    with pytest.raises(ValueError, match="b_t was quantized with"):
        gemm_fp8(a, w, **kw)


def test_autotune_keys_carry_the_layout():
    """A backend tuned on raw MX operands is not reused for gfx1250's GEMM layout (16x16 B, packed scales)."""
    from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import GEMMFP8KernelDispatcher
    from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
        GroupedGEMMFP8KernelDispatcher,
        GroupedGEMMFP8VariableKKernelDispatcher,
    )

    a, b, g = torch.empty(64, 128), torch.empty(4, 32, 128), torch.zeros(4, dtype=torch.int64)
    raw, packed = torch.empty(64, 4), torch.empty(2, 1, 32, 4)
    mx = ScalingGranularity.MX_BLOCKWISE
    common = dict(trans_a=False, trans_b=True, out_dtype=torch.bfloat16, granularity=mx)
    dense = dict(a=a, b=b[0], trans_c=False, b_scale_inv=raw, **common)
    grouped = dict(a=a, b=b, b_scales=raw, group_lens=g, group_offs=g, num_cu=None, **common)
    for make_key, kw, scale in (
        (GEMMFP8KernelDispatcher.make_key, dense, "a_scale_inv"),
        (GroupedGEMMFP8KernelDispatcher.make_key, grouped, "a_scales"),
        (GroupedGEMMFP8VariableKKernelDispatcher.make_key, dict(grouped, trans_c=False), "a_scales"),
    ):
        layouts = [{scale: s, "b_preshuffled": shuf} for s in (raw, packed) for shuf in (False, True)]
        assert len({make_key(**kw, **layout) for layout in layouts}) == 4


def test_scale_rule_read_on_gfx1250_only(monkeypatch):
    """PRIMUS_TURBO_MXFP8_SCALE_RULE belongs to gfx1250's quantizers: other archs never read it."""
    import primus_turbo.common.envs as envs
    import primus_turbo.pytorch.kernels.quantization.quantization_impl as qimpl

    monkeypatch.setenv("PRIMUS_TURBO_MXFP8_SCALE_RULE", "rne")
    envs.reset_cache()
    x, raw = torch.empty(32, 64, dtype=torch.bfloat16), ScalingRecipe()
    monkeypatch.setattr(qimpl, "is_gfx1250", lambda: False)
    assert qimpl._gfx1250_mxfp8(x, torch.float8_e4m3fn, raw, raw) is None
    monkeypatch.setattr(qimpl, "is_gfx1250", lambda: True)
    with pytest.raises(ValueError, match="PRIMUS_TURBO_MXFP8_SCALE_RULE='rne'"):
        qimpl._gfx1250_mxfp8(x, torch.float8_e4m3fn, raw, raw)


def test_fused_glu_e4m3_only_on_gfx1250(monkeypatch):
    """gfx1250's fused MXFP8 GLU / dGLU quantize to E4M3 both ways: HYBRID is refused before any GEMM runs."""
    import primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl as gimpl

    monkeypatch.setattr(gimpl, "is_gfx1250", lambda: True)
    gimpl._check_mx_glu_dispatch(MX, False, True)
    with pytest.raises(AssertionError, match="E4M3 only"):
        gimpl._check_mx_glu_dispatch(_cfg(Format.HYBRID), False, True)


def test_no_mx_on_gfx942(monkeypatch):
    monkeypatch.setattr(ops_utils, "is_gfx942", lambda: True)
    monkeypatch.setattr(ops_utils, "is_gfx1250", lambda: False)
    assert not mx_wgrad_accum_supported(False, torch.bfloat16)
    assert not mx_wgrad_accum_supported(True, torch.bfloat16)
    assert mx_wgrad_accum_backend(True) == BackendType.TRITON


def _accumulate(op, weight, inputs, grad_outputs, main_grad0, fused):
    """main_grad after the micro-batches: fused (beta=1 into main_grad) or main_grad += the returned wgrad."""
    w = torch.nn.Parameter(weight.clone())
    w.main_grad = main_grad0.clone()
    w.grad_added_to_main_grad = False
    for x, dy in zip(inputs, grad_outputs):
        out = op(x, w, fused)
        out.backward(dy)
        if not fused:
            w.main_grad.add_(w.grad.to(w.main_grad.dtype))
            w.grad = None
    return w.main_grad, w.grad_added_to_main_grad


@requires_mxfp8
@pytest.mark.parametrize("pinned", [None, BackendType.TRITON])
@pytest.mark.parametrize("main_grad_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("grouped", [False, True])
def test_fused_matches_unfused(grouped, main_grad_dtype, pinned):
    """The default backend's beta=1 wgrad, and Triton's pinned (the fallback; gfx1250 only for the dense one)."""
    torch.manual_seed(0)
    if pinned is not None:
        (GlobalBackendManager.set_grouped_gemm_backend if grouped else GlobalBackendManager.set_gemm_backend)(
            pinned, PrecisionType.FP8
        )
    if grouped:
        G, m, n, k = 4, 256, 512, 384
        group_lens = torch.full((G,), m, device="cuda", dtype=torch.int64)
        weight = torch.randn(G, n, k, device="cuda", dtype=torch.bfloat16) / 16
        rows = G * m

        def op(x, w, fused):
            return grouped_gemm_fp8(
                x,
                w,
                group_lens,
                trans_b=True,
                config=MX,
                fuse_bgrad_accum_pattern="megatron" if fused else None,
            )

    else:
        rows, n, k = 512, 768, 384
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / 16

        def op(x, w, fused):
            return gemm_fp8(
                x, w, trans_b=True, out_dtype=torch.bfloat16, config=MX,
                fuse_bgrad_accum_pattern="megatron" if fused else None,
            )  # fmt: skip

    if not mx_wgrad_accum_supported(grouped, main_grad_dtype):
        pytest.skip(f"no MXFP8 beta=1 wgrad into {main_grad_dtype} here")
    inputs = [torch.randn(rows, k, device="cuda", dtype=torch.bfloat16) / 16 for _ in range(2)]
    grad_outputs = [torch.randn(rows, n, device="cuda", dtype=torch.bfloat16) / 16 for _ in range(2)]
    main_grad0 = (torch.randn(weight.shape, device="cuda") / 16).to(main_grad_dtype)

    fused, fused_flag = _accumulate(op, weight, inputs, grad_outputs, main_grad0, fused=True)
    unfused, _ = _accumulate(op, weight, inputs, grad_outputs, main_grad0, fused=False)
    assert fused_flag
    # the same quantized operands either way: the fused epilogue rounds once, the unfused path twice
    torch.testing.assert_close(fused, unfused, rtol=1e-2, atol=1e-2)
    assert not torch.equal(fused, main_grad0), "main_grad was not accumulated into"


@requires_mxfp8
@pytest.mark.parametrize("main_grad_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("grouped", [False, True])
def test_dummy_wgrad_is_a_view_of_one_buffer(grouped, main_grad_dtype):
    """With the wgrad fused into main_grad, autograd gets a dummy: a view of one flat buffer per dtype (whatever the
    weight's shape), which AccumulateGrad takes over as the weight's .grad without a copy; ``zero=True`` zeroes
    just that view."""
    if not mx_wgrad_accum_supported(grouped, main_grad_dtype):
        pytest.skip(f"no MXFP8 beta=1 wgrad into {main_grad_dtype} here")
    torch.manual_seed(0)
    shape = (4, 512, 384) if grouped else (768, 384)
    rows = 4 * 256 if grouped else 512
    w = torch.nn.Parameter(torch.randn(shape, device="cuda", dtype=torch.bfloat16) / 16)
    w.main_grad = torch.zeros(shape, device="cuda", dtype=main_grad_dtype)
    w.grad_added_to_main_grad = False
    x = torch.randn(rows, shape[-1], device="cuda", dtype=torch.bfloat16)
    if grouped:
        lens = torch.full((4,), 256, device="cuda", dtype=torch.int64)
        out = grouped_gemm_fp8(x, w, lens, trans_b=True, config=MX, fuse_bgrad_accum_pattern="megatron")
    else:
        out = gemm_fp8(x, w, trans_b=True, out_dtype=torch.bfloat16, config=MX, fuse_bgrad_accum_pattern="megatron")
    out.backward(torch.randn_like(out))
    buf = ops_utils._dummy_wgrads[(torch.bfloat16, torch.cuda.current_device())]
    assert w.grad_added_to_main_grad and w.main_grad.abs().sum() > 0
    assert w.grad.shape == w.shape and w.grad.data_ptr() == buf.data_ptr(), "the dummy was copied, not taken over"
    small = ops_utils._get_dummy_wgrad([3, 5], torch.bfloat16, zero=True)
    assert small.data_ptr() == buf.data_ptr() and not small.any()
