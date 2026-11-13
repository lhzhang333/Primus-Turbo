###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MXFP8 wgrad accumulated into main_grad (fuse_bgrad_accum_pattern="megatron", beta=1) against the unfused wgrad
added by the framework, for gemm_fp8 and grouped_gemm_fp8, over two micro-batches on top of an existing gradient;
which backends / main_grad dtypes ``mx_wgrad_accum_supported`` reports; and which weight-cache layout
``mx_weight_layout_recipe`` asks for (the op layer's ``mx_gemm_layout`` agreeing)."""

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
)
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
    "grouped, pinned, arch, fly_accum, dtype, want",
    [
        (False, None, "gfx950", False, torch.bfloat16, True),  # hipBLASLt
        (False, None, "gfx950", False, torch.float32, True),
        # gfx1250: FlyDSL, once its kernel has the beta=1 epilogue (until then the unfused path)
        (False, None, "gfx1250", False, torch.bfloat16, False),
        (False, None, "gfx1250", True, torch.bfloat16, True),
        (False, None, "gfx1250", True, torch.float32, True),
        (False, None, "gfx1250", True, torch.float16, False),
        (
            False,
            BackendType.HIPBLASLT,
            "gfx1250",
            True,
            torch.bfloat16,
            False,
        ),  # hipBLASLt's MX GEMM is wrong there
        (
            False,
            BackendType.TRITON,
            "gfx950",
            False,
            torch.bfloat16,
            False,
        ),  # the dense Triton MX path is gfx1250's
        (False, BackendType.TRITON, "gfx1250", False, torch.float32, True),
        (True, None, "gfx1250", False, torch.bfloat16, False),
        (True, None, "gfx1250", True, torch.float32, True),
        (True, BackendType.TRITON, "gfx1250", False, torch.float32, True),
        (True, BackendType.FLYDSL, "gfx950", False, torch.bfloat16, True),  # gfx950 FlyDSL: 16-bit only
        (True, BackendType.FLYDSL, "gfx950", False, torch.float32, False),
        (True, BackendType.CK, "gfx950", False, torch.bfloat16, False),  # no MX beta=1 epilogue
    ],
)
def test_supported(monkeypatch, grouped, pinned, arch, fly_accum, dtype, want):
    monkeypatch.setattr(ops_utils, "is_gfx942", lambda: False)
    monkeypatch.setattr(ops_utils, "is_gfx950", lambda: arch == "gfx950")
    monkeypatch.setattr(ops_utils, "is_gfx1250", lambda: arch == "gfx1250")
    monkeypatch.setattr(ops_utils, "_mx1250_flydsl_accum", lambda grouped: fly_accum)
    choice = BackendChoice(backend=pinned) if pinned is not None else None
    getter = "get_grouped_gemm_backend" if grouped else "get_gemm_backend"
    monkeypatch.setattr(GlobalBackendManager, getter, classmethod(lambda cls, precision: choice))
    assert mx_wgrad_accum_supported(grouped, dtype) is want


@pytest.mark.parametrize(
    "arch, grouped, kernels, pinned, shape, want",
    [
        ("gfx1250", False, False, None, (64, 64), True),  # dense: the FlyDSL dense MX GEMM reads it
        (
            "gfx1250",
            True,
            False,
            None,
            (4, 64, 64),
            False,
        ),  # grouped: not until the grouped entry points are in
        ("gfx1250", True, True, None, (4, 64, 64), True),
        ("gfx1250", False, False, BackendType.TRITON, (64, 64), False),  # another FP8 backend pinned
        ("gfx1250", True, True, BackendType.TRITON, (4, 64, 64), False),
        ("gfx1250", False, False, None, (64, 72), False),  # no 16x16 blocks
        ("gfx950", False, False, None, (64, 64), False),
    ],
)
def test_weight_layout_recipe(monkeypatch, arch, grouped, kernels, pinned, shape, want):
    import primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl as gimpl

    monkeypatch.setattr(ops_utils, "is_gfx1250", lambda: arch == "gfx1250")
    monkeypatch.setattr(gimpl, "mx1250_grouped", lambda: (min, max) if kernels else (None, None))
    choice = BackendChoice(backend=pinned) if pinned is not None else None
    getter = "get_grouped_gemm_backend" if grouped else "get_gemm_backend"
    monkeypatch.setattr(GlobalBackendManager, getter, classmethod(lambda cls, precision: choice))
    assert mx_weight_layout_recipe(grouped, shape) == (
        ops_utils.MX_W if want else ScalingRecipe(use_2d_block=True)
    )
    assert ops_utils.mx_gemm_layout(torch.empty(shape), grouped) is want  # the op layer agrees


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
