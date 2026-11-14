###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MXFP8 strict mode (PRIMUS_TURBO_MX_STRICT): an MX op raises instead of taking a silent fallback."""

import pytest
import torch

import primus_turbo.common.envs as envs
from primus_turbo.pytorch.core.backend import (
    AutoKernelDispatcher,
    BackendChoice,
    BackendEntry,
    BackendType,
    GlobalBackendManager,
    KernelBackend,
)
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScaleDtype,
    ScalingGranularity,
    ScalingRecipe,
    check_mxfp8_support,
    float8_e4m3,
    mx_strict,
)

MX = Float8QuantConfig(
    format=Format.E4M3,
    granularity=ScalingGranularity.MX_BLOCKWISE,
    scale_dtype=ScaleDtype.E8M0,
    block_size=32,
)


@pytest.fixture
def strict(monkeypatch):
    def set_level(level: int):
        monkeypatch.setenv("PRIMUS_TURBO_MX_STRICT", str(level))
        envs.reset_cache("PRIMUS_TURBO_MX_STRICT")

    GlobalBackendManager.reset()
    yield set_level
    monkeypatch.delenv("PRIMUS_TURBO_MX_STRICT", raising=False)
    envs.reset_cache("PRIMUS_TURBO_MX_STRICT")
    GlobalBackendManager.reset()


def test_levels(strict):
    strict(0)
    mx_strict("off")
    mx_strict(lambda: pytest.fail("the message is built only to raise"), level=1)
    strict(1)
    with pytest.raises(RuntimeError, match="PRIMUS_TURBO_MX_STRICT=1: a fallback"):
        mx_strict("a fallback")
    with pytest.raises(RuntimeError, match="PRIMUS_TURBO_MX_STRICT=1: built"):
        mx_strict(lambda: "built")
    mx_strict("the HIP quantization", level=2)  # level 2 only
    strict(2)
    with pytest.raises(RuntimeError, match="PRIMUS_TURBO_MX_STRICT=2"):
        mx_strict("the HIP quantization", level=2)


@pytest.mark.parametrize("value", ["true", "on", "3", "-1", "1.0"])
def test_bad_value_raises(strict, value):
    """A safety switch fails loudly: anything but 0, 1, 2 (or unset / blank) is an error, not "off"."""
    strict(value)
    with pytest.raises(ValueError, match="PRIMUS_TURBO_MX_STRICT"):
        mx_strict("a fallback")


class _Never(KernelBackend):
    @staticmethod
    def can_handle(**kwargs) -> bool:
        return False

    @staticmethod
    def execute(**kwargs):
        raise AssertionError("not eligible")


class _Always(KernelBackend):
    @staticmethod
    def can_handle(**kwargs) -> bool:
        return True

    @staticmethod
    def execute(**kwargs):
        return "fallback"


class _Dispatcher(AutoKernelDispatcher):
    _backends = {
        BackendType.FLYDSL: BackendEntry(_Never, autotune=False),
        BackendType.TRITON: BackendEntry(_Always, autotune=False),
    }

    @classmethod
    def make_key(cls, **kwargs):
        return 0


@pytest.mark.parametrize(
    "granularity, a_dtype, refused",
    [
        (ScalingGranularity.MX_BLOCKWISE, float8_e4m3, True),  # MXFP8
        (ScalingGranularity.MX_BLOCKWISE, torch.uint8, False),  # MXFP4 (packed e2m1)
        (ScalingGranularity.TENSORWISE, float8_e4m3, False),
    ],
)
def test_dispatcher_fallback(strict, granularity, a_dtype, refused):
    """Only an MXFP8 call is refused its fallback, through dispatch and resolve alike."""
    flydsl = BackendChoice(backend=BackendType.FLYDSL)
    kw = dict(granularity=granularity, a=torch.empty(1, dtype=a_dtype))
    strict(0)
    assert _Dispatcher.dispatch(flydsl, None, **kw) == "fallback"
    strict(1)
    if refused:
        with pytest.raises(RuntimeError, match="default backend FLYDSL .* would fall back to TRITON"):
            _Dispatcher.dispatch(flydsl, None, **kw)
        with pytest.raises(RuntimeError, match="would fall back to TRITON"):
            _Dispatcher.resolve(BackendType.FLYDSL, None, **kw)
    else:
        assert _Dispatcher.dispatch(flydsl, None, **kw) == "fallback"
        assert _Dispatcher.resolve(BackendType.FLYDSL, None, **kw) == BackendType.TRITON


@pytest.mark.skipif(not torch.cuda.is_available() or not check_mxfp8_support()[0], reason="needs MXFP8")
def test_requantize_raises(strict):
    """A quantized operand without its transposed copy is re-quantized from its dequantized values: refused."""
    from primus_turbo.pytorch.core.quantized_tensor import QuantizedTensor
    from primus_turbo.pytorch.ops import gemm_fp8

    a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    qa = QuantizedTensor.quantize(
        a,
        float8_e4m3,
        ScalingGranularity.MX_BLOCKWISE,
        axis=-1,
        block_size=32,
        scaling_recipe=ScalingRecipe(),
    )
    strict(1)
    with pytest.raises(RuntimeError, match="no a_t, re-quantizing a"):
        gemm_fp8(qa, b, trans_b=True, out_dtype=torch.bfloat16, config=MX)
