###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""gemm_fp8 (MX_BLOCKWISE) on gfx1250: FP8GemmMXFunction on the FlyDSL MX GEMM, in strict mode (a fallback raises).
fwd / dgrad / wgrad against BF16 and against the fp32 emulation of the recipe (``tests/pytorch/ref/mxfp8_ref.py``:
the same quantization, so only the GEMM's accumulation order and the bf16 output rounding differ)."""

import pytest
import torch

import primus_turbo.common.envs as envs
from primus_turbo.pytorch.core.backend import GlobalBackendManager
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
from primus_turbo.pytorch.ops import gemm_fp8
from tests.pytorch.ref.mxfp8_ref import dequant, quant_2d, quant_dual, snr_db

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not is_gfx1250(), reason="gfx1250 MX GEMM")

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


def test_cached_weight():
    """A weight quantized ahead (``create_quantized_weight``, Primus' first-micro-batch cache) gives the same
    out and dx, bitwise."""
    x, w, dy = _data(512, 1536, 1024)
    q, q_t = create_quantized_weight(w, float8_e4m3, MX, need_weight_transpose_cache=True)
    res = []
    for b in (w, QuantizedTensorPair(q, q_t)):
        xg = x.clone().requires_grad_()
        out = gemm_fp8(xg, b, trans_b=True, out_dtype=torch.bfloat16, config=MX)
        out.backward(dy)
        res.append((out, xg.grad))
    assert all(torch.equal(a, b) for a, b in zip(*res))


def test_fallback_refused(strict):
    """N % 8 != 0 is not the FlyDSL kernel's: strict mode refuses the Triton fallback; without it, Triton's result."""
    x, w, _ = _data(256, 100, 384)
    with pytest.raises(RuntimeError, match="would fall back to TRITON"):
        gemm_fp8(x, w, trans_b=True, out_dtype=torch.bfloat16, config=MX)
    strict(0)
    out = gemm_fp8(x, w, trans_b=True, out_dtype=torch.bfloat16, config=MX)
    assert snr_db(out, x.float() @ w.float().t()) >= 25
