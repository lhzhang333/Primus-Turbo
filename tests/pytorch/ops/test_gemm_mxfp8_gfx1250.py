###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The gfx1250 MXFP8 dense GEMM (``primus_turbo.flydsl.gemm.gemm_mxfp8_kernel_gfx1250``) against the recipe
reference: bitwise on integer-valued operands for every config and for M / N / K tails (output pre-filled with
NaN, so a missed or stray store shows), SNR on random data, and run-to-run determinism."""

import pytest
import torch

from primus_turbo.pytorch.core.utils import is_gfx1250
from tests.pytorch.ref.mxfp8_ref import (
    gemm_ref,
    int_valued_operands,
    pack_scales,
    quant_rows,
    snr_db,
)

pytestmark = pytest.mark.skipif(not is_gfx1250(), reason="gfx1250 kernel")

if is_gfx1250():
    from primus_turbo.flydsl.gemm.gemm_mxfp8_gfx1250_configs import MXGemmConfig as C
    from primus_turbo.flydsl.gemm.gemm_mxfp8_kernel_gfx1250 import gemm_mxfp8_gfx1250

    CONFIGS = [
        C(256, 256, 2, 2, 1, 1),
        C(256, 256, 2, 2, 1, 4),
        C(256, 256, 2, 2, 2, 2),
        C(256, 256, 4, 1, 1, 1),
        C(256, 256, 3, 1, 1, 2),
        C(128, 256, 4, 1, 1, 4),
        C(256, 128, 4, 1, 4, 1),
        C(128, 128, 4, 2, 1, 1),
        C(256, 256, 2, 2, 1, 4, True),
    ]
else:
    CONFIGS = [None]


def _run(a, sa, b, sb, cfg, out=None):
    if cfg.scale_4d:
        sa, sb = pack_scales(sa), pack_scales(sb)
    return gemm_mxfp8_gfx1250(a, sa, b, sb, out=out, config=cfg)


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: "-".join(map(str, c)) if c else "none")
@pytest.mark.parametrize(
    "M, N, K",
    [
        (512, 1024, 1024),
        (300, 264, 576),  # every tail: M, N past the tile, K = 4.5 WMMA steps
        (257, 136, 1008),  # K not a multiple of 32: the last scale block is half data
        (1024, 512, 7168),
    ],
)
def test_gemm_mxfp8_int_exact(cfg, M, N, K):
    a, sa, b, sb = int_valued_operands(M, N, K, seed=M + N + K)
    out = torch.full((M, N), float("nan"), dtype=torch.bfloat16, device="cuda")
    _run(a, sa, b, sb, cfg, out)
    torch.testing.assert_close(out, gemm_ref(a, sa, b, sb).to(torch.bfloat16), rtol=0, atol=0)


def test_gemm_mxfp8_random_snr():
    M, N, K = 1024, 2048, 4096
    g = torch.Generator(device="cuda").manual_seed(5)
    a, sa = quant_rows(torch.randn(M, K, device="cuda", generator=g) * torch.logspace(-2, 2, K, device="cuda"))
    b, sb = quant_rows(torch.randn(N, K, device="cuda", generator=g))
    ref = gemm_ref(a, sa, b, sb)
    ref_bf16 = ref.to(torch.bfloat16)
    floor = snr_db(ref_bf16, ref)  # the bf16 output rounding alone: ~55.6 dB
    for cfg in CONFIGS[:2]:
        out = _run(a, sa, b, sb, cfg)
        # the WMMA accumulation error is invisible next to the bf16 output rounding
        assert snr_db(out, ref) > floor - 0.05


def test_gemm_mxfp8_deterministic():
    M, N, K = 1000, 2048, 7168
    g = torch.Generator(device="cuda").manual_seed(7)
    a, sa = quant_rows(torch.randn(M, K, device="cuda", generator=g))
    b, sb = quant_rows(torch.randn(N, K, device="cuda", generator=g))
    for cfg in CONFIGS[:2]:
        first = _run(a, sa, b, sb, cfg).clone()
        out = torch.empty_like(first)
        same = 0
        for _ in range(1000):
            _run(a, sa, b, sb, cfg, out)
            same += int(torch.equal(out, first))
        assert same == 1000


def test_gemm_mxfp8_rejects():
    a, sa, b, sb = int_valued_operands(256, 260, 256)
    with pytest.raises(ValueError, match="multiple of 8"):
        gemm_mxfp8_gfx1250(a, sa, b, sb)
    a, sa, b, sb = int_valued_operands(256, 256, 256)
    with pytest.raises(ValueError, match="E8M0"):
        gemm_mxfp8_gfx1250(a, sa.float(), b, sb)
