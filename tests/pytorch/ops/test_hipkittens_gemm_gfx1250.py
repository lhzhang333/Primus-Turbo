###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""HipKittens BF16 TN and NT GEMMs on gfx1250 (primus_turbo.hipkittens.gemm_gfx1250).

Integer operands in [-3, 3] keep every FP32 partial sum exact, so any correct kernel returns the same
BF16 result as ``torch.mm``: the numeric cases compare bit for bit. Random BF16 operands are compared
against an FP32 reference within BF16 output rounding.
"""

import pytest
import torch

from primus_turbo.pytorch.core.utils import is_gfx1250

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_gfx1250()), reason="HipKittens gfx1250 GEMM needs gfx1250"
)


def _gemm():
    from primus_turbo.hipkittens import gemm_gfx1250

    if not gemm_gfx1250.hk_gemm_available():
        pytest.skip("HipKittens gfx1250 headers or hipcc unavailable")
    return gemm_gfx1250


def _ints(rows, cols, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randint(-3, 4, (rows, cols), device="cuda", generator=g).to(torch.bfloat16)


# (M, N, K): whole-cluster blocks (the 224 and the 256 tile families), a ragged M edge (the
# DeepSeek-V3 vocabulary 129408 = 126 clusters of 1024 + 384), a ragged N edge, short and long K.
@pytest.mark.parametrize(
    "M,N,K",
    [
        (8192, 8192, 1024),
        (7168, 8192, 2048),
        (16384, 8192, 512),
        (129408, 1024, 256),
        (4096, 8704, 128),
        (24576, 8192, 1536),
        (7168, 8192, 16384),
    ],
)
def test_tn_matches_torch_bitwise(M, N, K):
    hk = _gemm()
    a, b = _ints(M, K, 1), _ints(N, K, 2)
    assert hk.hk_gemm_tn_supported(a, b)
    out = hk.hk_gemm_tn(a, b)
    torch.testing.assert_close(out, torch.mm(b, a.t()), rtol=0, atol=0)


@pytest.mark.parametrize("M,N,K", [(7168, 8192, 4096), (16384, 8192, 7168)])
def test_tn_random_close_to_fp32(M, N, K):
    hk = _gemm()
    g = torch.Generator(device="cuda").manual_seed(3)
    a = torch.randn(M, K, device="cuda", generator=g).bfloat16()
    b = torch.randn(N, K, device="cuda", generator=g).bfloat16()
    out = hk.hk_gemm_tn(a, b).float()
    ref = b.float() @ a.float().t()
    # BF16 output rounding: half an ulp of the largest magnitude, plus FP32 summation-order noise
    assert ((out - ref).abs().max() / ref.abs().max()).item() < 8e-3


def test_out_argument_and_layout():
    hk = _gemm()
    a, b = _ints(4096, 512, 4), _ints(8192, 512, 5)
    out = torch.empty(8192, 4096, dtype=torch.bfloat16, device="cuda")
    assert hk.hk_gemm_tn(a, b, out=out).data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, torch.mm(b, a.t()), rtol=0, atol=0)


# TN split-K: blocks whose tiles cover at most half the CUs and a long K (DeepSeek-V3's kv_up dgrad)
@pytest.mark.parametrize("M,N,K", [(512, 8192, 32768), (1024, 4096, 24576), (512, 8192 + 384, 16384)])
def test_tn_splitk_matches_torch_bitwise(M, N, K):
    hk = _gemm()
    plan = hk.hk_gemm_tn_plan(M, N, K)
    assert plan is not None and plan[3] > 1
    a, b = _ints(M, K, 21), _ints(N, K, 22)
    assert hk.hk_gemm_tn_supported(a, b)
    torch.testing.assert_close(hk.hk_gemm_tn(a, b), torch.mm(b, a.t()), rtol=0, atol=0)


@pytest.mark.parametrize(
    "M,N,K,why",
    [
        (512, 8192, 2048, "fewer tiles than CUs and K too short to split"),
        (1536, 8192, 24576, "192 tiles: more than half the CUs, no split-K"),
        (7168, 8192, 576, "K not a multiple of 128"),
        (1024, 1024, 128, "fewer tiles than CUs"),
    ],
)
def test_refuses(M, N, K, why):
    hk = _gemm()
    a, b = _ints(M, K, 6), _ints(N, K, 7)
    assert not hk.hk_gemm_tn_supported(a, b), why
    with pytest.raises(ValueError):
        hk.hk_gemm_tn(a, b)


def test_refuses_layout_and_dtype():
    hk = _gemm()
    a, b = _ints(8192, 1024, 8), _ints(8192, 1024, 9)
    assert hk.hk_gemm_tn_supported(a, b)
    assert not hk.hk_gemm_tn_supported(a.t(), b)  # not K-contiguous
    assert not hk.hk_gemm_tn_supported(a.float(), b.float())
    # the same values 16 bytes into an allocation: contiguous but not 256-byte aligned
    shifted = torch.empty(a.numel() + 64, dtype=a.dtype, device=a.device)[8 : 8 + a.numel()].view(a.shape)
    shifted.copy_(a)
    assert not hk.hk_gemm_tn_supported(shifted, b)


# NT (a [K, M], b [K, N] -> b.T @ a): whole clusters, the LM head's weight-gradient shape (a ragged N
# edge: 129408 = 126 clusters of 1024 + 384), the 2x4 cluster (M = 3 x 512), a ragged M edge.
@pytest.mark.parametrize(
    "M,N,K",
    [
        (8192, 8192, 1024),
        (7168, 129408, 512),
        (16384, 7168, 2048),
        (1536, 32768, 1024),
        (7424, 8192, 256),
    ],
)
def test_nt_matches_torch_bitwise(M, N, K):
    hk = _gemm()
    a, b = _ints(K, M, 11), _ints(K, N, 12)
    assert hk.hk_gemm_nt_supported(a, b)
    out = hk.hk_gemm_nt(a, b)
    torch.testing.assert_close(out, torch.mm(b.t(), a), rtol=0, atol=0)


def test_nt_random_close_to_fp32():
    hk = _gemm()
    g = torch.Generator(device="cuda").manual_seed(13)
    a = torch.randn(8192, 7168, device="cuda", generator=g).bfloat16()
    b = torch.randn(8192, 4096, device="cuda", generator=g).bfloat16()
    out = hk.hk_gemm_nt(a, b).float()
    ref = b.float().t() @ a.float()
    assert ((out - ref).abs().max() / ref.abs().max()).item() < 8e-3


def test_nt_refuses():
    hk = _gemm()
    a, b = _ints(8192, 4096, 14), _ints(8192, 4096, 15)  # 16 x 16 tiles: one full wave
    assert hk.hk_gemm_nt_supported(a, b)
    assert not hk.hk_gemm_nt_supported(a[:576], b[:576])  # K not a multiple of 128
    assert not hk.hk_gemm_nt_supported(a.t(), b)  # not contiguous
    assert not hk.hk_gemm_nt_supported(_ints(8192, 512, 16), _ints(8192, 1024, 17))  # fewer tiles than CUs
    with pytest.raises(ValueError):
        hk.hk_gemm_nt(a[:576], b[:576])


# NT with accumulation (out += b.T @ a, a weight gradient added straight into main_grad): the sum is taken in
# FP32 and rounded once, so on integer data it equals the exact sum rounded to BF16, as a GEMM with beta = 1
# returns it. Ragged M and N edges (and the corner they share) are added once.
@pytest.mark.parametrize(
    "M,N,K",
    [
        (8192, 8192, 1024),
        (7168, 129408, 512),
        (16384, 7168, 2048),
        (7424, 8192, 256),
        (7168 + 64, 4096 + 96, 1024),
    ],
)
def test_nt_accumulate_rounds_once(M, N, K):
    hk = _gemm()
    a, b = _ints(K, M, 31), _ints(K, N, 32)
    g = torch.Generator(device="cuda").manual_seed(33)
    c0 = torch.randint(-64, 65, (N, M), device="cuda", generator=g).to(torch.bfloat16)
    assert hk.hk_gemm_nt_supported(a, b)
    out = c0.clone()
    assert hk.hk_gemm_nt(a, b, out=out, accumulate=True).data_ptr() == out.data_ptr()
    exact = (c0.float() + b.float().t() @ a.float()).bfloat16()
    torch.testing.assert_close(out, exact, rtol=0, atol=0)
    torch.testing.assert_close(out, torch.addmm(c0, b.t(), a), rtol=0, atol=0)


def test_nt_accumulate_random_close_to_fp32():
    hk = _gemm()
    g = torch.Generator(device="cuda").manual_seed(34)
    a = torch.randn(8192, 7168, device="cuda", generator=g).bfloat16()
    b = torch.randn(8192, 4096, device="cuda", generator=g).bfloat16()
    c0 = (torch.randn(4096, 7168, device="cuda", generator=g) * 30).bfloat16()
    out = hk.hk_gemm_nt(a, b, out=c0.clone(), accumulate=True).float()
    ref = c0.float() + b.float().t() @ a.float()
    assert ((out - ref).abs().max() / ref.abs().max()).item() < 8e-3


def test_nt_accumulate_out_rules():
    hk = _gemm()
    a, b = _ints(1024, 4096, 35), _ints(1024, 4096, 36)
    with pytest.raises(ValueError):
        hk.hk_gemm_nt(a, b, accumulate=True)  # nothing to accumulate into
    # a parameter's slice of a gradient buffer: 16-byte (not 256-byte) aligned
    buf = torch.zeros(4096 * 4096 + 64, dtype=torch.bfloat16, device="cuda")
    out = buf[8 : 8 + 4096 * 4096].view(4096, 4096)
    assert hk.hk_gemm_nt_out_ok(out, a, b, accumulate=True)
    assert not hk.hk_gemm_nt_out_ok(out, a, b)
    with pytest.raises(ValueError):
        hk.hk_gemm_nt(a, b, out=out)
    hk.hk_gemm_nt(a, b, out=out, accumulate=True)
    hk.hk_gemm_nt(a, b, out=out, accumulate=True)
    torch.testing.assert_close(out, (2 * (b.float().t() @ a.float())).bfloat16(), rtol=0, atol=0)
    assert buf[:8].count_nonzero() == 0 and buf[8 + 4096 * 4096 :].count_nonzero() == 0
