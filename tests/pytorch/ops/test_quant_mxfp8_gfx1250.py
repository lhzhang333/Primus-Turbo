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
