###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The gfx1250 MXFP8 GEMMs (``primus_turbo.flydsl.gemm.gemm_mxfp8_kernel_gfx1250``) against the recipe reference:
dense, grouped-M and variable-K, bitwise on integer-valued operands (exact in fp32) for every config, M / N / K tails
and ragged / empty experts (outputs pre-filled with NaN, so a missed or stray store shows); fp32 out and beta = 1
(``out = round(out + fp32 result)``, with and without the early old-C fetch); SNR on random data; determinism; and
operands that end exactly on a 2 MiB allocation boundary (K tails must not read past them)."""

import pytest
import torch

from primus_turbo.pytorch.core.utils import is_gfx1250
from tests.pytorch.ref.mxfp8_ref import (
    PAD_SCALE,
    gemm_ref,
    int_valued_operands,
    pack_scales,
    quant_rows,
    snr_db,
)

pytestmark = pytest.mark.skipif(not is_gfx1250(), reason="gfx1250 kernel")

if is_gfx1250():
    import primus_turbo.flydsl.gemm.gemm_mxfp8_kernel_gfx1250 as K
    from primus_turbo.flydsl.gemm.gemm_mxfp8_gfx1250_configs import MXGemmConfig as C
    from primus_turbo.flydsl.gemm.gemm_mxfp8_kernel_gfx1250 import (
        gemm_mxfp8_gfx1250,
        grouped_gemm_mxfp8_gfx1250,
        grouped_gemm_mxfp8_variable_k_gfx1250,
    )

    # 7th / 8th fields: scales passed packed (the fast layout) or raw; B passed 16x16-preshuffled
    CONFIGS = [
        C(256, 256, 2, 2, 1, 1, True),
        C(256, 256, 2, 2, 1, 4, True),
        C(256, 256, 2, 2, 2, 2, True),
        C(256, 256, 4, 1, 1, 1, True),
        C(256, 256, 3, 1, 1, 2, True),
        C(128, 256, 3, 2, 1, 4, True),
        C(256, 128, 4, 1, 4, 1, True),
        C(128, 128, 4, 2, 1, 1, True),
        C(256, 256, 2, 2, 1, 4),
        C(128, 128, 4, 1, 1, 1),
        C(256, 256, 4, 1, 2, 2, True, True),
        C(128, 256, 3, 2, 1, 4, True, True),
    ]
else:
    CONFIGS = [None]

DEV = "cuda"


def _slack(t, extra=4096):
    """``t`` in a storage with ``extra`` bytes after it: a partial last K step reads the last row up to a step past its
    data (the MX quantizers allocate that slack); without it the kernels use kpair 1 or refuse K % 128 != 0."""
    n = t.numel() * t.element_size()
    v = torch.empty(n + extra, dtype=torch.uint8, device=t.device)[:n].view(t.dtype).view(t.shape)
    return v.copy_(t)


def _shuf(b):
    """The 16x16 preshuffle of the last two dims (``[.., N/16][K/16][16][16]``)."""
    *lead, N, Kd = b.shape
    return b.reshape(*lead, N // 16, 16, Kd // 16, 16).transpose(-3, -2).contiguous().reshape(b.shape)


def _exact(t):
    """``t`` in a fresh allocation of exactly its size (a multiple of 2 MiB >= 10 MiB: a segment of its own, so the
    tensor ends where the mapping ends)."""
    torch.cuda.empty_cache()
    x = torch.empty(t.numel() * t.element_size(), dtype=torch.uint8, device=t.device).view(t.dtype).view(t.shape)
    return x.copy_(t)


def _run(a, sa, b, sb, cfg, out=None):
    sa, sb = (pack_scales(sa), pack_scales(sb)) if cfg.scale_4d else (_slack(sa), _slack(sb))
    if cfg.b_preshuffle:
        b = _shuf(b)
    return gemm_mxfp8_gfx1250(_slack(a), sa, _slack(b), sb, out=out, config=cfg, b_preshuffled=cfg.b_preshuffle)


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: "-".join(map(str, c)) if c else "none")
@pytest.mark.parametrize(
    "M, N, K",
    [
        (512, 1024, 1024),
        (300, 264, 576),  # every tail: M, N past the tile, K = 4.5 WMMA steps
        (257, 136, 1008),  # K not a multiple of 32: the last scale block is half data
        (1024, 512, 7168),
        (384, 512, 640),  # a half last K step at kpair 2
    ],
)
def test_gemm_mxfp8_int_exact(cfg, M, N, K):
    if cfg.b_preshuffle and N % 16:
        pytest.skip("a preshuffled B needs N % 16 == 0")
    a, sa, b, sb = int_valued_operands(M, N, K, seed=M + N + K)
    out = torch.full((M, N), float("nan"), dtype=torch.bfloat16, device=DEV)
    _run(a, sa, b, sb, cfg, out)
    torch.testing.assert_close(out, gemm_ref(a, sa, b, sb).to(torch.bfloat16), rtol=0, atol=0)


@pytest.mark.parametrize("cfg", [C(256, 256, 2, 2, 1, 1), C(256, 256, 2, 2, 2, 2), C(128, 256, 3, 2, 1, 1),
                                 C(256, 256, 4, 1, 1, 4)] if is_gfx1250() else [None], ids=str)
@pytest.mark.parametrize("M, N, K", [(512, 1024, 1152), (300, 512, 640), (256, 768, 2048)])
@pytest.mark.parametrize("c_early", [True, False])
def test_gemm_mxfp8_fp32_out_and_accumulate(cfg, M, N, K, c_early, monkeypatch):
    """fp32 out == the exact result; beta = 1 into bf16 / fp32 == round(fp32 result + C), rounded once."""
    monkeypatch.setattr(K, "C_EARLY_ON", c_early)
    a, sa, b, sb = int_valued_operands(M, N, K, seed=M + K)
    ref = gemm_ref(a, sa, b, sb).float()
    sa, sb, a, b = pack_scales(sa), pack_scales(sb), _slack(a), _slack(_shuf(b))
    out = torch.full((M, N), float("nan"), device=DEV)
    gemm_mxfp8_gfx1250(a, sa, b, sb, out_dtype=torch.float32, out=out, config=cfg, b_preshuffled=True)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    for dt in (torch.bfloat16, torch.float32):
        c0 = (torch.randn(M, N, device=DEV) * 64).to(dt)
        out = c0.clone()
        gemm_mxfp8_gfx1250(a, sa, b, sb, out_dtype=dt, out=out, config=cfg, b_preshuffled=True, accumulate=True)
        torch.testing.assert_close(out, (ref + c0.float()).to(dt), rtol=0, atol=0)


def test_gemm_mxfp8_random_snr():
    M, N, K = 1024, 2048, 4096
    g = torch.Generator(device=DEV).manual_seed(5)
    a, sa = quant_rows(torch.randn(M, K, device=DEV, generator=g) * torch.logspace(-2, 2, K, device=DEV))
    b, sb = quant_rows(torch.randn(N, K, device=DEV, generator=g))
    ref = gemm_ref(a, sa, b, sb)
    floor = snr_db(ref.to(torch.bfloat16), ref)  # the bf16 output rounding alone: ~55.6 dB
    for cfg in (CONFIGS[1], CONFIGS[8], CONFIGS[10]):  # packed, raw, preshuffled B
        # the WMMA accumulation error is invisible next to the bf16 output rounding
        assert snr_db(_run(a, sa, b, sb, cfg), ref) > floor - 0.05
    out = gemm_mxfp8_gfx1250(a, pack_scales(sa), b, pack_scales(sb), out_dtype=torch.float32)
    assert snr_db(out, ref) > 60


def test_gemm_mxfp8_deterministic():
    M, N, K = 1000, 2048, 7168
    g = torch.Generator(device=DEV).manual_seed(7)
    a, sa = quant_rows(torch.randn(M, K, device=DEV, generator=g))
    b, sb = quant_rows(torch.randn(N, K, device=DEV, generator=g))
    for cfg in (CONFIGS[1], CONFIGS[3], CONFIGS[10]):
        first = _run(a, sa, b, sb, cfg).clone()
        out = torch.empty_like(first)
        same = 0
        for _ in range(1000):
            _run(a, sa, b, sb, cfg, out)
            same += int(torch.equal(out, first))
        assert same == 1000


def _grouped_operands(lens, N, Kd, seed):
    """A row copy padded to 32 rows per expert (zero rows, scale 127), b [G, N, K], int-valued."""
    pads = [-(-n // 32) * 32 for n in lens]
    a, sa, _, _ = int_valued_operands(max(sum(pads), 1), 16, Kd, seed=seed)
    a, sa = a[: sum(pads)].clone(), sa[: sum(pads)].clone()
    offs = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int64, device=DEV)
    offs_p = torch.tensor([0] + torch.tensor(pads).cumsum(0).tolist(), dtype=torch.int64, device=DEV)
    for g, n in enumerate(lens):
        a[int(offs_p[g]) + n: int(offs_p[g + 1])] = 0
        sa[int(offs_p[g]) + n: int(offs_p[g + 1])] = 127
    bb = [int_valued_operands(N, 16, Kd, seed=seed + 1 + g)[:2] for g in range(len(lens))]
    b, sb = torch.stack([x[0] for x in bb]), torch.stack([x[1] for x in bb])
    ref = torch.zeros(sum(lens), N, dtype=torch.float64, device=DEV)
    for g, n in enumerate(lens):
        r0, rp = int(offs[g]), int(offs_p[g])
        ref[r0: r0 + n] = gemm_ref(a[rp: rp + n], sa[rp: rp + n], b[g], sb[g])
    return a, sa, b, sb, offs, offs_p, ref


@pytest.mark.parametrize(
    "lens, N, Kd, cfg",
    [
        ([100, 0, 300, 37], 512, 1024, C(256, 256, 2, 2, 1, 4)),
        ([1, 513, 0, 0, 64, 255], 1024, 2048, C(256, 256, 4, 1, 1, 4)),
        ([256] * 8, 768, 576, C(128, 256, 3, 2, 1, 2)),
        ([77, 300, 0, 129], 512, 640, C(256, 256, 2, 2, 1, 2)),
        ([(37 * i) % 300 for i in range(40)], 256, 1024, C(256, 256, 2, 2, 1, 1)),  # > 32 experts
    ]
    if is_gfx1250()
    else [],
)
def test_grouped_gemm_mxfp8(lens, N, Kd, cfg, monkeypatch):
    """Routed experts fwd / dgrad: skewed, empty and one-row experts, K tails; packed scales + preshuffled B and the
    raw / row-major fallback; fp32 out; the XCD-grouped m tile order."""
    a, sa, b, sb, offs, offs_p, ref = _grouped_operands(lens, N, Kd, seed=len(lens) + Kd)
    m = sum(lens)
    sb4 = torch.stack([pack_scales(sb[g]) for g in range(len(lens))])
    for odt in (torch.bfloat16, torch.float32):
        out = torch.full((m, N), float("nan"), device=DEV, dtype=odt)
        grouped_gemm_mxfp8_gfx1250(_slack(a), pack_scales(sa), _slack(_shuf(b)), sb4, offs, offs_p, m, out_dtype=odt,
                                   out=out, config=cfg)
        torch.testing.assert_close(out, ref.to(odt), rtol=0, atol=0)
    out = torch.full((m, N), float("nan"), device=DEV, dtype=torch.bfloat16)
    grouped_gemm_mxfp8_gfx1250(_slack(a), _slack(sa), _slack(b), _slack(sb), offs, offs_p, m, out=out, config=cfg,
                               b_preshuffled=False)
    torch.testing.assert_close(out, ref.to(torch.bfloat16), rtol=0, atol=0)
    monkeypatch.setattr(K, "XCD_REMAP_ON", True)
    out = torch.full((m, N), float("nan"), device=DEV, dtype=torch.bfloat16)
    grouped_gemm_mxfp8_gfx1250(_slack(a), pack_scales(sa), _slack(_shuf(b)), sb4, offs, offs_p, m, out=out, config=cfg)
    torch.testing.assert_close(out, ref.to(torch.bfloat16), rtol=0, atol=0)


@pytest.mark.parametrize(
    "lens, M, N, cfg",
    [
        ([130, 0, 256, 1], 512, 768, C(256, 256, 2, 2, 2, 2)),
        ([0, 0], 256, 256, C(256, 256, 2, 2, 1, 1)),
        ([300, 77, 0, 1000, 1], 320, 512, C(256, 256, 2, 2, 1, 1)),
        ([2048, 128, 129, 640], 512, 768, C(128, 256, 3, 2, 1, 1)),
        ([5, 384, 1], 256, 256, C(128, 128, 4, 2, 1, 1)),
        ([2048, 2048], 256, 512, C(256, 256, 4, 1, 1, 2)),
    ]
    if is_gfx1250()
    else [],
)
def test_grouped_gemm_mxfp8_variable_k(lens, M, N, cfg, monkeypatch):
    """Routed experts wgrad over each expert's 128-padded tokens: empty experts (0, or out untouched with
    ``accumulate``), odd counts of 128 (a half last K step), M tails; bf16 / fp32 out, beta = 1 with and without the
    early old-C fetch."""
    pads = [-(-n // 128) * 128 for n in lens]
    T = max(sum(pads), 128)
    a, sa, b, sb = int_valued_operands(M, N, T, seed=T + M)
    offs = torch.tensor([0] + torch.tensor(pads).cumsum(0).tolist(), dtype=torch.int64, device=DEV)
    for g, n in enumerate(lens):  # padding tokens: zero codes, PAD_SCALE for the blocks with no token
        t0, t1 = int(offs[g]) + n, int(offs[g + 1])
        a[:, t0:t1], b[:, t0:t1] = 0, 0
        sa[:, -(-t0 // 32): t1 // 32], sb[:, -(-t0 // 32): t1 // 32] = PAD_SCALE, PAD_SCALE
    o = [int(x) for x in offs]
    ref = torch.stack(
        [gemm_ref(a[:, o[g]: o[g + 1]], sa[:, o[g] // 32: o[g + 1] // 32], b[:, o[g]: o[g + 1]],
                  sb[:, o[g] // 32: o[g + 1] // 32]) for g in range(len(lens))]
    ).float()
    sa, sb, b = pack_scales(sa), pack_scales(sb), _slack(_shuf(b))
    for dt, acc, ce, pe in ((torch.bfloat16, False, True, False), (torch.float32, False, True, False),
                            (torch.float32, True, True, False), (torch.bfloat16, True, True, False),
                            (torch.bfloat16, True, False, False), (torch.bfloat16, True, True, True)):
        monkeypatch.setattr(K, "C_EARLY_ON", ce)
        monkeypatch.setattr(K, "PERSIST_ON", pe)
        c0 = (torch.randn(len(lens), M, N, device=DEV) * 64).to(dt)
        out = c0.clone() if acc else torch.full(c0.shape, float("nan"), device=DEV, dtype=dt)
        grouped_gemm_mxfp8_variable_k_gfx1250(a, sa, b, sb, offs, out_dtype=dt, out=out, accumulate=acc, config=cfg)
        torch.testing.assert_close(out, ((ref + c0.float()) if acc else ref).to(dt), rtol=0, atol=0)


def test_grouped_gemm_mxfp8_variable_k_persistent(monkeypatch):
    """Persistent variable-K (beta = 1 into bf16, early old-C fetch): 768 output tiles on <= 256 workgroups, so each
    runs several tiles whose last K steps end in either slot (also with staggered starts); == the
    one-tile-per-workgroup launch bitwise and the reference."""
    lens = [128 * (i % 5) + (0 if i % 7 == 3 else 37 * (i % 3)) for i in range(24)]
    M, N = 1024, 2048
    pads = [-(-n // 128) * 128 for n in lens]
    T = sum(pads)
    a, sa, b, sb = int_valued_operands(M, N, T, seed=11)
    offs = torch.tensor([0] + torch.tensor(pads).cumsum(0).tolist(), dtype=torch.int64, device=DEV)
    o = [int(x) for x in offs]
    for g, n in enumerate(lens):
        a[:, o[g] + n: o[g + 1]], b[:, o[g] + n: o[g + 1]] = 0, 0
        sa[:, -(-(o[g] + n) // 32): o[g + 1] // 32], sb[:, -(-(o[g] + n) // 32): o[g + 1] // 32] = PAD_SCALE, PAD_SCALE
    ref = torch.stack([gemm_ref(a[:, o[g]: o[g + 1]], sa[:, o[g] // 32: o[g + 1] // 32], b[:, o[g]: o[g + 1]],
                                sb[:, o[g] // 32: o[g + 1] // 32]) for g in range(len(lens))]).float()
    sa, sb, b = pack_scales(sa), pack_scales(sb), _slack(_shuf(b))
    c0 = (torch.randn(len(lens), M, N, device=DEV) * 64).to(torch.bfloat16)
    outs = []
    for pe, sg in ((True, 0), (False, 0), (True, 8)):
        monkeypatch.setattr(K, "PERSIST_ON", pe)
        monkeypatch.setattr(K, "STAGGER_ON", sg)
        out = c0.clone()
        grouped_gemm_mxfp8_variable_k_gfx1250(a, sa, b, sb, offs, out=out, accumulate=True,
                                              config=C(256, 256, 2, 2, 1, 1))
        outs.append(out)
    torch.testing.assert_close(outs[0], (ref + c0.float()).to(torch.bfloat16), rtol=0, atol=0)
    assert torch.equal(outs[0], outs[1]) and torch.equal(outs[0], outs[2])


def test_gemm_mxfp8_tail_at_allocation_end():
    """K tails with an operand that ends exactly where its 10 MiB allocation does (no slack): the half last K step at
    kpair 2 reads only its first half of a row-major A, so nothing past it; a preshuffled b or raw scales without
    their slack make the entry use kpair 1; K % 128 != 0 without slack is refused."""
    M, N, Kd = 16384, 256, 640
    a, sa, b, sb = int_valued_operands(M, N, Kd, seed=1)
    ref = gemm_ref(a, sa, b, sb).to(torch.bfloat16)
    a = _exact(a)
    assert a.numel() == 10 << 20 and K._slack(a) == 0
    for cfg in (C(256, 256, 2, 2, 1, 1), C(256, 256, 2, 2, 2, 2)):
        out = gemm_mxfp8_gfx1250(a, pack_scales(sa), _slack(_shuf(b)), pack_scales(sb), config=cfg, b_preshuffled=True)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
        out = gemm_mxfp8_gfx1250(a, sa, _shuf(b), sb, config=cfg, b_preshuffled=True)  # no slack: kpair 1
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
    # variable-K over experts with odd counts of 128 tokens: a [2048, T] and a preshuffled b [2048, T] of 10 MiB
    lens = [384, 128, 1408, 2048, 1152]
    Mv, Nv, T = 2048, 2048, sum(lens)
    a, sa, b, sb = int_valued_operands(Mv, Nv, T, seed=2)
    offs = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int64, device=DEV)
    o = [int(x) for x in offs]
    ref = torch.stack([gemm_ref(a[:, o[g]: o[g + 1]], sa[:, o[g] // 32: o[g + 1] // 32], b[:, o[g]: o[g + 1]],
                                sb[:, o[g] // 32: o[g + 1] // 32]) for g in range(len(lens))]).to(torch.bfloat16)
    sa, sb, a = pack_scales(sa), pack_scales(sb), _exact(a)
    for bs in (_slack(_shuf(b)), _exact(_shuf(b))):  # kpair 2 with half steps; b without slack: kpair 1
        out = grouped_gemm_mxfp8_variable_k_gfx1250(a, sa, bs, sb, offs, config=C(256, 256, 2, 2, 1, 1))
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
    a, sa, b, sb = int_valued_operands(256, 256, 576)
    with pytest.raises(ValueError, match="pad K"):
        gemm_mxfp8_gfx1250(a, pack_scales(sa), b, pack_scales(sb), config=C(256, 256, 2, 2, 1, 1))


def test_gemm_mxfp8_rejects():
    a, sa, b, sb = int_valued_operands(256, 260, 256)
    with pytest.raises(ValueError, match="multiple of 8"):
        gemm_mxfp8_gfx1250(a, sa, b, sb)
    a, sa, b, sb = int_valued_operands(256, 256, 256)
    with pytest.raises(ValueError, match="E8M0"):
        gemm_mxfp8_gfx1250(a, sa.float(), b, sb)
