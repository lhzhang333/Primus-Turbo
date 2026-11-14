###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The gfx1250 MXFP8 GEMMs (``primus_turbo.flydsl.gemm.gemm_mxfp8_kernel_gfx1250``) against the recipe reference,
bitwise on integer-valued operands (exact in fp32): dense, grouped-M and variable-K; each config feature once (tile
shapes, pipeline depths, multicast clusters, packed / raw scales, preshuffled / row-major B, M / N / K tails, fp32
out, beta = 1 with and without the early old-C fetch, persistent workgroups), the entry points' default configs
(``config=None``: the production paths), operands without storage slack, SNR on random data and determinism.
Outputs sit between NaN guard rows: a missed store leaves NaN, a stray one past ``out`` hits a guard."""

import math

import pytest
import torch

from primus_turbo.flydsl.gemm.gemm_mxfp8_gfx1250_configs import MXGemmConfig as C
from primus_turbo.pytorch.core.utils import is_gfx1250
from tests.pytorch.ref.mxfp8_ref import (
    PAD_SCALE,
    gemm_ref,
    int_valued_operands,
    pack_scales,
    preshuffle_b16,
    quant_rows,
    snr_db,
)

pytestmark = pytest.mark.skipif(not is_gfx1250(), reason="gfx1250 kernel")

if is_gfx1250():
    import primus_turbo.flydsl.gemm.gemm_mxfp8_kernel_gfx1250 as KM
    from primus_turbo.flydsl.gemm.gemm_mxfp8_kernel_gfx1250 import (
        gemm_mxfp8_gfx1250,
        grouped_gemm_mxfp8_gfx1250,
        grouped_gemm_mxfp8_variable_k_gfx1250,
    )

DEV = "cuda"
BF16, F32 = torch.bfloat16, torch.float32

# The production configs (gemm_mxfp8_gfx1250_configs) that tests below run by name
DEFAULT = C(256, 256, 2, 2, 2, 2)  # the dense default
KP1_C2X4 = C(256, 256, 4, 1, 2, 4)  # dense o_proj fwd; the routed experts' grouped-M (c2x4)
KP1_C4X4 = C(256, 256, 4, 1, 4, 4)  # dense o_proj dgrad
PERSIST_C4X1 = C(256, 256, 2, 2, 4, 1, persist=True)  # dense q_up / kv_up fwd


def _slack(t, extra=4096):
    """``t`` in a storage with ``extra`` bytes after it (the MX quantizers allocate that slack after their outputs)."""
    n = t.numel() * t.element_size()
    v = torch.empty(n + extra, dtype=torch.uint8, device=t.device)[:n].view(t.dtype).view(t.shape)
    return v.copy_(t)


def _exact(t):
    """``t`` in a fresh allocation of exactly its size. At 10 MiB or more that is a segment of its own, so the tensor
    likely ends where the mapping does (best effort: the address range after it may still be mapped)."""
    torch.cuda.empty_cache()
    x = (
        torch.empty(t.numel() * t.element_size(), dtype=torch.uint8, device=t.device)
        .view(t.dtype)
        .view(t.shape)
    )
    return x.copy_(t)


def _guarded(shape, dtype, init=None):
    """An output of ``shape`` between two NaN guard rows (of its last dim), NaN itself unless ``init`` is given:
    (out, check), where check() asserts the guards are untouched."""
    buf = torch.full((math.prod(shape[:-1]) + 2, shape[-1]), float("nan"), dtype=dtype, device=DEV)
    out = buf[1:-1].view(shape)
    if init is not None:
        out.copy_(init)

    def check():
        assert torch.isnan(buf[0]).all() and torch.isnan(buf[-1]).all(), "a store past out"

    return out, check


def _dense(M, N, K, packed, bsh, seed):
    """Integer-valued dense operands as the kernel takes them (with storage slack) and the exact result."""
    a, sa, b, sb = int_valued_operands(M, N, K, seed=seed)
    ref = gemm_ref(a, sa, b, sb).float()
    sa, sb = (pack_scales(sa), pack_scales(sb)) if packed else (_slack(sa), _slack(sb))
    return _slack(a), sa, _slack(preshuffle_b16(b) if bsh else b), sb, ref


def _eq(out, ref):
    torch.testing.assert_close(out, ref.to(out.dtype), rtol=0, atol=0)


# (config, packed scales, preshuffled B, (M, N, K)): every tile shape, pipeline depth and cluster of the tables once
DENSE_CASES = [
    (
        C(256, 256, 2, 2, 1, 1),
        False,
        False,
        (300, 264, 576),
    ),  # raw / row-major; M, N tails; a half last K step
    (C(128, 128, 4, 1, 1, 1), False, False, (257, 136, 1008)),  # K % 32 != 0: the last scale block half data
    (C(128, 256, 3, 2, 1, 4), True, True, (640, 1024, 2048)),
    (C(256, 128, 4, 1, 4, 1), True, True, (1024, 512, 7168)),
    (C(128, 128, 4, 2, 1, 1), True, True, (384, 512, 640)),  # a half last K step at kpair 2
    (C(256, 256, 3, 1, 1, 2), True, False, (512, 1024, 1152)),  # row-major B at kpair 1
    (KP1_C2X4, True, True, (1024, 1024, 2048)),
    (KP1_C4X4, True, True, (1024, 2048, 1088)),  # a partial K step at kpair 1
    (DEFAULT, True, True, (512, 1024, 1024)),
]


@pytest.mark.parametrize(
    "cfg, packed, bsh, shape", DENSE_CASES, ids=lambda x: str(tuple(x)) if isinstance(x, tuple) else str(x)
)
def test_gemm_mxfp8_int_exact(cfg, packed, bsh, shape):
    M, N, K = shape
    a, sa, b, sb, ref = _dense(M, N, K, packed, bsh, seed=M + N + K)
    out, check = _guarded((M, N), BF16)
    gemm_mxfp8_gfx1250(a, sa, b, sb, out=out, config=cfg, b_preshuffled=bsh)
    _eq(out, ref)
    check()


@pytest.mark.parametrize(
    "cfg, shape",
    [(DEFAULT, (300, 512, 640)), (C(256, 256, 4, 1, 1, 4), (512, 1024, 1152))],
    ids=["kpair2-early-C", "kpair1"],
)
def test_gemm_mxfp8_fp32_out_and_accumulate(cfg, shape):
    """fp32 out == the exact result; beta = 1 into bf16 (kpair 2: the old C fetched during the last K step; kpair 1:
    read at the stores) and into fp32 == round(fp32 result + C), rounded once."""
    M, N, K = shape
    a, sa, b, sb, ref = _dense(M, N, K, True, True, seed=M + K)
    kinds = ((F32, False), (BF16, True)) if cfg.kpair == 2 else ((BF16, True), (F32, True))
    for dt, acc in kinds:
        c0 = (torch.randn(M, N, device=DEV) * 64).to(dt)
        out, check = _guarded((M, N), dt, c0 if acc else None)
        gemm_mxfp8_gfx1250(
            a, sa, b, sb, out_dtype=dt, out=out, config=cfg, b_preshuffled=True, accumulate=acc
        )
        _eq(out, (ref + c0.float()) if acc else ref)
        check()


def test_gemm_mxfp8_persistent():
    """Persistent dense, M past 32 m tiles so both the head chunks and the short last chunk of the schedule run:
    bf16 out on c4x1 (q_up / kv_up fwd), and the default config of beta = 1 into bf16 (a kpair-1 pick there becomes
    the persistent c2x2 with c_pre: the old C's first rows fetched during the previous tile's epilogue)."""
    M, N, K = 9216, 512, 640
    a, sa, b, sb, ref = _dense(M, N, K, True, True, seed=3)
    out, check = _guarded((M, N), BF16)
    gemm_mxfp8_gfx1250(a, sa, b, sb, out=out, config=PERSIST_C4X1, b_preshuffled=True)
    _eq(out, ref)
    check()
    M, N, K = 8704, 1536, 4096
    a, sa, b, sb, ref = _dense(M, N, K, True, True, seed=4)
    c0 = (torch.randn(M, N, device=DEV) * 64).to(BF16)
    out, check = _guarded((M, N), BF16, c0)
    gemm_mxfp8_gfx1250(a, sa, b, sb, out=out, b_preshuffled=True, accumulate=True)
    _eq(out, ref + c0.float())
    check()


def test_gemm_mxfp8_random_snr():
    M, N, K = 1024, 2048, 4224
    g = torch.Generator(device=DEV).manual_seed(5)
    a, sa = quant_rows(torch.randn(M, K, device=DEV, generator=g) * torch.logspace(-2, 2, K, device=DEV))
    b, sb = quant_rows(torch.randn(N, K, device=DEV, generator=g))
    ref = gemm_ref(a, sa, b, sb)
    floor = snr_db(ref.to(BF16), ref)  # the bf16 output rounding alone: ~55.6 dB
    a, sa, b, sb = _slack(a), pack_scales(sa), _slack(preshuffle_b16(b)), pack_scales(sb)
    # the WMMA accumulation error is invisible next to the bf16 output rounding
    assert snr_db(gemm_mxfp8_gfx1250(a, sa, b, sb, config=KP1_C2X4, b_preshuffled=True), ref) > floor - 0.05
    out = gemm_mxfp8_gfx1250(a, sa, b, sb, out_dtype=F32, config=DEFAULT, b_preshuffled=True)
    assert snr_db(out, ref) > 60


def _grouped(lens, N, K, seed):
    """A row copy padded to 32 rows per expert (zero rows, scale 127), b [G, N, K], int-valued; the exact result."""
    pads = [-(-n // 32) * 32 for n in lens]
    a, sa, _, _ = int_valued_operands(max(sum(pads), 1), 16, K, seed=seed)
    a, sa = a[: sum(pads)].clone(), sa[: sum(pads)].clone()
    offs = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int64, device=DEV)
    offs_p = torch.tensor([0] + torch.tensor(pads).cumsum(0).tolist(), dtype=torch.int64, device=DEV)
    for g, n in enumerate(lens):
        a[int(offs_p[g]) + n : int(offs_p[g + 1])] = 0
        sa[int(offs_p[g]) + n : int(offs_p[g + 1])] = 127
    bb = [int_valued_operands(N, 16, K, seed=seed + 1 + g)[:2] for g in range(len(lens))]
    b, sb = torch.stack([x[0] for x in bb]), torch.stack([x[1] for x in bb])
    ref = torch.zeros(sum(lens), N, dtype=torch.float64, device=DEV)
    for g, n in enumerate(lens):
        r0, rp = int(offs[g]), int(offs_p[g])
        ref[r0 : r0 + n] = gemm_ref(a[rp : rp + n], sa[rp : rp + n], b[g], sb[g])
    return a, sa, b, sb, offs, offs_p, ref.float()


# m tiles of 256 rows per expert: 1 3 0 0 1 1 3 (9). In 2x4 clusters (two consecutive m tiles share B): pairs that
# straddle two experts load B per workgroup, (6, 7) shares expert 6's, and (8, 9) has a dead second tile.
RAGGED = [1, 513, 0, 0, 64, 255, 700]


def test_grouped_gemm_mxfp8():
    """Routed experts fwd / dgrad with ragged, one-row and empty experts: the default config (the production c2x4),
    the raw-scale / row-major-B fallback with fp32 out, and no expert with rows (out untouched)."""
    N, K = 1024, 2048
    a, sa, b, sb, offs, offs_p, ref = _grouped(RAGGED, N, K, seed=7)
    m = sum(RAGGED)
    sb4 = torch.stack([pack_scales(sb[g]) for g in range(len(RAGGED))])
    out, check = _guarded((m, N), BF16)
    grouped_gemm_mxfp8_gfx1250(
        _slack(a), pack_scales(sa), _slack(preshuffle_b16(b)), sb4, offs, offs_p, m, out=out
    )
    _eq(out, ref)
    check()
    out, check = _guarded((m, N), F32)
    grouped_gemm_mxfp8_gfx1250(
        _slack(a),
        _slack(sa),
        _slack(b),
        _slack(sb),
        offs,
        offs_p,
        m,
        F32,
        out,
        C(256, 256, 2, 2, 1, 2),
        False,
    )
    _eq(out, ref)
    check()
    zero = torch.zeros_like(offs)
    out, check = _guarded((64, N), BF16)
    grouped_gemm_mxfp8_gfx1250(
        _slack(a), pack_scales(sa), _slack(preshuffle_b16(b)), sb4, zero, zero, 64, out=out
    )
    assert torch.isnan(out).all()
    check()


def test_grouped_gemm_mxfp8_many_experts():
    """48 experts (two 32-lane lookup chunks), ~2 m tiles each, K below 2048: the default config is persistent
    kpair 2 (a half last K step)."""
    lens = [(53 * i) % 900 for i in range(48)]
    N, K = 1024, 640
    a, sa, b, sb, offs, offs_p, ref = _grouped(lens, N, K, seed=11)
    m = sum(lens)
    sb4 = torch.stack([pack_scales(sb[g]) for g in range(len(lens))])
    out, check = _guarded((m, N), BF16)
    grouped_gemm_mxfp8_gfx1250(
        _slack(a), pack_scales(sa), _slack(preshuffle_b16(b)), sb4, offs, offs_p, m, out=out
    )
    _eq(out, ref)
    check()


def _vk(lens, M, N, seed):
    """Variable-K operands over each expert's 128-padded tokens (padding: zero codes, PAD_SCALE for the 32-blocks
    with no token), packed scales, a preshuffled b; the exact per-expert results [G, M, N] and the offsets."""
    pads = [-(-n // 128) * 128 for n in lens]
    T = max(sum(pads), 128)
    a, sa, b, sb = int_valued_operands(M, N, T, seed=seed)
    offs = torch.tensor([0] + torch.tensor(pads).cumsum(0).tolist(), dtype=torch.int64, device=DEV)
    o = [int(x) for x in offs]
    for g, n in enumerate(lens):
        t0, t1 = o[g] + n, o[g + 1]
        a[:, t0:t1], b[:, t0:t1] = 0, 0
        sa[:, -(-t0 // 32) : t1 // 32], sb[:, -(-t0 // 32) : t1 // 32] = PAD_SCALE, PAD_SCALE
    ref = torch.stack(
        [
            gemm_ref(
                a[:, o[g] : o[g + 1]],
                sa[:, o[g] // 32 : o[g + 1] // 32],
                b[:, o[g] : o[g + 1]],
                sb[:, o[g] // 32 : o[g + 1] // 32],
            )
            for g in range(len(lens))
        ]
    ).float()
    return a, pack_scales(sa), b, pack_scales(sb), offs, ref


def test_grouped_gemm_mxfp8_variable_k():
    """Routed experts wgrad, 40 experts (offsets read per tile, not from a lane table): empty experts, odd counts of
    128 tokens (a half last K step), an M tail. The default config, as production runs it: beta = 1 into a bf16
    main_grad, persistent with c_pre (2x2 clusters); and bf16 out without beta = 1 (persistent), fp32 out with beta = 1
    (not persistent, no early fetch); an empty expert leaves its slice unchanged (or zero)."""
    lens = [128 * (i % 5) + (0 if i % 7 == 3 else 37 * (i % 3)) for i in range(40)]
    M, N = 320, 2560
    a, sa, b, sb, offs, ref = _vk(lens, M, N, seed=13)
    a, b = _slack(a), _slack(preshuffle_b16(b))
    for dt, acc in ((BF16, True), (BF16, False), (F32, True)):
        c0 = (torch.randn(len(lens), M, N, device=DEV) * 64).to(dt)
        out, check = _guarded((len(lens), M, N), dt, c0 if acc else None)
        grouped_gemm_mxfp8_variable_k_gfx1250(a, sa, b, sb, offs, out_dtype=dt, out=out, accumulate=acc)
        _eq(out, (ref + c0.float()) if acc else ref)
        check()


def test_grouped_gemm_mxfp8_variable_k_narrow():
    """The default config of a narrow N into a bf16 main_grad (N <= 2048: persistent 1x1 with c_pre)."""
    lens = [300, 77, 0, 1000, 1]
    M, N = 512, 768
    a, sa, b, sb, offs, ref = _vk(lens, M, N, seed=17)
    c0 = (torch.randn(len(lens), M, N, device=DEV) * 64).to(BF16)
    out, check = _guarded((len(lens), M, N), BF16, c0)
    grouped_gemm_mxfp8_variable_k_gfx1250(
        _slack(a), sa, _slack(preshuffle_b16(b)), sb, offs, out=out, accumulate=True
    )
    _eq(out, ref + c0.float())
    check()


def test_gemm_mxfp8_deterministic():
    """The production configs repeat bit for bit (operands prepared once): dense c2x4, grouped-M c2x4 and the
    persistent variable-K wgrad into bf16."""
    g = torch.Generator(device=DEV).manual_seed(7)
    a, sa = quant_rows(torch.randn(1000, 7168, device=DEV, generator=g))
    b, sb = quant_rows(torch.randn(2048, 7168, device=DEV, generator=g))
    a, sa, b, sb = _slack(a), pack_scales(sa), _slack(preshuffle_b16(b)), pack_scales(sb)
    calls = [lambda out: gemm_mxfp8_gfx1250(a, sa, b, sb, out=out, config=KP1_C2X4, b_preshuffled=True)]
    ga, gsa, gb, gsb, offs, offs_p, _ = _grouped(RAGGED, 1024, 2048, seed=7)
    gsa, gsb4 = pack_scales(gsa), torch.stack([pack_scales(gsb[e]) for e in range(len(RAGGED))])
    ga, gb = _slack(ga), _slack(preshuffle_b16(gb))
    calls.append(
        lambda out: grouped_gemm_mxfp8_gfx1250(ga, gsa, gb, gsb4, offs, offs_p, sum(RAGGED), out=out)
    )
    lens = [128 * (i % 5) + (0 if i % 7 == 3 else 37 * (i % 3)) for i in range(40)]
    va, vsa, vb, vsb, voffs, _ = _vk(lens, 320, 2560, seed=13)
    va, vb = _slack(va), _slack(preshuffle_b16(vb))
    c0 = torch.randn(len(lens), 320, 2560, device=DEV).to(BF16)

    def vk(out):
        out.copy_(c0)
        grouped_gemm_mxfp8_variable_k_gfx1250(va, vsa, vb, vsb, voffs, out=out, accumulate=True)

    calls.append(vk)
    for call, shape in zip(calls, ((1000, 2048), (sum(RAGGED), 1024), (len(lens), 320, 2560))):
        first = torch.empty(shape, dtype=BF16, device=DEV)
        call(first)
        out = torch.empty_like(first)
        for _ in range(100):
            call(out)
            assert torch.equal(out, first)


def test_gemm_mxfp8_tail_at_allocation_end():
    """Operands that end exactly where their (10 MiB or more) allocation does, so a read past them could fault
    (best effort). With the default configs: a K tail at kpair 2 reads half rows of a row-major A, nothing past
    it; a preshuffled b or raw scales without their slack make the entry fall back to kpair 1 (never persistent);
    the variable-K wgrad's empty last expert reads inside the operands; K % 128 != 0 without slack is refused."""
    M, N, K = 16384, 256, 640  # dense default: 128x128, kpair 2 (a half last K step)
    a, sa, b, sb = int_valued_operands(M, N, K, seed=1)
    ref = gemm_ref(a, sa, b, sb).float()
    a = _exact(a)
    assert a.numel() == 10 << 20 and KM._slack(a) == 0
    sa, sb, bs = pack_scales(sa), pack_scales(sb), preshuffle_b16(b)
    _eq(gemm_mxfp8_gfx1250(a, sa, _slack(bs), sb, b_preshuffled=True), ref)
    _eq(gemm_mxfp8_gfx1250(a, sa, _exact(bs), sb, b_preshuffled=True), ref)  # no slack: kpair 1
    # beta = 1 into bf16, K % 256 == 128: the persistent c2x2 default needs b's slack; without it kpair 1
    M, N, K = 4096, 3072, 4224
    a, sa, b, sb = int_valued_operands(M, N, K, seed=2)
    ref = gemm_ref(a, sa, b, sb).float()
    c0 = (torch.randn(M, N, device=DEV) * 64).to(BF16)
    out = c0.clone()
    gemm_mxfp8_gfx1250(
        _slack(a),
        pack_scales(sa),
        _exact(preshuffle_b16(b)),
        pack_scales(sb),
        out=out,
        b_preshuffled=True,
        accumulate=True,
    )
    _eq(out, ref + c0.float())
    # grouped-M, K < 2048 (the persistent kpair-2 default), K % 256 == 128, b without its slack: kpair 1
    lens = [5000, 0, 6000, 4000]
    a, sa, b, sb, offs, offs_p, ref = _grouped(lens, 512, 640, seed=3)
    sb4 = torch.stack([pack_scales(sb[g]) for g in range(len(lens))])
    out = grouped_gemm_mxfp8_gfx1250(
        _slack(a), pack_scales(sa), _exact(preshuffle_b16(b)), sb4, offs, offs_p, 15000
    )
    _eq(out, ref)
    # variable-K, odd counts of 128 tokens and an empty last expert (its K step starts at T, the end of a and b)
    lens = [384, 128, 1408, 2048, 1152, 0]
    M, N = 2048, 2048
    a, sa, b, sb, offs, ref = _vk(lens, M, N, seed=4)
    a = _exact(a)
    for bs in (
        _slack(preshuffle_b16(b)),
        _exact(preshuffle_b16(b)),
    ):  # kpair 2 with half steps; no slack: kpair 1
        _eq(grouped_gemm_mxfp8_variable_k_gfx1250(a, sa, bs, sb, offs), ref)
    a, sa, b, sb = int_valued_operands(256, 256, 576)
    with pytest.raises(ValueError, match="pad K"):
        gemm_mxfp8_gfx1250(a, pack_scales(sa), b, pack_scales(sb), config=C(256, 256, 2, 2, 1, 1))


def test_gemm_mxfp8_rejects():
    a, sa, b, sb = int_valued_operands(256, 260, 256)
    with pytest.raises(ValueError, match="multiples of 8"):
        gemm_mxfp8_gfx1250(a, sa, b, sb)
    a, sa, b, sb = int_valued_operands(256, 256, 256)
    with pytest.raises(ValueError, match="E8M0"):
        gemm_mxfp8_gfx1250(a, sa.float(), b, sb)
    with pytest.raises(ValueError, match="persist=True needs"):
        gemm_mxfp8_gfx1250(a, sa, b, sb, config=C(256, 256, 4, 1, 1, 1, persist=True))
    with pytest.raises(ValueError, match="row stride"):  # overlapping output rows
        gemm_mxfp8_gfx1250(
            a,
            sa,
            b,
            sb,
            out=torch.empty(256 * 128 + 128, dtype=BF16, device=DEV).as_strided((256, 256), (128, 1)),
        )
    offs = torch.tensor([0, 128, 256, 256], dtype=torch.int64, device=DEV)
    with pytest.raises(ValueError, match="contiguous int64"):
        grouped_gemm_mxfp8_gfx1250(a, sa, b.view(1, 256, 256), sb.view(1, 256, 8), offs[::2], offs[::2], 256)
