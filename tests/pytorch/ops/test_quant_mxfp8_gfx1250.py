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

    assert (mxfp8_quant_gfx1250.PACK_ROW, mxfp8_quant_gfx1250.PACK_COL) == (PACK_ROW, PACK_COL)
    assert (mxfp8_quant_gfx1250.SHUF_ROW, mxfp8_quant_gfx1250.SHUF_COL) == (SHUF_ROW, SHUF_COL)
    return mxfp8_quant_gfx1250


PACK_ROW, PACK_COL, SHUF_ROW, SHUF_COL = 1, 2, 4, 8  # the kernels' layout bits


def _pack(s):
    """mxfp8_ref.pack_scales over any leading dims."""
    s = s.cpu().view(torch.uint8)
    return R.pack_scales(s) if s.ndim == 2 else torch.stack([_pack(t) for t in s])


def _lay(ref, lay):
    """Reference (q_row, s_row, q_col, s_col) in the layout ``lay``."""
    qr, sr, qc, sc = (t.cpu() for t in ref)
    return (R.preshuffle_b16(qr) if lay & SHUF_ROW else qr, _pack(sr) if lay & PACK_ROW else sr,
            R.preshuffle_b16(qc) if lay & SHUF_COL else qc, _pack(sc) if lay & PACK_COL else sc)


def _max_lay(rows, cols, feats, toks):
    """Packed scales, and 16x16 codes where the copy's dims are multiples of 16."""
    return PACK_ROW | PACK_COL | (SHUF_ROW if rows % 16 == cols % 16 == 0 else 0) | (
        SHUF_COL if feats % 16 == toks % 16 == 0 else 0)


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
    _check_grouped(Q.grouped_quant_mxfp8_raw(x, lens_d, offs, torch.float8_e4m3fn, rule=rule), x, lens, rule)


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


# ---- fused producers: GLU / dGLU (gfx950 phase-2 math) and gather (permute), MoE dual cast
def _ulp_close(got, ref, what):
    """bf16 within one ulp of the torch math (the kernels use the hardware exp2 / reciprocal), and
    nearly all exactly equal."""
    a, b = got.cpu().view(torch.int16).int(), ref.cpu().view(torch.int16).int()
    near = ((a - b).abs() <= 1) | ((got.cpu().float() - ref.cpu().float()).abs() <= 1e-35)
    assert near.all(), f"{what}: {int((~near).sum())} values more than an ulp off"
    assert (a == b).float().mean() > 0.995, f"{what}: only {float((a == b).float().mean()):.4f} exact"


def _moe_lens(case, seed):
    if case == "skew":  # one hot expert, empties, a single token
        return torch.tensor([1500, 0, 1, 300, 0, 0, 77, 640, 0, 2, 129, 0, 31, 0, 0, 256])
    return _lens(case, seed)


def _check_grouped(out, vals, lens, rule, lay=0):
    """An 8-tuple against mxfp8_ref.quant_grouped of the bf16 values it quantized: the allocation bounds, the
    32-row / 128-token padded lens and offsets of both copies, the rows / columns up to them in layout ``lay``."""
    q_row, s_row, q_col, s_col, lens_c, offs_c, lens_r, offs_r = R.quant_grouped(vals.cpu(), lens, rule)
    M, G, Mr, Mc = int(lens.sum()), len(lens), int(offs_r[-1]), int(offs_c[-1])
    assert out[0].shape[0] == -(-(M + 32 * G) // 32) * 32 and out[2].shape[1] == -(-(M + 128 * G) // 128) * 128
    for got, ref, what in zip(out[4:], (lens_r, offs_r, lens_c, offs_c), ("lens_row", "offs_row", "lens_col", "offs_col")):
        assert torch.equal(got.cpu(), ref.cpu()), f"{what}: {got.tolist()} vs {ref.tolist()}"
    qr, sr, _, sc = _lay((q_row, s_row, q_col, s_col), lay)
    _eq(out[0][:Mr], qr, "q_row")
    _eq(out[1][: Mr // 32] if lay & PACK_ROW else out[1][:Mr], sr, "s_row")
    _eq((R.unpreshuffle_b16(out[2].cpu()) if lay & SHUF_COL else out[2])[:, :Mc], q_col, "q_col")
    _eq(out[3][:, : Mc // 128] if lay & PACK_COL else out[3][:, : Mc // 32], sc, "s_col")


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("act,clamp", [("silu", None), ("silu", 7.0), ("gelu", None), ("gelu", 7.0)])
@pytest.mark.parametrize("I", [96, 2048])
@pytest.mark.parametrize("case", [[0, 5, 0], [31, 32, 33, 1, 0, 127, 128, 129], "random", "skew"])
def test_glu(Q, case, I, act, clamp, rule):
    lens = _moe_lens(case, I)
    M = int(lens.sum())
    g = torch.Generator(device="cuda").manual_seed(M + I)
    l1 = (torch.randn(M, 2 * I, generator=g, device="cuda") * 3).to(torch.bfloat16)
    probs = torch.rand(M, generator=g, device="cuda")
    probs[::17] = 0
    lens_d = lens.cuda()
    offs = torch.nn.functional.pad(torch.cumsum(lens_d, 0), (1, 0))
    act_v = torch.full((M, I), float("nan"), dtype=torch.bfloat16, device="cuda")
    out = Q.grouped_glu_quant_mxfp8(l1, probs, lens_d, offs, act, clamp, rule=rule, act_out=act_v)
    _ulp_close(act_v, R.glu_ref(l1.cpu(), probs.cpu(), act, clamp), "act")
    _check_grouped(out, act_v, lens, rule)


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("act,clamp", [("silu", None), ("silu", 7.0), ("gelu", None), ("gelu", 7.0)])
@pytest.mark.parametrize("I", [96, 2048])
@pytest.mark.parametrize("case", [[0, 5, 0], [31, 32, 33, 1, 0, 127, 128, 129], "random", "skew"])
def test_dglu(Q, case, I, act, clamp, rule):
    lens = _moe_lens(case, I + 1)
    M = int(lens.sum())
    g = torch.Generator(device="cuda").manual_seed(M + 3 * I)
    l1 = (torch.randn(M, 2 * I, generator=g, device="cuda") * 3).to(torch.bfloat16)
    dact = torch.randn(M, I, generator=g, device="cuda").to(torch.bfloat16)
    probs = torch.rand(M, generator=g, device="cuda")
    lens_d = lens.cuda()
    offs = torch.nn.functional.pad(torch.cumsum(lens_d, 0), (1, 0))
    grad_v = torch.full((M, 2 * I), float("nan"), dtype=torch.bfloat16, device="cuda")
    gp, *out = Q.grouped_dglu_quant_mxfp8(dact, l1, probs, lens_d, offs, act, clamp, rule=rule, grad_out=grad_v)
    ref_grad, ref_gp = R.dglu_ref(dact.cpu(), l1.cpu(), probs.cpu(), act, clamp)
    _ulp_close(grad_v, ref_grad, "grad_l1")
    torch.testing.assert_close(gp.cpu(), ref_gp, rtol=1e-4, atol=1e-4 * float(ref_gp.abs().max() + 1))
    _check_grouped(out, grad_v, lens, rule)


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("H", [64, 200, 7168])
@pytest.mark.parametrize("case", [[0, 5, 0], [31, 32, 33, 1, 0, 127, 128, 129], "random", "skew"])
@pytest.mark.parametrize("with_probs", [True, False])
def test_gather(Q, case, H, with_probs, rule):
    lens = _moe_lens(case, H)
    M = int(lens.sum())
    T = max(1, M // 3)
    g = torch.Generator(device="cuda").manual_seed(M + H)
    x = _x(T, H, seed=H)
    d2s = torch.randint(0, T, (M,), generator=g, device="cuda", dtype=torch.int32)
    if M > 4:
        d2s[1::9], d2s[3] = -1, T + 5  # rows of no token
    pp = torch.rand(M, generator=g, device="cuda") if with_probs else None
    if pp is not None:
        pp[::7] = 0
    lens_d = lens.cuda()
    offs = torch.nn.functional.pad(torch.cumsum(lens_d, 0), (1, 0))
    out = Q.grouped_gather_quant_mxfp8(x, d2s, lens_d, offs, pp, rule=rule)
    _check_grouped(out, R.gather_rows(x.cpu(), d2s.cpu(), None if pp is None else pp.cpu()), lens, rule)


# ---- layouts: packed scales / 16x16 codes, the same kernels with other store addressing
def _padded(t, shape):
    out = torch.zeros(shape, dtype=torch.uint8)
    out[tuple(slice(0, d) for d in t.shape)] = t.cpu().view(torch.uint8)
    return out


def _c128(n):
    return -(-n // 128) * 128


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("pad", [False, True])
@pytest.mark.parametrize("M,K", [(1, 32), (33, 64), (100, 200), (128, 128), (160, 336), (48, 2048), (300, 7168)])
def test_layout_dense(Q, M, K, pad, rule):
    x = _x(M, K, seed=M + 5 * K, special=not pad)  # (pad: a NaN block's codes in the K pad are 0xFF, not 0)
    KO, Mc = (_c128(K), _c128(M)) if pad else (K, M)
    lay = _max_lay(M, KO, K, Mc)
    shapes = ((M, KO), (M, -(-KO // 32)), (K, Mc), (K, -(-Mc // 32)))
    ref = _lay([_padded(t, sh) for t, sh in zip(R.quant_dual(x.cpu(), rule), shapes)], lay)
    for got, exp, what in zip(Q.quant_mxfp8_raw(x, rule=rule, pad=pad, layout=lay), ref, ("qr", "sr", "qc", "sc")):
        _eq(got, exp, what)
    for got, exp, what in zip(Q.quant_mxfp8_rows(x, rule=rule, pad=pad, layout=lay), ref[:2], ("rows q", "rows s")):
        _eq(got, exp, what)
    for got, exp, what in zip(Q.quant_mxfp8_cols(x, rule=rule, pad=pad, layout=lay), ref[2:], ("cols q", "cols s")):
        _eq(got, exp, what)


def _check_dense(out, vals, rule, pad, lay):
    """A dense dual cast (q_row, s_row, q_col, s_col) against mxfp8_ref.quant_dual of the bf16 values it quantized
    (``pad``: the token dim padded to 128, zero codes / byte-0 scales), in layout ``lay``."""
    M, W = vals.shape
    Mc = _c128(M) if pad else M
    shapes = ((M, W), (M, -(-W // 32)), (W, Mc), (W, -(-Mc // 32)))
    ref = _lay([_padded(t, sh) for t, sh in zip(R.quant_dual(vals.cpu(), rule), shapes)], lay)
    for got, exp, what in zip(out, ref, ("q_row", "s_row", "q_col", "s_col")):
        _eq(got, exp, what)


# dense GLU / dGLU (a dense SwiGLU MLP, the shared expert): probs None = no row weights, bit-identical to all ones
DENSE_GLU = [(1, 128, True), (300, 96, False), (257, 2048, True), (130, 256, True)]
GLU_MATH = [("silu", None, "nosat"), ("gelu", 7.0, "even")]  # activation, clamp, scale rule


@pytest.mark.parametrize("act,clamp,rule", GLU_MATH)
@pytest.mark.parametrize("M,I,pad", DENSE_GLU)
@pytest.mark.parametrize("with_probs", [False, True])
def test_glu_dense(Q, M, I, pad, act, clamp, with_probs, rule):
    g = torch.Generator(device="cuda").manual_seed(M + I)
    l1 = (torch.randn(M, 2 * I, generator=g, device="cuda") * 3).to(torch.bfloat16)
    probs = torch.rand(M, generator=g, device="cuda") if with_probs else None
    lay = _max_lay(M, I, I, _c128(M) if pad else M)
    act_v = torch.full((M, I), float("nan"), dtype=torch.bfloat16, device="cuda")
    out = Q.glu_quant_mxfp8(l1, probs, act, clamp, rule=rule, pad=pad, act_out=act_v, layout=lay)
    ones = torch.ones(M, device="cuda")
    _ulp_close(act_v, R.glu_ref(l1.cpu(), (ones if probs is None else probs).cpu(), act, clamp), "act")
    _check_dense(out, act_v, rule, pad, lay)
    if probs is None:  # = the probs path with ones, bit for bit
        for got, exp, what in zip(out, Q.glu_quant_mxfp8(l1, ones, act, clamp, rule=rule, pad=pad, layout=lay),
                                  ("q_row", "s_row", "q_col", "s_col")):
            _eq(got, exp, f"no probs vs ones: {what}")


@pytest.mark.parametrize("act,clamp,rule", GLU_MATH)
@pytest.mark.parametrize("M,I,pad", DENSE_GLU)
@pytest.mark.parametrize("with_probs", [False, True])
def test_dglu_dense(Q, M, I, pad, act, clamp, with_probs, rule):
    g = torch.Generator(device="cuda").manual_seed(M + 3 * I)
    l1 = (torch.randn(M, 2 * I, generator=g, device="cuda") * 3).to(torch.bfloat16)
    dact = torch.randn(M, I, generator=g, device="cuda").to(torch.bfloat16)
    probs = torch.rand(M, generator=g, device="cuda") if with_probs else None
    lay = _max_lay(M, 2 * I, 2 * I, _c128(M) if pad else M) & ~(0 if I % 128 == 0 else PACK_ROW)
    grad_v = torch.full((M, 2 * I), float("nan"), dtype=torch.bfloat16, device="cuda")
    gp, *out = Q.dglu_quant_mxfp8(dact, l1, probs, act, clamp, rule=rule, pad=pad, grad_out=grad_v, layout=lay)
    ones = torch.ones(M, device="cuda")
    ref_grad, ref_gp = R.dglu_ref(dact.cpu(), l1.cpu(), (ones if probs is None else probs).cpu(), act, clamp)
    _ulp_close(grad_v, ref_grad, "grad_l1")
    if probs is None:
        assert gp is None
    else:
        torch.testing.assert_close(gp.cpu(), ref_gp, rtol=1e-4, atol=1e-4 * float(ref_gp.abs().max() + 1))
    _check_dense(out, grad_v, rule, pad, lay)


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize(
    "B,N,K,pad", [(1, 64, 96, False), (3, 64, 96, True), (2, 256, 2048, False), (2, 129, 257, False),
                  (2, 129, 257, True), (1, 300, 7168, True)]
)
def test_layout_2d(Q, B, N, K, pad, rule):
    w = _x(B, N, K, seed=B + N + K, special=not pad)
    KO, Nc = (_c128(K), _c128(N)) if pad else (K, N)
    lay = _max_lay(N, KO, K, Nc)
    out = Q.quant_mxfp8_raw_batched(w, row_2d=True, col_2d=True, rule=rule, pad=pad, layout=lay)
    shapes = ((N, KO), (N, -(-KO // 32)), (K, Nc), (K, -(-Nc // 32)))
    for b in range(B):
        ref = _lay([_padded(t, sh) for t, sh in zip(R.quant_2d(w[b].cpu(), rule), shapes)], lay)
        for got, exp, what in zip(out, ref, ("q", "s_row", "q_t", "s_t")):
            _eq(got[b], exp, f"{what}[{b}]")


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("case", [[0, 5, 0], [31, 32, 33, 1, 0, 127, 128, 129], "skew"])
@pytest.mark.parametrize(
    "kind,dim", [("grouped", 64), ("grouped", 200), ("grouped", 2048), ("glu", 96), ("glu", 2048), ("dglu", 96),
                 ("dglu", 2048), ("gather", 64), ("gather", 7168)]
)
def test_layout_moe(Q, kind, dim, case, rule):
    """Every MoE producer's copies in the GEMM layouts (dGLU packs its row scales only for I % 128 == 0)."""
    lens = _moe_lens(case, dim)
    M = int(lens.sum())
    C = 2 * dim if kind == "dglu" else dim
    lay = PACK_COL | (SHUF_ROW | SHUF_COL if C % 16 == 0 else 0) | (0 if kind == "dglu" and dim % 128 else PACK_ROW)
    g = torch.Generator(device="cuda").manual_seed(M + dim)
    lens_d = lens.cuda()
    offs = torch.nn.functional.pad(torch.cumsum(lens_d, 0), (1, 0))
    l1 = (torch.randn(M, 2 * dim, generator=g, device="cuda") * 3).to(torch.bfloat16)
    probs = torch.rand(M, generator=g, device="cuda")
    vals = torch.full((M, C), float("nan"), dtype=torch.bfloat16, device="cuda")
    if kind == "grouped":
        vals = _x(M, dim, seed=M + dim)
        out = Q.grouped_quant_mxfp8_raw(vals, lens_d, offs, rule=rule, layout=lay)
    elif kind == "glu":
        out = Q.grouped_glu_quant_mxfp8(l1, probs, lens_d, offs, "silu", 7.0, rule=rule, act_out=vals, layout=lay)
    elif kind == "dglu":
        dact = torch.randn(M, dim, generator=g, device="cuda").to(torch.bfloat16)
        _, *out = Q.grouped_dglu_quant_mxfp8(dact, l1, probs, lens_d, offs, "gelu", 7.0, rule=rule, grad_out=vals,
                                             layout=lay)
    else:
        T = max(1, M // 3)
        x = _x(T, dim, seed=dim)
        d2s = torch.randint(-1, T, (M,), generator=g, device="cuda", dtype=torch.int32)
        probs[::7] = 0
        out = Q.grouped_gather_quant_mxfp8(x, d2s, lens_d, offs, probs, rule=rule, layout=lay)
        vals = R.gather_rows(x.cpu(), d2s.cpu(), probs.cpu())
    _check_grouped(out, vals, lens, rule, lay)


# ---- wiring: quantization_impl on gfx1250 (HIP layouts: K / M padded to 128, zero codes, byte-0 scales)
def _dispatch_names():
    from primus_turbo.pytorch.core.low_precision import (
        Float8QuantConfig,
        Format,
        ScaleDtype,
        ScalingGranularity,
        ScalingRecipe,
    )

    cfg = Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.MX_BLOCKWISE,
                            scale_dtype=ScaleDtype.E8M0, block_size=32)
    return ScalingGranularity.MX_BLOCKWISE, ScalingRecipe, cfg


@pytest.mark.parametrize("rule", RULES)
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
        for got, ref, shape, what in ((qr, rq, (200, Kp), "q_row"), (sr, rs, (200, Kp // 32), "s_row"),
                                      (qc, rqc, (330, Mp), "q_col"), (sc, rsc, (330, Mp // 32), "s_col")):
            _eq(got, _padded(ref, shape), what)
        for axis, ref, what in ((1, (rq, rs), "rows"), (0, (rqc, rsc), "cols")):
            got = quantize_fp8(x, E4, MX, block_size=32, axis=axis, scaling_recipe=Recipe())
            _eq(got[0], _padded(ref[0], got[0].shape), f"{what} q")
            _eq(got[1], _padded(ref[1], got[1].shape), f"{what} s")
        w = _x(3, 96, 200, seed=6, special=False)
        r2 = Recipe(use_2d_block=True)
        out = quantize_fp8_with_trans(w, E4, MX, block_size=32, scaling_recipe=r2, scaling_recipe_for_trans=r2)
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
            one = grouped_quantize_fp8(xg, E4, MX, lens_d, offs, block_size=32, axis=axis, scaling_recipe=Recipe())
            _eq(one[0][:n] if axis else one[0][:, :n], out[idx[0]][:n] if axis else out[idx[0]][:, :n], f"single {axis}")
            assert torch.equal(one[3].cpu(), ref[idx[3]]), f"single {axis} offs"
        src = _x(100, 330, seed=8, special=False)
        d2s = torch.randint(0, 100, (int(lens.sum()),), device="cuda", dtype=torch.int32)
        pp = torch.rand(d2s.numel(), device="cuda")
        pp[::5] = 0
        out = grouped_quantize_fp8_with_trans(src, E4, MX, lens_d, offs, block_size=32, dest2src=d2s,
                                              permuted_probs=pp, total_m=d2s.numel())
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
    rr, rc = Recipe(shuffle_scale=True), Recipe(shuffle_scale=True, shuffle_out=True)  # A rows, B-type columns
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
    for g_, e_, what in zip(got, _lay(raw, PACK_ROW | PACK_COL | SHUF_COL), ("q_row", "s_row", "q_col", "s_col")):
        _eq(g_, e_, f"activation {what}")
    lens = torch.tensor([0, 37, 1, 200, 0, 129])
    lens_d = lens.cuda()
    offs = torch.nn.functional.pad(torch.cumsum(lens_d, 0), (1, 0))
    xg = _x(int(lens.sum()), 384, seed=7, special=False)
    out = grouped_quantize_fp8_with_trans(xg, E4, MX, lens_d, offs, block_size=32, scaling_recipe=rr,
                                          scaling_recipe_for_trans=rc)
    _check_grouped(out, xg, lens, "nosat", PACK_ROW | PACK_COL | SHUF_COL)
    for axis, rec in ((-1, rr), (-2, rc)):  # dequantize: dense and grouped, each copy
        for kw, t in (({}, x), ({"group_lens": lens_d}, xg)):
            a = QuantizedTensor.quantize(t, E4, MX, block_size=32, scaling_recipe=rec, axis=axis, **kw)
            b = QuantizedTensor.quantize(t, E4, MX, block_size=32, scaling_recipe=Recipe(), axis=axis, **kw)
            _eq(a.dequantize(), b.dequantize(), f"dequantize {axis} {list(kw)}")
    with pytest.raises(NotImplementedError):  # fp16 / E5M2 / mixed block shapes: not these kernels, no layouts
        quantize_fp8_with_trans(x.half(), E4, MX, block_size=32, scaling_recipe=rr, scaling_recipe_for_trans=rc)
    with pytest.raises(NotImplementedError):
        quantize_fp8(x, torch.float8_e5m2, MX, block_size=32, axis=1, scaling_recipe=rr)
    with pytest.raises(NotImplementedError):
        quantize_fp8_with_trans(x, E4, MX, block_size=32, scaling_recipe=shuf2, scaling_recipe_for_trans=rc)
    with pytest.raises(AssertionError):  # 16x16 codes need rows % 16 == 0
        quantize_fp8(_x(100, 320, special=False), E4, MX, block_size=32, axis=1, scaling_recipe=Recipe(shuffle_out=True))


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
        lambda: quantize_fp8_with_trans(x, E4, MX, block_size=32, scaling_recipe=r2, scaling_recipe_for_trans=r1),
        lambda: grouped_quantize_fp8_with_trans(x, E4, MX, lens, offs, block_size=32, scaling_recipe=r2),
    ):
        with pytest.raises(RuntimeError, match="PRIMUS_TURBO_MX_STRICT=2"):
            call()


def test_operand_slack():
    """Every GEMM operand the quantizers allocate (codes and scales, both copies, any layout and producer, the cached
    weights) has >= 2 KiB of storage past its last byte: the gfx1250 MX GEMM's TDM loads address past a tensor's end."""
    from primus_turbo.flydsl.quantization import mxfp8_quant_gfx1250 as Q
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
    w = create_quantized_weight(_x(64, 96, special=False), E4, cfg, need_weight_transpose_cache=True,
                                scaling_recipe=shuf2)
    outs["weight"] = [t for qt_ in w for t in (qt_.qdata, qt_.scale_inv)]
    for name, ts in outs.items():
        for i, t in enumerate(ts):
            assert t is not None and slack(t) >= 2048, f"{name}[{i}]: only {slack(t)} bytes past the end"
