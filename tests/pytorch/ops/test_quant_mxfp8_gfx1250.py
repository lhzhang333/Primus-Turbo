###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""gfx1250 MXFP8 quantization kernels (primus_turbo/flydsl/quantization/mxfp8_quant_gfx1250.py) against the recipe
reference tests/pytorch/ref/mxfp8_ref.py, bit for bit (reference on the CPU: IEEE denormals, no device pow). Each
case compiles its own kernel (widths, expert counts, rules, layouts and producers are compile-time), so every
feature appears in a few cases rather than in a full sweep: K / M tails (K % 128, K % 8, M % 128 / 32 / 16 / 4),
both scale rules, 2-D blocks, every layout, each producer (dense, grouped, GLU / dGLU grouped and dense, gather),
empty experts, and inputs that end where their mapping does."""

import pytest
import torch

from primus_turbo.pytorch.core.utils import is_gfx1250
from tests.pytorch.ref import mxfp8_ref as R
from tests.pytorch.ref.mxfp8_ref import assert_bytes_equal as _eq
from tests.pytorch.ref.mxfp8_ref import mx_input as _x
from tests.pytorch.ref.mxfp8_ref import padded as _padded
from tests.pytorch.test_utils import tensor_at_mapping_end

pytestmark = pytest.mark.skipif(not is_gfx1250(), reason="gfx1250 kernels")
if is_gfx1250():
    from primus_turbo.flydsl.quantization import mxfp8_quant_gfx1250 as Q
    from primus_turbo.flydsl.quantization.mxfp8_quant_gfx1250 import (
        PACK_COL,
        PACK_ROW,
        SHUF_COL,
        SHUF_ROW,
    )


def _flags(lay):
    return dict(
        pack_row=bool(lay & PACK_ROW),
        shuf_row=bool(lay & SHUF_ROW),
        pack_col=bool(lay & PACK_COL),
        shuf_col=bool(lay & SHUF_COL),
    )


def _lay(ref, lay):
    """Reference (q_row, s_row, q_col, s_col) in the layout ``lay`` (the kernels' layout bits)."""
    return R.to_layout(ref, **_flags(lay))


def _check_grouped(out, vals, lens, rule, lay=0):
    R.check_grouped(out, vals, lens, rule, **_flags(lay))


def _max_lay(rows, cols, feats, toks):
    """Packed scales, and 16x16 codes where the copy's dims are multiples of 16."""
    return (
        PACK_ROW
        | PACK_COL
        | (SHUF_ROW if rows % 16 == cols % 16 == 0 else 0)
        | (SHUF_COL if feats % 16 == toks % 16 == 0 else 0)
    )


def _c128(n):
    return -(-n // 128) * 128


def _offs(lens):
    return torch.nn.functional.pad(torch.cumsum(lens, 0), (1, 0))


# K: 1 / 33 / 300 (K % 8 != 0: the inputs get storage past their end), 200 / 576 (K % 128 != 0: the last row's
# partial k-tile by masked loads), 128 / 7168; M: 1, 3, 100 (% 4), 128 (% 16), 130, 257, 300.
@pytest.mark.parametrize(
    "M,K,rule",
    [
        (1, 1, "nosat"),
        (3, 200, "even"),
        (100, 33, "nosat"),
        (128, 128, "even"),
        (130, 300, "nosat"),
        (257, 576, "nosat"),
        (300, 7168, "even"),
    ],
)
def test_dual(M, K, rule):
    x = _x(M, K, seed=M * 7 + K)
    out = Q.quant_mxfp8_raw(x, torch.float8_e4m3fn, rule=rule)
    for got, ref, what in zip(out, R.quant_dual(x.cpu(), rule), ("q_row", "s_row", "q_col", "s_col")):
        _eq(got, ref, what)
    q, s = Q.quant_mxfp8_rows(x, rule=rule)
    _eq(q, out[0], "rows-only q")
    _eq(s, out[1], "rows-only s")


@pytest.mark.parametrize(
    "B,N,K,rule", [(1, 1, 1, "nosat"), (3, 64, 96, "even"), (2, 129, 257, "nosat"), (1, 300, 7168, "nosat")]
)
def test_2d(B, N, K, rule):
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
    if case == "skew":  # one hot expert, empties, a single token
        return torch.tensor([1500, 0, 1, 300, 0, 0, 77, 640, 0, 2, 129, 0, 31, 0, 0, 256])
    return torch.tensor(case)


@pytest.mark.parametrize(
    "case,K,rule",
    [
        ([1], 64, "nosat"),
        ([0, 5, 0], 200, "even"),
        ([31, 32, 33, 1, 0, 127, 128, 129], 2048, "nosat"),
        ([256, 256, 256, 256], 200, "nosat"),
        ([0, 0, 300], 64, "even"),
        ([0, 0, 0], 200, "nosat"),  # no tokens: only the lens / offsets are written
        ("random", 2048, "even"),
        ("many", 64, "nosat"),
    ],
)
def test_grouped(case, K, rule):
    lens = _lens(case, K)
    M = int(lens.sum())
    x = _x(M, K, seed=K + M, special=M > 0)
    _check_grouped(Q.grouped_quant_mxfp8_raw(x, lens.cuda(), _offs(lens.cuda()), rule=rule), x, lens, rule)


def test_special_blocks():
    """Hand-made blocks: zero, denormal, NaN, +-Inf, exact powers, the rules' boundaries."""
    vals = [0.0, 2.0**-133, -(2.0**-130), 2.0**-126, 448.0, -448.0, 449.0, 1.9375 * 2**20, -1.875 * 2**-90,
            float("inf"), -float("inf"), float("nan"), 3.0e38, -3.0e38, 1.0, -0.0]  # fmt: skip
    x = torch.zeros(64, 64)
    for i, v in enumerate(vals):
        x[i, :32] = v * torch.linspace(-1, 1, 32) if i % 2 else v  # whole-block value or a ramp ending in it
        x[i, 32 + i] = v
        x[16 + i, 7] = v
    x = x.to(torch.bfloat16).cuda()
    for rule in ("nosat", "even"):
        ref = R.quant_dual(x.cpu(), rule)
        for got, exp, what in zip(Q.quant_mxfp8_raw(x, rule=rule), ref, ("qr", "sr", "qc", "sc")):
            _eq(got, exp, f"{rule} {what}")


# ---- fused producers: GLU / dGLU (gfx950 phase-2 math) and gather (permute), MoE dual cast
def _ulp_close(got, ref, what):
    """bf16 within one ulp of the torch math (the kernels use the hardware exp2 / reciprocal), and
    nearly all exactly equal."""
    a, b = got.cpu().view(torch.int16).int(), ref.cpu().view(torch.int16).int()
    near = ((a - b).abs() <= 1) | ((got.cpu().float() - ref.cpu().float()).abs() <= 1e-35)
    assert near.all(), f"{what}: {int((~near).sum())} values more than an ulp off"
    assert (a == b).float().mean() > 0.995, f"{what}: only {float((a == b).float().mean()):.4f} exact"


# I = 96: I % 128 != 0 (masked loads for the last row's partial k-tile of l1's halves and dact)
GLU_CASES = [
    ([0, 5, 0], 96, "silu", None, "nosat"),
    ([31, 32, 33, 1, 0, 127, 128, 129], 2048, "silu", 7.0, "even"),
    ("random", 96, "gelu", 7.0, "nosat"),
    ("skew", 2048, "gelu", None, "nosat"),
]


@pytest.mark.parametrize("case,I,act,clamp,rule", GLU_CASES)
def test_glu(case, I, act, clamp, rule):
    lens = _lens(case, I)
    M = int(lens.sum())
    g = torch.Generator(device="cuda").manual_seed(M + I)
    l1 = (torch.randn(M, 2 * I, generator=g, device="cuda") * 3).to(torch.bfloat16)
    probs = torch.rand(M, generator=g, device="cuda")
    probs[::17] = 0
    lens_d = lens.cuda()
    act_v = torch.full((M, I), float("nan"), dtype=torch.bfloat16, device="cuda")
    out = Q.grouped_glu_quant_mxfp8(l1, probs, lens_d, _offs(lens_d), act, clamp, rule=rule, act_out=act_v)
    _ulp_close(act_v, R.glu_ref(l1.cpu(), probs.cpu(), act, clamp), "act")
    _check_grouped(out, act_v, lens, rule)


@pytest.mark.parametrize("case,I,act,clamp,rule", GLU_CASES)
def test_dglu(case, I, act, clamp, rule):
    lens = _lens(case, I + 1)
    M = int(lens.sum())
    g = torch.Generator(device="cuda").manual_seed(M + 3 * I)
    l1 = (torch.randn(M, 2 * I, generator=g, device="cuda") * 3).to(torch.bfloat16)
    dact = torch.randn(M, I, generator=g, device="cuda").to(torch.bfloat16)
    probs = torch.rand(M, generator=g, device="cuda")
    lens_d = lens.cuda()
    grad_v = torch.full((M, 2 * I), float("nan"), dtype=torch.bfloat16, device="cuda")
    gp, *out = Q.grouped_dglu_quant_mxfp8(
        dact, l1, probs, lens_d, _offs(lens_d), act, clamp, rule=rule, grad_out=grad_v
    )
    ref_grad, ref_gp = R.dglu_ref(dact.cpu(), l1.cpu(), probs.cpu(), act, clamp)
    _ulp_close(grad_v, ref_grad, "grad_l1")
    torch.testing.assert_close(gp.cpu(), ref_gp, rtol=1e-4, atol=1e-4 * float(ref_gp.abs().max() + 1))
    _check_grouped(out, grad_v, lens, rule)


@pytest.mark.parametrize(
    "case,H,with_probs,rule",
    [
        ([0, 5, 0], 64, True, "nosat"),
        ([31, 32, 33, 1, 0, 127, 128, 129], 200, False, "even"),
        ("random", 7168, True, "nosat"),
        ("skew", 7168, False, "nosat"),
    ],
)
def test_gather(case, H, with_probs, rule):
    lens = _lens(case, H)
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
    out = Q.grouped_gather_quant_mxfp8(x, d2s, lens_d, _offs(lens_d), pp, rule=rule)
    _check_grouped(out, R.gather_rows(x.cpu(), d2s.cpu(), None if pp is None else pp.cpu()), lens, rule)


# ---- layouts: packed scales / 16x16 codes, the same kernels with other store addressing
@pytest.mark.parametrize(
    "M,K,pad,rule",
    [
        (1, 32, False, "nosat"),
        (33, 64, True, "nosat"),
        (100, 200, True, "even"),
        (160, 336, False, "nosat"),
        (300, 7168, True, "nosat"),
    ],
)
def test_layout_dense(M, K, pad, rule):
    x = _x(M, K, seed=M + 5 * K, special=not pad)  # (pad: a NaN block's codes in the K pad are 0xFF, not 0)
    KO, Mc = (_c128(K), _c128(M)) if pad else (K, M)
    lay = _max_lay(M, KO, K, Mc)
    shapes = ((M, KO), (M, -(-KO // 32)), (K, Mc), (K, -(-Mc // 32)))
    ref = _lay([_padded(t, sh) for t, sh in zip(R.quant_dual(x.cpu(), rule), shapes)], lay)
    for got, exp, what in zip(
        Q.quant_mxfp8_raw(x, rule=rule, pad=pad, layout=lay), ref, ("qr", "sr", "qc", "sc")
    ):
        _eq(got, exp, what)
    for got, exp, what in zip(
        Q.quant_mxfp8_rows(x, rule=rule, pad=pad, layout=lay), ref[:2], ("rows q", "rows s")
    ):
        _eq(got, exp, what)
    for got, exp, what in zip(
        Q.quant_mxfp8_cols(x, rule=rule, pad=pad, layout=lay), ref[2:], ("cols q", "cols s")
    ):
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
DENSE_GLU = [
    (1, 128, True, "silu", None, "nosat", False),
    (300, 96, False, "gelu", 7.0, "even", True),
    (257, 2048, True, "silu", None, "nosat", True),
    (130, 256, True, "gelu", 7.0, "nosat", False),
]


@pytest.mark.parametrize("M,I,pad,act,clamp,rule,with_probs", DENSE_GLU)
def test_glu_dense(M, I, pad, act, clamp, rule, with_probs):
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
        with_ones = Q.glu_quant_mxfp8(l1, ones, act, clamp, rule=rule, pad=pad, layout=lay)
        for got, exp, what in zip(out, with_ones, ("q_row", "s_row", "q_col", "s_col")):
            _eq(got, exp, f"no probs vs ones: {what}")


@pytest.mark.parametrize("M,I,pad,act,clamp,rule,with_probs", DENSE_GLU)
def test_dglu_dense(M, I, pad, act, clamp, rule, with_probs):
    g = torch.Generator(device="cuda").manual_seed(M + 3 * I)
    l1 = (torch.randn(M, 2 * I, generator=g, device="cuda") * 3).to(torch.bfloat16)
    dact = torch.randn(M, I, generator=g, device="cuda").to(torch.bfloat16)
    probs = torch.rand(M, generator=g, device="cuda") if with_probs else None
    lay = _max_lay(M, 2 * I, 2 * I, _c128(M) if pad else M) & ~(0 if I % 128 == 0 else PACK_ROW)
    grad_v = torch.full((M, 2 * I), float("nan"), dtype=torch.bfloat16, device="cuda")
    gp, *out = Q.dglu_quant_mxfp8(
        dact, l1, probs, act, clamp, rule=rule, pad=pad, grad_out=grad_v, layout=lay
    )
    ones = torch.ones(M, device="cuda")
    ref_grad, ref_gp = R.dglu_ref(dact.cpu(), l1.cpu(), (ones if probs is None else probs).cpu(), act, clamp)
    _ulp_close(grad_v, ref_grad, "grad_l1")
    if probs is None:
        assert gp is None
    else:
        torch.testing.assert_close(gp.cpu(), ref_gp, rtol=1e-4, atol=1e-4 * float(ref_gp.abs().max() + 1))
    _check_dense(out, grad_v, rule, pad, lay)


@pytest.mark.parametrize(
    "B,N,K,pad,rule",
    [(3, 64, 96, True, "nosat"), (2, 129, 257, False, "even"), (1, 300, 7168, True, "nosat")],
)
def test_layout_2d(B, N, K, pad, rule):
    w = _x(B, N, K, seed=B + N + K, special=not pad)
    KO, Nc = (_c128(K), _c128(N)) if pad else (K, N)
    lay = _max_lay(N, KO, K, Nc)
    out = Q.quant_mxfp8_raw_batched(w, row_2d=True, col_2d=True, rule=rule, pad=pad, layout=lay)
    shapes = ((N, KO), (N, -(-KO // 32)), (K, Nc), (K, -(-Nc // 32)))
    for b in range(B):
        ref = _lay([_padded(t, sh) for t, sh in zip(R.quant_2d(w[b].cpu(), rule), shapes)], lay)
        for got, exp, what in zip(out, ref, ("q", "s_row", "q_t", "s_t")):
            _eq(got[b], exp, f"{what}[{b}]")


@pytest.mark.parametrize(
    "kind,dim,case",
    [
        ("grouped", 64, [0, 5, 0]),
        ("grouped", 200, [31, 32, 33, 1, 0, 127, 128, 129]),
        ("grouped", 2048, "skew"),
        ("glu", 96, [0, 5, 0]),
        ("glu", 2048, "skew"),
        ("dglu", 96, [31, 32, 33, 1, 0, 127, 128, 129]),
        ("dglu", 2048, [0, 5, 0]),
        ("gather", 64, "skew"),
        ("gather", 7168, [31, 32, 33, 1, 0, 127, 128, 129]),
    ],
)
def test_layout_moe(kind, dim, case):
    """Every MoE producer's copies in the GEMM layouts (dGLU packs its row scales only for I % 128 == 0)."""
    lens = _lens(case, dim)
    M, rule = int(lens.sum()), "nosat"
    C = 2 * dim if kind == "dglu" else dim
    lay = (
        PACK_COL
        | (SHUF_ROW | SHUF_COL if C % 16 == 0 else 0)
        | (0 if kind == "dglu" and dim % 128 else PACK_ROW)
    )
    g = torch.Generator(device="cuda").manual_seed(M + dim)
    lens_d = lens.cuda()
    offs = _offs(lens_d)
    l1 = (torch.randn(M, 2 * dim, generator=g, device="cuda") * 3).to(torch.bfloat16)
    probs = torch.rand(M, generator=g, device="cuda")
    vals = torch.full((M, C), float("nan"), dtype=torch.bfloat16, device="cuda")
    if kind == "grouped":
        vals = _x(M, dim, seed=M + dim)
        out = Q.grouped_quant_mxfp8_raw(vals, lens_d, offs, rule=rule, layout=lay)
    elif kind == "glu":
        out = Q.grouped_glu_quant_mxfp8(
            l1, probs, lens_d, offs, "silu", 7.0, rule=rule, act_out=vals, layout=lay
        )
    elif kind == "dglu":
        dact = torch.randn(M, dim, generator=g, device="cuda").to(torch.bfloat16)
        _, *out = Q.grouped_dglu_quant_mxfp8(
            dact, l1, probs, lens_d, offs, "gelu", 7.0, rule=rule, grad_out=vals, layout=lay
        )
    else:
        T = max(1, M // 3)
        x = _x(T, dim, seed=dim)
        d2s = torch.randint(-1, T, (M,), generator=g, device="cuda", dtype=torch.int32)
        probs[::7] = 0
        out = Q.grouped_gather_quant_mxfp8(x, d2s, lens_d, offs, probs, rule=rule, layout=lay)
        vals = R.gather_rows(x.cpu(), d2s.cpu(), probs.cpu())
    _check_grouped(out, vals, lens, rule, lay)


def test_tail_at_mapping_end():
    """Inputs whose last byte ends their GPU mapping (a read past them faults), with K % 128 != 0: the tile load
    addresses whole 128-column rows, so the last row's partial k-tile must not come in by TDM. Every producer that
    loads by TDM (dense dual / rows / columns, 2-D blocks, grouped, GLU and dGLU grouped and dense) and the gather;
    K % 8 != 0 goes through a copy with room. The outputs are those of the same inputs in ordinary storage."""
    M, K, I = 257, 576, 96
    lens = torch.tensor([100, 0, 157], device="cuda")
    offs = _offs(lens)
    x, w, x300 = _x(M, K, seed=1), _x(3, 64, 96, seed=2), _x(130, 300, seed=3)
    g = torch.Generator(device="cuda").manual_seed(4)
    l1 = (torch.randn(M, 2 * I, generator=g, device="cuda") * 3).to(torch.bfloat16)
    dact = torch.randn(M, I, generator=g, device="cuda").to(torch.bfloat16)
    probs = torch.rand(M, generator=g, device="cuda")
    d2s = torch.arange(M, device="cuda", dtype=torch.int32).flip(0)  # the last row first
    cases = {
        "dual pad, GEMM layouts": (lambda a: Q.quant_mxfp8_raw(a, pad=True, layout=11), [x]),
        "rows": (lambda a: Q.quant_mxfp8_rows(a), [x]),
        "cols": (lambda a: Q.quant_mxfp8_cols(a, pad=True), [x]),
        "2d": (lambda a: Q.quant_mxfp8_raw_batched(a, row_2d=True, col_2d=True), [w]),
        "K % 8 != 0": (lambda a: Q.quant_mxfp8_raw(a), [x300]),
        "grouped": (lambda a: Q.grouped_quant_mxfp8_raw(a, lens, offs), [x]),
        "glu": (lambda a: Q.grouped_glu_quant_mxfp8(a, probs, lens, offs), [l1]),
        "glu dense": (lambda a: Q.glu_quant_mxfp8(a), [l1]),
        "dglu": (lambda a, b: Q.grouped_dglu_quant_mxfp8(a, b, probs, lens, offs), [dact, l1]),
        "dglu dense": (lambda a, b: Q.dglu_quant_mxfp8(a, b), [dact, l1]),
        "gather": (lambda a: Q.grouped_gather_quant_mxfp8(a, d2s, lens, offs), [x]),
    }
    for name, (fn, inputs) in cases.items():
        ref = [t for t in fn(*inputs) if t is not None]
        for i, t in enumerate(inputs):  # each input in turn at the end of its mapping
            with tensor_at_mapping_end(t.shape, t.dtype) as e:
                e.copy_(t)
                got = [u for u in fn(*[e if j == i else u for j, u in enumerate(inputs)]) if u is not None]
                for k, (a, b) in enumerate(zip(got, ref)):
                    _eq(a, b, f"{name}, input {i} at the mapping end: output {k}")
