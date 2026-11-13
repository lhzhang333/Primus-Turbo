###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MegaMoE in MXFP8 on gfx1250, one GPU (EP1): the MX rows on the wire, the combine-only launch, and the stages
bit for bit against the routed experts' grouped-path kernels run in the Mega pool order on the same quantized
operands (the rows from bf16 by the grouped path's gather-quant, the columns by mxfp8_ref from the dispatched fp8
rows); the staged ops against an fp32 reference."""

import os
import sys

import pytest
import torch

import primus_turbo.pytorch  # noqa: F401
from primus_turbo.flydsl.utils.glu_activation import GLUActivation
from primus_turbo.pytorch.core.utils import is_gfx1250

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ref"))
import mxfp8_ref as R  # noqa: E402

pytestmark = pytest.mark.skipif(not is_gfx1250(), reason="gfx1250 kernels")

from tests.pytorch.ops.test_mega_bf16_gfx1250_fused_moe import (  # noqa: E402,F401
    _reference,
    _snr,
    ep1_group,
)

SWIGLU = GLUActivation.swiglu()  # DeepSeek-V3's activation: plain SiLU-SwiGLU


def _inputs(T, H, I, E, topk, seed, skew=False):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(T, H, generator=g, device="cuda").to(torch.bfloat16)
    scores = torch.randn(T, E, generator=g, device="cuda")
    if skew:  # hot experts and two empty ones
        scores[:, : E // 4] += 2.0
        scores[:, -2:] -= 100.0
    topk_w, topk_idx = scores.softmax(-1).topk(topk, dim=-1)
    w1 = (torch.randn(E, 2 * I, H, generator=g, device="cuda") * 2 / H**0.5).to(torch.bfloat16)
    w2 = (torch.randn(E, H, I, generator=g, device="cuda") * 2 / I**0.5).to(torch.bfloat16)
    dy = torch.randn(T, H, generator=g, device="cuda").to(torch.bfloat16)
    return x, topk_idx, topk_w, w1, w2, dy


def _eq(got, want, what):
    a, b = got.contiguous().view(torch.uint8).cpu(), want.contiguous().view(torch.uint8).cpu()
    assert a.shape == b.shape, f"{what}: shape {tuple(a.shape)} vs {tuple(b.shape)}"
    bad = (a != b).nonzero()
    assert bad.numel() == 0, f"{what}: {bad.shape[0]} bytes differ, first at {bad[0].tolist()}"


def _fold(pool, h, T):
    """The combine at EP1: token t's pool rows summed in f32 in ``recv_token_to_pool_rows`` order, rounded to bf16."""
    rows = h.recv_token_to_pool_rows[:T].long()
    acc = torch.zeros(T, pool.shape[1], dtype=torch.float32, device=pool.device)
    for k in range(rows.shape[1]):
        ok = rows[:, k] >= 0
        acc[ok] += pool[rows[ok, k]].float()
    return acc.to(torch.bfloat16)


def _gate_scatter(grad_gate, h, T, topk):
    """``grad_topk_weights``: each real pool row's gate gradient at its route."""
    rows = h.recv_token_to_pool_rows[:T].long()
    out = torch.zeros(T * topk, dtype=torch.float32, device=grad_gate.device)
    p = rows[rows >= 0]
    out[h.pool_row_to_route[p].long()] = grad_gate[p]
    return out.view(T, topk)


def _ref_cols(t, h):
    """The pool's column copy of ``t`` by mxfp8_ref from its MX rows (``t``'s dispatch): codes, packed scales. An
    expert's tokens are its whole 256-row pool segment (padding rows zero), as the kernels see the pool."""
    q, s = R.quant_rows(t.cpu())
    P = h.pool_row_to_route.numel()
    seg = torch.diff(h.num_tokens_per_expert_prefix.cpu())
    out = R.quant_grouped_fp8(q, s, h.pool_row_to_recv_token[:P].cpu(), seg)
    return out[2], R.pack_scales(out[3])


def _cols(c, P, shuffled):
    """A column copy's written part (the pool's ``P`` tokens): codes in row-major order, packed scales."""
    codes = R.unpreshuffle_b16(c[0].cpu()) if shuffled else c[0].cpu()
    return codes[:, :P], c[1].cpu()[:, : P // 128]


@pytest.mark.parametrize(
    "T, H, I, E, topk, skew",
    [
        (512, 1024, 512, 8, 4, False),
        (1024, 7168, 2048, 32, 8, True),  # DeepSeek-V3's experts (EP1 slice), hot and empty experts
    ],
)
def test_stages_vs_grouped_path_bitwise_ep1(ep1_group, T, H, I, E, topk, skew):
    from primus_turbo.flydsl.gemm.gemm_mxfp8_kernel_gfx1250 import (
        grouped_gemm_mxfp8_gfx1250,
        grouped_gemm_mxfp8_variable_k_gfx1250,
    )
    from primus_turbo.flydsl.mega.gfx1250.dispatch_prologue_kernel import DispatchHandle
    from primus_turbo.flydsl.quantization import mxfp8_quant_gfx1250 as Q
    from primus_turbo.pytorch.kernels.fused_mega_moe import fused_mega_moe_gfx1250_fp8_impl as MX

    x, topk_idx, topk_w, w1, w2, dy = _inputs(T, H, I, E, topk, T + H, skew)
    l1, dw, handle, x_col, _ = MX.stage1_forward(x, w1, ep1_group, topk_idx, topk_w)
    y, act_col = MX.stage2_forward(l1, w2, handle, topk_idx, dw, SWIGLU)
    gl_row, gl_col, grad_gate, dW2 = MX.stage2_backward(dy, l1, dw, w2, handle, act_col, SWIGLU)
    dx, gtw, dW1 = MX.stage1_backward(gl_row, gl_col, grad_gate, x_col, w1, handle, topk_idx)
    h = DispatchHandle(*handle)
    P, lens, offs = h.pool_row_to_route.numel(), h.num_tokens_per_expert, h.num_tokens_per_expert_prefix
    rmap = h.pool_row_to_recv_token[:P]  # padding rows: out of range -> zero rows, as the pool's

    def gemm(a, w):
        return grouped_gemm_mxfp8_gfx1250(a[0], a[1], w[0], w[1], offs, offs, P)

    def wgrad(a, b, **kw):
        return grouped_gemm_mxfp8_variable_k_gfx1250(a[0], a[1], b[0], b[1], offs, **kw)

    W1, W1t = MX._weights(w1)[:2], MX._weights(w1)[2:]
    W2, W2t = MX._weights(w2)[:2], MX._weights(w2)[2:]
    # forward: rows quantized from bf16 x by the grouped path's gather-quant
    xr = Q.grouped_gather_quant_mxfp8(x, rmap, lens, offs, total_m=P, layout=Q.PACK_ROW)
    xr_mega = MX._send(x, h, 0, cols=False)  # the forward's rows once more: what the dispatch delivered
    _eq(xr_mega[0][:P], xr[0][:P], "x rows")
    _eq(xr_mega[1].view(-1)[: P * H // 32], xr[1].view(-1)[: P * H // 32], "x row scales")
    r_l1 = gemm(xr, W1)
    _eq(l1, r_l1, "l1")
    act = Q.grouped_glu_quant_mxfp8(r_l1, dw, lens, offs, "silu", None, layout=Q.PACK_ROW | Q.PACK_COL | Q.SHUF_COL)
    for got, want, what in zip(_cols(act_col, P, True), _cols(act[2:4], P, True), ("act columns", "scales")):
        _eq(got, want, what)
    _eq(y, _fold(gemm(act, W2), h, T), "y")
    # the column copy of x for dW1, from the dispatched fp8 rows
    for got, want, what in zip(_cols(x_col, P, True), _ref_cols(x, h), ("x columns", "x column scales")):
        _eq(got, want, what)
    # backward
    dyr = Q.grouped_gather_quant_mxfp8(dy, rmap, lens, offs, total_m=P, layout=Q.PACK_ROW)
    gp, *g = Q.grouped_dglu_quant_mxfp8(gemm(dyr, W2t), r_l1, dw, lens, offs, "silu", None,
                                        layout=Q.PACK_ROW | Q.PACK_COL)
    _eq(grad_gate, gp, "grad_gate")
    _eq(gl_row[0][:P], g[0][:P], "grad_l1 rows")
    _eq(gl_row[1].view(-1)[: P * 2 * I // 32], g[1].view(-1)[: P * 2 * I // 32], "grad_l1 row scales")
    for got, want, what in zip(_cols(gl_col, P, False), _cols(g[2:4], P, False), ("grad_l1 columns", "scales")):
        _eq(got, want, what)
    _eq(dx, _fold(gemm(g, W1t), h, T), "dx")
    _eq(gtw, _gate_scatter(gp, h, T, topk), "grad_topk_weights")
    dy_col = MX._send(dy, h, MX._COLS_A)[2:]  # the backward's own dispatch of dy, again (same pool)
    for got, want, what in zip(_cols(dy_col, P, False), _ref_cols(dy, h), ("dy columns", "dy column scales")):
        _eq(got, want, what)
    _eq(dW2, wgrad(dy_col, (act[2], act[3])), "dW2")
    _eq(dW1, wgrad((g[2], g[3]), x_col), "dW1")
    # beta = 1 into a bf16 main_grad: rounded once, round(main_grad + the fp32 wgrad)
    main_grad = torch.randn_like(w2)
    want = (main_grad.float() + wgrad(dy_col, act_col, out_dtype=torch.float32)).to(torch.bfloat16)
    assert MX._wgrad(dy_col, act_col, h, main_grad) is None
    _eq(main_grad, want, "dW2 into main_grad")


def test_combine_only_ep1(ep1_group):
    """The combine-only launch reads only real pool rows (padding rows NaN here) and folds them in route order."""
    from primus_turbo.flydsl.mega.gfx1250.bf16.gemm_combine_kernel import combine_only_gfx1250
    from primus_turbo.flydsl.mega.gfx1250.dispatch_prologue_kernel import DispatchHandle, run_dispatch_prologue

    T, H, E, topk = 333, 1024, 8, 4
    x, topk_idx, topk_w, w1, _, _ = _inputs(T, H, 256, E, topk, 7, skew=True)
    h, _ = run_dispatch_prologue(x, w1, ep1_group, topk_idx.to(torch.int64), topk_w)
    h = DispatchHandle(*h)
    P = h.pool_row_to_route.numel()
    pool = torch.full((P, H), float("nan"), dtype=torch.bfloat16, device="cuda")
    rows = h.recv_token_to_pool_rows[:T].long()
    real = rows[rows >= 0]
    pool[real] = torch.randn(real.numel(), H, device="cuda").to(torch.bfloat16)
    gate = torch.randn(P, device="cuda")
    for _ in range(2):  # the epoch flags stay consistent call after call
        y, gtw = combine_only_gfx1250(pool, h, topk_idx.to(torch.int64), grad_gate=gate)
        _eq(y, _fold(pool, h, T), "y")
        _eq(gtw, _gate_scatter(gate, h, T, topk), "grad_topk_weights")


@pytest.mark.parametrize("fuse", [False, True])
def test_staged_op_ep1(ep1_group, fuse):
    """``fused_mega_moe_fp8_stage1/2`` on gfx1250 against the fp32 reference (the grouped MLP's 20 dB gate), and the
    wgrads into a bf16 ``main_grad`` with ``fuse_wgrad_accum_pattern``."""
    from primus_turbo.pytorch.kernels.fused_mega_moe import advance_weight_generation
    from primus_turbo.pytorch.ops.moe.fused_mega_moe_fp8 import (
        fused_mega_moe_fp8_stage1,
        fused_mega_moe_fp8_stage2,
    )

    T, H, I, E, topk = 1024, 7168, 2048, 32, 8
    x, topk_idx, topk_w, w1, w2, dy = _inputs(T, H, I, E, topk, 11)
    leaves = [t.clone().requires_grad_() for t in (x, topk_w, w1, w2)]
    pattern = "megatron" if fuse else None
    if fuse:
        for w in leaves[2:]:
            w.main_grad = torch.zeros_like(w)
            w.grad_added_to_main_grad = False
    advance_weight_generation()
    l1, dw, handle, state = fused_mega_moe_fp8_stage1(leaves[0], topk_idx, leaves[1], leaves[2], ep1_group, pattern)
    y = fused_mega_moe_fp8_stage2(
        l1, dw, handle, state, topk_idx, leaves[1], leaves[3], ep1_group, SWIGLU, fuse_wgrad_accum_pattern=pattern
    )
    y.backward(dy)
    refs = [t.detach().float().requires_grad_() for t in (x, topk_w, w1, w2)]
    ref_y = _reference(refs[0], topk_idx, *refs[1:], SWIGLU)
    ref_y.backward(dy.float())
    grads = [leaves[0].grad, leaves[1].grad] + [w.main_grad if fuse else w.grad for w in leaves[2:]]
    snrs = {"y": _snr(y, ref_y.detach())}
    snrs.update({name: _snr(got, want.grad) for name, got, want in zip(("dx", "d topk_w", "dW1", "dW2"), grads, refs)})
    print(f"\n[MX Mega vs fp32, fuse={fuse}] " + ", ".join(f"{k} {v:.1f} dB" for k, v in snrs.items()))
    for name, snr in snrs.items():
        assert snr > 20, f"{name} SNR {snr:.1f} dB"
    if fuse:
        assert all(w.grad_added_to_main_grad for w in leaves[2:])
