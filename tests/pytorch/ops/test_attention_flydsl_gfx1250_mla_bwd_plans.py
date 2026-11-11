###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The two launch plans of the gfx1250 MLA backward (qk 192, v 128): the dS plan (k_dkdv96ds writes the bf16 dS^T
tiles into a scratch, k_dqds computes dQ = dS K from it) and the recompute plan (k_dkdv64 / k_dkdv, k_dqg96 / k_dqg).

The plan selection tests use meta tensors and run anywhere; the numeric tests need a gfx1250 card and compare the
plans bitwise, since both run the same arithmetic in the same order.
"""

import contextlib
import inspect

import pytest
import torch

from primus_turbo.pytorch.core.utils import is_gfx1250

needs_gfx1250 = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_gfx1250()), reason="needs a gfx1250 (MI455X) GPU"
)

D_QK, D_V = 192, 128
MLA_SCALE = 0.1352337788608801  # DeepSeek-V3: yarn mscale**2 / sqrt(192)


def _bwd():
    from primus_turbo.flydsl.attention.gfx1250_mla import bwd

    return bwd


def _meta(b, sq, skv, h):
    q = torch.empty(b, sq, h, D_QK, dtype=torch.bfloat16, device="meta")
    k = torch.empty(b, skv, h, D_QK, dtype=torch.bfloat16, device="meta")
    v = torch.empty(b, skv, h, D_V, dtype=torch.bfloat16, device="meta")
    out = torch.empty(b, sq, h, D_V, dtype=torch.bfloat16, device="meta")
    lse = torch.empty(b, h, sq, dtype=torch.float32, device="meta")
    return q, k, v, out, lse


@pytest.mark.parametrize(
    ("dims", "ds_plan"),
    [
        pytest.param((1, 4096, 4096, 256), True, id="dsv3_b2_folded"),
        pytest.param((2, 4096, 4096, 128), True, id="dsv3_b2"),
        pytest.param((1, 384, 384, 2), True, id="s384"),
        pytest.param((1, 256, 448, 2), True, id="s256_skv448"),
        pytest.param((1, 640, 704, 3), True, id="s640_skv704"),
        pytest.param((1, 320, 320, 2), False, id="s320_recompute"),
        pytest.param((1, 64, 128, 2), False, id="s64_recompute"),
    ],
)
def test_plan_selection(dims, ds_plan):
    """seqlen_q % 384 in {0, 256} takes the dS plan, other shapes the recompute plan (both behind the entry's gate:
    seqlen_kv % 64 == 0)."""
    bwd = _bwd()
    b, sq, skv, h = dims
    assert bwd.shape_unsupported_reason(b, sq, skv, h) is None
    assert (bwd.ds_plan_unsupported_reason(b, sq, skv, h) is None) == ds_plan
    q, k, v, out, lse = _meta(b, sq, skv, h)
    if ds_plan:
        launches, _ = bwd._ds_launches(out, q, k, v, out, lse, MLA_SCALE, True, None)
        names = [fn.func.__name__ for fn, _ in launches]
        assert names == ["launch_delta", "launch_dkdv96ds", "launch_dqds"]


@pytest.mark.parametrize(
    "dims",
    [
        pytest.param((1, 4096, 4096, 256), id="dsv3_b2_folded"),
        pytest.param((1, 384, 576, 2), id="s384_skv576"),
        pytest.param((2, 256, 320, 5), id="b2_s256_skv320_h5"),
        pytest.param((1, 1024, 1024, 40), id="s1024_h40"),
    ],
)
def test_ds_plan_grids_cover_every_row_and_head(dims):
    """k_dkdv96ds's blocks cover every kv row of every head, k_dqds's tiles every query of every head."""
    bwd = _bwd()
    b, sq, skv, h = dims
    q, k, v, out, lse = _meta(b, sq, skv, h)
    launches, _ = bwd._ds_launches(out, q, k, v, out, lse, MLA_SCALE, True, None)
    (_, _), (dkdv_fn, dkdv_args), (dq_fn, dq_args) = launches
    a = inspect.signature(dkdv_fn.func).bind(*dkdv_args).arguments
    assert a["HG"] * a["ngz"] == b * h
    # 96-row blocks, then one 64-row or two 32-row... tail blocks: the last block ends at or after skv
    assert (a["nblk"] - 1) * 96 < skv <= a["nblk"] * 96
    a = inspect.signature(dq_fn.func).bind(*dq_args).arguments
    assert a["HG"] * a["ngz"] == b * h
    assert a["q_off"] + 384 * a["ntile"] == sq and a["ny"] == a["ntile"] + int(a["q_off"] > 0)


# ---------------------------------------------------------------------------
# Numerics (gfx1250)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _nan_filled_allocations():
    """Buffers the host code allocates start as NaN (the dS scratch included), so an element no launch writes, or a
    scratch block read before it is written, shows up as a mismatch or a NaN."""

    def nan(alloc):
        def fn(*args, **kwargs):
            t = alloc(*args, **kwargs)
            return t.fill_(float("nan")) if t.is_floating_point() and t.device.type == "cuda" else t

        return fn

    with pytest.MonkeyPatch.context() as mp:
        for owner, name in ((torch, "empty"), (torch, "empty_like"), (torch.Tensor, "new_empty")):
            mp.setattr(owner, name, nan(getattr(owner, name)))
        yield


def _problem(b, sq, skv, h, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def r(s, d):
        return torch.randn(b, s, h, d, device="cuda", generator=g).to(torch.bfloat16)

    q, k, v, do = r(sq, D_QK), r(skv, D_QK), r(skv, D_V), r(sq, D_V)
    from primus_turbo.flydsl.attention.gfx1250_mla import fwd

    out, lse = fwd.flash_attn_mla_fwd(q, k, v, MLA_SCALE, True)
    return do, q, k, v, out, lse


def _grads(monkeypatch, problem, ds_plan):
    bwd = _bwd()
    monkeypatch.setattr(bwd, "DS_PLAN", ds_plan)
    with _nan_filled_allocations():
        grads = bwd.flash_attn_mla_bwd(*problem, MLA_SCALE, True)
    torch.cuda.synchronize()
    return grads


@needs_gfx1250
@pytest.mark.parametrize(
    "dims",
    [
        pytest.param((1, 384, 384, 2), id="s384"),
        pytest.param((1, 256, 256, 3), id="s256_head_tile_only"),
        pytest.param((2, 640, 640, 2), id="b2_s640"),
        pytest.param((1, 256, 448, 2), id="s256_skv448_tail64"),
        pytest.param((1, 384, 512, 2), id="s384_skv512_tail32"),
        pytest.param((1, 768, 832, 4), id="s768_skv832_tail64"),
        pytest.param((1, 1024, 1024, 40), id="s1024_h40_head_groups"),
        pytest.param((1, 4096, 4096, 16), id="s4096_h16"),
    ],
)
def test_ds_plan_equals_recompute_plan(monkeypatch, dims):
    bwd = _bwd()
    assert bwd.ds_plan_unsupported_reason(dims[0], dims[1], dims[2], dims[3]) is None
    problem = _problem(*dims)
    got = _grads(monkeypatch, problem, True)
    ref = _grads(monkeypatch, problem, False)
    for name, x, r in zip(("dq", "dk", "dv"), got, ref):
        assert torch.isfinite(x).all(), name
        assert torch.equal(x, r), name


@needs_gfx1250
def test_ds_plan_is_deterministic(monkeypatch):
    problem = _problem(1, 640, 640, 4, seed=3)
    a = _grads(monkeypatch, problem, True)
    b = _grads(monkeypatch, problem, True)
    assert all(torch.equal(x, y) for x, y in zip(a, b))


@needs_gfx1250
def test_scratch_oom_runs_the_recompute_plan(monkeypatch):
    """A dS scratch that cannot be allocated sends the call to the recompute plan, and no scratch at least as large is
    tried again on that device."""
    bwd = _bwd()
    problem = _problem(1, 384, 384, 2, seed=5)
    ref = _grads(monkeypatch, problem, False)
    nbytes = bwd._d.ds_layout_bytes(1, 384, 384, 2, 1)
    tried = []
    real_empty = torch.empty

    def empty(*args, **kwargs):
        size = args[0] if args else kwargs.get("size")
        if isinstance(size, tuple) and size == (nbytes // 2,) and kwargs.get("dtype") is torch.bfloat16:
            tried.append(size)
            raise torch.OutOfMemoryError("test")
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(bwd, "DS_PLAN", True)  # _grads(..., False) left it off
    monkeypatch.setattr(bwd, "_DS_OOM_BYTES", {})
    monkeypatch.setattr(torch, "empty", empty)
    got = bwd.flash_attn_mla_bwd(*problem, MLA_SCALE, True)
    got2 = bwd.flash_attn_mla_bwd(*problem, MLA_SCALE, True)
    monkeypatch.setattr(torch, "empty", real_empty)
    assert len(tried) == 1 and bwd._DS_OOM_BYTES == {problem[1].device: nbytes}
    for x, y, r in zip(got, got2, ref):
        assert torch.equal(x, r) and torch.equal(y, r)
