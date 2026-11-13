###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The two launch plans of the gfx1250 MLA backward (qk 192, v 128): the dS plan (k_dkdv96ds writes the bf16 dS^T
tiles into a scratch, k_dqds computes dQ = dS K from it, head chunk by head chunk on two streams within a scratch
budget) and the recompute plan (k_dkdv64 / k_dkdv, k_dqg96 / k_dqg).

The plan selection tests use meta tensors and run anywhere; the numeric tests need a gfx1250 card and compare the
plans bitwise, since both run the same arithmetic in the same order.
"""

import contextlib
import inspect

import pytest
import torch

import primus_turbo.common.envs as envs
from primus_turbo.pytorch.core.utils import is_gfx1250

needs_gfx1250 = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_gfx1250()), reason="needs a gfx1250 (MI455X) GPU"
)

D_QK, D_V = 192, 128
MLA_SCALE = 0.1352337788608801  # DeepSeek-V3: yarn mscale**2 / sqrt(192)


_BUDGET = "PRIMUS_TURBO_MLA_BWD_DS_SCRATCH_MB"


def _bwd():
    from primus_turbo.flydsl.attention.gfx1250_mla import bwd

    return bwd


def _set_budget(monkeypatch, mib):
    monkeypatch.setenv(_BUDGET, str(mib))
    envs.reset_cache(_BUDGET)


def _budget_for(b, sq, skv, h, hc):
    """The budget (MiB) whose first dS chunking is chunks of hc heads, for problems too small for the busy-chunk rule."""
    bwd = _bwd()
    nbytes = bwd.ds_scratch_bytes(b, sq, skv, 1) * hc * (1 if hc == h else 2)
    mib = -(-nbytes // (1 << 20))
    assert bwd.ds_chunkings(b, sq, skv, h, mib << 20)[0][0] == hc
    return mib


def _chunks(launches):
    """[(H0, HC)] of the dS plan's k_dqds launches."""
    out = []
    for fn, args in launches[1:]:
        a = inspect.signature(fn.func).bind(*args).arguments
        if fn.func.__name__ == "launch_dqds":
            out.append((a["H0"], a["HC"]))
    return out


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
        pytest.param((1, 512, 512, 2), True, id="s512_two_head_tiles"),
        pytest.param((1, 2048, 2048, 4), True, id="s2048"),
        pytest.param((1, 8192, 8192, 128), True, id="s8192_h128"),
        pytest.param((1, 128, 256, 2), True, id="s128_partial_head_tile"),
        pytest.param((1, 320, 320, 2), True, id="s320"),
        pytest.param((1, 64, 128, 2), True, id="s64"),
        pytest.param((1, 1984, 1984, 8), True, id="s1984"),
        pytest.param((1, 16448, 16448, 128), True, id="past_2_21_bhs"),
    ],
)
def test_plan_selection(dims, ds_plan):
    """Every shape the entry's gate takes (seqlen_q and seqlen_kv multiples of 64) takes the dS plan: the dQ head
    tiles cover any multiple of 32, the lowest one starting below query 0 where needed."""
    bwd = _bwd()
    b, sq, skv, h = dims
    assert bwd.shape_unsupported_reason(b, sq, skv, h) is None
    assert (bwd.ds_plan_unsupported_reason(b, sq, skv, h) is None) == ds_plan
    q, k, v, out, lse = _meta(b, sq, skv, h)
    if ds_plan:
        launches, _ = bwd._ds_launches(out, q, k, v, out, lse, MLA_SCALE, True, None)
        names = [fn.func.__name__ for fn, _ in launches]
        nchunks = (len(names) - 1) // 2
        assert nchunks >= 1 and names == ["launch_delta"] + ["launch_dkdv96ds", "launch_dqds"] * nchunks


@pytest.mark.parametrize(
    "dims",
    [
        pytest.param((1, 4096, 4096, 256), id="dsv3_b2_folded"),
        pytest.param((1, 384, 576, 2), id="s384_skv576"),
        pytest.param((2, 256, 320, 5), id="b2_s256_skv320_h5"),
        pytest.param((1, 1024, 1024, 40), id="s1024_h40"),
        pytest.param((1, 512, 512, 2), id="s512_two_head_tiles"),
        pytest.param((2, 896, 1024, 3), id="b2_s896_skv1024_h3"),
        pytest.param((1, 64, 64, 2), id="s64_partial_head_tile"),
        pytest.param((1, 320, 576, 2), id="s320_head_tiles_partial_and_whole"),
    ],
)
def test_ds_plan_grids_cover_every_row_and_head(dims):
    """The head chunks partition the heads; per chunk, k_dkdv96ds's blocks cover every kv row of every head of the
    chunk and k_dqds's tiles every query, both over the same heads and batches."""
    bwd = _bwd()
    b, sq, skv, h = dims
    q, k, v, out, lse = _meta(b, sq, skv, h)
    launches, _ = bwd._ds_launches(out, q, k, v, out, lse, MLA_SCALE, True, None)
    pairs = list(zip(launches[1::2], launches[2::2]))
    assert len(launches) == 1 + 2 * len(pairs)
    h_next = 0
    for (dkdv_fn, dkdv_args), (dq_fn, dq_args) in pairs:
        a = inspect.signature(dkdv_fn.func).bind(*dkdv_args).arguments
        assert a["H0"] == h_next and a["HG"] * a["ngz"] == b * a["HC"]
        # 96-row blocks, then one 64-row or two 32-row... tail blocks: the last block ends at or after skv
        assert (a["nblk"] - 1) * 96 < skv <= a["nblk"] * 96
        d = inspect.signature(dq_fn.func).bind(*dq_args).arguments
        assert (d["H0"], d["HC"]) == (a["H0"], a["HC"]) and d["HG"] * d["ngz"] == b * d["HC"]
        # 384-query tiles over [q_off, sq), 256-query head tiles from q_off down past query 0
        nhead = d["ny"] - d["ntile"]
        assert d["q_off"] + 384 * d["ntile"] == sq and 0 <= d["q_off"] <= 256 * nhead < d["q_off"] + 256
        h_next += a["HC"]
    assert h_next == h


@pytest.mark.parametrize(
    ("dims", "first_hc"),
    [
        # the most chunks whose k_dqds keeps 2 workgroups per CU, within the default budget
        pytest.param((1, 4096, 4096, 256), 64, id="dsv3_b2_folded"),
        pytest.param((2, 4096, 4096, 128), 32, id="dsv3_b2"),
        pytest.param((1, 8192, 8192, 128), 32, id="s8192_h128"),
        # too few workgroups for two busy chunks: the fewest chunks within the budget
        pytest.param((1, 1024, 1024, 16), 16, id="small_whole"),
        pytest.param((1, 32768, 32768, 64), 2, id="s32768_h64_budget"),
    ],
)
def test_ds_chunkings(dims, first_hc):
    bwd = _bwd()
    b, sq, skv, h = dims
    budget = envs.PRIMUS_TURBO_MLA_BWD_DS_SCRATCH_MB << 20
    tries = bwd.ds_chunkings(b, sq, skv, h, budget)
    per_head = bwd.ds_scratch_bytes(b, sq, skv, 1)
    assert tries[0][0] == first_hc
    for hc, nbytes in tries:
        assert nbytes == per_head * hc * (1 if hc == h else 2) <= budget
    # then smaller and smaller chunks, for a scratch that cannot be allocated, down to one head
    assert [hc for hc, _ in tries] == sorted({hc for hc, _ in tries}, reverse=True) and tries[-1][0] == 1


def test_ds_chunkings_round_up_uneven_heads():
    bwd = _bwd()
    assert [hc for hc, _ in bwd.ds_chunkings(1, 1024, 1024, 40, 1 << 40)] == [40, 20, 10, 5, 3, 2, 1]
    assert bwd.ds_chunkings(1, 4096, 4096, 256, 0) == []


def test_zero_budget_runs_the_recompute_plan(monkeypatch):
    bwd = _bwd()
    _set_budget(monkeypatch, 0)
    q, k, v, out, lse = _meta(1, 384, 384, 2)
    assert bwd._ds_launches(out, q, k, v, out, lse, MLA_SCALE, True, None) is None


def test_ds_plan_allocates_its_scratch_last(monkeypatch):
    """The dS scratch is the plan's last allocation: when it fits, the row constants and gradients (which the recompute
    plan needs too) already did, so the backward falls back instead of running out of memory after the scratch."""
    bwd = _bwd()
    b, sq, skv, h = 1, 384, 384, 2
    q, k, v, out, lse = _meta(b, sq, skv, h)
    sizes = []

    def record(alloc):
        def fn(*args, **kwargs):
            t = alloc(*args, **kwargs)
            sizes.append(t.numel())
            return t

        return fn

    for owner, name in ((torch, "empty"), (torch, "empty_like"), (torch.Tensor, "new_empty")):
        monkeypatch.setattr(owner, name, record(getattr(owner, name)))
    bwd._ds_launches(out, q, k, v, out, lse, MLA_SCALE, True, None)
    scratch = bwd.ds_chunkings(b, sq, skv, h, envs.PRIMUS_TURBO_MLA_BWD_DS_SCRATCH_MB << 20)[0][1] // 2
    assert sizes[-1] == scratch and sizes.count(scratch) == 1, sizes


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
        pytest.param((1, 4096, 4096, 256), id="dsv3_b2_folded"),
        pytest.param((1, 512, 512, 2), id="s512_two_head_tiles_only"),
        pytest.param((1, 896, 960, 2), id="s896_skv960_two_head_tiles"),
        pytest.param((1, 1664, 1664, 3), id="s1664_h3"),
        pytest.param((2, 2048, 2048, 4), id="b2_s2048"),
        pytest.param((1, 8192, 8192, 16), id="s8192_h16"),
        pytest.param((1, 64, 64, 2), id="s64_partial_head_tile"),
        pytest.param((1, 192, 448, 2), id="s192_skv448"),
        pytest.param((1, 64, 1024, 2), id="s64_skv1024_dead_pairs_unmasked"),
        pytest.param((2, 576, 576, 4), id="b2_s576"),
        pytest.param((1, 4160, 4160, 16), id="s4160_h16"),
        pytest.param((1, 16448, 16448, 128), id="s16448_h128_past_2_21_bhs"),
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
    nbytes = bwd.ds_scratch_bytes(1, 384, 384, 2)
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


@needs_gfx1250
@pytest.mark.parametrize(
    ("dims", "hc", "busy_workgroups"),
    [
        # two chunks of half the heads take the memory of one: only the busy-chunk rule picks them
        pytest.param((1, 1024, 1024, 40), 20, 60, id="h40_two_chunks"),
        pytest.param((1, 1024, 1024, 40), 10, None, id="h40_four_chunks"),
        pytest.param((1, 1024, 1024, 40), 3, None, id="h40_uneven_chunks"),
        pytest.param((2, 2048, 2048, 8), 2, None, id="b2_s2048_four_chunks"),
        pytest.param((1, 4096, 4096, 256), 64, None, id="dsv3_b2_folded_default"),
    ],
)
def test_chunked_ds_plan_equals_recompute_plan(monkeypatch, dims, hc, busy_workgroups):
    """Head chunks alternating between two streams, each stream's chunks on its own scratch, give the recompute
    plan's bits; the DeepSeek-V3 shape runs four chunks by default."""
    bwd = _bwd()
    b, sq, skv, h = dims
    if busy_workgroups is not None:
        monkeypatch.setattr(bwd, "_CHUNK_MIN_DQ_WORKGROUPS", busy_workgroups)
    elif hc != bwd.ds_chunkings(b, sq, skv, h, envs.PRIMUS_TURBO_MLA_BWD_DS_SCRATCH_MB << 20)[0][0]:
        _set_budget(monkeypatch, _budget_for(b, sq, skv, h, hc))
    assert bwd.ds_chunkings(b, sq, skv, h, envs.PRIMUS_TURBO_MLA_BWD_DS_SCRATCH_MB << 20)[0][0] == hc
    problem = _problem(*dims)
    got = _grads(monkeypatch, problem, True)
    ref = _grads(monkeypatch, problem, False)
    for name, x, r in zip(("dq", "dk", "dv"), got, ref):
        assert torch.isfinite(x).all(), name
        assert torch.equal(x, r), name


@needs_gfx1250
def test_chunked_ds_plan_under_cuda_graphs(monkeypatch):
    """A captured chunked backward (its side stream forks from the capture stream and joins it again) replays to the
    eager bits."""
    bwd = _bwd()
    dims = (1, 1024, 1024, 40)
    _set_budget(monkeypatch, _budget_for(*dims, 10))
    monkeypatch.setattr(bwd, "DS_PLAN", True)
    problem = _problem(*dims)
    eager = bwd.flash_attn_mla_bwd(*problem, MLA_SCALE, True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = bwd.flash_attn_mla_bwd(*problem, MLA_SCALE, True)
    graph.replay()
    torch.cuda.synchronize()
    assert all(torch.equal(x, y) for x, y in zip(got, eager))


@needs_gfx1250
def test_scratch_oom_tries_smaller_chunks(monkeypatch):
    """A dS scratch that cannot be allocated sends the call to smaller head chunks (the recompute plan only when none
    fits), and that size is not tried again on the device."""
    bwd = _bwd()
    dims = (1, 1024, 1024, 8)
    problem = _problem(*dims, seed=7)
    ref = _grads(monkeypatch, problem, False)
    tries = bwd.ds_chunkings(*dims, envs.PRIMUS_TURBO_MLA_BWD_DS_SCRATCH_MB << 20)
    first = tries[0][1]
    assert tries[0][0] == 8 and any(nbytes < first for _, nbytes in tries)
    real_empty = torch.empty

    def empty(*args, **kwargs):
        size = args[0] if args else kwargs.get("size")
        if size == (first // 2,) and kwargs.get("dtype") is torch.bfloat16:
            raise torch.OutOfMemoryError("test")
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(bwd, "DS_PLAN", True)
    monkeypatch.setattr(bwd, "_DS_OOM_BYTES", {})
    monkeypatch.setattr(torch, "empty", empty)
    got = bwd.flash_attn_mla_bwd(*problem, MLA_SCALE, True)
    launches, _ = bwd._ds_launches(*problem, MLA_SCALE, True, torch.cuda.current_stream())
    monkeypatch.setattr(torch, "empty", real_empty)
    torch.cuda.synchronize()
    assert bwd._DS_OOM_BYTES == {problem[1].device: first} and len(_chunks(launches)) > 1
    assert all(torch.equal(x, r) for x, r in zip(got, ref))
