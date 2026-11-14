###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The launch plan of the gfx1250 MLA backward (qk 192, v 128): k_delta_bshd, then k_dkdv96ds (dK, dV and the bf16
dS^T tiles into a scratch) and k_dqds (dQ = dS K from it) per head chunk, the chunks alternating between two streams
within the scratch budget PRIMUS_TURBO_MLA_BWD_DS_SCRATCH_MB.

The launch and chunking tests use meta tensors and run anywhere. The numeric tests need a gfx1250 card and compare
head chunkings with the whole scratch bitwise: every head's arithmetic is the same however the heads are chunked.
test_attention_flydsl_gfx1250_mla checks the gradients against an fp32 reference, and
test_attention_hipkittens_gfx1250_mla against the HipKittens kernels, bitwise.
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


@pytest.fixture(autouse=True)
def _default_budget(monkeypatch):
    """Every test starts from the default scratch budget, whatever the environment sets."""
    monkeypatch.delenv(_BUDGET, raising=False)
    envs.reset_cache(_BUDGET)


def _set_budget(monkeypatch, mib):
    monkeypatch.setenv(_BUDGET, str(mib))
    envs.reset_cache(_BUDGET)


def _force_chunks(monkeypatch, b, sq, skv, h, hc, streams=None):
    """Make the backward run chunks of hc heads: all heads in one scratch, or chunks on two streams (streams=1: one
    stream). The choice among chunkings is test_ds_chunkings'."""
    bwd = _bwd()
    streams = streams or (1 if hc == h else 2)
    plan = [(hc, streams, bwd.ds_scratch_bytes(b, sq, skv, 1) * hc * streams)]
    monkeypatch.setattr(bwd, "ds_chunkings", lambda *args: plan)


def _chunks(launches):
    """[(H0, HC)] of the k_dqds launches."""
    out = []
    for fn, args in launches[1:]:
        if fn.func.__name__ == "launch_dqds":
            a = inspect.signature(fn.func).bind(*args).arguments
            out.append((a["H0"], a["HC"]))
    return out


def _meta(b, sq, skv, h):
    q = torch.empty(b, sq, h, D_QK, dtype=torch.bfloat16, device="meta")
    k = torch.empty(b, skv, h, D_QK, dtype=torch.bfloat16, device="meta")
    v = torch.empty(b, skv, h, D_V, dtype=torch.bfloat16, device="meta")
    out = torch.empty(b, sq, h, D_V, dtype=torch.bfloat16, device="meta")
    lse = torch.empty(b, h, sq, dtype=torch.float32, device="meta")
    return q, k, v, out, lse


# ---------------------------------------------------------------------------
# Launches and chunking (meta tensors)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dims",
    [
        pytest.param((1, 4096, 4096, 256), id="dsv3_b2_folded"),
        pytest.param((2, 4096, 4096, 128), id="dsv3_b2"),
        pytest.param((1, 384, 384, 2), id="s384"),
        pytest.param((1, 256, 448, 2), id="s256_skv448"),
        pytest.param((1, 512, 512, 2), id="s512_two_head_tiles"),
        pytest.param((1, 2048, 2048, 4), id="s2048"),
        pytest.param((1, 8192, 8192, 128), id="s8192_h128"),
        pytest.param((1, 128, 256, 2), id="s128_partial_head_tile"),
        pytest.param((1, 320, 320, 2), id="s320"),
        pytest.param((1, 64, 128, 2), id="s64"),
        pytest.param((1, 1984, 1984, 8), id="s1984"),
        pytest.param((1, 16448, 16448, 128), id="past_2_21_bhs"),
    ],
)
def test_launch_sequence(dims):
    """Every shape the gate takes runs k_delta_bshd, then k_dkdv96ds and k_dqds per head chunk."""
    bwd = _bwd()
    b, sq, skv, h = dims
    assert bwd.shape_unsupported_reason(b, sq, skv, h) is None
    q, k, v, out, lse = _meta(b, sq, skv, h)
    launches, _ = bwd._launches(out, q, k, v, out, lse, MLA_SCALE, True, None)
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
def test_grids_cover_every_row_and_head(dims):
    """The head chunks partition the heads; per chunk, k_dkdv96ds's blocks cover every kv row of every head of the
    chunk and k_dqds's tiles every query, both over the same heads and batches."""
    bwd = _bwd()
    b, sq, skv, h = dims
    q, k, v, out, lse = _meta(b, sq, skv, h)
    launches, _ = bwd._launches(out, q, k, v, out, lse, MLA_SCALE, True, None)
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
        # one head keeps two workgroups per CU busy: one-head chunks, still on two streams
        pytest.param((96, 2048, 2048, 16), 1, id="b96_one_head_chunks"),
    ],
)
def test_ds_chunkings(dims, first_hc):
    bwd = _bwd()
    b, sq, skv, h = dims
    budget = envs.PRIMUS_TURBO_MLA_BWD_DS_SCRATCH_MB << 20
    tries = bwd.ds_chunkings(b, sq, skv, h, budget)
    per_head = bwd.ds_scratch_bytes(b, sq, skv, 1)
    assert tries[0][0] == first_hc
    for hc, streams, nbytes in tries:
        # one stream for all heads at once and for the last resort, one head at a time; two for chunks
        assert (streams == 1) == (hc == h or (hc, nbytes) == (1, per_head))
        assert nbytes == per_head * hc * streams <= budget
    # then smaller and smaller scratches, for one that cannot be allocated, down to one head on one stream
    assert [x[2] for x in tries] == sorted((x[2] for x in tries), reverse=True) and tries[-1] == (
        1,
        1,
        per_head,
    )


def test_ds_chunkings_round_up_uneven_heads():
    bwd = _bwd()
    per_head = bwd.ds_scratch_bytes(1, 1024, 1024, 1)
    tries = bwd.ds_chunkings(1, 1024, 1024, 40, 1 << 40)
    # (20 heads on two streams take the scratch of all 40 on one: no retry at the same size)
    assert [(hc, streams) for hc, streams, _ in tries] == [
        (40, 1),
        (10, 2),
        (5, 2),
        (3, 2),
        (2, 2),
        (1, 2),
        (1, 1),
    ]
    # below one head's scratch (0 included) only one head at a time is left
    assert bwd.ds_chunkings(1, 1024, 1024, 40, 0) == [(1, 1, per_head)]


def test_ds_chunkings_fall_back_to_smaller_scratch_only():
    """Odd head counts: the busy two-stream chunks can take more scratch than all heads at once; after them the
    backward still tries every smaller scratch, all heads at once included, largest first."""
    bwd = _bwd()
    per_head = bwd.ds_scratch_bytes(1, 1024, 1024, 1)
    tries = bwd.ds_chunkings(1, 1024, 1024, 3, 1 << 40, min_dq_workgroups=6)
    assert tries == [(2, 2, 4 * per_head), (3, 1, 3 * per_head), (1, 2, 2 * per_head), (1, 1, per_head)]


def test_zero_budget_runs_one_head_at_a_time(monkeypatch):
    bwd = _bwd()
    _set_budget(monkeypatch, 0)
    q, k, v, out, lse = _meta(1, 384, 384, 3)
    launches, _ = bwd._launches(out, q, k, v, out, lse, MLA_SCALE, True, None)
    assert _chunks(launches) == [(0, 1), (1, 1), (2, 1)]


def test_scratch_is_the_last_allocation(monkeypatch):
    """The dS scratch is the backward's last allocation: when it fits, the row constants and gradients already did,
    so a scratch that fits never leaves the backward short of memory."""
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
    bwd._launches(out, q, k, v, out, lse, MLA_SCALE, True, None)
    scratch = bwd.ds_chunkings(b, sq, skv, h, envs.PRIMUS_TURBO_MLA_BWD_DS_SCRATCH_MB << 20)[0][2] // 2
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


def _grads(problem):
    with _nan_filled_allocations():
        grads = _bwd().flash_attn_mla_bwd(*problem, MLA_SCALE, True)
    torch.cuda.synchronize()
    return grads


def _whole(monkeypatch, problem):
    """The gradients with every head in one scratch."""
    b, sq, h, _ = problem[1].shape
    with monkeypatch.context() as mp:
        _force_chunks(mp, b, sq, problem[2].shape[1], h, h)
        return _grads(problem)


@needs_gfx1250
@pytest.mark.parametrize(
    ("dims", "hc"),
    [
        pytest.param((1, 384, 384, 2), 1, id="s384"),
        pytest.param((1, 384, 384, 3), -1, id="s384_one_head_at_a_time"),
        pytest.param((1, 256, 256, 3), 1, id="s256_head_tile_only"),
        pytest.param((2, 640, 640, 2), 1, id="b2_s640"),
        pytest.param((1, 256, 448, 2), 1, id="s256_skv448_tail64"),
        pytest.param((1, 384, 512, 2), 1, id="s384_skv512_tail32"),
        pytest.param((1, 768, 832, 4), 2, id="s768_skv832_tail64"),
        pytest.param((1, 1024, 1024, 40), 10, id="s1024_h40_four_chunks"),
        pytest.param((1, 1024, 1024, 40), 3, id="s1024_h40_uneven_chunks"),
        pytest.param((2, 2048, 2048, 8), 2, id="b2_s2048"),
        pytest.param((1, 512, 512, 2), 1, id="s512_two_head_tiles_only"),
        pytest.param((1, 896, 960, 2), 1, id="s896_skv960_two_head_tiles"),
        pytest.param((1, 64, 64, 2), 1, id="s64_partial_head_tile"),
        pytest.param((1, 192, 448, 2), 1, id="s192_skv448"),
        pytest.param((1, 64, 1024, 2), 1, id="s64_skv1024_dead_pairs_unmasked"),
        pytest.param((2, 576, 576, 4), 2, id="b2_s576"),
        pytest.param((1, 4160, 4160, 16), 4, id="s4160_h16"),
        pytest.param((1, 8192, 8192, 16), 4, id="s8192_h16"),
        pytest.param((1, 16448, 16448, 128), 16, id="s16448_h128_past_2_21_bhs"),
    ],
)
def test_head_chunks_equal_the_whole_scratch(monkeypatch, dims, hc):
    """Head chunks alternating between two streams, each stream's chunks on its own scratch, give the bits of all
    heads in one scratch, and write every element (NaN-filled buffers)."""
    problem = _problem(*dims)
    ref = _whole(monkeypatch, problem)
    _force_chunks(
        monkeypatch, *dims, abs(hc), streams=1 if hc < 0 else None
    )  # -1: one head at a time, one stream
    got = _grads(problem)
    for name, x, r in zip(("dq", "dk", "dv"), got, ref):
        assert torch.isfinite(x).all() and torch.isfinite(r).all(), name
        assert torch.equal(x, r), name


@needs_gfx1250
@pytest.mark.parametrize(
    ("dims", "hc", "busy_workgroups"),
    [
        # two chunks of half the heads take the memory of one: only the busy-chunk rule picks them
        pytest.param((1, 1024, 1024, 40), 20, 60, id="h40_two_chunks"),
        pytest.param((1, 4096, 4096, 256), 64, None, id="dsv3_b2_folded_default"),
    ],
)
def test_default_chunking_equals_the_whole_scratch(monkeypatch, dims, hc, busy_workgroups):
    bwd = _bwd()
    b, sq, skv, h = dims
    problem = _problem(*dims)
    ref = _whole(monkeypatch, problem)
    if busy_workgroups is not None:
        monkeypatch.setattr(bwd, "_CHUNK_MIN_DQ_WORKGROUPS", busy_workgroups)
    assert bwd.ds_chunkings(b, sq, skv, h, envs.PRIMUS_TURBO_MLA_BWD_DS_SCRATCH_MB << 20)[0][0] == hc
    assert all(torch.equal(x, r) for x, r in zip(_grads(problem), ref))


@needs_gfx1250
def test_backward_is_deterministic():
    problem = _problem(1, 640, 640, 4, seed=3)
    a = _grads(problem)
    b = _grads(problem)
    assert all(torch.equal(x, y) for x, y in zip(a, b))


@needs_gfx1250
def test_chunked_backward_under_cuda_graphs(monkeypatch):
    """A captured chunked backward (its side stream forks from the capture stream and joins it again) replays to the
    eager bits."""
    bwd = _bwd()
    dims = (1, 1024, 1024, 40)
    _force_chunks(monkeypatch, *dims, 10)
    problem = _problem(*dims)
    eager = bwd.flash_attn_mla_bwd(*problem, MLA_SCALE, True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = bwd.flash_attn_mla_bwd(*problem, MLA_SCALE, True)
    graph.replay()
    torch.cuda.synchronize()
    assert all(torch.equal(x, y) for x, y in zip(got, eager))


@needs_gfx1250
def test_side_stream_per_caller_stream(monkeypatch):
    """Chunked backwards called on two streams run their chunks on two side streams of their own, and match."""
    bwd = _bwd()
    dims = (1, 1024, 1024, 8)
    _force_chunks(monkeypatch, *dims, 4)
    problem = _problem(*dims, seed=11)
    monkeypatch.setattr(bwd, "_SIDE_STREAMS", {})
    callers = [torch.cuda.Stream(), torch.cuda.Stream()]
    grads = []
    for s in callers:
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            grads.append(bwd.flash_attn_mla_bwd(*problem, MLA_SCALE, True))
    torch.cuda.synchronize()
    assert sorted(bwd._SIDE_STREAMS) == sorted((torch.cuda.current_device(), s.cuda_stream) for s in callers)
    assert all(torch.equal(x, y) for x, y in zip(*grads))


@needs_gfx1250
def test_scratch_oom_tries_smaller_chunks(monkeypatch):
    """A dS scratch that cannot be allocated sends the call to smaller head chunks, and that size is not tried again
    on the device."""
    bwd = _bwd()
    dims = (1, 1024, 1024, 8)
    problem = _problem(*dims, seed=7)
    ref = _grads(problem)
    tries = bwd.ds_chunkings(*dims, envs.PRIMUS_TURBO_MLA_BWD_DS_SCRATCH_MB << 20)
    first = tries[0][2]
    assert tries[0][0] == 8 and any(nbytes < first for _, _, nbytes in tries)
    real_empty = torch.empty

    def empty(*args, **kwargs):
        size = args[0] if args else kwargs.get("size")
        if size == (first // 2,) and kwargs.get("dtype") is torch.bfloat16:
            raise torch.OutOfMemoryError("test")
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(bwd, "_DS_OOM_BYTES", {})
    monkeypatch.setattr(torch, "empty", empty)
    got = bwd.flash_attn_mla_bwd(*problem, MLA_SCALE, True)
    launches, _ = bwd._launches(*problem, MLA_SCALE, True, torch.cuda.current_stream())
    monkeypatch.setattr(torch, "empty", real_empty)
    torch.cuda.synchronize()
    assert bwd._DS_OOM_BYTES == {problem[1].device: first} and len(_chunks(launches)) > 1
    assert all(torch.equal(x, r) for x, r in zip(got, ref))


@needs_gfx1250
def test_scratch_oom_for_one_head_raises(monkeypatch):
    """With no memory even for one head's scratch the backward raises torch.OutOfMemoryError, saying so."""
    bwd = _bwd()
    problem = _problem(1, 384, 384, 2, seed=5)
    real_empty = torch.empty

    def empty(*args, **kwargs):
        size = args[0] if args else kwargs.get("size")
        if isinstance(size, tuple) and len(size) == 1 and kwargs.get("dtype") is torch.bfloat16:
            raise torch.OutOfMemoryError("test")
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(bwd, "_DS_OOM_BYTES", {})
    monkeypatch.setattr(torch, "empty", empty)
    with pytest.raises(torch.OutOfMemoryError, match="even one head's dS scratch"):
        bwd.flash_attn_mla_bwd(*problem, MLA_SCALE, True)
