###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""HipKittens attention on gfx1250 (MI455X), head dim 128: ``attention_hipkittens_gfx1250_impl`` and the
HIPKITTENS backend there, built from csrc/kernels/attention/hipkittens/gfx1250.

The pins, the build script, the gate, the routing and the launch plans run on meta / CPU tensors. The numeric tests
need a gfx1250 card, the built code objects and the FlyDSL gfx1250 kernels (the fallback); their fp32 reference
runs on the CPU.
"""

import contextlib
import importlib.util
import json
import os
import struct
from pathlib import Path

import pytest
import torch

from primus_turbo.pytorch.core.backend import (
    BackendType,
    GlobalBackendManager,
    PrecisionType,
)
from primus_turbo.pytorch.core.utils import is_gfx1250
from primus_turbo.pytorch.kernels.attention import (
    attention_hipkittens_gfx1250_impl as hk,
)
from primus_turbo.pytorch.kernels.attention import attention_impl
from primus_turbo.pytorch.kernels.attention.attention_flydsl_gfx1250_impl import (
    _interface as _flydsl_interface,
)
from primus_turbo.pytorch.kernels.attention.attention_impl import (
    resolve_flash_attn_backend,
)
from primus_turbo.pytorch.ops import flash_attn_func
from tests.pytorch.ops.test_attention_flydsl_gfx1250 import (
    _FLYDSL_REASON,
    _check,
    _inputs,
    _meta,
    _reference,
    needs_flydsl,
)
from tests.pytorch.test_utils import compute_snr

KERNEL_DIR = Path(__file__).resolve().parents[3] / "csrc" / "kernels" / "attention" / "hipkittens" / "gfx1250"

needs_hk = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_gfx1250())
    or _FLYDSL_REASON is not None
    or hk.hk_gfx1250_unavailable_reason() is not None,
    reason="needs a gfx1250 (MI455X) GPU, the HipKittens code objects and the FlyDSL gfx1250 kernels",
)


def _hsaco():
    """The code-object build script (standard library only, loaded by path like setup.py does)."""
    spec = importlib.util.spec_from_file_location("_hk_gfx1250_hsaco", KERNEL_DIR / "hsaco.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _LogRecorder:
    """Stands in for the adapter's logger: records the warnings."""

    def __init__(self):
        self.warnings = []

    def warning(self, msg, *args, **kwargs):
        self.warnings.append(msg)


@contextlib.contextmanager
def _pinned(backend):
    """Pin the attention backend, so a refusal raises instead of quietly running another one."""
    GlobalBackendManager.set_attn_backend(backend, PrecisionType.BF16_FP16_FP32)
    try:
        yield
    finally:
        GlobalBackendManager.set_attn_backend(None, PrecisionType.BF16_FP16_FP32)


def _run(q, k, v, dout, causal=True, backend=BackendType.HIPKITTENS, **kwargs):
    q, k, v = (t.to("cuda").requires_grad_() for t in (q, k, v))
    with _pinned(backend):
        out, lse = flash_attn_func(q, k, v, causal=causal, return_lse=True, **kwargs)
    out.backward(dout.to("cuda"))
    return out, lse, q.grad, k.grad, v.grad


# ---------------------------------------------------------------------------
# Code objects and their build
# ---------------------------------------------------------------------------


def test_pins_and_kernels_match_the_manifest_and_sources():
    manifest = json.loads((KERNEL_DIR / "manifest.json").read_text())["kernels"]
    assert set(manifest) == set(hk._CODE_OBJECTS)
    for name, (pin, kernels) in hk._CODE_OBJECTS.items():
        assert manifest[name]["hsaco_sha256"] == pin, name
        src = (KERNEL_DIR / manifest[name]["src"]).read_text()
        for kernel in kernels:
            assert f"void {kernel}(" in src, kernel


@pytest.mark.parametrize(
    ("kernel", "size"),
    [("k_attn_fwd", 216), ("k_attn_bwd_prep", 88), ("k_attn_bwd_main", 192), ("k_attn_bwd_dq", 48)],
)
def test_argument_layouts_have_the_kernarg_sizes(kernel, size):
    """The packed arguments have the kernarg segment size of the kernel (k_attn_fwd: sizeof(KArgs); the
    backward kernels: the offsets documented in k_attn_bwd.hip)."""
    layouts = {k: layout for _, kernels in hk._CODE_OBJECTS.values() for k, (_, layout) in kernels.items()}
    assert struct.calcsize(layouts[kernel]) == size


def test_build_without_the_pinned_toolchain_installs_nothing(monkeypatch, tmp_path):
    """Another compiler cannot reproduce the pins: nothing is compiled, an unpinned file left behind is removed,
    and build() reports it instead of raising (it must never fail an install)."""
    hsaco = _hsaco()
    (tmp_path / "k_attn_fwd.hsaco").write_bytes(b"stale")
    monkeypatch.setattr(hsaco, "find_compiler", lambda cxx=None: "/usr/bin/true")
    monkeypatch.setattr(hsaco, "compiler_version", lambda cxx: "some other clang")
    log = []
    assert hsaco.build(bin_dir=tmp_path, log=log.append) is False
    assert not list(tmp_path.iterdir())
    assert any("not the pinned toolchain" in m for m in log)
    assert hsaco.unavailable_reason(tmp_path) == "k_attn_fwd.hsaco is not built"


def test_compile_command_is_the_manifest_command():
    hsaco = _hsaco()
    manifest = hsaco.load_manifest()
    for name, entry in manifest["kernels"].items():
        cmd = hsaco.compile_command("amdclang++", name, Path("/tmp/x.hsaco"))
        assert cmd == ["amdclang++", *manifest["flags"], f"-cuid={name}", *entry["extra_flags"], "-c",
                       entry["src"], "-o", "/tmp/x.hsaco"]  # fmt: skip


# ---------------------------------------------------------------------------
# Gate, layouts and launch plans (meta tensors)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [(4, 8192, 8192, 32, 8), (1, 64, 128, 4, 1), (2, 256, 512, 16, 4), (1, 1024, 1024, 128, 32)],
    ids=["llama3.1-8b", "smallest", "sq_lt_skv", "many_heads"],
)
def test_gate_accepts(monkeypatch, shape):
    monkeypatch.setattr(hk, "hk_gfx1250_unavailable_reason", lambda: None)
    for direction in ("fwd", "bwd"):
        assert hk.hk_gfx1250_unsupported_reason(direction, *_meta(*shape), True) is None, direction


@pytest.mark.parametrize(
    ("direction", "shape", "kwargs", "match"),
    [
        ("fwd", (1, 256, 256, 8, 4), {}, "one of"),  # G = 2
        ("bwd", (1, 256, 256, 32, 4), {}, "must be 4"),  # G = 8: forward only
        ("fwd", (1, 256, 256, 8, 2), {"causal": False}, "causal"),
        ("fwd", (1, 256, 288, 8, 2), {}, "multiples of 64"),
        ("bwd", (1, 256, 320, 8, 2), {}, "64 / 128"),
        ("fwd", (1, 512, 256, 8, 2), {}, "seqlen_q 512 > seqlen_kv 256"),
        ("bwd", (1, 256, 256, 8, 2), {"deterministic": True}, "deterministic"),
        ("fwd", (64, 8192, 8192, 32, 8), {}, "int32"),
        ("bwd", (16, 8192, 8192, 32, 8), {}, "int32"),
    ],
)
def test_gate_refuses(monkeypatch, direction, shape, kwargs, match):
    monkeypatch.setattr(hk, "hk_gfx1250_unavailable_reason", lambda: None)
    kwargs = dict(kwargs)
    causal = kwargs.pop("causal", True)
    reason = hk.hk_gfx1250_unsupported_reason(direction, *_meta(*shape), causal, **kwargs)
    assert reason is not None and match in reason, reason


def test_gate_refuses_other_dtypes_and_head_dims(monkeypatch):
    monkeypatch.setattr(hk, "hk_gfx1250_unavailable_reason", lambda: None)
    assert "bfloat16" in hk.hk_gfx1250_unsupported_reason(
        "fwd", *_meta(1, 128, 128, 4, 1, dtype=torch.float16), True
    )
    assert "head dim" in hk.hk_gfx1250_unsupported_reason("fwd", *_meta(1, 128, 128, 4, 1, d=64), True)


def test_tensor_layouts():
    x = torch.empty(2, 64, 4, 128, device="meta")
    sbhd = torch.empty(64, 2, 4, 128, device="meta").permute(1, 0, 2, 3)
    bhsd = torch.empty(2, 4, 64, 128, device="meta").transpose(1, 2)
    assert [hk.tensor_layout(t) for t in (x, sbhd, bhsd)] == ["bshd", "sbhd", None]
    assert (
        hk.tensor_layout(torch.empty(64, 1, 4, 128).permute(1, 0, 2, 3)) == "bshd"
    )  # b == 1: the same bytes
    kept = hk.hk_gfx1250_operands(x, sbhd, bhsd)
    assert kept[0] is x and kept[1] is sbhd and kept[2].is_contiguous()
    assert hk._strides(x) == (64 * 4 * 128, 4 * 128) and hk._strides(sbhd) == (4 * 128, 2 * 4 * 128)


def test_fdiv_is_exact():
    gen = torch.Generator().manual_seed(0)
    numerators = [0, 1, 2**31 - 1, *torch.randint(0, 2**31, (2000,), generator=gen).tolist()]
    for d in [1, 2, 3, 7, 8, 64, 255, 256, 257, 1000, 4096, 65535, 2**20 + 1]:
        m, sh = hk._fdiv(d)
        assert 0 < m < 2**32
        assert all(n // d == (n * m) >> sh for n in numerators), d


@pytest.mark.parametrize("layout", ["bshd", "sbhd"])
def test_forward_plan(layout):
    """Llama-3.1-8B: 4 x 8 head quads x 64 zig-zag pairs >= 256 CUs, so the persistent grid has one workgroup per CU;
    o comes in q's layout and every stride argument is the layout's."""
    b, s, hq, hkv = 4, 8192, 32, 8
    q, k, v = _meta(b, s, s, hq, hkv)
    if layout == "sbhd":
        q, k, v = (
            torch.empty(t.shape[1], t.shape[0], *t.shape[2:], device="meta").permute(1, 0, 2, 3)
            for t in (q, k, v)
        )
    (kernel, grid, values), (o, lse) = hk._fwd_plan(q, k, v, 0.088, True, cus=256)
    assert (kernel, grid) == ("k_attn_fwd", (256, 1, 1))
    assert hk.tensor_layout(o) == layout and lse.shape == (b, hq, s) and lse.dtype == torch.float32
    struct.Struct(hk._CODE_OBJECTS["k_attn_fwd"][1][kernel][1]).pack(
        *(0 if isinstance(x, torch.Tensor) else x for x in values)
    )
    assert values[21] == 1  # mode: pairs
    assert values[-8:] == (*hk._strides(q), *hk._strides(k), *hk._strides(v), *hk._strides(o))


def test_forward_plan_runs_one_item_per_workgroup_below_the_cu_count():
    q, k, v = _meta(1, 256, 256, 8, 2)  # 2 head quads x 2 pairs < 256 CUs
    (_, grid, values), _ = hk._fwd_plan(q, k, v, 0.088, True, cus=256)
    assert grid == (2 * 4, 1, 1) and values[21] == 0  # 2 head quads x 4 q blocks, mode: items


def test_backward_plan():
    """Three launches on one stream; dq / dk / dv come in the layouts of q / k / v."""
    b, sq, skv, hq, hkv = 2, 512, 1024, 16, 4
    q = torch.empty(sq, b, hq, 128, dtype=torch.bfloat16, device="meta").permute(1, 0, 2, 3)  # sbhd
    _, k, v = _meta(b, sq, skv, hq, hkv)
    do, o, lse = torch.empty_like(q), torch.empty_like(q), torch.empty(b, hq, sq, device="meta")
    launches, (dq, dk, dv) = hk._bwd_plan(do, q, k, v, o, lse, 0.088, True)
    assert [(name, grid) for name, grid, _ in launches] == [
        ("k_attn_bwd_prep", (sq // 64, hq, b)),
        ("k_attn_bwd_main", (hkv, skv // 128, b)),
        ("k_attn_bwd_dq", (sq // 32, hq, b)),
    ]
    assert [hk.tensor_layout(g) for g in (dq, dk, dv)] == ["sbhd", "bshd", "bshd"]
    kernels = hk._CODE_OBJECTS["k_attn_bwd"][1]
    for name, _, values in launches:
        struct.Struct(kernels[name][1]).pack(*(0 if isinstance(x, torch.Tensor) else x for x in values))


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def _resolve_kwargs(q, k, v):
    return dict(q=q, k=k, v=v, causal=True, window_size=(-1, -1), qkv_format="bshd", needs_backward=True)


@needs_flydsl
def test_hipkittens_takes_the_flydsl_head_dim_128_calls_on_gfx1250(monkeypatch):
    monkeypatch.setattr(attention_impl, "is_gfx1250", lambda: True)
    can = attention_impl.DenseAttnFwdHipkittensBackend.can_handle
    assert can(**_resolve_kwargs(*_meta(4, 8192, 8192, 32, 8)))
    assert can(**_resolve_kwargs(*_meta(1, 256, 256, 8, 4)))  # G = 2: both directions on FlyDSL
    q = torch.empty(1, 256, 8, 192, dtype=torch.bfloat16, device="meta")
    k, v = torch.empty_like(q), torch.empty(1, 256, 8, 128, dtype=torch.bfloat16, device="meta")
    assert not can(**_resolve_kwargs(q, k, v))  # the MLA head dims stay on FLYDSL
    assert not can(q=_meta(1, 256, 256, 8, 2)[0], k=None, v=None)


@needs_flydsl
def test_flydsl_stays_the_default_and_hipkittens_is_selected_when_pinned(monkeypatch):
    monkeypatch.setattr(attention_impl, "is_gfx1250", lambda: True)
    kwargs = _resolve_kwargs(*_meta(4, 8192, 8192, 32, 8))
    assert resolve_flash_attn_backend(False, None, **kwargs) == BackendType.FLYDSL
    assert resolve_flash_attn_backend(False, BackendType.HIPKITTENS, **kwargs) == BackendType.HIPKITTENS


# ---------------------------------------------------------------------------
# Numerics (gfx1250)
# ---------------------------------------------------------------------------


_SHAPES = {
    "gqa4": (2, 512, 512, 16, 4),
    "sq_lt_skv": (1, 256, 512, 8, 2),
    "items_grid": (1, 128, 128, 4, 1),
    "llama_heads": (1, 1024, 1024, 32, 8),
}


@needs_hk
@pytest.mark.parametrize("shape", list(_SHAPES.values()), ids=list(_SHAPES))
@pytest.mark.parametrize("layout", ["bshd", "sbhd"])
def test_matches_reference(monkeypatch, shape, layout):
    """Both directions on the HipKittens kernels (no fallback warning), in either layout."""
    log = _LogRecorder()
    monkeypatch.setattr(hk, "logger", log)
    q, k, v, dout = _inputs(*shape, layout=layout)
    _check(_run(q, k, v, dout), _reference(q, k, v, dout, True))
    assert log.warnings == []


@needs_hk
@pytest.mark.parametrize(
    ("shape", "kwargs", "falls_back"),
    [
        ((1, 256, 256, 32, 4), {}, ["backward"]),  # G = 8: the forward on HipKittens
        ((1, 256, 256, 32, 2), {}, ["backward"]),  # G = 16
        ((1, 256, 256, 8, 4), {}, ["forward", "backward"]),  # G = 2
        ((1, 256, 288, 8, 2), {}, ["forward", "backward"]),  # Skv % 64
        ((1, 256, 256, 8, 2), {"deterministic": True}, ["backward"]),
        ((1, 256, 256, 8, 2), {"causal": False}, ["forward", "backward"]),
    ],
    ids=["gqa8", "gqa16", "gqa2", "skv_x32", "deterministic", "non_causal"],
)
def test_fallbacks_match_reference(monkeypatch, shape, kwargs, falls_back):
    """A direction the HipKittens kernels cannot take runs on FlyDSL, with a warning, and the result is the same."""
    log = _LogRecorder()
    monkeypatch.setattr(hk, "logger", log)
    kwargs = dict(kwargs)
    causal = kwargs.pop("causal", True)
    q, k, v, dout = _inputs(*shape)
    _check(_run(q, k, v, dout, causal, **kwargs), _reference(q, k, v, dout, causal))
    assert sorted(m.split("this ")[1].split()[0] for m in log.warnings) == sorted(falls_back), log.warnings


@needs_hk
def test_sbhd_storage_is_read_and_written_in_place():
    """sbhd q / k / v (Megatron's layout): out and the gradients come back as views of sbhd storage."""
    q, k, v, dout = _inputs(2, 256, 256, 8, 2, layout="sbhd")
    out, lse, dq, dk, dv = _run(q, k, v, dout)
    for t in (out, dq, dk, dv):
        assert hk.tensor_layout(t) == "sbhd"


@needs_hk
def test_matches_flydsl_at_the_llama31_8b_training_shape():
    """b4 s8192 hq32 hkv8 against the FlyDSL kernels: the forward, the backward on the same forward output (dq is
    summed with fp32 atomics by both), and the autograd step on Megatron's layout, where the forwards' differences
    carry into the gradients."""
    q, k, v, dout = (t.cuda() for t in _inputs(4, 8192, 8192, 32, 8))
    scale = q.shape[-1] ** -0.5
    o_ref, lse_ref = _flydsl_interface().flash_attn_fwd(q, k, v, scale, True)
    o, lse = hk.flash_attn_hipkittens_gfx1250_forward_impl(q, k, v, scale, True)
    assert compute_snr(o_ref, o) > 45.0
    assert (lse - lse_ref).abs().max().item() < 1e-2
    grads_ref = _flydsl_interface().flash_attn_bwd(dout, q, k, v, o_ref, lse_ref, scale, True, False)
    grads = hk.flash_attn_hipkittens_gfx1250_backward_impl(dout, q, k, v, o_ref, lse_ref, scale, True)
    for name, a, b in zip(("dq", "dk", "dv"), grads, grads_ref):
        assert compute_snr(b, a) > 80.0, name
    got = _run(*_inputs(4, 8192, 8192, 32, 8, layout="sbhd"))
    ref = _run(*_inputs(4, 8192, 8192, 32, 8, layout="sbhd"), backend=BackendType.FLYDSL)
    for name, a, b in zip(("out", "lse", "dq", "dk", "dv"), got, ref):
        assert torch.isfinite(a).all(), name
        assert compute_snr(b, a) > 45.0, name


@needs_hk
def test_repeats():
    """out, lse, dK and dV are bitwise reproducible; dQ (fp32 atomics) to rounding."""
    q, k, v, dout = _inputs(2, 1024, 1024, 32, 8)
    first = _run(q, k, v, dout)
    for _ in range(3):
        again = _run(q, k, v, dout)
        for name, a, b in zip(("out", "lse", "dq", "dk", "dv"), first, again):
            if name == "dq":
                assert compute_snr(a.float().cpu(), b.float().cpu()) > 70.0
            else:
                assert torch.equal(a, b), name


@needs_hk
def test_runs_on_a_side_stream():
    q, k, v, dout = _inputs(1, 256, 256, 4, 1)
    ref = _reference(q, k, v, dout, True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        got = _run(q, k, v, dout)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    _check(got, ref)


@needs_hk
def test_cuda_graph_replays_the_kernels():
    """The kernels launch on the capturing stream: a captured forward + backward replays them with the results of an
    eager call (dQ to rounding: fp32 atomics)."""
    q, k, v, dout = (t.cuda() for t in _inputs(2, 512, 512, 16, 4, layout="sbhd"))
    scale = q.shape[-1] ** -0.5

    def step():
        out, lse = hk.flash_attn_hipkittens_gfx1250_forward_impl(q, k, v, scale, True)
        return (
            out,
            lse,
            *hk.flash_attn_hipkittens_gfx1250_backward_impl(dout, q, k, v, out, lse, scale, True),
        )

    eager = [t.clone() for t in step()]
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        step()  # warm-up on a side stream, as torch.cuda.graph expects
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = step()
    for t in captured:
        t.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    for name, a, b in zip(("out", "lse", "dq", "dk", "dv"), captured, eager):
        if name == "dq":
            assert compute_snr(b.float(), a.float()) > 70.0
        else:
            assert torch.equal(a, b), name


@needs_hk
def test_torch_compile_fullgraph():
    q, k, v, dout = _inputs(2, 256, 256, 8, 2, layout="sbhd")
    ref = _reference(q, k, v, dout, True)

    @torch.compile(fullgraph=True)
    def fn(q, k, v):
        return flash_attn_func(q, k, v, causal=True, return_lse=True)

    q_, k_, v_ = (t.to("cuda").requires_grad_() for t in (q, k, v))
    with _pinned(BackendType.HIPKITTENS):
        out, lse = fn(q_, k_, v_)
    out.backward(dout.to("cuda"))
    _check((out, lse, q_.grad, k_.grad, v_.grad), ref)


@needs_hk
@pytest.mark.skipif(
    os.environ.get("PRIMUS_TURBO_TEST_LARGE") != "1",
    reason="forward repeatability at seq 8192; set PRIMUS_TURBO_TEST_LARGE=1",
)
def test_forward_repeats_bitwise_at_seq_8192():
    """The forward is built in expert scheduling mode: 500 forwards of the Llama-3.1-8B training shape, every
    output finite and bitwise equal to the first."""
    q, k, v, _ = _inputs(4, 8192, 8192, 32, 8)
    q, k, v = (t.to("cuda") for t in (q, k, v))
    with _pinned(BackendType.HIPKITTENS):
        first = flash_attn_func(q, k, v, causal=True, return_lse=True)
        for x in first:
            assert torch.isfinite(x).all()
        for i in range(500):
            again = flash_attn_func(q, k, v, causal=True, return_lse=True)
            for a, b in zip(first, again):
                assert torch.equal(a, b), f"forward {i + 1} differs from the first"
