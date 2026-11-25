###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``primus_turbo.arch`` (architecture names and manifests) and ``primus_turbo.capabilities``.
Host-only: the same on every GPU."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import primus_turbo
from primus_turbo import arch, capabilities
from primus_turbo.arch import gfx1250

pytestmark = pytest.mark.gfx1250

_PKG_DIR = Path(primus_turbo.__file__).resolve().parent


@pytest.mark.parametrize(
    "name, expected",
    [
        ("gfx1250", "gfx1250"),
        ("gfx1250:sramecc+:xnack-", "gfx1250"),
        ("GFX950:sramecc+", "gfx950"),
        ("", ""),
        (None, ""),
    ],
)
def test_normalize_arch(name, expected):
    assert arch.normalize_arch(name) == expected


def test_device_arch_is_a_bare_gfx_id():
    # "" where this process has no ROCm device; on a device, an ID without target features.
    name = arch.device_arch()
    assert name == arch.normalize_arch(name)
    assert name == "" or name.startswith("gfx")


def test_manifest_lookup():
    assert arch.manifest("gfx1250") is gfx1250
    assert arch.manifest("gfx1250:sramecc+:xnack-") is gfx1250
    assert gfx1250.ARCH == "gfx1250" and gfx1250.ARCH in arch.KNOWN_ARCHS
    assert (
        arch.manifest("gfx000") is None and arch.manifest("__init__") is None and arch.manifest(None) is None
    )


def test_every_listed_module_is_in_the_package():
    """Not imported (a kernel package may refuse to import off its architecture): the file is there."""
    for table in (gfx1250.KERNELS, gfx1250.IMPLS):
        for name, module in table.items():
            assert arch.module_file(module) is not None, f"{name}: no module {module}"
    for feature, needs in gfx1250.FEATURES.items():
        assert needs, feature
        for need in needs:
            if not need.startswith("files:"):  # build products: there only in a build for that architecture
                assert need.partition(":")[0] in {**gfx1250.KERNELS, **gfx1250.IMPLS}, (feature, need)
        assert arch.needs_of(feature, "gfx1250"), feature


def test_has():
    assert capabilities.has("mega_moe.bf16", arch="gfx1250")
    assert capabilities.has("mega_moe.bf16", arch="gfx1250:sramecc+:xnack-")
    assert not capabilities.has("mega_moe.bf16", arch="gfx000")
    assert not capabilities.has("no.such.feature", arch="gfx1250")
    table = capabilities.features("gfx1250")
    assert set(table) == set(gfx1250.FEATURES)
    # every feature whose modules are part of the source is there; one that also needs build products
    # (the HipKittens code objects) is there exactly when they were built
    for feature, answer in table.items():
        files = [n for n in gfx1250.FEATURES[feature] if n.startswith("files:")]
        assert answer == all(any(_PKG_DIR.glob(n[len("files:") :])) for n in files), feature
    assert capabilities.features("gfx000") == {}


def test_a_feature_naming_something_unlisted_reads_false(monkeypatch):
    monkeypatch.setitem(gfx1250.FEATURES, "attention.flydsl.mha", ("no.such.family",))
    assert not capabilities.has("attention.flydsl.mha", arch="gfx1250")


def test_what_older_primus_releases_probe_is_still_there():
    """Primus releases that predate primus_turbo.capabilities look for two directories and import one
    function from one module path. Whoever moves one of them switches a feature off there without an
    error (keep an alias or a stub for a release)."""
    flydsl = _PKG_DIR / "flydsl"
    assert (flydsl / "attention" / "gfx1250").is_dir()
    assert (flydsl / "mega" / "gfx1250").is_dir()
    old = flydsl / "grouped_gemm" / "grouped_gemm_bf16_kernel_gfx1250.py"
    assert old.is_file() and "grouped_gemm_bf16_variable_k_accum_supported" in old.read_text()


def test_a_missing_module_reads_false(monkeypatch):
    monkeypatch.setitem(gfx1250.KERNELS, "attention.mha", "primus_turbo.flydsl.attention.no_such_package")
    assert not capabilities.has("attention.flydsl.mha", arch="gfx1250")
    assert capabilities.has("mega_moe.bf16", arch="gfx1250")


def test_default_arch_is_the_current_device(monkeypatch):
    monkeypatch.setattr(arch, "device_arch", lambda device=None: "gfx1250")
    assert capabilities.has("mega_moe.bf16")
    assert capabilities.features()["mega_moe.bf16"]
    monkeypatch.setattr(arch, "device_arch", lambda device=None: "")
    assert not capabilities.has("mega_moe.bf16")
    assert capabilities.features() == {}


def test_cli(capsys):
    assert capabilities._main(["--arch", "gfx1250", "mega_moe.bf16", "attention.flydsl.mha"]) == 0
    assert capsys.readouterr().out.split() == ["True", "True"]
    assert capabilities._main(["--arch", "gfx1250", "mega_moe.bf16", "no.such.feature"]) == 1
    assert capsys.readouterr().out.split() == ["True", "False"]
    assert capabilities._main(["--arch", "gfx1250"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines == [f"{feature} {answer}" for feature, answer in capabilities.features("gfx1250").items()]


def test_the_query_imports_neither_torch_nor_a_kernel():
    # A launch script asks before anything else is set up; it must stay cheap and unable to fail
    # on a kernel module's import.
    code = (
        "import sys; from primus_turbo import capabilities; "
        "assert capabilities.has('mega_moe.bf16', arch='gfx1250'); "
        "bad = [m for m in sys.modules if m == 'torch' or m.startswith(('flydsl', 'primus_turbo.flydsl'))]; "
        "assert not bad, bad"
    )
    root = os.path.dirname(_PKG_DIR)
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=root)
    assert run.returncode == 0, run.stderr
