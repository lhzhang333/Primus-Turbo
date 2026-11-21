###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``primus_turbo.capabilities`` and ``primus_turbo.common.arch``. Host-only: the same on every GPU."""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

import primus_turbo
from primus_turbo import capabilities
from primus_turbo.common.arch import KNOWN_ARCHS, device_arch, normalize_arch

pytestmark = pytest.mark.gfx1250

_PKG_DIR = Path(primus_turbo.__file__).resolve().parent


def _module_path(module: str) -> Path:
    base = _PKG_DIR.joinpath(*module.split(".")[1:])
    return base.with_suffix(".py") if base.with_suffix(".py").is_file() else base / "__init__.py"


def _defines(path: Path, name: str) -> bool:
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            return True
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return True
    return False


@pytest.mark.parametrize(
    "name, arch",
    [
        ("gfx1250", "gfx1250"),
        ("gfx1250:sramecc+:xnack-", "gfx1250"),
        ("GFX950:sramecc+", "gfx950"),
        ("", ""),
        (None, ""),
    ],
)
def test_normalize_arch(name, arch):
    assert normalize_arch(name) == arch


def test_device_arch_is_a_bare_gfx_id():
    # "" where this process has no ROCm device; on a device, an ID without target features.
    arch = device_arch()
    assert arch == normalize_arch(arch)
    assert arch == "" or arch.startswith("gfx")


def test_every_listed_module_is_in_the_package():
    """Not imported (a kernel package may refuse to import off its architecture): the file is
    there and defines the listed symbol. This is what makes a moved module a test failure on
    every architecture instead of a feature that silently reads False."""
    for feature, by_arch in capabilities._FEATURES.items():
        assert by_arch, feature
        for arch, modules in by_arch.items():
            assert arch in KNOWN_ARCHS, (feature, arch)
            assert modules, (feature, arch)
            for entry in modules:
                module, _, symbol = entry.partition(":")
                path = _module_path(module)
                assert path.is_file(), f"{feature} / {arch}: no module {module}"
                assert not symbol or _defines(path, symbol), f"{feature} / {arch}: {module} has no {symbol}"


def test_has():
    assert capabilities.has("mega_moe.bf16", arch="gfx1250")
    assert capabilities.has("mega_moe.bf16", arch="gfx1250:sramecc+:xnack-")
    assert not capabilities.has("mega_moe.bf16", arch="gfx000")
    assert not capabilities.has("no.such.feature", arch="gfx1250")
    table = capabilities.features("gfx1250")
    assert set(table) == set(capabilities._FEATURES)
    assert all(table.values())
    assert not any(capabilities.features("gfx000").values())


def test_a_missing_module_reads_false(monkeypatch):
    monkeypatch.setitem(
        capabilities._FEATURES,
        "attention.flydsl.mha",
        {"gfx1250": ("primus_turbo.flydsl.attention.no_such_package.interface",)},
    )
    assert not capabilities.has("attention.flydsl.mha", arch="gfx1250")
    assert capabilities.has("mega_moe.bf16", arch="gfx1250")


def test_default_arch_is_the_current_device(monkeypatch):
    monkeypatch.setattr(capabilities, "device_arch", lambda: "gfx1250")
    assert capabilities.has("mega_moe.bf16")
    assert capabilities.features()["mega_moe.bf16"]
    monkeypatch.setattr(capabilities, "device_arch", lambda: "")
    assert not capabilities.has("mega_moe.bf16")


def test_cli(capsys):
    assert capabilities._main(["--arch", "gfx1250", "mega_moe.bf16", "attention.flydsl.mha"]) == 0
    assert capsys.readouterr().out.split() == ["True", "True"]
    assert capabilities._main(["--arch", "gfx1250", "mega_moe.bf16", "no.such.feature"]) == 1
    assert capsys.readouterr().out.split() == ["True", "False"]
    assert capabilities._main(["--arch", "gfx1250"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines == [f"{feature} True" for feature in sorted(capabilities._FEATURES)]


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
