#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Check the architecture manifests (``primus_turbo/arch/<arch>.py``) against the tree.

Usage: ``python tools/check_arch_layout.py [--list]`` from anywhere in the repository.

For every architecture that has a manifest:

  * everything the manifest lists exists: the modules of ``KERNELS`` and ``IMPLS``, the paths of
    ``SOURCES`` (build products under ``primus_turbo/lib`` excepted), and every name and symbol
    ``FEATURES`` refers to;
  * every tracked path under ``primus_turbo/`` and ``csrc/`` whose name carries the architecture's
    tag is listed: the tag is a whole segment of a file or directory name, segments being
    separated by ``_`` or ``.`` (``attention/gfx1250_mla/``, ``grouped_gemm_bf16_kernel_gfx1250.py``);
  * the tag is spelled one way: no path under those two trees has the gfx ID glued to other text
    or uses the product name (``mi455x``, ``mi455``) as a name segment instead.

So a new architecture-specific file is registered in the pull request that adds it, and the
manifest stays the complete answer to "where is this architecture's code". Tests and benchmarks
are not listed: a test opts in by its name or by a marker (``tests/conftest.py``).

``--list`` prints what each manifest claims. Exits 1 if anything is reported, 2 when it is not run
in a source tree (next to an installed package there is no ``csrc/`` to check). Imports nothing of
Primus-Turbo: the manifests are data-only modules, loaded by path.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PKG = "primus_turbo"
CHECKED_TREES = ("primus_turbo/", "csrc/")
BUILD_PRODUCTS = ("primus_turbo/lib/",)
# Not source when the tree is walked instead of read from git: caches at any depth, build output
# and vendored code at the top only (primus_turbo/pytorch/dist/ is a source directory).
SKIP_PARTS = {"__pycache__", ".git"}
SKIP_TOP = {"build", "dist", "3rdparty"}


MANIFEST_FIELDS = ("ARCH", "PRODUCTS", "KERNELS", "IMPLS", "SOURCES", "FEATURES")


def known_archs() -> tuple:
    """``KNOWN_ARCHS`` of primus_turbo/arch/__init__.py, read without importing the package."""
    for node in ast.parse((REPO_ROOT / PKG / "arch" / "__init__.py").read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "KNOWN_ARCHS" for t in node.targets
        ):
            return tuple(ast.literal_eval(node.value))
    return ()


def load_manifests() -> tuple[dict, list[str]]:
    """Every primus_turbo/arch/<name>.py other than __init__.py is a manifest. (manifests, errors)."""
    out, errors = {}, []
    for path in sorted((REPO_ROOT / PKG / "arch").glob("*.py")):
        if path.name == "__init__.py":
            continue
        spec = importlib.util.spec_from_file_location(f"_arch_manifest_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        missing = [f for f in MANIFEST_FIELDS if not hasattr(module, f)]
        if missing:
            errors.append(f"primus_turbo/arch/{path.name}: not a manifest, it lacks {', '.join(missing)}")
            continue
        out[path.stem] = module
    return out, errors


def tracked_paths() -> list[str]:
    """Tracked files when REPO_ROOT is the top of a git checkout, otherwise (an unpacked sdist, also
    one that sits inside some other repository) every file of the two trees."""
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        if Path(top).resolve() == REPO_ROOT:
            out = subprocess.run(
                ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
            ).stdout
            paths = [p for p in out.splitlines() if p]
            if paths:
                return paths
    except (OSError, subprocess.CalledProcessError):
        pass
    return [
        p.relative_to(REPO_ROOT).as_posix()
        for tree in CHECKED_TREES
        for p in (REPO_ROOT / tree).rglob("*")
        if p.is_file()
        and not SKIP_PARTS & set(p.relative_to(REPO_ROOT).parts)
        and p.relative_to(REPO_ROOT).parts[0] not in SKIP_TOP
    ]


def module_path(module: str) -> tuple[str, bool] | None:
    """(repo-relative path, is a package) of a dotted module, None when the tree does not have it."""
    base = Path(*module.split("."))
    if (REPO_ROOT / base.with_suffix(".py")).is_file():
        return base.with_suffix(".py").as_posix(), False
    if (REPO_ROOT / base / "__init__.py").is_file():
        return base.as_posix(), True
    return None


def defines(path: Path, name: str) -> bool:
    """Whether the module binds ``name`` at module level: def / class / assignment / import, also
    inside a module-level ``if`` or ``try``."""

    def scan(body) -> bool:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
                return True
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
                return True
            if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == name:
                return True
            if isinstance(node, (ast.Import, ast.ImportFrom)) and any(
                (a.asname or a.name) == name for a in node.names
            ):
                return True
            if isinstance(node, ast.If) and (scan(node.body) or scan(node.orelse)):
                return True
            if isinstance(node, ast.Try) and (
                scan(node.body)
                or scan(node.orelse)
                or scan(node.finalbody)
                or any(scan(h.body) for h in node.handlers)
            ):
                return True
        return False

    return scan(ast.parse(path.read_text(encoding="utf-8")).body)


def claims(manifest) -> tuple[list[str], list[str]]:
    """(claimed path prefixes, errors) of one manifest."""
    prefixes, errors = [], []
    for table in ("KERNELS", "IMPLS"):
        for name, module in getattr(manifest, table).items():
            found = module_path(module)
            if found is None:
                errors.append(f"{table}[{name!r}]: no module {module}")
            else:
                prefixes.append(found[0] + ("/" if found[1] else ""))
    for source in manifest.SOURCES:
        if not (REPO_ROOT / source).exists() and not source.startswith(BUILD_PRODUCTS):
            errors.append(f"SOURCES: no path {source}")
        prefixes.append(source.rstrip("/") + "/")
    for feature, needs in manifest.FEATURES.items():
        for need in needs:
            if need.startswith("files:"):  # build products: not in the source tree
                continue
            name, _, symbol = need.partition(":")
            module = manifest.KERNELS.get(name) or manifest.IMPLS.get(name)
            if module is None:
                errors.append(f"FEATURES[{feature!r}]: {name!r} is neither in KERNELS nor in IMPLS")
                continue
            found = module_path(module)
            if symbol and found is not None:
                source = REPO_ROOT / found[0] / "__init__.py" if found[1] else REPO_ROOT / found[0]
                if not defines(source, symbol):
                    errors.append(f"FEATURES[{feature!r}]: {module} does not define {symbol}")
    return prefixes, errors


def check() -> tuple[list[str], dict]:
    listing = {}
    paths = [p for p in tracked_paths() if p.startswith(CHECKED_TREES)]
    manifests, errors = load_manifests()
    known = known_archs()
    for arch, manifest in manifests.items():
        if manifest.ARCH != arch:
            errors.append(f"primus_turbo/arch/{arch}.py: ARCH is {manifest.ARCH!r}")
        if arch not in known:
            errors.append(
                f"primus_turbo/arch/{arch}.py: {arch} is not in KNOWN_ARCHS of primus_turbo/arch/__init__.py"
            )
        prefixes, found = claims(manifest)
        errors += [f"primus_turbo/arch/{arch}.py: {e}" for e in found]
        listing[arch] = sorted(prefixes)
        aliases = {a for p in manifest.PRODUCTS for a in (p.lower(), p.lower().rstrip("x"))}
        for path in paths:
            if path.startswith(BUILD_PRODUCTS) or path == f"primus_turbo/arch/{arch}.py":
                continue
            names = [part for part in path.split("/")]
            tagged = any(arch in re.split(r"[_.]", part) for part in names)
            # a directory claim covers what is under it, a file claim that file only
            claimed = any(path.startswith(c) if c.endswith("/") else path == c for c in prefixes)
            if tagged and not claimed:
                errors.append(
                    f"{path}: carries the {arch} tag but primus_turbo/arch/{arch}.py does not list it"
                )
            if not tagged and arch in path.lower():
                errors.append(
                    f"{path}: '{arch}' must be a whole name segment (separated by '_', '.' or '/'), "
                    f"as in attention/{arch}_mla/ or grouped_gemm_bf16_kernel_{arch}.py"
                )
            for alias in sorted(aliases):
                if any(alias in re.split(r"[_.]", part.lower()) for part in names):
                    errors.append(f"{path}: use the gfx ID '{arch}' in paths, not the product name '{alias}'")
                    break
    return errors, listing


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--list", action="store_true", help="print what each manifest claims")
    args = parser.parse_args(argv)
    if not all((REPO_ROOT / tree).is_dir() for tree in CHECKED_TREES):
        # e.g. next to an installed package (tools/ is installed with it, csrc/ is not): the manifests'
        # sources are not there to be checked
        print(
            f"{REPO_ROOT} is not a Primus-Turbo source tree (it lacks one of {', '.join(CHECKED_TREES)})",
            file=sys.stderr,
        )
        return 2
    errors, listing = check()
    if args.list:
        for arch, prefixes in listing.items():
            print(f"{arch}:")
            for prefix in prefixes:
                print(f"  {prefix}")
    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
