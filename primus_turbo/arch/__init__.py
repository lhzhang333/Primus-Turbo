###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""GPU architectures: their names, and what Primus-Turbo has for each.

An architecture is named by its gfx ID (``gfx942``, ``gfx950``, ``gfx1250``). ROCm reports the
ID with target features attached (``gfx1250:sramecc+:xnack-``); ``normalize_arch`` strips them,
and new code should use it rather than splitting the name again.

**Manifests.** Architecture-specific code is not gathered in a directory of its own: a kernel
stays in the directory of its operation and carries the architecture as a tag in its name
(``grouped_gemm_bf16_kernel_gfx1250.py``, ``attention/gfx1250_mla/``), as the C++ sources do
(``*_gfx950.cu``). What says "this is the gfx1250 code, and this is what it provides" is the
architecture's manifest, ``primus_turbo/arch/<arch>.py``: one data-only module that lists the
kernel packages, the implementation-layer modules and the features built on them. It is

  * what ``primus_turbo.capabilities`` answers from (so code outside Primus-Turbo never has to
    know a path),
  * what ``tools/check_arch_layout.py`` (a pre-commit hook) holds the tree to: every path
    tagged with an architecture that has a manifest must be listed in it, and everything listed
    must exist (code that is specific to an architecture but carries no tag is invisible to the
    check, so tag it or list it),
  * the one file to read, or to route reviews by, for an architecture.

An architecture gets a manifest when someone needs one; gfx1250 (MI455X) has one today.

``device_arch`` answers "which architecture is this device". It does not answer which target
FlyDSL compiles for (``flydsl.runtime.device.get_rocm_arch``, which ``FLYDSL_GPU_ARCH``
overrides) or which architectures the C++ kernels were built for (``setup.py``).

Importing this package imports neither torch nor a kernel module, and neither does a query that
names the architecture. A query that leaves it out asks the current device, which imports torch
and initializes its device runtime: a launch script names the architecture.
"""

import importlib
from pathlib import Path
from typing import Dict, Optional

KNOWN_ARCHS = ("gfx942", "gfx950", "gfx1250")

# CUDA-style compute capability that ROCm's PyTorch reports for each architecture.
_COMPUTE_CAPABILITY = {(9, 4): "gfx942", (9, 5): "gfx950", (12, 5): "gfx1250"}

_ARCH_DIR = Path(__file__).resolve().parent
_PKG_DIR = _ARCH_DIR.parent


def normalize_arch(name: Optional[str]) -> str:
    """``"gfx1250:sramecc+:xnack-"`` -> ``"gfx1250"``; ``None`` or empty -> ``""``."""
    return str(name or "").split(":", 1)[0].strip().lower()


def device_arch(device=None) -> str:
    """Architecture of a torch device (default: the current one), ``""`` without a ROCm device.

    Read from the device's ``gcnArchName``; a PyTorch build that does not report it is
    identified by its compute capability instead.
    """
    try:
        import torch

        props = torch.cuda.get_device_properties(torch.cuda.current_device() if device is None else device)
    except Exception:
        return ""
    arch = normalize_arch(getattr(props, "gcnArchName", ""))
    if arch:
        return arch
    return _COMPUTE_CAPABILITY.get((getattr(props, "major", None), getattr(props, "minor", None)), "")


def manifest(arch: Optional[str]):
    """The manifest module of ``arch`` (``primus_turbo.arch.<arch>``), None when it has none."""
    arch = normalize_arch(arch)
    if arch not in KNOWN_ARCHS or not (_ARCH_DIR / f"{arch}.py").is_file():
        return None
    return importlib.import_module(f"{__name__}.{arch}")


def module_file(module: str) -> Optional[Path]:
    """The file of ``primus_turbo.<...>`` (a module or a package) in this install, None when it is
    not there. Not an import. A ``:<symbol>`` suffix is ignored."""
    parts = module.split(":", 1)[0].split(".")
    if parts[0] != "primus_turbo":
        return None
    base = _PKG_DIR.joinpath(*parts[1:])
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def needs_of(feature: str, arch: Optional[str]) -> tuple:
    """What a feature needs on ``arch``: modules (``"<module>"`` or ``"<module>:<symbol>"``) and, for
    build products, ``"files:<glob relative to the package>"``. ``()`` when the architecture's
    manifest does not list the feature (or there is no manifest)."""
    m = manifest(arch)
    if m is None:
        return ()
    out = []
    for need in m.FEATURES.get(feature, ()):
        if need.startswith("files:"):
            out.append(need)
            continue
        name, _, symbol = need.partition(":")
        # a name the manifest does not define stays as it is and is then simply not present
        module = m.KERNELS.get(name) or m.IMPLS.get(name) or name
        out.append(f"{module}:{symbol}" if symbol else module)
    return tuple(out)


def present(need: str) -> bool:
    """Whether one entry of ``needs_of`` is in this install. Not an import."""
    if need.startswith("files:"):
        return any(_PKG_DIR.glob(need[len("files:") :]))
    return module_file(need) is not None


def has(feature: str, arch: Optional[str] = None) -> bool:
    """Whether ``feature`` is available for ``arch`` (default: the current device's architecture).

    False for a feature or an architecture no manifest lists (to a caller written for a newer
    Primus-Turbo, a name this version does not know is a feature it does not have; ``features``
    gives the names it knows), and, with ``arch`` left out, when there is no ROCm device to ask.
    With ``arch`` given nothing is imported to answer: a feature is available when what its
    manifest names (module files, and for build products at least one matching file) is in the
    installed package.
    """
    needs = needs_of(feature, arch if arch else device_arch())
    return bool(needs) and all(present(n) for n in needs)


def features(arch: Optional[str] = None) -> Dict[str, bool]:
    """``{feature: has(feature, arch)}`` for every feature in the architecture's manifest."""
    arch = normalize_arch(arch) if arch else device_arch()
    m = manifest(arch)
    return {feature: has(feature, arch) for feature in sorted(m.FEATURES)} if m else {}
