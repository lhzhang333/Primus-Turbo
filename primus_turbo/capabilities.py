###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Which architecture-specific features this Primus-Turbo has.

The public way for a caller outside Primus-Turbo (Primus, a launch script) to ask "does
this Primus-Turbo have feature X for architecture Y", instead of looking for a directory
or importing an internal module::

    from primus_turbo import capabilities
    capabilities.has("mega_moe.bf16", arch="gfx1250")
    capabilities.features("gfx1250")          # {feature: bool}

    python -m primus_turbo.capabilities --arch gfx1250 mega_moe.bf16   # prints True / False

``_FEATURES`` maps a feature to the modules that implement it, per architecture. A feature
is available when every one of those module files is in the installed package; nothing is
imported to find out, so the query is cheap, needs no GPU and cannot fail on a kernel
module's own import. ``tests/pytorch/core/test_capabilities.py`` checks on every
architecture that each listed module file exists and defines the listed symbol, so whoever
moves or renames one has to update the table.

The table lists what callers ask about today, which is gfx1250 (MI455X). A feature that is
not listed for an architecture answers False; add a row when a caller needs one.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

from primus_turbo.common.arch import device_arch, normalize_arch

_PKG_DIR = Path(__file__).resolve().parent

# feature -> architecture -> modules ("<module>" or "<module>:<attribute>"), all required.
_FEATURES: Dict[str, Dict[str, Tuple[str, ...]]] = {
    # Dense multi-head / grouped-query attention (flash_attn_func, backend FLYDSL).
    "attention.flydsl.mha": {
        "gfx1250": (
            "primus_turbo.flydsl.attention.gfx1250.mha.interface",
            "primus_turbo.pytorch.kernels.attention.attention_flydsl_gfx1250_impl",
        ),
    },
    # Multi-head latent attention at the DeepSeek-V3 head dims.
    "attention.flydsl.mla": {
        "gfx1250": (
            "primus_turbo.flydsl.attention.gfx1250.mla.fwd",
            "primus_turbo.flydsl.attention.gfx1250.mla.bwd",
            "primus_turbo.pytorch.kernels.attention.attention_flydsl_gfx1250_mla_impl",
        ),
    },
    # Sparse MLA (DeepSeek-V4).
    "attention.flydsl.sparse_mla": {
        "gfx1250": (
            "primus_turbo.flydsl.attention.gfx1250.sparse_mla.fwd_kernel",
            "primus_turbo.flydsl.attention.gfx1250.sparse_mla.bwd_kernel",
        ),
    },
    # HipKittens kernels of the MLA backward.
    "attention.hipkittens.mla": {
        "gfx1250": ("primus_turbo.pytorch.kernels.attention.attention_hipkittens_gfx1250_mla_impl",),
    },
    # BF16 MegaMoE (fused_mega_moe): dispatch + GEMM + combine in fused kernels.
    "mega_moe.bf16": {
        "gfx1250": (
            "primus_turbo.flydsl.mega.gfx1250.bf16.dispatch_gemm_kernel",
            "primus_turbo.flydsl.mega.gfx1250.bf16.gemm_combine_kernel",
            "primus_turbo.pytorch.kernels.fused_mega_moe.fused_mega_moe_gfx1250_impl",
        ),
    },
    # BF16 grouped-GEMM weight gradient that adds into an fp32 main_grad (beta = 1).
    "grouped_gemm.flydsl.bf16.wgrad_accum": {
        "gfx1250": (
            "primus_turbo.flydsl.grouped_gemm.gfx1250.grouped_gemm_bf16_kernel"
            ":grouped_gemm_bf16_variable_k_accum_supported",
        ),
    },
    # FlyDSL intranode DeepEP (V2).
    "deep_ep.flydsl": {
        "gfx1250": ("primus_turbo.flydsl.deep_ep.intranode_v2_kernel",),
    },
}


def _module_file_exists(module: str) -> bool:
    """Whether ``primus_turbo.<...>`` is a module or package file of this install. Not an import."""
    parts = module.split(":", 1)[0].split(".")
    if parts[0] != "primus_turbo":
        return False
    base = _PKG_DIR.joinpath(*parts[1:])
    return base.with_suffix(".py").is_file() or (base / "__init__.py").is_file()


def has(feature: str, arch: Optional[str] = None) -> bool:
    """Whether ``feature`` is available for ``arch`` (default: the current device's architecture).

    False for a feature or an architecture the table does not list, and, with ``arch`` left
    out, when there is no ROCm device to ask.
    """
    arch = normalize_arch(arch) if arch else device_arch()
    modules = _FEATURES.get(feature, {}).get(arch)
    return bool(modules) and all(_module_file_exists(m) for m in modules)


def features(arch: Optional[str] = None) -> Dict[str, bool]:
    """``{feature: has(feature, arch)}`` for every feature in the table."""
    arch = normalize_arch(arch) if arch else device_arch()
    return {feature: has(feature, arch) for feature in sorted(_FEATURES)}


def _main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m primus_turbo.capabilities",
        description="Print whether this Primus-Turbo has each feature (True / False), one per line. "
        "With no feature, every feature in the table as '<feature> True|False'. "
        "Exit code 0 when every feature asked for is available, 1 otherwise.",
    )
    parser.add_argument("--arch", default=None, help="gfx ID, e.g. gfx1250 (default: the current device)")
    parser.add_argument("feature", nargs="*")
    args = parser.parse_args(argv)
    if args.feature:
        answers = [has(f, args.arch) for f in args.feature]
        for answer in answers:
            print(answer)
        return 0 if all(answers) else 1
    for feature, answer in features(args.arch).items():
        print(f"{feature} {answer}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
