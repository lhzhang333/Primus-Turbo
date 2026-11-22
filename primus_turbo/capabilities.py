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

The answers come from the architecture manifests (``primus_turbo/arch/<arch>.py``): a feature
is available when the files of the modules its manifest names are in the installed package.
Nothing is imported to find out, so the query is cheap, needs no GPU and cannot fail on a
kernel module's own import. A feature that no manifest lists for an architecture answers False.
"""

import argparse
import sys

from primus_turbo.arch import features, has

__all__ = ["features", "has"]


def _main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m primus_turbo.capabilities",
        description="Print whether this Primus-Turbo has each feature (True / False), one per line. "
        "With no feature, every feature of the architecture's manifest as '<feature> True|False'. "
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
