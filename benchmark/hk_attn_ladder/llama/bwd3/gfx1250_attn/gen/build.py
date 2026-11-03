###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""hkl.build's generator entry for the bwd3 tree (the Primus-Turbo gfx1250_attn package layout).

    python3 gen/build.py --out DIR --asmkit-root TREE [generator args...]

Runs tools/bwd1gen/build.py (which imports the package's own tools/asmkit, so --asmkit-root is ignored) with --out DIR
and the generator args. It leaves the generated code object in DIR/k_attn_bwd_v1.hsaco and the HIP reference (the
manifest's command as is) in DIR/ref/k_attn_bwd_v1.hsaco, where hkl.build looks for them."""

import runpy
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]


def main():
    args = sys.argv[1:]
    out, rest, i = None, [], 0
    while i < len(args):
        if args[i] == "--asmkit-root":
            i += 2
            continue
        if args[i] == "--out":
            out = args[i + 1]
            i += 2
            continue
        rest.append(args[i])
        i += 1
    if out is None:
        sys.exit("gen/build.py: --out DIR is required")
    script = PKG / "tools" / "bwd1gen" / "build.py"
    sys.argv = [str(script), "--out", out, *rest]
    rc = 0
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    sys.exit(rc)


if __name__ == "__main__":
    main()
