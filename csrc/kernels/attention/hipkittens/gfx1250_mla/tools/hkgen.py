###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""hkgen -- build a generated gfx1250 MLA kernel (k_mla_fwd, k_dkdv96ds) from its marker shell. CPU only.

    python3 tools/hkgen.py KERNEL --out DIR --cxx CXX

Compiles the shell with hsaco.py's command (DIR/shell/*.s is the compiler's .s), generates the regions of its
``; HKGEN`` markers (fwd.py / bwd.py: instruction order, s_set_vgpr_msb, expert-mode waits, hazard v_nops), splices
them behind their markers (DIR/KERNEL.s, 1024 VGPRs) and assembles DIR/KERNEL.hsaco with the clang driver's own
cc1as / lld commands. hsaco.py installs the result only when it has the manifest's sha256."""

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import asm as A  # noqa: E402
import bwd  # noqa: E402
import fwd  # noqa: E402

GENERATORS = {"k_mla_fwd": fwd, "k_dkdv96ds": bwd}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("kernel", choices=sorted(GENERATORS))
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--cxx", required=True, help="compiler (the manifest's toolchain)")
    a = ap.parse_args(argv)
    spec = importlib.util.spec_from_file_location("_gfx1250_mla_hsaco", HERE.parent / "hsaco.py")
    hsaco = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hsaco)
    out = Path(a.out).resolve()
    (out / "shell").mkdir(parents=True, exist_ok=True)
    cmd = hsaco.compile_command(a.cxx, a.kernel, out / "shell" / f"{a.kernel}.cc.hsaco")
    r = subprocess.run(cmd, cwd=hsaco.HIP_DIR, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{a.kernel}: shell compile failed:\n{r.stderr[-4000:]}")
    (s_path,) = (out / "shell").glob("*-gfx1250.s")
    shell = s_path.read_text().rstrip("\n").split("\n")
    spliced = out / f"{a.kernel}.s"
    spliced.write_text("\n".join(A.splice(shell, GENERATORS[a.kernel].generate(shell))) + "\n")
    A.assemble(cmd, hsaco.HIP_DIR, spliced, out / f"{a.kernel}.hsaco")
    return 0


if __name__ == "__main__":
    sys.exit(main())
