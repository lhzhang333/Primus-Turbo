###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Regenerate the assembly patches hsaco.py applies (``hip/asm/<entry>.s.patch``) and their manifest sha256s.

Two entries are generated:
  * ``k_attn_bwd_v1``: the ``k_hk_bwd1_g4`` body (from the resident operands to the dK / dV epilogue) comes from
    ``tools/bwd1gen`` (the prep and dQ-convert kernels of the code object stay as compiled);
  * ``k_attn_fwd_v1``: the item body (front, the stage loop, the last stages, the epilogue) comes from
    ``tools/fwd1gen`` (the persistent item loop, the schedule decode and the TDM descriptors stay as compiled).
Compiled with the manifest's flags (``-DHKGEN=1``), the HIP source is a marker shell with inline-asm markers
(``; HKGEN <REGION> ...``) where the generated code goes. A generator compiles the shell (``hsaco.compile_command``),
emits the regions (``tools/asmkit`` derives ``s_set_vgpr_msb``, the expert-mode waits and the hazard ``v_nop``s from
the instruction stream), splices and assembles them and checks the result; the patch is the difference between the
compiler's ``.s`` and the spliced ``.s``. Per entry:

  1. compile exactly as hsaco.py does and keep the compiler's ``.s``: its sha256 is ``asm_input_sha256``;
  2. run the generator (``tools/<gen>/build.py --out DIR --cxx CXX`` + its GEN_ARGS); its ``shell.s`` must be that
     very ``.s``;
  3. patch = unified diff (compiler ``.s`` -> generated ``.s``); hsaco.py must reproduce the generated ``.s`` from it
     and assemble the generator's code object byte for byte;
  4. write the patch and ``asm_input_sha256`` (``hsaco_sha256`` stays the pin tool's, after the GPU validation).

    python primus_turbo/hipkittens/gfx1250_attn/tools/regen_asm_patch.py [--entry NAME ...] [--check] [--cxx PATH]
        [--keep DIR]

``--check`` writes nothing and fails unless every regenerated patch and asm_input_sha256 equal the committed ones (and
the code object equals the pinned hsaco_sha256, when there is one). CPU only; needs the pinned toolchain.
"""

import argparse
import difflib
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
# entry -> (generator, the shipped schedule's arguments): ladder llama_bwd3 rung g10 / llama_fwd3 rung g8c
# (Primus-Turbo-hk-ladder benchmark/hk_attn_ladder)
ENTRIES = {
    "k_attn_bwd_v1": (TOOLS_DIR / "bwd1gen" / "build.py",
                      ["--va-rule", "class", "--vm-burst", "2", "--sig-before", "74", "--wait-before", "79",
                       "--tw-after", "66", "--atom-after", "58", "--xcnt-entry"]),
    "k_attn_fwd_v1": (TOOLS_DIR / "fwd1gen" / "build.py",
                      ["--trans-lat", "18", "--trans-rate", "4", "--valu-lat", "5", "--wmma-reuse", "ab"]),
}  # fmt: skip


def _load_hsaco():
    spec = importlib.util.spec_from_file_location("_gfx1250_attn_hsaco", TOOLS_DIR.parent / "hsaco.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _compile(hsaco, cxx, entry, out_dir: Path):
    """hsaco.py's compile of entry into ``out_dir``: (command, the compiler's .s)."""
    out_dir.mkdir(parents=True)
    cmd = hsaco.compile_command(cxx, entry, out_dir / f"{entry}.hsaco")
    r = subprocess.run(cmd, cwd=hsaco.HIP_DIR, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{entry}: compile failed:\n{r.stderr[-4000:]}")
    s_files = list(out_dir.glob(f"{entry}-hip-*.s"))
    if len(s_files) != 1:
        raise RuntimeError(f"{entry}: expected one saved .s, found {s_files}")
    return cmd, s_files[0]


def regenerate(hsaco, cxx, entry, tmp: Path, keep=None):
    """(patch text, asm_input_sha256, code object sha256) of entry; raises RuntimeError on a failed step."""
    generator, gen_args = ENTRIES[entry]
    k = hsaco.load_manifest()["kernels"][entry]
    if "-DHKGEN=1" not in k.get("extra_flags", []):
        raise RuntimeError(f"{entry}: the manifest's extra_flags must build the marker shell (-DHKGEN=1)")
    _, s_in = _compile(hsaco, cxx, entry, tmp / "cc")
    sha_in = _sha256(s_in)
    out = tmp / "gen"
    r = subprocess.run(
        [sys.executable, str(generator), "--out", str(out), "--cxx", cxx, *gen_args],
        capture_output=True,
        text=True,
    )
    print(r.stdout[-2000:], end="")
    if r.returncode != 0:
        raise RuntimeError(f"{entry}: generator failed:\n{r.stdout[-2000:]}\n{r.stderr[-4000:]}")
    if _sha256(out / "shell.s") != sha_in:
        raise RuntimeError(f"{entry}: the generator's shell .s differs from hsaco.py's compile")
    sha_out = _sha256(out / f"{entry}.hsaco")
    text_in, text_out = s_in.read_text(), (out / f"{entry}.s").read_text()
    diff = difflib.unified_diff(
        text_in.split("\n"),
        text_out.split("\n"),
        f"{entry}.s\tsha256 {sha_in}",
        f"{entry}.s\tgenerated",
        lineterm="",
    )
    patch = "\n".join(diff) + "\n"
    if hsaco.apply_unified_diff(text_in, patch) != text_out:
        raise RuntimeError(f"{entry}: hsaco.py's patcher does not reproduce the generated .s")
    (tmp / "p.patch").write_text(patch)
    cmd, _ = _compile(hsaco, cxx, entry, tmp / "rb")
    hsaco.patch_and_reassemble(
        entry, dict(k, asm_input_sha256=sha_in, asm_patch=str(tmp / "p.patch")), cmd, tmp / "rb", 600
    )
    if _sha256(tmp / "rb" / f"{entry}.hsaco") != sha_out:
        raise RuntimeError(f"{entry}: hsaco.py's build path does not reproduce the generator's code object")
    if keep is not None:
        keep.mkdir(parents=True, exist_ok=True)
        for f in (f"{entry}.s", f"{entry}.hsaco", "build.json"):
            shutil.copyfile(out / f, keep / (f"{entry}.{f}" if f == "build.json" else f))
    return patch, sha_in, sha_out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entry", nargs="*", choices=sorted(ENTRIES), help="entries (default: all)")
    ap.add_argument(
        "--check", action="store_true", help="write nothing; fail unless patches and sha256s match"
    )
    ap.add_argument("--cxx", default=None, help="compiler (default: $PRIMUS_TURBO_HK_CXX, PATH, $ROCM_PATH)")
    ap.add_argument("--keep", default=None, help="save the generated .s, code objects and build.json here")
    a = ap.parse_args(argv)
    hsaco = _load_hsaco()
    m = hsaco.load_manifest()
    cxx = hsaco.find_compiler(a.cxx)
    if cxx is None:
        print(f"no compiler (set ${hsaco.CXX_ENV} or --cxx)", file=sys.stderr)
        return 2
    version = hsaco.compiler_version(cxx)
    if version != m["toolchain"]:
        print(f"toolchain {version!r} != manifest {m['toolchain']!r}", file=sys.stderr)
        return 2
    ok, changed = True, False
    for entry in a.entry or sorted(ENTRIES):
        k = m["kernels"][entry]
        with tempfile.TemporaryDirectory() as tmp:
            try:
                patch, sha_in, sha_out = regenerate(
                    hsaco, cxx, entry, Path(tmp), Path(a.keep) if a.keep else None
                )
            except RuntimeError as e:
                print(e, file=sys.stderr)
                return 1
        dest = hsaco.HIP_DIR / k.get("asm_patch", f"asm/{entry}.s.patch")
        current = dest.is_file() and dest.read_text() == patch and k.get("asm_input_sha256") == sha_in
        pinned = k.get("hsaco_sha256")
        print(f"{entry}: {'current' if current else 'STALE'} (asm_input_sha256 {sha_in}; code object {sha_out}, "
              f"pinned {pinned})")  # fmt: skip
        if a.check:
            ok &= current and pinned in (None, sha_out)
            continue
        if not current:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(patch)
            k["asm_patch"], k["asm_input_sha256"] = str(dest.relative_to(hsaco.HIP_DIR)), sha_in
            if pinned != sha_out:
                k["hsaco_sha256"] = None  # a new code object: pin it after its GPU validation
            changed = True
            print(f"updated {dest} (hsaco_sha256 {'kept' if pinned == sha_out else 'cleared'})")
    if changed:
        hsaco.MANIFEST_PATH.write_text(json.dumps(m, indent=2) + "\n")
        print("updated manifest.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
