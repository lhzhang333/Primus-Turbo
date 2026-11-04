###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Build the HipKittens gfx1250 GEMM kernels at first use and load them with ctypes.

The kernels need the gfx1250 (``cdna5``) part of HipKittens, which is newer than the HipKittens the
package builds its gfx950 kernels against (``3rdparty/hipkittens``), so they are compiled here with
``hipcc`` rather than into ``primus_turbo.pytorch._C``. The shared object is cached under
``PRIMUS_TURBO_HK_GEMM_CACHE_DIR`` (default ``~/.cache/primus_turbo/hk_gemm``), keyed by the source,
the flags and the HipKittens include directory, so a change to any of them rebuilds. Concurrent
processes (ranks) take a file lock; the first builds, the others load its result.
"""

from __future__ import annotations

import ctypes
import fcntl
import functools
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import primus_turbo.common.envs as envs

_SRC = Path(__file__).with_name("hk_gemm_gfx1250.cpp")
_REPO = Path(__file__).resolve().parents[3]
_FLAGS = ["-DKITTENS_CDNA5", "--offload-arch=gfx1250", "-std=c++20", "-O3", "-w", "-shared", "-fPIC"]


class HipKittensGemmUnavailable(RuntimeError):
    """The gfx1250 HipKittens GEMM cannot be built or loaded here."""


def hipkittens_include() -> Path:
    """The HipKittens include directory with the gfx1250 (``cdna5``) headers."""
    configured = envs.PRIMUS_TURBO_HIPKITTENS_INCLUDE
    inc = Path(configured) if configured else _REPO / "3rdparty" / "hipkittens" / "include"
    if not (inc / "kittens.cuh").is_file() or not (inc / "cdna5").is_dir():
        raise HipKittensGemmUnavailable(
            f"no gfx1250 HipKittens headers in {inc} (kittens.cuh and cdna5/ needed); point "
            "PRIMUS_TURBO_HIPKITTENS_INCLUDE at the include/ directory of a HipKittens checkout that has cdna5"
        )
    return inc


def _hipcc() -> str:
    from torch.utils.cpp_extension import ROCM_HOME

    for cand in ([os.path.join(ROCM_HOME, "bin", "hipcc")] if ROCM_HOME else []) + [shutil.which("hipcc")]:
        if cand and os.path.isfile(cand):
            return cand
    raise HipKittensGemmUnavailable("hipcc not found (torch's ROCM_HOME/bin or PATH)")


def _key(inc: Path, hipcc: str) -> str:
    h = hashlib.sha256()
    h.update(_SRC.read_bytes())
    h.update(" ".join(_FLAGS).encode())
    h.update(str(inc.resolve()).encode())
    h.update(hipcc.encode())
    # the headers themselves: names, sizes and mtimes of the cdna5 tree and kittens.cuh
    for p in sorted([inc / "kittens.cuh", *(inc / "cdna5").rglob("*")]):
        if p.is_file():
            st = p.stat()
            h.update(f"{p.relative_to(inc)}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:16]


@functools.lru_cache(maxsize=None)
def load_library() -> ctypes.CDLL:
    """Build (once per source / flags / headers) and load the kernel library."""
    inc = hipkittens_include()
    hipcc = _hipcc()
    cache = Path(envs.PRIMUS_TURBO_HK_GEMM_CACHE_DIR or Path.home() / ".cache" / "primus_turbo" / "hk_gemm")
    cache.mkdir(parents=True, exist_ok=True)
    so = cache / f"libhk_gemm_gfx1250_{_key(inc, hipcc)}.so"
    if not so.is_file():
        with open(cache / ".build.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not so.is_file():
                with tempfile.TemporaryDirectory(dir=cache) as tmp:
                    out = Path(tmp) / so.name
                    cmd = [hipcc, *_FLAGS, f"-I{inc}", str(_SRC), "-o", str(out)]
                    r = subprocess.run(cmd, capture_output=True, text=True)
                    if r.returncode != 0:
                        raise HipKittensGemmUnavailable(
                            f"hipcc failed ({r.returncode}): {' '.join(cmd)}\n{r.stderr[-4000:]}"
                        )
                    os.replace(out, so)
    lib = ctypes.CDLL(str(so))
    lib.hk_gemm_tn.restype = ctypes.c_int
    lib.hk_gemm_tn.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int] * 5 + [
        ctypes.c_void_p
    ]
    lib.hk_gemm_tn_variant.restype = ctypes.c_int
    lib.hk_gemm_tn_variant.argtypes = [ctypes.c_int] + [ctypes.POINTER(ctypes.c_int)] * 4
    lib.hk_gemm_nt.restype = ctypes.c_int
    lib.hk_gemm_nt.argtypes = lib.hk_gemm_tn.argtypes
    lib.hk_gemm_nt_variant.restype = ctypes.c_int
    lib.hk_gemm_nt_variant.argtypes = lib.hk_gemm_tn_variant.argtypes
    lib.hk_gemm_tn_splitk.restype = ctypes.c_int
    lib.hk_gemm_tn_splitk.argtypes = [ctypes.c_int] + [ctypes.c_void_p] * 4 + [ctypes.c_int] * 6 + [ctypes.c_void_p]
    return lib
