###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The code the generators of the gfx1250 head-dim-128 attention kernels emit (gfx1250/sched: fwd_sched and
bwd_dkdv_sched, on the MLA package's assembly model). Host-only and standard library.

REGIONS pins the code that was validated on an MI455X, like test_attention_flydsl_gfx1250_mla_codegen.py does for
the MLA kernels; that file's docstring has the procedure after a deliberate change (read the difference of the
regions, rerun test_attention_flydsl_gfx1250.py and the timings on the card, then update the digests: this file run as
a script prints them).
"""

import hashlib
import importlib
import pathlib
import sys

import pytest

# sha256 of every region's asm text and constraint string
REGIONS = {
    ("fwd", "INIT"): "f0fd455c835dedb75c886c15859da50e5c524845424fbe9105c6a9eacbb7b48c",
    ("fwd", "INIT_C0"): "da9c4c9bac371c66c42e298a8f43a3f99423d614ea06f048272e54d58f83a969",
    ("fwd", "TILE_CL"): "83e12f5c1f79441c5ce42d6ef766cf3cf894108298c96dfa6116f5d842602bfd",
    ("fwd", "TILE_D0"): "ff1ddbaf26b51d37ede24d9c2345d9c8b813b8508f7f4fe5b6be54a6b3254f16",
    ("fwd", "TILE_D1"): "ac50687279b2d76936c93d84bd410fadfd3cff861af797a7cad2cc2e1b066960",
    ("fwd", "TILE_D2"): "4c402894bd4ed04097fa1520c0face9ecfa45e917b8f226d5a8cdf9e9993967e",
    ("fwd", "TILE_D3F"): "1eeefcf807fffd72a4a9f30ffc6fba30adefb24c50994004d28c47589b203480",
    ("fwd", "FINAL_A"): "87d4a494c84e3f70f5e5902055924499b972ca4aa5b0948102b0306e114b244d",
    ("bwd_dkdv", "MAIN"): "e6b87bf5f24a295901df10638f180e5cb7e1fe4b5bbbead6970eb6712a7acd59",
}
_PACKAGE = "primus_turbo.flydsl.attention.gfx1250.sched"


def _digest(sched, name):
    text, constraints = importlib.import_module(f"{_PACKAGE}.{sched}_sched").region_asm(name)
    return hashlib.sha256((text + "\n" + constraints).encode()).hexdigest()


def test_every_region_is_pinned():
    for sched in ("fwd", "bwd_dkdv"):
        names = importlib.import_module(f"{_PACKAGE}.{sched}_sched").OPERANDS
        assert {name for s, name in REGIONS if s == sched} == set(names)


@pytest.mark.parametrize(("sched", "name"), list(REGIONS), ids=["_".join(k) for k in REGIONS])
def test_region_is_the_validated_code(sched, name):
    assert _digest(sched, name) == REGIONS[sched, name], (
        f"{sched}_sched's region {name} changed: see this file's docstring before updating its digest"
    )


if __name__ == "__main__":  # the digests of the tree this file is in
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
    for key in REGIONS:
        print(f'    ("{key[0]}", "{key[1]}"): "{_digest(*key)}",')
