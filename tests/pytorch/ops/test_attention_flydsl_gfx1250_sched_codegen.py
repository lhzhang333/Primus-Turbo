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
    ("fwd", "INIT_C0"): "ecedb236f087d6e1fb2ad439dffeafd25905eb0f4efe20b42c40c74655dff639",
    ("fwd", "TILE_CL"): "4ce3bb11b99fc7a9c76ee2e827594042015106f161474a59957ec189906cbf7f",
    ("fwd", "TILE_D0"): "81248621ee71aa2ef579cb58ab1719befbf033dcd0e8a155b68f3eff96993988",
    ("fwd", "TILE_D1"): "6a01f7a0eab1099b6e8727610431cfa0e2b4d4223c27c9227f4affe511af47a2",
    ("fwd", "TILE_D2"): "28d53b6b888796cb5d128d112a1493b71e9657da25e8d90facc893d00b04550b",
    ("fwd", "TILE_D3F"): "2fbb341a6cc72593dc55eb4076afd0203e060b529b64bcde43318987733c28a5",
    ("fwd", "FINAL_A"): "aa17079c16501789b8da3ce367d9dbc4975f94b50310d44be16e5ab6f0426b1e",
    ("bwd_dkdv", "MAIN"): "426ce61bb0aacade520a94b4e9d305fc1f44d78cd24b647fe1f5948d2a674dc6",
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
