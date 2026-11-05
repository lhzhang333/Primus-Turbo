#!/usr/bin/env python3
"""Map gfx1250 WMMA operand layouts with wmma_probe.hsaco (one WMMA per case, register images from the host).

    PYTHONPATH=<Primus-Turbo-hk-mx> python3 wmma_layout.py [--json out.json]

Steps (each a batch of cases, one wave per case):
  1. rows of A: A one-hot at (lane, byte), B all ones -> the C entries that light up form row m(lane, byte)
  2. cols of B: B one-hot, A all ones -> column n(lane, byte); with 1 and 2 the C register map (lane, reg) -> (m, n)
  3. K pairing: A one-hot at p, B = ones at one (lane half, byte) for all 16 lanes of that half -> lit iff the
     B bytes of that (half, byte) carry the same internal k as p
  4. (MX) which lane's scale VGPR byte scales which row / column and which k block, per scaleAType / scaleBType
Prints a summary and the maps as JSON.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch

from primus_turbo.hipkittens.gfx1250_mla import hiprt

HERE = Path(__file__).resolve().parent
HSACO = HERE / "wmma_probe.hsaco"
E4M3_ONE, E4M3_TWO = 0x38, 0x40
BF16_ONE = (0x80, 0x3F)  # little-endian bytes of bf16 1.0
E8M0_ONE, E8M0_TWO = 127, 128

ap = argparse.ArgumentParser()
ap.add_argument("--json")
a = ap.parse_args()

dev = torch.device("cuda")
sha = hashlib.sha256(HSACO.read_bytes()).hexdigest()
MOD = hiprt.Module(HSACO, sha)


def kern(name, nptr):
    meta = {"block": 32, "kernarg_size": 8 * nptr,
            "args": [{"name": f"p{i}", "offset": 8 * i, "size": 8, "kind": "ptr"} for i in range(nptr)]}
    return hiprt.Kernel(MOD, name, meta)


def run(kname, A, B, SA=None, SB=None):
    """A, B: uint8 [ncase, 32, NB] register images; SA, SB: int32 [ncase, 32]. Returns C float [ncase, 32, 8]."""
    n = A.shape[0]
    Ai = A.contiguous().view(torch.int32)
    Bi = B.contiguous().view(torch.int32)
    C = torch.full((n, 32, 8), float("nan"), device=dev)
    s = torch.cuda.current_stream()
    if SA is None:
        kern(kname, 3).call((n, 1, 1), (32, 1, 1), [Ai, Bi, C], s)
    else:
        kern(kname, 5).call((n, 1, 1), (32, 1, 1), [Ai, Bi, SA.contiguous(), SB.contiguous(), C], s)
    torch.cuda.synchronize()
    return C.cpu()


def ones_img(n, nb, bf16=False):
    t = torch.zeros(n, 32, nb, dtype=torch.uint8)
    if bf16:
        t[:, :, 0::2] = BF16_ONE[0]
        t[:, :, 1::2] = BF16_ONE[1]
    else:
        t[:, :, :] = E4M3_ONE
    return t


def onehot_imgs(nb, bf16=False):
    """One case per (lane, element): element = byte (fp8) or bf16 pair."""
    step = 2 if bf16 else 1
    pos = [(l, e) for l in range(32) for e in range(0, nb, step)]
    t = torch.zeros(len(pos), 32, nb, dtype=torch.uint8)
    for i, (l, e) in enumerate(pos):
        if bf16:
            t[i, l, e], t[i, l, e + 1] = BF16_ONE
        else:
            t[i, l, e] = E4M3_ONE
    return pos, t


def lit(C, i, thr=0.5):
    return frozenset((l, r) for l in range(32) for r in range(8) if abs(float(C[i, l, r])) > thr)


def analyse(kname, nb, bf16=False):
    print(f"\n===== {kname} (operand bytes per lane {nb})", flush=True)
    pos, oh = onehot_imgs(nb, bf16)
    n = len(pos)
    # 1. rows of A
    CA = run(kname, oh.to(dev), ones_img(n, nb, bf16).to(dev))
    rows = {}
    for i, p in enumerate(pos):
        rows[p] = lit(CA, i)
    rowsets = sorted(set(rows.values()), key=lambda s: min(s))
    # 2. cols of B
    CB = run(kname, ones_img(n, nb, bf16).to(dev), oh.to(dev))
    cols = {}
    for i, p in enumerate(pos):
        cols[p] = lit(CB, i)
    colsets = sorted(set(cols.values()), key=lambda s: min(s))
    print(f"distinct A row sets {len(rowsets)} (sizes {sorted(set(len(s) for s in rowsets))}), "
          f"distinct B col sets {len(colsets)} (sizes {sorted(set(len(s) for s in colsets))})")
    # C map: (lane, reg) -> (row id, col id), row id = index of the row set, ordered by the A lane that lights it
    row_of_lane = {}
    for p, s in rows.items():
        row_of_lane.setdefault(s, set()).add(p[0])
    rid = {s: min(l % 16 for l in row_of_lane[s]) for s in rowsets}
    col_of_lane = {}
    for p, s in cols.items():
        col_of_lane.setdefault(s, set()).add(p[0])
    cid = {s: min(l % 16 for l in col_of_lane[s]) for s in colsets}
    cmap = {}
    for s in rowsets:
        for e in s:
            cmap.setdefault(e, [None, None])[0] = rid[s]
    for s in colsets:
        for e in s:
            cmap.setdefault(e, [None, None])[1] = cid[s]
    # A: row determined by lane % 16 only?
    a_row_by_lane = all(rid[rows[(l, e)]] == l % 16 for (l, e) in pos)
    b_col_by_lane = all(cid[cols[(l, e)]] == l % 16 for (l, e) in pos)
    print(f"A row == lane % 16: {a_row_by_lane};  B col == lane % 16: {b_col_by_lane}")
    print("C map (lane: (row, col) per reg 0..7):")
    for l in (0, 1, 15, 16, 17, 31):
        print(f"  lane {l:2d}: " + " ".join(f"({cmap[(l, r)][0]:2d},{cmap[(l, r)][1]:2d})" for r in range(8)))
    # 3. K pairing: A one-hot at p, B ones at (half h, element e) on all 16 lanes of half h
    step = 2 if bf16 else 1
    combos = [(h, e) for h in (0, 1) for e in range(0, nb, step)]
    cases = []
    A = torch.zeros(len(pos) * len(combos), 32, nb, dtype=torch.uint8)
    B = torch.zeros_like(A)
    i = 0
    for pa in pos:
        for (h, e) in combos:
            l, ea = pa
            if bf16:
                A[i, l, ea], A[i, l, ea + 1] = BF16_ONE
                for lb in range(16 * h, 16 * h + 16):
                    B[i, lb, e], B[i, lb, e + 1] = BF16_ONE
            else:
                A[i, l, ea] = E4M3_ONE
                B[i, 16 * h:16 * h + 16, e] = E4M3_ONE
            cases.append((pa, (h, e)))
            i += 1
    CK = []
    chunk = 32768
    for s0 in range(0, len(cases), chunk):
        CK.append(run(kname, A[s0:s0 + chunk].to(dev), B[s0:s0 + chunk].to(dev)))
    CK = torch.cat(CK)
    hits = (CK.abs() > 0.5).flatten(1).any(1)
    pair = defaultdict(list)
    for i, (pa, hb) in enumerate(cases):
        if hits[i]:
            pair[pa].append(hb)
    same = all(pair[(l, e)] == [(l // 16, e)] for (l, e) in pos)
    nmatch = {len(v) for v in pair.values()}
    print(f"K pairing: A (lane, elem) pairs with B (half(lane), same elem) for every position: {same}; "
          f"matches per A position: {sorted(nmatch)}")
    if not same:
        for (l, e) in pos[:: max(1, len(pos) // 24)]:
            print(f"  A lane {l:2d} elem {e:3d} -> B {pair[(l, e)]}")
    return {"cmap": {f"{l},{r}": cmap[(l, r)] for l in range(32) for r in range(8)},
            "a_row_by_lane": a_row_by_lane, "b_col_by_lane": b_col_by_lane, "pairing_same_half_elem": same,
            "pair": {f"{k[0]},{k[1]}": v for k, v in pair.items()}}


def analyse_mx():
    """Scale VGPR semantics of the MX 16x16x128 (layouts of A / B from analyse('k_probe_fp8_128'))."""
    nb = 64
    out = {}
    for ta in (0, 1):
        for tb in (0, 1):
            kname = f"k_probe_mx_{ta}{tb}"
            print(f"\n===== {kname}: scaleAType {ta}, scaleBType {tb}", flush=True)
            # all ones data; scale 1.0 everywhere except one byte (lane L, byte j) of SA (or SB) = 2.0
            cases = [("A", l, j) for l in range(32) for j in range(4)] + [("B", l, j) for l in range(32) for j in range(4)]
            n = len(cases)
            SA = torch.full((n, 32), 0x7F7F7F7F, dtype=torch.int32)
            SB = torch.full((n, 32), 0x7F7F7F7F, dtype=torch.int32)
            for i, (w, l, j) in enumerate(cases):
                t = SA if w == "A" else SB
                v = 0x7F7F7F7F & ~(0xFF << (8 * j)) | (E8M0_TWO << (8 * j))
                t[i, l] = v - (1 << 32) if v >= (1 << 31) else v
            C = run(kname, ones_img(n, nb).to(dev), ones_img(n, nb).to(dev), SA.to(dev), SB.to(dev))
            base = 128.0  # 128 products of 1 * 1
            res = {}
            for i, (w, l, j) in enumerate(cases):
                d = C[i] - base
                nz = (d.abs() > 0.5).nonzero().tolist()
                vals = sorted({round(float(d[x, y]), 2) for x, y in nz})
                res[(w, l, j)] = (len(nz), vals, nz[:3])
            for w in ("A", "B"):
                act = [(l, j, res[(w, l, j)][0], res[(w, l, j)][1]) for l in range(32) for j in range(4) if res[(w, l, j)][0]]
                lanes = sorted({x[0] for x in act})
                print(f"  scale{w}: active lanes {lanes}; per active byte: C entries changed {sorted({x[2] for x in act})}, "
                      f"delta {sorted({v for x in act for v in x[3]})}")
            out[kname] = {f"{w},{l},{j}": res[(w, l, j)][:2] for (w, l, j) in res}
    return out


props = torch.cuda.get_device_properties(0)
print(f"device {props.gcnArchName}  torch {torch.__version__}  probe {sha[:12]}")
R = {}
R["bf16_32"] = analyse("k_probe_bf16_32", 32, bf16=True)
R["fp8_64"] = analyse("k_probe_fp8_64", 32)
R["fp8_128"] = analyse("k_probe_fp8_128", 64)
R["mx"] = analyse_mx()
if a.json:
    with open(a.json, "w") as f:
        json.dump(R, f)
print("done")


def analyse_mx_blocks():
    """Which k block (scale byte j of the row's scale lane) covers each A / B register byte (lane, e)."""
    nb = 64
    res = {}
    for which in ("A", "B"):
        cases = [(l, e, j) for l in range(32) for e in range(nb) for j in range(4)]
        n = len(cases)
        Aimg = ones_img(n, nb) if which == "B" else torch.zeros(n, 32, nb, dtype=torch.uint8)
        Bimg = ones_img(n, nb) if which == "A" else torch.zeros(n, 32, nb, dtype=torch.uint8)
        SA = torch.full((n, 32), 0x7F7F7F7F, dtype=torch.int32)
        SB = torch.full((n, 32), 0x7F7F7F7F, dtype=torch.int32)
        for i, (l, e, j) in enumerate(cases):
            (Aimg if which == "A" else Bimg)[i, l, e] = E4M3_ONE
            v = 0x7F7F7F7F & ~(0xFF << (8 * j)) | (E8M0_TWO << (8 * j))
            v = v - (1 << 32) if v >= (1 << 31) else v
            (SA if which == "A" else SB)[i, l % 16] = v  # scale lane of row / column l % 16 (scaleType 0)
        C = run("k_probe_mx_00", Aimg.to(dev), Bimg.to(dev), SA.to(dev), SB.to(dev))
        mx = C.flatten(1).abs().max(1).values
        blk = {}
        for i, (l, e, j) in enumerate(cases):
            if abs(float(mx[i]) - 2.0) < 0.01:
                blk.setdefault((l, e), []).append(j)
        ok_single = all(len(blk.get((l, e), [])) == 1 for l in range(32) for e in range(nb))
        guess = all(blk.get((l, e), [None])[0] == e // 16 for l in range(32) for e in range(nb))
        print(f"\n===== MX k blocks of {which}: every byte in exactly one block: {ok_single}; block == byte // 16 "
              f"(both lane halves): {guess}")
        if not guess:
            for l in (0, 16):
                print(f"  lane {l}: " + " ".join(str(blk.get((l, e), ['?'])[0]) for e in range(nb)))
        res[which] = {f"{l},{e}": blk.get((l, e)) for l in range(32) for e in range(nb)}
    return res


R2 = analyse_mx_blocks()
if a.json:
    p = Path(a.json)
    R = json.loads(p.read_text())
    R["mx_blocks"] = R2
    p.write_text(json.dumps(R))
print("done blocks")
