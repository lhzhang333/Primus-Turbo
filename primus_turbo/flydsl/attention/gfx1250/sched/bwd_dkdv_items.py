###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

"""Host side of the persistent k_dkdv (bwd_dkdv_kernel) at head dim 128 with GQA: its work items and the table the
kernel reads. The MLA package's module (gfx1250_mla.bwd_dkdv_items) with kv heads as the unit of the lists and one
table row per q head of an item's group (sub-items: the first clears the accumulators and loads K / V, the last
stores dK / dV; bwd_dkdv_sched's flag bits). The rest of this text is the MLA module's.


An item is one 192-row kv block of one (batch, head) of the launch. The kernel's workgroups are persistent (at most
one per CU): workgroup g runs the items of its own list, back to back. The lists are made here, per problem shape,
by a simulation of a greedy dispatch on a cycle model of an item: workgroup g runs on XCD g % 8 (round-robin
dispatch), and XCD x takes the heads hb % 8 == x two at a time, their blocks from the longest on (a head's Q / dO
tiles then stay in that XCD's L2 while its blocks run); a workgroup whose XCD has no item left takes the next one of
the XCD with the most work left.

The table holds, per workgroup, per list slot and per wave, ROW_DW int32 words: every per-item value of the wave as
the kernel's generated code uses it (bwd_dkdv_sched.ROW), tensor addresses as byte offsets from the tensor's base (the
kernel adds the base). A list ends with a row of zeros. The table depends on the shape only and is cached per device.

The dS^T scratch the kernel fills (the HipKittens dslayout.h, version 2) is indexed here too: one region per (batch,
q head) of the launch, inside it the kv16 rows in order, row kt holding the q pairs from max(0, kt // 2 -
(seqlen_kv - seqlen_q) // 32) on as blocks of DSL_BLK_B bytes.
"""

import heapq

import numpy as np
import torch

from primus_turbo.flydsl.attention.gfx1250_mla import NUM_XCD
from primus_turbo.flydsl.attention.gfx1250_mla.bwd_dkdv_items import _TableCache

from . import bwd_dkdv_sched

ROW_DW = bwd_dkdv_sched.ROW_DW  # int32 words per (item, wave) row
NW = 4  # waves per workgroup
KV_W = bwd_dkdv_sched.KV_ROWS  # kv rows per wave
ROWS_B = NW * KV_W  # kv rows per item
D_QK, D_V = 128, 128
DSL_BLK_B = 1024  # a dS^T block: [16 kv][32 q] bf16
HEAD_ROUND = 2  # heads of an XCD whose blocks are dealt together
TAIL_ROUNDS = 2  # head rounds of an XCD dealt last, by length, to level the ends of the lists
# cycle model of an item (the dispatch simulation): per unmasked / masked query tile, per item. Measured at b2 s4096
# h128 (thread trace, barrier to barrier): 1246 / 1462 cycles per tile, an item's first tile 2200, and about 5000
# cycles per item outside its tiles; lists from these, from the measured item durations' fit or from 1350 / 2500 per
# item timed the same within 0.3%
TILE_CYC, MTILE_CYC, ITEM_CYC = 1250, 1350, 4500
TILE_CYC, MTILE_CYC = 1000, 1100  # head dim 128: 96 WMMAs per tile instead of 120 (estimate)
SUB_CYC = 1500  # a sub-item's ring start and its first tile's extra (estimate)

# A table is 0.5 KiB per item and per list end (3 MB at b2 s4096 h128, about 5 ms of host time to make). The cache
# holds MAX_TABLE_BYTES of them: more than the tables of one backward of any size the gate takes, however many head
# chunks it runs (15 to 30 MB at 2 GiB of k; 85 MB for tens of thousands of heads one at a time).
MAX_TABLE_BYTES = 128 << 20


_TABLES = _TableCache(MAX_TABLE_BYTES)


def nblk_kv(skv):
    """kv blocks of 192 rows; the last one's rows past skv (skv % 192 = 64 or 128: the last 128 or 64) are idle."""
    return -(-skv // ROWS_B)


def ds_row_offset(kt, sq, skv):
    """Blocks before kv16 row kt (an integer or an array) inside one (batch, q head) region of the dS^T scratch:
    under the mask, row k >= 2 m (m = (skv - sq) // 32 >= 0) skips its first k // 2 - m q pairs."""
    r = np.maximum(kt - 2 * ((skv - sq) // 32), 0)
    p = r >> 1
    return kt * (sq // 32) - (p * (p - 1) + (r & 1) * p)


def ds_blocks_per_head(sq, skv):
    """Blocks of one (batch, q head) region of the dS^T scratch."""
    return int(ds_row_offset(skv // 16, sq, skv))


def _tiles(bid, sq, skv):
    """(query tiles, masked ones, the first query pair) of kv block bid (an array): the block's tiles are the query
    pairs from its first row's diagonal on, the masked ones those up to its last row's."""
    cshift, nqt2 = skv - sq, sq // 32
    kv0g = bid * ROWS_B
    qs = np.maximum(kv0g - cshift, 0) // 32
    u = kv0g + ROWS_B - 1 - cshift
    qsf = np.minimum(np.where(u < 0, 0, (u + 31) // 32), nqt2)
    nm = np.minimum(np.maximum(qsf - qs, 0), nqt2 - qs)
    return nqt2 - qs, nm, qs


def item_lists(b, sq, skv, hc, g, num_cus):
    """[[(hb, bid)]] per workgroup: the items workgroup g runs, in order; hb = bat * hc + kv head of the launch
    (hc kv heads, each with its g q heads)."""
    nbh, nblk = b * hc, nblk_kv(skv)
    n, nm, _ = _tiles(np.arange(nblk), sq, skv)
    cost = (g * ((n - nm) * TILE_CYC + nm * MTILE_CYC + SUB_CYC) + ITEM_CYC).tolist()
    nwg = min(num_cus, nbh * nblk)
    queues, tail = [], []
    for x in range(NUM_XCD):
        heads = list(range(x, nbh, NUM_XCD))
        rounds = [heads[r : r + HEAD_ROUND] for r in range(0, len(heads), HEAD_ROUND)]
        ntail = min(TAIL_ROUNDS, max(len(rounds) - 1, 0))
        dealt = [[(hb, bid) for bid in range(nblk) for hb in hs] for hs in rounds]
        queues.append([it for r in dealt[: len(dealt) - ntail] for it in r])
        tail += [it for r in dealt[len(dealt) - ntail :] for it in r]
    pos = [0] * NUM_XCD
    left = [sum(cost[bid] for _, bid in q) for q in queues]
    lists = [[] for _ in range(nwg)]
    free = [(0, g) for g in range(nwg)]  # (cycle at which the workgroup is free, workgroup)
    for _ in range(sum(len(q) for q in queues)):
        t, g = heapq.heappop(free)
        x = g % NUM_XCD
        if pos[x] == len(queues[x]):
            x = max(range(NUM_XCD), key=lambda y: left[y])
        hb, bid = queues[x][pos[x]]
        pos[x] += 1
        left[x] -= cost[bid]
        lists[g].append((hb, bid))
        heapq.heappush(free, (t + cost[bid], g))
    # the last head rounds, longest item first, each to the workgroup of its XCD that is free first (to the first
    # free one of all when that is more than the item's length earlier, or still without an item): the lists end
    # level, and none is empty
    end = np.zeros(nwg, np.int64)
    for t, g in free:
        end[g] = t
    for hb, bid in sorted(tail, key=lambda it: (-cost[it[1]], it[0])):
        x = hb % NUM_XCD
        g = first = int(end.argmin())
        if x < nwg:
            own = x + NUM_XCD * int(end[x::NUM_XCD].argmin())
            if not (end[own] > end[first] + cost[bid] or end[own] > end[first] == 0):
                g = own
        lists[g].append((hb, bid))
        end[g] += cost[bid]
    return lists


def item_rows(hb, bid, gi, sq, skv, hq, hkv_n, h0, hc):
    """int32 [len(hb), NW, ROW_DW]: the table rows (bwd_dkdv_sched.ROW) of the sub-items (hb, bid, gi) (int64
    arrays; gi: the q head inside the kv head's group of G = hq / hkv_n) of a launch over the kv heads
    [h0, h0 + hc) of hkv_n."""
    hb, bid, gi = (np.asarray(x, np.int64)[:, None] for x in (hb, bid, gi))
    wv = np.arange(NW, dtype=np.int64)[None, :]
    cshift, mm, nkt = skv - sq, (skv - sq) >> 5, skv // 16
    g = hq // hkv_n
    bat = hb // hc
    hkv = h0 + hb - bat * hc
    qh = hkv * g + gi  # the sub-item's q head
    rg = (hb * g + gi)  # its region of the dS^T scratch: (batch, q head of the launch)
    kv0 = bid * ROWS_B + wv * KV_W
    n, nm, qs = _tiles(bid, sq, skv)
    row = np.zeros((hb.shape[0], NW, ROW_DW), np.int64)
    R = bwd_dkdv_sched.ROW
    row[..., R["n"]] = n
    row[..., R["nmask"]] = nm
    row[..., R["qps"]] = qs
    row[..., R["nl0"]] = (bat * hq + qh) * sq * 4
    row[..., R["thr0"]] = cshift - kv0
    row[..., R["ext"]] = np.clip(skv - kv0, 0, KV_W)
    # dS^T export: descriptor over [region hb, kv16 row kt0, + the true extent of the wave's valid kv16 rows)
    kt0 = kv0 >> 4
    ro0 = ds_row_offset(kt0, sq, skv)
    ro_end = ds_row_offset(np.maximum(np.minimum(kt0 + 3, nkt), kt0), sq, skv)
    for kh in range(3):
        kt = kt0 + kh
        f = (kt >> 1) - mm
        qpf = np.where(kt < nkt, np.where(f > 0, f, 0), 0x7FFFFFFF)
        row[..., R["qpf0"] + kh] = qpf
        row[..., R["rc0"] + kh] = (ds_row_offset(kt, sq, skv) - ro0 - qpf) * DSL_BLK_B
    ds_off = (rg * ds_blocks_per_head(sq, skv) + ro0) * DSL_BLK_B
    nrec = (ro_end - ro0) * DSL_BLK_B
    # buffer resources as [base | num_records << 57, num_records >> 7] (the base added in the kernel)
    nv_b, nk_b = skv * hkv_n * 2 * D_V, skv * hkv_n * 2 * D_QK  # one batch of v / k
    for name, off, nr in (
        ("rds", ds_off, nrec),
        ("rdv", bat * nv_b, nv_b),
        ("rdk", bat * nk_b, nk_b),
    ):
        row[..., R[name]] = off & 0xFFFFFFFF
        row[..., R[name] + 1] = (off >> 32) | ((nr << 25) & 0xFFFFFFFF)
        row[..., R[name] + 2] = nr >> 7
    # Q / dO: the wave's 8 rows of the item's first query pair (byte offsets from dO / Q)
    q_row = (bat * sq + 8 * wv + 32 * qs) * hq + qh
    row[..., R["ga0l"]] = q_row * 2 * D_V
    row[..., R["ga1l"]] = q_row * 2 * D_QK
    # dV / dK row voffset of the wave's first row inside its batch (K / V rows: the batch offset + this)
    kv_row = kv0 * hkv_n + hkv
    row[..., R["gvo"]] = kv_row * 2 * D_V
    row[..., R["rsv"]] = hkv_n * 4 * 2 * D_V  # 4 kv rows of v (and of k)
    # bit 0: the sub-item continues the one before it; bit 1: another one follows (no FINAL)
    row[..., R["flg"]] = (gi > 0) + 2 * (gi < g - 1)
    return (row & 0xFFFFFFFF).astype(np.uint32).view(np.int32)


def item_table(b, sq, skv, hq, hkv, h0, hc, num_cus, device):
    """(table int32 [workgroups, slots, NW, ROW_DW] on ``device``, workgroups, slots) of the launch over the kv heads
    [h0, h0 + hc) of every batch (hq q heads, hkv kv heads in all): workgroup g, wave w reads its rows
    [g, 0 .. , w] until the row of zeros; an item is G = hq / hkv rows, one per q head of its kv head. Cached; a
    new table costs a host-to-device copy (so a backward should have run once before a CUDA graph captures it)."""
    key = (b, sq, skv, hq, hkv, h0, hc, num_cus, str(device))
    capturing = device.type == "cuda" and torch.cuda.is_current_stream_capturing()
    hit = _TABLES.get(key, capturing)
    if hit is not None:
        return hit
    gq = hq // hkv
    lists = item_lists(b, sq, skv, hc, gq, num_cus)
    nslot = gq * max(len(x) for x in lists) + 1
    tbl = np.zeros((len(lists), nslot, NW, ROW_DW), np.int32)
    sub = range(gq)
    g = np.array([g for g, x in enumerate(lists) for _ in x for _ in sub], np.int64)
    s = np.array([s for x in lists for s in range(gq * len(x))], np.int64)
    hb = np.array([hb for x in lists for hb, _ in x for _ in sub], np.int64)
    bid = np.array([bid for x in lists for _, bid in x for _ in sub], np.int64)
    gi = np.array([i for x in lists for _ in x for i in sub], np.int64)
    tbl[g, s] = item_rows(hb, bid, gi, sq, skv, hq, hkv, h0, hc)
    return _TABLES.put(key, (torch.from_numpy(tbl).to(device), len(lists), nslot), capturing)
