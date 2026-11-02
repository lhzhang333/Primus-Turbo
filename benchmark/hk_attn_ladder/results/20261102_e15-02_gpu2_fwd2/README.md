# MLA forward, 2026-11-02 (e15-02 GPU2): patches 26-35

DeepSeek-V3 MLA forward, b2 s4096 h128, qk 192 / v 128, causal, SBHD, bf16, scale 0.1352337788608801
(`HKL_SCALE`). `hkl.ladder` with 12-14 rounds x 20 calls, cold (512 MB L2 flush before every call) and burst (20 ms
idle before every call), null control, FlyDSL = Primus-Turbo main ff4898e4 in the same process
(`HKL_TURBO_DIR=Primus-Turbo-v3`). Every main-line rung passes `hkl.test` on toy, s640, b1_s4096, b1_h32, prod and
prod_b4 with out / lse bitwise equal to FlyDSL.

## Main line

| rung | change | cold ms | burst ms | step (cycles) |
|---|---|---:|---:|---|
| f25b | f24a + ROWS64 (start of the day) | 0.739 | 0.647 | |
| f28a | f26a (sum tree by VGPR bank) + one RESC branch per q-tile pair | 0.736 | 0.643 | f26a 1.003-1.005x |
| f30a | f28a + TILE_C ring slots by inline asm from carried descriptors | 0.738 | 0.644 | f28a ~1.003x |
| f31a | f30a + diagonal prefixes skip the no-op PVs (D2: q tile 0, D3: 0-1, FINAL_A3 after D3: 0-2 and q tile 2's softmax) | 0.733 | 0.643 | 1.004x / 1.004x |
| f32a | f31a + TILE_C issues and advances its ring slot in generated code (descriptor quads in SGPRs) | 0.729 | 0.637 | 1.005-1.013x / 1.006-1.007x |
| f37j | f32a + all four Q fragments resident (patch 37 `--qres`: Q(3) in block 2, softmax scratch in the parity's sum-tree leaves, m / d in the high halves of the (m', -) / (corr, -) pairs, (peer, total) in consumed leaves) + `--lat WMMA=28,VALU=6` | 0.720-0.723 | 0.631-0.636 | cold 1.010-1.015x (clock +1%), burst 1.003-1.004x |

FlyDSL main in the same runs: 0.834-0.846 cold, 0.750-0.764 burst. f32a: 1.15x cold, 1.19x burst; f37j: 1.16-1.17x cold,
1.20x burst. The Q residency removes 24 of the 104 LDS reads per tile: f36x (timing ablation, the reads dropped on
f32a's registers) bounds it at -4.6% cold / -2.5% burst; with the re-planned registers the schedule lost most of the
cycle gain (f37a +1.2% cycles cold), the latency knobs recover it (f37b-k: WMMA 28 / VALU 6 best; the same knobs on
f32a without `--qres` are flat or worse, f37m / f37n).

## Side rungs and dead ends

| rung(s) | what | result |
|---|---|---|
| f25c-e | m1ds model with LDS bursts (DS_RATE 1-2) | 0.961-0.972x |
| f26a | sum-tree partial sums by VGPR bank: 28 -> 2 static conflicts per tile | 1.00x |
| f27a / f27c | O *= corr on every tile (no RESC branch) | 0.966x |
| f27b | DS_RATE 8 | 0.998x |
| f27d | rescale decision and O *= corr out of line | 0.974x |
| f29x | every-tile rescale with an SGPR multiplier (timing only) | = f26a, < f28a |
| f33a | FINAL_A3 laid out right after TILE_D3 (`__builtin_expect`) | 1.000x |
| f34a-c | m1ds with LDS loads in back-to-back pairs, 10-16 cycles apart | 0.980-0.985x |
| f35x | no VGPR bank conflicts in the packed fma (m') and the RESC multiply (timing only, wrong results) | 1.00x |
| f35r8 | DS_RATE 8 on f32a | 0.987-0.991x |
| f36x | Q(0) / Q(3) not re-read per tile (timing only, wrong results) | 1.048x cold, 1.025x burst: the bound for Q residency |
| f37a-i | `--qres` with the default knobs (f37a) and single-knob variants | f37a cold 1.004-1.007x, burst 0.996-0.999x |
| r12t4/8/16 (bwd) | dK/dV: the entry ticket only for the last 1/4, 1/8, 1/16 of the workgroups (bitwise == r10x) | 1.00x, 1.002-1.005x, 0.989x: the entry atomic is not on the critical path |

## What the traces say (f31a / f32a ATT, one wave, `tools/att_*.py`)

- TILE_C: ~1500-1550 cycles per tile against the 1280 WMMA floor; WMMA issue spacing mode 9, mean 9.5.
- Between two TILE_C regions (WMMA to WMMA): f31a 110 cycles (back edge ~34, three SGPR->TDM waits ~27, SALU,
  region entry ~12); f32a 63 (the TDM ops run inside the region, in the shadow of its first ~10 WMMAs). The rest is the
  loop's s_cmp -> s_cbranch (~10), the taken back edge (~24) and the region entry's full va_vdst drain (~23: LLVM's
  wait for the stage address VGPRs computed right before the asm, and the region's own unknown-entry wait).
- RESC pair branches: the skip (taken, +528 B) costs 25 cycles, the rescale path ~70-80 (64 v_pk_mul with no WMMA
  under them); each pair rescales on ~45-50% of the tiles on randn (FlyDSL's 1-nat threshold, per 16-row q tile;
  a numpy model of the decision gives 27% per q tile, 41% per pair). ~96 cycles per tile in all.
- Out-of-line blocks are cheap to reach (taken 11-18 cycles, back 10): f27d lost on the rescale work, not the jumps.
- Windows (WMMA to WMMA) over 8 cycles: two LDS loads in one window (~6 per tile, 13-33 cycles each), the RESC
  decision chain (v_cmp -> s_cmp -> s_cselect -> v_cndmask, 23-27 cycles), and full VALU windows (9-10 cycles).
- Instruction fetch: per block the kernel runs ~48 KB of code (8 KB hot TILE_C, 40 KB once per block); in ATT the first
  instruction of a 128-B line costs 4-21 cycles in TILE_D2 / D3 / FINAL_A3 (2 in TILE_C / D0 / D1). f33a's layout
  change did not move the timing.

## Microbenchmarks (micro/wmma_mix_gen.py, one wave per SIMD, 4 waves)

`wmma_mix_m.txt`, `wmma_mix_p.txt`, `wmma_mix_q.txt`: up to 7 `s_set_vgpr_msb` per WMMA window are free; a window holds
the WMMA plus at most 4 VALU (a 5th costs ~2 cycles); one ds_load_b128 per window is free (W + L + 4 VALU = 8.2
cycles), two per window take 16 cycles (the LDS limit of one 512-B load per 8 cycles per wave); back-to-back WMMA
pairs followed by a VALU run are slower than interleaving.

## Files

- `*_cold.md`, `*_burst.md`: the ladder tables of each step.
- `f25bp_probe.json`, `f31ap_probe.json`: `micro/fwd_probe.py` per-wave breakdowns (HKGEN_PROBE=2).
- `tools/`: ATT analysis scripts used here (`att_regions.py` per-region cycles, `att_branches.py` branch costs,
  `att_xdl_gaps.py` XDL idle by cause, `att_windows.py` WMMA windows by mix, `bankconf.py`, `tilestat.py`).
