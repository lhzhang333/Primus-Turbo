# Log: MXFP8 DSv3 MLA HK attention

## 2026-11-05

**Setup.** Branch `dev/lhz/dsv3_attn_hk_mxfp8` = origin/dev/mi455x_v4 429492b8 + merge of fork
`dev/lhz/dsv3-attn-hk-fwd-bwd` (f91a3f2f, no conflicts) + envs port (#605) + test fix; pushed to the fork.
Code objects rebuild to the pins (k_mla_fwd c29ac43f, k_dkdv96ds abeea38e), `regen_asm_patch.py --check` passes.
GPU2: HK pytest 97 pass, FlyDSL MLA 105 pass; `scratch_mx/mla_hk_bench.py` (8 rounds): HK fwd 0.724 / 0.727 ms
(hot / cold), bwd 2.303 / 2.288, autograd 3.043 / 3.029; FlyDSL 0.802 / 3.505 / 4.305.

**WMMA layouts** (`micro/wmma_probe.hip` + `micro/wmma_layout.py`, results/wmma_layout.json): for bf16 16x16x32,
fp8 16x16x64 and fp8 16x16x128 alike, A row = lane % 16, B column = lane % 16, C (lane l, reg r) = (row
r + 8 (l / 16), column l % 16), and A (lane, byte e) pairs with B (lane half, same byte e): the K order of a byte
is the same function of (half h, byte e) for A and B. MX 16x16x128 (E8M0): scaleAType / scaleBType 0 take the
scales from lanes 0..15, 1 from lanes 16..31; lane l's byte j scales row (A) / column (B) l % 16 for k block j, and
block j is bytes 16 j .. 16 j + 15 of every lane (both halves). Consequences:
- logical k of (half h, byte e) = 16 (e / 8) + 8 h + e % 8 makes block j = k in [32 j, 32 j + 32): standard MX
  blocks. Stored "MX-interleaved" (every 16 elements: the first 8 to half 0, the next 8 to half 1), a lane's 64 bytes
  are contiguous: 4 ds_load_b128 per operand.
- the S^T accumulator of kv16 tile t (lane half h: rows 8 h .. 8 h + 7) goes to bytes 8 t .. 8 t + 7 of the PV B
  operand: k = 16 t + 8 h + r = its kv, natural order, no lane exchange.
- one scale VGPR feeds two WMMAs (opsel 0: lanes 0..15, opsel 1: lanes 16..31), e.g. nope + rope of one q / kv tile.

**Numerics emulation** (`py/mx_numerics.py`, CPU in lhz-hk-cpu; b1 s2048 h8, causal, scale 0.13523; SNR dB against
fp32 on the bf16 inputs; mx_a = Q / K MX along d, V MX along kv, P E4M3 at 2^-7, dP / dV MX, dS bf16; mx_b = dS MX
too; +ks = k centred per channel over tokens (softmax-invariant); qkbf = QK bf16, the rest as named):

| data (logit std) | variant | o | lse | dq | dk | dv |
|---|---|---|---|---|---|---|
| randn (1.9 nats) | bf16 | 54.1 | 145.5 | 54.5 | 54.7 | 55.7 |
| | mx_a (OCP floor scales) | 22.7 | 50.2 | 22.0 | 22.0 | 22.4 |
| | mx_a+ks, ceil scales | 23.5 | 45.3 | 22.9 | 22.9 | 23.2 |
| | mx_b+ks, ceil | 23.5 | 45.3 | 21.7 | 21.8 | 23.2 |
| | qkbf_mx_a (floor) | 29.4 | 145.5 | 26.3 | 26.4 | 28.1 |
| soft (1.0) | mx_a+ks, ceil | 26.5 | 49.9 | 25.5 | 25.5 | 26.1 |
| kbias (5.8: k channel offsets) | mx_a (floor) | 16.4 | 35.2 | 13.7 | 16.2 | 16.3 |
| | mx_a+ks, ceil | 23.5 | (shifted) | 22.7 | 22.8 | 23.1 |
| mla-like (13.9: peaked) | mx_a (floor) | 12.5 | 31.0 | 5.4 | 5.5 | 12.6 |
| | qkbf_mx_a (floor) | 30.9 | 145.0 | 12.1 | 16.3 | 29.3 |

The QK error is the floor: E4M3's 3 mantissa bits give S ~28 dB (4% relative), i.e. a logit error of ~4% of the
logit spread (0.07 nats at 1.9 nats spread, 0.67 at 13.9), and the softmax turns it into the output error. K
centring removes bias-type outliers; the no-saturation scale rule (2^ceil(log2(amax / 448))) is worth ~1 dB over the
OCP floor rule (which clips the block maximum to 448 in ~a quarter of the blocks). Decisions: ceil scales, k
centring built in (lse corrected by scale * q . kbar), two QK modes (mx / bf16) behind one kernel, the training gate
decides.

GPU: from 03:50 GPU2 runs a foreign qwen38 server (178 GB), GPU3 another session (k3-hk), GPU1 a pr597 job; GPU0 is
idle but the ladder's logs/GPU0.FAULT (11-03, a foreign pmc SMU hang, pre-reboot) is in place and clearing it was
refused to me: CPU work until a GPU is free / the user clears it.

**MX forward m0 + quantization on the GPU** (GPU3 via `scratch_mx/g3.sh`, shared with the K3 session's lock; GPU2 taken by
a foreign qwen38 server). `py/test_fwd.py`: toy / small / mid / prod all PASS: quantization outputs bit-exact against the
torch twins (codes, exponents, pads, kbar, q . kbar); o / lse against the fp32 emulation of the kernel's algorithm
53-55 / 99-102 dB (kernel correct); against the fp32 reference on the bf16 inputs o 23.3-24.6 dB, lse 47-55 dB (the
recipe floor, as emulated); deterministic, finite.

Timing, b2 s4096 h128 (`py/bench_fwd.py`, hot / cold ms): bf16 HK fwd 0.732 / 0.737; MX m0 0.997 / 0.984; quantization
(kbar + Q rows + K rows + V cols) 0.33 -> 0.24 after: kbar from every 16th row (any constant shift of k is
softmax-invariant; the backward uses the same kbar), rows in X8's (b, h, s) order with the kbar slice loaded once per wave
(scattered byte stores of the (s, b, h) order were slower), cols with 4-byte LDS reads. Per kernel (`py/bench_quant.py`):
kbar 0.024, Q / K rows 0.080 / 0.078, V cols 0.088 ms: 5-8 TB/s, to be fused into the producers in Primus (P4).

m0 ablations (hot ms): full 1.005, no exp 0.977, no softmax 0.956, no QK WMMA 0.838, no PV WMMA 0.898, no K/V LDS
reads 0.895: m0 is latency-bound (per q tile QK -> max -> exp -> PV in series, all 92 fragment reads of a tile waited
for before any WMMA), not VALU-bound. ~5100 cycles per 128-kv tile against an issue bound of ~1500.

Rungs written while the node was down (compiled, not yet run): m1 (K fragments of tile t + 1 prefetched after the last
QK of tile t, B_{t+1} signalled after QK of q tile 0, QK(q_{i+1}) before softmax(q_i), diagonal tiles split, branch-free
rescale: 1024 VGPRs, 8 spills), m2 (m1 with Q read from LDS per tile: 1012 VGPRs, no spill), m3 (m2 with every slot
interleaved WMMA / softmax chunk by sched_barrier fences: slot i = PV(q_{i-1}) + QK(next) | softmax(q_i) chunks,
QK(q0) of tile t + 1 in slot 3, 16 / 24 / 24 / 24 WMMAs + an 8-WMMA tail per tile; 956 VGPRs, no spill).

04:31 UTC: GPU3 MES hang in the K3 session's Megatron job (TE arm), hive-wide mode2 reset stuck in
amdgpu_dpm_mode2_reset, kfd workers in D state: the node needs a power cycle. GPU work stopped.

**Turbo integration (P4 start, CPU only).** `primus_turbo/hipkittens/gfx1250_mla_mx` (kernels m0 / b0 + `mx.py`: build at
first use into a per-user cache, one builder per node via flock, ~4 s for the three sources) and `hk_mx` for
`PRIMUS_TURBO_MLA_FWD_IMPL` / `_BWD_IMPL`: any b / s / h strides (no fold, no copy), SBHD outputs, any backward after
any forward (an hk_mx forward saves its MX operands; an hk_mx backward after a bf16 forward quantizes q / k itself;
a bf16 backward after an hk_mx forward takes its out / lse), fallback hk_mx -> hk -> FlyDSL. Tests
`tests/pytorch/ops/test_attention_hipkittens_gfx1250_mla_mx.py`: settings / shape rules / routing 15 pass on CPU,
numerics pending. Found on the way: the root `.gitignore` has `*.hip`, the earlier commits had no kernel sources
(fixed: negation rules, 4703aca9).

**Static ISA analysis (`py/isa_stat.py`, CPU).** Per 128-kv tile and wave (main loop):

| rung / flags | VGPR | WMMA | VALU | TRANS | LDS | s_wait_alu | loop spills |
|---|---|---|---|---|---|---|---|
| m0 | 1021 | 96 | 1552 | 260 | 90 | - | 0 |
| m3 | 956 | 96 | 783 (390 packed) | 260 | 118 | - | 0 |
| m3 x | 956 | 96 | 783 | 260 | 118 | 38 | 0 |
| m4 x / xp | 894 / 918 | 100 | ~515 incl. the rescale branches (~275 without) | 259 | 114 | 6 | 0 |

(x = `-mllvm -amdgpu-expert-scheduling-mode`, which every bf16 HK kernel uses: without it LDS patterns were 4.6x slow
in the bf16 micro; xp = x + `-mllvm -greedy-regclass-priority-trumps-globalness`.) By the bf16 window rule (a WMMA
window hides <= ~4 VALU; a packed f32 op costs ~1 extra cycle) m3 is VALU-issue-bound (~1200 cyc / tile vs ~870 of
WMMA): hence m4.

**Rung m4** (`kern/k_mla_fwd_mx4.hip`, recipe m4): Q8 quantized from q * scale * log2(e) (k_mxq mode 32: S' in log2
units, no multiply); the nope WMMA's accumulator starts at Mt = -m (no exp-argument fma); the row sum is one more PV
WMMA against an all-ones V^T tile (the sum of the quantized P, the values PV takes; no pk_add); O / row-sum rescale only
in a wave-uniform lazy branch (TAU2 1.75 bits: P <= 431 < 448); rope as a WMMA over d 64..191 with E8M0 0 on d 64..127
(scale layout R2 = [s0..s3, 0, 0, s4, s5], k_mxq mode 16): 24-VGPR [nope | rope] fragments read as two overlapping
operands (LLVM allocates v[n:n+15] / v[n+8:n+23] without copies), no zero halves; Q double-buffered; LDS addresses
precomputed. Emulation `emu_fwd4` (CPU): o / lse within 0.05 dB of m3 on randn / kbias / peaked data.

**Rung b1** (`kern/k_mla_bwd_mx1.hip`, recipe m4): the S and dP WMMAs start at LSE2 / DELTA with the C modifier = neg
(builtin modC 1 = -C, 2 = |C|; scale operands may be SGPRs): P = exp2(S' - lse2) and dP - delta with no VALU; mask only
on the diagonal steps / tiles; dkdv 2 kv16 tiles per wave (3 spills), dq as b0 + the above; 749 / 911 VGPRs, no spills.

**Register allocation rules found (gfx1250, HIP C++, ~900-1000 VGPR kernels):**
- DS addresses and WMMA scale operands must live in v0..v255 (no VGPR-MSB remap); bank 0 fills and the greedy allocator
  spills them: `-greedy-regclass-priority-trumps-globalness` (as bf16 HK bwd) and no long-lived per-lane LDS offsets
  (recompute from a volatile `v_mbcnt` lane id where the addresses are formed).
- A branch between two instantiations of a big body inside one loop costs ~100 VGPRs (929 -> 1024 + spills): peel the
  special iterations into their own loop.
- Causal masks as per-element index compares hoist ~64 index registers: compare a compile-time constant (16 j + r)
  against one per-lane threshold.
- In the dq kernel the e8m0-bits exponent spills 468 B where mx_exp / exp2i does not (allocator noise; kept per kernel).
- Expert mode: a ds_load whose address a just-issued VALU wrote waits for every in-flight VALU (WMMAs included) once
  more than one class is pending; also WAR on fragment registers (double-buffer Q; V / K loads drain once per tile).

**GPU session plan (when the node is back):** test_fwd --tag {3, 3x, 4xp} toy..prod; bench_fwd --build --tags
,x,xp,2xp,3,3x,3xp,4,4x,4xp; test_bwd --fwd-tag 4xp --tag 1xp (and b0 --tag xp); bench_bwd --build --tags
,x,xp,1xp; then Turbo pytest (hk_mx numerics); update the package with the winners.

**Compiler bug found (clang, gfx1250 toolchain of the containers):** `__builtin_bit_cast(float, v[i])` on an
`ext_vector_type` element ignores the index and reads element 0 (the -O3 IR shows a `<0,0,0,0,4,4,4,4>` shuffle when
eight such casts build an f32x8 from two u32x4). It silently broke b0's dK/dV lse / delta reads and b1's C tuples
(caught on the CPU from the ISA: a ds_load_2addr of elements 0 / 4 broadcast to 8 registers). Fix: `mx::as_f(uint32_t)`
(value parameter). Rule: never bit_cast a vector element in place; cast whole vectors or go through a scalar.

**Rung b2** (`kern/k_mla_bwd_mx2.hip`): dK/dV split into k_dv_mx (S, P, dV) and k_dk_mx (S, dP, dS, dK) with 4 kv16
tiles per wave (b1 fused: 2, the register limit), one templated body; every LDS fragment feeds 4 WMMAs; S computed
twice (+20% WMMAs in total) for an LDS-bound -> WMMA-bound change (model: dK/dV ~1.4x). Steady loops: dv 96 WMMA /
144 VALU / 103 LDS; dk 144 / 664 / 171; VGPRs 489 / 1023, no spills; k_dv_mx fits 8 tiles per wave (953).

GPU session script: `scratch_mx/session_gpu.sh <gpu> [first step]` (each step through g<gpu>.sh with its own timeout,
logs in scratch_mx/logs/session_<n>.log, stops at the first failure).

**Training gate prepared** (outside the repo, `/home/lihuzhan/projects/2026_1023__v2/scripts`): `mx_gate.sh <tag> <rounds>
<arm>...` (DSv3 proxy, 4 GPUs, Primus worktree `Primus-v4-mx` = detached origin/dev/mi455x_v4 c6bd8c649, TURBO_DIR =
this worktree, own FlyDSL caches; arms fly / hk / mx (hk_mx both passes) / mxf (hk_mx forward + hk backward), `_warm`
fills caches; TRAIN_ITERS default 200) and `mx_gate_summary.py logs/<tag> --ref hk` (mean |dloss|, final-window loss,
grad-norm ratio, median iteration time). Order after the kernel tests pass: `mx_gate.sh gate1 1 hk_warm mx_warm` then
`mx_gate.sh gate2 2 hk mx mxf`.

## 2026-11-05 (b): heliosr-1b112-a30-4 (1 x MI455X, GPU0 shared with other sessions through logs/gpu0.lock)

**Reproduction** (container lhz-mx-dsv3, image tas-20261019; every GPU job through `scratch_mx/gx.sh`: GPU0 lock,
timeout, dmesg check). All GPU checks of the 11-05 session list pass: quantization (rc vs rows / cols bit-exact; the
fp32 q . kbar differs by summation order only, <= 1e-6, now compared with a tolerance), m4 / m3 forward toy..prod
(53 dB vs emulation), b0 / b1 / b2 backward toy..mid (46-53 dB), Turbo hk_mx pytest 28 pass. Lab fixes: KERNELS lacked
b2's k_dv_mx / k_dk_mx; recipe_of mapped backward tags 2* to r1. Timing (b2 s4096 h128, hot / cold ms): bf16 HK fwd
0.624 / 0.638, bwd 1.983 / 1.994; m0 0.880, m3xp 0.865, m4xp 0.713 / 0.699; fwd quant 0.207, quant_all 0.328; bwd b0
4.08, b1 3.02 (dkdv 1.92 + dq 1.05), b2 3.03 (dv 0.72 + dk 1.16 + dq 1.05), bwd quant 0.350.

**Training gate (1 GPU)**: DSv3 proxy 1-GPU variant (EP1, 32 experts, MBS 2, GBS 16, mock data), 200 iterations, same
seed: final-window loss hk 0.0319, mx 0.0320, mxf 0.0320, mxb 0.0319; mean |dloss| vs hk 0.0002-0.0003, grad-norm ratio
max 1.07 (mx) / 1.13 (mxf); median iteration hk 739 ms, mx 785, mxf 747, mxb 784 (MX not faster yet). Mock data
drives the loss to ~0.03, so this is a weak quality test (no tokenized data on this node).

**ATT of m4** (`scratch_mx/att.sh`, `py/att_tl.py`, `py/att_trip.py`): 3056 cycles per 128-kv tile and wave (WMMA
floor ~800). Per tile: 271 exps at >= 2 cycles each, issued in bursts of 8 between WMMA pairs (XDL idle half the
time); the rescale decision ~94 cycles per q tile (v_cmp -> SGPR -> s_cmp ~55, s_cmp -> branch 13, taken branch 25);
TDM issue ~300 (s_wait_tensorcnt 0xa that LLVM puts before reused descriptor SGPRs); 64 v_nop.

**Issue-rate micro** (`scratch_mx/micro/gen_issue*.py`, ATT): v_exp_f32 alone 1 / 2 cycles, alternating with VALU or
packed VALU 1 / cycle each; v_exp_f16 / bf16 half rate (4 cycles); MX 16x16x128 WMMA back to back 8 cycles (also
fp8 16x16x128 and bf16 16x16x32: 8); with WMMAs in flight other instructions issue at ~1 / 2 cycles: W + 3 T 10.7,
W + 6 V 12.9, W + 3 T + 3 V 12.8 cycles per window (also with v_nop / v_mov fillers: 12.5); matrix_a/b_reuse changes
nothing. Floor for 100 W + 256 T + ~210 V + ~120 LDS per tile: ~1350-1450 cycles.

**Rung m6** (`kern/k_mla_fwd_mx6.hip`, recipe m4 operands; emulation `emu_fwd6`): branch-free softmax with per-lane
integer shifts: exp2(S'') does not wait for the max (the shift goes into the P conversion scale); O and the row sum
accumulate relative to a fixed per-lane reference with the PV WMMAs' per-lane E8M0 B scale 2^(D - PE), so O is never
rescaled (guard when D > 48, once per tile); no branch per q tile; TAU2 0.75, PE 8. Numerics (CPU emulation, o / lse
dB vs fp32): randn m4 23.74 / 50.78, m6 23.49 / 50.54; kbias 23.95 -> 23.71; peaked 20.97 -> 20.94; spiky 22.40 ->
22.34 (the max element of P is no longer a power of two; P headroom variants all equal). GPU: 51-55 dB vs emu6,
toy..prod pass. Scheduling: sched_group_barrier windows [WMMA, VALU, TRANS, VALU, TRANS (, TRANS)] per slot (SCHED 1)
beat m4-style fences (SCHED 0: 0.751 ms). A peeled first tile costs ~300 VGPR spills (FIRST_PEEL 0: the first tile is
a runtime flag; unshifted exps need |S'| < 126 on the first tile). m6xp 0.680 -> 0.673 ms (pvs formed per PV group:
a long-lived WMMA scale VGPR must sit in v0..v255).

**XCD locality**: TDM-only skeleton of m6 (TDM_ONLY 1) = 0.242 ms: 2.9 GB moved per call; with the default order the
16 q blocks of a (b, h) go to 8 XCDs (dispatch L -> XCD L % 8), each XCD runs 16 (b, h) at once (~21 MB > 4 MB L2).
XCD_MAP 1 (k-th workgroup of XCD c takes q block nqb - 1 - k % nqb of bh = c + 8 (k / nqb)): TDM-only 0.172 ms, **m6xp
0.525 / 0.531 ms** (bf16 HK fwd 0.630). The same mapping should help the bf16 HK and the MX backward kernels.

**m6 register / shell fixes** (each measured by ATT, b2 s4096 h128):
- the 8-VGPR -m tuple per q tile replaced by one float m (the tuple formed per QK group): 1010 -> 930 VGPRs.
- the overflow guard (rescale O when D > 48) out of the tile loop, between chunks of GCHUNK 4 tiles: a branch with
  a big body inside the tile loop spilled the K-scale pairs (WMMA scale operands must sit in v0..v255; spill reload
  wait ~290-390 cycles per trip). The max may rise by <= ~56 log2 units (39 nats) within 4 tiles after a check.
- TDM: LLVM puts s_wait_tensorcnt 0xa before every TDM op (also in straight-line code: a per-wave cap of ~11 in
  flight); with 5 ops per tile on wave 0 (waves 1-3 NULL) and 3 tiles ahead, wave 0 waited 200+ cycles per tile.
  TDM_SPLIT: 2 TDM instructions per wave and tile (wave 0 K nope, 1 K rope + K scales, 2 V, 3 V scales), the wave's
  descriptor constants formed once; tensor_wait<4>.
- epilogue: the lse store by both lane halves (no exec change; each exec change waited s_wait_xcnt 0 for the
  in-flight output stores, ~4k cycles per block).
- result: 2548 cycles / trip (from 3056 in m4), no spill in the tile loop; **m6xp 0.475 / 0.479 ms** (bf16 HK 0.629 /
  0.636, m4 0.704); TDM-only skeleton 0.175 ms.
- ATT per block (m6c): only 57% of a wave's time is in the tile loop: per block ~18-22k cycles outside it (prologue:
  initial TDM wait ~1.3k; two diagonal tiles; epilogue) plus 1-3k cycles between blocks.
- epilogue: O through LDS (OSTAGE 1): bf16 rows written to a staging block (256 x 256 B after the Q region; LDS
  304 KB), barrier, one `tensor_store_from_lds` by wave 0. The scattered SBHD global stores (16 rows 64 KB apart per
  instruction) held their data VGPRs for hundreds of cycles each (s_wait_xcnt ~5.6k cycles per block). Probe
  (`scratch_mx/probe/tst.hip`): the TDM store does not honour LDS padding (rows after the first come out wrong), so
  the staging rows are unpadded (the ds_stores of 16 rows at one column conflict; still far cheaper). **m6xp 0.441 /
  0.447 ms**.

**Backward rung b3** (`kern/k_mla_bwd_mx3.hip` = b1 + the forward's two fixes): XCD-local workgroup order in k_dkdv_mx
(kv blocks of a (b, h) on one XCD, longest first) and k_dq_mx (as the forward), and the TDM ops spread over the waves
(dkdv: 11 per step -> 3 per wave, tensor_wait<3>; dq: 7 per tile -> 2 per wave, the block prologue 7 -> 2 per wave;
descriptor constants formed once, NULL descriptors past the end). No spills (749 / 910 VGPRs). Toy..mid pass with
b1's SNRs. b2 s4096 h128: dkdv 1.903 -> 1.284 ms, dq 1.065 -> 0.799, **backward 3.020 -> 2.132 ms** (bf16 HK 1.98).
- b3 epilogues through LDS (OSTAGE 1): dK / dV / dQ rows as bf16 into the idle ring, one TDM store each (dK wave 0,
  dV wave 1, dQ wave 0; rows >= S clipped by the descriptor, no exec change). dkdv 1.281 -> 1.139 ms, dq 0.798 ->
  0.736, **b3 backward 1.922 ms** (bf16 HK 1.97). b4 (= b2 + the same order / TDM split, no staging): dv 0.508 + dk
  0.800 + dq 0.800 = 2.19 (b3 at that point 2.14): the dV / dK split does not pay yet.
- LDS micro (`scratch_mx/micro/gen_lds.py`, ATT): with 4 waves loading, a ds_load costs ~6.2 cycles per wave whatever
  its width (b128 / b64): the instruction count, not the bytes, is the limit (~245 B/cycle/CU for b128). k_dkdv_mx
  issues 243 per 128-row step (~1500 of its 3700 cycles), the forward ~120 per tile (~750 of 2233).
