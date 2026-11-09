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
- b3 software pipelining (ATT: loads issued just before use, s_wait_dscnt 67-139 cycles per q16 tile; PF_A 1 let the
  scheduler sink each A-fragment load under the WMMAs into the registers they still read, ~36-cycle write-after-read
  waits, 20 per step): dkdv PF_A 2 (4 buffers, loads two groups ahead, fenced before the WMMAs) + PF_Q 1 (the next q16
  tile's loads fenced ahead of this tile's math): dkdv 1.139 -> 0.979 ms; dq PF_KV 1 (pair m + 1's K / V fragments
  fenced ahead of pair m, the dQ phase's K^T fragments 4 buffers / two ahead, the first two during pair 3): dq 0.741
  -> 0.671. **b3 backward 1.707 ms** (bf16 HK 1.981). VGPRs 764 / 933, no spills.
- forward rung m7 (`kern/k_mla_fwd_mx7.hip` = m6 defaults only): two q blocks per workgroup (nqb - 1 - p and p of one
  (b, h): 34 tiles each for nqb 16), the ring index running on across them (the first block's last three tile slots
  load the second block's first three tiles; the same kv tiles), pairs of a (b, h) on one XCD; the block body one
  instantiation inside a loop (two calls duplicated it: ~260 spills). Host grid (nqb / 2, H, B) when nqb % 2 == 0 and
  B H % 8 == 0 (pair_mode), else m6's. 0.440 -> 0.432 ms. QKFIRST: in slots 1, 2 the QK(q_{i+1}) WMMAs before PV(q_{i-1})
  (P(q_{i-1}) is converted at the very end of slot i - 1; ATT showed slot 1 opening with ~40 VALU while its PVs waited):
  **0.426 ms**. Package: forward m7 (mx.py launches the paired grid), pytest 28 pass.
- ATT of the pipelined b3: dkdv 2805 cycles / step (from 3714), now LDS-bound (240 loads at ~4.9 cycles, 39%); dq 3683
  cycles / tile. b4 with b3's pipelining and staging (NJ_DK 3: NJ 4 spills 28): dv 0.420 + dk 0.720 = 1.14 ms vs b3's
  dkdv 0.978: the S recomputation in both kernels does not pay; b3 stays.
- dq fragment addresses pinned per tile (bases + immediate offsets): ATT showed a v_add3 per dQ group computing the next
  K^T fragment address into a register the group's WMMAs still read (~31-cycle write-after-read stall + 4 v_nop, 12 per
  tile). dq 0.669 -> 0.625 ms, **b3 backward 1.651 ms** (bf16 HK 1.971).
- m7 ATT: 80% of the wave time in the tile loop now (m6 58%); per workgroup (two blocks) ~19.5k cycles outside it:
  first-block prologue TDM ~3.5-4k (the TDM issue queue stalls the wave), epilogue staging ds_stores ~1.5k per block
  (16-way bank conflicts: the TDM store forbids padding), the last block's TDM-store wait ~1k. PRO_PARTIAL (the first
  block's Q before its tiles, then tensor_wait<4>: compute starts once Q and tile 0 landed): 0.4246 -> 0.4238 ms (noise
  level; kept). The loop: 2236 cycles / tile, still with VALU-only runs (the max3 of q_i at the slot start) and WMMA-only
  runs where the sched_group_barrier windows meet dependences.
- dq DQ_SGB (sched_group_barrier windows [W, 2 T, 3 V] in the pair phase): 0.631 -> 0.703 ms, kept off.
- b3 ATT (per wave): dkdv 71% in the step loop, ~16.6k cycles outside (prologue TDM issue stalls ~3.4k: the wave's
  TDM queue is shallow and every block starts by pulling ~200 KB; staging 0.9k; final TDM-store wait 1.4k); dq 64%,
  ~27.6k outside (prologue ~5.3k, two diagonal tiles, staging 0.85k, final store 1.8k). Pairing blocks as in m7 would
  hide prologues and stores, but LDS is full (dq 320 KB: the second block's Q region and the first block's dQ staging
  collide); needs an LDS plan.
- dq DQ_PRO_PARTIAL (start on the Q / dO region + tile 0): 0.6256 -> 0.6253 ms (kept, noise level).
- **Training gate g200c** (package m7 + b3 as of 5c57ffdd, 1-GPU DSv3 proxy, 200 iterations): mx 735.0 vs hk 739.7
  ms / iteration (-0.6%, quantization included); final-window loss 0.0320 vs 0.0319. Earlier: m4 + b1 785.3 (g200),
  m6 + b3 without prefetch 745.6 (g200b).
- m7 DMASK (static mask classes for the two diagonal tiles: per (q16, k16) sub-tile fully kept / fully masked / mixed
  known at compile time, fully masked q tiles of the last diagonal tile skipped; DSKIP / DSOFT select either half):
  correct (toy..prod, the extra-shape sweep `scratch_mx/py/shapes_fwd.py`: xcd / paired / plain orders all pass), but
  either half perturbs the global register allocation (150-200 VGPR spills): 0.453 vs 0.423 ms. Kept off (DMASK 0 =
  the committed m7 code, 976 VGPRs, 1 spill).

## 2026-11-05 (c): backward write-after-read waits, rung b5 (cross-step pipelined dK / dV)

- b3 dkdv experiments (`kern/k_mla_bwd_mx3.hip` macros): QPIPE 1 (q16 tiles software-pipelined: S / dP WMMAs of tile
  qt + 1 beside the VALU of tile qt) 0.979 -> 0.972 ms; with sched_group_barrier windows (QSGB, 4 patterns) 1.01-1.05;
  QPIPE 2 (3 fragment buffers) slower. Timing probes (wrong numerics): without the per-tile lse / delta reloads -2.4%,
  without the per-tile scale reloads -2.9%, both -6.3% (the b32 scale loads cost more per instruction than b128).
- ATT (`py/att_tl.py` gap pairs): W -> L median 37 cycles, ~19 per step (~490 cycles / step): the dV / dK A-fragment
  loads went into the registers the previous group's WMMAs were still reading. The 4-buffer source scheme does not
  constrain the physical registers: a dead fragment's registers are reused at once and the allocator does not model
  the write-after-read wait. KEEP_WAR: an empty asm use of fragment d - 1 after the loads of group d (it stays
  allocated until then): dkdv 0.979 -> 0.947 ms, same 764 VGPRs. For QPIPE's q16 tiles (a whole QTile kept) it spills.
  PF_AD 3 / 4 (deeper A prefetch), QFENCE2 0: within 1%.
- **Rung b5** (`kern/k_mla_bwd_mx5.hip`): k_dkdv_mx pipelined across steps: the eight q16-tile slots of step s also
  issue the 20 dV / dK groups of step s - 1 (3, 3, 3, 3, 2, 2, 2, 2 per slot; their A fragments loaded one slot ahead,
  the previous slot's kept alive), P / dS of the previous step carried in registers; LDS ring split into a q part
  (Q / dO rows, scales, lse, delta) and an A part (dO^T, Q^T, scales), 3 stages each, TDM round r = q part of step r +
  A part of step r - 1, issued at the start of step r - 2 (two steps of lead for both). b3 ran each step's A phase
  after its q tiles with all four waves pulling ~200 KB of A fragments at once (~40-cycle load waits per group).
  dkdv 0.947 -> 0.909 ms (2361 cycles / step from 2598; 918 VGPRs, 1 spill in the peeled step 0). The step is now
  issue-bound with the stalls spread: L 714, T 399, V 390, W 354, S 188, C 155 cycles / step (230 loads at ~3.1).
- not taken: -fno-slp-vectorize (the pk_mul of dS = P dP needs operand pairs: ~120 v_mov per step; without SLP 784
  VGPRs, 188 fewer instructions, but dkdv only 0.904 and dq slower 0.644); lse / delta by buffer loads LTDT_D tiles
  ahead (1.000 / 0.978 / 0.962 ms for D 2 / 3 / 4: global latency).
- dq (b5 file, b3's kernel): the same wait on the dQ phase's K^T fragments (W -> L median 47 cycles, ~10 per tile):
  DQ_KEEP 1: dq 0.624 -> 0.598 ms (2975 cycles / tile from 3248), no extra VGPRs. Keeping the K / V pairs too: 112
  spills; their loads moved behind a fence after q tile 1..3 of the pair: 0.611-0.649 (slower).
- **b5 backward 1.584 ms** (b3 1.651; dkdv 0.909 + dq 0.598; bf16 HK 1.971). Toy..mid pass with b3's SNRs. Package:
  backward b5 (hip/k_mla_bwd_mx5.hip), pytest 28 pass.

## 2026-11-05 (d): quantization through TDM (the HBM limit was mis-measured)

- Bandwidth: torch's `copy_` reaches only 7.7 TB/s on gfx1250; a Triton copy of 1 GiB 18.5 TB/s, Megatron's MLA kv
  RoPE kernel 15.6 TB/s (`scratch_mx/py/bw2.py`). The quantization kernels (6.4-8.7 TB/s) were not near the limit.
- Package per call (`scratch_mx/py/pkg_time.py`, b2 s4096 h128): hk_mx forward 0.769 ms (quantization 0.334) vs bf16
  HK 0.625; backward 1.656 (dout quantization 0.097) vs 1.969.
- `k_mxq_rct192/128` (rows + cols in one pass, the 128-row bf16 tile by one TDM load, the four outputs staged in LDS and
  written by four TDM stores; bit-exact with k_mxq_rc): q 0.125 -> 0.095, k 0.119 -> 0.087, v 0.075 -> 0.049, dO
  0.097 -> 0.065 ms. Probes: TDM traffic alone 0.054 ms (q), the compute alone 0.070: compute-latency bound. RC_FAST:
  tree reductions (the per-block amax and the row dots were 32-long dependent chains), the kbar slice in registers,
  the 8-lane dot reduction by DPP instead of ds_bpermute: q 0.081, k 0.078, v 0.051, dO 0.070 (codes / scales still
  bit-exact; aux within 1e-6). A persistent variant (`k_mxq_rcp`: 3 input buffers, 2 staging sets, one 512-thread
  workgroup per CU) is slower (q 0.104): kept in the lab only.
- Package: quantization on k_mxq_rct; forward 0.672 ms (quantization 0.214), backward 1.670 (dout quantization 0.069);
  pytest 28 pass.
- k_mxq_kbar: the row loads of a thread issued 8 at a time (same summation order: bit-exact): 0.021 -> 0.020 ms.
- dkdv and dq on two streams: slower (1.70 vs 1.57 ms back to back; isolated sums 1.46: the difference is cache
  warmth, which the real backward pays too).
- **Training gate g120d** (package m7 + b5 + k_mxq_rct, 120 iterations): mx 727.5 vs hk 738.3 ms / iteration (-1.5%;
  before: -0.6%); final-window loss 0.0818 both, grad-norm ratio max 1.023.

## 2026-11-05 (e): MLA RoPE fused into the MX quantization and the backward epilogues (plan P4, first producer)

Megatron's apply_rope_fusion path (MI455X DSv3 config): `fused_apply_mla_rope_for_q` rotates q's d 128..191 in place
(0.060 ms), `fused_apply_mla_rope_for_kv` writes bf16 key / value copies (0.079 ms), the core attention then
quantizes query / key / value; the backward's RoPE kernels (kv 0.19 ms incl. the head sum of the k_pe gradient)
copy dk / dv into the kv gradient. Fused path (`mx.mla_rope_attention(q, kv, k_pe, cos, sin, scale)`, SBHD, autograd):
- `k_mxq_rctr192`: rows + cols of [x nope | rope(pe)]: the nope part by one TDM load into 384-byte LDS rows (pad 128 B
  after every 256), the pre-RoPE pe part (q: x + 128; k: k_pe with head stride 0) by a second TDM load into the X8T
  staging (free until the cols phase), the rotation in fp32 rounded to bf16 written into d 128..191, then rc_tile.
  q 0.096 ms (rct on the roped query 0.082), k 0.091 (0.078). `k_mxq_kbar_rope`: kbar of the roped k (0.016).
  Codes / scales / kbar bit-exact against torch RoPE (per-op fp32) + k_mxq_rct (`scratch_mx/py/rope_q_test.py`).
- backward: `k_dkdv_mxr` (b5's dkdv body, template SPLIT): dK's nope columns and dV straight into the kv gradient
  [s, b, h, 256] (one TDM store each), dK's rope columns into a per-head [s, b, h, 64] buffer (a third store);
  `k_rope_bwd_q` (inverse rotation of dq's d 128..191 in place, 0.030 ms), `k_rope_bwd_kpe` (fixed-order fp32 sum over
  the heads, inverse rotation: 0.089 -> 0.015 ms with 16-byte loads of 4 heads at a time). dkv bit-exact, the per-head
  buffer bit-exact, dq within 1 bf16 ulp (the compiler fuses a multiply-add; `#pragma clang fp contract(off)` does
  not override HIP's -ffp-contract=fast), dk_pe within the reference's own bf16 rounding of the head sum.
- Package op vs (torch RoPE + hk_mx): out bit-exact, dkv bit-exact, dq 117 dB / bit-exact on small shapes
  (`scratch_mx/py/rope_pkg_test.py`; pytest `test_rope_fused_matches_unfused`). Forward + backward through autograd
  at the proxy shape: 2.668 vs 2.865 ms (Megatron's RoPE kernels + hk_mx): -0.20 ms per layer and micro-batch.
- Primus (local worktree branch, not pushed): opt-in patch `PRIMUS_TURBO_MLA_ROPE_FUSION=1` wraps
  MultiLatentAttention.forward with the fused path for eligible calls (training-style, no inference / packed / CP /
  TP / SP / recompute / offload, YaRN + apply_rope_fusion, causal, 128 + 64 / 128 heads, s % 256 == 0), else
  Megatron's forward.
- **Training gate g120f** (120 iterations, same seed): bf16 HK 740.0, hk_mx 729.9 (-1.4%), **hk_mx + fused RoPE
  (mxr: Primus patch d909dab0 on local branch dev/lhz/mla_rope_fused_mx) 720.7 ms / iteration (-2.6%)**; final-window
  loss 0.0818 in all three, grad-norm ratio max 1.024; the log shows the fused path active, no fallback.

## 2026-11-05 (f): forward ablations, softmax tail sinking

- m7 timing ablations (`PROBE` bits, wrong results; base 0.425 ms): no exps 0.372, no max 0.392, no P conversions
  0.380, none of the three 0.212, no PV WMMAs 0.265, no K fragment loads 0.252, no V loads 0.407 (no QK WMMAs: slower,
  not interpretable). The costs interact: the loop is latency / dependency bound, not unit bound.
- ATT of slot 3: from the barrier to the first WMMA ~265 cycles per tile with no WMMA: the next tile's 56 K-fragment
  loads, then softmax(q2)'s tail (the lane max chains, the shift decision, the 8 P conversions) that PV(q2) needs.
  LLVM sank that tail past slot 2's sched_barrier into slot 3, next to its use (sched_barrier binds the machine
  scheduler, not the sinking passes). PIN_P 2 (an empty asm use of P(q_i) at the end of slot i): 0.425 -> 0.420 ms
  (2 spills); pinning P + D + m spills 14 (0.441); a tree-shaped lane max (MAXTREE): no gain.
- dkdv: the first TDM rounds before the resident K / V loads (PRO_TDM_FIRST), per-wave TDM stores of the staged rows
  without the barrier (EPI_WAVE): neutral (0.908-0.912), off. Per workgroup ~20% of the time is outside the step loop
  (steady state ~4.5k cycles before the first loop step incl. TDM issue stalls and the first wait, ~6k after:
  the last A phase, staging, the store wait); only one workgroup fits a CU, so it is exposed.
- Package per call now: forward 0.655 ms (quantization 0.210) vs bf16 HK 0.629; backward 1.646 vs 1.974.
- dkdv TDM_INC (lab b5 only, not yet in the package): each wave's 3 TDM ops keep their next step, global address and
  LDS stage in SGPRs and advance per round (the per-round descriptor math was ~87 SALU / step): dkdv 0.911 -> 0.903 ms,
  backward 1.563 -> 1.560; toy..mid and the fused-RoPE backward test pass. To do: copy to the package + pytest.

## State at the 2026-11-05 stop (~13:00 UTC; GPU handed over)

Package (branch head): forward m7 (PIN_P 2) 0.420 ms kernel, 0.655 ms per call with quantization (bf16 HK 0.629);
backward b5 1.584 (lab with TDM_INC 1.560), 1.646 per call (bf16 HK 1.974); fused RoPE path mla_rope_attention
(-0.20 ms per layer call vs Megatron's RoPE kernels + hk_mx). Training gate g120f: bf16 HK 740.0, hk_mx 729.9, hk_mx +
fused RoPE 720.7 ms / iteration, same loss. Next: (1) TDM_INC into the package; (2) the per-workgroup overhead of dkdv
(~20%: needs a persistent / paired design with an LDS plan for the 80 KB staging) and dq (~10%); (3) generated asm
schedules (port tools/asmkit + fwdgen / bwdgen to the MX kernels): the loops are 1.6-1.8x their issue floors;
(4) the fused-RoPE quantization kernels (q 0.096, k 0.091 ms vs ~0.055 memory floor).

## 2026-11-06: persistent dK / dV (b6, negative), TDM_INC into the package

GPU note: this morning the GPU runs ~1.5x slower than on 11-05 for every kernel (b5 dkdv 1.386 vs 0.903 ms, dq 0.93
vs 0.60; gfx clocks ~1300 MHz at the idle reading, 256 CUs present; another session is probing clocks). Only A / B
comparisons within one run are meaningful today.

- Package: b5 with TDM_INC (incremental TDM descriptors); pytest 30 pass, fused-RoPE package test pass.
- **b6** (`kern/k_mla_bwd_mx6.hip`, lab only): k_dkdv_mx persistent (grid 8 x workgroups per XCD, one per CU;
  workgroup x + 8 i takes items i, i + NW/8 .. of XCD x; item = ((b, h), pair p) -> kv blocks p and S/128 - 1 - p, S/128 + 1
  steps per item; q ring 3 stages, A ring 2 stages (round g = A part of step g + q part of step g + 2), an 80 KB staging
  area of its own: 319.5 KB). Bit-exact with b5 at five shapes (`scratch_mx/py/b6_test.py`), but slower every way:
  - one loop with the masked first step (+ the previous block's last dV / dK) and the normal step as two
    instantiations: 154 spills, 2.01 vs 1.39 ms (b5);
  - one step instantiation with a runtime diagonal mask (uniform branch per q16 tile): 881 VGPRs, 1 spill, but ~195
    SALU and 38 branches per step (TDM cursors, mask conditions with VALU -> SALU dependences): 1.72 ms;
  - nested loops (per block: the masked first step, the step loop, the last dV / dK, the epilogue; TDM rounds running
    on across blocks): 32 spills, two 16-VGPR tuples stored and reloaded every step in the low bank (v0..v255, where
    the DS addresses and WMMA scale operands must be): 2.25 ms. Allocator flags (no regclass priority, split-spill
    size, no misched) do not help.
  The persistent structure needs more low-bank registers than b5 leaves; parked (would need hand-assigned registers).
- LLVM machine scheduler `-misched=gcn-max-ilp` (lab tag `xi`, now in the package FLAGS): dq 0.930 -> 0.896 (-3.7%),
  forward m7 0.673 -> 0.659 (-2%), dkdv and the quantization kernels neutral (same-run A / B on the slow GPU); pytest
  30 pass, fused-RoPE package test pass. Package per call on the slow GPU: forward 1.004 (quantization 0.326), backward
  2.463 (+0.097 quantization) vs bf16 HK 0.862 / 2.431.
- The slow GPU: sustained MX backward 2.436 ms / call (1.54x the 11-05 1.584); during another session's run the gfx
  clocks sit at ~1304 MHz (max 2400) at ~850 W of a 2500 W limit, no throttle / violation status active. The bf16 HK
  kernels slow by less (fwd 1.37x, bwd 1.23x) than the MX ones (fwd 1.53x, bwd 1.50x): MX is closer to clock-bound.

## 2026-11-06 (b): why the dK / dV loop does not respond to scheduling

Everything below on the slow GPU (b5 dkdv ~1.385 ms, same-run A / B only).
- LLVM scheduling strategies: `-amdgpu-sched-strategy=coexec` (gfx1250 / gfx950 co-execution strategy): dkdv
  1.396 vs 1.385, dq 0.961 vs 0.896, forward neutral; `-amdgpu-enable-vopd=0`: +0.7 / +0.8 / +0.3%.
- **`py/resched.py`** (new lab tool): post-RA re-emission of an LLVM-compiled loop body (one straight-line block)
  through asmkit (`tools/asmkit`: s_set_vgpr_msb, expert-mode waits and hazard v_nops derived for the new order,
  fixed point over the back edge, a full drain at the loop entry / exit, per-kernel verify_obj). WMMA scale operands
  are not remapped by s_set_vgpr_msb (field map `-`). The identity round trip (`--mode identity`) is bit-exact and
  as fast as LLVM's build (1.398 vs 1.399). List schedules on a per-wave machine model (`--mode list`, `--prio`)
  are correct but never faster (1.389-1.51 ms): the model (WMMA windows, TRANS rate, LDS bucket, the measured
  s_set_vgpr_msb rule: an instruction that needs another mode issues >= 4 cycles after a DS instruction) predicts
  the LLVM order within 5% (`--calib <ATT dir>`: 2405 vs 2284 cycles / step) but not the effect of re-ordering.
- What the step costs (ATT, `scratch_mx/py/att_prof.py`, `att_msb.py`, `att_split.py`): ~2290 cycles / step and
  wave; the issue is close to a sum of per-instruction costs: 227 LDS loads at ~2.3-2.4 cycles each (issued in runs;
  dropping all 192 b128 loads as a probe: 2293 -> 1858 cycles / step, i.e. 2.3 cycles per load; kernel only -8.5%),
  88 WMMAs that hide ~4 other instructions each, ~450 VALU + 128 exp at 1-2 cycles. LLVM sinks the dS quantization
  of all four tile pairs (58 max3, 36 pk_mul, 18 cvt, 152 operand-forming v_mov) into the last slot: a ~550-cycle
  VALU-only tail. `-fno-slp-vectorize` keeps it in its slots (782 VGPRs instead of 920, no tail): dkdv unchanged
  (1.387), dq slower (0.932). Pinning the quantized values with an empty asm (PIN_Q) costs an
  `s_wait_alu depctr_va_vdst(0)` per pin (expert mode waits before every inline asm): 2.6-4.0 ms. A ping-pong
  2-step unroll (UNROLL2, no loop-carried P / dS copies) spills (1024 VGPRs + 756 B scratch).
- Per workgroup ~13k cycles outside the loop (32 workgroups per CU, ~27% of the kernel): first TDM round + barrier
  ~1.5-3k, the epilogue's bank-conflicted staging ds_stores ~2k (the TDM store forbids padding: 8-16-way), the
  drain before its barrier ~1.3k, the final TDM-store wait ~2k. Probes: no final wait (NO_FINAL_WAIT, unsafe):
  1.386 vs 1.383 (not on the critical path); dK / dV by global stores straight from the accumulators (EPI_DIRECT,
  correct): 1.71 ms (32-byte row pieces per instruction).

## 2026-11-06 (c): quantization kernels: bf16-native codes, no VOPD; dq static diagonal classes

- The quantization kernels slow down with the clocks (q 0.096 -> 0.143 ms on the slow GPU, Triton copy still 17.9
  TB/s): compute-latency bound. Probes today (rct q): TDM traffic alone 0.077 ms (~10 TB/s: per workgroup load ->
  wait -> compute -> store -> wait, 3 workgroups per CU), compute alone 0.087, both 0.124. ATT (`k_mxq_rctr192`):
  ~13-15k cycles per wave: kbar global load ~1k, TDM load wait ~1.5-2.3k, compute ~7k, TDM stores + wait ~2-3k;
  in the cols phase LLVM re-materializes 0xffff0000 into an SGPR before every VOPD bitop (209 s_mov_b32 / wave).
- **RC_BF16** (`k_mxq.hip`, now default): MX blocks straight from bf16 where no fp32 pre-processing is needed (rows
  without kbar / Y dots / pre: v; cols without kbar: q, v, dO): |x| max on the magnitude bits (v_pk_max_i16), codes by
  `v_cvt_scalef32_pk8_fp8_bf16`, the column pairs by v_perm_b32. Bit-exact (codes, scales, aux) with the f32 path.
  gfx1250 has no usable v_dot2_f32_bf16 builtin (needs dot12 / dot13), so dO rows keep the f32 path.
  With `-amdgpu-enable-vopd=0` for k_mxq.hip (package `SRC_FLAGS`): lab q-rope 0.145 -> 0.142, k-rope 0.139 -> 0.129,
  v 0.077 -> 0.071, dO 0.102 -> 0.097. Package per call: forward 1.004 -> 0.978 ms (quantization 0.326 -> 0.311),
  dout quantization 0.097 -> 0.089; pytest 30 pass, fused-RoPE package test pass.
- dq DQ_DSTAT (lab, off): static mask classes on the two diagonal kv tiles (kv tile n - 1: q tiles i < 2 fully masked
  -> skipped incl. their dQ WMMAs; kv tile n - 2: q tiles i >= 2 kept without the mask): correct, dq 0.895 -> 0.888.
- b6 (persistent dK / dV) without SLP: 804 (nested) / 883 VGPRs, no low-bank spills, bit-exact with b5 at five shapes,
  but still slower (1.79 / 1.74 vs 1.39 ms). ATT: the steady-state step is ~3230 cycles (b5 ~2290): with its own 80 KB
  staging area the A ring has 2 stages, so round g carries the A part of step g, needed at step g + 1 (one step of
  lead instead of b5's two): the TDM latency is exposed every step. Needs ~45 KB more LDS than exists; parked.
- Forward without SLP: 958 VGPRs + 8 spills, 0.721 vs 0.656 ms; DMASK + no SLP (936 VGPRs, no spill, correct): 0.666.
- 64-row quantization tiles (128 threads, ~49 KB LDS: twice the workgroups per CU): 30-50% slower (the cols' 32-byte
  TDM rows and the per-workgroup fixed work); reverted.
- resched `--coalesce` (a copy whose source one instruction defines is removed, that instruction writing the copy's
  destination): only 2 of the dkdv loop's 195 copies qualify; the rest re-shape tuples (exp results feeding the cvt
  8-tuple and the pk_mul pairs, WMMA / pk_mul / cvt outputs split across tuples): a register re-allocation, and the
  no-SLP build (no such copies) is not faster anyway.

## 2026-11-06 (d): the forward's tile loop re-scheduled after register allocation (package)

- `py/resched.py` on the forward (k_mla_fwd_mx's tile loop: one block, 786 real instructions: 100 W, 256 T, 266 V,
  114 L): every list schedule passes toy..prod and is faster than LLVM's order, unlike dK / dV (LDS-bound: LLVM's
  order stays, `--prio ready` 1.44-1.50 vs 1.386 ms). Forward (slow GPU, same run): LLVM 0.658-0.660, identity
  re-emission 0.652, height 0.646-0.649, orig 0.632-0.641, **first-ready-first (`--prio ready`) 0.610-0.618**;
  assumed LDS latency 24 / 32 / 48 cycles 0.6105 / 0.6094 / 0.6101 (96: 0.617), LDS gap 6 / burst 4 best (gap 10 /
  burst 2: 0.646).
- Package: `asm_resched.py` (the schedule + asmkit emission + per-kernel verify_obj, asmkit loaded from
  gfx1250_mla/tools by path), `mx.RESCHED = {"k_mla_fwd_mx7.hip": ("k_mla_fwd_mx",)}`: the forward source is
  compiled with -amdgpu-enable-vopd=0 -save-temps, its tile loop re-scheduled (first-ready-first, LDS latency 32), the
  code object assembled with the driver's commands and checked against the intended .s; any failure -> the plain
  build + a warning; `PRIMUS_TURBO_MLA_MX_RESCHED=0` (common.envs) turns it off; the cache key covers asm_resched and
  asmkit. setup.py now ships gfx1250_mla_mx/hip and asmkit. pytest 30 pass, fused-RoPE package test bit-exact.
  Package forward per call, alternating A / B: 0.9544 vs 0.9835 ms (-3.0%; medians of 3 runs each).
- asmkit va_vdst rule of the emitted loop: LLVM's 0.6099, `llvm_war_xdl` (the bf16 HK backward generator's) 0.6063,
  the per-class bound `class` 0.6022 ms (all toy..prod pass). Package: `llvm_war_xdl` (validated on hardware by the
  bf16 generator; `class` is only asmkit's model). pytest 30 pass, fused-RoPE test bit-exact.
- dq with a one-block tile loop (DQ_WAIT0: the per-tile wait + barrier also before tile 0, lab, off) re-scheduled:
  identity 0.903, ready 0.943, orig 0.968, height 0.904 vs LLVM 0.896 ms: LDS-heavy like dK / dV, LLVM's order stays.
- Forward model knobs (WMMA co-issue offsets, DS gap, TRANS occupancy): the package settings are on the plateau
  (0.6056-0.6059 ms); all offsets 0.614, TRANS 1 / 3 cycles 0.621 / 0.622.
- ATT: the forward tile loop 2052 -> 1768 cycles / tile (exp / VALU now alternate); outside the loop: per workgroup
  ~5-8k prologue (TDM queue + first data) and per block ~11-13k for the peeled diagonal tiles + epilogue: one
  straight-line block of ~2940 instructions (184 WMMA, 520 exp, ~1050 mask compares / selects).
- That block re-scheduled too (resched `--region`; package: the kernel's largest straight-line block with WMMAs
  besides the loop): 0.6059 -> 0.6007 ms (LLVM 0.659: -8.9% in all). Fixes on the way (caught by verify_obj): memory
  operand msb fields by position (scratch_store's data is operand 1 when the address is `off`), VOP3 with an SGPR dst
  (v_div_scale, carries) shift their source fields; asmkit derives no va_sdst / va_vcc waits, so a SALU / memory
  reader of an SGPR last written by a VALU gets an explicit `s_wait_alu depctr_va_sdst(0)` (one in the region).
  Package: pytest 30 pass, fused-RoPE test bit-exact; forward per call A / B 0.9675 vs 0.9915 ms (medians of 3).
- dq's two peeled diagonal tiles (1648 / 2115-instruction blocks: +540 / +1007 mask instructions over the loop body)
  re-scheduled (`--region`, the loop untouched): height 0.894 vs the no-VOPD build 0.902 / LLVM with VOPD 0.8965:
  within noise of the default build; not taken (the re-emission needs the no-VOPD build).
- Forward schedule search: first-ready-first with seeded random tie-breaking (12 seeds, jitter 8 / 24 cycles):
  0.626-0.636 vs 0.601 ms for the deterministic order (ties by LLVM's order): LLVM's order carries structure the model
  does not see; the package keeps the deterministic schedule.
- dq tail: on a traced SIMD 6% idle at the end of the dispatch (dkdv 0.1%): per XCD the last (b, h) pair's long q
  blocks finish last. m7-style pairing (q blocks i and 15 - i per workgroup: 34 tiles each) would remove it, but the
  dQ staging (96 KB) does not fit next to the next block's ring and Q / dO region (LDS full): parked.
- **RC_X8G** (`k_mxq.hip`, default): the row codes stored to global straight from the lanes (each lane's two 16-byte
  halves of an MX block; 8 lanes cover a row's 192 bytes: coalesced), no X8 staging and no X8 TDM store: LDS 101 ->
  77 KB (D 192), 4 workgroups per CU, with the register targets at 8 waves per SIMD (RCT_WPE192 / RCTR_WPE; rctr192
  at 128 VGPRs spills 2). Bit-exact. rct: q 0.1216 -> 0.1102, k 0.1166 -> 0.1057, v 0.0711 -> 0.0641, dO 0.0968 ->
  0.0950; fused RoPE: q 0.1432 -> 0.1404, k 0.1304 -> 0.1157. Package per call: forward 0.97 -> 0.93 ms (quantization
  0.310 -> 0.283), dout quantization 0.089 -> 0.086; fused-RoPE forward 1.0047 ms (1.0827 this morning). pytest 30
  pass, fused-RoPE test bit-exact.
- ATT of the new k_mxq_rctr192 (4 workgroups per CU): ~16.3k cycles per wave, now instruction-bound: VALU ~8.9k,
  SALU ~4.8k; ~2.2k of it exec handling of the rows phase (8 lanes per row, 6 live for D 192: v_cmpx /
  s_and_saveexec / s_or / branches around each per-lane `if`), the q . kbar dots (v_pk_fma ~27 cycles each).
- **RC_NODIV** (default): branch-free rows phase: the pad lanes recompute block NBLK - 1 (discarded) and write the pad
  scale bytes, the code offsets by selects (j < 4: chunk 128, else the 64-byte chunk at 128), only the two global code
  stores masked. Bit-exact. rct q 0.1102 -> 0.1029, k 0.1057 -> 0.0986 (v / dO unchanged: D 128 has no pad lanes);
  fused RoPE q 0.1393 -> 0.1323, k 0.1158 -> 0.1095. Package per call: forward 0.930 -> 0.903 ms (quantization 0.283
  -> 0.268); fused-RoPE forward 0.9712 ms (1.0827 this morning: -10.3%). pytest 30 pass, fused-RoPE test bit-exact.
- The pad lanes (which recompute block NBLK - 1 of their row) store those same bytes to the same addresses: no masked
  store region left in the rows phase. Codes / scales bit-exact; the fused-RoPE q kernel's aux (q . kbar) moves by
  ~1 ulp (LLVM contracts the dot differently in the changed code; within the tests' tolerance). Fused RoPE q 0.1333 ->
  0.1270, rct q / k -0.7 / -1%. pytest 30 pass, fused-RoPE test pass.
- k_mxq.hip without SLP (package SRC_FLAGS `-fno-slp-vectorize`; ~286 v_mov / v_mov_b64 per wave were tuple
  shuffles): fused-RoPE q 0.127 -> 0.124 ms, dout quantization 0.0952 -> 0.0924 (lab) / 0.0863 -> 0.0848 (package),
  k_rope_bwd_q 0.040 -> 0.036, k_rope_bwd_kpe 0.022 -> 0.018; k / v unchanged; codes bit-exact (aux by ulps).
  `-misched=gcn-max-ilp` for k_mxq: neutral. pytest 30 pass, fused-RoPE test pass.
- The package's re-scheduled forward was slower than the lab's (0.629 vs 0.600 ms in the lab harness, the same
  source): asm_resched's machine model lacked the s_set_vgpr_msb switch rule (>= 4 cycles after a DS instruction, >= 3
  after a WMMA). With it the package object runs 0.6002 ms (lab 0.6005, LLVM 0.6575), toy..prod pass. Package per
  call: forward 0.900 ms (1.004 this morning, -10.4%; bf16 HK 0.862), fused-RoPE forward 0.970, backward 2.452.
- `v_cvt_scalef32_pk8_fp8_{f32,bf16}` use only the exponent of the f32 scale operand (x / 2^floor(log2 scale), GPU
  test `scratch_mx/bt/cvtscale.py`): q's pre-scale cannot be folded into the conversion.
- Converter occupancy rules in the scheduler model (`--cvt-occ`, `--cvt-nowin`; ATT: a pk8 conversion blocks the
  next VALU / TRANS ~4 cycles, LDS / WMMA issue right after): 0.5999-0.6156 vs 0.6005 ms: no gain.
- dq workgroup order snake (DQ_ORDER 1: (b, h) groups alternate longest / shortest first): 0.917 vs 0.896 ms.
- RC_KBF16 (lab, off): kbar rounded to bf16 values, k - kbar by v_pk_add_bf16 and the bf16 MX path in rows and cols:
  codes equal MX(bf16(k - bf16(kbar))) (rows / cols, `scratch_mx/py/kbf16_check.py`), but only k -2.5..2.8% (fused-RoPE
  k 0.1096 -> 0.1065): not worth a new quantization semantics.
- dkdv EPI_PAD (lab, off): the epilogue's staging rows padded (pitch + 16 B: conflict-free ds_stores), each wave
  re-packs its 32 rows densely (32 consecutive 16-byte chunks per copy) into the idle q ring / the NULL A stage and
  TDM-stores its own rows, no barrier: bit-exact with b5 (five shapes), dkdv 1.392 vs 1.382 ms (+0.7%). With
  NO_FINAL_WAIT and EPI_WAVE also neutral, the epilogue's ATT costs (~2k staging + ~1.3k drain per wave) are not what
  bounds the kernel.
- Training gate g120h (all of today's package changes, slow GPU): bf16 HK 1001.6, hk_mx + fused RoPE 991.0 ms /
  iteration (-1.06%; this morning's g120g -0.67%), final-window loss 0.0818 both, grad-norm ratio max 1.025.
- dq DQ_ORDER 2 (per XCD every (b, h)'s long half of q blocks first, then the short halves: the dispatch tail made of
  short workgroups; each (b, h)'s K / V read in both passes): 0.896 -> 0.889 ms; with DQ_DSTAT 0.881 (-1.7%),
  bit-exact with b5 at five shapes. Both on in the package; pytest 30 pass, fused-RoPE test pass.
- dkdv DKDV_ORDER 2 (the same two-pass order: every (b, h)'s kv blocks 0 .. 15 first, then 16 .. 31): 1.389 -> 1.377
  ms (-0.8%), bit-exact. Package (on): pytest 30 pass, fused-RoPE test pass; per call forward 0.894, backward 2.410
  (bf16 HK 0.863 / 2.429 in the same run).
- The pass orders generalized (DKDV_ORDER / DQ_ORDER = number of passes): dkdv 2 / 4 / 8 passes 1.377 / 1.377 / 1.384,
  dq 0.881 / 0.875 / 0.880 ms (all bit-exact). Package: dkdv 2, dq 4 passes; pytest 30 pass, fused-RoPE test pass.
- dout quantization (rct128 mode 4: the dO . O row dots read O by per-lane global loads in the rows passes; ATT: the
  dot FMAs ~26 cycles each, branches / xcnt waits ~3k cycles per wave): RC_YPRE (O rows of both passes loaded before
  the tile's TDM wait): at the 128-VGPR target 36 B spills, 0.100 vs 0.092 ms; at 7 waves per SIMD (137 VGPRs, 3
  workgroups per CU) 0.0911 vs 0.0921. Off.
- **k_dq_mxr** (fused-RoPE path; `attn_bwd_rope` no longer launches k_rope_bwd_q): dQ's inverse RoPE in the dq
  epilogue. The pairs it combines (d 128 + 8 m + r and the value 32 further) are d16 tiles 8 + p / 10 + p of the same
  lane. The block's cos / sin rows (256 x 128 B each, rows padded to 144 B) are TDM-loaded into the idle ring stage of
  tile n during tile n - 2 (waves 0 / 1 in place of their NULL ops); the dQ staging is split into halves in the other two
  stages (two TDM stores); irope8 moved from k_mxq.hip to mx_common.h. Bit-exact with k_dq_mx + k_rope_bwd_q (five
  shapes, realistic and random cos / sin, `scratch_mx/py/dqr_test.py`): the rope columns' products dq * scale pass an
  empty asm, else LLVM folds fpext(fptrunc(x * scale)) into v_fma_mixlo_bf16 (one rounding of the exact product: 2.5e-6
  of the inputs an ulp off, the rotated values more where the two terms cancel; `#pragma clang fp contract(off)` and
  __fmul_rn do not stop it: HIP's __fmul_rn is a plain contractible multiply). The rotation itself is exact per op even
  when contracted (bf16 x bf16 products are exact in f32). Lab: k_dq_mxr 0.8956 vs k_dq_mx 0.8940 ms, k_dq_mx +
  k_rope_bwd_q 0.9283; with per-lane global cos / sin loads instead of the TDM (32 poorly coalesced 16-byte loads per
  lane, latency exposed in the epilogue) 0.904-0.908. Package (profiler, two runs): k_dq_mxr 0.922 / 0.915 vs k_dq_mx
  0.904 / 0.907 + k_rope_bwd_q 0.037 ms; backward per call 2.459 / 2.443 vs 2.469 / 2.473 ms (events 2.538 vs 2.549):
  the rope math adds ~400 VALU per wave to an epilogue nothing overlaps (one workgroup per CU). pytest 30 pass,
  fused-RoPE test pass (dq bitwise with the unfused composition).

## 2026-11-06 (e): conflict-free epilogue staging (skewed regions), fused dQ RoPE

- ATT of the package forward (`scratch_mx/py/att_segs.py`: a wave's timeline as loop runs and the runs outside):
  a warm workgroup ~83k cycles, the loop ~52.5k, outside ~30k (36%): prologue ~5k, per block ~11-13k of diagonal
  tiles + epilogue, in which the O staging's 32 ds_store_b128 per lane cost ~1.6-1.9k plus ~1.5k in the following
  `s_wait_dscnt 0` (256-byte rows: a store's 16 rows hit one bank group, 16-way). The first workgroup per CU also
  pays ~16k cycles of instruction-cache misses in its first diagonal region (a ~1k-cycle stall every 256 bytes of
  code, `scratch_mx/py/att_top.py`), once per CU per launch.
- **Skewed staging regions** (the TDM store ignores LDS padding, so a padded pitch is not possible): the staging is
  split into 16 dense regions by row class (row % 16 = j, the lane's lr), each a plain 2-D tile, the region bases
  stepping by rows * pitch + 16 B; one 2-D TDM store per region with a global row stride of 16 rows, spread over the
  waves. A ds_store's 16 rows land in 16 bank groups (2-way: the minimum for 512 bytes) with no extra instruction.
  Forward OSKEW (O, 256-byte rows): lab 0.6821 -> 0.6398 ms (-6.2%); dq DQ_SKEW (384-byte rows, 8-way before):
  0.9006 -> 0.8723 (-3.1%; the fused-RoPE k_dq_mxr with its two 128-row halves 0.8956 -> 0.8732); dkdv EPI_SKEW (dK
  384 / 256 + 128, dV 256-byte rows): 1.3910 -> 1.3391 (-3.7%). All bit-exact at five shapes (+ the forward's extra
  shape sweep). (EPI_PAD's conflict-free stores lost their gain to its re-pack copies; the regions need none.)
- Forward QFIRST (the second block's Q loads issued in the first block's epilogue ahead of the O stores: its start
  waits for its Q only): lab 0.6420 -> 0.6260 ms. With the package's re-scheduling the forward gains less: three
  copies of each code object interleaved (`scratch_mx/py/pkg_obj_ab.py`; identical objects differ by up to ~1%
  between loads), attn_fwd per call: old 0.9407, OSKEW 0.9207, OSKEW + QFIRST (branch-free: NULL ops when there is no
  second block, else the branch split the re-scheduled region) 0.9205 ms. The re-scheduled tile loop runs ~1857 vs
  ~1762 cycles per trip with the new register allocation (same instruction mix: 100 W, 256 T, ~266 V, 114 L, ~520
  s_set_vgpr_msb per trip), which eats half of the epilogue's gain. asm_resched is deterministic (outputs identical
  across PYTHONHASHSEED values).
- Package (profiler, fused-RoPE fwd + bwd, b2 s4096 h128, slow GPU): GPU time per step 3.545 -> 3.435 ms:
  k_dkdv_mxr 1.405 -> 1.328, k_dq_mxr 0.913 -> 0.886, k_mla_fwd_mx 0.605 -> 0.600. pytest 30 pass, fused-RoPE test
  pass (out / dq / dkv bitwise with the unfused composition).
- Forward **DMASK** (static mask classes for the two diagonal tiles, fully masked q tiles of the last one skipped;
  11-05 rejected for 150-200 VGPR spills) retried on the new code: 8 spills in the LLVM build, bit-exact at five
  shapes and the extra-shape sweep. Lab (LLVM order) 0.6442 -> 0.6292 ms; package (re-scheduled region shrinks 3074
  -> 1908 instructions), attn_fwd per call (3 copies each): 0.9233 -> 0.8841 ms; DSKIP alone 0.9058. Under DMASK the
  combinations (re-scheduled, 3 copies): OSKEW + QFIRST 0.8863, OSKEW 0.8933, neither 0.9114: all on. Package
  profile: k_mla_fwd_mx 0.600 -> 0.558 ms, fused-RoPE fwd + bwd GPU time 3.435 -> 3.389 ms per step. pytest 30
  pass, fused-RoPE test pass.
- Forward option sweep under the package's re-scheduling (two copies each, attn_fwd per call): base 0.8867,
  SCHED 0 0.8881, MAXTREE 0.8932, PRO_PARTIAL 0 0.8866, TDM_SPLIT 0 0.8868, GCHUNK 8 0.8839 (not taken: the overflow
  guard's margin), **QKFIRST 0** 0.8801 (four copies: 0.8770 vs 0.8839; LLVM order 0.6057 vs 0.6226 ms): QKFIRST off.
  Package: k_mla_fwd_mx 0.551 ms, fused-RoPE fwd + bwd 3.377 ms per step. pytest 30 pass, fused-RoPE test pass,
  extra-shape sweep pass.
- Backward option sweep (bit-exact, same run): dkdv base 1.3263 ms, PF_KV 0 1.3265, DQ_WAIT0 1.3283,
  DQ_PRO_PARTIAL 0 1.3286, MS_PA 0 1.3291, KEEP_WAR 0 1.3344, PRO_TDM_FIRST 1.3401, TDM_INC 0 1.3476: dkdv stays.
  dq: **DQ_WAIT0** (the one-block tile loop) k_dq_mx 0.8570 vs 0.8653 ms but k_dq_mxr 0.8770 vs 0.8717: on for
  k_dq_mx only (k_dq_mxr 0.8696 vs 0.8702 after); PF_KV 0 1.172 (!). pytest 30 pass, fused-RoPE test pass.
- **Quantization: column-code staging** (`k_mxq.hip` RC_XSKEW, default): rc_tile's cols phase stores each lane's two
  columns (rows d0 = 2 dp, d0 + 1 of 128 B) for 32 consecutive dp per instruction: one bank group (32-way). As 16
  regions (dp % 8, row parity), bases stepping by D / 16 rows + 16 B (2-way, the minimum), two 2-D TDM stores per
  wave (global row stride 16 rows). Bit-exact (codes, scales, aux; X8T byte-equal). Lab: q 0.1042 -> 0.0960, k 0.1003
  -> 0.0858, v 0.0654 -> 0.0498, dO 0.0935 -> 0.0871 ms. Package profile: k_mxq_rctr192 (q, k) 0.231 -> 0.209,
  k_mxq_rct128 (v, dO) 0.150 -> 0.131; fused-RoPE fwd + bwd 3.377 -> 3.347 ms per step. pytest 30 pass, fused-RoPE
  test pass. Variants, all the same or slower: 4 / 8 regions (8 / 4-way), 16 / 24 / 48 regions of consecutive rows
  (with a remapped task order: 2-way), wave 0 issuing all 16 stores, a dense re-pack into the dead input tile + one
  store (one more barrier: +5%).
- Pitfall: timing probes numbered >= 2 in k_mxq also skip the XS / X8T / XTS TDM stores (`RCT_PROBE < 2` guards):
  probes 5..8 of this work looked 20-45% faster only because they wrote no outputs (`scratch_mx/py/x8t_check.py`
  fills the outputs with 0xAB and checks they were written).
- **Quantization: padded tile rows** (RC_TPAD, default): the rows phase's 16-byte reads of dense 256 / 384-byte bf16
  rows hit 4 bank groups (8 / 10-way, modelled). rct128: 272-byte rows (TDM load pad 16 B per 256 B: 2-way), v 0.0497
  -> 0.0465, dO 0.0877 -> 0.0854 ms; rctr192: 400-byte rows (the rope gap 128 -> 144 B), k -1%, q same. rct192 with 16
  B after every 512 B of its stream (384 is no power of two; the TDM pads by the stream, verified bit-exact): 4-way but
  q +5%, k +3% (address math): left dense. Bit-exact. Package: k_mxq_rct128 0.131 -> 0.126 ms (two calls), fused-RoPE
  fwd + bwd 3.341 ms per step. pytest 30 pass, fused-RoPE test pass.
- rctr192 **RC_RPRE** (default): each thread's cos / sin rows for the two rotation items loaded before the tile's TDM
  wait (their latency under the load): q 0.1203 -> 0.1172, k 0.0921 -> 0.0897 ms, bit-exact, no extra VGPRs (105).
- Forward re-schedule with an msb preference (tie-break to an instruction needing no s_set_vgpr_msb switch; or + 1
  cycle per switch): the loop's emitted lines 1328 -> 1135 / 1016, but attn_fwd 0.8761 -> 0.8986 / 0.9229 ms (3
  copies): the W / T / V interleaving matters more than the switches (T <-> V switches fill the TRANS unit's idle
  slot; ATT: T -> V with a switch 1.70 cycles vs T -> T 1.77). Not taken.
- **Code prefetch** (PF_CODE): the first workgroup per CU of each launch misses the instruction cache line by line in
  its once-per-block regions (dq ATT: that workgroup's diagonal + epilogue 60k vs 8k cycles; forward ~16k extra).
  `prefetch_code<NB>()` (mx_common.h: `__builtin_amdgcn_s_getpc` + NB `__builtin_amdgcn_s_prefetch_inst` of 32 lines
  at 4 KB steps) at the kernel start. A first inline-asm version (`s_prefetch_inst_pc_rel`, 16 x 4 KB) page-faulted
  (no-retry fault, client TCP; no reset): k_dq_mxr is the code object's last function, the range passed the end of
  .text. Package fwd + bwd profile (two runs each): k_dq_mxr 0.8905 / 0.8825 -> 0.8618 / 0.8625 / 0.8581 ms; dkdv
  (6 x 4 KB) 1.3261 / 1.3274 -> 1.3290 / 1.3298, forward (7 x 4 KB) 0.550 either way: on for the dq kernels only (10 x
  4 KB of 43.9 / 47.8 KB). mx.py checks the built object (ELF symbol table vs the .text end) and rebuilds with
  PF_CODE 0 when a prefetch could pass the end. Fused-RoPE fwd + bwd 3.302 ms per step. pytest 30 pass, fused-RoPE
  test pass.
- Training gate **g120i** (all of the above, slow GPU, 120 iterations, 1-GPU DSv3 proxy): bf16 HK 1004.7, hk_mx +
  fused RoPE 981.9 ms / iteration (-2.27%; g120h this morning -1.06%), final-window loss 0.0818 both, grad-norm ratio
  max 1.025.
- dkdv ATT after the skewed staging (warm workgroups): prologue (incl. the peeled diagonal step) ~6.2-6.6k cycles,
  loop ~2270 cycles / step x 24-26, epilogue ~2.8-3.3k: 14% outside the loop; the cold first workgroup per CU +7.7k.
  Forward trip profile (100-cycle buckets): two WMMA-free stretches per trip (~300 + ~200 cycles of softmax bursts).
  The forward's 8 VGPR spills are all in the prologue (none in the tile loop).
- dq trip profile (ATT, k_dq_mxr, 2857 cycles / tile): per kv16 pair and q tile a VALU-only stretch (~100-200 cycles:
  dS = P dP, its |max| and the pk8 conversions), and the 48 dQ WMMAs at the end of the trip (~400 cycles of W + L, no
  VALU). Overlapping them needs the dQ of tile t in tile t + 1 (dS of two tiles live: +64 VGPRs over 905, and tile t's
  K^T stage is recycled at the start of t + 1 with the 3-stage ring): not possible without an LDS / register re-plan.
  The |dS| max as a max3 tree (DQ_AMTREE, lab): same v_max count (LLVM already formed max3), 1 spill, 0.8957 vs
  0.8610 ms: dropped.
- **lse in log2 units**: the forward kernel writes lse2 = m + log2(l) (it multiplied by LN2, and the backward's first
  op multiplied by LOG2E again); `saved[0]` is lse2, attn_bwd / attn_bwd_rope use it directly, attn_fwd forms the
  API lse (natural, + q . kbar) as before (bit-identical), the fused-RoPE autograd path skips it (`need_lse`). Two
  elementwise kernels fewer per fused-RoPE step (3 us each + a ~3.4 us gap); the fused and unfused paths still match
  bitwise (rope test), pytest 30 pass, extra-shape sweep pass. The lab's fwd_mx returns natural units as before.
- From ~10:20 UTC the GPU runs every kernel 5-7% slower than at 10:15 (same clocks 1.30 GHz, no throttle flags, no
  other process; forward 0.551 -> 0.590, dkdv 1.328 -> 1.384-1.395, dq 0.86 -> 0.97 ms per call): compare within one
  run only.
- Forward time vs workgroup rounds (lab, b2 s4096, h 16 / 32 / 64 / 128 = 1 / 2 / 4 / 8 rounds of 256): 87.5 / 163.8
  / 312.6 / 611.3 us: ~75 us per round, ~12 us of launch + cold start: no dispatch-level loss (the per-workgroup
  time is what counts).
- Forward re-schedule with a kind preference among equally early candidates (3 copies): none 0.8819, WMMA 0.9023,
  TRANS 0.8812, DS_LD 0.8861 ms: the default stays.
- k_rope_bwd_kpe with 4 (b, s) rows per workgroup (16 loads in flight per thread, same summation order: bit-exact):
  22.8 / 21.9 -> 22.0 / 21.5 us (lab): not latency-bound enough to matter; reverted.
- dS for the dQ pass from dk / dv (a GEMM-only dq): dk / dv's S^T / dP^T are in the "lane = kv" orientation (A = Q,
  B = K), the dQ operand needs dS quantized along kv with lane = q: a q <-> kv transpose through LDS per step (~42 KB
  free in dk / dv) on an LDS-bound kernel (+15-20% estimated) and a bf16 rounding before the quantization (no longer
  bit-exact): not pursued.
- **Persistent forward** (FWD_PERSIST, lab, reverted): grid (256, 1, 1), workgroup w takes the pairs w, w + 256, ..
  (34 tiles each; ring stage offset (34 j) & 3), the next pair's tiles 0..2 issued in the current pair's last three
  tiles and its Q in the second block's epilogue; bit-exact (also with 8 workgroups: many pairs each). Slower: 0.6170
  / 0.6323 / 0.6305 vs 0.6119-0.6140 ms over three versions (two loops instantiated the block twice: 2x code; then
  the per-item state live through the tile loop cost SGPRs: 58 -> 121-214 SGPR spills, the tile loop 2076 -> 2252
  cycles / trip, more than the ~4k cycles of start-up wait it hides per pair).
- The q quantization's row dots q . kbar (mode 2, the aux) only form the API lse: the fused-RoPE autograd path
  (need_lse False) quantizes q with mode 48: same codes / scales, 116.2 -> 104.7 us per call (-10%). pytest 30 pass,
  fused-RoPE test pass.
- Forward loop costs (ATT, 4 trips, package build): v_exp 1.7 cycles each (256 / trip), WMMA 4.1 (100), ds_load
  2.0 (114), v_max3 1.5 (144), pk8 conversions 3.4 (34): issue-bound, a sum of per-instruction costs. A model probe
  without VGPR anti / output dependences (the bound of renaming spare VGPRs) predicts no gain (1485 -> 1558 model
  cycles: the greedy order gets worse; the model is ~285 cycles below the measured trip). A third S buffer in the
  source changes nothing (SSA: LLVM's allocation decides the reuse).
- `v_exp_f16` / `v_exp_bf16` (micro-benchmark `scratch_mx/bt/expbench`, 4 waves per SIMD, 8 chains per lane):
  ~5.3 cycles per exp per SIMD vs v_exp_f32 ~2.5: no faster 16-bit transcendental (and bf16 inputs would be too
  coarse for exp2(s - m)).

## 2026-11-09: rung b7, dS materialized by dK / dV, dQ as a GEMM (design, CPU builds)

GPU still at sclk ~1304 MHz (amd-smi, idle 855 W): same-run A / B only.
- Why: k_dq_mx spends 96 of its 144 WMMAs per tile and all of its exps recomputing S / P / dP that k_dkdv_mx already
  has (per-step budget, 11-06 slow GPU: dkdv 1.33, dq 0.86, forward 0.55, quantization 0.34 of 3.30 ms).
- b7: k_dkdv_mxd / k_dkdv_mxrd = k_dkdv_mx / _mxr + the f32 dS = P (dP - delta) as bf16 (one global_store_b128 of 512
  contiguous bytes per (slot, kv16 tile): 64 v_cvt_pk_bf16 + 16 stores per step and wave) into causal 128 x 128 tiles
  (per (b, h): index qb (qb + 1) / 2 + kb, a q block's tiles consecutive; chunk 16 w + 2 qt + j = dkdv wave w's kv16
  tile j at slot qt; 4.33 GB at b2 s4096 h128). k_dqg_mx / _mxr: grid and order as k_dq_mx, a 3-stage TDM ring of
  [dS (t, 2 qb), dS (t, 2 qb + 1), K8T, KTS] (one op per wave), the B operand by ds_load_tr16_b128 (8-lane transposed
  16-bit loads; lane map tr_off, to be confirmed on the GPU by scratch_mx/bt/trload.py), MX along kv from bf16 as k_dq_mx
  (16 values + the peer lane per block), 48 dQ WMMAs per tile, k_dq_mx's epilogue (skewed staging, fused inverse RoPE).
  No masks: dkdv writes exact zeros where P is masked. Expected memory-bound (4.33 GB of dS at ~15 TB/s ~0.3 ms).
  Recipe change: dQ uses MX(bf16(dS)) instead of MX(dS): ~3% of the codes move by one E4M3 ulp (double rounding),
  dK / dV unchanged (bitwise).
- Build: with SLP the dS stores make the SLP vectorizer's block scheduling pull 36 of the step's dV / dK WMMAs (slots
  1..6) into slot 7's load region (IR level, before the sched_barriers matter): 986 VGPRs + 96 spilled in the loop.
  `-fno-slp-vectorize`: every dkdv variant 782 VGPRs, no scratch (k_dkdv_mxrd: one 4-byte value, 7 reloads / trip);
  k_dqg 675 VGPRs, no scratch. Lab tags: 5xi (EXPERT_I, b5 kernels), 5ni (EXPERT_I + no SLP: b7).
- Tests / timing: `scratch_mx/py/b7_test.py` (dK / dV bitwise vs b5, dS vs the emulation, dQ SNR vs emu / b5 / fp32
  reference; dS buffer poisoned), `scratch_mx/py/b7_time.py`; mxlab.bwd_mx7, ds_unpack, emu_bwd(dq_bf16=, ds_out=).
- GPU (same slow clock): `ds_load_tr16_b128` / `global_load_tr16_b128` = 8-lane groups, lane c gets column c (16-bit) of
  the 8 rows its group's lanes address (`scratch_mx/bt/trload.py`): tr_off as assumed. b7_test (toy shapes): dK / dV
  bitwise = b5 (SLP and no-SLP builds), every dS byte written, dS 55.5 dB vs the emulation (= bf16 rounding), dQ vs
  the emulation of b7's recipe 48-52 dB, vs the fp32 reference 21.62 dB (b5 21.65).
- First timing (b2 s4096 h128): dkdv_mx 1.327, dkdv_mxd 1.549 (+17%), dq_mx 0.848, dqg_mx 0.536 (8.3 TB/s of dS).
  dkdv: ATT +253 cycles / trip; store-pair stalls (VALU -> VMEM data ~23 cycles), an s_wait_xcnt (78) where the RA
  reused the store data registers. Probes: stores to one L2-resident tile 1.516, conversions without stores 1.347:
  the store instructions themselves, not the HBM writes. Store placement (DSO_LATE): 0 1.543, 1 1.542, 2 1.531,
  **3 (the step's 16 stores at its end) 1.486**, 4 (two bursts) 1.644; non-temporal stores no change. ATT (DSO_LATE 2):
  trips 2400-3600 with sporadic 400-1100-cycle ds_load issue stalls 160-360 cycles after a store pair (base: steady
  2200-2300): VMEM stores interleaved with the LDS stream; fewer with the burst.
- dqg ATT: 1704 cycles / tile, LDS-issue bound (48 K8T b128 per wave per tile: all four waves read the whole K8T tile)
  + ~600 VALU of quantization; not HBM-bound. DQG_V 1 (12 K8T fragments resident, q tiles pipelined: loads of i + 2,
  quantization of i + 1 next to i's 12 WMMAs; E8M0 on the bits): 0.518; DQG_AMAX 1 (signed and unsigned int16 max
  trees instead of AND + max) and DQG_ORDER 1 (transposed loads first, K8T next to the first quantization): 0.495 ms.
  All bitwise identical with the first version (b7_ab.py, five shapes). ATT: 1639 cycles / tile, the first ~800 without
  WMMAs (the K8T + dS loads of all waves at once).
- Lab, same run: dkdv_mxr 1.325 + dq_mxr 0.857 = 2.182 -> dkdv_mxrd 1.484 + dqg_mxr 0.511 = 1.995 ms (-8.6%).
- Package: b7 on by default (PRIMUS_TURBO_MLA_MX_DQ_GEMM, PRIMUS_TURBO_MLA_MX_DS_MAX_BYTES 16 GiB, OOM -> recompute
  kernel), k_mla_bwd_mx5.hip built without SLP. pytest 30 pass, fused-RoPE test pass (fused == unfused bitwise).
  Profile (fused-RoPE fwd + bwd per step): 3.544 (b5 kernels, no-SLP build) -> 3.276 ms.
- A lab timing script bug (k_dkdv_mxr given dK / dV of head stride 192 / 128: the kernel's DK / DV are kv-gradient
  views with head stride 256) page-faulted (no-retry, TCP) and the queue was reset; the GPU recovered. gx.sh's dmesg
  window now starts 30 s early (dmesg runs ~10 s ahead of the wall clock: the fault was not caught).
- Training gate **g120j** (slow GPU, 120 iterations): bf16 HK 1001.9, hk_mx + fused RoPE (b7) 975.7 ms / iteration
  (-2.61%; g120i -2.27%), final-window loss 0.0818 both, grad-norm ratio max 1.014.
- DK_BF16 (dK's dS operand MX-quantized from the stored bf16 values, dQ's recipe): dkdv_mxd 1.709 vs 1.490 ms: off.
- **DQG_PAIR** (k_dqg: q blocks nqb - 1 - p and p per workgroup, 2 nqb + 2 tiles each; grid (nqb / 2, H, B), the
  kernel takes the paired path only when launched with it): stages padded to 98816 B so the first block's dQ staging
  fits its last tile's stage; that tile issues the block's cos / sin (ROPE) or a NULL op in place of the next block's tile
  1, which is issued after the epilogue's stores; the next block's first wait (<= 1 outstanding) covers its tile 0 and
  the stores (the staging stage takes its tile 2). Bitwise = unpaired (five shapes). dqg_mx 0.496 -> 0.483, dqg_mxr
  0.504 -> 0.496 ms (the block switch still exposes one TDM latency: three stages hold tiles 0..2 of the next block or
  the staging).
- Package profile, same harness and run (prof_pkg.py, fused-RoPE fwd + bwd per step, two rounds each): the 11-06 package
  (db260777) 3.500 / 3.507, this one 3.256 / 3.251 ms (-7.1%; both include 0.205 ms of the script's gradient adds).
- dkdv post-RA re-scheduling again (resched.py on k_dkdv_mxd, no-SLP build): the model predicts LLVM's order to the
  cycle (2452 vs ATT 2456 / trip) and 1810 for first-ready-first, but measured identity 1.493, orig 1.517, height
  1.579, ready 1.594 vs 1.477 ms (bitwise): still no gain from re-ordering dK / dV.
- LTDT_DPP (lab, off): a q16 tile's lse / delta C tuples from one ds_load_b32 each + 8 DPP row_share broadcasts
  (row_share works on gfx1250) instead of 2 + 2 ds_load_b128 (-14% of the loop's LDS bytes, +128 VALU / step):
  bitwise, dkdv_mx 1.330 -> 1.514, dkdv_mxd 1.489 -> 1.669 ms: the loop is not LDS-bandwidth bound (per-wave issue).
- The fused-RoPE forward's quantizations in one launch: **k_mxq_qkv** (q and k rctr192 argument sets as structs, v as
  rct128 rows + cols; grid (S / 128, H, 3 B); rctr_body / rct_body take (kt, h, b)): two separate rctr192 launches 0.200
  -> 0.185 ms in a two-way version, q + k + v 0.2285 -> 0.222 ms in the three-way one (package profile); fused == unfused
  bitwise, pytest 30 pass. Package fused-RoPE fwd + bwd per step 3.25 -> ~3.23 ms.
- dkdv re-scheduling with longer LDS latencies in the model (ready 96 / 160, height 128, orig 128): 1.574-1.585 /
  1.493 / 1.511 vs 1.478 ms: no setting helps.
- dqg probes (wrong results, timing only): no K8T fragment loads 0.491 -> 0.449 ms (a K8T-from-global path with a
  fragment-major layout could gain at most ~8%: not pursued), no dS quantization 0.397 (the VALU of the quantization
  is the larger cost; each value is quantized once already).
- DKDV_SGB (sched_group_barrier issue patterns per dkdv slot): [W 2T 4V 2DS] x 12 2.299, [W T 2V DS] x 16 1.784,
  S / dP WMMAs first then [W 3V 2DS] 1.510 vs 1.486 ms (bitwise): off.
- **dqg re-scheduled after register allocation** (asm_resched): the tile loop made one basic block (the per-tile wait +
  barrier also at tile 0; issue() by selects only: per-wave / per-block base addresses and bytes per tile precomputed),
  the halves' max of the bf16 amax by a rotate + packed max (no 16-bit register halves: asmkit's parser does not read
  `v.l / v.h`). Lab, bitwise with LLVM's order: dqg_mx LLVM 0.490, identity 0.471, ready 0.466, **height / LDS latency
  96 / LDS burst 2: 0.447, dqg_mxr 0.503 -> 0.460** (height 64 / 128 / 160, gap 4 / 8: 0.454-0.467 / 0.467-0.489).
  Package: asm_resched gained the height priority and an LDS latency option; RESCHED per source with options
  (k_mla_bwd_mx5.hip: k_dqg_mx, k_dqg_mxr; the prefetch range is checked on the re-scheduled object too). Package
  profile: k_dqg_mxr 0.555 -> 0.510-0.516 ms, fused-RoPE fwd + bwd per step 3.22 -> 3.19-3.21. pytest 30 pass,
  fused-RoPE test pass.
- Forward re-schedule sweep with the package scheduler (SLP on, as the package): ready / 32 (the package's) 0.556, ready
  / 64 0.557, height / 32 / 64 / 96 0.598-0.607, height 96 burst 2 0.591 ms (bitwise): the forward keeps ready / 32.
- Training gate **g120k** (qkv launch, paired + re-scheduled dqg; another session's e2e jobs ran between the two arms):
  bf16 HK 1002.2, hk_mx + fused RoPE 975.1 ms / iteration (-2.70%; g120j -2.61%), loss 0.0818 both, grad-norm ratio
  max 1.014.
- dkdv budget (11-09 ATT + probes): per step and CU ~860 LDS loads (4 waves x ~215) plus ~95 KB of TDM writes into LDS
  (~80% of LDS time at an assumed 256 B / cycle), VALU near full too: fewer LDS loads for more VALU (LTDT_DPP) and fewer
  scale loads (DKDV_SCPROBE, a probe: 1.334 -> 1.557 ms, LLVM's code changes around it) are slower. A generator would
  fight the same two limits; a larger reuse (48 kv rows per wave) needs another tile / dS layout.
- Tests: test_dq_gemm_matches_recompute (dK / dV bitwise between the dQ GEMM and the recompute kernel, dQ > 30 dB apart,
  both over the fp32 floor; paired grid, odd q blocks, b h % 8 != 0), test_dq_gemm_workspace_cap (the cap falls back,
  bitwise). pytest 34 pass.
- asmkit (gfx1250_mla/tools/asmkit/isa.py regs_of): 16-bit register halves (`v168.l`, `v253.l /*v765.l*/`) parse as
  their register (parse_line already marks such writes partial); inputs that parsed before are unchanged (the bf16 HK
  regen_asm_patch.py --check passes: both generated kernels current).
- With it: dqg's block of its last two tiles re-scheduled too (regions): dqg_mx 0.446 -> 0.439, dqg_mxr 0.460 -> 0.458
  ms, bitwise; package on (k_dqg_mxr 0.490-0.498 in the profile). k_mxq_qkv's loop re-scheduled (ready): 0.2260 vs
  0.2228 ms: not taken.
- A GPU fault marker (scratch_mx/logs/GPU0.FAULT) was written by mistake: the MES / queue-reset lines came from another
  session's pytest sweep that ended just before this job (dmesg's clock lags the wall clock by ~15 s, and gx.sh's window
  had been widened). gx.sh now checks only the kernel-log lines added during the job (line count at the start).
- dkdv re-scheduled with a bwdgen-like LDS model (one LDS op per 8 / 11 cycles, no burst, latency 70 / 90; height /
  ready): 1.546 / 1.585 / 1.569 vs 1.497 ms (bitwise): the dK / dV loop stays LLVM's.
- SLOT_ORDER 1 (every kv16 tile's S / dP WMMAs of a dkdv slot first, then the softmax per tile): 1.489 vs 1.490 ms
  (bitwise): LLVM normalizes the order.
- Forward prologue (ATT, package build): ~8.9k cycles per workgroup: the first TDM data (s_wait_tensorcnt 0xa, ~2k:
  the 11th op of the wave waits for the first to land), the barrier after it (~1.6k), a cold s_load of the kernel
  arguments (~1k, first wave), SGPR spills to VGPR lanes (58 writelane + 39 readlane, ~0.46k): per-workgroup latency
  (8 workgroups per CU: ~9% of the kernel); only overlapping workgroups (persistent: SGPR-bound on 11-06) would hide it.
