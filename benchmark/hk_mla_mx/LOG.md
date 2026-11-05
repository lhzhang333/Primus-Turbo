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
