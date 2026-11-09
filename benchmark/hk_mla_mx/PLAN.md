# DSv3 MLA attention on MXFP8 (HipKittens-style HIP kernels, gfx1250 / MI455X)

Branch `dev/lhz/dsv3_attn_hk_mxfp8` (fork lhzhang333/Primus-Turbo), cut from `origin/dev/mi455x_v4` (429492b8)
plus the bf16 HK MLA package of fork `dev/lhz/dsv3-attn-hk-fwd-bwd` (f91a3f2f: fwd f37j, bwd r10x).
Work notes: `LOG.md`. Everything measured on e15-02 GPU2 (`scratch_mx/g2.sh`: GPU2 lock + ladder `run_gpu.sh`).

## 0. Problem and baseline

DeepSeek-V3 MLA, the proxy's per-GPU shape: b 2, s 4096, h 128, d_qk 192 (128 nope + 64 rope), d_v 128, MHA, causal,
SBHD, softmax scale 0.13523 (yarn). Baseline (2026-11-05, GPU2, this branch, bf16):

| | FlyDSL | HK bf16 | bf16 arch ceiling (roofline 10-29) |
|---|---|---|---|
| fwd | 0.800 ms | 0.724 / 0.727 (hot / cold) | ~0.65 |
| bwd | 3.505 | 2.303 / 2.288 | ~1.80 |
| fwd + bwd (autograd) | 4.305 | 3.043 / 3.029 | ~2.45 |

Hardware (micro 11-05, `scratch_fp8_probe/isa/wmma_rate.hip`, toggling data, at the 2500 W cap): bf16 16x16x32 WMMA
9 cyc, 3.1 PF; fp8 16x16x64 5.75 cyc, 8.3 PF; fp8 16x16x128 8.75 cyc, 9.7 PF; MX-scaled 16x16x128 9.0 cyc, 9.3 PF.
FP8 is ~4x bf16 per cycle and ~3x sustained. The bf16 forward already spends 2.74 VALU + 0.83 exp per WMMA (the
hideable budget is ~3 + 1): with FP8 the softmax, not the WMMA, becomes the forward's limit.

## 1. Recipe (MXFP8: E4M3 data, E8M0 scale per 32 elements along each GEMM's reduction dim)

Forward (S^T = K Q^T, O^T = V^T P^T as the bf16 kernel):
- S: Q and K MX along d (per token and head, 6 blocks of 32: nope 4 + rope 2). nope: one MX 16x16x128 WMMA;
  rope: v0 an MX 16x16x128 with the second half zero (exact MX), alternative (rung) a plain fp8 16x16x64 on per-head
  power-of-2 scales folded into the nope scales and the softmax constant.
- P V: P quantized in the kernel to E4M3 with a fixed E8M0 scale (P <= e with the lazy rescale threshold of 1 nat:
  2^-7 keeps 128 P <= 348 < 448), V MX along kv (per head and channel, blocks of 32 tokens), stored transposed
  ([b, h, d, s]) by the quantization kernel so the V^T fragments load without a transposing LDS read.
- fp32 softmax state, bf16 out, fp32 lse (as bf16 HK).

Backward (k_delta, dK/dV kernel, dQ kernel, as bf16 HK):
- recompute S from the same Q8 / K8 as the forward; dP = dO V^T: dO and V MX along d_v (rowwise);
  dV = P^T dO: P fixed-scale in-kernel, dO MX along q (colwise).
- A (conservative): dS bf16, dK = dS^T Q and dQ = dS K in bf16.
- B (aggressive): dS MX in the kernel (colwise for dK, rowwise per 32 kv for dQ, written to the dS scratch in fp8
  with its scales: half the scratch traffic), Q / K colwise MX. Taken only if the numerics gate holds.

Quantized operands are produced by new HIP kernels (one pass per tensor, row + col where both are needed); later
fused into the producers (RoPE / up-projection epilogues) in Primus.

## 2. Gates

- Kernel correctness: against an fp32 emulation of the same recipe (same quantized operands, same P rounding)
  >= 40 dB on o / lse / dq / dk / dv at every test shape; finite, deterministic (bitwise across calls).
- Recipe quality: SNR against the fp32 reference on the bf16 inputs, measured per variant on random and on
  outlier-heavy data; recorded next to bf16 HK. A variant is taken only if it does not lose more than ~1 dB to the
  best exact-MX variant (decided per variant, numbers in LOG.md).
- Training: DSv3 proxy loss / grad-norm trajectory of the MX attention against bf16 HK over >= 200 iterations
  (same seeds); no divergence, final-window loss within noise of bf16. Before any default switch.

## 3. Phases (each step a rung: built, checked against the gate, timed against its parent)

- P0 setup and facts: branch + bf16 port (done), numerics emulation of every recipe variant (SNR table),
  WMMA fp8 / MX operand-layout discovery (A, B, C, scale VGPR lanes, scaleType / opsel), issue-model micro
  (fp8 WMMA spacing, VALU / TRANS co-issue, v_cvt_scalef32_pk8_fp8_f32, packed f32 math), 8-bit LDS / TDM paths.
- P1 quantization kernels: Q / K row (+ col), V col-transposed (+ row), dO row + col; >= 70% of HBM bandwidth.
- P2 forward: HIP reference kernel (exact recipe, 128-row kv tiles, 4 waves x 64 q rows, TDM ring) -> correctness;
  then rungs: Q8 resident, fragment pipelining, packed softmax (pk_fma / pk_add / max3), pk8 P conversion,
  lazy rescale tuning, diagonal specialization, persistent schedule, rope fold, generated asm schedule (fwdgen).
  Target <= 0.35 ms (>= 2x bf16 HK), stretch 0.30.
- P3 backward: A first (dK/dV kernel with S / dP / dV in MX, dQ unchanged), then B if the numerics allow; tiles of
  128 q rows (MX K = 128), rings, generated schedules. Target A <= 1.3 ms, B <= 1.0 ms.
- P4 integration: Turbo op + env switch (`PRIMUS_TURBO_MLA_FWD_IMPL` / `_BWD_IMPL` = `hk_mx`), tests, Primus flag,
  DSv3 proxy e2e (throughput and the training gate); quantization fused into the producers.
- P5 extreme tuning: power-aware schedules, asm generators, ATT-driven fixes, end-to-end overlap.

## 3b. Status (2026-11-05 06:15; details in LOG.md)

| piece | rung | state |
|---|---|---|
| quantization | k_mxq rows / cols, kbar (every 16th row) | GPU: bit-exact vs twins; fwd set 0.24 ms |
| | k_mxq_rc (rows + cols in one read; both passes' operands at once) | compiled, untested |
| forward | m0 (reference) | GPU: 53 dB vs emulation, 0.99 ms (bf16 HK 0.73): latency-bound |
| | m1 / m2 / m3 (pipelined; m3 sched_barrier-interleaved) | compiled, untested; m3 VALU-issue-bound by ISA count |
| | m4 (Q pre-scaled log2, C-init -m, ones-WMMA row sum, lazy rescale branch, overlapped rope, R2 scales) | compiled (~275 VALU / tile vs 783), untested |
| | m5 (= m4 + P headroom 14, integer rescale shift via the P conversion scale, Q resident) | draft, after m4's review |
| backward | b0 (reference, dS lane-local, dQ recomputed) | compiled, untested (lse/delta read bug fixed: clang bit_cast) |
| | b1 (m4 recipe, negated-C lse2 / delta, peeled diagonal) | compiled, untested |
| | b2 (dV / dK split, 4 kv16 tiles per wave: WMMA- instead of LDS-bound) | compiled, untested |
| Turbo | hk_mx (m4 + b1 + rc; fp8-only saved tensors) | CPU tests pass; GPU numerics pending |
| e2e | training gate scripts (mx_gate.sh, summary) | prepared |

## 3c. Status (2026-11-09, heliosr-1b112-a30-4, GPU stuck at ~1.3 GHz: same-run numbers only; details in LOG.md)

| piece | package rung | per call, b2 s4096 h128 (package profile, slow GPU) |
|---|---|---|
| forward quantization | k_mxq_kbar_rope + k_mxq_qkv (q, k with RoPE, v rows + cols in one launch) | 0.013 + 0.223 ms |
| forward | m7 (DMASK, OSKEW, QFIRST), tile loop + diagonal block re-scheduled after RA (asm_resched ready / 32) | 0.585 |
| dO quantization | k_mxq_rct128 mode 4 (O by TDM: RC_YTDM) | 0.071 |
| dK / dV | b7 k_dkdv_mxrd = b5 + bf16 dS tiles stored (the step's 16 stores at its end) | 1.534 |
| dQ | b7 k_dqg_mxr: dQ = scale dS K over the stored dS (no recompute), paired q blocks, re-scheduled (height / 96) | 0.492 |
| fused-RoPE fwd + bwd per step | (+ k_rope_bwd_kpe 0.016) | ~2.94 (3.30 on 11-06) |

Training gate g120k: -2.70% iteration time vs bf16 HK, same loss. Exhausted on dK / dV (LOG.md 11-09): every post-RA
list schedule (LDS latency 32..500, height / ready / orig, bwdgen-like LDS model, WMMA chain latency), LLVM scheduler
strategies, sched_group_barrier patterns, fewer LDS loads for VALU (DPP lse / delta), slot source orders. ATT: the loop
is bound by LDS (4 waves x ~215 loads per step + the TDM writes) and VALU at once; re-orders gain VALU / TRANS
efficiency but lose more in LDS latency (several hundred cycles under load) and WMMA stalls the models lack.

Next (each several days):
1. **Quantization fused into the q / kv up-projection GEMMs (P4)**: the GEMM epilogue writes the MX operands (rows +
   cols + scales, RoPE on the rope columns) instead of bf16 q / kv: -0.22 ms quantization, the bf16 write + read of
   q / kv, smaller saved activations. Open: k's centring by kbar (a mean over s, a separate pass before the GEMM or a
   recipe change), the GEMM (Turbo's own HIP / Triton GEMM with a custom epilogue instead of TE / hipBLASLt), Primus'
   MLA module.
2. **dK / dV generator** (tools/bwdgen-style: explicit registers, slot scheduler calibrated by ATT): the loop's
   ~1000 instructions per step vs ~2450 cycles; realistic -10..15% of 1.53 ms. Calibrate first: LDS latency /
   rate under four loading waves, WMMA operand-port stalls next to VALU, the pk8 converter's occupancy.
3. A larger kv reuse per wave (48 kv rows: fewer LDS bytes per WMMA) needs 192-row kv blocks against 128-row q steps /
   dS tiles / dQ tiles: a re-tiling of dK / dV and dQ together.
4. Re-tune at full clocks (2.4 GHz) once the node is fixed: every A / B of 11-06 / 11-09 ran at 1.3 GHz.

## 4. Rules

GPU2 only, through `scratch_mx/g2.sh` (lock, FAULT interlock, dmesg check, hard timeout); new kernels on toy
shapes first; no `rocprofv3 --pmc`; commits with `-c user.name=lihuzhan -c user.email=lihuzhan@amd.com`, pushed to
the fork only; progress note to the user about every 30 minutes.
