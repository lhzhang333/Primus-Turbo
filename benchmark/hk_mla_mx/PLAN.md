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

## 4. Rules

GPU2 only, through `scratch_mx/g2.sh` (lock, FAULT interlock, dmesg check, hard timeout); new kernels on toy
shapes first; no `rocprofv3 --pmc`; commits with `-c user.name=lihuzhan -c user.email=lihuzhan@amd.com`, pushed to
the fork only; progress note to the user about every 30 minutes.
