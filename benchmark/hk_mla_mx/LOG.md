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
