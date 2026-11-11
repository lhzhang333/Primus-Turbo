###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Torch reference of the gfx1250 MXFP8 training recipe (tests and benchmarks only).

Recipe: E4M3 data, one E8M0 scale per 32 consecutive elements along each GEMM's reduction dim.
Activations and gradients are quantized twice: rows (blocks along the last dim) and a transposed
column copy (blocks along the first dim, stored ``[C, R]``). Weights use 32x32 blocks, so both
copies hold the same fp8 values.

Scale rules (``rule``):

* ``"nosat"``: the smallest 2^e with amax / 2^e <= 448, so nothing saturates (the MXFP8 attention's
  rule). E8M0 byte = exp_field(amax) - 8 + (mantissa(amax) > 0x600000), clamped to >= 0; amax = 0
  gives byte 0 (2^-127); a non-finite amax gives 0xFF (NaN), so Inf / NaN inputs come out as NaN.
* ``"even"``: Turbo's HIP / FlyDSL CDNA4 rule (torchao's "EVEN"): amax rounded half-up to 3 mantissa
  bits, exponent - 8, scaled values clamped to +-448. The exponent only moves up at mantissa >= 1.9375,
  so scaled block maxima in (448, 496) clip.

Scale layout of the GEMM (aiter's mx32 layout): ``[ceil(R/32), ceil(K/128), 32, 4]`` uint8, where
``s4[g, t, r, j]`` scales row ``32 g + r`` over K block ``4 t + j`` (elements ``128 t + 32 j ..``).
One int32 of it is the WMMA scale operand of one row for one 128-wide K step. Pad bytes are 0x00.
"""

import torch

MX = 32
E4M3_MAX = 448.0


def _e8m0_bytes(amax: torch.Tensor, rule: str) -> torch.Tensor:
    """E8M0 byte (int32) of a float32 amax tensor."""
    bits = amax.contiguous().view(torch.int32)
    exp = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    if rule == "nosat":
        byte = exp - 8 + (mant > 0x600000).to(torch.int32)
    elif rule == "even":
        # round the mantissa to 3 bits (half up), then take the exponent
        byte = (((bits + (1 << 19)) >> 23) & 0xFF) - 8
    else:
        raise ValueError(f"unknown scale rule {rule!r}")
    byte = byte.clamp_(min=0)
    return torch.where(exp == 0xFF, torch.full_like(byte, 0xFF), byte)


def _scale_of(byte: torch.Tensor) -> torch.Tensor:
    """2^(byte - 127) as float32 (byte 0xFF -> NaN)."""
    s = torch.ldexp(torch.ones_like(byte, dtype=torch.float32), (byte - 127).to(torch.float32))
    return torch.where(byte == 0xFF, torch.full_like(s, float("nan")), s)


def _quant_blocks(xb: torch.Tensor, amax: torch.Tensor, rule: str):
    """xb [..., 32] float32 with its block amax [...] -> (fp8 [..., 32], E8M0 bytes [...] uint8)."""
    byte = _e8m0_bytes(amax, rule)
    v = xb / _scale_of(byte).unsqueeze(-1)  # exact: power-of-two scale
    if rule == "even":
        v = v.clamp(-E4M3_MAX, E4M3_MAX)
    return v.to(torch.float8_e4m3fn), byte.to(torch.uint8)


def quant_rows(x: torch.Tensor, rule: str = "nosat"):
    """[R, C] -> (fp8 [R, C], raw E8M0 [R, ceil(C/32)]). C is zero-padded to a multiple of 32 for amax."""
    R, C = x.shape
    Cp = -(-C // MX) * MX
    xf = torch.nn.functional.pad(x.float(), (0, Cp - C)).view(R, Cp // MX, MX)
    q, s = _quant_blocks(xf, xf.abs().amax(-1), rule)
    return q.view(R, Cp)[:, :C].contiguous(), s


def quant_dual(x: torch.Tensor, rule: str = "nosat"):
    """Activation / gradient: rows of x and rows of x^T (= column blocks of x).

    Returns (q_row [R, C], s_row [R, ceil(C/32)], q_col [C, R], s_col [C, ceil(R/32)])."""
    q_row, s_row = quant_rows(x, rule)
    q_col, s_col = quant_rows(x.t(), rule)
    return q_row, s_row, q_col, s_col


def quant_2d(w: torch.Tensor, rule: str = "nosat"):
    """Weight: one scale per 32x32 block, shared by both copies.

    Returns (q [N, K], s_row [N, ceil(K/32)], q_t [K, N] = q^T, s_t [K, ceil(N/32)])."""
    N, K = w.shape
    Np, Kp = -(-N // MX) * MX, -(-K // MX) * MX
    wf = torch.nn.functional.pad(w.float(), (0, Kp - K, 0, Np - N))
    blk = wf.view(Np // MX, MX, Kp // MX, MX).permute(0, 2, 1, 3)  # [N/32, K/32, 32, 32]
    amax = blk.abs().amax((-1, -2))
    q, s = _quant_blocks(blk.reshape(*amax.shape, MX * MX), amax, rule)
    q = q.view(Np // MX, Kp // MX, MX, MX).permute(0, 2, 1, 3).reshape(Np, Kp)[:N, :K].contiguous()
    s_row = s.repeat_interleave(MX, 0)[:N]  # [N, K/32]
    s_t = s.t().repeat_interleave(MX, 0)[:K]  # [K, N/32]
    return q, s_row, q.t().contiguous(), s_t.contiguous()


def pack_scales(s: torch.Tensor) -> torch.Tensor:
    """Raw E8M0 [R, ceil(K/32)] -> the GEMM layout [ceil(R/32), ceil(K/128), 32, 4], pad bytes 0x00."""
    R, KB = s.shape
    Rp, KBp = -(-R // 32) * 32, -(-KB // 4) * 4
    sp = torch.zeros(Rp, KBp, dtype=torch.uint8, device=s.device)
    sp[:R, :KB] = s
    return sp.view(Rp // 32, 32, KBp // 4, 4).permute(0, 2, 1, 3).contiguous()


def unpack_scales(s4: torch.Tensor, R: int, K: int) -> torch.Tensor:
    """The GEMM layout -> raw E8M0 [R, ceil(K/32)]."""
    G, T = s4.shape[:2]
    return s4.permute(0, 2, 1, 3).reshape(G * 32, T * 4)[:R, : -(-K // MX)].contiguous()


def dequant(q: torch.Tensor, s: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    """fp8 [R, C] with raw E8M0 [R, ceil(C/32)] -> [R, C] in ``dtype``."""
    R, C = q.shape
    scale = _scale_of(s.to(torch.int32)).repeat_interleave(MX, 1)[:, :C]
    return (q.to(torch.float32) * scale).to(dtype)


def gemm_ref(a, sa, b, sb, dtype=torch.float64) -> torch.Tensor:
    """NT MX GEMM reference: dequant(a) @ dequant(b)^T in ``dtype`` (float64 = exact for small K)."""
    return dequant(a, sa, dtype) @ dequant(b, sb, dtype).t()


def snr_db(x: torch.Tensor, ref: torch.Tensor) -> float:
    ref = ref.double()
    noise = (x.double() - ref).pow(2).mean()
    return float(10 * torch.log10(ref.pow(2).mean() / noise.clamp_min(1e-300)))


def int_valued_operands(M: int, N: int, K: int, device="cuda", seed: int = 0):
    """fp8 operands holding small integers and scales near 2^0: products and sums are exact in fp32
    (|C| < 2^24), so a correct kernel matches ``gemm_ref`` bitwise in fp32."""
    g = torch.Generator(device=device).manual_seed(seed)
    a = torch.randint(-4, 5, (M, K), generator=g, device=device).to(torch.float8_e4m3fn)
    b = torch.randint(-4, 5, (N, K), generator=g, device=device).to(torch.float8_e4m3fn)
    kb = -(-K // MX)
    sa = torch.randint(126, 129, (M, kb), generator=g, device=device).to(torch.uint8)
    sb = torch.randint(126, 129, (N, kb), generator=g, device=device).to(torch.uint8)
    return a, sa, b, sb
