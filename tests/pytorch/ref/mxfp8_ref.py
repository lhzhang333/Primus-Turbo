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
  gives byte 0 (2^-127); a non-finite amax gives 0xFF (NaN), so Inf / NaN inputs come out as NaN
  (every code of such a block is 0xFF, what gfx1250's scaled conversion makes of a NaN scale).
* ``"even"``: Turbo's HIP / FlyDSL CDNA4 rule (torchao's "EVEN"): amax rounded half-up to 3 mantissa
  bits, exponent - 8, scaled values clamped to +-448. The exponent only moves up at mantissa >= 1.9375,
  so scaled block maxima in (448, 496) clip.

Scale layout of the GEMM (aiter's mx32 layout): ``[ceil(R/32), ceil(K/128), 32, 4]`` uint8, where
``s4[g, t, r, j]`` scales row ``32 g + r`` over K block ``4 t + j`` (elements ``128 t + 32 j ..``).
One int32 of it is the WMMA scale operand of one row for one 128-wide K step. Pad bytes are 0x00.

MoE (``quant_grouped``): rows as ``quant_rows`` per expert padded to 32 rows, the column copy per expert
padded to 128 tokens; pad rows and blocks without a token get scale ``PAD_SCALE`` (127). Its fused
producers: ``glu_ref`` / ``dglu_ref`` (the grouped MLP's activation and its gradient, bf16 before
quantization) and ``gather_rows`` (permute). B operands of the GEMM may be 16x16-preshuffled
(``preshuffle_b16``).
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


def _pow2(e: torch.Tensor) -> torch.Tensor:
    """2^e as float32 built from its bits, exact (no pow / ldexp, which need not be exact on a GPU); e >= -127."""
    e = e.to(torch.int32)
    bits = torch.where(e > -127, (e + 127) << 23, torch.full_like(e, 0x400000))
    return bits.view(torch.float32)


def _scale_of(byte: torch.Tensor) -> torch.Tensor:
    """2^(byte - 127) as float32 (byte 0xFF -> NaN)."""
    s = _pow2(byte.to(torch.int32).clamp(max=254) - 127)
    return torch.where(byte == 0xFF, torch.full_like(s, float("nan")), s)


def _quant_blocks(xb: torch.Tensor, amax: torch.Tensor, rule: str):
    """xb [..., 32] float32 with its block amax [...] -> (fp8 [..., 32], E8M0 bytes [...] uint8)."""
    byte = _e8m0_bytes(amax, rule)
    # x * 2^(127 - byte): exact, and a normal float for every finite byte (<= 247), so no denormal
    # arithmetic on any device; non-finite blocks are overwritten below
    v = xb * _pow2(127 - byte.clamp(max=247)).unsqueeze(-1)
    if rule == "even":
        v = v.clamp(-E4M3_MAX, E4M3_MAX)
    q = v.to(torch.float8_e4m3fn).view(torch.uint8)
    q = torch.where((byte == 0xFF).unsqueeze(-1), torch.full_like(q, 0xFF), q)  # NaN scale: NaN codes
    return q.view(torch.float8_e4m3fn), byte.to(torch.uint8)


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


PAD_SCALE = 127  # E8M0 byte of a column block that holds no token (padding only): 1.0


def quant_grouped(x: torch.Tensor, group_lens: torch.Tensor, rule: str = "nosat", align: int = 128, row_align: int = 32):
    """MoE activation / gradient ``x [M, C]`` in expert order, ``group_lens [G]`` tokens per expert.

    Rows: ``quant_rows(x)`` with each expert's rows zero-padded to a multiple of ``row_align`` (32: Turbo's
    ``group_offs_padded_rowwise``, so an expert starts on a 32-row scale block of the packed layout; 1:
    unpadded). Columns: per expert, its tokens transposed and zero-padded to a multiple of ``align``, so no
    32-block straddles two experts: ``q_col [C, sum(pad_g)]``, ``s_col [C, sum(pad_g) / 32]``. A pad row's
    blocks and a column block holding no token get ``PAD_SCALE`` (127 = 1.0: Turbo's grouped contract,
    ``grouped_gemm_fp8_impl._alloc_mx_act_buffers``, whose quantizer gives every zero block 1.0); a block
    with tokens is quantized normally (all-zero tokens: byte 0, like any zero block). Codes of padding are 0.

    Returns (q_row, s_row, q_col, s_col, lens_col [G] int64, offs_col [G + 1] int64, lens_row, offs_row)."""
    lens = [int(n) for n in group_lens.tolist()]
    rpads = [-(-n // row_align) * row_align for n in lens]
    q_row = torch.zeros(sum(rpads), x.shape[1], dtype=torch.uint8, device=x.device)
    s_row = torch.full((sum(rpads), -(-x.shape[1] // MX)), PAD_SCALE, dtype=torch.uint8, device=x.device)
    qr, sr = quant_rows(x, rule)
    o = orow = 0
    for n, pn in zip(lens, rpads):
        q_row[orow : orow + n], s_row[orow : orow + n] = qr[o : o + n].view(torch.uint8), sr[o : o + n]
        o, orow = o + n, orow + pn
    q_row = q_row.view(torch.float8_e4m3fn)
    pads = [-(-n // align) * align for n in lens]
    q_col = torch.zeros(x.shape[1], sum(pads), dtype=torch.uint8, device=x.device)
    s_col = torch.zeros(x.shape[1], sum(pads) // MX, dtype=torch.uint8, device=x.device)
    o = oc = 0
    for n, pn in zip(lens, pads):
        if pn:
            xg = torch.zeros(pn, x.shape[1], dtype=x.dtype, device=x.device)
            xg[:n] = x[o : o + n]
            q, s = quant_rows(xg.t(), rule)
            s[:, -(-n // MX) :] = PAD_SCALE
            q_col[:, oc : oc + pn] = q.view(torch.uint8)
            s_col[:, oc // MX : (oc + pn) // MX] = s
        o, oc = o + n, oc + pn
    lens_col, lens_row = (torch.tensor(v, dtype=torch.int64, device=x.device) for v in (pads, rpads))
    offs_col, offs_row = (torch.nn.functional.pad(torch.cumsum(v, 0), (1, 0)) for v in (lens_col, lens_row))
    return q_row, s_row, q_col.view(torch.float8_e4m3fn), s_col, lens_col, offs_col, lens_row, offs_row


_GELU_A2, _GELU_B, _GELU_C2 = 2 * 0.7978845608028654, 0.044715, 6 * 0.7978845608028654 * 0.044715


def _glu_in(l1, clamp_limit):
    """gate / up halves of l1 [M, 2I] in fp32, clamped (min(gate, L), clamp(up, -L, L)), with kept masks."""
    g, u = l1.float().chunk(2, dim=1)
    if not clamp_limit:
        return g, u, None, None
    gc, uc = g.clamp(max=clamp_limit), u.clamp(-clamp_limit, clamp_limit)
    return gc, uc, gc == g, uc == u


def glu_ref(l1, probs, activation="silu", clamp_limit=None):
    """MoE GLU forward before its quantization (bf16 [M, I]): f(gate) * up * probs[row] in fp32 as the
    gfx950 / gfx1250 phase-2 kernels compute it: silu as g / (1 + exp2(-g log2 e)), gelu (tanh form) as
    g * sigmoid(2 sqrt(2/pi) g (1 + 0.044715 g^2)). The kernels' exp2 / reciprocal are the hardware's, so
    their bf16 values may be an ulp off this; their quantization is checked on their own values."""
    g, u, _, _ = _glu_in(l1, clamp_limit)
    gate = g / (1 + torch.exp2(g * -1.4426950408889634)) if activation == "silu" else g * torch.sigmoid(
        g * _GELU_A2 * (1 + g * g * _GELU_B)
    )
    return ((gate * u) * probs.float()[:, None]).to(torch.bfloat16)


def dglu_ref(grad_act, l1, probs, activation="silu", clamp_limit=None):
    """MoE GLU backward before quantization: (grad_l1 = [dgate | dup] [M, 2I] bf16, grad_probs [M] fp32)
    with d = grad_act, a = f(gate): dgate = d p up f'(gate), dup = d p a (zero where the clamp bit),
    grad_probs = sum_j d a up."""
    g, u, kg, ku = _glu_in(l1, clamp_limit)
    if activation == "silu":
        sg = torch.sigmoid(g)
        a, da = sg * g, sg * (1 + g - sg * g)
    else:
        sg = torch.sigmoid(g * _GELU_A2 * (1 + g * g * _GELU_B))
        a, da = g * sg, sg + g * sg * (1 - sg) * (_GELU_A2 + g * g * _GELU_C2)
    d = grad_act.float()
    dp = d * probs.float()[:, None]
    dg, du = dp * u * da, dp * a
    if kg is not None:
        dg, du = torch.where(kg, dg, 0.0), torch.where(ku, du, 0.0)
    return torch.cat([dg, du], 1).to(torch.bfloat16), (d * a * u).sum(1)


def gather_rows(x, dest2src, permuted_probs=None):
    """The permuted activation: row d = x[dest2src[d]]; zero where dest2src is out of range or the
    row's prob is exactly 0 (the routing padding rule)."""
    idx = dest2src.long()
    ok = (idx >= 0) & (idx < x.shape[0])
    if permuted_probs is not None:
        ok &= permuted_probs != 0
    return torch.where(ok[:, None], x[idx.clamp(0, x.shape[0] - 1)], torch.zeros((), dtype=x.dtype))


def pack_scales(s: torch.Tensor) -> torch.Tensor:
    """Raw E8M0 [R, ceil(K/32)] -> the GEMM layout [ceil(R/32), ceil(K/128), 32, 4], pad bytes 0x00."""
    R, KB = s.shape
    Rp, KBp = -(-R // 32) * 32, -(-KB // 4) * 4
    sp = torch.zeros(Rp, KBp, dtype=torch.uint8, device=s.device)
    sp[:R, :KB] = s
    return sp.view(Rp // 32, 32, KBp // 4, 4).permute(0, 2, 1, 3).contiguous()


def preshuffle_b16(q: torch.Tensor) -> torch.Tensor:
    """A GEMM B operand [.., N, K] in the 16x16-preshuffled byte order of the gfx1250 MX GEMM
    (``[N/16][K/16][16][16]``: each 16x16 block's 256 bytes contiguous), same shape and dtype."""
    *b, N, K = q.shape
    u = q.contiguous().view(torch.uint8).view(*b, N // 16, 16, K // 16, 16)
    return u.transpose(-3, -2).contiguous().view(*b, N, K).view(q.dtype)


def unpreshuffle_b16(q: torch.Tensor) -> torch.Tensor:
    """Inverse of ``preshuffle_b16``."""
    *b, N, K = q.shape
    u = q.contiguous().view(torch.uint8).view(*b, N // 16, K // 16, 16, 16)
    return u.transpose(-3, -2).contiguous().view(*b, N, K).view(q.dtype)


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
