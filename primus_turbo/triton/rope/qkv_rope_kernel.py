###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Triton fused QKV rotary position embedding: the same contract as the FlyDSL kernels in
``primus_turbo.flydsl.rope.rope_kernel``.

Forward: split a packed ``[S, B, NG, (NPG + 2) * D]`` QKV tensor (per group: NPG q heads, one k head,
one v head) into contiguous q ``[S, B, NG * NPG, D]``, k and v ``[S, B, NG, D]``, rotating q and k
(non-interleaved rotate-half) and copying v:

    lo, hi = x[:D/2], x[D/2:]
    out_lo = lo * cos(f) - hi * sin(f)
    out_hi = hi * cos(f) + lo * sin(f)

with ``f = freqs[s, j]`` (only the low half of the ``[S, 1, 1, D]`` table is read; Megatron's tables
repeat it), cos / sin and the products in fp32, rounded once to the output dtype.

Backward: the transposed rotation of dq and dk, dv copied, into a packed ``[S, B, NG, (NPG + 2) * D]``
gradient. dq / dk / dv may be any ``[S, B, heads, D]`` views with a unit last-dim stride (for example
the ``[B, S, ...]`` buffers an attention backward hands back, transposed): the kernel reads them
through their strides instead of copying them first.

A program owns ``ROWS`` consecutive (s, b) rows of one KV group, so cos / sin are taken once per row
for all the group's heads; the grid is ``(ceil(S * B / ROWS), NG)``.
"""

import triton
import triton.language as tl


@triton.jit
def qkv_rope_fwd_kernel(
    QKV,
    Q_FREQS,
    K_FREQS,
    Q,
    K,
    V,
    n_rows,
    B,
    stride_qf,
    stride_kf,
    NG: tl.constexpr,
    NPG: tl.constexpr,
    NPG_P2: tl.constexpr,
    D: tl.constexpr,
    ROWS: tl.constexpr,
):
    HALF: tl.constexpr = D // 2
    W: tl.constexpr = (NPG + 2) * D
    g = tl.program_id(1)
    r = (tl.program_id(0) * ROWS + tl.arange(0, ROWS)).to(tl.int64)
    j = tl.arange(0, HALF)
    h = tl.arange(0, NPG_P2)
    rmask = r < n_rows
    m2 = rmask[:, None]
    m3 = rmask[:, None, None] & (h < NPG)[None, :, None]
    s = r // B

    fq = tl.load(Q_FREQS + s[:, None] * stride_qf + j[None, :], mask=m2, other=0.0)
    fk = tl.load(K_FREQS + s[:, None] * stride_kf + j[None, :], mask=m2, other=0.0)
    cq = tl.cos(fq)[:, None, :]
    sq = tl.sin(fq)[:, None, :]
    ck = tl.cos(fk)
    sk = tl.sin(fk)

    row = QKV + r * (NG * W) + g * W
    src = row[:, None, None] + h[None, :, None] * D + j[None, None, :]
    lo = tl.load(src, mask=m3).to(tl.float32)
    hi = tl.load(src + HALF, mask=m3).to(tl.float32)
    dst = Q + r[:, None, None] * (NG * NPG * D) + (g * NPG + h[None, :, None]) * D + j[None, None, :]
    tl.store(dst, (lo * cq - hi * sq).to(Q.dtype.element_ty), mask=m3)
    tl.store(dst + HALF, (hi * cq + lo * sq).to(Q.dtype.element_ty), mask=m3)

    ksrc = row[:, None] + NPG * D + j[None, :]
    klo = tl.load(ksrc, mask=m2).to(tl.float32)
    khi = tl.load(ksrc + HALF, mask=m2).to(tl.float32)
    kdst = K + r[:, None] * (NG * D) + g * D + j[None, :]
    tl.store(kdst, (klo * ck - khi * sk).to(K.dtype.element_ty), mask=m2)
    tl.store(kdst + HALF, (khi * ck + klo * sk).to(K.dtype.element_ty), mask=m2)

    d = tl.arange(0, D)
    v = tl.load(row[:, None] + (NPG + 1) * D + d[None, :], mask=m2)
    tl.store(V + r[:, None] * (NG * D) + g * D + d[None, :], v, mask=m2)


@triton.jit
def qkv_rope_bwd_kernel(
    DQ,
    DK,
    DV,
    Q_FREQS,
    K_FREQS,
    DQKV,
    n_rows,
    B,
    stride_qf,
    stride_kf,
    stride_dq_s,
    stride_dq_b,
    stride_dq_h,
    stride_dk_s,
    stride_dk_b,
    stride_dk_h,
    stride_dv_s,
    stride_dv_b,
    stride_dv_h,
    NG: tl.constexpr,
    NPG: tl.constexpr,
    NPG_P2: tl.constexpr,
    D: tl.constexpr,
    ROWS: tl.constexpr,
):
    HALF: tl.constexpr = D // 2
    W: tl.constexpr = (NPG + 2) * D
    g = tl.program_id(1)
    r = (tl.program_id(0) * ROWS + tl.arange(0, ROWS)).to(tl.int64)
    j = tl.arange(0, HALF)
    h = tl.arange(0, NPG_P2)
    rmask = r < n_rows
    m2 = rmask[:, None]
    m3 = rmask[:, None, None] & (h < NPG)[None, :, None]
    s = r // B
    b = r % B

    fq = tl.load(Q_FREQS + s[:, None] * stride_qf + j[None, :], mask=m2, other=0.0)
    fk = tl.load(K_FREQS + s[:, None] * stride_kf + j[None, :], mask=m2, other=0.0)
    cq = tl.cos(fq)[:, None, :]
    sq = tl.sin(fq)[:, None, :]
    ck = tl.cos(fk)
    sk = tl.sin(fk)

    out = DQKV + r * (NG * W) + g * W
    src = DQ + (s * stride_dq_s + b * stride_dq_b)[:, None, None]
    src = src + (g * NPG + h[None, :, None]) * stride_dq_h + j[None, None, :]
    lo = tl.load(src, mask=m3).to(tl.float32)
    hi = tl.load(src + HALF, mask=m3).to(tl.float32)
    dst = out[:, None, None] + h[None, :, None] * D + j[None, None, :]
    tl.store(dst, (lo * cq + hi * sq).to(DQKV.dtype.element_ty), mask=m3)
    tl.store(dst + HALF, (hi * cq - lo * sq).to(DQKV.dtype.element_ty), mask=m3)

    ksrc = DK + (s * stride_dk_s + b * stride_dk_b)[:, None] + g * stride_dk_h + j[None, :]
    klo = tl.load(ksrc, mask=m2).to(tl.float32)
    khi = tl.load(ksrc + HALF, mask=m2).to(tl.float32)
    kdst = out[:, None] + NPG * D + j[None, :]
    tl.store(kdst, (klo * ck + khi * sk).to(DQKV.dtype.element_ty), mask=m2)
    tl.store(kdst + HALF, (khi * ck - klo * sk).to(DQKV.dtype.element_ty), mask=m2)

    d = tl.arange(0, D)
    v = tl.load(DV + (s * stride_dv_s + b * stride_dv_b)[:, None] + g * stride_dv_h + d[None, :], mask=m2)
    tl.store(out[:, None] + (NPG + 1) * D + d[None, :], v, mask=m2)
