###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import triton
import triton.language as tl


@triton.jit
def swiglu_with_mask_fwd_kernel(
    # pointers
    x_ptr,
    probs_ptr,
    row_mask_ptr,
    out_ptr,
    # sizes
    num_tokens: tl.constexpr,
    # strides
    stride_x_token,
    stride_probs_token,
    stride_out_token,
    # clamp
    clamp_limit,
    # metas
    LOAD_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_CLAMP: tl.constexpr,
):
    pid = tl.program_id(0)

    compute_type = tl.float32
    data_type = x_ptr.dtype.element_ty
    idx_type = tl.int64

    half_stride_x_token = stride_x_token // 2
    loop = (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    for i in range(0, loop):
        row_idx = (i * BLOCK_SIZE + pid).to(idx_type)
        row_mask = row_idx < num_tokens
        col_off = tl.arange(0, LOAD_WIDTH)
        col_mask = col_off < half_stride_x_token

        extra_mask = tl.load(row_mask_ptr + row_idx, mask=row_mask)
        mask = (row_mask & col_mask) & (extra_mask != 0)

        up_ptr = x_ptr + row_idx * stride_x_token
        down_ptr = up_ptr + half_stride_x_token

        up = tl.load(up_ptr + col_off, mask=mask).to(compute_type)
        down = tl.load(down_ptr + col_off, mask=mask).to(compute_type)

        if HAS_CLAMP:
            limit = clamp_limit.to(compute_type)
            up = tl.minimum(up, limit)
            down = tl.minimum(tl.maximum(down, -limit), limit)

        up = tl.fdiv(up, (1.0 + tl.exp(-up)))
        out = up * down

        probs = tl.load(probs_ptr + row_idx * stride_probs_token, mask=row_mask)
        out = out * probs

        tl.store(out_ptr + row_idx * stride_out_token + col_off, out.to(data_type), mask=mask)


@triton.jit
def swiglu_with_mask_bwd_kernel(
    # pointers
    grad_out_ptr,
    x_ptr,
    probs_ptr,
    row_mask_ptr,
    grad_x_ptr,
    grad_probs_ptr,
    # sizes
    num_tokens: tl.constexpr,
    # strides
    stride_grad_out_token,
    stride_x_token,
    stride_probs_token,
    stride_grad_x_token,
    stride_grad_probs_token,
    # clamp
    clamp_limit,
    # metas
    LOAD_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_CLAMP: tl.constexpr,
):
    pid = tl.program_id(0)

    compute_type = tl.float32
    grad_x_data_type = grad_x_ptr.dtype.element_ty
    grad_probs_data_type = grad_probs_ptr.dtype.element_ty
    idx_type = tl.int64

    half_stride_x_token = stride_x_token // 2
    loop = (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    for i in range(0, loop):
        row_idx = (i * BLOCK_SIZE + pid).to(idx_type)
        row_mask = row_idx < num_tokens
        col_off = tl.arange(0, LOAD_WIDTH)
        col_mask = col_off < half_stride_x_token

        extra_mask = tl.load(row_mask_ptr + row_idx, mask=row_mask)
        mask = (row_mask & col_mask) & (extra_mask != 0)

        up_ptr = x_ptr + row_idx * stride_x_token
        down_ptr = up_ptr + half_stride_x_token

        up = tl.load(up_ptr + col_off, mask=mask).to(compute_type)
        down = tl.load(down_ptr + col_off, mask=mask).to(compute_type)

        if HAS_CLAMP:
            limit = clamp_limit.to(compute_type)
            up_kept = up <= limit
            down_kept = (down >= -limit) & (down <= limit)
            up = tl.minimum(up, limit)
            down = tl.minimum(tl.maximum(down, -limit), limit)

        sigmoid = tl.sigmoid(up)
        silu = sigmoid * up

        grad_out = tl.load(grad_out_ptr + row_idx * stride_grad_out_token + col_off, mask=mask).to(
            compute_type
        )

        grad_probs = grad_out * silu * down
        grad_probs_sum = tl.sum(grad_probs, dtype=compute_type)

        tl.store(
            grad_probs_ptr + row_idx * stride_grad_probs_token,
            grad_probs_sum.to(grad_probs_data_type),
            mask=row_mask,
        )

        probs = tl.load(probs_ptr + row_idx * stride_probs_token, mask=row_mask).to(compute_type)

        grad_out_with_probs = grad_out * probs
        grad_down = grad_out_with_probs * silu
        grad_silu = sigmoid * (1.0 + up * (1.0 - sigmoid))
        grad_up = grad_out_with_probs * (down * grad_silu)

        if HAS_CLAMP:
            grad_up = tl.where(up_kept, grad_up, 0.0)
            grad_down = tl.where(down_kept, grad_down, 0.0)

        tl.store(
            grad_x_ptr + row_idx * stride_grad_x_token + col_off, grad_up.to(grad_x_data_type), mask=mask
        )
        tl.store(
            grad_x_ptr + row_idx * stride_grad_x_token + stride_grad_x_token // 2 + col_off,
            grad_down.to(grad_x_data_type),
            mask=mask,
        )


@triton.jit
def swiglu_fwd_kernel(
    # pointers
    x_ptr,
    probs_ptr,
    out_ptr,
    # sizes
    num_tokens: tl.constexpr,
    # strides
    stride_x_token: tl.constexpr,
    stride_probs_token: tl.constexpr,
    stride_out_token: tl.constexpr,
    # clamp
    clamp_limit,
    # metas
    LOAD_WIDTH: tl.constexpr,
    HAS_CLAMP: tl.constexpr,
):
    pid = tl.program_id(0)

    compute_type = tl.float32
    data_type = x_ptr.dtype.element_ty

    half_stride_x_token = stride_x_token // 2

    row_idx = pid.to(tl.int64)
    row_mask = row_idx < num_tokens
    col_off = tl.arange(0, LOAD_WIDTH)
    col_mask = col_off < half_stride_x_token

    mask = row_mask & col_mask

    up_ptr = x_ptr + row_idx * stride_x_token
    down_ptr = up_ptr + half_stride_x_token

    up = tl.load(up_ptr + col_off, mask=mask).to(compute_type)
    down = tl.load(down_ptr + col_off, mask=mask).to(compute_type)

    if HAS_CLAMP:
        limit = clamp_limit.to(compute_type)
        up = tl.minimum(up, limit)
        down = tl.minimum(tl.maximum(down, -limit), limit)

    up = tl.fdiv(up, (1.0 + tl.exp(-up)))
    out = up * down

    probs = tl.load(probs_ptr + row_idx * stride_probs_token)
    out = out * probs

    tl.store(out_ptr + row_idx * stride_out_token + col_off, out.to(data_type), mask=mask)


@triton.jit
def swiglu_bwd_kernel(
    # pointers
    grad_out_ptr,
    x_ptr,
    probs_ptr,
    grad_x_ptr,
    grad_probs_ptr,
    # sizes
    num_tokens: tl.constexpr,
    # strides
    stride_grad_out_token: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_probs_token: tl.constexpr,
    stride_grad_x_token: tl.constexpr,
    stride_grad_probs_token: tl.constexpr,
    # clamp
    clamp_limit,
    # metas
    LOAD_WIDTH: tl.constexpr,
    HAS_CLAMP: tl.constexpr,
):
    pid = tl.program_id(0)

    compute_type = tl.float32
    grad_x_data_type = grad_x_ptr.dtype.element_ty
    grad_probs_data_type = grad_probs_ptr.dtype.element_ty

    half_stride_x_token = stride_x_token // 2

    row_idx = pid.to(tl.int64)
    row_mask = row_idx < num_tokens
    col_off = tl.arange(0, LOAD_WIDTH)
    col_mask = col_off < half_stride_x_token

    mask = row_mask & col_mask

    up_ptr = x_ptr + row_idx * stride_x_token
    down_ptr = up_ptr + half_stride_x_token

    up = tl.load(up_ptr + col_off, mask=mask).to(compute_type)
    down = tl.load(down_ptr + col_off, mask=mask).to(compute_type)

    if HAS_CLAMP:
        limit = clamp_limit.to(compute_type)
        up_kept = up <= limit
        down_kept = (down >= -limit) & (down <= limit)
        up = tl.minimum(up, limit)
        down = tl.minimum(tl.maximum(down, -limit), limit)

    sigmoid = tl.sigmoid(up)
    silu = sigmoid * up

    grad_out = tl.load(grad_out_ptr + row_idx * stride_grad_out_token + col_off, mask=mask).to(compute_type)

    grad_probs = grad_out * silu * down
    grad_probs_sum = tl.sum(grad_probs, dtype=compute_type)

    tl.store(
        grad_probs_ptr + row_idx * stride_grad_probs_token,
        grad_probs_sum.to(grad_probs_data_type),
        mask=row_mask,
    )

    probs = tl.load(probs_ptr + row_idx * stride_probs_token).to(compute_type)

    grad_out_with_probs = grad_out * probs
    grad_down = grad_out_with_probs * silu
    grad_silu = sigmoid * (1.0 + up * (1.0 - sigmoid))
    grad_up = grad_out_with_probs * (down * grad_silu)

    if HAS_CLAMP:
        grad_up = tl.where(up_kept, grad_up, 0.0)
        grad_down = tl.where(down_kept, grad_down, 0.0)

    tl.store(grad_x_ptr + row_idx * stride_grad_x_token + col_off, grad_up.to(grad_x_data_type), mask=mask)
    tl.store(
        grad_x_ptr + row_idx * stride_grad_x_token + stride_grad_x_token // 2 + col_off,
        grad_down.to(grad_x_data_type),
        mask=mask,
    )


@triton.jit
def swiglu_dense_fwd_kernel(
    x_ptr,
    out_ptr,
    num_rows,
    half_cols,
    stride_x_row,
    stride_out_row,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """``out = silu(x[:, :N]) * x[:, N:]`` over a 2-D grid of [BLOCK_M, BLOCK_N] tiles, fp32 math."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < num_rows) & (cols[None, :] < half_cols)
    x_offs = rows[:, None] * stride_x_row + cols[None, :]
    gate = tl.load(x_ptr + x_offs, mask=mask).to(tl.float32)
    up = tl.load(x_ptr + x_offs + half_cols, mask=mask).to(tl.float32)
    out = gate * tl.sigmoid(gate) * up
    tl.store(
        out_ptr + rows[:, None] * stride_out_row + cols[None, :],
        out.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def swiglu_dense_bwd_kernel(
    grad_out_ptr,
    x_ptr,
    grad_x_ptr,
    num_rows,
    half_cols,
    stride_grad_out_row,
    stride_x_row,
    stride_grad_x_row,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Gradient of :func:`swiglu_dense_fwd_kernel`; both halves of ``grad_x`` in one pass, fp32 math."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < num_rows) & (cols[None, :] < half_cols)
    x_offs = rows[:, None] * stride_x_row + cols[None, :]
    gate = tl.load(x_ptr + x_offs, mask=mask).to(tl.float32)
    up = tl.load(x_ptr + x_offs + half_cols, mask=mask).to(tl.float32)
    grad_out = tl.load(grad_out_ptr + rows[:, None] * stride_grad_out_row + cols[None, :], mask=mask).to(
        tl.float32
    )
    sigmoid = tl.sigmoid(gate)
    # the expression order of Megatron's swiglu_back
    grad_gate = grad_out * sigmoid * (1 + gate * (1 - sigmoid)) * up
    grad_up = grad_out * (gate * sigmoid)
    grad_x_offs = rows[:, None] * stride_grad_x_row + cols[None, :]
    tl.store(grad_x_ptr + grad_x_offs, grad_gate.to(grad_x_ptr.dtype.element_ty), mask=mask)
    tl.store(grad_x_ptr + grad_x_offs + half_cols, grad_up.to(grad_x_ptr.dtype.element_ty), mask=mask)
