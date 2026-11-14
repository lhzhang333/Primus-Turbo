###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The placeholder gradients of the fused-wgrad ops (``fuse_bgrad_accum_pattern="megatron"``): views of one flat
buffer per (dtype, device) handed out by ``get_dummy_wgrad``, which autograd takes over as the weight's ``.grad``
without a copy."""

import pytest
import torch

import primus_turbo.pytorch.ops.utils as ops_utils
from primus_turbo.pytorch.ops import DUMMY_WGRAD_FLAT_POOL, get_dummy_wgrad, grouped_gemm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def test_pool_views():
    """One buffer per (dtype, device), however the device is named: views of its prefix whatever the shape; a larger
    shape grows it (earlier views stay valid); zero=True zeroes just the requested view."""
    assert DUMMY_WGRAD_FLAT_POOL
    fp16, dev = torch.float16, torch.device("cuda", torch.cuda.current_device())
    small = get_dummy_wgrad([3, 5], fp16, "cuda")
    assert small.shape == (3, 5) and small.is_contiguous() and not small.requires_grad
    assert get_dummy_wgrad([2, 2], fp16, dev).data_ptr() == small.data_ptr()
    # the ops' internal call takes the same pool
    assert ops_utils._get_dummy_wgrad([1], fp16).data_ptr() == small.data_ptr()
    big = get_dummy_wgrad([small.numel() + (1 << 20)], fp16, dev.index)  # grows the buffer
    assert get_dummy_wgrad([3], fp16, "cuda").data_ptr() == big.data_ptr() and small.shape == (3, 5)
    big.fill_(1)
    zeroed = get_dummy_wgrad([4, 4], fp16, "cuda", zero=True)
    assert not zeroed.any() and big[16:].eq(1).all()
    other_dtype = get_dummy_wgrad([4, 4], torch.bfloat16, "cuda")  # one buffer per dtype
    assert other_dtype.data_ptr() != zeroed.data_ptr()


def test_fused_wgrad_takes_a_pool_view():
    """bf16 grouped_gemm with its wgrad fused into an fp32 main_grad: main_grad gets the wgrad, and the weight's
    .grad is a view of the pool (taken over by AccumulateGrad without a copy), not a buffer per weight shape."""
    G, M, N, K = 2, 128, 256, 128
    w = torch.nn.Parameter(torch.randn(G, N, K, device="cuda", dtype=torch.bfloat16))
    w.main_grad, w.grad_added_to_main_grad = torch.zeros(G, N, K, device="cuda"), False
    x = torch.randn(G * M, K, device="cuda", dtype=torch.bfloat16)
    lens = torch.full((G,), M, device="cuda", dtype=torch.int64)
    grouped_gemm(x, w, lens, trans_b=True, fuse_bgrad_accum_pattern="megatron").sum().backward()
    assert w.grad_added_to_main_grad and w.main_grad.abs().sum() > 0
    assert w.grad.shape == w.shape
    assert w.grad.data_ptr() == get_dummy_wgrad([1], torch.bfloat16, "cuda").data_ptr()
