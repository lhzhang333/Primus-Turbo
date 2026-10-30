###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Two rows a pass in the grid-stride RMSNorm backward: dx and the dgamma partials bit for bit against one."""

import pytest
import torch

from primus_turbo.triton.normalization import rmsnorm_kernel as rk

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("B, H", [(8192, 7168), (1001, 4096), (777, 5120), (300, 2880)])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.parametrize("zero_centered", [False, True])
def test_two_rows_a_pass_is_bit_identical(B, H, num_warps, zero_centered):
    block_h = 1 << (H - 1).bit_length()
    programs = 128
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(B, H, device="cuda", generator=g).to(torch.bfloat16)
    dy = torch.randn(B, H, device="cuda", generator=g).to(torch.bfloat16)
    gamma = (0.1 * torch.randn(H, device="cuda", generator=g)).to(torch.bfloat16)
    rstd = torch.rsqrt(x.float().pow(2).mean(-1) + 1e-6)

    def run(rows):
        dx = torch.full_like(x, float("nan"))
        part = torch.full((programs, H), float("nan"), device="cuda", dtype=torch.float32)
        rk.rmsnorm_bwd_kernel_grid_stride.fn[(programs,)](
            dy,
            x,
            gamma,
            rstd,
            dx,
            part,
            x.stride(0),
            x.stride(1),
            dy.stride(0),
            dy.stride(1),
            dx.stride(0),
            dx.stride(1),
            part.stride(0),
            B=B,
            H=H,
            BLOCK_H=block_h,
            num_programs=programs,
            ZERO_CENTERED=zero_centered,
            ROWS_PER_ITER=rows,
            num_warps=num_warps,
            num_stages=1,
        )
        return dx, part

    (dx1, p1), (dx2, p2) = run(1), run(2)
    assert not dx1.isnan().any() and not p1.isnan().any()
    assert torch.equal(dx1.view(torch.int16), dx2.view(torch.int16))
    assert torch.equal(p1.view(torch.int32), p2.view(torch.int32))
