###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MegaMoE in MXFP8 on gfx1250 across every visible GPU (EP2+), against the fp32 reference of the single-GPU bf16
suite: each rank compares its own tokens and its own experts' gradients (the MX recipe's ~23 dB; 20 dB gate, the
weakest rank deciding). Also: y / dx bitwise across calls on the same inputs (the pool order only moves the
wgrads), the wgrads into a bf16 main_grad, and an opt-in soak (MEGA_MX_SOAK=<calls>)."""

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import run_tests

import primus_turbo.pytorch  # noqa: F401
from primus_turbo.flydsl.utils.glu_activation import GLUActivation
from primus_turbo.pytorch.core.utils import is_gfx1250

if not is_gfx1250():
    pytest.skip("gfx1250 kernels", allow_module_level=True)

from primus_turbo.pytorch.kernels.fused_mega_moe import advance_weight_generation  # noqa: E402
from primus_turbo.pytorch.ops.moe.fused_mega_moe_fp8 import (  # noqa: E402
    fused_mega_moe_fp8_stage1,
    fused_mega_moe_fp8_stage2,
)
from tests.pytorch.ops.test_mega_bf16_gfx1250_fused_moe import (  # noqa: E402
    _reference,
    _snr,
)


class FusedMegaMoEMXFP8Gfx1250EPTest(MultiProcContinuousTest):
    timeout = timedelta(seconds=900)

    @classmethod
    def backend_str(cls) -> str:
        return "nccl"

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def _gather_rows(self, t):
        """Every rank's rows of ``t`` rank-major, and where this rank's start; counts may differ."""
        n = torch.tensor([t.shape[0]], device=self.device)
        counts = [torch.empty_like(n) for _ in range(self.world_size)]
        dist.all_gather(counts, n)
        counts = [int(c) for c in counts]
        buf = t.new_zeros((max(counts),) + t.shape[1:])
        buf[: t.shape[0]] = t
        out = [torch.empty_like(buf) for _ in counts]
        dist.all_gather(out, buf)
        return torch.cat([o[:c] for o, c in zip(out, counts)]), sum(counts[: self.rank])

    def _inputs(self, num_tokens, hidden, inter, experts_per_rank, num_topk, skew=0.0, seed=0):
        gen = torch.Generator(device=self.device).manual_seed(1234 + 17 * seed + self.rank)

        def randn(*shape, scale=1.0):
            return (torch.randn(*shape, device=self.device, generator=gen) * scale).to(torch.bfloat16)

        num_experts = experts_per_rank * self.world_size
        x, dy = randn(num_tokens, hidden), randn(num_tokens, hidden)
        logits = torch.randn(num_tokens, num_experts, device=self.device, generator=gen)
        logits[:, :experts_per_rank] += skew  # crowd rank 0's experts
        topk_w, topk_idx = logits.softmax(-1).topk(num_topk, dim=-1)
        w1 = randn(experts_per_rank, 2 * inter, hidden, scale=2 / hidden**0.5)
        w2 = randn(experts_per_rank, hidden, inter, scale=2 / inter**0.5)
        return x, topk_idx, topk_w, w1, w2, dy

    def _step(self, x, topk_idx, topk_w, w1, w2, dy, activation, fuse=False):
        """One staged forward + backward; ``(y, dx, d topk_w, dW1, dW2)`` (the wgrads from main_grad with ``fuse``).
        A step of its own: the MX weight cache is keyed by address, which fresh weight clones may reuse."""
        advance_weight_generation()
        leaves = [t.clone().requires_grad_() for t in (x, topk_w, w1, w2)]
        pattern = "megatron" if fuse else None
        if fuse:
            for w in leaves[2:]:
                w.main_grad = torch.zeros_like(w)
                w.grad_added_to_main_grad = False
        group = dist.group.WORLD
        l1, dw, handle, state = fused_mega_moe_fp8_stage1(leaves[0], topk_idx, leaves[1], leaves[2], group, pattern)
        y = fused_mega_moe_fp8_stage2(
            l1, dw, handle, state, topk_idx, leaves[1], leaves[3], group, activation, fuse_wgrad_accum_pattern=pattern
        )
        y.backward(dy)
        wgrads = [w.main_grad if fuse else w.grad for w in leaves[2:]]
        return [y.detach(), leaves[0].grad, leaves[1].grad] + wgrads

    def _check(self, num_tokens, hidden, inter, experts_per_rank, num_topk, *, activation=None, token_step=0,
               skew=0.0, fuse=False, calls=2):
        torch.cuda.set_device(self.device)
        activation = activation or GLUActivation.swiglu()
        num_tokens += token_step * self.rank
        experts = slice(self.rank * experts_per_rank, (self.rank + 1) * experts_per_rank)
        for call in range(calls):
            x, topk_idx, topk_w, w1, w2, dy = self._inputs(num_tokens, hidden, inter, experts_per_rank, num_topk,
                                                           skew, call)
            got = self._step(x, topk_idx, topk_w, w1, w2, dy, activation, fuse)
            gx, row0 = self._gather_rows(x)
            refs = [t.detach().float().requires_grad_()
                    for t in (gx, self._gather_rows(topk_w)[0], self._gather_rows(w1)[0], self._gather_rows(w2)[0])]
            yr = _reference(refs[0], self._gather_rows(topk_idx)[0], *refs[1:], activation)
            yr.backward(self._gather_rows(dy)[0].float())
            rows = slice(row0, row0 + num_tokens)
            wants = [yr.detach()[rows], refs[0].grad[rows], refs[1].grad[rows], refs[2].grad[experts],
                     refs[3].grad[experts]]
            snr = torch.tensor([_snr(g, w) for g, w in zip(got, wants)], device=self.device)
            dist.all_reduce(snr, op=dist.ReduceOp.MIN)
            for name, s in zip(("y", "dx", "d topk_w", "dW1", "dW2"), snr.tolist()):
                self.assertGreater(s, 20, f"{name} SNR {s:.1f} dB on the weakest rank")

    @skip_if_lt_x_gpu(2)
    def test_deepseek_v3(self):
        # DeepSeek-V3's expert shape with its EP8 expert count per rank; the wgrads into main_grad too
        self._check(1024, 7168, 2048, 32, 8)
        self._check(1024, 7168, 2048, 32, 8, fuse=True, calls=1)

    @skip_if_lt_x_gpu(2)
    def test_ragged_skewed(self):
        # different token counts per rank and most routes crowding rank 0's experts (empty experts elsewhere)
        self._check(333, 7168, 2048, 32, 8, token_step=7, skew=3.0)

    @skip_if_lt_x_gpu(2)
    def test_one_token_clamped(self):
        # one token per rank, a small expert shape, Megatron's clamp form of the activation
        self._check(1, 2048, 512, 4, 4, activation=GLUActivation.swiglu(7.0))

    @skip_if_lt_x_gpu(2)
    def test_repeat_bitwise(self):
        """y / dx / d topk_w bitwise across calls on the same inputs: the fold and reduce orders are fixed; only the
        wgrads see the pool order, which the dispatch prologue's atomics may change."""
        torch.cuda.set_device(self.device)
        calls = int(os.environ.get("MEGA_MX_SOAK", "3"))
        inputs = self._inputs(1024, 7168, 2048, 32, 8, skew=1.0)
        first = self._step(*inputs, GLUActivation.swiglu())
        bad = torch.zeros(1, device=self.device)
        for _ in range(calls - 1):
            again = self._step(*inputs, GLUActivation.swiglu())
            for a, b in zip(first[:3], again[:3]):
                bad += int(not torch.equal(a.view(torch.uint8), b.view(torch.uint8)))
        dist.all_reduce(bad)
        self.assertEqual(int(bad.item()), 0, "y / dx / d topk_w changed between calls")


if __name__ == "__main__":
    run_tests()
