###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import functools
import inspect
from typing import Tuple

import torch

_torch_custom_op_wrapper = torch.library.custom_op

from primus_turbo.flydsl.gemm.gemm_fp8_kernel import gemm_fp8_tensorwise_flydsl_kernel
from primus_turbo.flydsl.gemm.gemm_mxfp8_kernel import gemm_mxfp8_flydsl_kernel
from primus_turbo.pytorch.core.backend import (
    AutoKernelDispatcher,
    BackendChoice,
    BackendEntry,
    BackendType,
    GlobalBackendManager,
    KernelBackend,
    PrecisionType,
    TuneCache,
)
from primus_turbo.pytorch.core.low_precision import (
    ScalingGranularity,
    float8_e4m3,
    float8_e5m2,
)
from primus_turbo.pytorch.core.utils import (
    build_ck,
    is_gfx942,
    is_gfx950,
    is_gfx1250,
)
from primus_turbo.triton.gemm.gemm_fp8_kernel import (
    gemm_fp8_blockwise_triton_kernel,
    gemm_fp8_rowwise_triton_kernel,
    gemm_fp8_tensorwise_triton_kernel,
)


def get_gemm_logical_shape(
    a: torch.Tensor, b: torch.Tensor, trans_a: bool, trans_b: bool
) -> Tuple[int, int, int]:
    assert a.ndim == 2 and b.ndim == 2, (
        f"Expected both a and b to be 2D tensors, but got a.ndim={a.ndim}, b.ndim={b.ndim}"
    )
    M = a.shape[1] if trans_a else a.shape[0]
    Ka = a.shape[0] if trans_a else a.shape[1]
    Kb = b.shape[1] if trans_b else b.shape[0]
    N = b.shape[0] if trans_b else b.shape[1]
    assert Ka == Kb, f"GEMM K mismatch: a has K={Ka}, b has K={Kb}"
    return M, N, Ka


_COMMON_SUPPORTED_DTYPES = (
    (float8_e4m3, float8_e4m3, torch.float16),
    (float8_e4m3, float8_e4m3, torch.bfloat16),
    (float8_e5m2, float8_e5m2, torch.float16),
    (float8_e5m2, float8_e5m2, torch.bfloat16),
)

_HYBRID_SUPPORTED_DTYPES = (
    (float8_e4m3, float8_e5m2, torch.float16),
    (float8_e4m3, float8_e5m2, torch.bfloat16),
    (float8_e5m2, float8_e4m3, torch.float16),
    (float8_e5m2, float8_e4m3, torch.bfloat16),
)


class GEMMFP8HipBLASLtBackend(KernelBackend):
    SUPPORTED_GRANULARITIES = {
        ScalingGranularity.TENSORWISE,
        ScalingGranularity.MX_BLOCKWISE,
    }

    # (a_dtype, b_dtype, c_dtype)
    SUPPORTED_DTYPES = set(_COMMON_SUPPORTED_DTYPES + _HYBRID_SUPPORTED_DTYPES)

    # (trans_a, trans_b, trans_c)
    SUPPORTED_LAYOUTS = (
        (False, False, False),
        (False, True, False),
        (True, False, False),
    )

    @staticmethod
    def can_handle(
        a: torch.Tensor,
        a_scale_inv: torch.Tensor,
        trans_a: bool,
        b: torch.Tensor,
        b_scale_inv: torch.Tensor,
        trans_b: bool,
        out_dtype: torch.dtype,
        trans_c: bool,
        granularity: ScalingGranularity,
        inplace_add_to_out: bool = False,
        out: torch.Tensor | None = None,
        **kwargs,
    ) -> bool:
        supported = True
        if granularity == ScalingGranularity.MX_BLOCKWISE:
            supported &= not is_gfx942()
            # gfx1250: hipBLASLt's MX GEMM disagrees with its own dequantized operands (~8 dB SNR, 2026-11-11), so the
            # dispatcher takes the Triton MX path there
            supported &= not is_gfx1250()
        # check ScalingGranularity
        supported &= granularity in GEMMFP8HipBLASLtBackend.SUPPORTED_GRANULARITIES
        # check dtype
        supported &= (a.dtype, b.dtype, out_dtype) in GEMMFP8HipBLASLtBackend.SUPPORTED_DTYPES

        if inplace_add_to_out:
            supported &= out is not None and out.is_contiguous()
            supported &= out is not None and out.dtype in (
                torch.float32,
                torch.bfloat16,
                torch.float16,
            )

        # TODO:
        # check layout
        # supported &= (trans_a, trans_b, trans_c) in GEMMFP8HipBLASLtBackend.SUPPORTED_LAYOUTS
        # TODO:
        # check shape

        return supported

    @staticmethod
    def execute(
        a: torch.Tensor,
        a_scale_inv: torch.Tensor,
        trans_a: bool,
        b: torch.Tensor,
        b_scale_inv: torch.Tensor,
        trans_b: bool,
        out_dtype: torch.dtype,
        trans_c: bool,
        granularity: ScalingGranularity,
        inplace_add_to_out: bool = False,
        out: torch.Tensor | None = None,
        **kwargs,
    ):
        beta = 1.0 if inplace_add_to_out else 0.0
        return torch.ops.primus_turbo_cpp_extension.hipblaslt_gemm_fp8(
            a,
            a_scale_inv,
            b,
            b_scale_inv,
            out_dtype,
            trans_a,
            trans_b,
            trans_c,
            granularity.name,
            beta,
            out,
        )


class GEMMFP8CKBackend(KernelBackend):
    SUPPORTED_GRANULARITIES = {
        ScalingGranularity.TENSORWISE,
        ScalingGranularity.ROWWISE,
        ScalingGranularity.BLOCKWISE,
    }

    SUPPORTED_DTYPES = set(_COMMON_SUPPORTED_DTYPES + _HYBRID_SUPPORTED_DTYPES)

    @staticmethod
    def can_handle(
        a: torch.Tensor,
        a_scale_inv: torch.Tensor,
        trans_a: bool,
        b: torch.Tensor,
        b_scale_inv: torch.Tensor,
        trans_b: bool,
        out_dtype: torch.dtype,
        trans_c: bool,
        granularity: ScalingGranularity,
        inplace_add_to_out: bool = False,
        **kwargs,
    ) -> bool:
        supported = True
        # This backend has no beta=1 accumulate epilogue.
        supported &= not inplace_add_to_out
        # check the CK backend was compiled into this build
        supported &= build_ck()
        supported &= not is_gfx1250()
        # check ScalingGranularity
        supported &= granularity in GEMMFP8CKBackend.SUPPORTED_GRANULARITIES
        # check dtype
        supported &= (a.dtype, b.dtype, out_dtype) in GEMMFP8CKBackend.SUPPORTED_DTYPES

        if trans_c:
            lhs, rhs = b, a
            trans_lhs, trans_rhs = (not trans_b), (not trans_a)
        else:
            lhs, rhs = a, b
            trans_lhs, trans_rhs = trans_a, trans_b

        k = lhs.shape[0] if trans_lhs else lhs.shape[1]
        n = rhs.shape[0] if trans_rhs else rhs.shape[1]

        # NT / NN layout (transA == False): the contraction dim k must be a
        # multiple of 32.
        if not trans_lhs:
            supported &= k % 32 == 0
            # BLOCKWISE additionally requires k, n multiples of 128 and k >= 128.
            if granularity == ScalingGranularity.BLOCKWISE:
                supported &= (k % 128 == 0) and (n % 128 == 0) and (k >= 128)

        return supported

    @staticmethod
    def execute(
        a: torch.Tensor,
        a_scale_inv: torch.Tensor,
        trans_a: bool,
        b: torch.Tensor,
        b_scale_inv: torch.Tensor,
        trans_b: bool,
        out_dtype: torch.dtype,
        trans_c: bool,
        granularity: ScalingGranularity,
    ):
        if trans_c:
            lhs, rhs = b, a
            lhs_scale_inv, rhs_scale_inv = b_scale_inv, a_scale_inv
            trans_lhs = not trans_b
            trans_rhs = not trans_a
        else:
            lhs, rhs = a, b
            lhs_scale_inv, rhs_scale_inv = a_scale_inv, b_scale_inv
            trans_lhs = trans_a
            trans_rhs = trans_b

        return torch.ops.primus_turbo_cpp_extension.ck_gemm_fp8(
            lhs, rhs, lhs_scale_inv, rhs_scale_inv, trans_lhs, trans_rhs, out_dtype, granularity.name
        )


_MX_ONE_GROUP_OFFS = {}


def _mx_dense_triton(a, a_scale_inv, b, b_scale_inv, out_dtype, beta=0.0, out=None):
    """MX_BLOCKWISE NT GEMM C = A B^T on the Triton variable-K MX kernel with one group spanning the reduction dim
    (beta=1 accumulates into ``out``). K must be a multiple of 128, as the HIP / FlyDSL MX quantization pads it."""
    from primus_turbo.triton.grouped_gemm.grouped_gemm_fp8_kernel import (
        grouped_gemm_mxfp8_variable_k_triton_kernel,
    )

    m, k = a.shape
    n = b.shape[0]
    key = (k, a.device)
    if key not in _MX_ONE_GROUP_OFFS:
        _MX_ONE_GROUP_OFFS[key] = torch.tensor([0, k], dtype=torch.int64, device=a.device)
    c = grouped_gemm_mxfp8_variable_k_triton_kernel(
        a,
        a_scale_inv,
        b,
        b_scale_inv,
        _MX_ONE_GROUP_OFFS[key],
        m,
        n,
        1,
        out_dtype=out_dtype,
        beta=beta,
        out=None if out is None else out.view(1, m, n),
    )
    return c[0]


@functools.lru_cache(maxsize=None)
def mx1250_gemm():
    """(``gemm_mxfp8_gfx1250``, whether it has the beta=1 epilogue: its ``accumulate`` argument, the K line's
    fp32-out / beta=1 step). gfx1250's MX GEMM, imported on first use."""
    from primus_turbo.flydsl.gemm.gemm_mxfp8_kernel_gfx1250 import gemm_mxfp8_gfx1250

    return gemm_mxfp8_gfx1250, "accumulate" in inspect.signature(gemm_mxfp8_gfx1250).parameters


def _mx1250_rows_ok(t: torch.Tensor) -> bool:
    """An operand of gfx1250's MX GEMMs: E4M3, K contiguous, 16-byte aligned rows."""
    ok = t.dtype == torch.float8_e4m3fn and t.stride(-1) == 1
    return ok and t.stride(-2) % 16 == 0 and t.data_ptr() % 16 == 0


def _mx1250_scale_ok(s: torch.Tensor, rows: int, k: int, batch: tuple = ()) -> bool:
    """E8M0 scales of a ``rows`` x ``k`` operand as gfx1250's MX GEMMs take them: raw ``[*batch, rows, ceil(k/32)]``
    (k/32 contiguous) or packed ``[*batch, ceil(rows/32), ceil(k/128), 32, 4]`` (contiguous)."""
    if s.element_size() != 1:
        return False
    if s.dim() == len(batch) + 4:
        return s.shape == (*batch, -(-rows // 32), -(-k // 128), 32, 4) and s.is_contiguous()
    return s.shape == (*batch, rows, -(-k // 32)) and s.stride(-1) == 1 and (not batch or s.is_contiguous())


def mx1250_operands_ok(a, sa, b, sb, k: int, batch: tuple = ()) -> bool:
    """``a [M, k]`` and ``b [*batch, N, k]`` with their scales, both raw or both packed, for gfx1250's MX GEMMs."""
    ok = a.shape[-1] == b.shape[-1] == k and k % 16 == 0 and _mx1250_rows_ok(a) and _mx1250_rows_ok(b)
    ok = ok and sa.dim() + len(batch) == sb.dim() and _mx1250_scale_ok(sa, a.shape[-2], k)
    return ok and _mx1250_scale_ok(sb, b.shape[-2], k, batch)


def mx1250_out_ok(out, shape, accumulate: bool, out_dtype) -> bool:
    """The output of gfx1250's MX GEMMs: bf16 (N % 8 == 0); beta=1 (``accumulate``) into a bf16 / fp32 ``out`` of
    ``shape`` with 16-byte aligned rows."""
    if shape[-1] % 8:
        return False
    if not accumulate:
        return out_dtype == torch.bfloat16
    ok = out is not None and out.dtype in (torch.bfloat16, torch.float32) and out.shape == shape
    return ok and out.is_contiguous() and out.data_ptr() % 16 == 0


def _mx1250_can_handle(
    a, a_scale_inv, trans_a, b, b_scale_inv, trans_b, out_dtype, trans_c, accumulate, out, bsh
):
    """``gemm_mxfp8_gfx1250``: NT, ``b`` row-major or (``bsh``) in dense 16x16 blocks (N % 16 == 0); beta=1 once
    the kernel has that epilogue."""
    if trans_a or not trans_b or trans_c or a.dim() != 2 or b.dim() != 2:
        return False
    (m, k), n = a.shape, b.shape[0]
    ok = mx1250_operands_ok(a, a_scale_inv, b, b_scale_inv, k) and (
        not bsh or (n % 16 == 0 and b.stride(0) == k)
    )
    return ok and (not accumulate or mx1250_gemm()[1]) and mx1250_out_ok(out, (m, n), accumulate, out_dtype)


class GEMMFP8TritonBackend(KernelBackend):
    """Triton persistent-kernel backend for FP8 GEMM.

    Supports:
      - TENSORWISE: per-tensor scaling (all layouts), including HYBRID format
      - ROWWISE: per-row/per-col vector scaling (all layouts)
      - BLOCKWISE: block-wise scaling with three layouts:
          NT/RCR (forward), NN/RRR (grad_X), TN/CRR (grad_W)
      - MX_BLOCKWISE (gfx1250 only): NT, raw E8M0 [rows, K/32] scales, K % 128 == 0, beta=1 into a bf16 / fp16 /
        fp32 ``out``: the grouped variable-K MX kernel with one group (``_mx_dense_triton``), the fallback of the
        FlyDSL gfx1250 MX GEMM (hipBLASLt's MX GEMM is wrong there)
    """

    SUPPORTED_GRANULARITIES = {
        ScalingGranularity.TENSORWISE,
        ScalingGranularity.ROWWISE,
        ScalingGranularity.BLOCKWISE,
        ScalingGranularity.MX_BLOCKWISE,
    }

    SUPPORTED_DTYPES = set(_COMMON_SUPPORTED_DTYPES + _HYBRID_SUPPORTED_DTYPES)

    @staticmethod
    def can_handle(
        a: torch.Tensor,
        a_scale_inv: torch.Tensor,
        trans_a: bool,
        b: torch.Tensor,
        b_scale_inv: torch.Tensor,
        trans_b: bool,
        out_dtype: torch.dtype,
        trans_c: bool,
        granularity: ScalingGranularity,
        inplace_add_to_out: bool = False,
        out: torch.Tensor | None = None,
        **kwargs,
    ) -> bool:
        if granularity == ScalingGranularity.MX_BLOCKWISE:
            m, k = a.shape
            # gfx1250 only: other archs keep their MX backends (and auto-tune candidates) as they are
            supported = is_gfx1250() and (not trans_a) and trans_b and not trans_c
            supported &= a.dim() == 2 and b.dim() == 2 and b.shape[1] == k and k % 128 == 0
            # raw [rows, K/32] scales and a row-major b only
            supported &= a_scale_inv.dim() == 2 and b_scale_inv.dim() == 2 and not kwargs.get("b_preshuffled")
            supported &= a.dtype in (float8_e4m3, float8_e5m2) and b.dtype in (float8_e4m3, float8_e5m2)
            supported &= out_dtype in (torch.bfloat16, torch.float16)
            if inplace_add_to_out:
                supported &= out is not None and out.is_contiguous() and tuple(out.shape) == (m, b.shape[0])
                supported &= out is not None and out.dtype in (torch.bfloat16, torch.float16, torch.float32)
            return supported
        supported = True
        supported &= granularity in GEMMFP8TritonBackend.SUPPORTED_GRANULARITIES
        supported &= (a.dtype, b.dtype, out_dtype) in GEMMFP8TritonBackend.SUPPORTED_DTYPES
        # Only the TENSORWISE kernel implements the beta=1 epilogue; the ROWWISE /
        # BLOCKWISE ones would ignore `out` and silently produce no gradient.
        if inplace_add_to_out:
            supported &= granularity == ScalingGranularity.TENSORWISE
        return supported

    @staticmethod
    def execute(
        a: torch.Tensor,
        a_scale_inv: torch.Tensor,
        trans_a: bool,
        b: torch.Tensor,
        b_scale_inv: torch.Tensor,
        trans_b: bool,
        out_dtype: torch.dtype,
        trans_c: bool,
        granularity: ScalingGranularity,
        inplace_add_to_out: bool = False,
        out: torch.Tensor | None = None,
        **kwargs,
    ):
        beta = 1.0 if inplace_add_to_out else 0.0

        if granularity == ScalingGranularity.MX_BLOCKWISE:
            return _mx_dense_triton(a, a_scale_inv, b, b_scale_inv, out_dtype, beta=beta, out=out)
        if granularity == ScalingGranularity.TENSORWISE:
            return gemm_fp8_tensorwise_triton_kernel(
                a,
                a_scale_inv,
                b,
                b_scale_inv,
                trans_a=trans_a,
                trans_b=trans_b,
                out_dtype=out_dtype,
                trans_c=trans_c,
                beta=beta,
                out=out,
            )

        assert not inplace_add_to_out, (
            f"Fused accumulation into `out` is only implemented for TENSORWISE, got {granularity}"
        )
        if granularity == ScalingGranularity.ROWWISE:
            return gemm_fp8_rowwise_triton_kernel(
                a,
                a_scale_inv,
                b,
                b_scale_inv,
                trans_a=trans_a,
                trans_b=trans_b,
                out_dtype=out_dtype,
                trans_c=trans_c,
            )
        elif granularity == ScalingGranularity.BLOCKWISE:
            return gemm_fp8_blockwise_triton_kernel(
                a,
                a_scale_inv,
                b,
                b_scale_inv,
                trans_a=trans_a,
                trans_b=trans_b,
                out_dtype=out_dtype,
                trans_c=trans_c,
            )
        else:
            raise ValueError(f"Unsupported granularity for FP8 Triton: {granularity}")


class GEMMFP8TurboBackend(KernelBackend):
    """Hand-tuned MXFP8 GEMM kernel for GFX950 (MI350/MI355).

    Supports MX_BLOCKWISE only. NT layout. Tile 256x256x128.
    Shape constraints: m,n % 16 == 0, k % 128 == 0, k >= 384.
    """

    SUPPORTED_GRANULARITIES = {
        ScalingGranularity.MX_BLOCKWISE,
    }

    SUPPORTED_DTYPES = set(_COMMON_SUPPORTED_DTYPES + _HYBRID_SUPPORTED_DTYPES)

    @staticmethod
    def can_handle(
        a: torch.Tensor,
        a_scale_inv: torch.Tensor,
        trans_a: bool,
        b: torch.Tensor,
        b_scale_inv: torch.Tensor,
        trans_b: bool,
        out_dtype: torch.dtype,
        trans_c: bool,
        granularity: ScalingGranularity,
        inplace_add_to_out: bool = False,
        **kwargs,
    ) -> bool:
        supported = True
        # This backend has no beta=1 accumulate epilogue.
        supported &= not inplace_add_to_out
        supported &= is_gfx950()
        supported &= granularity in GEMMFP8TurboBackend.SUPPORTED_GRANULARITIES
        supported &= (a.dtype, b.dtype, out_dtype) in GEMMFP8TurboBackend.SUPPORTED_DTYPES
        supported &= not trans_a and trans_b and not trans_c
        m, n, k = get_gemm_logical_shape(a, b, trans_a, trans_b)
        supported &= m % 16 == 0 and n % 16 == 0 and k % 128 == 0 and k >= 384
        return supported

    @staticmethod
    def execute(
        a: torch.Tensor,
        a_scale_inv: torch.Tensor,
        trans_a: bool,
        b: torch.Tensor,
        b_scale_inv: torch.Tensor,
        trans_b: bool,
        out_dtype: torch.dtype,
        trans_c: bool,
        granularity: ScalingGranularity,
    ):
        return torch.ops.primus_turbo_cpp_extension.turbo_gemm_fp8(
            a, a_scale_inv, b, b_scale_inv, out_dtype, trans_a, trans_b, trans_c, granularity.name
        )


class GEMMFP8FlyDSLBackend(KernelBackend):
    """FlyDSL 8-wave fp8 dense GEMM backend (gfx950), and gfx1250's MX_BLOCKWISE GEMM.

    gfx1250 MX_BLOCKWISE: ``gemm_mxfp8_gfx1250`` (``_mx1250_can_handle``), the default MX backend there.

    TENSORWISE: scalar a_scale/b_scale, bf16/fp16 out, arbitrary M/N/K, layouts
    NT/NN/TN (TT unsupported). trans_c via post-hoc transpose.

    MX_BLOCKWISE: NT only, per-operand E4M3/E5M2 (incl. hybrid), bf16/fp16 out,
    per-1x32 raw E8M0 2D scales [M,K//32]/[N,K//32]. The kernel repacks the scales
    to its preshuffled layout, so execute() does no host-side padding.
    """

    SUPPORTED_GRANULARITIES = {ScalingGranularity.TENSORWISE, ScalingGranularity.MX_BLOCKWISE}
    SUPPORTED_DTYPES = set(_COMMON_SUPPORTED_DTYPES + _HYBRID_SUPPORTED_DTYPES)

    @staticmethod
    def can_handle(
        a: torch.Tensor,
        a_scale_inv: torch.Tensor,
        trans_a: bool,
        b: torch.Tensor,
        b_scale_inv: torch.Tensor,
        trans_b: bool,
        out_dtype: torch.dtype,
        trans_c: bool,
        granularity: ScalingGranularity,
        inplace_add_to_out: bool = False,
        out: torch.Tensor | None = None,
        **kwargs,
    ) -> bool:
        if granularity == ScalingGranularity.MX_BLOCKWISE and is_gfx1250():
            return _mx1250_can_handle(
                a, a_scale_inv, trans_a, b, b_scale_inv, trans_b, out_dtype, trans_c, inplace_add_to_out, out,
                kwargs.get("b_preshuffled", False),
            )  # fmt: skip
        supported = True
        if inplace_add_to_out:
            supported &= out is not None and out.dtype == out_dtype
            supported &= out_dtype in (torch.bfloat16, torch.float16)
        # gfx950 (CDNA4) only: kernel uses mfma_f32_16x16x128_f8f6f4, absent on gfx942-.
        supported &= is_gfx950()
        supported &= granularity in GEMMFP8FlyDSLBackend.SUPPORTED_GRANULARITIES
        supported &= (a.dtype, b.dtype, out_dtype) in GEMMFP8FlyDSLBackend.SUPPORTED_DTYPES
        m, n, k = get_gemm_logical_shape(a, b, trans_a, trans_b)

        if granularity == ScalingGranularity.MX_BLOCKWISE:
            # NT only; per-operand E4M3/E5M2; raw E8M0 2D scales [M,K//32]/[N,K//32].
            supported &= (not trans_a) and trans_b
            supported &= k % 128 == 0 and k >= 256
            supported &= a_scale_inv.shape == (m, k // 32) and b_scale_inv.shape == (n, k // 32)
            supported &= a_scale_inv.element_size() == 1 and b_scale_inv.element_size() == 1
            supported &= not (inplace_add_to_out and trans_c)
            return supported

        # TENSORWISE: NT/NN/TN native (TT unsupported), scalar per-tensor scales.
        supported &= not (trans_a and trans_b)
        supported &= k > 128  # software pipeline needs >= 2 K tiles: ceil(K/128) >= 2
        supported &= a_scale_inv.numel() == 1 and b_scale_inv.numel() == 1
        return supported

    @staticmethod
    def execute(
        a: torch.Tensor,
        a_scale_inv: torch.Tensor,
        trans_a: bool,
        b: torch.Tensor,
        b_scale_inv: torch.Tensor,
        trans_b: bool,
        out_dtype: torch.dtype,
        trans_c: bool,
        granularity: ScalingGranularity,
        inplace_add_to_out: bool = False,
        out: torch.Tensor | None = None,
        **kwargs,
    ):
        if granularity == ScalingGranularity.MX_BLOCKWISE and is_gfx1250():
            bsh = kwargs.get("b_preshuffled", False)
            if inplace_add_to_out:
                return mx1250_gemm()[0](
                    a, a_scale_inv, b, b_scale_inv, out.dtype, out=out, accumulate=True, b_preshuffled=bsh
                )
            return mx1250_gemm()[0](a, a_scale_inv, b, b_scale_inv, b_preshuffled=bsh)
        if granularity == ScalingGranularity.MX_BLOCKWISE:
            res = gemm_mxfp8_flydsl_kernel(
                a,
                a_scale_inv.view(torch.uint8),
                b,
                b_scale_inv.view(torch.uint8),
                out_dtype=out_dtype,
                beta=1.0 if inplace_add_to_out else 0.0,
                out=out if inplace_add_to_out else None,
            )
            return res.t().contiguous() if trans_c else res

        if trans_c:
            lhs, rhs = b, a
            lhs_scale_inv, rhs_scale_inv = b_scale_inv, a_scale_inv
            trans_lhs, trans_rhs = (not trans_b), (not trans_a)
        else:
            lhs, rhs = a, b
            lhs_scale_inv, rhs_scale_inv = a_scale_inv, b_scale_inv
            trans_lhs, trans_rhs = trans_a, trans_b
        return gemm_fp8_tensorwise_flydsl_kernel(
            lhs,
            lhs_scale_inv,
            rhs,
            rhs_scale_inv,
            trans_a=trans_lhs,
            trans_b=trans_rhs,
            out_dtype=out_dtype,
            trans_c=False,
            beta=1.0 if inplace_add_to_out else 0.0,
            out=out if inplace_add_to_out else None,
        )


_GEMM_FP8_BACKENDS = {
    BackendType.TURBO: BackendEntry(GEMMFP8TurboBackend),
    BackendType.HIPBLASLT: BackendEntry(GEMMFP8HipBLASLtBackend),
    BackendType.CK: BackendEntry(GEMMFP8CKBackend),
    BackendType.TRITON: BackendEntry(GEMMFP8TritonBackend),
    BackendType.FLYDSL: BackendEntry(GEMMFP8FlyDSLBackend),
}


class GEMMFP8KernelDispatcher(AutoKernelDispatcher):
    _backends = _GEMM_FP8_BACKENDS
    _cache = TuneCache(1024)

    @classmethod
    def make_key(cls, a, b, trans_a, trans_b, trans_c, out_dtype, granularity, **kwargs):
        m, n, k = get_gemm_logical_shape(a, b, trans_a, trans_b)
        return (m, n, k, a.dtype, b.dtype, out_dtype, trans_a, trans_b, trans_c, granularity)


@_torch_custom_op_wrapper("primus_turbo::gemm_fp8_impl", mutates_args=(), device_types="cuda")
def gemm_fp8_impl(
    a: torch.Tensor,
    a_scale_inv: torch.Tensor,
    trans_a: bool,
    b: torch.Tensor,
    b_scale_inv: torch.Tensor,
    trans_b: bool,
    out_dtype: torch.dtype,
    trans_c: bool,
    granularity: int,
    default_backend: int,
    b_preshuffled: bool = False,
) -> torch.Tensor:
    """``b_preshuffled``: ``b`` in 16x16 blocks, gfx1250's MX GEMM layout (only its FlyDSL backend takes it)."""
    default_backend_choice = BackendChoice(backend=BackendType(default_backend))
    user_backend_choice = GlobalBackendManager.get_gemm_backend(PrecisionType.FP8)
    granularity_enum = ScalingGranularity(granularity)

    kwargs = dict(
        a=a,
        b=b,
        a_scale_inv=a_scale_inv,
        b_scale_inv=b_scale_inv,
        out_dtype=out_dtype,
        trans_a=trans_a,
        trans_b=trans_b,
        trans_c=trans_c,
        granularity=granularity_enum,
    )
    if b_preshuffled:
        kwargs["b_preshuffled"] = True

    return GEMMFP8KernelDispatcher.dispatch(default_backend_choice, user_backend_choice, **kwargs)


@_torch_custom_op_wrapper("primus_turbo::gemm_fp8_accum_impl", mutates_args={"out"}, device_types="cuda")
def gemm_fp8_accum_impl(
    a: torch.Tensor,
    a_scale_inv: torch.Tensor,
    trans_a: bool,
    b: torch.Tensor,
    b_scale_inv: torch.Tensor,
    trans_b: bool,
    out_dtype: torch.dtype,
    trans_c: bool,
    granularity: int,
    out: torch.Tensor,
    default_backend: int,
    b_preshuffled: bool = False,
) -> None:
    """Dense FP8 GEMM that accumulates into ``out`` instead of returning.

    Computes ``out += op(A) @ op(B)``, folding the accumulation into the GEMM
    epilogue (beta=1)
    """
    default_backend_choice = BackendChoice(backend=BackendType(default_backend))
    user_backend_choice = GlobalBackendManager.get_gemm_backend(PrecisionType.FP8)
    granularity_enum = ScalingGranularity(granularity)

    kwargs = dict(
        a=a,
        b=b,
        a_scale_inv=a_scale_inv,
        b_scale_inv=b_scale_inv,
        out_dtype=out_dtype,
        trans_a=trans_a,
        trans_b=trans_b,
        trans_c=trans_c,
        granularity=granularity_enum,
        inplace_add_to_out=True,
        out=out,
    )
    if b_preshuffled:
        kwargs["b_preshuffled"] = True

    # The tuner benchmarks a backend by launching it repeatedly, so letting it tune on
    # the caller's buffer would accumulate the wgrad once per warmup and timing
    # iteration.
    if GlobalBackendManager.auto_tune_enabled() and not GEMMFP8KernelDispatcher._is_graph_capturing():
        GEMMFP8KernelDispatcher.tune(**{**kwargs, "out": torch.zeros_like(out)})

    GEMMFP8KernelDispatcher.dispatch(default_backend_choice, user_backend_choice, **kwargs)


@gemm_fp8_impl.register_fake
def gemm_fp8_impl_meta(
    a: torch.Tensor,
    a_scale_inv: torch.Tensor,
    trans_a: bool,
    b: torch.Tensor,
    b_scale_inv: torch.Tensor,
    trans_b: bool,
    out_dtype: torch.dtype,
    trans_c: bool,
    granularity: int,
    default_backend: int,
    b_preshuffled: bool = False,
) -> torch.Tensor:
    m, n, _ = get_gemm_logical_shape(a, b, trans_a, trans_b)
    if trans_c:
        m, n = n, m
    return torch.empty(m, n, dtype=out_dtype, device=a.device)


@gemm_fp8_accum_impl.register_fake
def gemm_fp8_accum_impl_meta(
    a: torch.Tensor,
    a_scale_inv: torch.Tensor,
    trans_a: bool,
    b: torch.Tensor,
    b_scale_inv: torch.Tensor,
    trans_b: bool,
    out_dtype: torch.dtype,
    trans_c: bool,
    granularity: int,
    out: torch.Tensor,
    default_backend: int,
    b_preshuffled: bool = False,
) -> None:
    m, n, _ = get_gemm_logical_shape(a, b, trans_a, trans_b)
    if trans_c:
        m, n = n, m
    assert tuple(out.shape) == (m, n), f"out shape {tuple(out.shape)} must equal {(m, n)}"
    return None
