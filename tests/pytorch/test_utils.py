###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import contextlib
import ctypes

import pytest
import torch

from primus_turbo.pytorch.core.low_precision import is_fp8_dtype
from primus_turbo.pytorch.kernels.attention.attention_impl import (
    FlashAttnDenseDispatcher,
    FlashAttnVarlenDispatcher,
    resolve_flash_attn_backend,
)


def is_ROCM():
    return torch.cuda.is_available() and torch.version.hip


# TODO: Need to check again whether these values are reasonable.
def get_tolerances(dtype):
    if dtype == torch.float32:
        return dict(rtol=1e-4, atol=1e-4)
    elif dtype == torch.float16:
        return dict(rtol=1e-2, atol=1e-2)
    elif dtype == torch.bfloat16:
        return dict(rtol=1e-2, atol=1e-2)
    elif is_fp8_dtype(dtype):
        return dict(rtol=1e-1, atol=1e-1)
    elif dtype == torch.float4_e2m1fn_x2:
        return dict(rtol=0.5, atol=0.5)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


###################################################################


# Relative Error
# Note: x is ref
def relative_error(x: torch.Tensor, y: torch.Tensor):
    x, y = x.float(), y.float()
    return (torch.norm(x - y) / torch.norm(x)).detach().item()


# MSE Error
def mean_squared_error(x: torch.Tensor, y: torch.Tensor):
    x, y = x.float(), y.float()
    return torch.mean((x - y) ** 2).item()


# Max Abs Error
def max_abs_error(x: torch.Tensor, y: torch.Tensor):
    x, y = x.float(), y.float()
    return torch.max(torch.abs(x - y)).item()


# Cosine Similarity
def cosine_similarity(x: torch.Tensor, y: torch.Tensor):
    x, y = x.flatten().float(), y.flatten().float()
    return torch.nn.functional.cosine_similarity(x, y, dim=0).item()


# Symmetric Similarity
def symmetric_similarity_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


# SNR
# Note: x is ref
def compute_snr(x: torch.Tensor, y: torch.Tensor):
    x, y = x.float(), y.float()
    signal_power = torch.norm(x).pow(2)
    noise_power = torch.norm(x - y).pow(2)
    return 10 * torch.log10(signal_power / (noise_power + 1e-12)).detach().item()


def pinned_backend_takes(backend, varlen=False, **kwargs) -> bool:
    """Whether a pinned backend implements these inputs (True when nothing is pinned).

    A False also asserts the resolver refuses them outright: a backend the caller pinned must
    never be swapped for another one behind their back. So the combos this returns False for
    stay covered -- by that refusal -- instead of dropping out of the run as skips.
    """
    if backend is None:
        return True
    dispatcher = FlashAttnVarlenDispatcher if varlen else FlashAttnDenseDispatcher
    if dispatcher._backends[backend].impl.can_handle(**kwargs):
        return True
    with pytest.raises(ValueError, match="cannot handle the given inputs"):
        resolve_flash_attn_backend(varlen=varlen, user_backend=backend, **kwargs)
    return False


class _HipMemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _HipMemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleType", ctypes.c_int),
        ("location", _HipMemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", ctypes.c_uint32),
    ]


class _HipMemAccessDesc(ctypes.Structure):
    _fields_ = [("location", _HipMemLocation), ("flags", ctypes.c_int)]


class _CudaArray:
    def __init__(self, ptr, nbytes):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }


@contextlib.contextmanager
def tensor_at_mapping_end(shape, dtype):
    """A CUDA tensor whose last byte is the last mapped byte of its virtual address range: HIP's virtual memory API
    reserves a page more than it maps, so a kernel that reads past the tensor faults instead of reading a neighbour
    (for tests of out-of-bounds addressing). Skips the test where the API is not available. Unmapped on exit, after
    a device synchronize; the tensor must not be used afterwards."""
    torch.zeros(1, device="cuda")  # the runtime is up, so its library is loaded
    if not torch.version.hip:
        pytest.skip("HIP virtual memory API")
    path = next((line.split()[-1] for line in open("/proc/self/maps") if "libamdhip64" in line), None)
    if path is None:
        pytest.skip("libamdhip64 not loaded")
    hip = ctypes.CDLL(path)
    dev = torch.cuda.current_device()
    loc = _HipMemLocation(type=1, id=dev)  # hipMemLocationTypeDevice
    prop = _HipMemAllocationProp(type=1, requestedHandleType=0, location=loc)  # pinned, no export handle
    gran = ctypes.c_size_t()
    if hip.hipMemGetAllocationGranularity(ctypes.byref(gran), ctypes.byref(prop), 0):
        pytest.skip("hipMemGetAllocationGranularity failed")
    nbytes = torch.Size(shape).numel() * torch.empty((), dtype=dtype).element_size()
    page = gran.value
    size = max(1, -(-nbytes // page)) * page
    va, handle = ctypes.c_void_p(), ctypes.c_void_p()
    no_flags = ctypes.c_ulonglong(0)
    if hip.hipMemAddressReserve(
        ctypes.byref(va), ctypes.c_size_t(size + page), ctypes.c_size_t(page), None, no_flags
    ):
        pytest.skip("hipMemAddressReserve failed")
    mapped = False
    try:
        if hip.hipMemCreate(ctypes.byref(handle), ctypes.c_size_t(size), ctypes.byref(prop), no_flags):
            pytest.skip("hipMemCreate failed")
        if hip.hipMemMap(va, ctypes.c_size_t(size), ctypes.c_size_t(0), handle, no_flags):
            pytest.skip("hipMemMap failed")
        mapped = True
        access = _HipMemAccessDesc(location=loc, flags=3)  # read-write
        if hip.hipMemSetAccess(va, ctypes.c_size_t(size), ctypes.byref(access), ctypes.c_size_t(1)):
            pytest.skip("hipMemSetAccess failed")
        raw = torch.as_tensor(_CudaArray(va.value + size - nbytes, nbytes), device="cuda")
        yield raw.view(dtype).view(shape)
    finally:
        torch.cuda.synchronize()  # nothing may still use the range
        if mapped:
            hip.hipMemUnmap(va, ctypes.c_size_t(size))
        if handle.value:
            hip.hipMemRelease(handle)
        hip.hipMemAddressFree(va, ctypes.c_size_t(size + page))
