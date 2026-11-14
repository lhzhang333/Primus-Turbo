###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

"""Small helpers shared by the gfx1250 FlyDSL kernels: the device check and typed LDS loads / stores."""

import functools

import flydsl.expr as fx
import torch


@functools.lru_cache(maxsize=16)
def _gcn_arch(device_index: int) -> str:
    return str(getattr(torch.cuda.get_device_properties(device_index), "gcnArchName", "") or "")


def require_gfx1250(device: torch.device, who: str) -> None:
    """Raise unless ``device`` is a gfx1250 GPU (the kernels use wave32, WMMA and the TDM engine)."""
    if device.type != "cuda":
        raise RuntimeError(f"{who}: needs ROCm device tensors, got device '{device}'")
    index = device.index if device.index is not None else torch.cuda.current_device()
    arch = _gcn_arch(index)
    if "gfx1250" not in arch:
        raise RuntimeError(f"{who}: runs on gfx1250 only, but device {index} reports gcnArchName={arch!r}")


def make_lds_copy_ops(bits: int):
    """(load, store) of ``bits`` (32 / 64 / 128) at an LDS byte address: ``load(base, offset)`` returns the i32
    vector, ``store(base, offset, data)`` writes one."""
    if bits not in (32, 64, 128):
        raise ValueError(f"bits must be 32/64/128, got {bits}")
    layout = fx.make_layout(bits // fx.Int32.width, 1)
    atom = fx.make_copy_atom(fx.UniversalCopy(bits), fx.Int32)
    ptr_ty = fx.PointerType.get(elem_ty=fx.Int32.ir_type, address_space=fx.AddressSpace.Shared, alignment=bits // 8)

    def _view(lds_base_idx, byte_offset):
        addr_i32 = fx.Int32(lds_base_idx) + fx.Int32(byte_offset)
        return fx.Tensor(fx.make_view(fx.inttoptr(ptr_ty, addr_i32), layout))

    def load(lds_base_idx, byte_offset):
        rmem = fx.make_rmem_tensor(layout, fx.Int32)
        fx.copy_atom_call(atom, _view(lds_base_idx, byte_offset), rmem)
        return rmem.load()

    def store(lds_base_idx, byte_offset, data):
        rmem = fx.make_rmem_tensor(layout, fx.Int32)
        rmem.store(data)
        fx.copy_atom_call(atom, rmem, _view(lds_base_idx, byte_offset))

    return load, store
