/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// Waits, workgroup barriers, fences and scheduling controls of the attention kernels. Every wait is
// an LLVM intrinsic (visible to the waitcnt pass and to expert-mode dependency tracking);
// s_wait_dscnt has no clang builtin and is reached through an asm label.
#pragma once
#include "tdm.h"
#include "types.h"

extern "C" __device__ void hka_llvm_s_wait_dscnt(short) __asm("llvm.amdgcn.s.wait.dscnt");

namespace hk_attn {

// s_wait_dscnt N: all but the N newest LDS ops of this wave have completed.
template <int N> HKA_DEV void wait_dscnt() {
    hka_llvm_s_wait_dscnt((short) N);
}

// Instruction classes of sched_barrier / sched_group_barrier (AMDGPU SchedGroupMask).
namespace sg {
inline constexpr int VALU = 0x002, WMMA = 0x008, VMEM = 0x010, DS_READ = 0x100, TRANS = 0x400;
} // namespace sg
// No instruction may be scheduled across it except the classes in MASK.
template <int MASK = 0> HKA_DEV void sched_barrier() {
    __builtin_amdgcn_sched_barrier(MASK);
}
// The next N instructions of class MASK form a scheduling group of pipeline ID.
template <int MASK, int N, int ID = 0> HKA_DEV void sched_group_barrier() {
    __builtin_amdgcn_sched_group_barrier(MASK, N, ID);
}
// IR-level fence (no instruction): memory accesses are not moved across it.
HKA_DEV void compiler_fence() {
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
}
// compiler_fence + sched_barrier: nothing crosses it, neither in IR passes nor in the machine
// schedulers.
HKA_DEV void order_fence() {
    compiler_fence();
    sched_barrier<0>();
}

// Workgroup release fence restricted to LDS: `s_wait_dscnt 0` only, in-flight buffer stores and
// atomics are not drained.
inline constexpr int MO_RELEASE = 3;
HKA_DEV void         lds_release() {
    __builtin_amdgcn_fence(MO_RELEASE, "workgroup", "local");
}

// Split workgroup barrier (barrier id -1).
HKA_DEV void barrier_signal() {
    __builtin_amdgcn_s_barrier_signal(-1);
}
HKA_DEV void barrier_wait() {
    __builtin_amdgcn_s_barrier_wait((short) -1);
}
// LDS protocol: every LDS access of this wave before the signal is complete and visible to every
// wave after its wait. Signal = s_wait_dscnt 0 + s_barrier_signal; global accesses and TDM ops stay
// in flight. Between signal and wait a wave may do any work that does not touch the protected LDS.
HKA_DEV void split_signal_lds() {
    sched_barrier<0>();
    lds_release();
    barrier_signal();
    sched_barrier<0>();
}
// Wait = s_barrier_wait only: LDS loads issued between signal and wait stay in flight.
HKA_DEV void split_wait_lds() {
    sched_barrier<0>();
    barrier_wait();
    compiler_fence();
    sched_barrier<0>();
}
// TDM ring signal: retire all but this wave's TW newest TDM ops, then the LDS signal (the TDM
// writes of a wave become visible to the workgroup through the barrier after its tensor wait).
template <int TW> HKA_DEV void split_signal_tdm_lds() {
    sched_barrier<0>();
    wait_tensorcnt<TW>();
    lds_release();
    barrier_signal();
    sched_barrier<0>();
}
HKA_DEV void lds_barrier() {
    split_signal_lds();
    split_wait_lds();
}

// SCHED_MODE.DISABLE_XDL_ARB_STALL = 1 (s_setreg hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1): this
// wave issues WMMAs back to back without yielding the matrix pipe; kernel lifetime.
HKA_DEV void lock_simd() {
    __builtin_amdgcn_s_setreg(26 | (2 << 6) | (0 << 11), 1u);
}

} // namespace hk_attn
