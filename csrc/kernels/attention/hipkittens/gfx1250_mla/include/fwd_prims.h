/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// fwd_prims.h -- the gfx1250 builtins of k_mla_fwd (wave32), namespace f1. Each wrapper lowers to
// one instruction, or to the fence / barrier sequence named. Self-contained: the backward kernels
// have their own (dkdv_prims.h, dq_prims.h).
#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <utility>

namespace f1 {

typedef __bf16                                   bf16;
typedef __bf16                                   bf16x8 __attribute__((ext_vector_type(8)));
typedef __bf16                                   bf16x16 __attribute__((ext_vector_type(16)));
typedef float                                    f32x2 __attribute__((ext_vector_type(2)));
typedef float                                    f32x8 __attribute__((ext_vector_type(8)));
typedef uint32_t                                 u32x4 __attribute__((ext_vector_type(4)));
typedef int32_t                                  i32x4 __attribute__((ext_vector_type(4)));
typedef int32_t                                  i32x8 __attribute__((ext_vector_type(8)));
typedef int                                      v4i __attribute__((vector_size(16)));
typedef __attribute__((address_space(3))) bf16x8 lds_bf16x8;
typedef __attribute__((address_space(3))) unsigned char lds_u8;
typedef __attribute__((address_space(3))) u32x4         lds_u32x4;

#define F1_INLINE __device__ __forceinline__

// ---- compile-time loops: f(std::integral_constant<int, I>{}) for I in [0, N)
template <int N, typename F, int... I>
F1_INLINE void static_for_impl(F &&f, std::integer_sequence<int, I...>) {
    (f(std::integral_constant<int, I>{}), ...);
}
template <int N, typename F> F1_INLINE void static_for(F &&f) {
    static_for_impl<N>(static_cast<F &&>(f), std::make_integer_sequence<int, N>{});
}
template <int I> using ic = std::integral_constant<int, I>;

// ---- vectors, WMMA
F1_INLINE bf16x16 cat(bf16x8 a, bf16x8 b) {
    return __builtin_shufflevector(a, b, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
}
// fptrunc f32 -> bf16 (v_cvt_pk_bf16_f32, round to nearest even), FlyDSL's .to(BFloat16)
F1_INLINE bf16x8 to_bf16x8(f32x8 v) {
    return bf16x8{(bf16) v[0], (bf16) v[1], (bf16) v[2], (bf16) v[3],
                  (bf16) v[4], (bf16) v[5], (bf16) v[6], (bf16) v[7]};
}
// WMMA 16x16x32 bf16 -> f32 (v_wmma_f32_16x16x32_bf16: no negation, no C modifier, no reuse hints)
F1_INLINE f32x8 wmma(bf16x16 a, bf16x16 b, f32x8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x32_bf16(false, a, false, b, (short) 0, c, false, false);
}
F1_INLINE f32x8 zero8() {
    return f32x8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
}

// ---- LDS (addresses are byte offsets in the workgroup's LDS)
F1_INLINE uint32_t lds_addr(const void *p) {
    return (uint32_t) (uintptr_t) p;
}
F1_INLINE const lds_u8 *lds_ptr(uint32_t a) {
    return (const lds_u8 *) (uintptr_t) a;
}
// fragment reads: ds_load_b128 and the transposing ds_load_tr16_b128, base + immediate offset
F1_INLINE bf16x8 lds_load_b128(const lds_u8 *p, int off) {
    return *(const lds_bf16x8 *) (p + off);
}
F1_INLINE bf16x8 lds_tr16(const lds_u8 *p, int off) {
    return __builtin_amdgcn_ds_load_tr16_b128_v8bf16((lds_bf16x8 *) (p + off));
}
// output staging: ds_store_b128, base + immediate offset
F1_INLINE void lds_store_b128(uint32_t a, int off, u32x4 v) {
    *(lds_u32x4 *) ((lds_u8 *) (uintptr_t) a + off) = v;
}
// 16 bytes from LDS to global memory, asynchronous (global_store_async_from_lds_b128): retired by
// s_wait_asynccnt
F1_INLINE void store_async_from_lds(unsigned long long gaddr, uint32_t lds) {
    __builtin_amdgcn_global_store_async_from_lds_b128(
        (__attribute__((address_space(1))) v4i *) gaddr,
        (__attribute__((address_space(3))) v4i *) (uintptr_t) lds, 0, 0);
}

// ---- buffer stores: resource word 3 as FlyDSL's make_buffer_tensor on gfx1250
inline constexpr int             RSRC_FLAGS = 159744;
F1_INLINE __amdgpu_buffer_rsrc_t rsrc(const void *base, int nbytes) {
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<void *>(base), (short) 0, (long) nbytes,
                                             RSRC_FLAGS);
}

// ---- Tensor Data Mover: one 2-D tensor_load_to_lds (descriptor groups 0 and 1; groups 2..4 zero)
F1_INLINE void tdm_load(i32x4 g0, i32x8 g1) {
    const i32x4 z4 = {0, 0, 0, 0};
    const i32x8 z8 = {0, 0, 0, 0, 0, 0, 0, 0};
    __builtin_amdgcn_tensor_load_to_lds(g0, g1, z4, z4, z8, 0);
}
// s_wait_tensorcnt N: at most N of this wave's TDM ops in flight
template <int N> F1_INLINE void tensor_wait() {
    __builtin_amdgcn_s_wait_tensorcnt((short) N);
}

// ---- scheduling, barriers
template <int MASK> F1_INLINE void sched_barrier() {
    __builtin_amdgcn_sched_barrier(MASK);
}
inline constexpr int MO_ACQUIRE = 2, MO_RELEASE = 3;
// llvm.amdgcn.s.wait.dscnt through an asm label: there is no builtin for it, and the legacy
// __builtin_amdgcn_s_waitcnt cannot be selected on gfx1250
extern "C" __device__ void f1_llvm_s_wait_dscnt(short) __asm("llvm.amdgcn.s.wait.dscnt");
F1_INLINE void             wait_ds0() {
    f1_llvm_s_wait_dscnt(0);
}
// retire this wave's LDS reads and TDM copies, release (workgroup, LDS), arrive at the barrier
// (no wait)
F1_INLINE void drain_and_arrive() {
    sched_barrier<0>();
    f1_llvm_s_wait_dscnt(0);
    tensor_wait<0>();
    __builtin_amdgcn_fence(MO_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier_signal(-1);
    sched_barrier<0>();
}
// wait at the workgroup barrier, acquire (workgroup, LDS)
F1_INLINE void bar_wait() {
    sched_barrier<0>();
    __builtin_amdgcn_s_barrier_wait(-1);
    __builtin_amdgcn_fence(MO_ACQUIRE, "workgroup", "local");
    sched_barrier<0>();
}

// ---- wave state, lane exchange, transcendentals
// wave index in the workgroup as a uniform SGPR
F1_INLINE int wave_id(int tid) {
    return __builtin_amdgcn_readfirstlane(tid) >> 5;
}
// an SGPR 0 LLVM cannot see through, hoist or fold (volatile asm), uniform for its divergence
// analysis (readfirstlane: an asm result is divergent)
F1_INLINE int opaque0() {
    int z;
    asm volatile("s_mov_b32 %0, 0" : "=s"(z));
    return __builtin_amdgcn_readfirstlane(z);
}
// exp2 as a bare v_exp_f32 (without llvm.exp2.f32's denormal-range scaling)
F1_INLINE float exp2_raw(float x) {
    return __builtin_amdgcn_exp2f(x);
}
// the value of lane l ^ 16 (v_permlanex16_b32 with FlyDSL's selects)
F1_INLINE float peer(float x) {
    const unsigned u = __builtin_bit_cast(unsigned, x);
    const unsigned v = __builtin_amdgcn_permlanex16(u, u, 0x76543210u, 0xFEDCBA98u, false, false);
    return __builtin_bit_cast(float, v);
}

} // namespace f1
