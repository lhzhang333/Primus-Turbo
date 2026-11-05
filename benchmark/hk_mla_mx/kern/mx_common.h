/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// mx_common.h -- gfx1250 (wave32) helpers of the MXFP8 MLA kernels: types, the MX WMMA, E4M3 conversion, E8M0 scale
// rule, LDS fragment loads, TDM descriptors, barriers. Self-contained (HIP runtime only).
//
// Operand layout facts (benchmark/hk_mla_mx/micro, 2026-11-05): for the fp8 16x16x128 and MX 16x16x128 WMMAs A row =
// lane % 16, B column = lane % 16, C (lane l, reg r) = (row r + 8 (l / 16), column l % 16); A and B pair on (lane half
// h, byte e), and with logical k(h, e) = 16 (e / 8) + 8 h + e % 8 the MX k block j (32 consecutive k) is bytes
// 16 j .. 16 j + 15 of every lane. Scale VGPRs: scaleType 0 reads lanes 0..15, 1 lanes 16..31; byte j of lane l
// scales row / column l % 16, k block j.
//
// "MX-interleaved" storage of a 128-element chunk (the K range of one x128 WMMA): logical k at byte
// 64 h + 8 c + r (c = k / 16, h = (k % 16) / 8, r = k % 8), so half h of a lane loads 64 contiguous bytes. A 64-element
// chunk (the rope part, zero-padded to 128 in the WMMA): byte 32 h + 8 c + r (c < 4).
#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <utility>

namespace mx {

typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef float f32x4 __attribute__((ext_vector_type(4)));
typedef float f32x2 __attribute__((ext_vector_type(2)));
typedef int32_t i32x4 __attribute__((ext_vector_type(4)));
typedef int32_t i32x8 __attribute__((ext_vector_type(8)));
typedef int32_t i32x16 __attribute__((ext_vector_type(16)));
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef __bf16 bf16;
typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));
typedef __attribute__((address_space(3))) unsigned char lds_u8;
typedef __attribute__((address_space(3))) u32x4 lds_u32x4;
typedef __attribute__((address_space(3))) u32x2 lds_u32x2;
typedef __attribute__((address_space(3))) uint32_t lds_u32;

#define MX_INLINE __device__ __forceinline__

template <int N, typename F, int... I>
MX_INLINE void static_for_impl(F &&f, std::integer_sequence<int, I...>) {
    (f(std::integral_constant<int, I>{}), ...);
}
template <int N, typename F> MX_INLINE void static_for(F &&f) {
    static_for_impl<N>(static_cast<F &&>(f), std::make_integer_sequence<int, N>{});
}

constexpr float E4M3_MAX = 448.0f;
constexpr float LOG2E = 1.4426950408889634f;
constexpr float LN2 = 0.69314718055994531f;

// ---- E8M0 scale rule: the smallest power of two 2^e with amax / 2^e <= 448 (no saturation); amax = 0 -> e = -127.
// frexp: amax = m 2^E, m in [0.5, 1): amax / 2^(E - 9) = 512 m in [256, 512), > 448 iff m > 0.875.
MX_INLINE int mx_exp(float amax) {
    if (!(amax > 0.f)) return -127;
    int E;
    const float m = __builtin_frexpf(amax, &E);
    int e = E - 9 + (m > 0.875f ? 1 : 0);
    return e < -127 ? -127 : (e > 127 ? 127 : e);
}
MX_INLINE float exp2i(int e) {  // 2^e for e in [-127, 127] (subnormal at -127)
    return __builtin_ldexpf(1.0f, e);
}

// ---- 8 f32 -> 8 E4M3 (x / scale, round to nearest even), gfx1250's only fp8 converter: the PK8 scale form
MX_INLINE u32x2 cvt8_e4m3(f32x8 v, float scale) {
    return __builtin_amdgcn_cvt_scalef32_pk8_fp8_f32(v, scale);
}

// ---- MX 16x16x128 WMMA, E4M3 x E4M3 -> f32, E8M0 scales; SA / SB: scaleAType / scaleBType (0: lanes 0..15, 1: 16..31)
template <int SA, int SB>
MX_INLINE f32x8 wmma_mx(i32x16 a, i32x16 b, f32x8 c, int sa, int sb) {
    return __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(0, a, 0, b, (short)0, c, SA, 0, sa, SB, 0, sb, false,
                                                            false);
}

// the same with a C modifier: MODC 1 = -C (D = A B - C), 2 = |C|
template <int SA, int SB, int MODC>
MX_INLINE f32x8 wmma_mxc(i32x16 a, i32x16 b, f32x8 c, int sa, int sb) {
    return __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(0, a, 0, b, (short)MODC, c, SA, 0, sa, SB, 0, sb, false,
                                                            false);
}

// ---- mx_exp on the bits: the E8M0 byte (mx_exp + 127) of amax >= 0, and its scale 2^(byte - 127) as a float
// (byte 0: 2^-127, subnormal). ef = IEEE exponent field, f = mantissa: mx_exp = ef - 135 + (f > 0x600000).
MX_INLINE int e8m0_byte(float amax) {
    const int byte = (int)((__builtin_bit_cast(unsigned, amax) + 0x1FFFFFu) >> 23) - 8;
    return byte < 0 ? 0 : byte;
}
MX_INLINE float e8m0_scale(int byte) {
    const unsigned u = (unsigned)byte << 23;
    return __builtin_bit_cast(float, u > 0x400000u ? u : 0x400000u);
}

// ---- LDS
MX_INLINE uint32_t lds_addr(const void *p) { return (uint32_t)(uintptr_t)p; }
MX_INLINE const lds_u8 *lds_ptr(uint32_t a) { return (const lds_u8 *)(uintptr_t)a; }
MX_INLINE u32x4 lds_b128(const lds_u8 *p, int off) { return *(const lds_u32x4 *)(p + off); }
MX_INLINE uint32_t lds_b32(const lds_u8 *p, int off) { return *(const lds_u32 *)(p + off); }

// ---- TDM (tensor_load_to_lds), groups 0 and 1 of a 2-D tile; groups 2..4 zero
MX_INLINE void tdm_load(i32x4 g0, i32x8 g1) {
    const i32x4 z4 = {0, 0, 0, 0};
    const i32x8 z8 = {0, 0, 0, 0, 0, 0, 0, 0};
    __builtin_amdgcn_tensor_load_to_lds(g0, g1, z4, z4, z8, 0);
}
template <int N> MX_INLINE void tensor_wait() { __builtin_amdgcn_s_wait_tensorcnt((short)N); }
// group 1: w0 (element size, LDS row padding), dim 0 unbounded, tdim1 rows in bounds, tile0 columns x tile1 rows, the
// row stride in elements
MX_INLINE i32x8 tdm_g1(unsigned w0, unsigned tdim1, unsigned tile0, unsigned tile1, unsigned long long stride_el) {
    return i32x8{(int)w0, (int)0xFFFF0000u, (int)(((tdim1 & 0xFFFFu) << 16) | 0x7FFFu),
                 (int)(((tdim1 >> 16) & 0xFFFFu) | (tile0 << 16)), (int)tile1, (int)(unsigned)stride_el,
                 (int)((unsigned)(stride_el >> 32) & 0xFFFFu), 0};
}
// group 0: LDS byte address, global address; count 0 = NULL descriptor (moves nothing)
MX_INLINE i32x4 tdm_g0(unsigned lds, unsigned long long ga, int count = 1) {
    return i32x4{count, (int)lds, (int)(unsigned)ga, (int)((unsigned)(ga >> 32) | 0x80000000u)};
}
// word 0 of group 1: data size [17:16] = log2(bytes), pad enable [20], pad interval [24:22] = log2(dwords) - 1,
// pad amount [31:25] = dwords - 1
constexpr unsigned tdm_w0(unsigned elem_log2, unsigned interval_dw, unsigned amount_dw) {
    unsigned il = 0;
    while ((2u << il) < interval_dw) ++il;
    return amount_dw == 0 ? (elem_log2 << 16)
                          : ((elem_log2 << 16) | (1u << 20) | (il << 22) | ((amount_dw - 1u) << 25));
}

// ---- barriers, wave state
inline constexpr int MO_ACQUIRE = 2, MO_RELEASE = 3;
extern "C" __device__ void mx_llvm_s_wait_dscnt(short) __asm("llvm.amdgcn.s.wait.dscnt");
MX_INLINE void bar_signal() {
    __builtin_amdgcn_sched_barrier(0);
    mx_llvm_s_wait_dscnt(0);
    __builtin_amdgcn_fence(MO_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_sched_barrier(0);
}
MX_INLINE void bar_wait() {
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier_wait(-1);
    __builtin_amdgcn_fence(MO_ACQUIRE, "workgroup", "local");
    __builtin_amdgcn_sched_barrier(0);
}
MX_INLINE void lock_simd() { __builtin_amdgcn_s_setreg(154, 1); }
MX_INLINE int wave_id(int tid) { return __builtin_amdgcn_readfirstlane(tid) >> 5; }
MX_INLINE float exp2_raw(float x) { return __builtin_amdgcn_exp2f(x); }
// bit_cast of a 32-bit value to float. Use it for vector elements: clang's __builtin_bit_cast(float, v[i]) on an
// ext_vector_type element ignores the index (always element 0; seen at -O3 as a <0,0,0,0,4,4,4,4> shuffle).
MX_INLINE float as_f(uint32_t x) { return __builtin_bit_cast(float, x); }
MX_INLINE float peer(float x) {  // the value of lane l ^ 16
    const unsigned u = __builtin_bit_cast(unsigned, x);
    const unsigned v = __builtin_amdgcn_permlanex16(u, u, 0x76543210u, 0xFEDCBA98u, false, false);
    return __builtin_bit_cast(float, v);
}

}  // namespace mx
