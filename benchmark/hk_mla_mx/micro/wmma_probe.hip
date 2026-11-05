// wmma_probe.hip -- one WMMA per wave on register images given by the host, to map gfx1250's fp8 / MX operand
// layouts (A, B, the f32 C, the E8M0 scale VGPRs). Case w = workgroup w (one wave32): the lane's A / B registers are
// A[w][lane][0..NI), B[w][lane][0..NI) (int32 words, byte j of word i = byte 4 i + j of the register image), the
// scale VGPRs SA[w][lane], SB[w][lane]; it writes C[w][lane][0..8) (the f32 accumulator, C starts at 0).
#include <hip/hip_runtime.h>

typedef int v8i __attribute__((ext_vector_type(8)));
typedef int v16i __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef __bf16 v16bf __attribute__((ext_vector_type(16)));

template <int NI>
__device__ inline void load_img(const int* __restrict__ p, int (&r)[NI]) {
#pragma unroll
  for (int i = 0; i < NI; ++i) r[i] = p[i];
}
__device__ inline void store_c(float* __restrict__ c, v8f v) {
#pragma unroll
  for (int i = 0; i < 8; ++i) c[i] = v[i];
}

extern "C" __global__ __launch_bounds__(32) void k_probe_bf16_32(const int* A, const int* B, float* C) {
  const int w = blockIdx.x, l = threadIdx.x;
  int a[8], b[8];
  load_img<8>(A + (w * 32 + l) * 8, a);
  load_img<8>(B + (w * 32 + l) * 8, b);
  v8i av = {a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]};
  v8i bv = {b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]};
  v8f c = {};
  c = __builtin_amdgcn_wmma_f32_16x16x32_bf16(false, __builtin_bit_cast(v16bf, av), false,
                                              __builtin_bit_cast(v16bf, bv), (short)0, c, false, false);
  store_c(C + (w * 32 + l) * 8, c);
}

extern "C" __global__ __launch_bounds__(32) void k_probe_fp8_64(const int* A, const int* B, float* C) {
  const int w = blockIdx.x, l = threadIdx.x;
  int a[8], b[8];
  load_img<8>(A + (w * 32 + l) * 8, a);
  load_img<8>(B + (w * 32 + l) * 8, b);
  v8i av = {a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]};
  v8i bv = {b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]};
  v8f c = {};
  c = __builtin_amdgcn_wmma_f32_16x16x64_fp8_fp8(av, bv, (short)0, c, false, false);
  store_c(C + (w * 32 + l) * 8, c);
}

extern "C" __global__ __launch_bounds__(32) void k_probe_fp8_128(const int* A, const int* B, float* C) {
  const int w = blockIdx.x, l = threadIdx.x;
  int a[16], b[16];
  load_img<16>(A + (w * 32 + l) * 16, a);
  load_img<16>(B + (w * 32 + l) * 16, b);
  v16i av, bv;
#pragma unroll
  for (int i = 0; i < 16; ++i) av[i] = a[i], bv[i] = b[i];
  v8f c = {};
  c = __builtin_amdgcn_wmma_f32_16x16x128_fp8_fp8(av, bv, (short)0, c, false, false);
  store_c(C + (w * 32 + l) * 8, c);
}

// MX-scaled 16x16x128, e4m3 x e4m3, E8M0 scales; TA / TB = the scaleAType / scaleBType immediates (opsel)
#define MX_PROBE(TA, TB)                                                                                         \
  extern "C" __global__ __launch_bounds__(32) void k_probe_mx_##TA##TB(const int* A, const int* B, const int* SA, \
                                                                         const int* SB, float* C) {              \
    const int w = blockIdx.x, l = threadIdx.x;                                                                   \
    int a[16], b[16];                                                                                            \
    load_img<16>(A + (w * 32 + l) * 16, a);                                                                      \
    load_img<16>(B + (w * 32 + l) * 16, b);                                                                      \
    v16i av, bv;                                                                                                 \
    _Pragma("unroll") for (int i = 0; i < 16; ++i) av[i] = a[i], bv[i] = b[i];                                   \
    const int sa = SA[w * 32 + l], sb = SB[w * 32 + l];                                                          \
    v8f c = {};                                                                                                  \
    c = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(0, av, 0, bv, (short)0, c, TA, 0, sa, TB, 0, sb, false, \
                                                         false);                                                 \
    store_c(C + (w * 32 + l) * 8, c);                                                                            \
  }
MX_PROBE(0, 0)
MX_PROBE(0, 1)
MX_PROBE(1, 0)
MX_PROBE(1, 1)
