// tdmst: does tensor_store_from_lds write a 16 x 128 bf16 tile from LDS (272-B rows) to global rows r0.. with a row
// stride of STRIDE elements, clipped to tdim1 rows, and nothing else? One workgroup, one wave.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>
typedef int i32x4 __attribute__((ext_vector_type(4)));
typedef int i32x8 __attribute__((ext_vector_type(8)));
constexpr unsigned W0_V = 0x07510000u;  // k_mla_fwd.hip: 128 columns, pad 8 elements -> 272-B rows
__device__ inline i32x8 tdm_g1(unsigned w0, unsigned tdim1, unsigned tile0, unsigned tile1, unsigned long long stride_el) {
  return i32x8{(int)w0, (int)0xFFFF0000u, (int)(((tdim1 & 0xFFFFu) << 16) | 0x7FFFu),
               (int)(((tdim1 >> 16) & 0xFFFFu) | (tile0 << 16)), (int)tile1, (int)(unsigned)stride_el,
               (int)((unsigned)(stride_el >> 32) & 0xFFFFu), 0};
}
__device__ inline i32x4 tdm_g0(unsigned lds, unsigned long long ga, int count = 1) {
  return i32x4{count, (int)lds, (int)(unsigned)ga, (int)((unsigned)(ga >> 32) | 0x80000000u)};
}
extern "C" __global__ __attribute__((amdgpu_flat_work_group_size(32, 32))) void k(unsigned short* out, int r0, int stride,
                                                                                 int tdim1) {
  __shared__ __attribute__((aligned(1024))) unsigned short lds[16 * 136];
  const int lane = threadIdx.x;
  for (int r = 0; r < 16; ++r)
    for (int c = lane; c < 136; c += 32) lds[r * 136 + c] = (unsigned short)(c < 128 ? (r * 1000 + c + 1) : 0xDEAD);
  __syncthreads();
  const unsigned la = (unsigned)(uintptr_t)lds;
  const unsigned long long ga = (unsigned long long)(uintptr_t)(out + (size_t)r0 * stride);
  const i32x4 g0 = tdm_g0(la, ga, 1);
  // the padded LDS row as a 136-column tile (272 contiguous bytes), dim 0 bounded at 128 columns: the 8 pad columns
  // are out of bounds and dropped; no LDS padding (store direction reads LDS linearly)
  i32x8 g1 = tdm_g1(0x00010000u, (unsigned)tdim1, 136, 16, (unsigned long long)stride);
  g1[1] = (int)(128u << 16);              // tdim0 lo16 = 128
  g1[2] = (int)((((unsigned)tdim1) & 0xFFFFu) << 16);  // tdim0 hi16 = 0
  asm volatile("s_wait_dscnt 0x0\n\ttensor_store_from_lds %0, %1\n\ts_wait_tensorcnt 0x0" ::"s"(g0), "s"(g1) : "memory");
}
int main() {
  const int rows = 40, stride = 512;  // elements per global row (e.g. H * D_V); 128 used
  int fails = 0;
  for (int t = 0; t < 3; ++t) {
    const int r0 = t == 0 ? 3 : (t == 1 ? 10 : 20), tdim1 = t == 0 ? 16 : (t == 1 ? 5 : 0);
    std::vector<unsigned short> h((size_t)rows * stride, 0x1111);
    unsigned short* d;
    hipMalloc(&d, h.size() * 2);
    hipMemcpy(d, h.data(), h.size() * 2, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(k, dim3(1), dim3(32), 0, 0, d, r0, stride, tdim1);
    if (hipDeviceSynchronize() != hipSuccess) { printf("launch failed\n"); return 2; }
    hipMemcpy(h.data(), d, h.size() * 2, hipMemcpyDeviceToHost);
    hipFree(d);
    int bad = 0, written = 0;
    for (int r = 0; r < rows; ++r)
      for (int c = 0; c < stride; ++c) {
        const unsigned short v = h[(size_t)r * stride + c];
        const int rr = r - r0;
        const bool in = rr >= 0 && rr < 16 && rr < tdim1 && c < 128;
        const unsigned short want = in ? (unsigned short)(rr * 1000 + c + 1) : 0x1111;
        if (in) ++written;
        if (v != want) { if (bad < 5) printf("  t%d r%d c%d got %04x want %04x\n", t, r, c, v, want); ++bad; }
      }
    printf("case %d (r0 %d, tdim1 %d): %d elements expected written, %d mismatches\n", t, r0, tdim1, written, bad);
    fails += bad != 0;
  }
  printf(fails ? "FAIL\n" : "PASS\n");
  return fails != 0;
}
