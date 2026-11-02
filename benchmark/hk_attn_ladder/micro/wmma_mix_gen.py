#!/usr/bin/env python3
"""Generate micro/wmma_mix.cpp: shader cycles of one repetition of an issue pattern (one wave per SIMD, expert
mode, lock_simd). Pattern letters: W = independent WMMA (4 accumulators round robin), w = WMMA on one dependent
chain, P = v_pk_fma_f32, F = v_fma_f32, E = v_exp_f32, C = v_cvt_pk_bf16_f32, M = v_mov_b32, D = v_mov_b32_dpp
row_ror:8 bank_mask:0xc, L / T = ds_load_b128 / ds_load_tr16_b128, X = s_set_vgpr_msb 0 (every register here is
below v256: an MSB write that changes nothing), A / U = v_add_f32_e32 / v_mul_f32_e32, K / Q = v_pk_add_f32 /
v_pk_mul_f32, 3 = v_max3_num_f32, N = v_max_num_f32_e32, S = s_add_co_i32 (SALU), Z = s_wait_dscnt 0x3f (never
waits here); every VALU on its own registers (no dependences inside a repetition).
    python3 micro/wmma_mix_gen.py [PATTERN*REPS ...] > micro/wmma_mix.cpp"""
import sys

PATS = ['W*64', 'L*64', 'T*64', 'WL*64', 'WLL*64', 'WT*64', 'WTT*64', 'WWL*64', 'WLP*64', 'WLPE*64', 'WPPEL*64', 'WTPPE*64', 'WWLL*32', 'WWWLLL*16', 'LLLLWWWW*16']  # fmt: skip
REP = 16
if len(sys.argv) > 1:
    PATS = sys.argv[1:]


def ins(c, j, wk):
    if c == "W":
        a = 64 + 8 * (wk % 4)
        return f"v_wmma_f32_16x16x32_bf16 v[{a}:{a + 7}], v[96:103], v[104:111], v[{a}:{a + 7}]"
    if c == "w":
        return "v_wmma_f32_16x16x32_bf16 v[64:71], v[96:103], v[104:111], v[64:71]"
    if c in "abcd":  # chain a..d: each WMMA depends on the previous WMMA of its own chain
        a = 64 + 8 * "abcd".index(c)
        return f"v_wmma_f32_16x16x32_bf16 v[{a}:{a + 7}], v[96:103], v[104:111], v[{a}:{a + 7}]"
    r = 112 + 2 * (j % 32)
    if c == "P":
        return f"v_pk_fma_f32 v[{r}:{r + 1}], v[176:177], v[178:179], v[{r}:{r + 1}]"
    if c == "F":
        return f"v_fma_f32 v{r}, v176, v177, v{r}"
    if c == "E":
        return f"v_exp_f32 v{r}, v{180 + j % 8}"
    if c == "C":
        return f"v_cvt_pk_bf16_f32 v{r}, v176, v177"
    if c == "M":
        return f"v_mov_b32 v{r}, v{180 + j % 8}"
    if c == "D":
        return f"v_mov_b32_dpp v{r}, v{180 + j % 8} row_ror:8 row_mask:0xf bank_mask:0xc"
    if c == "X":
        return "s_set_vgpr_msb 0"
    if c == "A":
        return f"v_add_f32_e32 v{r}, v176, v{r}"
    if c == "U":
        return f"v_mul_f32_e32 v{r}, v176, v{r}"
    if c == "K":
        return f"v_pk_add_f32 v[{r}:{r + 1}], v[176:177], v[{r}:{r + 1}]"
    if c == "Q":
        return f"v_pk_mul_f32 v[{r}:{r + 1}], v[176:177], v[{r}:{r + 1}]"
    if c == "3":
        return f"v_max3_num_f32 v{r}, v176, v177, v{r}"
    if c == "N":
        return f"v_max_num_f32_e32 v{r}, v176, v{r}"
    if c == "S":
        return f"s_add_co_i32 s{96 + j % 4}, s{96 + j % 4}, 1"
    if c == "Z":
        return "s_wait_dscnt 0x3f"
    # VGPR bank tests (bank = register % 4; r % 4 is 0 or 2): e = v_fma_f32 with three sources in three banks; f / g =
    # v_pk_add_f32 with
    # the pairs on the same / other bank pair; h = v_pk_fma_f32 with an SGPR-pair c (two VGPR pairs, other banks);
    # i = v_max3_num_f32 with sources in three banks; j = v_add_f32 with both sources in one bank, no read of v{r}
    if c == "e":
        return f"v_fma_f32 v{r}, v{177 + r % 4}, v{179 + r % 4}, v{r}"
    if c == "f":
        return f"v_pk_add_f32 v[{r}:{r + 1}], v[{176 + r % 4}:{177 + r % 4}], v[{r}:{r + 1}]"
    if c == "g":
        return f"v_pk_add_f32 v[{r}:{r + 1}], v[{178 + r % 4}:{179 + r % 4}], v[{r}:{r + 1}]"
    if c == "h":
        return f"v_pk_fma_f32 v[{r}:{r + 1}], v[{r}:{r + 1}], s[96:97], v[{178 + r % 4}:{179 + r % 4}] op_sel_hi:[1,0,1]"
    if c == "i":
        return f"v_max3_num_f32 v{r}, v{177 + r % 4}, v{179 + r % 4}, v{r}"
    if c == "j":
        return f"v_add_f32_e32 v{r}, v176, v180"
    # explicit banks: q4 = a bank-0 destination register cycling over 16 (stride 4)
    q4 = 112 + 4 * (j % 16)
    if c == "k":  # dst-read source and the other source both bank 0
        return f"v_add_f32_e32 v{q4}, v176, v{q4}"
    if c == "l":  # dst-read source bank 0, other source bank 1
        return f"v_add_f32_e32 v{q4}, v177, v{q4}"
    if c == "m":  # two non-dst sources, both bank 0, varying (other registers of the 16-cycle)
        return f"v_add_f32_e32 v{q4}, v{180 + 4 * (j % 2)}, v{q4 + 64 if q4 + 64 < 176 else q4 - 64}"
    if c == "n":  # two non-dst varying sources in banks 0 / 1
        return f"v_add_f32_e32 v{q4}, v181, v{q4 + 64 if q4 + 64 < 176 else q4 - 64}"
    if c == "o":  # fma: dst-read bank 0 + sources banks 0, 1 (one conflict)
        return f"v_fma_f32 v{q4}, v176, v177, v{q4}"
    if c == "p":  # fma: banks 0 (dst-read), 1, 2
        return f"v_fma_f32 v{q4}, v177, v178, v{q4}"
    if c == "t":  # pk_fma: three VGPR pairs (6 bank reads over 4 banks)
        return f"v_pk_fma_f32 v[{q4}:{q4 + 1}], v[176:177], v[178:179], v[{q4}:{q4 + 1}]"
    if c == "u":  # pk_fma: pairs banks (0,1) dst-read, (2,3), SGPR c
        return f"v_pk_fma_f32 v[{q4}:{q4 + 1}], v[178:179], s[96:97], v[{q4}:{q4 + 1}] op_sel_hi:[1,0,1]"
    if c == "v":  # pk_mul: pairs (0,1) dst-read and (0,1)
        return f"v_pk_mul_f32 v[{q4}:{q4 + 1}], v[176:177], v[{q4}:{q4 + 1}]"
    if c in "LT":
        d = 192 + 4 * (j % 16)
        mn = "ds_load_b128" if c == "L" else "ds_load_tr16_b128"
        return f"{mn} v[{d}:{d + 3}], v188 offset:{(j % 32) * 512}"
    raise ValueError(c)


clob = ", ".join([f'"v{r}"' for r in range(64, 256)] + [f'"s{r}"' for r in range(96, 100)] + ['"scc"'])
out = [r'''// GENERATED by micro/wmma_mix_gen.py: shader cycles per repetition of issue patterns (see the generator).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(x)                                                                       \
  do {                                                                                 \
    hipError_t e = (x);                                                                \
    if (e != hipSuccess) {                                                             \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e)); \
      exit(1);                                                                         \
    }                                                                                  \
  } while (0)
#ifndef WAVES
#define WAVES 4
#endif
constexpr int ITERS = 128;
''']
init = (["v_mov_b32 v96, 0x3f803f80"] + [f"v_mov_b32 v{r}, v96" for r in range(97, 112)] +
        [f"v_mov_b32 v{r}, 1.0" for r in range(112, 188)] + ["s_wait_alu depctr_va_vdst(0)"])
for p, pat_rep in enumerate(PATS):
    pat, rep = pat_rep.split("*")
    body, j, wk = [], 0, 0
    for _ in range(int(rep)):
        for c in pat:
            body.append(ins(c, j, wk))
            if c in "Wwabcd":
                wk += 1
            else:
                j += 1
    out.append(f'__global__ __launch_bounds__(32 * WAVES, 1) void k{p}(unsigned long long* cyc) {{')
    out.append("  __builtin_amdgcn_s_setreg(154, 1);")
    out.append("  __shared__ __attribute__((aligned(16))) unsigned char lds[WAVES * 16384];")
    out.append("  const unsigned la = (unsigned)(uintptr_t)(lds + (threadIdx.x / 32) * 16384 + (threadIdx.x % 32) * 16);")
    out.append('  asm volatile("v_mov_b32 v188, %0" :: "v"(la) : "v188");')
    out.append("  asm volatile(" + " ".join(f'"{l}\\n"' for l in init) + f" ::: {clob});")
    out.append("  const unsigned long long t0 = __builtin_readcyclecounter();")
    out.append("  _Pragma(\"unroll 1\") for (int it = 0; it < ITERS; ++it) asm volatile(" + " ".join(f'"{l}\\n"' for l in body)
               + f" ::: {clob});")
    out.append('  asm volatile("s_wait_dscnt 0x0\\n s_wait_alu depctr_va_vdst(0)" ::: "memory");')
    out.append("  const unsigned long long t1 = __builtin_readcyclecounter();")
    out.append("  if (threadIdx.x % 32 == 0) cyc[blockIdx.x * WAVES + threadIdx.x / 32] = t1 - t0;")
    out.append("}")
out.append("typedef void (*K)(unsigned long long*);")
out.append("static const K KS[] = {" + ", ".join(f"k{p}" for p in range(len(PATS))) + "};")
out.append("static const char* NAMES[] = {" + ", ".join(f'"{p}"' for p in PATS) + "};")
out.append(r'''int main() {
  hipDeviceProp_t p;
  CHECK(hipGetDeviceProperties(&p, 0));
  const int grid = p.multiProcessorCount;
  unsigned long long* cyc;
  CHECK(hipMalloc(&cyc, (size_t)grid * WAVES * 8));
  std::vector<unsigned long long> h(grid * WAVES);
  printf("pattern                   cycles/rep  (one wave per SIMD, median)\n");
  for (int i = 0; i < (int)(sizeof(KS) / sizeof(KS[0])); ++i) {
    for (int r = 0; r < 3; ++r) KS[i]<<<grid, 32 * WAVES>>>(cyc);
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(h.data(), cyc, h.size() * 8, hipMemcpyDeviceToHost));
    std::sort(h.begin(), h.end());
    printf("%-25s %10.2f\n", NAMES[i], (double)h[h.size() / 2] / (double)ITERS);
  }
  return 0;
}''')
print("\n".join(out))
