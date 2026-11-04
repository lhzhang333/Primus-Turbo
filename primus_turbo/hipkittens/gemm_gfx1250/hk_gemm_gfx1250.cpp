// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2025 HipKittens Project Contributors
//
// Adapted from the HipKittens gfx1250 BF16 GEMM ladder (kernels/cdna5/gemm/bf16fp32/gfx1250, level 12:
// https://github.com/HazyResearch/HipKittens). Modified by the Primus-Turbo team: the tile, wave grid and
// cluster are template parameters, and the grid may cover a leading block of the matrix only.
//
// C[M, N] (column-major) = A[M, K] . B[N, K]^T, BF16 in and out, FP32 accumulation. Both operands are
// K-contiguous ("TN"), so a linear layer's forward y = x W^T is A = W, B = x, C = y (row-major [tokens, out]).
//
// One workgroup computes a BM x BN tile with 8 waves (two per SIMD, a WM x WN grid). A and B panels of
// 128 along K come in through TDM into a two-slot LDS ring; a CM x CN workgroup cluster multicasts them
// (workgroups of a cluster that share a row of tiles share the A panel, those sharing a column share B).
// Split barriers, a pinned LDS-read / WMMA interleave, and an LDS-staged coalesced epilogue as in the ladder.
//
// Built at run time by build.py (hipcc, -DKITTENS_CDNA5 --offload-arch=gfx1250) and called through ctypes.
#include "kittens.cuh"

using namespace kittens;

namespace hk_gemm {

using gl_ab = gl<bf16, -1, -1, -1, -1>;
using gl_c = gl<bf16, -1, -1, -1, -1, ducks::gl_layout::col_major>;
template<int R, int C> using lds_tile = st<bf16, R, C, ducks::st_shape::st_16x32_padded<>>;
template<int R, int C> using reg_tile = rt<bf16, R, C, ducks::rt_layout::row, ducks::rt_shape::rt_16x32>;
template<int R, int C> using acc_tile = rt_fl<R, C, ducks::rt_layout::col, ducks::rt_shape::rt_16x16>;
struct globals { gl_ab a, b; gl_c c; };
template<int ROWS> constexpr int ds_loads = ROWS / 16 * 2;  // ds_load_b128 per ROWS x 32 operand slice

constexpr int BLOCK_K = 128, K_STEP = 32, KS = BLOCK_K / K_STEP;

// Pin the issue order to 4 groups of ND LDS reads then NW WMMAs (sched_group_barrier masks: 0x100 = DS
// read, 0x008 = MFMA/WMMA). A function, not inlined by hand: inlined it spills (HipKittens note).
template<int NW, int ND> __device__ __forceinline__ void pin_interleave() {
    for (int i = 0; i < 4; ++i) {
        __builtin_amdgcn_sched_group_barrier(0x100, ND, 0);
        __builtin_amdgcn_sched_group_barrier(0x008, NW, 0);
    }
}

// WMMAs [BEGIN, END) of mma_ABt(d, a, b, d) in a zigzag over the rows of D, so each row starts on the B
// fragment the previous one ended on.
template<int BEGIN, int END, class D, class A, class B>
__device__ __forceinline__ void mma_ABt_part(D& d, const A& a, const B& b) {
    [&]<int... I>(std::integer_sequence<int, I...>) {
        ([&] {
            constexpr int i = BEGIN + I, n = i / D::width, t = i % D::width;
            constexpr int m = n % 2 ? D::width - 1 - t : t;
            mma_ABt_base<(t > 0), false>(d.tiles[n][m], a.tiles[n][0], b.tiles[m][0], d.tiles[n][m]);
        }(), ...);
    }(std::make_integer_sequence<int, END - BEGIN>{});
}

template<int BM, int BN, int WM, int WN>
struct Cfg {
    static constexpr int WARP_M = BM / WM, WARP_N = BN / WN, THREADS = WM * WN * WARP_THREADS;
    static constexpr int NWMMA = (WARP_M / 16) * (WARP_N / 16);  // WMMAs per wave per K step of 32
    static constexpr int NDS = ds_loads<WARP_M> + ds_loads<WARP_N>;
    static constexpr int PRE_BAR = (NWMMA * 3) / 8;               // WMMAs issued before the stage flip
    using A_tile = lds_tile<BM, BLOCK_K>;
    using B_tile = lds_tile<BN, BLOCK_K>;
    using C_tile = st<bf16, BM, BN, ducks::st_shape::st_16x32_padded<128, 8>, ducks::st_layout::col>;
    static_assert(WARP_M % 16 == 0 && WARP_N % 16 == 0, "wave tile must be a multiple of 16");
    static_assert(NWMMA % 4 == 0, "four interleave groups");
    static constexpr size_t lds_bytes() {
        return 2 * (sizeof(A_tile) + sizeof(B_tile)) > sizeof(C_tile) ? 2 * (sizeof(A_tile) + sizeof(B_tile))
                                                                       : sizeof(C_tile);
    }
};

template<int BM, int BN, int WM, int WN, int CM, int CN>
__global__ __cluster_dims__(CM, CN, 1) __launch_bounds__(WM * WN * WARP_THREADS, 1)
void tn_kernel(const globals g) {
    using C = Cfg<BM, BN, WM, WN>;
    using A_tile = typename C::A_tile;
    using B_tile = typename C::B_tile;
    using C_tile = typename C::C_tile;
    constexpr int WARP_M = C::WARP_M, WARP_N = C::WARP_N, THREADS = C::THREADS;
    constexpr int NWM = C::NWMMA, NW = NWM / 4, ND = (C::NDS + 3) / 4, PRE_BAR = C::PRE_BAR;
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    A_tile (&A_lds)[2] = al.allocate_in<segment<0>, A_tile, 2>();
    B_tile (&B_lds)[2] = al.allocate_in<segment<0>, B_tile, 2>();
    sched::lock_simd();

    const int bm = blockIdx.x, bn = blockIdx.y, w = warpid(), nk = g.a.cols() / BLOCK_K;
    const int a_off = w / WN * WARP_M * BLOCK_K;
    const int b_off = w % WN * WARP_N * BLOCK_K;
    reg_tile<WARP_M, K_STEP> A_reg[2];
    reg_tile<WARP_N, K_STEP> B_reg[2];
    acc_tile<WARP_M, WARP_N> acc;
    zero(acc);

    // a workgroup is cx + CM * cy in its cluster: the CN with the same cx share A, the CM with the same cy share B
    uint32_t a_mask = 0, b_mask = 0;
#pragma unroll
    for (int j = 0; j < CN; ++j) a_mask |= 1u << (bm % CM + CM * j);
#pragma unroll
    for (int i = 0; i < CM; ++i) b_mask |= 1u << (i + CM * (bn % CN));

    // count = 0 is a null descriptor: a TDM op cannot be EXEC-masked
    auto fill = [&](int slot, int k, uint32_t count = 1) {
        if (w == 0) tdm::load_async(A_lds[slot], g.a, {0, 0, bm, k}, g.a.rows(), g.a.cols(), g.a.cols(), a_mask, count);
        if (w == 1) tdm::load_async(B_lds[slot], g.b, {0, 0, bn, k}, g.b.rows(), g.b.cols(), g.b.cols(), b_mask, count);
    };

    fill(0, 0);
    sync::wait_tdm<0>();
    sched::compiler_fence();
    cluster::sync();
    sched::compiler_fence();
    load(A_reg[0], A_lds[0], a_off);
    load(B_reg[0], B_lds[0], b_off);
    for (int k = 0; k < nk; ++k) {
        const int cur = k % 2, nxt = (k + 1) % 2;
        fill(nxt, k + 1 < nk ? k + 1 : 0, k + 1 < nk);
        sched::compiler_fence();
        for (int s = 0; s < KS - 1; ++s) {
            load(A_reg[(s + 1) % 2], A_lds[cur], a_off + (s + 1) * K_STEP);
            load(B_reg[(s + 1) % 2], B_lds[cur], b_off + (s + 1) * K_STEP);
            mma_ABt(acc, A_reg[s % 2], B_reg[s % 2], acc);
            pin_interleave<NW, ND>();
            sched::compiler_fence();
        }
        constexpr int L = (KS - 1) % 2;
        mma_ABt_part<0, PRE_BAR>(acc, A_reg[L], B_reg[L]);
        sched::compiler_fence();
        sync::wait_ds<0>();
        sync::wait_tdm<0>();
        sched::compiler_fence();
        sync::arrive();
        if (w == 0) cluster::arrive();  // the cluster barrier counts workgroups: one signal each
        sched::compiler_fence();
        sync::wait();
        sched::compiler_fence();
        load(A_reg[0], A_lds[nxt], a_off);
        load(B_reg[0], B_lds[nxt], b_off);
        mma_ABt_part<PRE_BAR, NWM>(acc, A_reg[L], B_reg[L]);
        pin_interleave<(NWM - PRE_BAR + 3) / 4, ND>();
        sched::compiler_fence();
        cluster::wait();
        sched::compiler_fence();
    }
    sync::wait_tdm<0>();
    sync::wait_ds<0>();
    C_tile& C_lds = *(C_tile*)&__shm[0];  // the C staging tile reuses the operand rings
    sync::wait_ds<0>();
    sync::sync();
    store(C_lds, acc, w / WN * WARP_M, w % WN * WARP_N);
    sync::wait_ds<0>();
    sync::sync();
    store<THREADS>(g.c, C_lds, {0, 0, bm, bn});
}

template<int BM, int BN, int WM, int WN, int CM, int CN>
int launch_tn(const void* a, const void* b, void* c, int M, int N, int K, int m_tiles, int n_tiles, hipStream_t s) {
    using C = Cfg<BM, BN, WM, WN>;
    if (m_tiles <= 0 || n_tiles <= 0 || m_tiles % CM || n_tiles % CN || m_tiles * BM > M || n_tiles * BN > N ||
        K % BLOCK_K)
        return 2;
    globals g{gl_ab((bf16*)a, 1, 1, M, K), gl_ab((bf16*)b, 1, 1, N, K), gl_c((bf16*)c, 1, 1, M, N)};
    auto k = tn_kernel<BM, BN, WM, WN, CM, CN>;
    constexpr size_t lds = C::lds_bytes();
    if (hipFuncSetAttribute((const void*)k, hipFuncAttributeMaxDynamicSharedMemorySize, lds) != hipSuccess) return 3;
    hipLaunchKernelGGL(k, dim3(m_tiles, n_tiles), dim3(C::THREADS), lds, s, g);
    return hipGetLastError() == hipSuccess ? 0 : 4;
}

}  // namespace hk_gemm

// ---------------------------------------------------------------- C API (ctypes)
// TN variants: id -> (BM, BN, CM, CN); the grid covers m_tiles x n_tiles tiles from the origin, which must be
// multiples of the cluster. The global views carry the whole matrices (lda = ldb = K, ldc = M).
extern "C" int hk_gemm_tn_variant(int v, int* bm, int* bn, int* cm, int* cn) {
    switch (v) {
        case 0: *bm = 256; *bn = 256; *cm = 4; *cn = 4; return 0;  // the ladder's L12
        case 1: *bm = 224; *bn = 256; *cm = 4; *cn = 4; return 0;  // 7168 = 32 x 224 along M
        case 2: *bm = 256; *bn = 256; *cm = 2; *cn = 4; return 0;  // M a multiple of 512
        default: return 1;
    }
}

extern "C" int hk_gemm_tn(int v, const void* a, const void* b, void* c, int M, int N, int K, int m_tiles, int n_tiles,
                          void* stream) {
    hipStream_t s = (hipStream_t)stream;
    switch (v) {
        case 0: return hk_gemm::launch_tn<256, 256, 4, 2, 4, 4>(a, b, c, M, N, K, m_tiles, n_tiles, s);
        case 1: return hk_gemm::launch_tn<224, 256, 2, 4, 4, 4>(a, b, c, M, N, K, m_tiles, n_tiles, s);
        case 2: return hk_gemm::launch_tn<256, 256, 4, 2, 2, 4>(a, b, c, M, N, K, m_tiles, n_tiles, s);
        default: return 1;
    }
}
