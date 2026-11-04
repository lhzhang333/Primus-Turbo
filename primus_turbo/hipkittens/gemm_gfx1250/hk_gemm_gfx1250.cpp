// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2025 HipKittens Project Contributors
//
// Adapted from the HipKittens gfx1250 BF16 GEMM ladder (kernels/cdna5/gemm/bf16fp32/gfx1250, level 12:
// https://github.com/HazyResearch/HipKittens). Modified by the Primus-Turbo team: the tile, wave grid and
// cluster are template parameters, and the grid may cover a leading block of the matrix only.
//
// TN: C[M, N] (column-major) = A[M, K] . B[N, K]^T, BF16 in and out, FP32 accumulation. Both operands are
// K-contiguous, so a linear layer's forward y = x W^T is A = W, B = x, C = y (row-major [tokens, out]).
// NT: C[M, N] (column-major) = A[K, M]^T . B[K, N], both operands M / N-contiguous (K outermost), so a linear
// layer's weight gradient dW = dY^T x is A = x [tokens, in], B = dY [tokens, out], C = dW (row-major [out, in]).
//
// One workgroup computes a BM x BN tile with 8 waves (two per SIMD, a WM x WN grid). A and B panels of
// 128 along K come in through TDM into a two-slot LDS ring; a CM x CN workgroup cluster multicasts them
// (workgroups of a cluster that share a row of tiles share the A panel, those sharing a column share B).
// Split barriers, a pinned LDS-read / WMMA interleave, and an LDS-staged coalesced epilogue as in the ladder.
// The NT kernel is the same with K-major LDS panels ([128][BM], as the TDM deposits K-outer operands) read into
// the WMMA operands with ds_load_tr16_b128, which transposes 16x8 blocks on the way; A and B are read the same
// way, so their K orders agree. A panel row is padded by 16 B every 256 B (32 B per 512-B row): each half-wave
// of a ds_load_tr16_b128 reads 8 rows x 32 B, and a row pitch of 8 banks (mod 64) keeps those 8 reads on
// disjoint banks (16 B of padding per row, a pitch of 4 banks, makes them collide in pairs).
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


// ---------------------------------------------------------------- NT: K-major panels, transposing LDS reads
// a K-major LDS panel: BLOCK_K rows of R elements, 8 pad elements (16 B) after every 128 (R a multiple of 128)
constexpr int KMAJOR_PAD_INTERVAL = 128;
template<int R>
using kmajor_tile = st<bf16, BLOCK_K, R, ducks::st_shape::st_16x32_padded<KMAJOR_PAD_INTERVAL, 8>>;
template<int R> constexpr int kmajor_row_bytes = (R + 8 * (R / KMAJOR_PAD_INTERVAL)) * 2;
// byte offset of element (row, col) of a panel, col a multiple of 16 (no pad inside a 16-element block)
template<int R> __device__ __forceinline__ constexpr int kmajor_off(int row, int col) {
    return row * kmajor_row_bytes<R> + col * 2 + (col / KMAJOR_PAD_INTERVAL) * 16;
}

typedef __bf16 bf16x8_t __attribute__((ext_vector_type(8)));
typedef __attribute__((address_space(3))) bf16x8_t lds_bf16x8_t;
typedef __attribute__((address_space(3))) unsigned char lds_u8_t;

template<class TILE> __device__ __forceinline__ const lds_u8_t* lds_base(const TILE& t) {
    return (const lds_u8_t*)(uintptr_t)(&t.data[0]);
}

// This lane's byte offset in a K-major panel for ds_load_tr16_b128: the lane supplies the address of the 8
// elements at K row (lane / 16) * 8 + lane % 8, columns ((lane / 8) % 2) * 8 .. + 7 of a 16 x 16 block.
template<int R> __device__ __forceinline__ int tr_lane_off() {
    const int lane = kittens::laneid();
    return ((lane / 16) * 8 + lane % 8) * kmajor_row_bytes<R> + ((lane / 8) % 2) * 16;
}

// The WMMA operand slice of columns [col0, col0 + ROWS) x K rows [k0, k0 + 32) of a K-major panel: per 16
// columns, ds_load_tr16_b128 of K rows k0 .. k0 + 15 and k0 + 16 .. k0 + 31.
template<int ROWS, int R>
__device__ __forceinline__ void load_tr(reg_tile<ROWS, K_STEP>& dst, const lds_u8_t* lane_base, int col0, int k0) {
#pragma unroll
    for (int i = 0; i < ROWS / 16; ++i) {
        const int off = kmajor_off<R>(k0, col0 + 16 * i);
        const bf16x8_t lo = __builtin_amdgcn_ds_load_tr16_b128_v8bf16((lds_bf16x8_t*)(lane_base + off));
        const bf16x8_t hi =
            __builtin_amdgcn_ds_load_tr16_b128_v8bf16((lds_bf16x8_t*)(lane_base + off + 16 * kmajor_row_bytes<R>));
        using packed_t = std::remove_cvref_t<decltype(dst.tiles[0][0].data[0])>;
        const packed_t* lo_p = reinterpret_cast<const packed_t*>(&lo);
        const packed_t* hi_p = reinterpret_cast<const packed_t*>(&hi);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            dst.tiles[i][0].data[j] = lo_p[j];
            dst.tiles[i][0].data[4 + j] = hi_p[j];
        }
    }
}

template<int BM, int BN, int WM, int WN>
struct CfgNT : Cfg<BM, BN, WM, WN> {
    static_assert(BM % KMAJOR_PAD_INTERVAL == 0 && BN % KMAJOR_PAD_INTERVAL == 0, "panel rows of whole pad intervals");
    using A_tile = kmajor_tile<BM>;
    using B_tile = kmajor_tile<BN>;
    using C_tile = typename Cfg<BM, BN, WM, WN>::C_tile;
    static constexpr size_t lds_bytes() {
        return 2 * (sizeof(A_tile) + sizeof(B_tile)) > sizeof(C_tile) ? 2 * (sizeof(A_tile) + sizeof(B_tile))
                                                                       : sizeof(C_tile);
    }
};

template<int BM, int BN, int WM, int WN, int CM, int CN>
__global__ __cluster_dims__(CM, CN, 1) __launch_bounds__(WM * WN * WARP_THREADS, 1)
void nt_kernel(const globals g) {
    using C = CfgNT<BM, BN, WM, WN>;
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

    // g.a is [K, M] and g.b is [K, N], row-major
    const int bm = blockIdx.x, bn = blockIdx.y, w = warpid(), nk = g.a.rows() / BLOCK_K;
    const int a_col = w / WN * WARP_M;
    const int b_col = w % WN * WARP_N;
    const lds_u8_t* a_base[2] = {lds_base(A_lds[0]) + tr_lane_off<BM>(), lds_base(A_lds[1]) + tr_lane_off<BM>()};
    const lds_u8_t* b_base[2] = {lds_base(B_lds[0]) + tr_lane_off<BN>(), lds_base(B_lds[1]) + tr_lane_off<BN>()};
    reg_tile<WARP_M, K_STEP> A_reg[2];
    reg_tile<WARP_N, K_STEP> B_reg[2];
    acc_tile<WARP_M, WARP_N> acc;
    zero(acc);

    uint32_t a_mask = 0, b_mask = 0;
#pragma unroll
    for (int j = 0; j < CN; ++j) a_mask |= 1u << (bm % CM + CM * j);
#pragma unroll
    for (int i = 0; i < CM; ++i) b_mask |= 1u << (i + CM * (bn % CN));

    auto fill = [&](int slot, int k, uint32_t count = 1) {
        if (w == 0) tdm::load_async(A_lds[slot], g.a, {0, 0, k, bm}, g.a.rows(), g.a.cols(), g.a.cols(), a_mask, count);
        if (w == 1) tdm::load_async(B_lds[slot], g.b, {0, 0, k, bn}, g.b.rows(), g.b.cols(), g.b.cols(), b_mask, count);
    };

    fill(0, 0);
    sync::wait_tdm<0>();
    sched::compiler_fence();
    cluster::sync();
    sched::compiler_fence();
    load_tr<WARP_M, BM>(A_reg[0], a_base[0], a_col, 0);
    load_tr<WARP_N, BN>(B_reg[0], b_base[0], b_col, 0);
    for (int k = 0; k < nk; ++k) {
        const int cur = k % 2, nxt = (k + 1) % 2;
        fill(nxt, k + 1 < nk ? k + 1 : 0, k + 1 < nk);
        sched::compiler_fence();
        for (int s = 0; s < KS - 1; ++s) {
            load_tr<WARP_M, BM>(A_reg[(s + 1) % 2], a_base[cur], a_col, (s + 1) * K_STEP);
            load_tr<WARP_N, BN>(B_reg[(s + 1) % 2], b_base[cur], b_col, (s + 1) * K_STEP);
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
        if (w == 0) cluster::arrive();
        sched::compiler_fence();
        sync::wait();
        sched::compiler_fence();
        load_tr<WARP_M, BM>(A_reg[0], a_base[nxt], a_col, 0);
        load_tr<WARP_N, BN>(B_reg[0], b_base[nxt], b_col, 0);
        mma_ABt_part<PRE_BAR, NWM>(acc, A_reg[L], B_reg[L]);
        pin_interleave<(NWM - PRE_BAR + 3) / 4, ND>();
        sched::compiler_fence();
        cluster::wait();
        sched::compiler_fence();
    }
    sync::wait_tdm<0>();
    sync::wait_ds<0>();
    C_tile& C_lds = *(C_tile*)&__shm[0];
    sync::wait_ds<0>();
    sync::sync();
    store(C_lds, acc, w / WN * WARP_M, w % WN * WARP_N);
    sync::wait_ds<0>();
    sync::sync();
    store<THREADS>(g.c, C_lds, {0, 0, bm, bn});
}

template<int BM, int BN, int WM, int WN, int CM, int CN>
int launch_nt(const void* a, const void* b, void* c, int M, int N, int K, int m_tiles, int n_tiles, hipStream_t s) {
    using C = CfgNT<BM, BN, WM, WN>;
    if (m_tiles <= 0 || n_tiles <= 0 || m_tiles % CM || n_tiles % CN || m_tiles * BM > M || n_tiles * BN > N ||
        K % BLOCK_K)
        return 2;
    globals g{gl_ab((bf16*)a, 1, 1, K, M), gl_ab((bf16*)b, 1, 1, K, N), gl_c((bf16*)c, 1, 1, M, N)};
    auto k = nt_kernel<BM, BN, WM, WN, CM, CN>;
    constexpr size_t lds = C::lds_bytes();
    if (hipFuncSetAttribute((const void*)k, hipFuncAttributeMaxDynamicSharedMemorySize, lds) != hipSuccess) return 3;
    hipLaunchKernelGGL(k, dim3(m_tiles, n_tiles), dim3(C::THREADS), lds, s, g);
    return hipGetLastError() == hipSuccess ? 0 : 4;
}


// ---------------------------------------------------------------- TN split-K: grid z splits K
// For long-K GEMMs with fewer tiles than CUs (a dgrad onto a narrow input): workgroup (x, y, z) accumulates
// K-blocks [z nk / S, (z + 1) nk / S) of its tile and stores the FP32 partial tile straight from registers
// into split z of the workspace ([S][N_block][M_block], column-major blocks); reduce_kernel sums the S
// partials into C.
using gl_f = gl<float, -1, -1, -1, -1, ducks::gl_layout::col_major>;
struct globals_sk { gl_ab a, b; gl_f ws; };

template<int BM, int BN, int WM, int WN, int CM, int CN>
__global__ __cluster_dims__(CM, CN, 1) __launch_bounds__(WM * WN * WARP_THREADS, 1)
void tn_sk_kernel(const globals_sk g) {
    using C = Cfg<BM, BN, WM, WN>;
    using A_tile = typename C::A_tile;
    using B_tile = typename C::B_tile;
    constexpr int WARP_M = C::WARP_M, WARP_N = C::WARP_N;
    constexpr int NWM = C::NWMMA, NW = NWM / 4, ND = (C::NDS + 3) / 4, PRE_BAR = C::PRE_BAR;
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    A_tile (&A_lds)[2] = al.allocate_in<segment<0>, A_tile, 2>();
    B_tile (&B_lds)[2] = al.allocate_in<segment<0>, B_tile, 2>();
    sched::lock_simd();

    const int bm = blockIdx.x, bn = blockIdx.y, z = blockIdx.z, w = warpid();
    const int nk_all = g.a.cols() / BLOCK_K, S = gridDim.z;
    const int k0 = (int)((int64_t)nk_all * z / S), nk = (int)((int64_t)nk_all * (z + 1) / S) - k0;
    const int a_off = w / WN * WARP_M * BLOCK_K;
    const int b_off = w % WN * WARP_N * BLOCK_K;
    reg_tile<WARP_M, K_STEP> A_reg[2];
    reg_tile<WARP_N, K_STEP> B_reg[2];
    acc_tile<WARP_M, WARP_N> acc;
    zero(acc);

    uint32_t a_mask = 0, b_mask = 0;
#pragma unroll
    for (int j = 0; j < CN; ++j) a_mask |= 1u << (bm % CM + CM * j);
#pragma unroll
    for (int i = 0; i < CM; ++i) b_mask |= 1u << (i + CM * (bn % CN));

    auto fill = [&](int slot, int k, uint32_t count = 1) {
        if (w == 0) tdm::load_async(A_lds[slot], g.a, {0, 0, bm, k}, g.a.rows(), g.a.cols(), g.a.cols(), a_mask, count);
        if (w == 1) tdm::load_async(B_lds[slot], g.b, {0, 0, bn, k}, g.b.rows(), g.b.cols(), g.b.cols(), b_mask, count);
    };

    fill(0, k0);
    sync::wait_tdm<0>();
    sched::compiler_fence();
    cluster::sync();
    sched::compiler_fence();
    load(A_reg[0], A_lds[0], a_off);
    load(B_reg[0], B_lds[0], b_off);
    for (int k = 0; k < nk; ++k) {
        const int cur = k % 2, nxt = (k + 1) % 2;
        fill(nxt, k + 1 < nk ? k0 + k + 1 : k0, k + 1 < nk);
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
        if (w == 0) cluster::arrive();
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
    store(g.ws, acc, {0, z, bm * WM + w / WN, bn * WN + w % WN});
}

// C[col * ldc + row] = bf16(sum_z ws[z][col][row]) over the mb x nb block, 8 rows per thread (mb % 8 == 0)
__global__ void sk_reduce_kernel(const float* __restrict__ ws, bf16* __restrict__ c, int mb, int nb, int ldc, int S) {
    const int64_t i = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) * 8;
    const int64_t n = (int64_t)mb * nb;
    if (i >= n) return;
    float acc[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = 0.f;
    for (int s = 0; s < S; ++s) {
        const float4* q = reinterpret_cast<const float4*>(ws + s * n + i);
        const float4 v0 = q[0], v1 = q[1];
        acc[0] += v0.x; acc[1] += v0.y; acc[2] += v0.z; acc[3] += v0.w;
        acc[4] += v1.x; acc[5] += v1.y; acc[6] += v1.z; acc[7] += v1.w;
    }
    const int64_t col = i / mb, row = i % mb;
    bf16* out = c + col * ldc + row;
#pragma unroll
    for (int j = 0; j < 8; ++j) out[j] = (bf16)acc[j];
}

template<int BM, int BN, int WM, int WN, int CM, int CN>
int launch_tn_sk(const void* a, const void* b, void* c, void* ws, int M, int N, int K, int m_tiles, int n_tiles,
                 int splits, hipStream_t s) {
    using C = Cfg<BM, BN, WM, WN>;
    const int mb = m_tiles * BM, nb = n_tiles * BN;
    if (m_tiles <= 0 || n_tiles <= 0 || m_tiles % CM || n_tiles % CN || mb > M || nb > N || K % BLOCK_K ||
        splits < 1 || K / BLOCK_K < splits || mb % 8)
        return 2;
    globals_sk g{gl_ab((bf16*)a, 1, 1, M, K), gl_ab((bf16*)b, 1, 1, N, K), gl_f((float*)ws, 1, splits, mb, nb)};
    auto k = tn_sk_kernel<BM, BN, WM, WN, CM, CN>;
    constexpr size_t lds = 2 * (sizeof(typename C::A_tile) + sizeof(typename C::B_tile));
    if (hipFuncSetAttribute((const void*)k, hipFuncAttributeMaxDynamicSharedMemorySize, lds) != hipSuccess) return 3;
    hipLaunchKernelGGL(k, dim3(m_tiles, n_tiles, splits), dim3(C::THREADS), lds, s, g);
    const int64_t n = (int64_t)mb * nb;
    hipLaunchKernelGGL(sk_reduce_kernel, dim3((unsigned)((n / 8 + 255) / 256)), dim3(256), 0, s, (const float*)ws,
                       (bf16*)c, mb, nb, M, splits);
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

// NT variants: id -> (BM, BN, CM, CN). A is [K, M], B is [K, N] (lda = M, ldb = N), C [M, N] column-major (ldc = M).
extern "C" int hk_gemm_nt_variant(int v, int* bm, int* bn, int* cm, int* cn) {
    switch (v) {
        case 0: *bm = 256; *bn = 256; *cm = 4; *cn = 4; return 0;
        case 1: *bm = 256; *bn = 256; *cm = 2; *cn = 4; return 0;  // M a multiple of 512
        default: return 1;
    }
}

extern "C" int hk_gemm_nt(int v, const void* a, const void* b, void* c, int M, int N, int K, int m_tiles, int n_tiles,
                          void* stream) {
    hipStream_t s = (hipStream_t)stream;
    switch (v) {
        case 0: return hk_gemm::launch_nt<256, 256, 4, 2, 4, 4>(a, b, c, M, N, K, m_tiles, n_tiles, s);
        case 1: return hk_gemm::launch_nt<256, 256, 4, 2, 2, 4>(a, b, c, M, N, K, m_tiles, n_tiles, s);
        default: return 1;
    }
}

// TN split-K: the TN variants' tiles and clusters; ws holds splits x (m_tiles BM) x (n_tiles BN) floats
extern "C" int hk_gemm_tn_splitk(int v, const void* a, const void* b, void* c, void* ws, int M, int N, int K,
                                 int m_tiles, int n_tiles, int splits, void* stream) {
    hipStream_t s = (hipStream_t)stream;
    switch (v) {
        case 0: return hk_gemm::launch_tn_sk<256, 256, 4, 2, 4, 4>(a, b, c, ws, M, N, K, m_tiles, n_tiles, splits, s);
        case 1: return hk_gemm::launch_tn_sk<224, 256, 2, 4, 4, 4>(a, b, c, ws, M, N, K, m_tiles, n_tiles, splits, s);
        case 2: return hk_gemm::launch_tn_sk<256, 256, 4, 2, 2, 4>(a, b, c, ws, M, N, K, m_tiles, n_tiles, splits, s);
        default: return 1;
    }
}
