/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// k_mxq.hip -- MXFP8 (E4M3 data, E8M0 scales, blocks of 32, scale rule mx_exp: no saturation) quantization of the
// MLA attention operands for the gfx1250 MXFP8 kernels.
//
//   k_mxq_kbar      kbar[b, h, d] = mean over s of K[b, s, h, d] (fp32), d < 192: the k centring (softmax-invariant)
//   k_mxq_rows192   X [b, s, h, 192] bf16 -> X8 [b, h, s, 192] E4M3 MX-interleaved (nope 128 + rope 64, mx_common.h),
//                   XS [b, h, s, 8] E8M0 (nope blocks 0..3, rope 0..1, two pad bytes 127); Q and K (rowwise along d)
//   k_mxq_rows128   X [b, s, h, 128] -> X8 [b, h, s, 128] (one 128 chunk), XS [b, h, s, 4]; V and dO (along d)
//   k_mxq_cols128   X [b, s, h, 128] -> X8T [b, h, 128, s] (s MX-interleaved per 128), XTS [b, h, s / 128, 128, 4]
//                   (per s tile: d-major, the 4 blocks of 32 s); V (forward) and dO (backward), along s
//   k_mxq_cols192   the same for 192 columns (Q, K along s: the backward's dK / dQ operands)
//   k_mxq_rc192/128 rows + cols of a tensor in one pass (one read): the forward's and the backward's operands at once
// Row kernels' mode: bit 0 subtract kbar (K), bit 1 aux[row] = x . kbar (Q: the lse correction), bit 2 aux[row] =
// x . Y[row] (dO: delta = rowsum(dO * O)), bit 3 also write the dequantized values as bf16 XD [b, s, h, D]
// (contiguous) for a bf16 consumer of the same operand, bit 4 (D = 192) scale layout R2: [s0..s3, 0, 0, s4, s5] (the
// rope WMMA over d 64..191: E8M0 0 = 2^-127 silences d 64..127), bit 5 quantize x * pre (after the aux dots; Q:
// pre = softmax scale * log2(e), the logits in log2 units). Col kernels' mode: bit 0 subtract kbar.
// Inputs: any b / s / h strides (elements), the d axis contiguous and 16-byte aligned.
#include "mx_common.h"

#ifndef RCT_PROBE
#define RCT_PROBE 0
#endif
#ifndef RC_BF16
#define RC_BF16 1  // rc_tile: bf16-native MX blocks where no fp32 pre-processing is needed (rows without kbar / pre, cols
                   // without kbar): |x| max on packed bf16 bits, v_cvt_scalef32_pk8_fp8_bf16 (bit-exact with the f32 path)
#endif
#ifndef RC_X8G
#define RC_X8G 1  // rct / rctr: the row codes stored to global straight from the lanes (no X8 staging / TDM store: LDS
                  // 101 -> 77 KB for D 192, 4 workgroups per CU)
#endif
#ifndef RC_FAST
#define RC_FAST 1  // tree reductions (amax, row dots) and the kbar slice in registers (rc_tile)
#endif

using namespace mx;

namespace {
// store 32 converted values of one MX block j (logical k 32 j .. 32 j + 31 of a chunk of CH = 128 or 64 elements) to
// the chunk's MX-interleaved bytes: k = 16 c + 8 h + r -> byte (CH / 2) h + 8 c + r
template <int CH>
__device__ __forceinline__ void store_block(unsigned char *chunk, int j, u32x2 q0, u32x2 q1, u32x2 q2, u32x2 q3) {
    // q0: k 0..7 (c = 2j, h 0), q1: k 8..15 (c 2j, h 1), q2: k 16..23 (c 2j + 1, h 0), q3: k 24..31 (c 2j + 1, h 1)
    const u32x4 h0 = {q0[0], q0[1], q2[0], q2[1]};
    const u32x4 h1 = {q1[0], q1[1], q3[0], q3[1]};
    *(u32x4 *)(chunk + 16 * j) = h0;
    *(u32x4 *)(chunk + CH / 2 + 16 * j) = h1;
}

__device__ __forceinline__ f32x8 bf8_to_f32(u32x4 w) {
    const bf16x8 v = __builtin_bit_cast(bf16x8, w);
    f32x8 r;
    static_for<8>([&](auto i) { r[(int)i] = (float)v[(int)i]; });
    return r;
}
__device__ __forceinline__ u32x4 f32_to_bf8(f32x8 v) {
    bf16x8 r;
    static_for<8>([&](auto i) { r[(int)i] = (bf16)v[(int)i]; });
    return __builtin_bit_cast(u32x4, r);
}

// the E4M3 value of a byte as fp32 (decode, for the dequantized copy): exact
__device__ __forceinline__ float e4m3_to_f32(unsigned v) {
    const unsigned s = (v >> 7) & 1, e = (v >> 3) & 15, m = v & 7;
    float x = e == 0 ? __builtin_ldexpf((float)m, -9) : __builtin_ldexpf((float)(8 + m), (int)e - 10);
    return s ? -x : x;
}

// one row lane's MX block: 32 values v[4][8] -> amax, exponent, 4 x 8 E4M3 codes; optional dequantized fp32 copy
struct Block {
    int e;
    u32x2 q[4];
};
__device__ __forceinline__ Block quant_block(const f32x8 (&v)[4]) {
#if RC_FAST
    // a tree (the max is order-independent: the same codes): a 32-long dependent chain of max ops cost ~80 cycles
    float a[4];
    static_for<4>([&](auto i) {
        const f32x8 &x = v[(int)i];
        const float m0 = __builtin_fmaxf(__builtin_fabsf(x[0]), __builtin_fabsf(x[1])),
                    m1 = __builtin_fmaxf(__builtin_fabsf(x[2]), __builtin_fabsf(x[3])),
                    m2 = __builtin_fmaxf(__builtin_fabsf(x[4]), __builtin_fabsf(x[5])),
                    m3 = __builtin_fmaxf(__builtin_fabsf(x[6]), __builtin_fabsf(x[7]));
        a[(int)i] = __builtin_fmaxf(__builtin_fmaxf(m0, m1), __builtin_fmaxf(m2, m3));
    });
    const float amax = __builtin_fmaxf(__builtin_fmaxf(a[0], a[1]), __builtin_fmaxf(a[2], a[3]));
#else
    float amax = 0.f;
    static_for<4>([&](auto i) {
        static_for<8>([&](auto e) { amax = __builtin_fmaxf(amax, __builtin_fabsf(v[(int)i][(int)e])); });
    });
#endif
    Block blk;
    blk.e = mx_exp(amax);
    const float sc = exp2i(blk.e);
    static_for<4>([&](auto i) { blk.q[(int)i] = cvt8_e4m3(v[(int)i], sc); });
    return blk;
}
__device__ __forceinline__ void dequant_block(const Block &blk, f32x8 (&out)[4]) {
    static_for<4>([&](auto i) {
        static_for<8>([&](auto e) {
            const unsigned w = blk.q[(int)i][(int)e / 4];
            out[(int)i][(int)e] = __builtin_ldexpf(e4m3_to_f32((w >> (8 * ((int)e % 4))) & 0xFF), blk.e);
        });
    });
}

#if RC_BF16
typedef short i16x2 __attribute__((ext_vector_type(2)));
// the MX block of 32 bf16 values packed in w[4] (values 8 i .. 8 i + 7 in w[i]): amax from the magnitude bits (the
// integer order of |x| bits is the float order), the codes straight from bf16; the same codes as quant_block on f32
__device__ __forceinline__ Block quant_block_bf16(const u32x4 (&w)[4]) {
    auto ab = [](unsigned x) { return __builtin_bit_cast(i16x2, x & 0x7fff7fffu); };
    i16x2 m[4];
    static_for<4>([&](auto ii) {
        constexpr int i = ii;
        m[i] = __builtin_elementwise_max(__builtin_elementwise_max(ab(w[i][0]), ab(w[i][1])),
                                         __builtin_elementwise_max(ab(w[i][2]), ab(w[i][3])));
    });
    const i16x2 mm = __builtin_elementwise_max(__builtin_elementwise_max(m[0], m[1]), __builtin_elementwise_max(m[2], m[3]));
    const int mb = mm[0] > mm[1] ? (int)mm[0] : (int)mm[1];
    const float amax = __builtin_bit_cast(float, (unsigned)mb << 16);
    Block blk;
    blk.e = mx_exp(amax);
    const float sc = exp2i(blk.e);
    static_for<4>([&](auto i) {
        blk.q[(int)i] = __builtin_amdgcn_cvt_scalef32_pk8_fp8_bf16(__builtin_bit_cast(bf16x8, w[(int)i]), sc);
    });
    return blk;
}
#endif

// rows kernel body: D = 192 (8 lanes per row: 6 blocks + 2 pad) or 128 (4 lanes per row). A wave owns NR * RPW
// consecutive rows of one (b, h) (rows in X8's (b, h, s) order: contiguous stores), loading the lane's kbar slice
// once. Requires S % (NR * RPW) == 0.
template <int D>
__device__ __forceinline__ void rows_body(const bf16 *__restrict__ X, unsigned char *__restrict__ X8,
                                          unsigned char *__restrict__ XS, const float *__restrict__ kbar,
                                          const bf16 *__restrict__ Y, float *__restrict__ aux, bf16 *__restrict__ XD,
                                          int B, int S, int H, long long sb, long long ss, long long sh, long long yb,
                                          long long ys, long long yh, int mode, float pre) {
    constexpr int NBLK = D / 32;           // 6 or 4
    constexpr int LPR = D == 192 ? 8 : 4;  // lanes per row
    constexpr int RS = D == 192 ? 8 : 4;   // scale bytes per row
    constexpr int RPW = 32 / LPR;          // rows per wave per group
    constexpr int NR = 8;                  // row groups per wave
    const int t = threadIdx.x, lane = t & 31, j = lane % LPR;
    // block 256 = 8 waves (no blockDim: it is a hidden kernarg)
    const long long wrow0 = ((long long)blockIdx.x * 8 + (t >> 5)) * (NR * RPW);
    const long long nrows = (long long)B * H * S;
    if (wrow0 >= nrows) return;
    const int bh = (int)(wrow0 / S), h = bh % H, b = bh / H;
    const int s0 = (int)(wrow0 % S);
    f32x8 kb[4];
    if ((mode & 3) && j < NBLK) {
        const float *kp = kbar + (long long)bh * D + 32 * j;
        static_for<4>([&](auto i) {
            const f32x4 k0 = *(const f32x4 *)(kp + 8 * (int)i), k1 = *(const f32x4 *)(kp + 8 * (int)i + 4);
            kb[(int)i] = f32x8{k0[0], k0[1], k0[2], k0[3], k1[0], k1[1], k1[2], k1[3]};
        });
    }
    for (int g = 0; g < NR; ++g) {
        const int s = s0 + g * RPW + lane / LPR;
        const long long row = (long long)bh * S + s;
        float dot = 0.f;
        if (j < NBLK) {
            const bf16 *src = X + b * sb + (long long)s * ss + h * sh + 32 * j;
            f32x8 v[4];
            static_for<4>([&](auto i) { v[(int)i] = bf8_to_f32(*(const u32x4 *)(src + 8 * (int)i)); });
            if (mode & 1) {
                static_for<4>([&](auto i) { v[(int)i] = v[(int)i] - kb[(int)i]; });
            } else if (mode & 2) {
                static_for<4>([&](auto i) { static_for<8>([&](auto e) { dot += v[(int)i][(int)e] * kb[(int)i][(int)e]; }); });
            }
            if (mode & 4) {
                const bf16 *yp = Y + b * yb + (long long)s * ys + h * yh + 32 * j;
                static_for<4>([&](auto i) {
                    const f32x8 y = bf8_to_f32(*(const u32x4 *)(yp + 8 * (int)i));
                    static_for<8>([&](auto e) { dot += v[(int)i][(int)e] * y[(int)e]; });
                });
            }
            if (mode & 32) {
                static_for<4>([&](auto i) { v[(int)i] = v[(int)i] * pre; });
            }
            const Block blk = quant_block(v);
            unsigned char *dst = X8 + row * D;
            if (D == 128 || j < 4)
                store_block<128>(dst, j, blk.q[0], blk.q[1], blk.q[2], blk.q[3]);
            else
                store_block<64>(dst + 128, j - 4, blk.q[0], blk.q[1], blk.q[2], blk.q[3]);
            XS[row * RS + ((D == 192 && (mode & 16) && j >= 4) ? j + 2 : j)] = (unsigned char)(blk.e + 127);
            if (mode & 8) {
                f32x8 dq[4];
                dequant_block(blk, dq);
                bf16 *xd = XD + (((long long)b * S + s) * H + h) * D + 32 * j;
                static_for<4>([&](auto i) { *(u32x4 *)(xd + 8 * (int)i) = f32_to_bf8(dq[(int)i]); });
            }
        } else if (j < RS) {
            // pad scales: 127 = 2^0 for the rope WMMA's zero half (layout R1), 0 = 2^-127 silences d 64..127 (R2)
            XS[row * RS + ((mode & 16) ? j - 2 : j)] = (mode & 16) ? 0 : 127;
        }
        if (mode & 6) {
            static_for<LPR == 8 ? 3 : 2>([&](auto k) { dot += __shfl_xor(dot, 1 << (int)k, LPR); });
            if (j == 0) aux[row] = dot;
        }
    }
}

// cols kernel body: one 128-row (s) x D tile of X -> X8T rows d (s interleaved), XTS [.., D, 4]. 256 threads: the
// tile is loaded row-major into LDS, then task = (2 consecutive d, block j of 32 s): 32 ds_load_b32, 2 MX blocks.
// (A 4-tile variant with full-line stores was not faster: 0.093 vs 0.088 ms for V at b2 s4096 h128.)
template <int D>
__device__ __forceinline__ void cols_body(const bf16 *__restrict__ X, unsigned char *__restrict__ X8T,
                                          unsigned char *__restrict__ XTS, const float *__restrict__ kbar, int S,
                                          int H, long long sb, long long ss, long long sh, int mode,
                                          unsigned char *tile) {
    constexpr int XP = D * 2 + 16;  // LDS row pitch (bytes) of the bf16 tile
    constexpr int TPR = D / 8;      // 16-B chunks per row
    constexpr int NT = 256;
    const int kt = blockIdx.x, h = blockIdx.y, b = blockIdx.z, t = threadIdx.x;
    const bf16 *src = X + b * sb + (long long)(kt * 128) * ss + h * sh;
    for (int i = t; i < 128 * TPR; i += NT) {
        const int r = i / TPR, c = (i % TPR) * 8;
        *(u32x4 *)(tile + r * XP + 2 * c) = *(const u32x4 *)(src + (long long)r * ss + c);
    }
    __syncthreads();
    for (int task = t; task < (D / 2) * 4; task += NT) {
        const int dp = task % (D / 2), j = task / (D / 2), d0 = 2 * dp;
        float kb0 = 0.f, kb1 = 0.f;
        if (mode & 1) {
            const float *kp = kbar + ((long long)b * H + h) * D + d0;
            kb0 = kp[0];
            kb1 = kp[1];
        }
        f32x8 v0[4], v1[4];
        static_for<4>([&](auto i) {
            static_for<8>([&](auto e) {
                const int r = 32 * j + 8 * (int)i + (int)e;
                const uint32_t w = *(const uint32_t *)(tile + r * XP + 2 * d0);
                v0[(int)i][(int)e] = __builtin_bit_cast(float, w << 16) - kb0;
                v1[(int)i][(int)e] = __builtin_bit_cast(float, w & 0xFFFF0000u) - kb1;
            });
        });
        const Block b0 = quant_block(v0), b1 = quant_block(v1);
        unsigned char *dst = X8T + (((long long)b * H + h) * D + d0) * S + kt * 128;
        store_block<128>(dst, j, b0.q[0], b0.q[1], b0.q[2], b0.q[3]);
        store_block<128>(dst + S, j, b1.q[0], b1.q[1], b1.q[2], b1.q[3]);
        unsigned char *xs = XTS + ((((long long)b * H + h) * (S / 128) + kt) * D + d0) * 4 + j;
        xs[0] = (unsigned char)(b0.e + 127);
        xs[4] = (unsigned char)(b1.e + 127);
    }
}

// rows + cols of one 128-row (s) tile in one pass (the tile read once): the rows kernel's outputs (X8 / XS / aux, mode
// bits 0..5 as k_mxq_rows) from the LDS tile, then the cols kernel's (X8T / XTS: kbar subtracted when mode & 1, never
// pre-scaled). 256 threads; LDS: the bf16 tile + kbar.
template <int D>
__device__ __forceinline__ void rc_body(const bf16 *__restrict__ X, unsigned char *__restrict__ X8,
                                        unsigned char *__restrict__ XS, unsigned char *__restrict__ X8T,
                                        unsigned char *__restrict__ XTS, const float *__restrict__ kbar,
                                        const bf16 *__restrict__ Y, float *__restrict__ aux, int S, int H, long long sb,
                                        long long ss, long long sh, long long yb, long long ys, long long yh, int mode,
                                        float pre, unsigned char *tile, float *kb_lds) {
    constexpr int XP = D * 2 + 16;
    constexpr int TPR = D / 8;
    constexpr int NT = 256;
    constexpr int NBLK = D / 32;
    constexpr int LPR = D == 192 ? 8 : 4;
    constexpr int RS = D == 192 ? 8 : 4;
    constexpr int RPP = NT / LPR;  // rows per pass
    const int kt = blockIdx.x, h = blockIdx.y, b = blockIdx.z, t = threadIdx.x;
    const long long bh = (long long)b * H + h;
    const bf16 *src = X + b * sb + (long long)(kt * 128) * ss + h * sh;
    for (int i = t; i < 128 * TPR; i += NT) {
        const int r = i / TPR, c = (i % TPR) * 8;
        *(u32x4 *)(tile + r * XP + 2 * c) = *(const u32x4 *)(src + (long long)r * ss + c);
    }
    if (mode & 3) {
        for (int i = t; i < D; i += NT) kb_lds[i] = kbar[bh * D + i];
    }
    __syncthreads();
    // ---- rows: LPR lanes per row (lane j < NBLK: block j of 32), RPP rows per pass
    const int j = t % LPR;
    for (int r0 = 0; r0 < 128; r0 += RPP) {
        const int r = r0 + t / LPR;
        const long long row = bh * S + (long long)kt * 128 + r;
        float dot = 0.f;
        if (j < NBLK) {
            f32x8 v[4];
            static_for<4>([&](auto i) {
                v[(int)i] = bf8_to_f32(*(const u32x4 *)(tile + r * XP + 2 * (32 * j + 8 * (int)i)));
            });
            if (mode & 1) {
                static_for<4>([&](auto i) {
                    static_for<8>([&](auto e) { v[(int)i][(int)e] -= kb_lds[32 * j + 8 * (int)i + (int)e]; });
                });
            } else if (mode & 2) {
                static_for<4>([&](auto i) {
                    static_for<8>([&](auto e) { dot += v[(int)i][(int)e] * kb_lds[32 * j + 8 * (int)i + (int)e]; });
                });
            }
            if (mode & 4) {
                const bf16 *yp = Y + b * yb + (long long)(kt * 128 + r) * ys + h * yh + 32 * j;
                static_for<4>([&](auto i) {
                    const f32x8 y = bf8_to_f32(*(const u32x4 *)(yp + 8 * (int)i));
                    static_for<8>([&](auto e) { dot += v[(int)i][(int)e] * y[(int)e]; });
                });
            }
            if (mode & 32) {
                static_for<4>([&](auto i) { v[(int)i] = v[(int)i] * pre; });
            }
            const Block blk = quant_block(v);
            unsigned char *dst = X8 + row * D;
            if (D == 128 || j < 4)
                store_block<128>(dst, j, blk.q[0], blk.q[1], blk.q[2], blk.q[3]);
            else
                store_block<64>(dst + 128, j - 4, blk.q[0], blk.q[1], blk.q[2], blk.q[3]);
            XS[row * RS + ((D == 192 && (mode & 16) && j >= 4) ? j + 2 : j)] = (unsigned char)(blk.e + 127);
        } else if (j < RS) {
            XS[row * RS + ((mode & 16) ? j - 2 : j)] = (mode & 16) ? 0 : 127;
        }
        if (mode & 6) {
            static_for<LPR == 8 ? 3 : 2>([&](auto k) { dot += __shfl_xor(dot, 1 << (int)k, LPR); });
            if (j == 0) aux[row] = dot;
        }
    }
    // ---- cols: task = (2 consecutive d, block jb of 32 s)
    for (int task = t; task < (D / 2) * 4; task += NT) {
        const int dp = task % (D / 2), jb = task / (D / 2), d0 = 2 * dp;
        const float kb0 = (mode & 1) ? kb_lds[d0] : 0.f, kb1 = (mode & 1) ? kb_lds[d0 + 1] : 0.f;
        f32x8 v0[4], v1[4];
        static_for<4>([&](auto i) {
            static_for<8>([&](auto e) {
                const int r = 32 * jb + 8 * (int)i + (int)e;
                const uint32_t w = *(const uint32_t *)(tile + r * XP + 2 * d0);
                v0[(int)i][(int)e] = __builtin_bit_cast(float, w << 16) - kb0;
                v1[(int)i][(int)e] = __builtin_bit_cast(float, w & 0xFFFF0000u) - kb1;
            });
        });
        const Block b0 = quant_block(v0), b1 = quant_block(v1);
        unsigned char *dst = X8T + (bh * D + d0) * S + kt * 128;
        store_block<128>(dst, jb, b0.q[0], b0.q[1], b0.q[2], b0.q[3]);
        store_block<128>(dst + S, jb, b1.q[0], b1.q[1], b1.q[2], b1.q[3]);
        unsigned char *xs = XTS + ((bh * (S / 128) + kt) * D + d0) * 4 + jb;
        xs[0] = (unsigned char)(b0.e + 127);
        xs[4] = (unsigned char)(b1.e + 127);
    }
}

// rows + cols of one bf16 tile in LDS (unpadded rows of D * 2 bytes) into LDS staging: x8 rows (128 x D), xs (128 x RS),
// x8t (D x 128), xts (D x 4); NT threads
template <int D, int NT>
__device__ __forceinline__ void rc_tile(const unsigned char *tile, unsigned char *x8, unsigned char *xs,
                                        unsigned char *x8t, unsigned char *xts, const float *kb_lds,
                                        const bf16 *__restrict__ Y, float *__restrict__ aux, int b, int h, int kt,
                                        long long bh, int S, long long yb, long long ys, long long yh, int mode,
                                        float pre) {
    constexpr int XR = D * 2;
    constexpr int NBLK = D / 32;
    constexpr int LPR = D == 192 ? 8 : 4;
    constexpr int RS = D == 192 ? 8 : 4;
    constexpr int RPP = NT / LPR;
    const int t = threadIdx.x;
    // ---- rows: LPR lanes per row (lane j < NBLK: block j of 32), RPP rows per pass
    const int j = t % LPR;
#if RC_FAST
    f32x8 kbr[4];  // the lane's kbar slice (its block j is the same in every pass)
    if ((mode & 3) && j < NBLK) {
        static_for<4>([&](auto i) {
            const f32x4 k0 = *(const f32x4 *)(kb_lds + 32 * j + 8 * (int)i),
                        k1 = *(const f32x4 *)(kb_lds + 32 * j + 8 * (int)i + 4);
            kbr[(int)i] = f32x8{k0[0], k0[1], k0[2], k0[3], k1[0], k1[1], k1[2], k1[3]};
        });
    }
#endif
    for (int r0 = 0; r0 < (RCT_PROBE == 3 ? 0 : 128); r0 += RPP) {  // probe 3: no rows phase
        const int r = r0 + t / LPR;
        const long long row = bh * S + (long long)kt * 128 + r;
        float dot = 0.f;
#if RC_BF16
        if (j < NBLK && (mode & 39) == 0) {  // no kbar, no Y dots, no pre: bf16 codes
            u32x4 w[4];
            static_for<4>([&](auto i) { w[(int)i] = *(const u32x4 *)(tile + r * XR + 2 * (32 * j + 8 * (int)i)); });
            const Block blk = quant_block_bf16(w);
            unsigned char *dst = x8 + r * D;
            if (D == 128 || j < 4)
                store_block<128>(dst, j, blk.q[0], blk.q[1], blk.q[2], blk.q[3]);
            else
                store_block<64>(dst + 128, j - 4, blk.q[0], blk.q[1], blk.q[2], blk.q[3]);
            xs[r * RS + ((D == 192 && (mode & 16) && j >= 4) ? j + 2 : j)] = (unsigned char)(blk.e + 127);
        } else
#endif
        if (j < NBLK) {
            f32x8 v[4];
            static_for<4>([&](auto i) {
                v[(int)i] = bf8_to_f32(*(const u32x4 *)(tile + r * XR + 2 * (32 * j + 8 * (int)i)));
            });
#if RC_FAST
            // four partial dots (8-long chains) instead of one 32-long chain
            float pd[4] = {0.f, 0.f, 0.f, 0.f};
            if (mode & 1) {
                static_for<4>([&](auto i) { v[(int)i] = v[(int)i] - kbr[(int)i]; });
            } else if (mode & 2) {
                static_for<4>([&](auto i) {
                    static_for<8>([&](auto e) { pd[(int)i] += v[(int)i][(int)e] * kbr[(int)i][(int)e]; });
                });
            }
            if (mode & 4) {
                const bf16 *yp = Y + b * yb + (long long)(kt * 128 + r) * ys + h * yh + 32 * j;
                static_for<4>([&](auto i) {
                    const f32x8 y = bf8_to_f32(*(const u32x4 *)(yp + 8 * (int)i));
                    static_for<8>([&](auto e) { pd[(int)i] += v[(int)i][(int)e] * y[(int)e]; });
                });
            }
            dot = (pd[0] + pd[1]) + (pd[2] + pd[3]);
#else
            if (mode & 1) {
                static_for<4>([&](auto i) {
                    static_for<8>([&](auto e) { v[(int)i][(int)e] -= kb_lds[32 * j + 8 * (int)i + (int)e]; });
                });
            } else if (mode & 2) {
                static_for<4>([&](auto i) {
                    static_for<8>([&](auto e) { dot += v[(int)i][(int)e] * kb_lds[32 * j + 8 * (int)i + (int)e]; });
                });
            }
            if (mode & 4) {
                const bf16 *yp = Y + b * yb + (long long)(kt * 128 + r) * ys + h * yh + 32 * j;
                static_for<4>([&](auto i) {
                    const f32x8 y = bf8_to_f32(*(const u32x4 *)(yp + 8 * (int)i));
                    static_for<8>([&](auto e) { dot += v[(int)i][(int)e] * y[(int)e]; });
                });
            }
#endif
            if (mode & 32) {
                static_for<4>([&](auto i) { v[(int)i] = v[(int)i] * pre; });
            }
            const Block blk = quant_block(v);
            unsigned char *dst = x8 + r * D;
            if (D == 128 || j < 4)
                store_block<128>(dst, j, blk.q[0], blk.q[1], blk.q[2], blk.q[3]);
            else
                store_block<64>(dst + 128, j - 4, blk.q[0], blk.q[1], blk.q[2], blk.q[3]);
            xs[r * RS + ((D == 192 && (mode & 16) && j >= 4) ? j + 2 : j)] = (unsigned char)(blk.e + 127);
        } else if (j < RS) {
            xs[r * RS + ((mode & 16) ? j - 2 : j)] = (mode & 16) ? 0 : 127;
        }
        if (mode & 6) {
#if RC_FAST
            // xor-1 / xor-2 / xor-4 sums by DPP (quad_perm [1,0,3,2], [2,3,0,1], row_xmask 4): VALU, not ds_bpermute
            dot += __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, dot), 0xB1, 0xF, 0xF, false));
            dot += __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, dot), 0x4E, 0xF, 0xF, false));
            if constexpr (LPR == 8)
                dot += __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, dot), 0x164, 0xF, 0xF, false));
#else
            static_for<LPR == 8 ? 3 : 2>([&](auto k) { dot += __shfl_xor(dot, 1 << (int)k, LPR); });
#endif
            if (j == 0) aux[row] = dot;
        }
    }
    // ---- cols: task = (2 consecutive d, block jb of 32 s)
#if RC_BF16
    if ((mode & 1) == 0) {  // no kbar: the two columns' bf16 pairs (rows 2 k, 2 k + 1) by v_perm, bf16 codes
        for (int task = t; task < (RCT_PROBE == 4 ? 0 : (D / 2) * 4); task += NT) {
            const int dp = task % (D / 2), jb = task / (D / 2), d0 = 2 * dp;
            u32x4 c0[4], c1[4];
            static_for<4>([&](auto ii) {
                static_for<4>([&](auto kk) {
                    constexpr int i = ii, k = kk;
                    const int r = 32 * jb + 8 * i + 2 * k;
                    const uint32_t wa = *(const uint32_t *)(tile + r * XR + 2 * d0);
                    const uint32_t wb = *(const uint32_t *)(tile + (r + 1) * XR + 2 * d0);
                    c0[i][k] = __builtin_amdgcn_perm(wb, wa, 0x05040100u);
                    c1[i][k] = __builtin_amdgcn_perm(wb, wa, 0x07060302u);
                });
            });
            const Block b0 = quant_block_bf16(c0), b1 = quant_block_bf16(c1);
            unsigned char *dst = x8t + d0 * 128;
            store_block<128>(dst, jb, b0.q[0], b0.q[1], b0.q[2], b0.q[3]);
            store_block<128>(dst + 128, jb, b1.q[0], b1.q[1], b1.q[2], b1.q[3]);
            xts[d0 * 4 + jb] = (unsigned char)(b0.e + 127);
            xts[d0 * 4 + 4 + jb] = (unsigned char)(b1.e + 127);
        }
        return;
    }
#endif
    for (int task = t; task < (RCT_PROBE == 4 ? 0 : (D / 2) * 4); task += NT) {  // probe 4: no cols phase
        const int dp = task % (D / 2), jb = task / (D / 2), d0 = 2 * dp;
        const float kb0 = (mode & 1) ? kb_lds[d0] : 0.f, kb1 = (mode & 1) ? kb_lds[d0 + 1] : 0.f;
        f32x8 v0[4], v1[4];
        static_for<4>([&](auto i) {
            static_for<8>([&](auto e) {
                const int r = 32 * jb + 8 * (int)i + (int)e;
                const uint32_t w = *(const uint32_t *)(tile + r * XR + 2 * d0);
                v0[(int)i][(int)e] = __builtin_bit_cast(float, w << 16) - kb0;
                v1[(int)i][(int)e] = __builtin_bit_cast(float, w & 0xFFFF0000u) - kb1;
            });
        });
        const Block b0 = quant_block(v0), b1 = quant_block(v1);
        unsigned char *dst = x8t + d0 * 128;
        store_block<128>(dst, jb, b0.q[0], b0.q[1], b0.q[2], b0.q[3]);
        store_block<128>(dst + 128, jb, b1.q[0], b1.q[1], b1.q[2], b1.q[3]);
        xts[d0 * 4 + jb] = (unsigned char)(b0.e + 127);
        xts[d0 * 4 + 4 + jb] = (unsigned char)(b1.e + 127);
    }
}

// rc_body with the Tensor Data Mover: the bf16 tile by one TDM load (wave 0), the outputs staged in LDS (unpadded: the
// TDM store ignores LDS padding) and written by four TDM stores (X8 rows, XS, X8T, XTS), so a workgroup moves its 48 KB
// in and out as four large transfers instead of thousands of 16-byte accesses (k_mxq_rc: ~6.5 TB/s of ~18).
template <int D>
__device__ __forceinline__ void rct_body(const bf16 *__restrict__ X, unsigned char *__restrict__ X8,
                                         unsigned char *__restrict__ XS, unsigned char *__restrict__ X8T,
                                         unsigned char *__restrict__ XTS, const float *__restrict__ kbar,
                                         const bf16 *__restrict__ Y, float *__restrict__ aux, int S, int H,
                                         long long sb, long long ss, long long sh, long long yb, long long ys,
                                         long long yh, int mode, float pre, unsigned char *smem) {
    constexpr int XR = D * 2;  // bf16 row bytes (unpadded)
    constexpr int NT = 256;
    constexpr int NBLK = D / 32;
    constexpr int LPR = D == 192 ? 8 : 4;
    constexpr int RS = D == 192 ? 8 : 4;
    constexpr int RPP = NT / LPR;
    constexpr int O_X8 = 128 * XR, O_XS = O_X8 + (RC_X8G ? 0 : 128 * D), O_X8T = O_XS + 128 * RS,
                  O_XTS = O_X8T + D * 128, O_KB = O_XTS + D * 4;
    const int kt = blockIdx.x, h = blockIdx.y, b = blockIdx.z, t = threadIdx.x;
    const int wv = __builtin_amdgcn_readfirstlane(t >> 5);
    const long long bh = (long long)b * H + h;
    const unsigned lds0 = (unsigned)(uintptr_t)smem;
    constexpr unsigned W0 = tdm_w0(0, 0, 0);
    if (wv == 0 && RCT_PROBE < 2) {
        const unsigned long long ga = (unsigned long long)(uintptr_t)(X + b * sb + (long long)(kt * 128) * ss + h * sh);
        tdm_load(tdm_g0(lds0, ga), tdm_g1(W0, 128, XR, 128, 2ull * (unsigned long long)ss));
    }
    float *kb_lds = (float *)(smem + O_KB);
    if (mode & 3) {
        for (int i = t; i < D; i += NT) kb_lds[i] = kbar[bh * D + i];
    }
    if (wv == 0) tensor_wait<0>();
    __syncthreads();
#if RCT_PROBE != 1  // timing probes: 1 = no compute (TDM traffic only), 2 = no TDM (compute only)
    rc_tile<D, NT>(smem, RC_X8G ? X8 + (bh * S + (long long)kt * 128) * D : smem + O_X8, smem + O_XS, smem + O_X8T,
                   smem + O_XTS, kb_lds, Y, aux, b, h, kt, bh, S, yb, ys, yh, mode, pre);
#endif
    __syncthreads();
    if (wv == 0 && RCT_PROBE < 2) {
        const long long row0 = bh * S + (long long)kt * 128;
        if (!RC_X8G)
            tdm_store(tdm_g0(lds0 + O_X8, (unsigned long long)(uintptr_t)(X8 + row0 * D)), tdm_g1(W0, 128, D, 128, D));
        tdm_store(tdm_g0(lds0 + O_XS, (unsigned long long)(uintptr_t)(XS + row0 * RS)),
                  tdm_g1(W0, 1, 128 * RS, 1, 128 * RS));
        tdm_store(tdm_g0(lds0 + O_X8T, (unsigned long long)(uintptr_t)(X8T + (bh * D) * S + (long long)kt * 128)),
                  tdm_g1(W0, D, 128, D, (unsigned long long)S));
        tdm_store(tdm_g0(lds0 + O_XTS, (unsigned long long)(uintptr_t)(XTS + ((bh * (S / 128) + kt) * D) * 4)),
                  tdm_g1(W0, 1, D * 4, 1, D * 4));
        tensor_wait<0>();
    }
}

// persistent rows + cols through TDM: workgroup w walks tiles i = w, w + gridDim.x, ... (i: h fastest, then the 128-row
// tile kt, then b: concurrent workgroups read neighbouring heads). Three input buffers (the load of the workgroup's tile
// k + 2 is issued while tile k computes) and two staging sets (tile k's four stores drain while tile k + 1 computes);
// per iteration one barrier pair; the TDM ops of wave 0 in issue order: load(0), 4 NULL, load(1), then per iteration k
// stores(k) x 4, load(k + 2): at the start of iteration k, tensorcnt <= 5 means load(k) and stores(k - 2) are done.
template <int D, int NT>
__device__ __forceinline__ void rcp_body(const bf16 *__restrict__ X, unsigned char *__restrict__ X8,
                                         unsigned char *__restrict__ XS, unsigned char *__restrict__ X8T,
                                         unsigned char *__restrict__ XTS, const float *__restrict__ kbar,
                                         const bf16 *__restrict__ Y, float *__restrict__ aux, int B, int S, int H,
                                         long long sb, long long ss, long long sh, long long yb, long long ys,
                                         long long yh, int mode, float pre, unsigned char *smem) {
    constexpr int XR = D * 2;
    constexpr int RS = D == 192 ? 8 : 4;
    constexpr int IN_B = 128 * XR;
    constexpr int ST_XS = 128 * D, ST_X8T = ST_XS + 128 * RS, ST_XTS = ST_X8T + D * 128, ST_B = ST_XTS + D * 4;
    constexpr int O_ST = 3 * IN_B, O_KB = O_ST + 2 * ST_B;
    constexpr unsigned W0 = tdm_w0(0, 0, 0);
    const int KT = S / 128, ntiles = KT * H * B;
    const int t = threadIdx.x;
    const int wv = __builtin_amdgcn_readfirstlane(t >> 5);
    const unsigned lds0 = (unsigned)(uintptr_t)smem;
    const int w = (int)blockIdx.x, nw = (int)gridDim.x;
    auto load = [&](int k) {
        const int i = w + k * nw;
        const bool real = i < ntiles;
        const int ii = real ? i : 0, h = ii % H, kt = (ii / H) % KT, b = ii / (H * KT);
        const unsigned long long ga = (unsigned long long)(uintptr_t)(X + b * sb + (long long)(kt * 128) * ss + h * sh);
        tdm_load(tdm_g0(lds0 + (unsigned)((k % 3) * IN_B), ga, real ? 1 : 0),
                 tdm_g1(W0, 128, XR, 128, 2ull * (unsigned long long)ss));
    };
    if (wv == 0) {
        load(0);
        static_for<4>([&](auto) { tdm_load(tdm_g0(lds0, 0, 0), tdm_g1(W0, 1, 4, 1, 4)); });
        load(1);
    }
    for (int k = 0; w + k * nw < ntiles; ++k) {
        const int i = w + k * nw;
        const int h = i % H, kt = (i / H) % KT, b = i / (H * KT);
        const long long bh = (long long)b * H + h;
        float *kb_lds = (float *)(smem + O_KB + (k % 2) * D * 4);
        if (mode & 3) {
            for (int e = t; e < D; e += NT) kb_lds[e] = kbar[bh * D + e];
        }
        if (wv == 0) tensor_wait<5>();
        __syncthreads();  // tile k and the kbar slice visible; staging k % 2 free
        unsigned char *st = smem + O_ST + (k % 2) * ST_B;
        rc_tile<D, NT>(smem + (k % 3) * IN_B, st, st + ST_XS, st + ST_X8T, st + ST_XTS, kb_lds, Y, aux, b, h, kt, bh, S,
                       yb, ys, yh, mode, pre);
        __syncthreads();  // staging k % 2 written; input buffer k % 3 consumed
        if (wv == 0) {
            const unsigned sl = lds0 + O_ST + (unsigned)((k % 2) * ST_B);
            const long long row0 = bh * S + (long long)kt * 128;
            tdm_store(tdm_g0(sl, (unsigned long long)(uintptr_t)(X8 + row0 * D)), tdm_g1(W0, 128, D, 128, D));
            tdm_store(tdm_g0(sl + ST_XS, (unsigned long long)(uintptr_t)(XS + row0 * RS)),
                      tdm_g1(W0, 1, 128 * RS, 1, 128 * RS));
            tdm_store(tdm_g0(sl + ST_X8T, (unsigned long long)(uintptr_t)(X8T + (bh * D) * S + (long long)kt * 128)),
                      tdm_g1(W0, D, 128, D, (unsigned long long)S));
            tdm_store(tdm_g0(sl + ST_XTS, (unsigned long long)(uintptr_t)(XTS + ((bh * KT + kt) * D) * 4)),
                      tdm_g1(W0, 1, D * 4, 1, D * 4));
            load(k + 2);
        }
    }
    if (wv == 0) tensor_wait<0>();
}

// ---- MLA RoPE fused into the quantization (DeepSeek's interleaved rotary on d 128..191, Megatron's
// fused_apply_mla_rope_for_q / _for_kv math): for the pe input x (64 values, pairs x1 = x[2 i], x2 = x[2 i + 1], i < 32)
// and the position's cos / sin rows c, sn (64 bf16): out[i] = x1 c[i] - x2 sn[i], out[32 + i] = x2 c[32 + i] + x1 sn[32 +
// i], computed in fp32 (the compiler may fuse a multiply-add: within 1 bf16 ulp of per-op rounding) and rounded to bf16.
__device__ __forceinline__ void rope8(const u32x4 x0, const u32x4 x1, const u32x4 cl, const u32x4 sl, const u32x4 cr,
                                      const u32x4 sr, u32x4 &left, u32x4 &right) {
    // x0, x1: pe values 16 m .. 16 m + 15 (pairs 8 m .. 8 m + 7); cl / sl: cos / sin 8 m .. +7; cr / sr: 32 + 8 m .. +7
    const bf16x8 xa = __builtin_bit_cast(bf16x8, x0), xb = __builtin_bit_cast(bf16x8, x1);
    const bf16x8 c0 = __builtin_bit_cast(bf16x8, cl), s0 = __builtin_bit_cast(bf16x8, sl);
    const bf16x8 c1 = __builtin_bit_cast(bf16x8, cr), s1 = __builtin_bit_cast(bf16x8, sr);
    bf16x8 lo, hi;
    static_for<8>([&](auto kk) {
        constexpr int k = kk;
        const float a = (float)(k < 4 ? xa[2 * k] : xb[2 * (k - 4)]);
        const float b = (float)(k < 4 ? xa[2 * k + 1] : xb[2 * (k - 4) + 1]);
        lo[k] = (bf16)__fsub_rn(__fmul_rn(a, (float)c0[k]), __fmul_rn(b, (float)s0[k]));
        hi[k] = (bf16)__fadd_rn(__fmul_rn(b, (float)c1[k]), __fmul_rn(a, (float)s1[k]));
    });
    left = __builtin_bit_cast(u32x4, lo);
    right = __builtin_bit_cast(u32x4, hi);
}

// rct_body for D = 192 with the pe columns roped on the way in: the nope part (d 0..127) of X by one TDM load into
// 384-byte LDS rows (pad 128 B after every 256), the pre-RoPE pe part (64 values per row) of PE (q: X + 128; k: k_pe,
// shared by the heads: ph = 0) by a second TDM load into a side buffer; the rotation writes d 128..191 of the rows.
__device__ __forceinline__ void rctr_body(const bf16 *__restrict__ X, unsigned char *__restrict__ X8,
                                          unsigned char *__restrict__ XS, unsigned char *__restrict__ X8T,
                                          unsigned char *__restrict__ XTS, const float *__restrict__ kbar,
                                          float *__restrict__ aux, int S, int H, long long sb, long long ss,
                                          long long sh, const bf16 *__restrict__ PE, long long pb, long long ps,
                                          long long ph, const bf16 *__restrict__ COS, const bf16 *__restrict__ SIN,
                                          int mode, float pre, unsigned char *smem) {
    constexpr int D = 192, XR = D * 2, NT = 256, RS = 8;
    constexpr int O_X8 = 128 * XR, O_XS = O_X8 + (RC_X8G ? 0 : 128 * D), O_X8T = O_XS + 128 * RS,
                  O_XTS = O_X8T + D * 128, O_KB = O_XTS + D * 4,
                  O_PE = O_X8T;  // pe side buffer (128 x 128 B) in the X8T staging, written only in the cols phase
    const int kt = blockIdx.x, h = blockIdx.y, b = blockIdx.z, t = threadIdx.x;
    const int wv = __builtin_amdgcn_readfirstlane(t >> 5);
    const long long bh = (long long)b * H + h;
    const unsigned lds0 = (unsigned)(uintptr_t)smem;
    if (wv == 0) {
        const unsigned long long gx = (unsigned long long)(uintptr_t)(X + b * sb + (long long)(kt * 128) * ss + h * sh);
        tdm_load(tdm_g0(lds0, gx), tdm_g1(tdm_w0(0, 64, 32), 128, 256, 128, 2ull * (unsigned long long)ss));
        const unsigned long long gp = (unsigned long long)(uintptr_t)(PE + b * pb + (long long)(kt * 128) * ps + h * ph);
        tdm_load(tdm_g0(lds0 + O_PE, gp), tdm_g1(tdm_w0(0, 0, 0), 128, 128, 128, 2ull * (unsigned long long)ps));
    }
    float *kb_lds = (float *)(smem + O_KB);
    if (mode & 3) {
        for (int i = t; i < D; i += NT) kb_lds[i] = kbar[bh * D + i];
    }
    if (wv == 0) tensor_wait<0>();
    __syncthreads();
    // rotation: item (row r, group m of 8 pairs): 512 items, 2 per thread
    for (int it = t; it < 128 * 4; it += NT) {
        const int r = it >> 2, m = it & 3;
        const unsigned char *pe = smem + O_PE + r * 128 + 32 * m;
        const bf16 *cs = COS + (long long)(kt * 128 + r) * 64, *sn = SIN + (long long)(kt * 128 + r) * 64;
        u32x4 lo, hi;
        rope8(*(const u32x4 *)pe, *(const u32x4 *)(pe + 16), *(const u32x4 *)(cs + 8 * m), *(const u32x4 *)(sn + 8 * m),
              *(const u32x4 *)(cs + 32 + 8 * m), *(const u32x4 *)(sn + 32 + 8 * m), lo, hi);
        *(u32x4 *)(smem + r * XR + 2 * (128 + 8 * m)) = lo;
        *(u32x4 *)(smem + r * XR + 2 * (160 + 8 * m)) = hi;
    }
    __syncthreads();
    rc_tile<D, NT>(smem, RC_X8G ? X8 + (bh * S + (long long)kt * 128) * D : smem + O_X8, smem + O_XS, smem + O_X8T,
                   smem + O_XTS, kb_lds, nullptr, aux, b, h, kt, bh, S, 0, 0, 0, mode & ~4, pre);
    __syncthreads();
    if (wv == 0) {
        constexpr unsigned W0 = tdm_w0(0, 0, 0);
        const long long row0 = bh * S + (long long)kt * 128;
        if (!RC_X8G)
            tdm_store(tdm_g0(lds0 + O_X8, (unsigned long long)(uintptr_t)(X8 + row0 * D)), tdm_g1(W0, 128, D, 128, D));
        tdm_store(tdm_g0(lds0 + O_XS, (unsigned long long)(uintptr_t)(XS + row0 * RS)),
                  tdm_g1(W0, 1, 128 * RS, 1, 128 * RS));
        tdm_store(tdm_g0(lds0 + O_X8T, (unsigned long long)(uintptr_t)(X8T + (bh * D) * S + (long long)kt * 128)),
                  tdm_g1(W0, D, 128, D, (unsigned long long)S));
        tdm_store(tdm_g0(lds0 + O_XTS, (unsigned long long)(uintptr_t)(XTS + ((bh * (S / 128) + kt) * D) * 4)),
                  tdm_g1(W0, 1, D * 4, 1, D * 4));
        tensor_wait<0>();
    }
}
}  // namespace

// k_mxq_rctr192: rows + cols of [X nope | rope(PE)] in one pass (q: PE = X + 128; k: X = kv, PE = k_pe, ph = 0); grid
// (S / 128, H, B), 256 threads
#ifndef RCTR_WPE
#define RCTR_WPE (RC_X8G ? 8 : 6)  // as RCT_WPE192 (LDS 77 KB with RC_X8G: 4 workgroups per CU)
#endif
extern "C" __global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(RCTR_WPE))) void k_mxq_rctr192(
    const bf16 *__restrict__ X, unsigned char *__restrict__ X8, unsigned char *__restrict__ XS,
    unsigned char *__restrict__ X8T, unsigned char *__restrict__ XTS, const float *__restrict__ kbar,
    float *__restrict__ aux, int S, int H, long long sb, long long ss, long long sh, const bf16 *__restrict__ PE,
    long long pb, long long ps, long long ph, const bf16 *__restrict__ COS, const bf16 *__restrict__ SIN, int mode,
    float pre) {
    __shared__ __attribute__((aligned(1024))) unsigned char smem[128 * 384 + (RC_X8G ? 0 : 128 * 192) + 128 * 8 +
                                                                 192 * 128 + 192 * 4 + 192 * 4];
    rctr_body(X, X8, XS, X8T, XTS, kbar, aux, S, H, sb, ss, sh, PE, pb, ps, ph, COS, SIN, mode, pre, smem);
}

// ---- the RoPE backward of the fused path (Megatron's rotary_bwd_q / _kv math): for the gradient g of the roped pe
// values (halves gl = g[0..31], gr = g[32..63]) the input gradient x1[i] = gl c[i] + gr sn[32 + i] (to pe[2 i]), x2[i] =
// -gl sn[i] + gr c[32 + i] (to pe[2 i + 1]), fp32 per op, bf16 out.
__device__ __forceinline__ void irope8(const f32x8 &gl, const f32x8 &gr, const u32x4 cl, const u32x4 sl, const u32x4 cr,
                                       const u32x4 sr, u32x4 &o0, u32x4 &o1) {
    const bf16x8 c0 = __builtin_bit_cast(bf16x8, cl), s0 = __builtin_bit_cast(bf16x8, sl);
    const bf16x8 c1 = __builtin_bit_cast(bf16x8, cr), s1 = __builtin_bit_cast(bf16x8, sr);
    bf16x8 a, b;
    static_for<8>([&](auto kk) {
        constexpr int k = kk;
        const bf16 x1 = (bf16)__fadd_rn(__fmul_rn(gl[k], (float)c0[k]), __fmul_rn(gr[k], (float)s1[k]));
        const bf16 x2 = (bf16)__fsub_rn(__fmul_rn(gr[k], (float)c1[k]), __fmul_rn(gl[k], (float)s0[k]));
        if constexpr (k < 4) {
            a[2 * k] = x1;
            a[2 * k + 1] = x2;
        } else {
            b[2 * (k - 4)] = x1;
            b[2 * (k - 4) + 1] = x2;
        }
    });
    o0 = __builtin_bit_cast(u32x4, a);
    o1 = __builtin_bit_cast(u32x4, b);
}

// k_rope_bwd_q: in place on G [b, s, h, 192] (d 128..191: the roped gradient -> the pre-RoPE one); 4 lanes per row
// (the lanes of a row in one wave: all of its loads before its stores); grid (B S H / 64), 256 threads
extern "C" __global__ __launch_bounds__(256) void k_rope_bwd_q(bf16 *__restrict__ G, int B, int S, int H, long long gb,
                                                               long long gs, long long gh,
                                                               const bf16 *__restrict__ COS,
                                                               const bf16 *__restrict__ SIN) {
    const long long row = (long long)blockIdx.x * 64 + (threadIdx.x >> 2);
    const int m = threadIdx.x & 3;
    if (row >= (long long)B * S * H) return;
    const int h = (int)(row % H), s = (int)((row / H) % S), b = (int)(row / ((long long)H * S));
    bf16 *g = G + b * gb + (long long)s * gs + h * gh + 128;
    const bf16 *cs = COS + (long long)s * 64, *sn = SIN + (long long)s * 64;
    const f32x8 gl = bf8_to_f32(*(const u32x4 *)(g + 8 * m)), gr = bf8_to_f32(*(const u32x4 *)(g + 32 + 8 * m));
    u32x4 o0, o1;
    irope8(gl, gr, *(const u32x4 *)(cs + 8 * m), *(const u32x4 *)(sn + 8 * m), *(const u32x4 *)(cs + 32 + 8 * m),
           *(const u32x4 *)(sn + 32 + 8 * m), o0, o1);
    *(u32x4 *)(g + 16 * m) = o0;
    *(u32x4 *)(g + 16 * m + 8) = o1;
}

// k_rope_bwd_kpe: GP [b, s, 64] = the pre-RoPE gradient of k_pe from the heads' roped gradients GH [b, s, h, 64] (fp32
// sum over h in a fixed order, then the inverse rotation); grid (B S), 256 threads: thread = (head slice hs of 32, value
// group of 8), 16-byte loads of heads hs, hs + 32, .. issued together, then the 32 slices summed through LDS
extern "C" __global__ __launch_bounds__(256) void k_rope_bwd_kpe(const bf16 *__restrict__ GH, bf16 *__restrict__ GP,
                                                                 int B, int S, int H, long long hb, long long hs_,
                                                                 long long hh, long long pb, long long ps,
                                                                 const bf16 *__restrict__ COS,
                                                                 const bf16 *__restrict__ SIN) {
    __shared__ float part[32][64 + 4];
    __shared__ float tot[64];
    const int bs = blockIdx.x, b = bs / S, s = bs % S;
    const int t = threadIdx.x, vg = t & 7, hsl = t >> 3;
    const bf16 *gh = GH + b * hb + (long long)s * hs_ + 8 * vg;
    f32x8 acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    int h = hsl;
    for (; h + 96 < H; h += 128) {  // 4 heads at a time
        u32x4 w[4];
        static_for<4>([&](auto u) { w[(int)u] = *(const u32x4 *)(gh + (long long)(h + 32 * (int)u) * hh); });
        static_for<4>([&](auto u) { acc = acc + bf8_to_f32(w[(int)u]); });
    }
    for (; h < H; h += 32) acc = acc + bf8_to_f32(*(const u32x4 *)(gh + (long long)h * hh));
    static_for<8>([&](auto e) { part[hsl][8 * vg + (int)e] = acc[(int)e]; });
    __syncthreads();
    if (t < 64) {
        float sum = 0.f;
        for (int k = 0; k < 32; ++k) sum += part[k][t];  // fixed order: deterministic
        tot[t] = sum;
    }
    __syncthreads();
    if (t < 4) {
        const int m = t;  // pairs 8 m .. 8 m + 7
        f32x8 gl, gr;
        static_for<8>([&](auto k) {
            gl[(int)k] = tot[8 * m + (int)k];
            gr[(int)k] = tot[32 + 8 * m + (int)k];
        });
        const bf16 *cs = COS + (long long)s * 64, *sn = SIN + (long long)s * 64;
        u32x4 o0, o1;
        irope8(gl, gr, *(const u32x4 *)(cs + 8 * m), *(const u32x4 *)(sn + 8 * m), *(const u32x4 *)(cs + 32 + 8 * m),
               *(const u32x4 *)(sn + 32 + 8 * m), o0, o1);
        bf16 *gp = GP + b * pb + (long long)s * ps + 16 * m;
        *(u32x4 *)gp = o0;
        *(u32x4 *)(gp + 8) = o1;
    }
}

// k_mxq_kbar_rope: kbar of k = [X nope | rope(PE)] (as k_mxq_kbar; the roped pe values rounded to bf16 as quantized)
extern "C" __global__ __launch_bounds__(768) void k_mxq_kbar_rope(const bf16 *__restrict__ X, float *__restrict__ kbar,
                                                                  int S, int H, long long sb, long long ss,
                                                                  long long sh, const bf16 *__restrict__ PE,
                                                                  long long pb, long long ps, long long ph,
                                                                  const bf16 *__restrict__ COS,
                                                                  const bf16 *__restrict__ SIN, int stride) {
    constexpr int D = 192;
    __shared__ float part[32][D + 4];
    const int bh = blockIdx.x, b = bh / H, h = bh % H;
    const int t = threadIdx.x, cg = t % 24, rg = t / 24;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int s = rg * stride; s < S; s += 32 * stride) {
        f32x8 v;
        if (cg < 16) {
            v = bf8_to_f32(*(const u32x4 *)(X + b * sb + (long long)s * ss + h * sh + 8 * cg));
        } else {
            const int m = (cg - 16) & 3;
            const bf16 *pe = PE + b * pb + (long long)s * ps + h * ph + 16 * m;
            const bf16 *cs = COS + (long long)s * 64, *sn = SIN + (long long)s * 64;
            u32x4 lo, hi;
            rope8(*(const u32x4 *)pe, *(const u32x4 *)(pe + 8), *(const u32x4 *)(cs + 8 * m), *(const u32x4 *)(sn + 8 * m),
                  *(const u32x4 *)(cs + 32 + 8 * m), *(const u32x4 *)(sn + 32 + 8 * m), lo, hi);
            v = bf8_to_f32(cg < 20 ? lo : hi);
        }
        static_for<8>([&](auto i) { acc[(int)i] += v[(int)i]; });
    }
    static_for<8>([&](auto i) { part[rg][8 * cg + (int)i] = acc[(int)i]; });
    __syncthreads();
    if (t < D) {
        float sum = 0.f;
        for (int g = 0; g < 32; ++g) sum += part[g][t];
        kbar[(long long)bh * D + t] = sum / (float)((S + stride - 1) / stride);
    }
}

// grid (B * H), block 768 = 24 column groups of 8 d x 32 row groups. The mean of every stride-th row: any constant
// shift of k is softmax-invariant, so an approximate mean centres k as well (and the backward uses the same kbar).
extern "C" __global__ __launch_bounds__(768) void k_mxq_kbar(const bf16 *__restrict__ K, float *__restrict__ kbar,
                                                             int S, int H, long long sb, long long ss, long long sh,
                                                             int stride) {
    constexpr int D = 192;
    __shared__ float part[32][D + 4];
    const int bh = blockIdx.x, b = bh / H, h = bh % H;
    const int t = threadIdx.x, cg = t % 24, rg = t / 24;
    const bf16 *base = K + b * sb + h * sh + 8 * cg;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#if RC_FAST
    // the row loads of a thread issued together (8 at a time: one latency instead of eight), summed in the same order
    int s = rg * stride;
    for (; s + 7 * 32 * stride < S; s += 8 * 32 * stride) {
        u32x4 w[8];
        static_for<8>([&](auto u) { w[(int)u] = *(const u32x4 *)(base + (long long)(s + (int)u * 32 * stride) * ss); });
        static_for<8>([&](auto u) {
            const f32x8 v = bf8_to_f32(w[(int)u]);
            static_for<8>([&](auto i) { acc[(int)i] += v[(int)i]; });
        });
    }
    for (; s < S; s += 32 * stride) {
        const f32x8 v = bf8_to_f32(*(const u32x4 *)(base + (long long)s * ss));
        static_for<8>([&](auto i) { acc[(int)i] += v[(int)i]; });
    }
#else
    for (int s = rg * stride; s < S; s += 32 * stride) {
        const f32x8 v = bf8_to_f32(*(const u32x4 *)(base + (long long)s * ss));
        static_for<8>([&](auto i) { acc[(int)i] += v[(int)i]; });
    }
#endif
    static_for<8>([&](auto i) { part[rg][8 * cg + (int)i] = acc[(int)i]; });
    __syncthreads();
    if (t < D) {
        float sum = 0.f;
        for (int g = 0; g < 32; ++g) sum += part[g][t];  // fixed order: deterministic
        kbar[(long long)bh * D + t] = sum / (float)((S + stride - 1) / stride);
    }
}

#define ROWS_KERNEL(D)                                                                                               \
    extern "C" __global__ __launch_bounds__(256) void k_mxq_rows##D(                                                 \
        const bf16 *__restrict__ X, unsigned char *__restrict__ X8, unsigned char *__restrict__ XS,                  \
        const float *__restrict__ kbar, const bf16 *__restrict__ Y, float *__restrict__ aux, bf16 *__restrict__ XD, \
        int B, int S, int H, long long sb, long long ss, long long sh, long long yb, long long ys, long long yh,     \
        int mode, float pre) {                                                                                       \
        rows_body<D>(X, X8, XS, kbar, Y, aux, XD, B, S, H, sb, ss, sh, yb, ys, yh, mode, pre);                       \
    }
ROWS_KERNEL(192)
ROWS_KERNEL(128)

#define COLS_KERNEL(D)                                                                                               \
    extern "C" __global__ __launch_bounds__(256) void k_mxq_cols##D(                                                 \
        const bf16 *__restrict__ X, unsigned char *__restrict__ X8T, unsigned char *__restrict__ XTS,                \
        const float *__restrict__ kbar, int S, int H, long long sb, long long ss, long long sh, int mode) {          \
        __shared__ __attribute__((aligned(16))) unsigned char tile[128 * (D * 2 + 16)];                              \
        cols_body<D>(X, X8T, XTS, kbar, S, H, sb, ss, sh, mode, tile);                                               \
    }
COLS_KERNEL(128)
COLS_KERNEL(192)

// rows + cols in one pass: grid (S / 128, H, B), 256 threads (rc_body)
#define RC_KERNEL(D)                                                                                                 \
    extern "C" __global__ __launch_bounds__(256) void k_mxq_rc##D(                                                   \
        const bf16 *__restrict__ X, unsigned char *__restrict__ X8, unsigned char *__restrict__ XS,                  \
        unsigned char *__restrict__ X8T, unsigned char *__restrict__ XTS, const float *__restrict__ kbar,            \
        const bf16 *__restrict__ Y, float *__restrict__ aux, int S, int H, long long sb, long long ss, long long sh, \
        long long yb, long long ys, long long yh, int mode, float pre) {                                             \
        __shared__ __attribute__((aligned(16))) unsigned char tile[128 * (D * 2 + 16)];                              \
        __shared__ float kb_lds[D];                                                                                  \
        rc_body<D>(X, X8, XS, X8T, XTS, kbar, Y, aux, S, H, sb, ss, sh, yb, ys, yh, mode, pre, tile, kb_lds);        \
    }
RC_KERNEL(192)
RC_KERNEL(128)

// rows + cols in one pass through TDM: grid (S / 128, H, B), 256 threads (rct_body); LDS 101 KB (D 192) / 67 KB
#ifndef RCT_WPE192
#define RCT_WPE192 (RC_X8G ? 8 : 6)  // waves per EU the register allocation targets (LDS: 4 / 3 workgroups per CU)
#endif
#ifndef RCT_WPE128
#define RCT_WPE128 8  // 4 workgroups per CU
#endif
#define RCT_KERNEL(D)                                                                                                \
    extern "C" __global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(RCT_WPE##D)))                    \
    void k_mxq_rct##D(                                                                                               \
        const bf16 *__restrict__ X, unsigned char *__restrict__ X8, unsigned char *__restrict__ XS,                  \
        unsigned char *__restrict__ X8T, unsigned char *__restrict__ XTS, const float *__restrict__ kbar,            \
        const bf16 *__restrict__ Y, float *__restrict__ aux, int S, int H, long long sb, long long ss, long long sh, \
        long long yb, long long ys, long long yh, int mode, float pre) {                                             \
        __shared__ __attribute__((aligned(1024))) unsigned char smem[128 * D * 2 + (RC_X8G ? 0 : 128 * D) + 128 * 8 + D * 128 + \
                                                                     D * 4 + D * 4];                                 \
        rct_body<D>(X, X8, XS, X8T, XTS, kbar, Y, aux, S, H, sb, ss, sh, yb, ys, yh, mode, pre, smem);               \
    }
RCT_KERNEL(192)
RCT_KERNEL(128)

// persistent rows + cols through TDM: grid (number of workgroups, 1, 1), 512 threads (rcp_body); LDS 245 KB (D 192)
#define RCP_KERNEL(D)                                                                                                \
    extern "C" __global__ __launch_bounds__(512) void k_mxq_rcp##D(                                                  \
        const bf16 *__restrict__ X, unsigned char *__restrict__ X8, unsigned char *__restrict__ XS,                  \
        unsigned char *__restrict__ X8T, unsigned char *__restrict__ XTS, const float *__restrict__ kbar,            \
        const bf16 *__restrict__ Y, float *__restrict__ aux, int B, int S, int H, long long sb, long long ss,        \
        long long sh, long long yb, long long ys, long long yh, int mode, float pre) {                               \
        __shared__ __attribute__((aligned(1024))) unsigned char smem[3 * 128 * D * 2 +                               \
                                                                     2 * (128 * D + 128 * 8 + D * 128 + D * 4) +     \
                                                                     2 * D * 4];                                     \
        rcp_body<D, 512>(X, X8, XS, X8T, XTS, kbar, Y, aux, B, S, H, sb, ss, sh, yb, ys, yh, mode, pre, smem);      \
    }
RCP_KERNEL(192)
RCP_KERNEL(128)
