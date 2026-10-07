// src/kernels/cuda/verify_kernels.cu - see include/strata/kernels/verify_kernels.hpp.
//
// The per-token arithmetic of every kernel here is transcribed from its single-token original (fused_gdn.cu,
// elementwise.cu) with the same operation order, so a verify window reproduces plain decode bit for bit.
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/dp4a.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int S = 128;          // GDN state size
constexpr int RG = 4;
constexpr int RPG = S / RG;

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

__global__ void __launch_bounds__(S) gdn_conv_l2_multi_kernel(const float* __restrict__ hist,
                                                              const float* __restrict__ qkv,
                                                              const float* __restrict__ w, float* __restrict__ h,
                                                              int C, int qk_heads, float eps, int t_begin) {
    __shared__ float part[S / 32];
    const int t = t_begin + blockIdx.y;
    const int c = blockIdx.x * S + threadIdx.x;
    // the window of token t: [hist0, hist1, hist2, x_0, ..., x_t], its last four entries
    float win[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int src = t + j;          // index into [hist(3) | x...]
        win[j] = src < 3 ? hist[c * 3 + src] : qkv[(size_t) (src - 3) * C + c];
    }
    const float v0 = win[0], v1 = win[1], v2 = win[2], x = qkv[(size_t) t * C + c];
    float sum = v0 * w[c * 4] + v1 * w[c * 4 + 1] + v2 * w[c * 4 + 2] + x * w[c * 4 + 3];
    float y = sum / (1.0f + __expf(-sum));
    if ((int) blockIdx.x < qk_heads) {
        float sq = y * y;
        for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
        if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
        __syncthreads();
        const float ss = part[0] + part[1] + part[2] + part[3];
        y *= rsqrtf(ss + eps);
    }
    h[(size_t) t * C + c] = y;
}

__global__ void gdn_conv_commit_kernel(float* __restrict__ hist, const float* __restrict__ qkv, int C,
                                       const int32_t* __restrict__ n_keep) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int n = *n_keep;
    if (n <= 0) return;
    float seq[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int src = n + j;          // the last three of [hist(3) | x_0..x_{n-1}]
        seq[j] = src < 3 ? hist[c * 3 + src] : qkv[(size_t) (src - 3) * C + c];
    }
    hist[c * 3] = seq[0];
    hist[c * 3 + 1] = seq[1];
    hist[c * 3 + 2] = seq[2];
}

// 8 weights of row `base` (16-byte chunk j of a 16-bit row; 32 bytes of an F32 one), widened to FP32
template <int WF>
__device__ __forceinline__ void wload8(const void* base, int j, float* w) {
    if constexpr (WF == 2) {
        const float4* p = reinterpret_cast<const float4*>(base) + 2 * j;
        const float4 a = __ldg(p), b = __ldg(p + 1);
        w[0] = a.x; w[1] = a.y; w[2] = a.z; w[3] = a.w; w[4] = b.x; w[5] = b.y; w[6] = b.z; w[7] = b.w;
    } else {
        const uint4 v = __ldg(reinterpret_cast<const uint4*>(base) + j);
        const uint32_t u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            if constexpr (WF == 0) { w[2 * k] = __uint_as_float(u[k] << 16); w[2 * k + 1] = __uint_as_float(u[k] & 0xffff0000u); }
            else { w[2 * k] = __half2float(__ushort_as_half((uint16_t) u[k])); w[2 * k + 1] = __half2float(__ushort_as_half((uint16_t) (u[k] >> 16))); }
        }
    }
}

template <int WF>
__global__ void __launch_bounds__(256) gdn_ab_multi_kernel(const float* __restrict__ x, const void* __restrict__ wa,
                                                           const void* __restrict__ wb,
                                                           const float* __restrict__ dt,
                                                           const float* __restrict__ ssm_a, float* __restrict__ gate,
                                                           float* __restrict__ beta, int n, int h_v, int T) {
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= 2 * h_v) return;
    const bool is_beta = row >= h_v;
    const int r = is_beta ? row - h_v : row;
    const void* wrow = reinterpret_cast<const char*>(is_beta ? wb : wa) + (size_t) r * n * (WF == 2 ? 4 : 2);
    float acc[kVerifyMaxT];
#pragma unroll
    for (int t = 0; t < kVerifyMaxT; ++t) acc[t] = 0.0f;
    for (int j = lane; j < n / 8; j += 32) {
        float w[8];
        wload8<WF>(wrow, j, w);
#pragma unroll
        for (int t = 0; t < kVerifyMaxT; ++t) {
            if (t >= T) break;
            const float* xt = x + (size_t) t * n;
            const float4 xa = *reinterpret_cast<const float4*>(xt + j * 8);
            const float4 xb = *reinterpret_cast<const float4*>(xt + j * 8 + 4);
            float a = acc[t];
            a = fmaf(w[0], xa.x, a); a = fmaf(w[1], xa.y, a);
            a = fmaf(w[2], xa.z, a); a = fmaf(w[3], xa.w, a);
            a = fmaf(w[4], xb.x, a); a = fmaf(w[5], xb.y, a);
            a = fmaf(w[6], xb.z, a); a = fmaf(w[7], xb.w, a);
            acc[t] = a;
        }
    }
#pragma unroll
    for (int t = 0; t < kVerifyMaxT; ++t) {
        if (t >= T) break;
        float a = acc[t];
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (lane != 0) continue;
        if (is_beta) {
            beta[(size_t) t * h_v + r] = 1.0f / (1.0f + __expf(-a));
        } else {
            const float v = a + dt[r];
            const float sp = v > 20.0f ? v : log1pf(__expf(v));
            gate[(size_t) t * h_v + r] = sp * ssm_a[r];
        }
    }
}

// gdn_ab_multi_kernel for TT tokens and n = 2560 (10 chunks per lane): the loops have fixed trip counts, so every
// chunk's weight and activation loads can be issued up front; the same FMAs in the same order
template <int WF, int TT>
__global__ void __launch_bounds__(256) gdn_ab_multi_t_kernel(const float* __restrict__ x, const void* __restrict__ wa,
                                                             const void* __restrict__ wb, const float* __restrict__ dt,
                                                             const float* __restrict__ ssm_a, float* __restrict__ gate,
                                                             float* __restrict__ beta, int h_v) {
    constexpr int n = 2560, CH = n / 8 / 32;
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= 2 * h_v) return;
    const bool is_beta = row >= h_v;
    const int r = is_beta ? row - h_v : row;
    const void* wrow = reinterpret_cast<const char*>(is_beta ? wb : wa) + (size_t) r * n * (WF == 2 ? 4 : 2);
    float acc[TT];
#pragma unroll
    for (int t = 0; t < TT; ++t) acc[t] = 0.0f;
#pragma unroll
    for (int it = 0; it < CH; ++it) {
        const int j = lane + 32 * it;
        float w[8];
        wload8<WF>(wrow, j, w);
#pragma unroll
        for (int t = 0; t < TT; ++t) {
            const float* xt = x + (size_t) t * n;
            const float4 xa = *reinterpret_cast<const float4*>(xt + j * 8);
            const float4 xb = *reinterpret_cast<const float4*>(xt + j * 8 + 4);
            float a = acc[t];
            a = fmaf(w[0], xa.x, a); a = fmaf(w[1], xa.y, a);
            a = fmaf(w[2], xa.z, a); a = fmaf(w[3], xa.w, a);
            a = fmaf(w[4], xb.x, a); a = fmaf(w[5], xb.y, a);
            a = fmaf(w[6], xb.z, a); a = fmaf(w[7], xb.w, a);
            acc[t] = a;
        }
    }
#pragma unroll
    for (int t = 0; t < TT; ++t) {
        float a = acc[t];
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (lane != 0) continue;
        if (is_beta) {
            beta[(size_t) t * h_v + r] = 1.0f / (1.0f + __expf(-a));
        } else {
            const float v = a + dt[r];
            const float sp = v > 20.0f ? v : log1pf(__expf(v));
            gate[(size_t) t * h_v + r] = sp * ssm_a[r];
        }
    }
}

// State-only commit kernel: each head's 128 independent state columns are split across 4 blocks of 32 columns
// (48 * 4 = 192 blocks of 128 threads across all SMs, vs 48 blocks of 512 threads), with no sq/o/norm/z/y work.
__global__ void __launch_bounds__(32 * RG) gdn_step_commit_kernel(float* __restrict__ state,
                                                                  const float* __restrict__ hbuf, int C,
                                                                  const float* __restrict__ gate,
                                                                  const float* __restrict__ beta,
                                                                  int h_k, int h_v,
                                                                  const int32_t* __restrict__ n_keep) {
    const int n = *n_keep;
    if (n <= 0) return;
    __shared__ float sk[S];
    __shared__ float red[RG][32];
    const int head = blockIdx.x;
    const int col_local = threadIdx.x;
    const int col = blockIdx.y * 32 + col_local;
    const int rg = threadIdx.y;
    const int tid = rg * 32 + col_local;    // 0 .. 127
    const int qh = head % h_k;
    const int qk = S * h_k;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
    for (int t = 0; t < n; ++t) {
        const float* ht = hbuf + (size_t) t * C;
        __syncthreads();
        sk[tid] = ht[qk + qh * S + tid];
        __syncthreads();
        const float g = __expf(gate[(size_t) t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
        red[rg][col_local] = kv;
        __syncthreads();
        const float kv_col = red[0][col_local] + red[1][col_local] + red[2][col_local] + red[3][col_local];
        const float delta = (ht[2 * qk + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
#pragma unroll
        for (int r = 0; r < RPG; ++r) s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
}

// S26 (STRATA_QFUSE=1): the q8_1 image of a 32-column block, written by the warp that holds those columns with
// native_quantize_q8_1_kernel's quantizer (the same XOR-tree max and sum, d = amax / 127, roundf(x / d), ds = (d, sum)).
struct GdnQ81 { half2 ds; int8_t qs[32]; };
__device__ __forceinline__ void gdn_q8_1_store(GdnQ81* __restrict__ xq, size_t idx, float xi) {
    // no contraction: xi is a product here, and fma(a, b, shfl) would round the first sum unlike the separate
    // quantizer, which reads xi from memory (S26 harness: ds.sum differed by 1 ulp without this)
#pragma clang fp contract(off)
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    const float d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
    xq[idx / 32].qs[idx % 32] = q;
    if (idx % 32 == 0) xq[idx / 32].ds = make_half2(d, sum);
}

template<bool Q>
__global__ void __launch_bounds__(S * RG) gdn_step_norm_multi_kernel(float* __restrict__ state,
                                                                     const float* __restrict__ hbuf, int C,
                                                                     const float* __restrict__ gate,
                                                                     const float* __restrict__ beta,
                                                                     const float* __restrict__ z,
                                                                     const float* __restrict__ gamma, float eps,
                                                                     float* __restrict__ y, int h_k, int h_v, int T,
                                                                     const int32_t* __restrict__ n_keep, int t_out_begin,
                                                                     GdnQ81* __restrict__ xq) {
    __shared__ float sk[S], sq[S];
    __shared__ float red[RG][S];
    __shared__ float wsum[S * RG / 32];
    const int head = blockIdx.x;
    const int col = threadIdx.x;
    const int rg = threadIdx.y;
    const int tid = rg * S + col;
    const int qh = head % h_k;
    const int qk = S * h_k;             // q at [0, qk), k at [qk, 2qk), v at [2qk, ...)
    const int value_dim = S * h_v;
    const int n = n_keep ? *n_keep : T;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
    for (int t = 0; t < n; ++t) {
        const float* ht = hbuf + (size_t) t * C;
        const bool need_out = (t >= t_out_begin);
        __syncthreads();                // the previous token is done with sk/sq/red/wsum
        if (tid < S) {
            sk[tid] = ht[qk + qh * S + tid];
            if (need_out) sq[tid] = ht[qh * S + tid];
        }
        __syncthreads();
        const float g = __expf(gate[(size_t) t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
        red[rg][col] = kv;
        __syncthreads();
        const float kv_col = red[0][col] + red[1][col] + red[2][col] + red[3][col];
        const float delta = (ht[2 * qk + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
        if (!need_out) {
#pragma unroll
            for (int r = 0; r < RPG; ++r) s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
            continue;
        }
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
            o = fmaf(s[r], sq[rg * RPG + r], o);
        }
        __syncthreads();
        red[rg][col] = o;
        __syncthreads();
        float oc = 0.0f, sq_part = 0.0f;
        if (rg == 0) {
            oc = (red[0][col] + red[1][col] + red[2][col] + red[3][col]) * rsqrtf((float) S);
            sq_part = oc * oc;
        }
        for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part += __shfl_xor_sync(0xffffffffu, sq_part, o2);
        if ((tid & 31) == 0) wsum[tid >> 5] = sq_part;
        __syncthreads();
        if (rg == 0) {
            const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            const float scale = rsqrtf(ss / (float) S + eps);
            const float zz = z[(size_t) t * value_dim + head * S + col];
            if constexpr (Q) {
                const float yv = oc * scale * gamma[col] * (1.0f / (1.0f + __expf(-zz)));
                y[(size_t) t * value_dim + head * S + col] = yv;
                gdn_q8_1_store(xq, (size_t) (t - t_out_begin) * value_dim + head * S + col, yv);
            } else {
                y[(size_t) t * value_dim + head * S + col] = oc * scale * gamma[col] * (1.0f / (1.0f + __expf(-zz)));
            }
        }
    }
    if (n_keep != nullptr && n > 0) {
#pragma unroll
        for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
    }
}

// S25 (STRATA_GDN_SPLIT=1): the same recurrence with each head's 128 state columns over 4 blocks of 32 columns
// (128 threads: the same (column, row group) threads, each with the same 32 state rows), so 4x the blocks of
// gdn_step_norm_multi_kernel - which ran one 512-thread block per head (32 blocks) and read the 2 MB state of a layer
// at ~60 GB/s. Every column's arithmetic is the original's; the output norm, which needs all 128 columns of a head,
// moves to gdn_out_norm_kernel: this kernel leaves the unnormalized output `oc` in y and the second kernel sums the
// squares by the same warps (columns 32w..32w+31, the same butterfly) in the same order.
// S26: OUT = false (the commit, whose outputs nobody reads) drops the output reduction and its two barriers; the
// norm kernel below has the old kernel's code shape (S x RG threads, rg 0 holding oc), which makes it bitwise equal -
// the 128-thread version differed from the old kernel by 1 ulp in ~7% of the outputs.
constexpr int GS_COLS = 32;
template<bool OUT>
__global__ void __launch_bounds__(GS_COLS * RG) gdn_step_split_kernel(float* __restrict__ state,
                                                                      const float* __restrict__ hbuf, int C,
                                                                      const float* __restrict__ gate,
                                                                      const float* __restrict__ beta,
                                                                      float* __restrict__ y, int h_k, int h_v, int T,
                                                                      const int32_t* __restrict__ n_keep,
                                                                      int t_out_begin) {
    __shared__ float sk[S], sq[S];
    __shared__ float red[RG][GS_COLS];
    const int head = blockIdx.x;
    const int c0 = blockIdx.y * GS_COLS;
    const int lc = threadIdx.x;                 // local column
    const int col = c0 + lc;
    const int rg = threadIdx.y;
    const int tid = rg * GS_COLS + lc;
    const int qh = head % h_k;
    const int qk = S * h_k;
    const int value_dim = S * h_v;
    const int n = n_keep ? *n_keep : T;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
    for (int t = 0; t < n; ++t) {
        const float* ht = hbuf + (size_t) t * C;
        __syncthreads();
        if (tid < S) { sk[tid] = ht[qk + qh * S + tid]; sq[tid] = ht[qh * S + tid]; }
        __syncthreads();
        const float g = __expf(gate[(size_t) t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
        red[rg][lc] = kv;
        __syncthreads();
        const float kv_col = red[0][lc] + red[1][lc] + red[2][lc] + red[3][lc];
        const float delta = (ht[2 * qk + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
            o = fmaf(s[r], sq[rg * RPG + r], o);
        }
        if (!OUT) continue;
        __syncthreads();
        red[rg][lc] = o;
        __syncthreads();
        if (rg == 0 && t >= t_out_begin)
            y[(size_t) t * value_dim + head * S + col] =
                (red[0][lc] + red[1][lc] + red[2][lc] + red[3][lc]) * rsqrtf((float) S);
    }
    if (n_keep != nullptr && n > 0) {
#pragma unroll
        for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
    }
}


// S26: the state-only replay (the commit) with its per-token inputs prefetched - gate, beta and the v column of every
// row up front, the next row's k / q while the current one computes. The arithmetic and its order are
// gdn_step_split_kernel<false>'s (gfx1151 harness: state bitwise equal; 44.5 -> 39.1 us at 1 row, 47.4 -> 40.5 at 2,
// 53.8 -> 42.9 at 4, h_v 48). With outputs the prefetch lost (register pressure), so the window keeps the plain split.
template<int MAXT>
__global__ void __launch_bounds__(GS_COLS * RG) gdn_state_replay_kernel(float* __restrict__ state,
                                                                        const float* __restrict__ hbuf, int C,
                                                                        const float* __restrict__ gate,
                                                                        const float* __restrict__ beta, int h_k,
                                                                        int h_v, int T, const int32_t* __restrict__ n_keep) {
    __shared__ float sk[S], sq[S];
    __shared__ float red[RG][GS_COLS];
    const int head = blockIdx.x;
    const int c0 = blockIdx.y * GS_COLS;
    const int lc = threadIdx.x;
    const int col = c0 + lc;
    const int rg = threadIdx.y;
    const int tid = rg * GS_COLS + lc;
    const int qh = head % h_k;
    const int qk = S * h_k;
    const int n = n_keep ? *n_keep : T;
    float gv[MAXT], bv[MAXT], vv[MAXT];
#pragma unroll
    for (int t = 0; t < MAXT; ++t) {
        if (t < n) {
            gv[t] = gate[(size_t) t * h_v + head];
            bv[t] = beta[(size_t) t * h_v + head];
            vv[t] = hbuf[(size_t) t * C + 2 * qk + head * S + col];
        }
    }
    float kn = 0.0f, qn = 0.0f;
    if (n > 0) { kn = hbuf[qk + qh * S + tid]; qn = hbuf[qh * S + tid]; }
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
#pragma unroll
    for (int t = 0; t < MAXT; ++t) {
        if (t >= n) break;
        __syncthreads();
        sk[tid] = kn; sq[tid] = qn;
        __syncthreads();
        if (t + 1 < n) {
            const float* hn = hbuf + (size_t) (t + 1) * C;
            kn = hn[qk + qh * S + tid]; qn = hn[qh * S + tid];
        }
        const float g = __expf(gv[t]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
        red[rg][lc] = kv;
        __syncthreads();
        const float kv_col = red[0][lc] + red[1][lc] + red[2][lc] + red[3][lc];
        const float delta = (vv[t] - g * kv_col) * bv[t];
        float o = 0.0f;   // unused: kept because the measured code had it (without it: 51 us instead of 39 at 1 row)
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
            o = fmaf(s[r], sq[rg * RPG + r], o);
        }
        (void) o;
    }
    if (n_keep != nullptr && n > 0) {
#pragma unroll
        for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
    }
}
// y = oc * rsqrt(mean(oc^2) + eps) * gamma * sigmoid(z) per head and token: gdn_step_norm_multi_kernel's tail with its
// code shape (S x RG threads; rg 0 holds oc, the other row groups contribute 0 to wsum), reading oc from y
template<bool Q>
__global__ void __launch_bounds__(S * RG) gdn_out_norm_kernel(const float* __restrict__ z, const float* __restrict__ gamma,
                                                             float eps, float* __restrict__ y, int h_v, int T,
                                                             const int32_t* __restrict__ n_keep, int t_out_begin,
                                                             GdnQ81* __restrict__ xq) {
    __shared__ float wsum[S * RG / 32];
    const int head = blockIdx.x, t = blockIdx.y + t_out_begin, col = threadIdx.x, rg = threadIdx.y;
    const int tid = rg * S + col;
    const int n = n_keep ? *n_keep : T;
    if (t >= n) return;
    const int value_dim = S * h_v;
    float oc = 0.0f, sq_part = 0.0f;
    if (rg == 0) {
        oc = y[(size_t) t * value_dim + head * S + col];
        sq_part = oc * oc;
    }
    for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part += __shfl_xor_sync(0xffffffffu, sq_part, o2);
    if ((tid & 31) == 0) wsum[tid >> 5] = sq_part;
    __syncthreads();
    if (rg == 0) {
        const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
        const float scale = rsqrtf(ss / (float) S + eps);
        const float zz = z[(size_t) t * value_dim + head * S + col];
        if constexpr (Q) {
            const float yv = oc * scale * gamma[col] * (1.0f / (1.0f + __expf(-zz)));
            y[(size_t) t * value_dim + head * S + col] = yv;
            gdn_q8_1_store(xq, (size_t) (t - t_out_begin) * value_dim + head * S + col, yv);
        } else {
            y[(size_t) t * value_dim + head * S + col] = oc * scale * gamma[col] * (1.0f / (1.0f + __expf(-zz)));
        }
    }
}

__global__ void embedding_gather_dev_kernel(const uint8_t* __restrict__ codes, const float* __restrict__ scales,
                                            const float* __restrict__ offsets, const int32_t* __restrict__ tokens,
                                            int64_t n, int code_bits, int code_bias, int group_elems,
                                            unsigned long long row_codes, unsigned long long row_groups,
                                            float* __restrict__ out) {
    const int t = blockIdx.y;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned long long token = (unsigned long long) tokens[t];
    const uint8_t* c = codes + token * row_codes;
    const float* sc = scales + token * row_groups;
    const float* of = offsets ? offsets + token * row_groups : nullptr;
    const int per_byte = 8 / code_bits;
    const unsigned mask = (1u << code_bits) - 1u;
    const int code = (c[i / per_byte] >> ((i % per_byte) * code_bits)) & mask;
    const int64_t group = i / group_elems;
    const float product = __fmul_rn((float) (code + code_bias), sc[group]);
    out[(size_t) t * n + i] = __fadd_rn(product, of ? of[group] : 0.0f);
}

__global__ void broadcast_streams_kernel(const float* __restrict__ x, float* __restrict__ R, int64_t n, int hc) {
    const int t = blockIdx.y;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * hc) return;
    R[(size_t) t * n * hc + i] = x[(size_t) t * n + i % n];
}

__global__ void copy_indexed_kernel(float* __restrict__ dst, const float* __restrict__ src, int64_t stride,
                                    const int32_t* __restrict__ index, int64_t n) {
    const int idx = *index;
    if (idx < 0) return;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        dst[i] = src[(size_t) idx * stride + i];
}

__global__ void fetch_blobs_kernel(const unsigned long long* __restrict__ src, const int32_t* __restrict__ n,
                                   uint4* __restrict__ dst, long long per) {
    // four independent 16-byte loads in flight per thread before their stores: the link's latency is covered with
    // fewer blocks, which leaves SMs to the VRAM experts running beside this copy (STRATA_FETCH_BLOCKS)
    const long long total = (long long) *n * per;
    const long long S = (long long) gridDim.x * blockDim.x;
    long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    for (; i + 3 * S < total; i += 4 * S) {
        uint4 v[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const long long j = i + u * S, k = j / per;
            v[u] = ((const uint4*) src[k])[j - k * per];
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) dst[i + u * S] = v[u];
    }
    for (; i < total; i += S) {
        const long long k = i / per;
        dst[i] = ((const uint4*) src[k])[i - k * per];
    }
}

__global__ void fetch_blobs_strided_kernel(const unsigned long long* __restrict__ src, const int32_t* __restrict__ n,
                                           uint8_t* __restrict__ dst, long long per, int q0, int qstep) {
    const int nq = *n > q0 ? (*n - q0 + qstep - 1) / qstep : 0;
    const long long total = (long long) nq * per;
    const long long S = (long long) gridDim.x * blockDim.x;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < total; i += S) {
        const long long j = i / per, off = i - j * per;
        const int q = q0 + (int) j * qstep;
        ((uint4*) (dst + (size_t) q * (size_t) per * 16))[off] = ((const uint4*) src[q])[off];
    }
}

__global__ void wait_and_clear_kernel(uint32_t* flag) {
    volatile uint32_t* f = flag;
    while (*f == 0) __nanosleep(100);
    __threadfence_system();
    *f = 0;
}

__global__ void partner_fetch_kernel(PartnerFetchArgs a) {
    __shared__ unsigned long long sp[32];
    __shared__ int s_n;
    __shared__ int s_stop;
    for (int k = 0; k < a.steps; ++k) {
        const uint32_t ring = (uint32_t) k + 1;
        if (threadIdx.x == 0) {
            volatile const uint32_t* f = a.flagA;
            uint32_t v;
            while ((v = *f) < ring) __nanosleep(200);
            s_stop = v == 0xffffffffu;
            __threadfence_system();
        }
        __syncthreads();
        if (s_stop) return;
        const int grp = k % a.G, l = a.lb + k / a.G;
        const volatile int32_t* pl = a.plan + (size_t) grp * (size_t) a.plan_i32;
        if (threadIdx.x == 0) s_n = pl[2];
        __syncthreads();
        const int n2 = s_n < 32 ? s_n : 32;
        const volatile unsigned long long* p2 = (const volatile unsigned long long*) (pl + a.ptr_off) + a.capx;
        if ((int) threadIdx.x < n2) sp[threadIdx.x] = p2[threadIdx.x];
        __syncthreads();
        const long long per16 = a.blob_bytes[l] / 16;
        uint8_t* stage = a.staging + (size_t) (grp * a.per) * (size_t) a.max_blob;
        const int nodd = n2 / 2;   // q = 1, 3, ... < n2
        const long long total = (long long) nodd * per16;
        const long long S = (long long) gridDim.x * blockDim.x;
        long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
        for (; i + 3 * S < total; i += 4 * S) {
            uint4 v[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const long long x = i + u * S, j = x / per16;
                v[u] = ((const uint4*) sp[2 * j + 1])[x - j * per16];
            }
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const long long x = i + u * S, j = x / per16;
                ((uint4*) (stage + (size_t) (2 * j + 1) * (size_t) a.blob_bytes[l]))[x - j * per16] = v[u];
            }
        }
        for (; i < total; i += S) {
            const long long j = i / per16;
            ((uint4*) (stage + (size_t) (2 * j + 1) * (size_t) a.blob_bytes[l]))[i - j * per16] =
                ((const uint4*) sp[2 * j + 1])[i - j * per16];
        }
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) {
            const unsigned prev = atomicAdd(&a.arrive[k], 1u);
            if (prev == gridDim.x - 1) {
                __threadfence_system();
                *(volatile uint32_t*) (a.done + k) = 1u;
            }
        }
        __syncthreads();
    }
}

__global__ void rebase_ptrs_kernel(unsigned long long* ptr, const int32_t* n, unsigned long long base, long long bytes) {
    const int k = threadIdx.x;
    if (k < *n) ptr[k] = base + (unsigned long long) k * (unsigned long long) bytes;
}

__global__ void add_streams_broadcast_kernel(const float* __restrict__ h, const float* __restrict__ e,
                                             float* __restrict__ R, int64_t n, int hc) {
    const int t = blockIdx.y;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * hc) return;
    R[(size_t) t * n * hc + i] = h[(size_t) t * n * hc + i] + e[(size_t) t * n + i % n];
}

__global__ void ident_hits_kernel(const int32_t* __restrict__ ids, int n, int32_t* __restrict__ slot,
                                  int32_t* __restrict__ dst, int32_t* __restrict__ count) {
    const int i = threadIdx.x;
    if (i < n) { slot[i] = ids[i]; dst[i] = i; }
    if (i == 0) *count = n;
}

// E = the widest element the row size divides into (16, 4 or 1 bytes): a Q6_K head row of 2560 values is 2100 bytes
template<typename E>
__global__ void gather_rows_kernel(const E* __restrict__ src, long long row_e, const int32_t* __restrict__ ids,
                                   long long n, E* __restrict__ dst) {
    const long long total = n * row_e;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (long long) gridDim.x * blockDim.x) {
        const long long r = i / row_e, o = i - r * row_e;
        dst[i] = src[(long long) ids[r] * row_e + o];
    }
}

__global__ void map_ids_kernel(int32_t* ids, const int32_t* __restrict__ table, int n) {
    const int i = threadIdx.x;
    if (i < n) ids[i] = table[ids[i]];
}

__global__ void row_top_prob_kernel(const float* __restrict__ logits, int n_vocab, const int32_t* __restrict__ ids,
                                    float* __restrict__ probs) {
    __shared__ float part[32];
    const int t = blockIdx.x;
    const float* l = logits + (size_t) t * n_vocab;
    const float m = l[ids[t]];
    float s = 0.0f;
    for (int i = threadIdx.x; i < n_vocab; i += blockDim.x) s += __expf(l[i] - m);
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot = 0.0f;
        for (int w = 0; w < (int) (blockDim.x >> 5); ++w) tot += part[w];
        probs[t] = 1.0f / tot;
    }
}

__global__ void mtp_select_kernel(const float* __restrict__ R_src, int64_t stride, const int32_t* __restrict__ ids,
                                  const int32_t* __restrict__ row_dev, float* __restrict__ R_dst,
                                  int32_t* __restrict__ tok_dst, int32_t* out, int j, const float* probs, float* out_p) {
    const int row = *row_dev;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < stride; i += (int64_t) gridDim.x * blockDim.x)
        R_dst[i] = R_src[(size_t) row * stride + i];
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const int32_t tok = ids[row];
        *tok_dst = tok;
        if (probs != nullptr && out_p != nullptr) ((volatile float*) out_p)[j] = probs[row];
        __threadfence_system();
        if (out != nullptr) ((volatile int32_t*) out)[j] = tok;
    }
}

__global__ void dense_steps_kernel(const int32_t* __restrict__ cells, int n, int32_t* __restrict__ steps) {
    const int i = threadIdx.x;
    if (i >= n) return;
    const int c = cells[i];
    steps[i * 4 + 0] = c;
    steps[i * 4 + 1] = c + 1;
    steps[i * 4 + 2] = (c + 1) / 4;
    steps[i * 4 + 3] = c + 1;
}

}  // namespace

void fetch_blobs(const unsigned long long* src, const int32_t* n, uint8_t* dst, int64_t blob_bytes, int cap, void* stream) {
    if (cap <= 0) return;
    if (blob_bytes % 16 != 0) { std::fprintf(stderr, "fetch_blobs: blob size must be a multiple of 16\n"); std::exit(1); }
    static const unsigned blocks = [] {
        const char* v = std::getenv("STRATA_FETCH_BLOCKS");
        const int x = v ? std::atoi(v) : 0;
        return x > 0 && x <= 4096 ? (unsigned) x : 24u;   // 24 x 256 beside the VRAM experts (AC922, 2 GPUs: 43.7 ms/window vs 45.8)
    }();
    static const unsigned threads = [] {
        const char* v = std::getenv("STRATA_FETCH_THREADS");
        const int x = v ? std::atoi(v) : 0;
        return x == 512 || x == 1024 ? (unsigned) x : 256u;
    }();
    fetch_blobs_kernel<<<blocks, threads, 0, (cudaStream_t) stream>>>(src, n, (uint4*) dst, (long long) (blob_bytes / 16));
    check("fetch_blobs");
}

void fetch_blobs_strided(const unsigned long long* src, const int32_t* n, uint8_t* dst, int64_t blob_bytes, int q0,
                         int qstep, void* stream) {
    if (blob_bytes % 16 != 0) { std::fprintf(stderr, "fetch_blobs: blob size must be a multiple of 16\n"); std::exit(1); }
    fetch_blobs_strided_kernel<<<24, 256, 0, (cudaStream_t) stream>>>(src, n, dst, (long long) (blob_bytes / 16), q0, qstep);
    check("fetch_blobs_strided");
}

void wait_and_clear(uint32_t* flag, void* stream) {
    wait_and_clear_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(flag);
    check("wait_and_clear");
}

void partner_fetch(const PartnerFetchArgs& a, int blocks, void* stream) {
    partner_fetch_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>(a);
    check("partner_fetch");
}

void rebase_ptrs(unsigned long long* ptr, const int32_t* n, uint8_t* base, int64_t blob_bytes, void* stream) {
    rebase_ptrs_kernel<<<1, 128, 0, (cudaStream_t) stream>>>(ptr, n, (unsigned long long) base, (long long) blob_bytes);
    check("rebase_ptrs");
}

void add_streams_broadcast(const float* h, const float* e, float* R, int64_t n_embd, int hc, int n_tok, void* stream) {
    add_streams_broadcast_kernel<<<dim3((unsigned) ((n_embd * hc + 255) / 256), (unsigned) n_tok), 256, 0,
                                   (cudaStream_t) stream>>>(h, e, R, n_embd, hc);
    check("add_streams_broadcast");
}

void ident_hits(const int32_t* ids, int n, int32_t* slot, int32_t* dst, int32_t* count, void* stream) {
    if (n < 1 || n > 1024) { std::fprintf(stderr, "ident_hits: n out of range\n"); std::exit(1); }
    ident_hits_kernel<<<1, 1024, 0, (cudaStream_t) stream>>>(ids, n, slot, dst, count);
    check("ident_hits");
}

void mtp_select(const float* R_src, int64_t R_stride, const int32_t* ids, const int32_t* row_dev, float* R_dst,
                int32_t* tok_dst, int32_t* out, int j, void* stream, const float* probs, float* out_p) {
    mtp_select_kernel<<<1, 256, 0, (cudaStream_t) stream>>>(R_src, R_stride, ids, row_dev, R_dst, tok_dst, out, j,
                                                            probs, out_p);
    check("mtp_select");
}

void gather_rows(const uint8_t* src, int64_t row_bytes, const int32_t* ids, int64_t n, uint8_t* dst, void* stream) {
    cudaStream_t s = (cudaStream_t) stream;
    if (row_bytes % 16 == 0)
        gather_rows_kernel<<<48 * 8, 256, 0, s>>>((const uint4*) src, row_bytes / 16, ids, n, (uint4*) dst);
    else if (row_bytes % 4 == 0)
        gather_rows_kernel<<<48 * 8, 256, 0, s>>>((const uint32_t*) src, row_bytes / 4, ids, n, (uint32_t*) dst);
    else
        gather_rows_kernel<<<48 * 8, 256, 0, s>>>(src, row_bytes, ids, n, dst);
    check("gather_rows");
}

void map_ids(int32_t* ids, const int32_t* table, int n, void* stream) {
    map_ids_kernel<<<1, 64, 0, (cudaStream_t) stream>>>(ids, table, n);
    check("map_ids");
}

void row_top_prob(const float* logits, int n_rows, int n_vocab, const int32_t* ids, float* probs, void* stream) {
    row_top_prob_kernel<<<n_rows, 1024, 0, (cudaStream_t) stream>>>(logits, n_vocab, ids, probs);
    check("row_top_prob");
}

namespace {
__global__ void window_ids_kernel(int32_t* steps, int window, int32_t* ids, long long stride) {
    const int q = blockIdx.y;
    int32_t* st = steps + q * 4;
    const int n_kv = st[1];
    const int start = n_kv > window ? n_kv - window : 0;
    const int width = n_kv - start;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < width; j += gridDim.x * blockDim.x)
        ids[q * stride + j] = start + j;
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0) st[3] = width;
}
}  // namespace

void window_ids(int32_t* steps, int n, int window, int32_t* ids, int64_t ids_stride, void* stream) {
    window_ids_kernel<<<dim3(8, (unsigned) n), 256, 0, (cudaStream_t) stream>>>(steps, window, ids, (long long) ids_stride);
    check("window_ids");
}

void dense_steps(const int32_t* cells, int n, int32_t* steps, void* stream) {
    dense_steps_kernel<<<1, 64, 0, (cudaStream_t) stream>>>(cells, n, steps);
    check("dense_steps");
}

void gdn_conv_l2_multi(const float* history, const float* qkv, const float* conv_w, float* h, int channels,
                       int qk_heads, float eps, int n_tok, void* stream, int t_begin) {
    if (!history || !qkv || !conv_w || !h || channels % S != 0 || n_tok < 1 || n_tok > kVerifyMaxT) {
        std::fprintf(stderr, "gdn_conv_l2_multi: invalid arguments\n");
        std::exit(1);
    }
    gdn_conv_l2_multi_kernel<<<dim3((unsigned) (channels / S), (unsigned) n_tok), S, 0, (cudaStream_t) stream>>>(
        history, qkv, conv_w, h, channels, qk_heads, eps, t_begin);
    check("gdn_conv_l2_multi");
}

void gdn_conv_commit(float* history, const float* qkv, int channels, const int32_t* n_keep, void* stream) {
    gdn_conv_commit_kernel<<<(unsigned) ((channels + 255) / 256), 256, 0, (cudaStream_t) stream>>>(history, qkv,
                                                                                                 channels, n_keep);
    check("gdn_conv_commit");
}

void gdn_ab_multi(const float* x, const void* w_alpha, const void* w_beta, const float* dt, const float* ssm_a,
                  float* gate, float* beta, int n_embd, int h_v, int n_tok, void* stream, WForm form) {
    if (n_embd % 8 != 0 || n_tok < 1 || n_tok > kVerifyMaxT) {
        std::fprintf(stderr, "gdn_ab_multi: invalid arguments\n");
        std::exit(1);
    }
    const unsigned blocks = (unsigned) ((2 * h_v + 7) / 8);
    static const bool generic = [] { const char* v = std::getenv("STRATA_HC_DOWN_GENERIC"); return v && v[0] == '1'; }();
    if (n_embd == 2560 && !generic) {
        const cudaStream_t st = (cudaStream_t) stream;
#define STRATA_AB_T(TT) case TT: \
        if (form == WForm::Bf16) gdn_ab_multi_t_kernel<0, TT><<<blocks, 256, 0, st>>>(x, w_alpha, w_beta, dt, ssm_a, gate, beta, h_v); \
        else if (form == WForm::F16) gdn_ab_multi_t_kernel<1, TT><<<blocks, 256, 0, st>>>(x, w_alpha, w_beta, dt, ssm_a, gate, beta, h_v); \
        else gdn_ab_multi_t_kernel<2, TT><<<blocks, 256, 0, st>>>(x, w_alpha, w_beta, dt, ssm_a, gate, beta, h_v); \
        check("gdn_ab_multi"); return;
        switch (n_tok) { STRATA_AB_T(1) STRATA_AB_T(2) STRATA_AB_T(3) STRATA_AB_T(4) STRATA_AB_T(5) STRATA_AB_T(6) STRATA_AB_T(7) STRATA_AB_T(8) default: break; }
#undef STRATA_AB_T
    }
    if (form == WForm::Bf16)
        gdn_ab_multi_kernel<0><<<blocks, 256, 0, (cudaStream_t) stream>>>(x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok);
    else if (form == WForm::F16)
        gdn_ab_multi_kernel<1><<<blocks, 256, 0, (cudaStream_t) stream>>>(x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok);
    else
        gdn_ab_multi_kernel<2><<<blocks, 256, 0, (cudaStream_t) stream>>>(x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok);
    check("gdn_ab_multi");
}

void gdn_step_norm_multi(float* state, const float* h, int conv_channels, const float* gate, const float* beta,
                         const float* z, const float* gamma, float eps, float* y, int h_k, int h_v, int n_tok,
                         const int32_t* n_keep, void* stream, int t_out_begin, void* xq_out) {
    GdnQ81* const xq = static_cast<GdnQ81*>(xq_out);   // S26: non-null = also the outputs' q8_1 (STRATA_QFUSE)
    if (!state || !h || !gate || !beta || !z || !gamma || !y || h_k <= 0 || h_v % h_k || n_tok < 1 ||
        n_tok > kVerifyMaxT) {
        std::fprintf(stderr, "gdn_step_norm_multi: invalid arguments\n");
        std::exit(1);
    }
    static const bool split = [] { const char* v = std::getenv("STRATA_GDN_SPLIT"); return v && v[0] == '1'; }();
    // S26: the split runs where it is faster (gfx1151 harness, h_v 48: the commit 8-19% per call; a window of 2+ rows
    // 3-13%; one row: no gain, the old kernel stays) - bitwise equal in every case
    if (split && (t_out_begin >= n_tok || n_tok - t_out_begin >= 2)) {
        const dim3 grid((unsigned) h_v, S / GS_COLS), block(GS_COLS, RG);
        if (t_out_begin >= n_tok) {   // no outputs (the commit): the state only
            gdn_state_replay_kernel<kVerifyMaxT><<<grid, block, 0, (cudaStream_t) stream>>>(
                state, h, conv_channels, gate, beta, h_k, h_v, n_tok, n_keep);
        } else {
            gdn_step_split_kernel<true><<<grid, block, 0, (cudaStream_t) stream>>>(
                state, h, conv_channels, gate, beta, y, h_k, h_v, n_tok, n_keep, t_out_begin);
            const dim3 ng((unsigned) h_v, (unsigned) (n_tok - t_out_begin));
            if (xq) gdn_out_norm_kernel<true><<<ng, dim3(S, RG), 0, (cudaStream_t) stream>>>(z, gamma, eps, y, h_v, n_tok, n_keep, t_out_begin, xq);
            else gdn_out_norm_kernel<false><<<ng, dim3(S, RG), 0, (cudaStream_t) stream>>>(z, gamma, eps, y, h_v, n_tok, n_keep, t_out_begin, nullptr);
        }
        check("gdn_step_norm_multi (split)");
        return;
    }
    static const bool commit_split = [] {
        const char* e = std::getenv("STRATA_GDN_COMMIT_SPLIT");
        return !e || e[0] != '0';
    }();
    if (commit_split && n_keep != nullptr && t_out_begin >= n_tok) {
        gdn_step_commit_kernel<<<dim3((unsigned) h_v, 4u), dim3(32, RG), 0, (cudaStream_t) stream>>>(
            state, h, conv_channels, gate, beta, h_k, h_v, n_keep);
        check("gdn_step_commit");
        return;
    }
    if (xq && t_out_begin < n_tok)
        gdn_step_norm_multi_kernel<true><<<(unsigned) h_v, dim3(S, RG), 0, (cudaStream_t) stream>>>(
            state, h, conv_channels, gate, beta, z, gamma, eps, y, h_k, h_v, n_tok, n_keep, t_out_begin, xq);
    else
        gdn_step_norm_multi_kernel<false><<<(unsigned) h_v, dim3(S, RG), 0, (cudaStream_t) stream>>>(
            state, h, conv_channels, gate, beta, z, gamma, eps, y, h_k, h_v, n_tok, n_keep, t_out_begin, nullptr);
    check("gdn_step_norm_multi");
}

namespace {
__global__ void wait_flag_ge_kernel(const volatile uint32_t* flag, uint32_t value) {
    while (*flag < value) strata_spin_pause();
    __threadfence_system();
}
}  // namespace

namespace {
// one thread per window entry: up to kVerifyMaxT tokens x 10 routed experts (80), so 128 (#646 had 64: a window of
// 7+ tokens lost its last entries)
constexpr int kResidentPlanMax = 128;
static_assert(kVerifyMaxT * 10 <= kResidentPlanMax, "resident_plan: one thread per entry");
__global__ void __launch_bounds__(kResidentPlanMax) resident_plan_kernel(const int32_t* __restrict__ ids, int n, int k, const int32_t* __restrict__ res,
                                     int n_expert, const uint8_t* cache_base, const unsigned long long* slot_off,
                                     long long blob, int32_t* __restrict__ pl, long long capx, uint32_t* skip,
                                     uint32_t ring, volatile uint32_t* plan_err) {
    __shared__ int32_t s_ids[kResidentPlanMax];
    __shared__ unsigned long long s_ptr[kResidentPlanMax];
    __shared__ int32_t s_first[kResidentPlanMax];
    __shared__ int32_t s_cnt[kResidentPlanMax];
    __shared__ int32_t s_gstart[kResidentPlanMax];
    __shared__ int s_bad;
    const int tid = threadIdx.x;
    if (tid == 0) s_bad = 0;
    __syncthreads();

    int32_t eid = -1;
    if (tid < n) {
        eid = ids[tid];
        s_ids[tid] = eid;
        const int32_t slot = (eid >= 0 && eid < n_expert) ? res[eid] : -1;
        if (slot < 0) {
            atomicOr(&s_bad, 1);
        } else {
            s_ptr[tid] = (unsigned long long) (cache_base + (slot_off ? (size_t) slot_off[slot] : (size_t) slot * (size_t) blob));
        }
    }
    __syncthreads();
    if (s_bad) {
        if (tid == 0 && skip != nullptr) *skip = 0;
        if (tid == 0 && skip == nullptr) {   // #871: the all-resident graph has no host plan to fall back on
            pl[0] = 0; pl[1] = 0; pl[2] = 0;   // an empty plan: no expert runs on a stale pointer
            if (plan_err != nullptr) { *plan_err = 1; __threadfence_system(); }
        }
        return;
    }

    int first_j = tid;
    int rank_in_group = 0;
    int count_same = 0;
    if (tid < n) {
        for (int j = 0; j < n; ++j) {
            if (s_ids[j] == eid) {
                if (j < first_j) first_j = j;
                if (j < tid) ++rank_in_group;
                ++count_same;
            }
        }
        const bool is_first = (first_j == tid);
        s_first[tid] = is_first ? 1 : 0;
        s_cnt[tid] = is_first ? count_same : 0;
    }
    __syncthreads();

    int32_t* counts = pl;
    int32_t* start = pl + 4;
    int32_t* dst = start + capx + 1;
    int32_t* tok = dst + capx;
    const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
    unsigned long long* ptr = (unsigned long long*) (pl + ptr_off);
    int32_t* start2 = pl + ptr_off + 4 * capx;

    if (tid < n && first_j == tid) {
        int grp_idx = 0;
        int ent_start = 0;
        for (int j = 0; j < tid; ++j) {
            grp_idx += s_first[j];
            ent_start += s_cnt[j];
        }
        ptr[grp_idx] = s_ptr[tid];
        start[grp_idx] = ent_start;
        s_gstart[tid] = ent_start;
    }
    __syncthreads();

    if (tid < n) {
        const int out_idx = s_gstart[first_j] + rank_in_group;
        dst[out_idx] = tid;
        tok[out_idx] = tid / k;
    }
    if (tid == 0) {
        int groups = 0;
        for (int j = 0; j < n; ++j) groups += s_first[j];
        start[groups] = n;
        start2[0] = n;
        counts[0] = groups;
        counts[1] = n;
        counts[2] = 0;
        if (skip != nullptr) {
            __threadfence();
            *skip = ring;
        }
    }
}
// The same plan in one block of 128 threads (n <= 128): thread i owns entry i. Groups are the distinct experts in
// order of first occurrence; group g's entries are its occurrences in increasing i - exactly the loop above
// (S26: the one-thread loop took ~44 us per call on gfx1151, 48 per window).
__global__ void resident_plan_par_kernel(const int32_t* __restrict__ ids, int n, int k, const int32_t* __restrict__ res,
                                         int n_expert, const uint8_t* cache_base, const unsigned long long* slot_off,
                                         long long blob, int32_t* __restrict__ pl, long long capx, uint32_t* skip,
                                         uint32_t ring) {
    __shared__ int32_t s_id[128];
    __shared__ int s_first[128], s_size[128];
    __shared__ int s_bad;
    const int i = threadIdx.x;
    if (i == 0) s_bad = 0;
    __syncthreads();
    int32_t e = -1;
    if (i < n) {
        e = ids[i];
        s_id[i] = e;
        if (e < 0 || e >= n_expert || res[e] < 0) s_bad = 1;
    }
    __syncthreads();
    if (s_bad) { if (i == 0) *skip = 0; return; }
    int first = i, rank = 0, size = 0;
    if (i < n) {
        for (int j = 0; j < n; ++j) {
            const bool same = s_id[j] == e;
            if (same && j < first) first = j;
            if (same && j < i) ++rank;
            size += same;
        }
    }
    if (i < n) { s_first[i] = first; s_size[i] = size; }
    __syncthreads();
    int32_t* counts = pl;
    int32_t* start = pl + 4;
    int32_t* dst = start + capx + 1;
    int32_t* tok = dst + capx;
    const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
    unsigned long long* ptr = (unsigned long long*) (pl + ptr_off);
    int32_t* start2 = pl + ptr_off + 4 * capx;
    if (i < n) {
        // group of entry i = the number of first occurrences before `first`; its start = their sizes' sum
        int group = 0, gstart = 0;
        for (int j = 0; j < first; ++j)
            if (s_first[j] == j) { ++group; gstart += s_size[j]; }
        if (i == first) {
            const int32_t slot = res[e];
            ptr[group] = (unsigned long long) (cache_base + (slot_off ? (size_t) slot_off[slot] : (size_t) slot * (size_t) blob));
            start[group] = gstart;
        }
        dst[gstart + rank] = i;
        tok[gstart + rank] = i / k;
        if (i == 0) {
            int groups = 0;
            for (int j = 0; j < n; ++j) groups += s_first[j] == j;
            start[groups] = n;
            start2[0] = n;
            counts[0] = groups;
            counts[1] = n;
            counts[2] = 0;
        }
    }
    __threadfence();
    __syncthreads();
    if (i == 0) *skip = ring;
}
__global__ void wait_flag_ge_or_kernel(const volatile uint32_t* flag, uint32_t value, const volatile uint32_t* skip) {
    if (*skip == value) return;
    while (*flag < value) strata_spin_pause();
    __threadfence_system();
}
__global__ void copy_i32_unless_kernel(int32_t* __restrict__ dst, const volatile int32_t* src, int n,
                                       const uint32_t* skip, uint32_t value) {
    if (*skip == value) return;
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];
}
__global__ void copy_or_zero_kernel(float4* __restrict__ dst, const volatile float4* src, long long n4,
                                    const uint32_t* skip, uint32_t value) {
    const bool zero = *skip == value;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (long long) gridDim.x * blockDim.x)
        dst[i] = zero ? make_float4(0.f, 0.f, 0.f, 0.f) : const_cast<const float4*>(src)[i];
}
}  // namespace

void resident_plan(const int32_t* ids, int n_entries, int k, const int32_t* res_layer, int n_expert,
                   const uint8_t* cache_base, const unsigned long long* slot_off, long long blob, int32_t* plan,
                   long long capx, uint32_t* skip, uint32_t ring, void* stream, uint32_t* plan_err) {
    resident_plan_kernel<<<1, kResidentPlanMax, 0, (cudaStream_t) stream>>>(ids, n_entries, k, res_layer, n_expert, cache_base, slot_off,
                                                              blob, plan, capx, skip, ring, plan_err);
    check("resident_plan");
}
void wait_flag_ge_or(const uint32_t* flag, uint32_t value, const uint32_t* skip, void* stream) {
    wait_flag_ge_or_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(flag, value, skip);
    check("wait_flag_ge_or");
}
void copy_i32_from_mapped_unless(int32_t* dst, const int32_t* src, long long n, const uint32_t* skip, uint32_t value,
                                 void* stream) {
    if (n <= 0) return;
    copy_i32_unless_kernel<<<1, 128, 0, (cudaStream_t) stream>>>(dst, (const volatile int32_t*) src, (int) n, skip, value);
    check("copy_i32_from_mapped_unless");
}
void copy_or_zero_from_mapped(float* dst, const float* src, long long n, const uint32_t* skip, uint32_t value,
                              void* stream) {
    if (n <= 0) return;
    const long long n4 = n / 4;
    const int blocks = (int) ((n4 + 255) / 256 < 64 ? (n4 + 255) / 256 : 64);
    copy_or_zero_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>((float4*) dst, (const volatile float4*) src, n4, skip,
                                                                    value);
    check("copy_or_zero_from_mapped");
}

void wait_flag_ge(const uint32_t* flag, uint32_t value, void* stream) {
    wait_flag_ge_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(flag, value);
    check("wait_flag_ge");
}

void embedding_gather_dev(const uint8_t* codes, const float* scales, const float* offsets, const int32_t* tokens,
                          int n_tok, int64_t n, int code_bits, int code_bias, int group_elems, uint64_t row_codes,
                          uint64_t row_groups, float* out, void* stream) {
    embedding_gather_dev_kernel<<<dim3((unsigned) ((n + 255) / 256), (unsigned) n_tok), 256, 0,
                                  (cudaStream_t) stream>>>(codes, scales, offsets, tokens, n, code_bits, code_bias,
                                                           group_elems, row_codes, row_groups, out);
    check("embedding_gather_dev");
}

void broadcast_streams(const float* x, float* R, int64_t n_embd, int hc, int n_tok, void* stream) {
    broadcast_streams_kernel<<<dim3((unsigned) ((n_embd * hc + 255) / 256), (unsigned) n_tok), 256, 0,
                               (cudaStream_t) stream>>>(x, R, n_embd, hc);
    check("broadcast_streams");
}

void copy_indexed(float* dst, const float* src, int64_t stride, const int32_t* index, int64_t n, void* stream) {
    const unsigned blocks = (unsigned) ((n + 255) / 256 < 64 ? (n + 255) / 256 : 64);
    copy_indexed_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>(dst, src, stride, index, n);
    check("copy_indexed");
}

// a GPU timestamp (ns, %globaltimer) into buf[i] - the verify window's stage profiler
namespace { __global__ void gpu_stamp_kernel(unsigned long long* buf, int i) {
    unsigned long long t;
#if defined(STRATA_HIP_GFX906)
    t = wall_clock64() * 40ull;   // gfx906: the wall clock runs at 25 MHz (hipDeviceAttributeWallClockRate) -> ns
#elif defined(__HIPCC__)
    t = wall_clock64() * 10ull;   // gfx10.3 / gfx11 / gfx12: a constant 100 MHz counter, in ns
#else
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
#endif
    buf[i] = t;
} }
void gpu_stamp(unsigned long long* buf, int i, void* stream) {
    gpu_stamp_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(buf, i);
}

}  // namespace strata::kernels
