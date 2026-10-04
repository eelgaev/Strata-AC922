// Fused prefill expert products: see fused_expert.hpp.
#include "strata/kernels/fused_expert.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::kernels {
namespace {

// a block: 4 warps (2 x 2), TM tokens x BN weight rows, K in steps of BK; each warp a 32 x 32 tile, and its 16-row
// halves past the expert's last token are skipped. Two threads dequantize a weight row.
constexpr int TM = 64, BN = 64, BK = 64, LDE = BN + 4, NT = 128, TPR = NT / BN;
// the X and W tiles are [rows][BK] halves without padding, their 16-byte chunks XOR-swizzled by the row: the fragment
// loads of a quarter warp read rows r..r+3 and r+8..r+11 (or + 16, 24), which then fall in 8 distinct chunks
__device__ __forceinline__ int swz(int r) { return (r & 3) | (((r >> 3) & 1) << 2); }
__device__ __forceinline__ int soff(int r, int c) { return r * BK + ((c ^ swz(r)) << 3); }
constexpr int SMEM = (TM + BN) * BK * 2 > TM * LDE * 4 ? (TM + BN) * BK * 2 : TM * LDE * 4;
using Acc = float[8];   // an m8n8k4 accumulator

struct Tiles {
    const uint8_t* blob[kFusedExpertMax];
    int64_t row0[kFusedExpertMax];
    int ne[kFusedExpertMax], tile0[kFusedExpertMax + 1];
    int n;
};

// get_scale_min_k4 on the 12 scale bytes held in registers (s0..s2): a byte-indexed array would put the block header
// in local memory, and that store waits on the prefetch's global load
__device__ __forceinline__ int sbyte(uint32_t s0, uint32_t s1, uint32_t s2, int b) {
    const uint32_t w = b < 4 ? s0 : (b < 8 ? s1 : s2);
    return (int) ((w >> (8 * (b & 3))) & 0xFF);
}
__device__ __forceinline__ void scale_min_k4(int j, uint32_t s0, uint32_t s1, uint32_t s2, int& d, int& m) {
    if (j < 4) { d = sbyte(s0, s1, s2, j) & 63; m = sbyte(s0, s1, s2, j + 4) & 63; }
    else {
        const int a = sbyte(s0, s1, s2, j + 4), lo = sbyte(s0, s1, s2, j - 4), hi = sbyte(s0, s1, s2, j);
        d = (a & 0x0F) | ((lo >> 6) << 4); m = (a >> 4) | ((hi >> 6) << 4);
    }
}
// SwiGLU as swiglu_il_kernel computes it (hf_sat included)
__device__ __forceinline__ __half silu_mul(float g, float u) {
    const float h = g / (1.0f + __expf(-g)) * u;
    return __float2half(isnan(h) ? h : fminf(fmaxf(h, -65504.0f), 65504.0f));
}
__device__ __forceinline__ void locate(const Tiles& g, int nrt, int& e, int& rt, int& ne, int64_t& row0) {
    const int t = (int) blockIdx.x;
    e = 0;
    while (e + 1 < g.n && g.tile0[e + 1] <= t) ++e;
    const int tt = (t - g.tile0[e]) / nrt;
    rt = (t - g.tile0[e]) % nrt;
    ne = min(TM, g.ne[e] - tt * TM);
    row0 = g.row0[e] + (int64_t) tt * TM;
}

// the X tile [TM][BK] through registers (fetched one step ahead); rows >= ne are zero
struct XRegs { uint4 v[TM * BK / 8 / NT]; };
__device__ __forceinline__ void fetch_x(XRegs& x, const __half* X, int64_t ld, int ne, int64_t k0) {
#pragma unroll
    for (int j = 0; j < TM * BK / 8 / NT; ++j) {
        const int i = threadIdx.x + NT * j, r = i / (BK / 8), c = (i % (BK / 8)) * 8;
        x.v[j] = r < ne ? *(const uint4*) (X + r * ld + k0 + c) : make_uint4(0, 0, 0, 0);
    }
}
__device__ __forceinline__ void commit_x(__half* Xs, const XRegs& x) {
#pragma unroll
    for (int j = 0; j < TM * BK / 8 / NT; ++j) {
        const int i = threadIdx.x + NT * j, r = i / (BK / 8);
        *(uint4*) &Xs[soff(r, i % (BK / 8))] = x.v[j];
    }
}
__device__ __forceinline__ void mma884(Acc& d, uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                 "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}
// a warp's 32 x 32 tile as 4 x 4 blocks of 8 x 8; quad pair q computes M blocks (q & 1) + 2 i and N blocks
// 2 (q >> 1) + j, its lane t (0..7) holding row t of A and column t of B (4 k values each, one 8-byte piece)
__device__ __forceinline__ void qp_coords(int& q, int& t) {
    const int lane = threadIdx.x & 31;
    q = (lane >> 2) & 3;
    t = (lane & 3) + ((lane >> 4) << 2);
}
// the step's products in fresh accumulators, then added to the running sum with ordinary FP32 adds: Volta's tensor
// cores truncate when they accumulate, and a 160-step chain (K = 2560) of truncations biases the sum toward zero.
// Rows past the expert's last token (ne) are skipped 16 at a time.
__device__ __forceinline__ void mma_tile(const __half* Xs, const __half* Ws, Acc (&tot)[2][2], int ne) {
    const int w = threadIdx.x / 32, wm = w / 2, wn = w % 2;
    if (32 * wm >= ne) return;
    const bool f1 = 32 * wm + 16 < ne;   // warp-uniform
    int q, t;
    qp_coords(q, t);
    const int ra0 = 32 * wm + 8 * (q & 1) + t, ra1 = ra0 + 16, rb0 = 32 * wn + 16 * (q >> 1) + t, rb1 = rb0 + 8;
    Acc acc[2][2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.0f;
    // independent accumulators back to back (a dependent HMMA waits out the previous one's latency)
    if (f1) {
#pragma unroll
        for (int c = 0; c < BK / 8; ++c) {   // a 16-byte chunk: 8 k values, two k4 steps
            const uint4 b0 = *(const uint4*) &Ws[soff(rb0, c)], b1 = *(const uint4*) &Ws[soff(rb1, c)];
            const uint4 a0 = *(const uint4*) &Xs[soff(ra0, c)], a1 = *(const uint4*) &Xs[soff(ra1, c)];
            mma884(acc[0][0], a0.x, a0.y, b0.x, b0.y);
            mma884(acc[0][1], a0.x, a0.y, b1.x, b1.y);
            mma884(acc[1][0], a1.x, a1.y, b0.x, b0.y);
            mma884(acc[1][1], a1.x, a1.y, b1.x, b1.y);
            mma884(acc[0][0], a0.z, a0.w, b0.z, b0.w);
            mma884(acc[0][1], a0.z, a0.w, b1.z, b1.w);
            mma884(acc[1][0], a1.z, a1.w, b0.z, b0.w);
            mma884(acc[1][1], a1.z, a1.w, b1.z, b1.w);
        }
    } else {
#pragma unroll
        for (int c = 0; c < BK / 8; ++c) {
            const uint4 b0 = *(const uint4*) &Ws[soff(rb0, c)], b1 = *(const uint4*) &Ws[soff(rb1, c)];
            const uint4 a0 = *(const uint4*) &Xs[soff(ra0, c)];
            mma884(acc[0][0], a0.x, a0.y, b0.x, b0.y);
            mma884(acc[0][1], a0.x, a0.y, b1.x, b1.y);
            mma884(acc[0][0], a0.z, a0.w, b0.z, b0.w);
            mma884(acc[0][1], a0.z, a0.w, b1.z, b1.w);
        }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) tot[i][j][e] += acc[i][j][e];
}
// the m8n8k4 FP32 accumulator layout: element e of lane l is row (l & 1) + (e & 2) (+ 4 for l >= 16), column
// (e & 4) + (l & 2) + (e & 1)
__device__ __forceinline__ void store_acc(float (*E)[LDE], Acc (&acc)[2][2]) {
    const int w = threadIdx.x / 32, wm = w / 2, wn = w % 2, lane = threadIdx.x & 31;
    int q, t;
    qp_coords(q, t);
    (void) t;
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int r = 32 * wm + 8 * ((q & 1) + 2 * i) + (lane & 1) + (e & 2) + ((lane >> 4) << 2);
                const int c = 32 * wn + 8 * (2 * (q >> 1) + j) + (e & 4) + (lane & 2) + (e & 1);
                E[r][c] = acc[i][j][e];
            }
}

// gate/up (Q4_K; interleaved row 2r = gate r, 2r + 1 = up r) and SwiGLU -> H
__global__ void __launch_bounds__(NT) gu_q4k_kernel(Tiles g, FusedExpertLayout L, const __half* __restrict__ X,
                                                    __half* __restrict__ H) {
    extern __shared__ __align__(16) unsigned char sm[];
    __half* Xs = (__half*) sm;
    __half* Ws = (__half*) (sm + TM * BK * 2);
    int e, rt, ne;
    int64_t row0;
    locate(g, (int) (2 * L.n_ff / BN), e, rt, ne, row0);
    Acc acc[2][2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.0f;
    // TPR threads per weight row: the low (sub-block 2 jj) or high (2 jj + 1) nibbles of a 64-value chunk, NV 16-byte
    // pieces of its 32 quant bytes each
    constexpr int NV = 4 / TPR;
    const int wr = threadIdx.x / TPR, hi = threadIdx.x % 2, v0 = (threadIdx.x % TPR) / 2 * NV, R = rt * BN + wr;
    const uint8_t* rowp = g.blob[e] + ((R & 1) ? L.up_off : 0) + (size_t) (R >> 1) * L.gu_row;
    const __half* Xb = X + row0 * L.n_embd;
    XRegs xr;
    uint4 qr[NV], hr;
    auto fetch_w = [&](int64_t k0) {
        const uint8_t* b = rowp + (size_t) (k0 / 256) * 144;
        const int jj = (int) (k0 % 256) / 64;
        hr = *(const uint4*) b;   // dm, scales[12]
#pragma unroll
        for (int v = 0; v < NV; ++v) qr[v] = ((const uint4*) (b + 16 + 32 * jj))[v0 + v];
    };
    auto commit_w = [&](int64_t k0) {
        const int jj = (int) (k0 % 256) / 64;
        const float2 dm = __half22float2(*(const __half2*) &hr.x);
        int s, m;
        scale_min_k4(2 * jj + hi, hr.y, hr.z, hr.w, s, m);
        // dq_q4_k's formulas: d1 = dall * sc, m1 = dmin * m, value d1 * q - m1
        const float d1 = dm.x * (uint8_t) s, m1 = dm.y * (uint8_t) m;
        const int sh = 4 * hi;

#pragma unroll
        for (int v = 0; v < NV; ++v) {
            const uint8_t* q = (const uint8_t*) &qr[v];
            __align__(16) __half o[16];
#pragma unroll
            for (int l = 0; l < 16; ++l) o[l] = __float2half(d1 * ((q[l] >> sh) & 0xF) - m1);
            *(uint4*) &Ws[soff(wr, hi * 4 + 2 * (v0 + v))] = *(const uint4*) &o[0];
            *(uint4*) &Ws[soff(wr, hi * 4 + 2 * (v0 + v) + 1)] = *(const uint4*) &o[8];
        }
    };
    fetch_x(xr, Xb, L.n_embd, ne, 0);
    fetch_w(0);
    for (int64_t k0 = 0; k0 < L.n_embd; k0 += BK) {
        commit_x(Xs, xr);
        commit_w(k0);
        __syncthreads();
        if (k0 + BK < L.n_embd) { fetch_x(xr, Xb, L.n_embd, ne, k0 + BK); fetch_w(k0 + BK); }
        mma_tile(Xs, Ws, acc, ne);
        __syncthreads();
    }
    float (*E)[LDE] = (float (*)[LDE]) sm;
    store_acc(E, acc);
    __syncthreads();
    for (int i = threadIdx.x; i < TM * (BN / 2); i += NT) {
        const int r = i / (BN / 2), c = i % (BN / 2);
        if (r < ne) H[(row0 + r) * L.n_ff + rt * (BN / 2) + c] = silu_mul(E[r][2 * c], E[r][2 * c + 1]);
    }
}

// down (Q5_1 = 7 or Q8_0 = 8) -> D (FP32)
template<int DT>
__global__ void __launch_bounds__(NT) down_kernel(Tiles g, FusedExpertLayout L, const __half* __restrict__ H,
                                                  float* __restrict__ D) {
    extern __shared__ __align__(16) unsigned char sm[];
    __half* Xs = (__half*) sm;
    __half* Ws = (__half*) (sm + TM * BK * 2);
    int e, rt, ne;
    int64_t row0;
    locate(g, (int) (L.n_embd / BN), e, rt, ne, row0);
    Acc acc[2][2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.0f;
    // TPR threads per weight row: one of the step's two 32-value blocks (hi), NI of its values (from i0) each
    constexpr int NI = 32 / (TPR / 2);
    const int wr = threadIdx.x / TPR, hi = threadIdx.x % 2, i0 = (threadIdx.x % TPR) / 2 * NI;
    constexpr int BB = DT == 7 ? 24 : 34;   // block bytes
    const uint8_t* rowp = g.blob[e] + L.down_off + (size_t) (rt * BN + wr) * L.d_row;
    const __half* Hb = H + row0 * L.n_ff;
    XRegs xr;
    uint2 w5[3];      // Q5_1: d, m, qh, qs[16] (8-byte aligned)
    uint16_t w8[17];  // Q8_0: d, qs[32] (2-byte aligned)
    auto fetch_w = [&](int64_t k0) {
        const uint8_t* b = rowp + (size_t) (k0 / 32 + hi) * BB;
        if constexpr (DT == 7) {
            w5[0] = ((const uint2*) b)[0]; w5[1] = ((const uint2*) b)[1]; w5[2] = ((const uint2*) b)[2];
        } else {
#pragma unroll
            for (int i = 0; i < 17; ++i) w8[i] = ((const uint16_t*) b)[i];
        }
    };
    auto commit_w = [&]() {
        __align__(16) __half y[NI];
        if constexpr (DT == 7) {
            // dq_q5_1's formulas; value i and i + 16 come from byte i: this thread's bytes are i0 / 2 .. + NI / 2
            const float2 dm = __half22float2(*(const __half2*) &w5[0].x);
            const uint32_t qh = w5[0].y;
            const uint8_t* q = (const uint8_t*) &w5[1];
            const int b0 = i0 / 2;
#pragma unroll
            for (int l = 0; l < NI / 2; ++l) {
                const int i = b0 + l;
                const int xh_0 = ((qh >> (i + 0)) << 4) & 0x10, xh_1 = ((qh >> (i + 12))) & 0x10;
                y[l] = __float2half((float) ((q[i] & 0xf) | xh_0) * dm.x + dm.y);
                y[l + NI / 2] = __float2half((float) ((q[i] >> 4) | xh_1) * dm.x + dm.y);
            }
#pragma unroll
            for (int l = 0; l < NI / 16; ++l) {
                *(uint4*) &Ws[soff(wr, hi * 4 + b0 / 8 + l)] = *(const uint4*) &y[8 * l];
                *(uint4*) &Ws[soff(wr, hi * 4 + 2 + b0 / 8 + l)] = *(const uint4*) &y[NI / 2 + 8 * l];
            }
        } else {
            const float d = __half2float(*(const __half*) &w8[0]);
            const int8_t* q = (const int8_t*) &w8[1];
#pragma unroll
            for (int i = 0; i < NI; ++i) y[i] = __float2half((float) q[i0 + i] * d);
#pragma unroll
            for (int i = 0; i < NI / 8; ++i) *(uint4*) &Ws[soff(wr, hi * 4 + i0 / 8 + i)] = *(const uint4*) &y[8 * i];
        }
    };
    fetch_x(xr, Hb, L.n_ff, ne, 0);
    fetch_w(0);
    for (int64_t k0 = 0; k0 < L.n_ff; k0 += BK) {
        commit_x(Xs, xr);
        commit_w();
        __syncthreads();
        if (k0 + BK < L.n_ff) { fetch_x(xr, Hb, L.n_ff, ne, k0 + BK); fetch_w(k0 + BK); }
        mma_tile(Xs, Ws, acc, ne);
        __syncthreads();
    }
    float (*E)[LDE] = (float (*)[LDE]) sm;
    store_acc(E, acc);
    __syncthreads();
    for (int i = threadIdx.x; i < TM * BN; i += NT) {
        const int r = i / BN, c = i % BN;
        if (r < ne) D[(row0 + r) * L.n_embd + rt * BN + c] = E[r][c];
    }
}

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}
}  // namespace

bool fused_expert_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) noexcept {
    return gu_type == 12 && (d_type == 7 || d_type == 8) && n_embd % 256 == 0 && n_ff % BN == 0 && n_ff % BK == 0;
}

void fused_expert_run(const FusedExpertLayout& L, const FusedExpertGroup& in, const uint16_t* X, uint16_t* H, float* D,
                      void* stream) {
    if (in.n <= 0) return;
    if (!fused_expert_supported(L.gu_type, L.d_type, L.n_embd, L.n_ff) || in.n > kFusedExpertMax) {
        std::fprintf(stderr, "fused_expert_run: types %d/%d, %d experts\n", L.gu_type, L.d_type, in.n);
        std::exit(1);
    }
    static const bool once = [] {
        cudaFuncSetAttribute(gu_q4k_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        cudaFuncSetAttribute(down_kernel<7>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        cudaFuncSetAttribute(down_kernel<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        return true;
    }();
    (void) once;
    Tiles gu{}, dn{};
    int n = 0;
    for (int i = 0; i < in.n; ++i) {
        if (in.ne[i] <= 0) continue;
        gu.blob[n] = dn.blob[n] = in.blob[i];
        gu.row0[n] = dn.row0[n] = in.row0[i];
        gu.ne[n] = dn.ne[n] = in.ne[i];
        const int tt = (in.ne[i] + TM - 1) / TM;
        gu.tile0[n + 1] = gu.tile0[n] + tt * (int) (2 * L.n_ff / BN);
        dn.tile0[n + 1] = dn.tile0[n] + tt * (int) (L.n_embd / BN);
        ++n;
    }
    if (n == 0) return;
    gu.n = dn.n = n;
    const cudaStream_t s = (cudaStream_t) stream;
    gu_q4k_kernel<<<gu.tile0[n], NT, SMEM, s>>>(gu, L, (const __half*) X, (__half*) H);
    check("fused_expert gate/up");
    if (L.d_type == 7) down_kernel<7><<<dn.tile0[n], NT, SMEM, s>>>(dn, L, (const __half*) H, D);
    else down_kernel<8><<<dn.tile0[n], NT, SMEM, s>>>(dn, L, (const __half*) H, D);
    check("fused_expert down");
}

}  // namespace strata::kernels
