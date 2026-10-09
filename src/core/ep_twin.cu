// src/core/ep_twin.cu - see include/strata/core/ep_twin.hpp.
#include "strata/core/ep_twin.hpp"
#include "strata/core/on_device.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cstdio>

namespace strata::core {
namespace {

constexpr int kPlanMax = 128;   // entries per step: kVerifyMaxT (8) tokens x 10 routed, as resident_plan

__global__ void ep_epoch_kernel(int* epoch) { epoch[0] += 1; }

__device__ __forceinline__ void ep_raise(int* done, int* flag, const int* epoch) {
    // after the block's writes: one system fence per block; the last block raises the flag
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        if (atomicAdd(done, 1) == (int) (gridDim.x * gridDim.y) - 1) {
            *done = 0;
            __threadfence_system();
            *(volatile int*) flag = *(volatile const int*) epoch;
        }
    }
}

// the group's q8_1 rows (16-byte units) and ids into the twin's mailbox
__global__ void ep_push_input_kernel(const uint4* __restrict__ xq, int xq16, const int32_t* __restrict__ ids, int n_ent,
                                     uint4* m_xq, int32_t* m_ids, int* flag, const int* epoch, int* done) {
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
    for (int i = i0; i < xq16; i += stride) m_xq[i] = xq[i];
    for (int i = i0; i < n_ent; i += stride) m_ids[i] = ids[i];
    ep_raise(done, flag, epoch);
}

__global__ void ep_wait_kernel(const int* flag, const int* epoch) {
    const int want = *(volatile const int*) epoch;
    while (*(volatile const int*) flag < want) {}
    __threadfence_system();
}

// resident_plan (verify_kernels.cu) without the all-resident bail-out: the entries whose expert this twin holds form
// the groups, every other entry is skipped.  Same plan layout.
__global__ void __launch_bounds__(kPlanMax) ep_owned_plan_kernel(const int32_t* __restrict__ ids, int n, int k,
                                                                 const int32_t* __restrict__ res, int n_expert,
                                                                 const uint8_t* cache_base,
                                                                 const unsigned long long* slot_off,
                                                                 int32_t* __restrict__ pl, long long capx) {
    __shared__ int32_t s_ids[kPlanMax];
    __shared__ int32_t s_excl[kPlanMax];
    __shared__ int32_t s_wsum[4];
    const int tid = threadIdx.x;
    int32_t eid = -1, slot = -1;
    if (tid < n) {
        eid = ids[tid];
        s_ids[tid] = eid;
        slot = (eid >= 0 && eid < n_expert) ? res[eid] : -1;
    }
    __syncthreads();
    int first_j = tid, rank_in_group = 0, count_same = 0;
    if (tid < n) {
        for (int j = 0; j < n; ++j)
            if (s_ids[j] == eid) {
                if (j < first_j) first_j = j;
                if (j < tid) ++rank_in_group;
                ++count_same;
            }
    }
    const bool is_first = tid < n && first_j == tid && slot >= 0;
    const int my_pack = ((is_first ? count_same : 0) << 16) | (is_first ? 1 : 0);
    const int lane = tid & 31, warp = tid >> 5;
    int pref = my_pack;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const int up = __shfl_up_sync(0xffffffffu, pref, d);
        if (lane >= d) pref += up;
    }
    s_excl[tid] = pref - my_pack;
    if (lane == 31) s_wsum[warp] = pref;
    __syncthreads();
    int32_t* counts = pl;
    int32_t* start = pl + 4;
    int32_t* dst = start + capx + 1;
    int32_t* tok = dst + capx;
    const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
    unsigned long long* ptr = (unsigned long long*) (pl + ptr_off);
    if (is_first) {
        int tot = s_excl[tid];
#pragma unroll
        for (int w = 0; w < 4; ++w)
            if (w < warp) tot += s_wsum[w];
        ptr[tot & 0xffff] = (unsigned long long) (cache_base + slot_off[slot]);
        start[tot & 0xffff] = tot >> 16;
    }
    if (tid < n && slot >= 0) {
        int fj_tot = s_excl[first_j];
#pragma unroll
        for (int w = 0; w < 4; ++w)
            if (w < (first_j >> 5)) fj_tot += s_wsum[w];
        const int out_idx = (fj_tot >> 16) + rank_in_group;
        dst[out_idx] = tid;
        tok[out_idx] = tid / k;
    }
    if (tid == 0) {
        const int sum = s_wsum[0] + s_wsum[1] + s_wsum[2] + s_wsum[3];
        const int groups = sum & 0xffff, entries = sum >> 16;
        start[groups] = entries;
        counts[0] = groups;
        counts[1] = entries;
        counts[2] = 0;
    }
}

// the rows of the entries this twin holds, and the mask of which they are, into the stage's mailbox; grid (bx, n_ent)
__global__ void ep_push_rows_kernel(const float4* __restrict__ rows, const int32_t* __restrict__ ids,
                                    const int32_t* __restrict__ res, int n_expert, int row4, float4* m_rows,
                                    int32_t* m_mask, int* flag, const int* epoch, int* done) {
    const int e = blockIdx.y;
    const int32_t eid = ids[e];
    const bool mine = eid >= 0 && eid < n_expert && res[eid] >= 0;
    if (blockIdx.x == 0 && threadIdx.x == 0) m_mask[e] = mine ? 1 : 0;
    if (mine)
        for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < row4; c += gridDim.x * blockDim.x)
            m_rows[(int64_t) e * row4 + c] = rows[(int64_t) e * row4 + c];
    ep_raise(done, flag, epoch);
}

// the stage side: wait for the twin's rows, copy the masked ones into the expert rows; grid (bx, n_ent)
__global__ void ep_merge_kernel(float4* parts, const float4* __restrict__ m_rows, const int32_t* __restrict__ m_mask,
                                int row4, const int* flag, const int* epoch) {
    if (threadIdx.x == 0) {
        const int want = *(volatile const int*) epoch;
        while (*(volatile const int*) flag < want) {}
    }
    __syncthreads();
    const int e = blockIdx.y;
    if (((volatile const int32_t*) m_mask)[e] == 0) return;
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < row4; c += gridDim.x * blockDim.x)
        parts[(int64_t) e * row4 + c] = __ldcv(m_rows + (int64_t) e * row4 + c);   // written by the twin: past L1
}

bool ok(cudaError_t e, const char* what, std::string& err) {
    if (e == cudaSuccess) return true;
    err = std::string("ep twin: ") + what + ": " + cudaGetErrorString(e);
    (void) cudaGetLastError();
    return false;
}

}  // namespace

EpTwin::~EpTwin() {
    if (twin_dev_ >= 0) {
        const OnDevice on(twin_dev_);
        if (ts_) cudaStreamSynchronize(ts_);
        for (auto& x : exec_) if (x) cudaGraphExecDestroy(x);
        for (void* p : {(void*) slot_off_d_, (void*) res_d_, (void*) in_xq_, (void*) in_ids_, (void*) in_flag_, (void*) t_epoch_,
                        (void*) t_done_, (void*) plan_, scratch_, (void*) rows_})
            if (p) cudaFree(p);
        if (ts_) cudaStreamDestroy(ts_);
    }
    if (prim_dev_ >= 0) {
        const OnDevice on(prim_dev_);
        for (void* p : {(void*) out_rows_, (void*) out_mask_, (void*) out_flag_, (void*) p_epoch_, (void*) p_done_})
            if (p) cudaFree(p);
    }
}

bool EpTwin::init(int prim_dev, int twin_dev, int64_t lb, int64_t le, int64_t n_expert, int k, int64_t n_embd, int max_t,
                  const uint8_t* cache_base, const std::vector<uint64_t>& slot_off, const std::vector<int32_t>& twin_res,
                  std::string& err) {
    if (k * max_t > kPlanMax || max_t > 16) { err = "ep twin: window too large"; return false; }
    lb_ = lb; le_ = le; n_expert_ = n_expert; k_ = k; n_embd_ = n_embd; max_t_ = max_t;
    capx_ = (int64_t) max_t * k;
    plan_i32_ = (((4 + (capx_ + 1) + 2 * capx_) + 1) & ~1ll) + 6 * capx_ + 64;
    xq_row_ = n_embd / 32 * 36;
    cache_base_ = cache_base;
    {
        const OnDevice on(twin_dev);
        if (!ok(cudaStreamCreateWithFlags(&ts_, cudaStreamNonBlocking), "stream", err) ||
            !ok(cudaMalloc((void**) &slot_off_d_, slot_off.size() * 8), "alloc", err) ||
            !ok(cudaMemcpy(slot_off_d_, slot_off.data(), slot_off.size() * 8, cudaMemcpyHostToDevice), "slot offsets", err) ||
            !ok(cudaMalloc((void**) &res_d_, twin_res.size() * 4), "alloc", err) ||
            !ok(cudaMemcpy(res_d_, twin_res.data(), twin_res.size() * 4, cudaMemcpyHostToDevice), "residency", err) ||
            !ok(cudaMalloc((void**) &in_xq_, (size_t) kMaxSteps * max_t * xq_row_), "alloc", err) ||
            !ok(cudaMalloc((void**) &in_ids_, (size_t) kMaxSteps * capx_ * 4), "alloc", err) ||
            !ok(cudaMalloc((void**) &in_flag_, kMaxSteps * 4), "alloc", err) ||
            !ok(cudaMemset(in_flag_, 0, kMaxSteps * 4), "memset", err) ||
            !ok(cudaMalloc((void**) &t_epoch_, 4), "alloc", err) || !ok(cudaMemset(t_epoch_, 0, 4), "memset", err) ||
            !ok(cudaMalloc((void**) &t_done_, 64), "alloc", err) || !ok(cudaMemset(t_done_, 0, 64), "memset", err) ||
            !ok(cudaMalloc((void**) &plan_, (size_t) plan_i32_ * 4), "alloc", err) ||
            !ok(cudaMalloc(&scratch_, strata::kernels::native_expert_scratch_bytes(capx_, 640)), "alloc", err) ||
            !ok(cudaMalloc((void**) &rows_, (size_t) capx_ * n_embd * 4), "alloc", err))
            return false;
        cudaDeviceSynchronize();
    }
    {
        const OnDevice on(prim_dev);
        if (!ok(cudaMalloc((void**) &out_rows_, (size_t) 2 * capx_ * n_embd * 4), "alloc", err) ||
            !ok(cudaMalloc((void**) &out_mask_, (size_t) 2 * capx_ * 4), "alloc", err) ||
            !ok(cudaMalloc((void**) &out_flag_, kMaxSteps * 4), "alloc", err) ||
            !ok(cudaMemset(out_flag_, 0, kMaxSteps * 4), "memset", err) ||
            !ok(cudaMalloc((void**) &p_epoch_, 4), "alloc", err) || !ok(cudaMemset(p_epoch_, 0, 4), "memset", err) ||
            !ok(cudaMalloc((void**) &p_done_, 64), "alloc", err) || !ok(cudaMemset(p_done_, 0, 64), "memset", err))
            return false;
        cudaDeviceSynchronize();
    }
    prim_dev_ = prim_dev;
    twin_dev_ = twin_dev;
    return true;
}

bool EpTwin::capture(int T, int G, std::string& err) {
    if (T < 1 || T > max_t_) { err = "ep twin: window size out of range"; return false; }
    if (exec_[T] != nullptr) return true;
    if ((le_ - lb_) * G > kMaxSteps) { err = "ep twin: too many steps"; return false; }
    const OnDevice on(twin_dev_);
    const auto& lay = strata::kernels::cpu::expert_layout();
    const int tb[2] = {0, (T + 1) / 2}, te[2] = {G == 2 ? (T + 1) / 2 : T, T};
    const int row4 = (int) (n_embd_ / 4);
    if (!ok(cudaStreamBeginCapture(ts_, cudaStreamCaptureModeThreadLocal), "begin capture", err)) return false;
    ep_epoch_kernel<<<1, 1, 0, ts_>>>(t_epoch_);
    for (int64_t l = lb_; l < le_; ++l)
        for (int grp = 0; grp < G; ++grp) {
            const int step = (int) ((l - lb_) * G + grp), n = te[grp] - tb[grp], ne = n * k_;
            const int32_t* ids = in_ids_ + (size_t) step * capx_;
            const int32_t* res = res_d_ + l * n_expert_;
            ep_wait_kernel<<<1, 1, 0, ts_>>>(in_flag_ + step, t_epoch_);
            ep_owned_plan_kernel<<<1, kPlanMax, 0, ts_>>>(ids, ne, k_, res, (int) n_expert_, cache_base_, slot_off_d_,
                                                          plan_, capx_);
            const long long ptr_off = ((4 + (capx_ + 1) + 2 * capx_) + 1) & ~1ll;
            const auto& f = lay.fmt[(size_t) l];
            const strata::kernels::NativeExpertLayout L = strata::kernels::native_expert_layout(f.gu_type, f.d_type,
                                                                                                f.n_embd, f.n_ff);
            strata::kernels::native_expert_grouped(L, (const unsigned long long*) (plan_ + ptr_off), plan_ + 4, plan_,
                                                   plan_ + 4 + capx_ + 1, plan_ + 4 + capx_ + 1 + capx_, ne, ne,
                                                   in_xq_ + (size_t) step * max_t_ * xq_row_, scratch_, rows_, ts_);
            ep_push_rows_kernel<<<dim3(2, (unsigned) ne), 320, 0, ts_>>>(
                (const float4*) rows_, ids, res, (int) n_expert_, row4,
                (float4*) (out_rows_ + (size_t) grp * capx_ * n_embd_), out_mask_ + (size_t) grp * capx_,
                out_flag_ + step, t_epoch_, t_done_);
        }
    cudaGraph_t g = nullptr;
    if (!ok(cudaStreamEndCapture(ts_, &g), "end capture", err)) return false;
    const cudaError_t ie = cudaGraphInstantiate(&exec_[T], g, 0);
    cudaGraphDestroy(g);
    return ok(ie, "instantiate", err);
}

bool EpTwin::launch(int T, std::string& err) {
    const OnDevice on(twin_dev_);
    return ok(cudaGraphLaunch(exec_[T], ts_), "launch", err);
}

bool EpTwin::sync(std::string& err) {
    const OnDevice on(twin_dev_);
    return ok(cudaStreamSynchronize(ts_), "sync", err);
}

void EpTwin::rec_begin(cudaStream_t s) const { ep_epoch_kernel<<<1, 1, 0, s>>>(p_epoch_); }

void EpTwin::rec_push_input(int step, const void* xq, int n_tok, const int32_t* ids, int n_ent, cudaStream_t s) const {
    ep_push_input_kernel<<<4, 256, 0, s>>>((const uint4*) xq, (int) (n_tok * xq_row_ / 16), ids, n_ent,
                                           (uint4*) (in_xq_ + (size_t) step * max_t_ * xq_row_),
                                           in_ids_ + (size_t) step * capx_, in_flag_ + step, p_epoch_, p_done_);
}

void EpTwin::rec_merge(int step, int grp, float* parts, int n_ent, cudaStream_t s) const {
    ep_merge_kernel<<<dim3(2, (unsigned) n_ent), 320, 0, s>>>((float4*) parts,
                                                             (const float4*) (out_rows_ + (size_t) grp * capx_ * n_embd_),
                                                             out_mask_ + (size_t) grp * capx_, (int) (n_embd_ / 4),
                                                             out_flag_ + step, p_epoch_);
}

}  // namespace strata::core
