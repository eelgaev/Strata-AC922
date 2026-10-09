// include/strata/core/ep_twin.hpp - --ep-twins (opt-in): expert parallelism inside an NVLink pair.
//
// A layer split's stage runs on one GPU; while it runs, its NVLink partner is idle (its own stage comes later in the
// window).  With --ep-twins the pair's two expert caches hold half of each stage's cached experts each (the same set
// per stage as without, so the outputs do not change), and the partner - the stage's twin - computes the routed
// entries it holds while the stage's GPU computes the rest.  Per layer (and token group) the stage's verify graph
// pushes the experts' q8_1 input and the routed ids into the twin's mailbox; the twin's own captured graph waits for
// them, plans the entries it owns on the device, runs native_expert_grouped and pushes those rows back with a mask;
// the stage merges them into its expert rows before the combine, which then reads every row in entry order as
// before.  Every value is computed by the same kernels from the same inputs: bit-identical to the run without twins.
// The flags are stamped with a per-window epoch each graph advances when it starts (no resets between windows).
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

class EpTwin {
public:
    static constexpr int kMaxSteps = 128;   ///< (layers of a stage) x (token groups of a window)
    EpTwin() = default;
    ~EpTwin();
    EpTwin(const EpTwin&) = delete;
    EpTwin& operator=(const EpTwin&) = delete;

    /// The stage runs layers [lb, le) on `prim_dev`; the twin is `twin_dev` with the expert cache at `cache_base`
    /// (slot byte offsets `slot_off`).  `twin_res` (n_layers x n_expert): the slot in that cache of each expert the
    /// twin computes for this stage, -1 for every other.  Peer access between the two devices must be enabled.
    bool init(int prim_dev, int twin_dev, int64_t lb, int64_t le, int64_t n_expert, int k, int64_t n_embd, int max_t,
              const uint8_t* cache_base, const std::vector<uint64_t>& slot_off, const std::vector<int32_t>& twin_res,
              std::string& err);
    bool ready() const { return twin_dev_ >= 0; }
    int twin_device() const { return twin_dev_; }

    /// The twin's graph for a window of T tokens in G token groups (the stage's own split), captured on first use.
    bool capture(int T, int G, std::string& err);
    /// Launch it (after the stage's graph launch, so both run together).
    bool launch(int T, std::string& err);
    /// Wait for the twin's stream.
    bool sync(std::string& err);

    // ---- recorded into the STAGE's graph (on its stream, its device current) ----
    /// first node of every stage graph: the stage's window epoch
    void rec_begin(cudaStream_t s) const;
    /// step `step` (= (layer - lb) * G + group): the group's q8_1 rows (`n_tok` x n_embd/32 blocks of 36 bytes) and its
    /// routed ids (`n_ent` = n_tok x k) into the twin's mailbox, then the twin's flag
    void rec_push_input(int step, const void* xq, int n_tok, const int32_t* ids, int n_ent, cudaStream_t s) const;
    /// wait for the twin's rows of `step` and copy the ones it computed (its mask) into `parts` (entry rows)
    void rec_merge(int step, int grp, float* parts, int n_ent, cudaStream_t s) const;

private:
    int prim_dev_ = -1, twin_dev_ = -1;
    int64_t lb_ = 0, le_ = 0, n_expert_ = 0, n_embd_ = 0;
    int k_ = 0, max_t_ = 0;
    int64_t capx_ = 0, plan_i32_ = 0, xq_row_ = 0;
    cudaStream_t ts_ = nullptr;                      ///< the twin's stream
    // twin device
    const uint8_t* cache_base_ = nullptr;
    unsigned long long* slot_off_d_ = nullptr;
    int32_t* res_d_ = nullptr;                       ///< twin_res on the twin
    uint8_t* in_xq_ = nullptr;                       ///< [step][max_t x xq_row_]
    int32_t* in_ids_ = nullptr;                      ///< [step][capx]
    int* in_flag_ = nullptr;                         ///< [step]
    int* t_epoch_ = nullptr;
    int* t_done_ = nullptr;
    int32_t* plan_ = nullptr;
    void* scratch_ = nullptr;
    float* rows_ = nullptr;                          ///< capx x n_embd
    // stage device
    float* out_rows_ = nullptr;                      ///< [group][capx x n_embd]
    int32_t* out_mask_ = nullptr;                    ///< [group][capx]
    int* out_flag_ = nullptr;                        ///< [step]
    int* p_epoch_ = nullptr;
    int* p_done_ = nullptr;
    cudaGraphExec_t exec_[17] = {};
};

}  // namespace strata::core
