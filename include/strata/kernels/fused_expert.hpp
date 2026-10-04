#pragma once
// Fused prefill expert products (STRATA_FUSED_EXPERTS=1): the GGUF blocks of a native pack's expert are dequantized
// tile by tile into shared memory and multiplied on the FP16 tensor cores (Volta wmma, FP32 accumulation), so the
// FP16 copy of the expert is never written to or read from VRAM, and up to kFusedExpertMax experts share one launch.
// The weights are the same FP16 values iq_dequant_*_f16 produces; only the summation order differs from cuBLAS.
#include <cstddef>
#include <cstdint>

namespace strata::kernels {

constexpr int kFusedExpertMax = 32;

struct FusedExpertLayout {
    int gu_type = 0, d_type = 0;
    int64_t n_embd = 0, n_ff = 0;
    size_t gu_row = 0, d_row = 0, up_off = 0, down_off = 0;
};

struct FusedExpertGroup {
    const uint8_t* blob[kFusedExpertMax] = {};
    int64_t row0[kFusedExpertMax] = {};   // first row of the expert's tokens in X / H / D
    int ne[kFusedExpertMax] = {};         // its token count
    int n = 0;
};

// gate/up Q4_K or Q5_K with down Q5_1 or Q8_0, n_embd % 256 == 0, n_ff % 64 == 0
bool fused_expert_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) noexcept;

// H[rows][n_ff] (FP16) = SwiGLU of X[rows][n_embd] (FP16) times gate/up (interleaved as iq_dequant_gu_f16),
// then D[rows][n_embd] (FP32) = H times down, for every expert of `g`
void fused_expert_run(const FusedExpertLayout& L, const FusedExpertGroup& g, const uint16_t* X, uint16_t* H, float* D,
                      void* stream);

}  // namespace strata::kernels
