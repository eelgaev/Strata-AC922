#pragma once
// POWER9 VSX row kernels for the CPU expert share (STRATA_VSX_EXPERTS=1): UD-Q4_K_XL's Q4_K gate/up (Q8_K activations)
// and Q5_1 down (Q8_1 activations), each weight block decoded once for all the tokens routed to the expert. ggml's
// arithmetic per block (exact integer dots, the same scale products) summed in another order: ~2e-7 relative to
// ggml-cpu's vec_dot, not bitwise.
#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {
/// true on a POWER9 build with STRATA_VSX_EXPERTS set
bool kq_vsx_on() noexcept;
/// gate/up rows [r0, r1): ff[t][r] = silu(gate . x_t) * (up . x_t), act[t] = Q8_K rows of n_embd
void kq_vsx_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n_embd, const void* const* act, int nt,
                    float* const* ff, int r0, int r1);
/// Q5_1 rows [r0, r1) of row_bytes each (n_ff values, n_ff % 128 == 0): out[t][r] = row . h_t, hq[t] = Q8_1 rows
void kq_vsx_q51_rows(const uint8_t* W, size_t row_bytes, int n_ff, const void* const* hq, int nt, float* const* out,
                     int r0, int r1);
}  // namespace strata::kernels::cpu
