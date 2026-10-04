// src/kernels/cpu/kq_vsx.cpp - see kq_vsx.hpp. Measured on an IBM AC922 (one POWER9 core, 3.8 GHz, vs ggml-cpu's POWER9
// vec_dot): 640 Q4_K rows 1.7x / 2.1x / 2.3x at 1 / 2 / 3 tokens, 2560 Q5_1 rows 4.3x / 6.5x / 7.7x.
#include "strata/kernels/cpu/kq_vsx.hpp"

#include <cstdlib>
#include <cstring>
#include <cmath>

#if defined(__POWER9_VECTOR__)
#include <altivec.h>
#undef vector
#undef bool
#undef pixel

namespace strata::kernels::cpu {
namespace {
struct bq4K { uint16_t d, dmin; uint8_t scales[12]; uint8_t qs[128]; };
struct bq8K { float d; int8_t qs[256]; int16_t bsums[16]; };
struct bq51 { uint16_t d, m; uint8_t qh[4]; uint8_t qs[16]; };
struct bq81 { uint16_t d, s; int8_t qs[32]; };
static_assert(sizeof(bq4K) == 144 && sizeof(bq8K) == 292 && sizeof(bq51) == 24 && sizeof(bq81) == 36, "layout");

typedef __vector signed char vs8;
typedef __vector unsigned char vu8;
typedef __vector signed short vs16;
typedef __vector unsigned short vu16;
typedef __vector signed int vs32;
typedef __vector unsigned int vu32;
typedef __vector float vf32;
inline float h2f(uint16_t h) {
    const vu16 v = {h, 0, 0, 0, 0, 0, 0, 0};
    return vec_extract_fp32_from_shorth(v)[0];
}

// four int32x4 partial-sum vectors -> one vector of their four totals
static inline vs32 reduce4(vs32 a, vs32 b, vs32 c, vs32 d) {
    const vs32 ab = vec_add(vec_mergeh(a, b), vec_mergel(a, b));   // a0+a2, b0+b2, a1+a3, b1+b3
    const vs32 cd = vec_add(vec_mergeh(c, d), vec_mergel(c, d));
    const vs32 lo = (vs32) vec_mergeh((__vector unsigned long long) ab, (__vector unsigned long long) cd);   // a0+a2 b0+b2 c0+c2 d0+d2
    const vs32 hi = (vs32) vec_mergel((__vector unsigned long long) ab, (__vector unsigned long long) cd);   // a1+a3 b1+b3 c1+c3 d1+d3
    return vec_add(lo, hi);
}
static inline float hsum(vf32 v) {
    v = vec_add(v, vec_sld(v, v, 8));
    v = vec_add(v, vec_sld(v, v, 4));
    return vec_extract(v, 0);
}

// ---- Q4_K x Q8_K: rows [r0, r1) of a row-major Q4_K matrix (nsb superblocks per row) for nt tokens
constexpr int MT = 8;   // MAXT
void q4k_rows(const uint8_t* W, size_t row_bytes, int nsb, const bq8K* const* y, int nt, float* const* out, int r0, int r1) {
    const vu8 m4 = vec_splats((unsigned char) 0xF), s4 = vec_splats((unsigned char) 4);
    const vs32 z = vec_splats(0);
    for (int r = r0; r < r1; ++r) {
        const bq4K* x = (const bq4K*) (W + (size_t) r * row_bytes);
        vf32 acc[MT];
        for (int t = 0; t < nt; ++t) acc[t] = vec_splats(0.0f);
        for (int b = 0; b < nsb; ++b) {
            const bq4K& xb = x[b];
            // scales and mins: ggml's utmp unpacking in GPRs, then one vector load of the 16 bytes
            uint32_t u[4];
            memcpy(u, xb.scales, 12);
            u[3] = ((u[2] >> 4) & 0x0f0f0f0f) | (((u[1] >> 6) & 0x03030303) << 4);
            const uint32_t ua = u[1] & 0x3f3f3f3f;
            u[1] = (u[2] & 0x0f0f0f0f) | (((u[0] >> 6) & 0x03030303) << 4);
            u[2] = ua;
            u[0] &= 0x3f3f3f3f;
            const vu8 utm = vec_xl(0, (const unsigned char*) u);   // scales 0..7, mins 0..7 (all < 64)
            const vs16 scs = (vs16) vec_mergeh(utm, vec_splats((unsigned char) 0)), mns = (vs16) vec_mergel(utm, vec_splats((unsigned char) 0));
            const vs32 SC0 = vec_unpackh(scs), SC1 = vec_unpackl(scs);
            const vs16 MN0 = vec_mergeh(mns, mns), MN1 = vec_mergel(mns, mns);
            const vu16 hd = {xb.d, xb.dmin, 0, 0, 0, 0, 0, 0};
            const vf32 fd = vec_extract_fp32_from_shorth(hd);
            const float xd = fd[0], xm = fd[1];
            vu8 q[16];   // per 64-value chunk j: lo0, lo1, hi0, hi1
#pragma GCC unroll 4
            for (int j = 0; j < 4; ++j) {
                const vu8 a = vec_xl(0, xb.qs + 32 * j), c = vec_xl(16, xb.qs + 32 * j);
                q[4 * j + 0] = vec_and(a, m4); q[4 * j + 1] = vec_and(c, m4);
                q[4 * j + 2] = vec_sr(a, s4);  q[4 * j + 3] = vec_sr(c, s4);
            }
            for (int t = 0; t < nt; ++t) {
                const bq8K& yb = y[t][b];
                vs32 P[8];
#pragma GCC unroll 4
                for (int j = 0; j < 4; ++j) {
                    const int8_t* yq = yb.qs + 64 * j;
                    P[2 * j] = vec_msum(vec_xl(16, yq), q[4 * j + 1], vec_msum(vec_xl(0, yq), q[4 * j + 0], z));
                    P[2 * j + 1] = vec_msum(vec_xl(48, yq), q[4 * j + 3], vec_msum(vec_xl(32, yq), q[4 * j + 2], z));
                }
                const vs32 R0 = reduce4(P[0], P[1], P[2], P[3]), R1 = reduce4(P[4], P[5], P[6], P[7]);
                const vs32 isum = vec_add(vec_mul(R0, SC0), vec_mul(R1, SC1));
                const vs16 B0 = vec_xl(0, yb.bsums), B1 = vec_xl(16, yb.bsums);
                const vs32 msum = vec_msum(MN1, B1, vec_msum(MN0, B0, z));
                const vf32 vd = vec_splats(xd * yb.d), vm = vec_splats(xm * yb.d);
                acc[t] = vec_madd(vec_ctf(isum, 0), vd, acc[t]);
                acc[t] = vec_nmsub(vec_ctf(msum, 0), vm, acc[t]);
            }
        }
        for (int t = 0; t < nt; ++t) out[t][r] = hsum(acc[t]);
    }
}

// ---- Q5_1 x Q8_1: rows [r0, r1), nb blocks per row (a multiple of 4), nt tokens
void q51_rows(const uint8_t* W, size_t row_bytes, int nb, const bq81* const* y, int nt, float* const* out, int r0, int r1) {
    const vu8 m4 = vec_splats((unsigned char) 0xF), s4 = vec_splats((unsigned char) 4), b10 = vec_splats((unsigned char) 0x10);
    const vu8 bits = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    const vu8 sel0 = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1}, sel1 = {2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3};
    const vs32 z = vec_splats(0);
    // per token: the activation scales d and s as floats, four blocks to a vector
    float yd[MT][32], ys[MT][32];
    for (int t = 0; t < nt; ++t)
        for (int b = 0; b < nb; ++b) { yd[t][b] = h2f(y[t][b].d); ys[t][b] = h2f(y[t][b].s); }
    for (int r = r0; r < r1; ++r) {
        const bq51* x = (const bq51*) (W + (size_t) r * row_bytes);
        vf32 acc[MT];
        for (int t = 0; t < nt; ++t) acc[t] = vec_splats(0.0f);
        for (int b = 0; b < nb; b += 4) {
            // the four 24-byte blocks as six vectors; d/m and qh gathered by permutes
            const uint8_t* g = (const uint8_t*) (x + b);
            const vu8 V0 = vec_xl(0, g), V1 = vec_xl(16, g), V2 = vec_xl(32, g), V3 = vec_xl(48, g), V4 = vec_xl(64, g), V5 = vec_xl(80, g);
            const vu8 sp = {0, 1, 24, 25, 2, 3, 26, 27, 4, 5, 6, 7, 28, 29, 30, 31};   // d_a d_b m_a m_b qh_a qh_b
            const vu8 P01 = vec_perm(V0, V1, sp), P23 = vec_perm(V3, V4, sp);
            const vu8 sdm = {0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23};
            const vu8 sqh = {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31};
            const vu16 dm = (vu16) vec_perm(P01, P23, sdm);    // d0 d1 d2 d3 m0 m1 m2 m3
            const vu8 QH = vec_perm(P01, P23, sqh);            // qh0 qh1 qh2 qh3 (4 bytes each)
            const vf32 D = vec_extract_fp32_from_shorth(dm), M = vec_extract_fp32_from_shortl(dm);
            const vu8 smid = {8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
            const vu8 QS[4] = {vec_perm(V0, V1, smid), V2, vec_perm(V3, V4, smid), V5};
            vu8 qa[4], qb[4];
#pragma GCC unroll 4
            for (int k = 0; k < 4; ++k) {
                // byte j of the expansion holds bit j of qh (lanes 0-15: bits 0-15, then bits 16-31), moved to 0x10
                const vu8 s0 = {(unsigned char) (4 * k), (unsigned char) (4 * k), (unsigned char) (4 * k), (unsigned char) (4 * k),
                                (unsigned char) (4 * k), (unsigned char) (4 * k), (unsigned char) (4 * k), (unsigned char) (4 * k),
                                (unsigned char) (4 * k + 1), (unsigned char) (4 * k + 1), (unsigned char) (4 * k + 1), (unsigned char) (4 * k + 1),
                                (unsigned char) (4 * k + 1), (unsigned char) (4 * k + 1), (unsigned char) (4 * k + 1), (unsigned char) (4 * k + 1)};
                const vu8 s1 = vec_add(s0, vec_splats((unsigned char) 2));
                const vu8 h0 = vec_and(vec_cmpeq(vec_and(vec_perm(QH, QH, s0), bits), bits), b10);
                const vu8 h1 = vec_and(vec_cmpeq(vec_and(vec_perm(QH, QH, s1), bits), bits), b10);
                qa[k] = vec_or(vec_and(QS[k], m4), h0);
                qb[k] = vec_or(vec_sr(QS[k], s4), h1);
            }
            for (int t = 0; t < nt; ++t) {
                vs32 S[4];
#pragma GCC unroll 4
                for (int k = 0; k < 4; ++k) {
                    const int8_t* yq = y[t][b + k].qs;
                    S[k] = vec_msum(vec_xl(16, yq), qb[k], vec_msum(vec_xl(0, yq), qa[k], z));
                }
                const vs32 tot = reduce4(S[0], S[1], S[2], S[3]);
                acc[t] = vec_madd(vec_ctf(tot, 0), vec_mul(D, vec_xl(0, yd[t] + b)), acc[t]);
                acc[t] = vec_madd(M, vec_xl(0, ys[t] + b), acc[t]);
            }
        }
        for (int t = 0; t < nt; ++t) out[t][r] = hsum(acc[t]);
    }
}

}  // namespace

bool kq_vsx_on() noexcept {
    static const bool on = [] { const char* v = std::getenv("STRATA_VSX_EXPERTS"); return v != nullptr && v[0] && v[0] != '0'; }();
    return on;
}
void kq_vsx_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n_embd, const void* const* act, int nt,
                    float* const* ff, int r0, int r1) {
    const bq8K* y[MT];
    for (int t = 0; t < nt; ++t) y[t] = (const bq8K*) act[t];
    float g[MT], u[MT];
    float* gp[MT];
    float* up[MT];
    for (int r = r0; r < r1; ++r) {
        for (int t = 0; t < nt; ++t) { gp[t] = g + t - r; up[t] = u + t - r; }   // out[t][r] lands in g[t] / u[t]
        q4k_rows(blob, gu_row, n_embd / 256, y, nt, gp, r, r + 1);
        q4k_rows(blob + up_off, gu_row, n_embd / 256, y, nt, up, r, r + 1);
        for (int t = 0; t < nt; ++t) ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
    }
}
void kq_vsx_q51_rows(const uint8_t* W, size_t row_bytes, int n_ff, const void* const* hq, int nt, float* const* out,
                     int r0, int r1) {
    const bq81* y[MT];
    for (int t = 0; t < nt; ++t) y[t] = (const bq81*) hq[t];
    q51_rows(W, row_bytes, n_ff / 32, y, nt, out, r0, r1);
}
}  // namespace strata::kernels::cpu
#else
namespace strata::kernels::cpu {
bool kq_vsx_on() noexcept { return false; }
void kq_vsx_gu_rows(const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) { std::abort(); }
void kq_vsx_q51_rows(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) { std::abort(); }
}  // namespace strata::kernels::cpu
#endif
