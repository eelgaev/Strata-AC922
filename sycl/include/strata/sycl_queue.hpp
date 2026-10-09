// include/strata/sycl_queue.hpp - the SYCL port's one addition to the engine's API surface.
//
// Every launcher takes `void* stream`, a cudaStream_t where null means the default stream. dpct migrates the
// cast to `(dpct::queue_ptr) stream` and dereferences it, so a null stream is a null sycl::queue* and a crash.
// q_of() is that cast with CUDA's null-stream meaning restored: the default in-order queue.
#pragma once
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdlib>

namespace strata {
inline sycl::queue* q_of(const void* stream) {
    return stream ? (sycl::queue*) stream : &dpct::get_in_order_queue();
}
}  // namespace strata

namespace strata {
// A large device fill as compute kernels of at most `chunk` bytes, each waited for.  On the Arc Pro B70 (xe driver) one
// queue.memset of many GiB runs on the blitter engine and times out ("Engine memory CAT error", GT reset); kernels do not.
// Small fills (<= chunk) and non-multiple-of-8 tails fall back to memset.
inline void big_fill_zero(sycl::queue& q, void* p, size_t bytes, size_t chunk = (size_t)256 << 20) {
    static const bool plain = [] { const char* v = std::getenv("STRATA_CHUNKED_FILL"); return v && v[0] == '0'; }();
    if (plain) { q.memset(p, 0, bytes); q.wait(); return; }
    uint8_t* b = (uint8_t*)p;
    size_t off = 0;
    while (off < bytes) {
        size_t n = bytes - off < chunk ? bytes - off : chunk;
        if (n >= (1u << 20) && ((uintptr_t)(b + off) & 7) == 0) {
            size_t w = n / 8;
            uint64_t* d = (uint64_t*)(b + off);
            q.parallel_for(sycl::range<1>(w), [=](sycl::id<1> i) { d[i] = 0; });
            if (n & 7) q.memset(b + off + w * 8, 0, n & 7);
        } else {
            q.memset(b + off, 0, n);
        }
        q.wait();
        off += n;
    }
}
}  // namespace strata

// ---- host memory a running kernel polls for the host's stores (the doorbell flags, the CPU experts' result rows).
// On an Arc A750 (Alchemist, i915) a kernel never sees a host store to ordinary host USM, with or without
// system-scope atomics: every ring wait runs to its bound and the window reads stale rows (NaN logits). Host memory
// from zeMemAllocHost with ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED does work there (sycl/probe/doorbell.cpp mode 2,
// "HANDSHAKE OK"), so those buffers come from it when the Level Zero headers are present (libze-dev), else from
// sycl::malloc_host. STRATA_HOST_UNCACHED=0 turns it off.
#if defined(__has_include)
#if __has_include(<level_zero/ze_api.h>) && __has_include(<sycl/ext/oneapi/backend/level_zero.hpp>)
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#define STRATA_HAVE_ZE 1
#endif
#endif
#include <cstdlib>
#include <mutex>
#include <unordered_set>
namespace strata {
namespace detail {
inline std::unordered_set<void*>& uncached_set() { static std::unordered_set<void*> s; return s; }
inline std::mutex& uncached_mu() { static std::mutex m; return m; }
}  // namespace detail
inline void* host_malloc_polled(size_t bytes, sycl::queue& q) {
#ifdef STRATA_HAVE_ZE
    static const bool on = [] { const char* v = std::getenv("STRATA_HOST_UNCACHED"); return !(v && v[0] == '0'); }();
    if (on && q.get_backend() == sycl::backend::ext_oneapi_level_zero) {
        auto ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
        ze_host_mem_alloc_desc_t d{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED};
        void* p = nullptr;
        if (zeMemAllocHost(ctx, &d, bytes, 64, &p) == ZE_RESULT_SUCCESS && p != nullptr) {
            std::lock_guard<std::mutex> lk(detail::uncached_mu());
            detail::uncached_set().insert(p);
            return p;
        }
    }
#endif
    return sycl::malloc_host(bytes, q);
}
inline void host_free_polled(void* p, sycl::queue& q) {
    if (p == nullptr) return;
#ifdef STRATA_HAVE_ZE
    {
        std::lock_guard<std::mutex> lk(detail::uncached_mu());
        if (detail::uncached_set().erase(p)) {
            zeMemFree(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context()), p);
            return;
        }
    }
#endif
    sycl::free(p, q);
}
}  // namespace strata

// The kernel driver of the Intel GPU ("xe", "i915", or "" when unknown), from sysfs.  The driver decides what the GPU
// may read: an Arc A750 (i915) reads ordinary host memory a kernel is handed; an Arc Pro B70 (xe) page-faults on it
// (ccs0, FaultType 0, address in the CPU's mmap range) and the card times the job out.
#include <filesystem>
#include <fstream>
#include <string>
namespace strata {
inline const std::string& intel_gpu_driver() {
    static const std::string drv = [] {
        std::error_code ec;
        for (const auto& e : std::filesystem::directory_iterator("/sys/class/drm", ec)) {
            const std::string n = e.path().filename().string();
            if (n.rfind("card", 0) != 0 || n.find('-') != std::string::npos) continue;
            std::ifstream vf(e.path() / "device" / "vendor");
            std::string vendor;
            if (!(vf >> vendor) || vendor != "0x8086") continue;
            const auto d = std::filesystem::read_symlink(e.path() / "device" / "driver", ec);
            if (!ec) return d.filename().string();
        }
        return std::string();
    }();
    return drv;
}
}  // namespace strata
