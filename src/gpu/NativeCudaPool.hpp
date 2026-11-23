#pragma once
// RTX 4070 SUPER / 12 GiB k28/s6 native CUDA Candidate-D buffer pool.
// No SYCL, AdaptiveCpp or OpenSSL headers: future foundation for replacing
// GpuBufferPool + GpuPipeline in the CUDA-only executable.
#include "gpu/NativeCudaRuntime.hpp"
#include <cstddef>
#include <cstdint>

namespace pos2gpu {
struct NativeCudaPoolPlan {
    static constexpr int k = 28, strength = 6;
    uint64_t total_xs = 0;
    uint64_t capacity = 0;
    size_t storage_bytes = 0;
    size_t pair_a_bytes = 0;
    size_t pair_b_bytes = 0;
    size_t xs_sort_cub_bytes = 0;
    size_t sort_and_index_scratch_bytes = 0;
    size_t pinned_bytes = 0;
    size_t index_t1_bytes = 0;
    size_t index_t23_bytes = 0;
    size_t device_bytes() const noexcept {
        return storage_bytes+pair_a_bytes+pair_b_bytes+
               sort_and_index_scratch_bytes+sizeof(uint64_t);
    }
};

// CUB sizing uses CUDA device state but does not allocate GPU memory.
// Full Candidate-D sizing matches the proven SYCL pooled path.
NativeCudaPoolPlan native_cuda_pool_plan(int cuda_ordinal);

class NativeCudaPool final {
public:
    explicit NativeCudaPool(int cuda_ordinal, size_t margin_bytes = 128ull<<20);
    ~NativeCudaPool() noexcept;
    NativeCudaPool(NativeCudaPool const&) = delete;
    NativeCudaPool& operator=(NativeCudaPool const&) = delete;

    NativeCudaPoolPlan const& plan() const noexcept { return plan_; }
    NativeCudaSession& session() noexcept { return session_; }

    // Allocation-lifetime alias contract, valid only after earlier phases
    // complete on the CUDA default stream (native kernels synchronize).
    void* d_storage() const noexcept { return storage_; }
    void* d_pair_b() const noexcept { return pair_b_; }
    void* d_scratch() const noexcept { return scratch_; }
    uint64_t* d_count() const noexcept { return count_; }
    void* ensure_pair_a();
    uint64_t* ensure_pinned();
private:
    NativeCudaSession session_;
    NativeCudaPoolPlan plan_;
    void* storage_ = nullptr;
    void* pair_a_ = nullptr;
    void* pair_b_ = nullptr;
    void* scratch_ = nullptr;
    uint64_t* count_ = nullptr;
    uint64_t* pinned_ = nullptr;
};
} // namespace pos2gpu
