// Native Candidate-D layout and CUDA Runtime-managed pooled buffers.
// The pool is intentionally separate from the hybrid SYCL implementation.
#include "gpu/NativeCudaPool.hpp"
#include "gpu/SortCubInternal.cuh"
#include "gpu/XsCandidateGpu.hpp"
#include "host/PoolSizing.hpp"
#include "host/CompactDLayout.hpp"
#include <algorithm>
#include <stdexcept>
#include <string>
#include <cstdio>

namespace pos2gpu {
NativeCudaPoolPlan native_cuda_pool_plan(int cuda_ordinal) {
    // Initialize the same device that will own all CUB queries/allocations.
    cudaError_t const set_error = cudaSetDevice(cuda_ordinal);
    if (set_error != cudaSuccess)
        throw std::runtime_error(std::string("native CUDA pool device selection: ") +
                                 cudaGetErrorString(set_error));
    NativeCudaPoolPlan p{};
    p.total_xs = 1ull << p.k;
    p.capacity = match_phase_capacity(p.k,2);
    if (p.capacity > UINT32_MAX || p.total_xs > UINT32_MAX)
        throw std::runtime_error("native CUDA pool: u32 index capacity exceeded");

    p.storage_bytes = std::max(size_t(p.total_xs)*sizeof(XsCandidateGpu),
                               size_t(p.capacity)*3*sizeof(uint32_t));
    p.pair_a_bytes = size_t(p.capacity)*sizeof(uint64_t);
    p.pair_b_bytes = size_t(p.capacity)*sizeof(uint64_t);
    p.pinned_bytes = size_t(p.capacity)*sizeof(uint64_t);

    // CUB DoubleBuffer queries are allocation-free, same as GpuBufferPool.
    cub_sort_pairs_u32_u32(nullptr,p.xs_sort_cub_bytes,
        nullptr,nullptr,nullptr,nullptr,p.total_xs,0,p.k);

    if (!compact_d_layout_fits(true,true,true,size_t(p.total_xs),
            size_t(p.capacity),p.storage_bytes,p.pair_b_bytes,
            p.xs_sort_cub_bytes))
        throw std::runtime_error("native CUDA pool Candidate D layout mismatch");

    // T1: 17 u64 coarse; 16*2^18+1 u64 fine; 16*2^24+1 u32 direct.
    // T2/T3: 257 u64 coarse; 256*2^16+1 u32 fine;
    //        256*2^20+1 u32 direct, all sharing the same region.
    p.index_t1_bytes = 17*sizeof(uint64_t)
        + ((size_t(16)<<18)+1)*sizeof(uint64_t)
        + ((size_t(16)<<24)+1)*sizeof(uint32_t);
    p.index_t23_bytes = 257*sizeof(uint64_t)
        + ((size_t(256)<<16)+1)*sizeof(uint32_t)
        + ((size_t(256)<<20)+1)*sizeof(uint32_t);

    size_t sort_pairs=0,sort_u64=0,sort_meta=0;
    cub_sort_pairs_u32_u32(nullptr,sort_pairs,
        nullptr,nullptr,nullptr,nullptr,p.capacity,0,p.k);
    cub_sort_keys_u64(nullptr,sort_u64,nullptr,nullptr,p.capacity,0,2*p.k);
    cub_sort_pairs_u32_u64(nullptr,sort_meta,
        nullptr,nullptr,nullptr,nullptr,p.capacity,0,p.k);
    p.sort_and_index_scratch_bytes = std::max(
        {sort_pairs,sort_u64,sort_meta,p.index_t1_bytes,p.index_t23_bytes});
    return p;
}

NativeCudaPool::NativeCudaPool(int cuda_ordinal,size_t margin_bytes)
    : session_(cuda_ordinal),plan_(native_cuda_pool_plan(cuda_ordinal))
{
    size_t free_bytes=0,total_bytes=0;
    session_.memory_info(free_bytes,total_bytes);
    if (plan_.device_bytes() > free_bytes ||
        margin_bytes > free_bytes - plan_.device_bytes()) {
        char buffer[280];
        std::snprintf(buffer,sizeof(buffer),
            "native CUDA pool: insufficient free VRAM: need %.3f GiB + %.3f GiB safety, "
            "free %.3f / total %.3f GiB",
            plan_.device_bytes()/double(1ull<<30),
            margin_bytes/double(1ull<<30),
            free_bytes/double(1ull<<30),
            total_bytes/double(1ull<<30));
        throw std::runtime_error(buffer);
    }
    try {
        storage_ = session_.allocate_device(plan_.storage_bytes);
        pair_b_ = session_.allocate_device(plan_.pair_b_bytes);
        scratch_ = session_.allocate_device(plan_.sort_and_index_scratch_bytes);
        count_ = static_cast<uint64_t*>(session_.allocate_device(sizeof(uint64_t)));
        // pair_a is lazy until Xs has finished, matching the proven layout.
    } catch (...) {
        // Constructor failure does not call the destructor.
        if (count_) session_.release_device(count_);
        if (scratch_) session_.release_device(scratch_);
        if (pair_b_) session_.release_device(pair_b_);
        if (storage_) session_.release_device(storage_);
        throw;
    }
}
void* NativeCudaPool::ensure_pair_a() {
    if(!pair_a_)pair_a_=session_.allocate_device(plan_.pair_a_bytes);
    return pair_a_;
}
uint64_t* NativeCudaPool::ensure_pinned() {
    if(!pinned_)pinned_=static_cast<uint64_t*>(
        session_.allocate_pinned(plan_.pinned_bytes));
    return pinned_;
}
NativeCudaPool::~NativeCudaPool() noexcept {
    // Destructor must not throw, even if device was reset externally.
    try {
        if(pinned_)session_.release_pinned(pinned_);
        if(count_)session_.release_device(count_);
        if(scratch_)session_.release_device(scratch_);
        if(pair_a_)session_.release_device(pair_a_);
        if(pair_b_)session_.release_device(pair_b_);
        if(storage_)session_.release_device(storage_);
    } catch (...) {
        // Preserve noexcept: failures are only diagnostic at shutdown.
    }
}
} // namespace pos2gpu
