#pragma once
// CUDA Runtime host facade, intentionally independent of SYCL/AdaptiveCpp.
// Future replacement for SyclBackend and GPU buffer/queue management.
// These synchronous methods prioritize correctness during migration.
#include <cuda_runtime_api.h>
#include <cstddef>
#include <string>
#include <vector>

namespace pos2gpu {
struct NativeCudaDeviceInfo {
    int ordinal;
    std::string name;
    std::size_t total_bytes;
    int sm_major;
    int sm_minor;
};
std::vector<NativeCudaDeviceInfo> native_cuda_devices();

class NativeCudaSession final {
public:
    explicit NativeCudaSession(int device_ordinal);
    ~NativeCudaSession() noexcept;
    NativeCudaSession(NativeCudaSession const&) = delete;
    NativeCudaSession& operator=(NativeCudaSession const&) = delete;
    int ordinal() const noexcept { return ordinal_; }
    cudaStream_t stream() const noexcept { return stream_; }
    void select() const;
    void* allocate_device(std::size_t bytes) const;
    void* allocate_pinned(std::size_t bytes) const;
    void release_device(void* ptr) const;
    void release_pinned(void* ptr) const;
    void copy_h2d(void* device_dst, void const* host_src, std::size_t bytes) const;
    void copy_d2h(void* host_dst, void const* device_src, std::size_t bytes) const;
    void copy_d2d(void* dst, void const* src, std::size_t bytes) const;
    void zero_device(void* ptr, std::size_t bytes) const;
    void sync() const;
    void memory_info(std::size_t& free_bytes, std::size_t& total_bytes) const;
private:
    int ordinal_;
    cudaStream_t stream_{};
};
} // namespace pos2gpu
