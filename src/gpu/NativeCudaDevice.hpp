#pragma once
// Select the actual CUDA device owning an AdaptiveCpp CUDA USM pointer.
// SYCL enumeration and CUDA runtime ordinals need not match on multi-GPU hosts.
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
namespace pos2gpu {
inline void cuda_use_pointer_device(void const* ptr) {
    cudaPointerAttributes attrs{};
    cudaError_t status=cudaPointerGetAttributes(&attrs,ptr);
    if(status!=cudaSuccess)throw std::runtime_error(
        std::string("native CUDA device detection failed: ")+cudaGetErrorString(status));
    if(attrs.type!=cudaMemoryTypeDevice && attrs.type!=cudaMemoryTypeManaged)
        throw std::runtime_error("native CUDA input is not CUDA device memory");
    status=cudaSetDevice(attrs.device);
    if(status!=cudaSuccess)throw std::runtime_error(
        std::string("native CUDA device selection failed: ")+cudaGetErrorString(status));
}
}
