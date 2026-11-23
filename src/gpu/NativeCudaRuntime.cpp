#include "gpu/NativeCudaRuntime.hpp"
#include <stdexcept>
#include <string>

namespace pos2gpu {
namespace {
void check(cudaError_t status, char const* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("native CUDA runtime ") + operation +
                                 ": " + cudaGetErrorString(status));
}
}
std::vector<NativeCudaDeviceInfo> native_cuda_devices() {
    int n = 0;
    check(cudaGetDeviceCount(&n), "cudaGetDeviceCount");
    std::vector<NativeCudaDeviceInfo> out;
    out.reserve(n);
    for (int i=0; i<n; ++i) {
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop,i), "cudaGetDeviceProperties");
        out.push_back({i,prop.name,prop.totalGlobalMem,prop.major,prop.minor});
    }
    return out;
}
NativeCudaSession::NativeCudaSession(int device_ordinal): ordinal_(device_ordinal) {
    select();
    check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
          "cudaStreamCreateWithFlags");
}
NativeCudaSession::~NativeCudaSession() noexcept {
    if (!stream_) return;
    (void)cudaSetDevice(ordinal_);
    (void)cudaStreamSynchronize(stream_);
    (void)cudaStreamDestroy(stream_);
}
void NativeCudaSession::select() const {
    check(cudaSetDevice(ordinal_), "cudaSetDevice");
}
void* NativeCudaSession::allocate_device(std::size_t bytes) const {
    if(!bytes) return nullptr;
    select();
    void* p=nullptr;
    check(cudaMalloc(&p,bytes),"cudaMalloc");
    return p;
}
void* NativeCudaSession::allocate_pinned(std::size_t bytes) const {
    if(!bytes) return nullptr;
    select();
    void* p=nullptr;
    check(cudaMallocHost(&p,bytes),"cudaMallocHost");
    return p;
}
void NativeCudaSession::release_device(void* ptr) const {
    if(!ptr)return;
    select();
    sync();
    check(cudaFree(ptr),"cudaFree");
}
void NativeCudaSession::release_pinned(void* ptr) const {
    if(!ptr)return;
    select();
    sync();
    check(cudaFreeHost(ptr),"cudaFreeHost");
}
void NativeCudaSession::copy_h2d(void* dst,void const* src,std::size_t bytes) const {
    if(!bytes)return;
    select();
    check(cudaMemcpyAsync(dst,src,bytes,cudaMemcpyHostToDevice,stream_),"H2D");
    sync();
}
void NativeCudaSession::copy_d2h(void* dst,void const* src,std::size_t bytes) const {
    if(!bytes)return;
    select();
    check(cudaMemcpyAsync(dst,src,bytes,cudaMemcpyDeviceToHost,stream_),"D2H");
    sync();
}
void NativeCudaSession::copy_d2d(void* dst,void const* src,std::size_t bytes) const {
    if(!bytes)return;
    select();
    check(cudaMemcpyAsync(dst,src,bytes,cudaMemcpyDeviceToDevice,stream_),"D2D");
    sync();
}
void NativeCudaSession::zero_device(void* ptr,std::size_t bytes) const {
    if(!bytes)return;
    select();
    check(cudaMemsetAsync(ptr,0,bytes,stream_),"memset");
    sync();
}
void NativeCudaSession::sync() const {
    select();
    check(cudaStreamSynchronize(stream_),"stream synchronization");
}
void NativeCudaSession::memory_info(std::size_t& free_bytes,std::size_t& total_bytes) const {
    select();
    check(cudaMemGetInfo(&free_bytes,&total_bytes),"cudaMemGetInfo");
}
} // namespace pos2gpu
