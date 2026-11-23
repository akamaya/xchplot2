// CUDA top-bit histogram + atomic bucket partition for TwoLevelSort.
// Per-bucket sorting is already provided by SortCuda.cu (CUB).
#include "gpu/NativeCudaTwoLevel.hpp"
#include "gpu/NativeCudaDevice.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <climits>
#include <stdexcept>
#include <string>
namespace pos2gpu {
namespace {
constexpr unsigned kThreads=256;
inline unsigned blocks(uint64_t n) {
    uint64_t b=(n+kThreads-1)/kThreads;
    if (!b || b>UINT_MAX) throw std::invalid_argument("CUDA two-level sort: grid overflow");
    return unsigned(b);
}
inline void check(cudaError_t e,char const* where) {
    if (e!=cudaSuccess) throw std::runtime_error(
        std::string("CUDA two-level sort ")+where+": "+cudaGetErrorString(e));
}
inline void finish() {
    check(cudaGetLastError(),"launch");
    check(cudaDeviceSynchronize(),"execution");
}
inline void validate(int offset,int bits) {
    if(offset<0||bits<1||bits>16||offset+bits>32)
        throw std::invalid_argument("CUDA two-level sort: invalid top-bit range");
}
__device__ __forceinline__ uint32_t bucket(uint32_t key,int off,int bits) {
    uint32_t const mask=(1u<<bits)-1u;
    return (key>>off)&mask;
}
__global__ void hist_kernel(uint32_t const* keys,uint32_t* hist,uint64_t n,
                            int off,int bits) {
    uint64_t i=uint64_t(blockIdx.x)*kThreads+threadIdx.x;
    if(i<n) atomicAdd(&hist[bucket(keys[i],off,bits)],1u);
}
__global__ void partition_kernel(uint32_t const* keys,uint32_t const* vals,
    uint32_t* part_keys,uint32_t* part_vals,uint32_t* cursors,
    uint64_t n,int off,int bits) {
    uint64_t i=uint64_t(blockIdx.x)*kThreads+threadIdx.x;
    if(i>=n)return;
    uint32_t const key=keys[i];
    uint32_t const b=bucket(key,off,bits);
    uint32_t const pos=atomicAdd(&cursors[b],1u);
    part_keys[pos]=key;
    part_vals[pos]=vals[i];
}
}
void cuda_two_level_histogram(uint32_t const* keys,uint32_t* hist,uint64_t n,
    int off,int bits) {
    if(!n)return; validate(off,bits);cuda_use_pointer_device(hist);
    hist_kernel<<<blocks(n),kThreads>>>(keys,hist,n,off,bits);finish();
}
void cuda_two_level_partition(uint32_t const* keys,uint32_t const* vals,
    uint32_t* part_keys,uint32_t* part_vals,uint32_t* cursors,
    uint64_t n,int off,int bits) {
    if(!n)return;validate(off,bits);cuda_use_pointer_device(part_keys);
    partition_kernel<<<blocks(n),kThreads>>>(keys,vals,part_keys,part_vals,
        cursors,n,off,bits);finish();
}
} // namespace pos2gpu
