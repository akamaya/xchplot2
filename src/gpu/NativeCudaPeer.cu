// Native CUDA two-pass peer scatter: exact SYCL bucket routing and atomics.
#include "gpu/NativeCudaPeer.hpp"
#include "gpu/NativeCudaDevice.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <climits>
#include <stdexcept>
#include <string>
namespace pos2gpu::detail {
namespace {
constexpr unsigned kThreads=256;
inline unsigned grid(uint64_t n) {
    uint64_t const b=(n+kThreads-1)/kThreads;
    if (!b || b>UINT_MAX) throw std::invalid_argument("CUDA peer grid overflow");
    return unsigned(b);
}
inline void finish() {
    cudaError_t e=cudaGetLastError();
    if (e!=cudaSuccess) throw std::runtime_error(std::string("CUDA peer launch: ")+cudaGetErrorString(e));
    e=cudaDeviceSynchronize();
    if (e!=cudaSuccess) throw std::runtime_error(std::string("CUDA peer execution: ")+cudaGetErrorString(e));
}
inline void validate(uint32_t N,int first,int last,int max_bits) {
    if (!N || first<0 || last<=first || last>max_bits)
        throw std::invalid_argument("CUDA peer: invalid bit range or destination count");
}
__device__ __forceinline__ uint32_t bucket32(uint32_t k,uint32_t N,int b,int e) {
    int const bits=e-b;
    if (bits>=32) return uint32_t((uint64_t(k)*N)>>32);
    uint32_t const mask=(1u<<bits)-1u;
    uint32_t const val=(k>>b)&mask;
    return uint32_t((uint64_t(val)*N)>>bits);
}
__device__ __forceinline__ uint32_t bucket64(uint64_t k,uint32_t N,int b,int e) {
    int const bits=e-b;
    if (bits>=64) return uint32_t((uint64_t(uint32_t(k>>32))*N)>>32);
    if (bits>32) {
        uint64_t const mask=(1ull<<(bits-32))-1ull;
        uint64_t const val=(k>>(b+32))&mask;
        return uint32_t((val*N)>>(bits-32));
    }
    uint64_t const mask=(1ull<<bits)-1ull;
    uint64_t const val=(k>>b)&mask;
    return uint32_t((val*N)>>bits);
}
template<class K>
__device__ __forceinline__ uint32_t bucket(K k,uint32_t N,int b,int e) {
    if constexpr(sizeof(K)==4) return bucket32(uint32_t(k),N,b,e);
    else return bucket64(uint64_t(k),N,b,e);
}
template<class K>
__global__ void count_kernel(K const* keys,uint64_t n,uint32_t N,
                              int b,int e,uint32_t* count) {
    uint64_t const i=uint64_t(blockIdx.x)*kThreads+threadIdx.x;
    if(i<n) atomicAdd(&count[bucket(keys[i],N,b,e)],1u);
}
template<class K,class A,class B,bool HasA,bool HasB>
__global__ void scatter_kernel(
    K const* keys,A const* va,B const* vb,uint64_t n,uint32_t N,
    int b,int e,uint32_t const* offsets,uint32_t* cursors,
    K* ok,A* oa,B* ob) {
    uint64_t const i=uint64_t(blockIdx.x)*kThreads+threadIdx.x;
    if(i>=n) return;
    K k=keys[i];
    uint32_t const dest=bucket(k,N,b,e);
    uint32_t const pos=offsets[dest]+atomicAdd(&cursors[dest],1u);
    ok[pos]=k;
    if constexpr(HasA) oa[pos]=va[i];
    if constexpr(HasB) ob[pos]=vb[i];
}
template<class K,class A,class B,bool HA,bool HB>
void scatter(K const* keys,A const* va,B const* vb,uint64_t n,uint32_t N,
    int b,int e,uint32_t const* starts,uint32_t* cur,K* ok,A* oa,B* ob) {
    if(!n)return;
    validate(N,b,e,sizeof(K)*8);
    cuda_use_pointer_device(ok);
    scatter_kernel<K,A,B,HA,HB><<<grid(n),kThreads>>>(
        keys,va,vb,n,N,b,e,starts,cur,ok,oa,ob);
    finish();
}
}
void cuda_peer_count_u32(uint32_t const* p,uint64_t n,uint32_t N,int b,int e,uint32_t* dst) {
    if(!n)return;validate(N,b,e,32);cuda_use_pointer_device(dst);
    count_kernel<<<grid(n),kThreads>>>(p,n,N,b,e,dst);finish();
}
void cuda_peer_count_u64(uint64_t const* p,uint64_t n,uint32_t N,int b,int e,uint32_t* dst) {
    if(!n)return;validate(N,b,e,64);cuda_use_pointer_device(dst);
    count_kernel<<<grid(n),kThreads>>>(p,n,N,b,e,dst);finish();
}
void cuda_peer_scatter_u32_u32(uint32_t const* k,uint32_t const* v,uint64_t n,uint32_t N,
    int b,int e,uint32_t const* start,uint32_t* cur,uint32_t* ok,uint32_t* ov) {
    scatter<uint32_t,uint32_t,uint32_t,true,false>(k,v,nullptr,n,N,b,e,start,cur,ok,ov,nullptr);
}
void cuda_peer_scatter_u32_u64(uint32_t const* k,uint64_t const* v,uint64_t n,uint32_t N,
    int b,int e,uint32_t const* start,uint32_t* cur,uint32_t* ok,uint64_t* ov) {
    scatter<uint32_t,uint64_t,uint32_t,true,false>(k,v,nullptr,n,N,b,e,start,cur,ok,ov,nullptr);
}
void cuda_peer_scatter_u32_u64u32(uint32_t const* k,uint64_t const* va,uint32_t const* vb,
    uint64_t n,uint32_t N,int b,int e,uint32_t const* start,uint32_t* cur,
    uint32_t* ok,uint64_t* oa,uint32_t* ob) {
    scatter<uint32_t,uint64_t,uint32_t,true,true>(k,va,vb,n,N,b,e,start,cur,ok,oa,ob);
}
void cuda_peer_scatter_u64_keys(uint64_t const* k,uint64_t n,uint32_t N,int b,int e,
    uint32_t const* start,uint32_t* cur,uint64_t* ok) {
    scatter<uint64_t,uint32_t,uint32_t,false,false>(
        k,nullptr,nullptr,n,N,b,e,start,cur,ok,nullptr,nullptr);
}
} // namespace pos2gpu::detail
