// Native CUDA pipeline primitives preserving the verified SYCL semantics.
#include "gpu/NativeCudaPipeline.hpp"
#include <cuda_runtime.h>
#include "gpu/NativeCudaDevice.hpp"
#include <climits>
#include <stdexcept>
#include <string>
namespace pos2gpu {
namespace {
constexpr int threads=256;
unsigned blocks(uint64_t n) {
    uint64_t b=(n+threads-1)/threads;
    if(!b||b>UINT_MAX)throw std::invalid_argument("native CUDA primitive grid invalid");
    return unsigned(b);
}
void finish(){
    cudaError_t e=cudaGetLastError();
    if(e!=cudaSuccess)throw std::runtime_error(std::string("CUDA pipeline launch: ")+cudaGetErrorString(e));
    e=cudaDeviceSynchronize();
    if(e!=cudaSuccess)throw std::runtime_error(std::string("CUDA pipeline execution: ")+cudaGetErrorString(e));
}
__global__ void init(uint32_t* d,uint64_t n){
    uint64_t i=uint64_t(blockIdx.x)*threads+threadIdx.x;
    if(i<n)d[i]=uint32_t(i);
}
template<typename T>
__global__ void gather(T const* src,uint32_t const* ids,T* dst,uint64_t n){
    uint64_t i=uint64_t(blockIdx.x)*threads+threadIdx.x;
    if(i<n)dst[i]=src[ids[i]];
}
__global__ void permute(uint64_t const* sm,uint32_t const* sx,uint32_t const* ids,
                        uint64_t* dm,uint32_t* dx,uint64_t n) {
    uint64_t i=uint64_t(blockIdx.x)*threads+threadIdx.x;
    if(i<n){uint32_t j=ids[i];dm[i]=sm[j];dx[i]=sx[j];}
}
__global__ void inverse(uint32_t const* ids,uint32_t* out,uint64_t n){
    uint64_t i=uint64_t(blockIdx.x)*threads+threadIdx.x;
    if(i<n)out[ids[i]]=uint32_t(i);
}
template<typename T>
__global__ void scatter(T const* src,uint32_t const* ids,T* dst,uint64_t n){
    uint64_t i=uint64_t(blockIdx.x)*threads+threadIdx.x;
    if(i<n)dst[ids[i]]=src[i];
}
__global__ void permute_scatter(uint64_t const* sm,uint32_t const* sx,uint32_t const* ids,
                                uint64_t* dm,uint32_t* dx,uint64_t n) {
    uint64_t i=uint64_t(blockIdx.x)*threads+threadIdx.x;
    if(i<n){uint32_t j=ids[i];dm[j]=sm[i];dx[j]=sx[i];}
}
__global__ void merge(uint32_t const* ak,uint32_t const* av,uint64_t na,
                      uint32_t const* bk,uint32_t const* bv,uint64_t nb,
                      uint32_t* ok,uint32_t* ov,uint64_t n) {
    uint64_t p=uint64_t(blockIdx.x)*threads+threadIdx.x;
    if(p>=n)return;
    uint64_t lo=(p>nb)?p-nb:0;
    uint64_t hi=(p<na)?p:na;
    while(lo<hi){
        uint64_t i=lo+(hi-lo+1)/2;
        uint64_t j=p-i;
        uint32_t const a=ak[i-1];
        uint32_t const b=(j<nb)?bk[j]:0xffffffffu;
        if(a>b)hi=i-1;else lo=i;
    }
    uint64_t i=lo,j=p-i;
    bool take_a=(i<na) && (j>=nb || ak[i]<=bk[j]);
    if(take_a){ok[p]=ak[i];ov[p]=av[i];}
    else{ok[p]=bk[j];ov[p]=bv[j];}
}
} // namespace
void cuda_init_u32_identity(uint32_t* a,uint64_t n){if(!n)return;cuda_use_pointer_device(a);init<<<blocks(n),threads>>>(a,n);finish();}
void cuda_gather_u64(uint64_t const* a,uint32_t const* b,uint64_t* c,uint64_t n){if(!n)return;cuda_use_pointer_device(c);gather<<<blocks(n),threads>>>(a,b,c,n);finish();}
void cuda_gather_u32(uint32_t const* a,uint32_t const* b,uint32_t* c,uint64_t n){if(!n)return;cuda_use_pointer_device(c);gather<<<blocks(n),threads>>>(a,b,c,n);finish();}
void cuda_permute_t2(uint64_t const* a,uint32_t const* b,uint32_t const* c,uint64_t* d,uint32_t* e,uint64_t n){if(!n)return;cuda_use_pointer_device(d);permute<<<blocks(n),threads>>>(a,b,c,d,e,n);finish();}
void cuda_inverse_u32(uint32_t const* a,uint32_t* b,uint64_t n){if(!n)return;cuda_use_pointer_device(b);inverse<<<blocks(n),threads>>>(a,b,n);finish();}
void cuda_scatter_u64(uint64_t const* a,uint32_t const* b,uint64_t* c,uint64_t n){if(!n)return;cuda_use_pointer_device(c);scatter<<<blocks(n),threads>>>(a,b,c,n);finish();}
void cuda_scatter_u32(uint32_t const* a,uint32_t const* b,uint32_t* c,uint64_t n){if(!n)return;cuda_use_pointer_device(c);scatter<<<blocks(n),threads>>>(a,b,c,n);finish();}
void cuda_permute_t2_scatter(uint64_t const* a,uint32_t const* b,uint32_t const* c,uint64_t* d,uint32_t* e,uint64_t n){if(!n)return;cuda_use_pointer_device(d);permute_scatter<<<blocks(n),threads>>>(a,b,c,d,e,n);finish();}
void cuda_merge_pairs_stable_2way_u32_u32(uint32_t const* ak,uint32_t const* av,uint64_t na,uint32_t const* bk,uint32_t const* bv,uint64_t nb,uint32_t* ok,uint32_t* ov,uint64_t n) {
    if(!n)return;cuda_use_pointer_device(ok);merge<<<blocks(n),threads>>>(ak,av,na,bk,bv,nb,ok,ov,n);finish();
}
} // namespace pos2gpu
