// Native CUDA AES Xs generation and packing.
// The mixed-key stream feeds the already-native CUB sort.
#include "gpu/NativeCudaXs.hpp"
#include <cuda_runtime.h>
#include "gpu/NativeCudaDevice.hpp"
#include <climits>
#include <stdexcept>
#include <string>
namespace pos2gpu {
namespace {
constexpr int threads=256;
void check(cudaError_t e,char const* op) {
    if(e!=cudaSuccess)throw std::runtime_error(
        std::string("native CUDA Xs ")+op+": "+cudaGetErrorString(e));
}
unsigned blocks(uint64_t n) {
    uint64_t b=(n+threads-1)/threads;
    if(!b||b>UINT_MAX)throw std::invalid_argument("native CUDA Xs: invalid grid");
    return unsigned(b);
}
void finish() {
    check(cudaGetLastError(),"launch");
    check(cudaDeviceSynchronize(),"execution");
}
__global__ void gen(AesHashKeys keys,uint32_t* out_keys,uint32_t* out_vals,
                    uint64_t start,uint64_t count,int k,uint32_t xor_value,
                    uint32_t const* tables)
{
    __shared__ uint32_t t[4*256];
    for(int i=threadIdx.x;i<1024;i+=threads)t[i]=tables[i];
    __syncthreads();
    uint64_t const idx=uint64_t(blockIdx.x)*threads+threadIdx.x;
    if(idx>=count)return;
    uint32_t const x=uint32_t(start+idx);
    out_keys[idx]=g_x_smem(keys,x^xor_value,k,t);
    out_vals[idx]=x;
}
__global__ void pack(uint32_t const* keys,uint32_t const* vals,
                     XsCandidateGpu* out,uint64_t n)
{
    uint64_t const i=uint64_t(blockIdx.x)*threads+threadIdx.x;
    if(i<n)out[i]=XsCandidateGpu{keys[i],vals[i]};
}
} // namespace
void cuda_xs_generate(AesHashKeys keys,uint32_t* ko,uint32_t* vo,
                      uint64_t pos,uint64_t n,int k,uint32_t xor_value,
                      uint32_t const* tables) {
    if(!n)return;
    if(k<18||k>28)throw std::invalid_argument("CUDA Xs requires k18..28");
    cuda_use_pointer_device(ko);
    gen<<<blocks(n),threads>>>(keys,ko,vo,pos,n,k,xor_value,tables);
    finish();
}
void cuda_xs_pack(uint32_t const* keys,uint32_t const* vals,
                  XsCandidateGpu* out,uint64_t n) {
    if(!n)return;
    cuda_use_pointer_device(out);
    pack<<<blocks(n),threads>>>(keys,vals,out,n);
    finish();
}
} // namespace pos2gpu
