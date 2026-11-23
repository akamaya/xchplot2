// Direct24/Direct20 and coarse/fine indices: native CUDA builders matching
// the verified fixed-profile SYCL code, including sentinels and skew fallback.
#include "gpu/NativeCudaIndex.hpp"
#include <cuda_runtime.h>
#include "gpu/NativeCudaDevice.hpp"
#include <stdexcept>
#include <string>
#include <climits>
namespace pos2gpu {
namespace {
constexpr unsigned N=256;
inline void check(cudaError_t e,char const* at) {
    if(e!=cudaSuccess)throw std::runtime_error(
        std::string("native CUDA index ")+at+": "+cudaGetErrorString(e));
}
inline void finish() {
    check(cudaGetLastError(),"launch");
    check(cudaDeviceSynchronize(),"execution");
}
inline unsigned grid_for(uint64_t n) {
    uint64_t blocks=(n+N-1)/N;
    if(blocks==0||blocks>UINT_MAX)throw std::invalid_argument("CUDA index grid invalid");
    return static_cast<unsigned>(blocks);
}
__global__ void t1_buckets(XsCandidateGpu const* xs,uint64_t total,
                           int target_bits,uint32_t buckets,uint64_t* out) {
    uint32_t b=blockIdx.x*blockDim.x+threadIdx.x;
    if(b>buckets)return;
    if(b==buckets){out[b]=total;return;}
    uint64_t lo=0,hi=total;
    while(lo<hi){uint64_t m=lo+((hi-lo)>>1);if((xs[m].match_info>>target_bits)<b)lo=m+1;else hi=m;}
    out[b]=lo;
}
__global__ void t23_buckets(uint32_t const* mi,uint64_t total,
                            int target_bits,uint32_t buckets,uint64_t* out) {
    uint32_t b=blockIdx.x*blockDim.x+threadIdx.x;
    if(b>buckets)return;
    if(b==buckets){out[b]=total;return;}
    uint64_t lo=0,hi=total;
    while(lo<hi){uint64_t m=lo+((hi-lo)>>1);if((mi[m]>>target_bits)<b)lo=m+1;else hi=m;}
    out[b]=lo;
}
__global__ void t1_fine(XsCandidateGpu const* xs,uint64_t const* coarse,
                        int target_bits,int fine_bits,uint32_t buckets,uint64_t* fine) {
    uint32_t const idx=blockIdx.x*blockDim.x+threadIdx.x;
    uint32_t const n=buckets*(1u<<fine_bits);
    if(idx>=n)return;
    uint32_t const fine_count=1u<<fine_bits;
    uint32_t const bucket=idx/fine_count,key=idx%fine_count;
    uint64_t lo=coarse[bucket],hi=coarse[bucket+1];
    uint32_t const mask=(1u<<target_bits)-1u,shift=target_bits-fine_bits;
    while(lo<hi){uint64_t const m=lo+((hi-lo)>>1);
        uint32_t const v=(xs[m].match_info&mask)>>shift;
        if(v<key)lo=m+1;else hi=m;}
    fine[idx]=lo;
    if(idx==n-1)fine[n]=coarse[buckets];
}
__global__ void t23_fine(uint32_t const* mi,uint64_t const* coarse,
                         int target_bits,int fine_bits,uint32_t buckets,uint32_t* fine) {
    uint32_t const idx=blockIdx.x*blockDim.x+threadIdx.x;
    uint32_t const n=buckets*(1u<<fine_bits);
    if(idx>=n)return;
    uint32_t const fine_count=1u<<fine_bits;
    uint32_t const bucket=idx/fine_count,key=idx%fine_count;
    uint64_t lo=coarse[bucket],hi=coarse[bucket+1];
    uint32_t const mask=(1u<<target_bits)-1u,shift=target_bits-fine_bits;
    while(lo<hi){uint64_t const m=lo+((hi-lo)>>1);
        uint32_t const v=(mi[m]&mask)>>shift;
        if(v<key)lo=m+1;else hi=m;}
    fine[idx]=uint32_t(lo);
    if(idx==n-1)fine[n]=uint32_t(coarse[buckets]);
}
__global__ void build24(XsCandidateGpu const* xs,uint64_t const* fine,
                        uint64_t total,uint32_t* direct) {
    uint32_t const g=blockIdx.x*blockDim.x+threadIdx.x;
    constexpr uint32_t n=16u*(1u<<18);
    if(g>=n)return;
    uint64_t const start=fine[g],end=fine[g+1];
    uint32_t const base=(g&((1u<<18)-1u))<<6;
    uint64_t const dest=uint64_t(g)<<6;
    if(end-start<=1024u){
        uint64_t pos=start;
        for(uint32_t s=0;s<64;s++){
            uint32_t const target=base+s;
            while(pos<end&&(xs[pos].match_info&0xffffffu)<target)++pos;
            direct[dest+s]=uint32_t(pos);
        }
    }else{
        for(uint32_t s=0;s<64;s++){
            uint32_t const target=base+s;
            uint64_t lo=start,hi=end;
            while(lo<hi){uint64_t const mid=lo+((hi-lo)>>1);
                if((xs[mid].match_info&0xffffffu)<target)lo=mid+1;else hi=mid;}
            direct[dest+s]=uint32_t(lo);
        }
    }
    if(g==n-1)direct[uint64_t(n)<<6]=uint32_t(total);
}
__global__ void build20(uint32_t const* mi,uint32_t const* fine,uint32_t* direct) {
    uint32_t const g=blockIdx.x*blockDim.x+threadIdx.x;
    constexpr uint32_t n=256u*(1u<<16);
    if(g>=n)return;
    uint32_t const start=fine[g],end=fine[g+1];
    uint32_t const base=(g&0xffffu)<<4;
    uint64_t const dest=uint64_t(g)<<4;
    if(end-start<=1024u){
        uint32_t pos=start;
        for(uint32_t s=0;s<16;s++){
            uint32_t const target=base|s;
            while(pos<end&&(mi[pos]&0xfffffu)<target)++pos;
            direct[dest+s]=pos;
        }
    }else{
        for(uint32_t s=0;s<16;s++){
            uint32_t const target=base|s;
            uint32_t lo=start,hi=end;
            while(lo<hi){uint32_t const m=lo+((hi-lo)>>1);
                if((mi[m]&0xfffffu)<target)lo=m+1;else hi=m;}
            direct[dest+s]=lo;
        }
    }
    if(g==n-1)direct[uint64_t(n)<<4]=fine[n];
}
} // namespace
void cuda_t1_bucket_offsets(XsCandidateGpu const* p,uint64_t n,int bits,uint32_t buckets,uint64_t* out){
    cuda_use_pointer_device(p);
    if(bits!=24||buckets!=16)throw std::invalid_argument("CUDA T1 buckets: k28/s6 only");
    t1_buckets<<<grid_for(uint64_t(buckets)+1),N>>>(p,n,bits,buckets,out);finish();
}
void cuda_t1_fine_offsets(XsCandidateGpu const* p,uint64_t const* coarse,int bits,int fine_bits,uint32_t buckets,uint64_t* out){
    cuda_use_pointer_device(p);
    if(bits!=24||fine_bits!=18||buckets!=16)throw std::invalid_argument("CUDA T1 fine: k28/s6 only");
    t1_fine<<<grid_for(uint64_t(buckets)<<fine_bits),N>>>(p,coarse,bits,fine_bits,buckets,out);finish();
}
void cuda_t1_direct24(XsCandidateGpu const* p,uint64_t const* fine,uint64_t total,uint32_t buckets,uint32_t* out){
    cuda_use_pointer_device(p);
    if(buckets!=16||total>UINT32_MAX)throw std::invalid_argument("CUDA Direct24: invalid layout");
    build24<<<grid_for(uint64_t(buckets)<<18),N>>>(p,fine,total,out);finish();
}
void cuda_t23_bucket_offsets(uint32_t const* p,uint64_t n,int bits,uint32_t buckets,uint64_t* out){
    cuda_use_pointer_device(p);
    if(bits!=20||buckets!=256)throw std::invalid_argument("CUDA T2/T3 buckets: k28/s6 only");
    t23_buckets<<<grid_for(uint64_t(buckets)+1),N>>>(p,n,bits,buckets,out);finish();
}
void cuda_t23_fine_offsets(uint32_t const* p,uint64_t const* coarse,int bits,int fine_bits,uint32_t buckets,uint32_t* out){
    cuda_use_pointer_device(p);
    if(bits!=20||fine_bits!=16||buckets!=256)throw std::invalid_argument("CUDA T2/T3 fine: k28/s6 only");
    t23_fine<<<grid_for(uint64_t(buckets)<<fine_bits),N>>>(p,coarse,bits,fine_bits,buckets,out);finish();
}
void cuda_t23_direct20(uint32_t const* p,uint32_t const* fine,uint32_t* direct,uint32_t buckets){
    cuda_use_pointer_device(p);
    if(buckets!=256)throw std::invalid_argument("CUDA Direct20: invalid layout");
    build20<<<grid_for(uint64_t(buckets)<<16),N>>>(p,fine,direct);finish();
}
} // namespace pos2gpu
