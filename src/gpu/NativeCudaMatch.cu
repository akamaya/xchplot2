// Native CUDA Compact matching, ported from the verified SYCL k28/s6 path.
// Target: NVIDIA Ada 12 GiB. Exact Direct24 / Direct20 indices and the
// same V4INIT shared16 T0 AES layout, pairing compaction and overflow path.
// No SYCL headers in this nvcc translation unit.
#include "gpu/NativeCudaMatch.hpp"
#include "gpu/MatchKernelCommon.cuh"
#include <cuda_runtime.h>
#include "gpu/NativeCudaDevice.hpp"
#include <cub/block/block_scan.cuh>
#include <climits>
#include <stdexcept>
#include <string>

namespace pos2gpu {
namespace {
constexpr int kThreads = 256;
constexpr int kT1Cap = 512;
constexpr int kT2Cap = 480;
constexpr int kT3Cap = 330;

inline void check_cuda(cudaError_t status, char const* where) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("native CUDA match: ") + where +
                                 ": " + cudaGetErrorString(status));
}
inline dim3 make_grid(uint64_t l_count_max, uint32_t begin, uint32_t end) {
    if (end <= begin) return dim3(1,1,1);
    uint64_t const bx = (l_count_max + kThreads - 1) / kThreads;
    if (bx == 0 || bx > INT_MAX || end - begin > 65535u)
        throw std::invalid_argument("native CUDA match: invalid grid dimensions");
    return dim3(static_cast<unsigned>(bx), end - begin);
}

// 16 replicas of T0, identical to the verified SYCL V4INIT layout.
// The source buffer has four standard AES tables; T0 occupies [0,256).
__device__ __forceinline__ uint32_t const* load_shared16_t0(
    uint32_t const* device_t0, uint4* shared_v4)
{
    #pragma unroll 1
    for (int j = threadIdx.x; j < 1024; j += kThreads) {
        uint32_t x = device_t0[j >> 2];
        shared_v4[j] = make_uint4(x, x, x, x);
    }
    __syncthreads();
    return reinterpret_cast<uint32_t const*>(shared_v4) + (threadIdx.x & 15);
}

__device__ __forceinline__ uint32_t round_word(
    AesState const& s, AesState const& key, uint32_t const* t, int c)
{
    uint32_t const a = t[(s.w[c] & 255u) * 16u];
    uint32_t const b = t[((s.w[(c+1)&3] >> 8) & 255u) * 16u];
    uint32_t const d = t[((s.w[(c+2)&3] >> 16) & 255u) * 16u];
    uint32_t const e = t[(s.w[(c+3)&3] >> 24) * 16u];
    return a ^ ((b << 8) | (b >> 24)) ^ ((d << 16) | (d >> 16)) ^
           ((e >> 8) | (e << 24)) ^ key.w[c];
}
__device__ __forceinline__ AesState round_all(
    AesState const& s, AesState const& key, uint32_t const* t)
{
    AesState out;
    #pragma unroll
    for (int c=0;c<4;c++) out.w[c] = round_word(s,key,t,c);
    return out;
}
// The last AES round is partial, exactly Experiment A.  All preceding
// rounds are full AESENC-compatible rounds with alternating keys.
__device__ __forceinline__ AesState before_last(
    AesState state, AesHashKeys const& keys, int pairs, uint32_t const* t)
{
    #pragma unroll 2
    for (int i=0; i<pairs-1; ++i) {
        state = round_all(state,keys.round_key_1,t);
        state = round_all(state,keys.round_key_2,t);
    }
    return round_all(state,keys.round_key_1,t);
}
__device__ __forceinline__ uint32_t target(
    AesHashKeys const& keys, uint32_t table, uint32_t match_key, uint64_t meta,
    uint32_t const* t, int extra)
{
    AesState state = set_int_vec_i128(int32_t(meta>>32),int32_t(meta),
                                      int32_t(match_key),int32_t(table));
    state = before_last(state,keys,kAesMatchingTargetRounds << extra,t);
    return round_word(state,keys.round_key_2,t,0);
}
template<int Table>
__device__ __forceinline__ Result128 pair_result(
    AesHashKeys const& keys, uint64_t l, uint64_t r, uint32_t const* t,
    int extra, uint32_t test_mask)
{
    AesState s = set_int_vec_i128(int32_t(r>>32),int32_t(r),
                                  int32_t(l>>32),int32_t(l));
    s = before_last(s, keys, kAesPairingRounds << extra, t);
    Result128 result{{0,0,0,0}};
    result.r[3] = round_word(s,keys.round_key_2,t,3);
    if ((result.r[3] & test_mask) == 0) {
        if constexpr (Table == 1 || Table == 2)
            result.r[0] = round_word(s,keys.round_key_2,t,0);
        if constexpr (Table == 2) {
            result.r[1] = round_word(s,keys.round_key_2,t,1);
            result.r[2] = round_word(s,keys.round_key_2,t,2);
        }
    }
    return result;
}
__device__ __forceinline__ uint64_t reserve(uint64_t* counter) {
    return atomicAdd(reinterpret_cast<unsigned long long*>(counter),1ULL);
}

template<int Cap>
struct SharedScan {
    typename cub::BlockScan<uint32_t,kThreads>::TempStorage temp;
    uint32_t count;
};
template<int Cap>
__device__ __forceinline__ uint32_t exclusive_count(
    SharedScan<Cap>& scan, uint32_t count, uint32_t* total)
{
    uint32_t offset;
    cub::BlockScan<uint32_t,kThreads>(scan.temp).ExclusiveSum(count,offset);
    if (threadIdx.x == kThreads-1) *total = offset+count;
    __syncthreads();
    return offset;
}

__global__ void t1_kernel(
    AesHashKeys keys, XsCandidateGpu const* xs, uint64_t const* offsets,
    uint32_t const* direct, uint32_t const* t0, uint32_t num_keys, int k,
    int section_bits, int extra, uint32_t target_mask, uint32_t test_mask,
    uint32_t info_mask, uint64_t* out_meta, uint32_t* out_mi,
    uint64_t* out_count, uint64_t capacity, uint32_t begin_bucket)
{
    __shared__ alignas(16) uint4 t0_local[1024];
    __shared__ uint32_t pairs_l[kT1Cap],pairs_r[kT1Cap];
    __shared__ SharedScan<kT1Cap> scan;
    uint32_t const* table = load_shared16_t0(t0,t0_local);
    uint32_t const bucket = begin_bucket + blockIdx.y;
    uint32_t const section_l = bucket/num_keys;
    uint32_t const match_key = bucket%num_keys;
    uint32_t const section_r=matching_section_r(section_l,section_bits);
    uint32_t const r_bucket=section_r*num_keys+match_key;
    uint64_t const start=offsets[section_l*num_keys], stop=offsets[(section_l+1)*num_keys];
    uint64_t const l=start+uint64_t(blockIdx.x)*kThreads+threadIdx.x;
    uint32_t x=0,count=0; uint64_t first=0;
    if(l<stop) {
        x=xs[l].x;
        uint32_t const t=target(keys,1u,match_key,uint64_t(x),table,extra)&target_mask;
        uint64_t const key=(uint64_t(r_bucket)<<24)|t;
        first=direct[key]; count=direct[key+1]-uint32_t(first);
    }
    auto emit=[&](uint32_t a,uint32_t b) {
        auto result=pair_result<1>(keys,uint64_t(a),uint64_t(b),table,extra,test_mask);
        if(result.r[3]&test_mask) return;
        uint64_t const oi=reserve(out_count);
        if(oi<capacity) {
            out_meta[oi]=(uint64_t(a)<<k)|b;
            out_mi[oi]=result.r[0]&info_mask;
        }
    };
    uint32_t scan_n=(count>kT1Cap)?kT1Cap+1:count;
    uint32_t const offset=exclusive_count(scan,scan_n,&scan.count);
    uint32_t const n=scan.count;
    if(n>kT1Cap) {
        for(uint32_t j=0;j<count;j++) emit(x,xs[first+j].x);
        return;
    }
    for(uint32_t j=0;j<count;j++) {
        uint32_t const i=offset+j;
        pairs_l[i]=x; pairs_r[i]=xs[first+j].x;
    }
    __syncthreads();
    for(uint32_t i=threadIdx.x;i<n;i+=kThreads) emit(pairs_l[i],pairs_r[i]);
}

__global__ void t2_kernel(
    AesHashKeys keys, uint64_t const* meta, uint64_t const* offsets,
    uint32_t const* direct, uint32_t const* t0, uint32_t num_keys,
    int k, int section_bits, uint32_t target_mask, uint32_t test_mask,
    uint32_t info_mask, int half_k,
    uint64_t* out_meta, uint32_t* out_mi, uint32_t* out_xbits,
    uint64_t* out_count, uint64_t capacity, uint32_t begin_bucket)
{
    __shared__ alignas(16) uint4 t0_local[1024];
    __shared__ uint64_t pairs_l[kT2Cap],pairs_r[kT2Cap];
    __shared__ SharedScan<kT2Cap> scan;
    uint32_t const* table=load_shared16_t0(t0,t0_local);
    uint32_t const bucket=begin_bucket+blockIdx.y;
    uint32_t const section_l=bucket/num_keys,match_key=bucket%num_keys;
    uint32_t const section_r=matching_section_r(section_l,section_bits);
    uint32_t const r_bucket=section_r*num_keys+match_key;
    uint64_t const start=offsets[section_l*num_keys],stop=offsets[(section_l+1)*num_keys];
    uint64_t const l=start+uint64_t(blockIdx.x)*kThreads+threadIdx.x;
    uint64_t ml=0,first=0,hi=0;uint32_t count=0;
    if(l<stop){
        ml=meta[l];
        uint32_t const t=target(keys,2u,match_key,ml,table,0)&target_mask;
        uint64_t const key=(uint64_t(r_bucket)<<20)|t;
        first=direct[key]; hi=direct[key+1];
        for(uint64_t r=first;r<hi;r++) if(++count>kT2Cap) break;
    }
    auto emit=[&](uint64_t a,uint64_t b) {
        auto result=pair_result<2>(keys,a,b,table,0,test_mask);
        if(result.r[3]&test_mask) return;
        uint64_t const oi=reserve(out_count);
        if(oi<capacity) {
            uint64_t const full=uint64_t(result.r[1])|(uint64_t(result.r[2])<<32);
            uint64_t const mask=(2*k==64)?~uint64_t(0):((uint64_t(1)<<(2*k))-1);
            out_meta[oi]=full&mask;
            out_mi[oi]=result.r[0]&info_mask;
            uint32_t const xl=uint32_t((a>>k)>>half_k),xr=uint32_t((b>>k)>>half_k);
            out_xbits[oi]=(xl<<half_k)|xr;
        }
    };
    uint32_t const off=exclusive_count(scan,count,&scan.count);
    uint32_t const n=scan.count;
    if(n>kT2Cap){
        if(l<stop) for(uint64_t r=first;r<hi;r++) emit(ml,meta[r]);
        return;
    }
    for(uint32_t j=0;j<count;j++) {pairs_l[off+j]=ml;pairs_r[off+j]=meta[first+j];}
    __syncthreads();
    for(uint32_t i=threadIdx.x;i<n;i+=kThreads)emit(pairs_l[i],pairs_r[i]);
}

__global__ void t3_kernel(
    AesHashKeys keys, FeistelKey fk, uint64_t const* meta,uint32_t const* xbits,
    uint64_t const* offsets,uint32_t const* direct,uint32_t const* t0,
    uint32_t num_keys,int k,int section_bits,uint32_t target_mask,
    uint32_t test_mask,uint64_t* out_fragments,uint64_t* out_count,
    uint64_t capacity,uint32_t begin_bucket)
{
    __shared__ alignas(16) uint4 t0_local[1024];
    __shared__ uint64_t pairs_l[kT3Cap],pairs_r[kT3Cap];
    __shared__ uint32_t pairs_xl[kT3Cap],pairs_xr[kT3Cap];
    __shared__ SharedScan<kT3Cap> scan;
    uint32_t const* table=load_shared16_t0(t0,t0_local);
    uint32_t const bucket=begin_bucket+blockIdx.y;
    uint32_t const section_l=bucket/num_keys,match_key=bucket%num_keys;
    uint32_t const section_r=matching_section_r(section_l,section_bits);
    uint32_t const r_bucket=section_r*num_keys+match_key;
    uint64_t const start=offsets[section_l*num_keys],stop=offsets[(section_l+1)*num_keys];
    uint64_t const l=start+uint64_t(blockIdx.x)*kThreads+threadIdx.x;
    uint64_t ml=0,first=0,hi=0;uint32_t count=0,xb=0;
    if(l<stop){
        ml=meta[l];xb=xbits[l];
        uint32_t const t=target(keys,3u,match_key,ml,table,0)&target_mask;
        uint64_t const key=(uint64_t(r_bucket)<<20)|t;
        first=direct[key];hi=direct[key+1];
        for(uint64_t r=first;r<hi;r++)if(++count>kT3Cap)break;
    }
    auto emit=[&](uint64_t a,uint64_t b,uint32_t xa,uint32_t xb2){
        auto result=pair_result<3>(keys,a,b,table,0,test_mask);
        if(result.r[3]&test_mask)return;
        uint64_t const fragment=feistel_encrypt(fk,(uint64_t(xa)<<k)|uint64_t(xb2));
        uint64_t const oi=reserve(out_count);
        if(oi<capacity)out_fragments[oi]=fragment;
    };
    uint32_t const off=exclusive_count(scan,count,&scan.count);
    uint32_t const n=scan.count;
    if(n>kT3Cap){
        if(l<stop)for(uint64_t r=first;r<hi;r++)emit(ml,meta[r],xb,xbits[r]);
        return;
    }
    for(uint32_t j=0;j<count;j++){
        uint32_t const i=off+j;
        pairs_l[i]=ml;pairs_r[i]=meta[first+j];
        pairs_xl[i]=xb;pairs_xr[i]=xbits[first+j];
    }
    __syncthreads();
    for(uint32_t i=threadIdx.x;i<n;i+=kThreads)
        emit(pairs_l[i],pairs_r[i],pairs_xl[i],pairs_xr[i]);
}
inline void finish() {
    check_cuda(cudaGetLastError(),"kernel launch");
    check_cuda(cudaDeviceSynchronize(),"kernel execution");
}
} // namespace

void cuda_match_t1(AesHashKeys keys,XsCandidateGpu const* xs,
    uint64_t const* offsets,uint32_t const* direct,uint32_t const* t0,
    uint32_t num_keys,int k,int section_bits,int extra,uint32_t target_mask,
    int test_bits,int info_bits,uint64_t* out_meta,uint32_t* out_mi,
    uint64_t* out_count,uint64_t capacity,uint64_t max_rows,
    uint32_t begin_bucket,uint32_t end_bucket)
{
    if(end_bucket<=begin_bucket)return;
    cuda_use_pointer_device(xs);
    if(k!=28||num_keys!=4||section_bits!=2||extra!=4)
        throw std::invalid_argument("CUDA T1 requires k28/s6");
    dim3 grid=make_grid(max_rows,begin_bucket,end_bucket);
    uint32_t const tm=(test_bits>=32)?0xffffffffu:((1u<<test_bits)-1u);
    uint32_t const im=(info_bits>=32)?0xffffffffu:((1u<<info_bits)-1u);
    t1_kernel<<<grid,kThreads>>>(keys,xs,offsets,direct,t0,num_keys,k,
        section_bits,extra,target_mask,tm,im,out_meta,out_mi,out_count,capacity,begin_bucket);
    finish();
}
void cuda_match_t2(AesHashKeys keys,uint64_t const* meta,uint64_t const* offsets,
    uint32_t const* direct,uint32_t const* t0,uint32_t num_keys,
    int k,int section_bits,uint32_t target_mask,int test_bits,int info_bits,
    int half_k,uint64_t* out_meta,uint32_t* out_mi,uint32_t* out_xbits,
    uint64_t* out_count,uint64_t capacity,uint64_t max_rows,
    uint32_t begin_bucket,uint32_t end_bucket)
{
    if(end_bucket<=begin_bucket)return;
    cuda_use_pointer_device(meta);
    if(k!=28||num_keys!=64||section_bits!=2)
        throw std::invalid_argument("CUDA T2 requires k28/s6");
    dim3 grid=make_grid(max_rows,begin_bucket,end_bucket);
    uint32_t const tm=(test_bits>=32)?0xffffffffu:((1u<<test_bits)-1u);
    uint32_t const im=(info_bits>=32)?0xffffffffu:((1u<<info_bits)-1u);
    t2_kernel<<<grid,kThreads>>>(keys,meta,offsets,direct,t0,num_keys,k,
        section_bits,target_mask,tm,im,half_k,out_meta,out_mi,out_xbits,
        out_count,capacity,begin_bucket);
    finish();
}
void cuda_match_t3(AesHashKeys keys,FeistelKey fk,uint64_t const* meta,
    uint32_t const* xbits,uint64_t const* offsets,uint32_t const* direct,
    uint32_t const* t0,uint32_t num_keys,int k,int section_bits,
    uint32_t target_mask,int test_bits,uint64_t* out_fragments,
    uint64_t* out_count,uint64_t capacity,uint64_t max_rows,
    uint32_t begin_bucket,uint32_t end_bucket)
{
    if(end_bucket<=begin_bucket)return;
    cuda_use_pointer_device(meta);
    if(k!=28||num_keys!=64||section_bits!=2)
        throw std::invalid_argument("CUDA T3 requires k28/s6");
    dim3 grid=make_grid(max_rows,begin_bucket,end_bucket);
    uint32_t const tm=(test_bits>=32)?0xffffffffu:((1u<<test_bits)-1u);
    t3_kernel<<<grid,kThreads>>>(keys,fk,meta,xbits,offsets,direct,t0,num_keys,
        k,section_bits,target_mask,tm,out_fragments,out_count,capacity,begin_bucket);
    finish();
}
} // namespace pos2gpu
