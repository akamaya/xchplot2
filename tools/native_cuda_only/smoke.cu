// Standalone nvcc-only GPU tests. No SYCL or AdaptiveCpp includes.
#include "gpu/NativeCudaXs.hpp"
#include "gpu/NativeCudaPipeline.hpp"
#include "gpu/NativeCudaPeer.hpp"
#include "gpu/NativeCudaTwoLevel.hpp"
#include "gpu/AesTables.inl"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
void ck(cudaError_t e, char const* op) {
    if(e!=cudaSuccess)throw std::runtime_error(std::string(op)+": "+cudaGetErrorString(e));
}
template<class T> T* alloc(size_t n) {
    T* p=nullptr;ck(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)),"cudaMalloc");
    return p;
}
template<class T> void upload(T* p,std::vector<T> const& h) {
    ck(cudaMemcpy(p,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice),"H2D");
}
template<class T> std::vector<T> download(T const* p,size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost),"D2H");
    return h;
}
pos2gpu::AesState cpu_round(pos2gpu::AesState s,pos2gpu::AesState key) {
    auto byte=[](uint32_t w,int n){return (w>>(8*n))&255u;};
    pos2gpu::AesState out{};
    for(int c=0;c<4;++c){
        out.w[c]=pos2gpu::aes_tables::T0[byte(s.w[c],0)]
           ^ pos2gpu::aes_tables::T1[byte(s.w[(c+1)&3],1)]
           ^ pos2gpu::aes_tables::T2[byte(s.w[(c+2)&3],2)]
           ^ pos2gpu::aes_tables::T3[byte(s.w[(c+3)&3],3)]
           ^ key.w[c];
    }
    return out;
}
uint32_t cpu_gx(pos2gpu::AesHashKeys const& keys,uint32_t x) {
    pos2gpu::AesState s=pos2gpu::set_int_vec_i128(0,0,0,int32_t(x));
    for(int r=0;r<pos2gpu::kAesGRounds;++r){
        s=cpu_round(s,keys.round_key_1);
        s=cpu_round(s,keys.round_key_2);
    }
    return s.w[0]&0x0fffffffu;
}
void require(bool good,char const* what){if(!good)throw std::runtime_error(what);}
}

int main() {
 try {
    int num=0;ck(cudaGetDeviceCount(&num),"device count");
    int chosen=-1;
    for(int i=0;i<num;++i){
        cudaDeviceProp prop{};
        ck(cudaGetDeviceProperties(&prop,i),"device properties");
        if(std::string(prop.name).find("RTX 4070 SUPER")!=std::string::npos)chosen=i;
    }
    require(chosen>=0,"RTX 4070 SUPER not detected; refusing to test another GPU");
    ck(cudaSetDevice(chosen),"set RTX 4070 SUPER");
    std::printf("[native-cuda-smoke] CUDA ordinal=%d RTX 4070 SUPER\n",chosen);
    constexpr size_t n=4096;
    uint8_t plot_id[32];
    for(int i=0;i<32;++i) plot_id[i]=uint8_t((i*29+7)&255);
    pos2gpu::AesHashKeys keys=pos2gpu::make_keys(plot_id);
    std::vector<uint32_t> tables(1024);
    for(int i=0;i<256;++i){
        tables[i]=pos2gpu::aes_tables::T0[i];
        tables[i+256]=pos2gpu::aes_tables::T1[i];
        tables[i+512]=pos2gpu::aes_tables::T2[i];
        tables[i+768]=pos2gpu::aes_tables::T3[i];
    }
    uint32_t *dt=alloc<uint32_t>(1024),*dk=alloc<uint32_t>(n),*dv=alloc<uint32_t>(n);
    upload(dt,tables);
    pos2gpu::cuda_xs_generate(keys,dk,dv,0,n,28,0,dt);
    auto generated=download(dk,n),values=download(dv,n);
    for(size_t i=0;i<n;++i){
        if(values[i]!=i||generated[i]!=cpu_gx(keys,uint32_t(i)))
            throw std::runtime_error("Xs AES parity mismatch at x="+std::to_string(i));
    }
    std::printf("[native-cuda-smoke] Xs AES: %zu/%zu parity PASS\n",n,n);

    uint32_t *d_hist=alloc<uint32_t>(4);
    ck(cudaMemset(d_hist,0,4*sizeof(uint32_t)),"clear histogram");
    pos2gpu::cuda_two_level_histogram(dk,d_hist,n,26,2);
    auto hist=download(d_hist,4);
    std::vector<uint32_t> expected(4,0),starts(4,0),zero(4,0);
    for(uint32_t v:generated)++expected[(v>>26)&3u];
    require(hist==expected,"two-level histogram mismatch");
    for(size_t j=1;j<4;++j) starts[j]=starts[j-1]+expected[j-1];
    uint32_t *d_cursor=alloc<uint32_t>(4),*d_start=alloc<uint32_t>(4);
    uint32_t *d_pk=alloc<uint32_t>(n),*d_pv=alloc<uint32_t>(n);
    upload(d_cursor,starts);
    pos2gpu::cuda_two_level_partition(dk,dv,d_pk,d_pv,d_cursor,n,26,2);
    auto out_keys=download(d_pk,n),out_vals=download(d_pv,n);
    std::vector<std::pair<uint32_t,uint32_t>> actual,ref;
    for(size_t i=0;i<n;++i){actual.emplace_back(out_keys[i],out_vals[i]);ref.emplace_back(generated[i],values[i]);}
    std::sort(actual.begin(),actual.end());std::sort(ref.begin(),ref.end());
    require(actual==ref,"two-level partition multiset mismatch");
    std::printf("[native-cuda-smoke] two-level histogram/partition PASS\n");

    ck(cudaMemset(d_hist,0,4*sizeof(uint32_t)),"clear peer histogram");
    pos2gpu::detail::cuda_peer_count_u32(dk,n,4,0,28,d_hist);
    require(download(d_hist,4)==expected,"distributed peer histogram mismatch");
    upload(d_start,starts);upload(d_cursor,zero);
    pos2gpu::detail::cuda_peer_scatter_u32_u32(
        dk,dv,n,4,0,28,d_start,d_cursor,d_pk,d_pv);
    out_keys=download(d_pk,n);out_vals=download(d_pv,n);
    actual.clear();
    for(size_t i=0;i<n;++i) actual.emplace_back(out_keys[i],out_vals[i]);
    std::sort(actual.begin(),actual.end());
    require(actual==ref,"distributed peer scatter multiset mismatch");
    std::printf("[native-cuda-smoke] CUDA peer histogram/scatter PASS\n");

    uint32_t *d_idx=alloc<uint32_t>(n),*d_gather=alloc<uint32_t>(n);
    std::vector<uint32_t> indices(n);
    for(size_t i=0;i<n;++i)indices[i]=uint32_t(n-1-i);
    upload(d_idx,indices);
    pos2gpu::cuda_gather_u32(dk,d_idx,d_gather,n);
    auto gathered=download(d_gather,n);
    for(size_t i=0;i<n;++i)
        require(gathered[i]==generated[n-1-i],"CUDA gather mismatch");
    std::printf("[native-cuda-smoke] gather PASS\n");

    ck(cudaFree(dt),"free dt");ck(cudaFree(dk),"free dk");ck(cudaFree(dv),"free dv");
    ck(cudaFree(d_hist),"free hist");ck(cudaFree(d_cursor),"free cursor");
    ck(cudaFree(d_start),"free starts");ck(cudaFree(d_pk),"free pk");
    ck(cudaFree(d_pv),"free pv");ck(cudaFree(d_idx),"free idx");
    ck(cudaFree(d_gather),"free gather");
    std::puts("[native-cuda-smoke] OK (no AdaptiveCpp/SYCL)");
    return 0;
 }catch(std::exception const& e){
    std::fprintf(stderr,"[native-cuda-smoke] FAILED: %s\n",e.what());
    return 1;
 }
}
