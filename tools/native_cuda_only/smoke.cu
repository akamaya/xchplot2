// Standalone nvcc-only GPU tests. No SYCL or AdaptiveCpp includes.
#include "gpu/NativeCudaXs.hpp"
#include "gpu/NativeCudaRuntime.hpp"
#include "gpu/NativeCudaPool.hpp"
#include "gpu/FeistelCipherGpu.cuh"
#include "gpu/SortCubInternal.cuh"
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
    // Test the independent CUDA Runtime host facade: device discovery,
    // pinned buffers, CUDA memory and synchronous transfers (no SYCL).
    pos2gpu::NativeCudaSession session(chosen);
    std::size_t free_bytes=0,total_bytes=0;
    session.memory_info(free_bytes,total_bytes);
    require(free_bytes>0 && total_bytes>=free_bytes,"CUDA memory info invalid");
    auto* dev_tmp=session.allocate_device(1024);
    auto* host_tmp=static_cast<uint8_t*>(session.allocate_pinned(1024));
    require(dev_tmp && host_tmp,"CUDA native allocations failed");
    for(int i=0;i<1024;++i)host_tmp[i]=uint8_t(i&255);
    session.copy_h2d(dev_tmp,host_tmp,1024);
    std::fill(host_tmp,host_tmp+1024,uint8_t(0));
    session.copy_d2h(host_tmp,dev_tmp,1024);
    for(int i=0;i<1024;++i)
        require(host_tmp[i]==uint8_t(i&255),"native runtime transfer mismatch");
    session.release_device(dev_tmp);
    session.release_pinned(host_tmp);
    std::puts("[native-cuda-smoke] CUDA Runtime session/pinned transfer PASS");

    std::printf("[native-cuda-smoke] CUDA ordinal=%d RTX 4070 SUPER\n",chosen);
    constexpr size_t n=4096;
    uint8_t plot_id[32];
    for(int i=0;i<32;++i) plot_id[i]=uint8_t((i*29+7)&255);
    pos2gpu::AesHashKeys keys=pos2gpu::make_keys(plot_id);
    // SHA256 round-key schedule: independently generated reference for
    // plot_id[i] = (i*29+7)&255 using the upstream bit-extraction rules.
    {
        auto fk=pos2gpu::make_feistel_key(plot_id,28);
        uint32_t const golden[4][3]={
            {24494912u,130966106u,118182932u},
            {239297506u,71721121u,12732245u},
            {200835073u,77432689u,3115480u},
            {27242478u,118085515u,111350572u}};
        require(fk.k==28,"Feistel key k mismatch");
        for(int r=0;r<4;++r){
            auto const& key=fk.round_key.round[r];
            require(key.b==golden[r][0]&&key.c==golden[r][1]&&
                    key.d==golden[r][2],"Feistel SHA256 key schedule mismatch");
        }
        std::puts("[native-cuda-smoke] Windows CNG Feistel key schedule PASS");
    }
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

    // CUB sort: exact key order and stable value order; no SYCL queue.
    {
        auto expected_pairs=ref;
        std::stable_sort(expected_pairs.begin(),expected_pairs.end(),
            [](auto const& a,auto const& b){return a.first<b.first;});
        uint32_t* sorted_keys=alloc<uint32_t>(n);
        uint32_t* sorted_vals=alloc<uint32_t>(n);
        size_t scratch_bytes=0;
        upload(dk,generated);
        upload(dv,values);
        pos2gpu::cub_sort_pairs_u32_u32(nullptr,scratch_bytes,dk,sorted_keys,
            dv,sorted_vals,n,0,28);
        require(scratch_bytes!=0,"CUB sort sizing returned zero");
        void* tmp=nullptr;ck(cudaMalloc(&tmp,scratch_bytes),"CUB scratch allocation");
        pos2gpu::cub_sort_pairs_u32_u32(tmp,scratch_bytes,dk,sorted_keys,
            dv,sorted_vals,n,0,28);
        auto actual_k=download(sorted_keys,n),actual_v=download(sorted_vals,n);
        for(size_t i=0;i<n;++i){
            if(actual_k[i]!=expected_pairs[i].first ||
               actual_v[i]!=expected_pairs[i].second)
                throw std::runtime_error("CUB u32/u32 stable sort mismatch");
        }
        ck(cudaFree(tmp),"CUB scratch release");
        ck(cudaFree(sorted_keys),"sorted keys release");
        ck(cudaFree(sorted_vals),"sorted vals release");
        std::printf("[native-cuda-smoke] CUDA CUB u32/u32 sort: %zu PASS\n",n);
    }
    // Exact u32-key/u64-value sort used by Candidate D T1.
    {
        uint32_t* out_key=alloc<uint32_t>(n);
        uint64_t* in_meta=alloc<uint64_t>(n);
        uint64_t* out_meta=alloc<uint64_t>(n);
        std::vector<uint64_t> metas(n);
        std::vector<std::pair<uint32_t,uint64_t>> expected_pairs;
        for(size_t i=0;i<n;++i){
            metas[i]=(uint64_t(i)<<32)|uint64_t(17*i+5);
            expected_pairs.emplace_back(generated[i],metas[i]);
        }
        std::stable_sort(expected_pairs.begin(),expected_pairs.end(),
            [](auto const& a,auto const& b){return a.first<b.first;});
        upload(dk,generated);upload(in_meta,metas);
        size_t bytes=0;
        pos2gpu::cub_sort_pairs_u32_u64(nullptr,bytes,dk,out_key,
            in_meta,out_meta,n,0,28);
        void* temp=nullptr;ck(cudaMalloc(&temp,bytes),"CUB u32/u64 scratch");
        pos2gpu::cub_sort_pairs_u32_u64(temp,bytes,dk,out_key,
            in_meta,out_meta,n,0,28);
        auto keys2 = download(out_key, n);
        auto meta2 = download(out_meta, n);
        for(size_t i=0;i<n;++i)
            if(keys2[i]!=expected_pairs[i].first ||
               meta2[i]!=expected_pairs[i].second)
                throw std::runtime_error("CUB u32/u64 stable sort mismatch");
        ck(cudaFree(temp),"free CUB u32/u64 scratch");
        ck(cudaFree(out_key),"free keys u32/u64");
        ck(cudaFree(in_meta),"free in meta");
        ck(cudaFree(out_meta),"free out meta");
        std::printf("[native-cuda-smoke] CUDA CUB u32/u64 sort: %zu PASS\n",n);
    }
    // 56-bit T3 fragment sort uses the third exact CUB entry point.
    {
        std::vector<uint64_t> fragment(n);
        for(size_t i=0;i<n;++i)
            fragment[i]=((uint64_t(n-1-i)<<24) | uint64_t(i*13));
        auto sorted_expected=fragment;
        std::sort(sorted_expected.begin(),sorted_expected.end());
        uint64_t* in=alloc<uint64_t>(n);
        uint64_t* out=alloc<uint64_t>(n);
        upload(in,fragment);
        size_t bytes=0;
        pos2gpu::cub_sort_keys_u64(nullptr,bytes,in,out,n,0,56);
        void* temp=nullptr;ck(cudaMalloc(&temp,bytes),"CUB u64 scratch");
        pos2gpu::cub_sort_keys_u64(temp,bytes,in,out,n,0,56);
        require(download(out,n)==sorted_expected,"CUB u64 keys sort mismatch");
        ck(cudaFree(temp),"free CUB u64 scratch");
        ck(cudaFree(in),"free u64 in");
        ck(cudaFree(out),"free u64 out");
        std::printf("[native-cuda-smoke] CUDA CUB u64 sort: %zu PASS\n",n);
    }
    // Full-size k28 Candidate D layout must match the verified pool.
    // This query DOES NOT allocate multi-GiB buffers.
    {
        auto plan=pos2gpu::native_cuda_pool_plan(chosen);
        require(plan.total_xs==(1ull<<28),"Candidate D Xs count mismatch");
        require(plan.capacity>plan.total_xs,"Candidate D match cap too small");
        require(plan.storage_bytes>=12*plan.capacity,"Candidate D storage too small");
        require(plan.pair_b_bytes>=8*plan.total_xs,"Candidate D pair B too small");
        require(plan.sort_and_index_scratch_bytes>=plan.index_t1_bytes &&
                plan.sort_and_index_scratch_bytes>=plan.index_t23_bytes,
                "Candidate D exact index scratch too small");
        std::printf("[native-cuda-smoke] Candidate D pool: %.3f GiB VRAM "
                    "(storage %.3f, pair A %.3f, pair B %.3f, "
                    "scratch %.3f), planning PASS (no large alloc)\n",
            plan.device_bytes()/double(1ull<<30),
            plan.storage_bytes/double(1ull<<30),
            plan.pair_a_bytes/double(1ull<<30),
            plan.pair_b_bytes/double(1ull<<30),
            plan.sort_and_index_scratch_bytes/double(1ull<<30));
    }

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
