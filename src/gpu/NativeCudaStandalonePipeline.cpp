// Standalone NVIDIA CUDA compute pipeline, no SYCL or AdaptiveCpp.
// Exact Candidate-D aliases and sequencing copied conceptually from the
// validated hybrid GpuPipeline, without reintroducing legacy ON/OFF switches.
// This deliberately produces only sorted T3 fragments + SHA-256 diagnostics;
// the gplot writer/ProofValidator integration is a separate later step.
#include "gpu/NativeCudaStandalonePipeline.hpp"
#include "gpu/NativeCudaPool.hpp"
#include "gpu/NativeCudaXs.hpp"
#include "gpu/NativeCudaIndex.hpp"
#include "gpu/NativeCudaMatch.hpp"
#include "gpu/NativeCudaPipeline.hpp"
#include "gpu/SortCubInternal.cuh"
#include "gpu/AesTables.inl"
#include "gpu/XsCandidateGpu.hpp"
#include "host/PoolSizing.hpp"
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

namespace pos2gpu {
namespace {
void require(bool good,char const* message) {
    if(!good)throw std::runtime_error(message);
}
template<class T> T* offset(void* p,size_t bytes) {
    return reinterpret_cast<T*>(static_cast<uint8_t*>(p)+bytes);
}
uint64_t counter(NativeCudaPool& pool) {
    uint64_t count=0;
    pool.session().copy_d2h(&count,pool.d_count(),sizeof(count));
    require(count<=pool.plan().capacity,"CUDA-only: match output exceeds Candidate D capacity");
    return count;
}
std::array<uint8_t,32> digest_fragments(NativeCudaPool& pool,
                                        uint64_t const* fragments,uint64_t count) {
    auto& session=pool.session();
    constexpr size_t tile_bytes=8u<<20;
    void* pinned=session.allocate_pinned(tile_bytes);
    BCRYPT_ALG_HANDLE algorithm=nullptr;
    BCRYPT_HASH_HANDLE hash=nullptr;
    try {
        if(BCryptOpenAlgorithmProvider(&algorithm,BCRYPT_SHA256_ALGORITHM,nullptr,0)<0)
            throw std::runtime_error("SHA-256 BCryptOpenAlgorithmProvider failed");
        if(BCryptCreateHash(algorithm,&hash,nullptr,0,nullptr,0,0)<0)
            throw std::runtime_error("SHA-256 BCryptCreateHash failed");
        auto const* src=reinterpret_cast<uint8_t const*>(fragments);
        uint64_t const total_bytes=count*sizeof(uint64_t);
        for(uint64_t off=0;off<total_bytes;off+=tile_bytes) {
            size_t n=size_t(std::min<uint64_t>(tile_bytes,total_bytes-off));
            session.copy_d2h(pinned,src+off,n);
            if(BCryptHashData(hash,static_cast<PUCHAR>(pinned),ULONG(n),0)<0)
                throw std::runtime_error("SHA-256 BCryptHashData failed");
        }
        std::array<uint8_t,32> digest{};
        if(BCryptFinishHash(hash,digest.data(),ULONG(digest.size()),0)<0)
            throw std::runtime_error("SHA-256 BCryptFinishHash failed");
        (void)BCryptDestroyHash(hash);
        (void)BCryptCloseAlgorithmProvider(algorithm,0);
        session.release_pinned(pinned);
        return digest;
    } catch(...) {
        if(hash)(void)BCryptDestroyHash(hash);
        if(algorithm)(void)BCryptCloseAlgorithmProvider(algorithm,0);
        session.release_pinned(pinned);
        throw;
    }
}
}

NativeCudaPipelineResult native_cuda_run_k28s6(
    std::array<uint8_t,32> const& plot_id,int ordinal)
{
    auto const started=std::chrono::steady_clock::now();
    NativeCudaPool pool(ordinal);
    auto& session=pool.session();
    NativeCudaPoolPlan const& p=pool.plan();
    uint64_t const n=p.total_xs,cap=p.capacity;
    require(n==(1ull<<28) && cap==match_phase_capacity(28,2),
            "CUDA-only: k28/s6 Candidate D plan mismatch");
    require(p.storage_bytes>=size_t(n)*8+p.xs_sort_cub_bytes,
            "CUDA-only: insufficient Xs CUB workspace in storage");
    require(p.pair_b_bytes>=size_t(n)*8,
            "CUDA-only: insufficient Xs output in pair B");
    require(p.storage_bytes>=size_t(cap)*12,
            "CUDA-only: insufficient storage for three cap-sized arrays");
    require(p.pair_a_bytes>=size_t(cap)*8,
            "CUDA-only: insufficient T1/T2/T3 match meta storage");
    require(p.pair_b_bytes>=size_t(cap)*8,
            "CUDA-only: insufficient sorted T2 meta storage");

    // The uploaded tables are needed by the verified shared16 T0 AES
    // matcher and by the full four-table native Xs generation.
    std::array<uint32_t,1024> tables{};
    for(int i=0;i<256;++i) {
        tables[i+0*256]=aes_tables::T0[i];
        tables[i+1*256]=aes_tables::T1[i];
        tables[i+2*256]=aes_tables::T2[i];
        tables[i+3*256]=aes_tables::T3[i];
    }
    void* table_memory=session.allocate_device(tables.size()*sizeof(uint32_t));
    // Preserve correct lifetime if allocation/index/matcher throws.
    try {
        session.copy_h2d(table_memory,tables.data(),tables.size()*sizeof(uint32_t));
        auto const* t0=static_cast<uint32_t const*>(table_memory);
        AesHashKeys keys=make_keys(plot_id.data());

        auto* storage=static_cast<uint8_t*>(pool.d_storage());
        auto* pair_b=static_cast<uint8_t*>(pool.d_pair_b());
        auto* scratch=static_cast<uint8_t*>(pool.d_scratch());
        auto* xs=reinterpret_cast<XsCandidateGpu*>(storage);
        // Candidate D: Xs source keys/values in storage's first 8*n;
        // sort destination keys/values in pair_b's first 8*n.
        uint32_t* keys_in=reinterpret_cast<uint32_t*>(storage);
        uint32_t* vals_in=keys_in+n;
        uint32_t* keys_out=reinterpret_cast<uint32_t*>(pair_b);
        uint32_t* vals_out=keys_out+n;

        std::fprintf(stderr,"[cuda-only] Xs generate/sort/pack...\n");
        cuda_xs_generate(keys,keys_in,vals_in,0,n,28,0,t0);
        size_t cub_xs=p.xs_sort_cub_bytes;
        cub_sort_pairs_u32_u32(storage+size_t(n)*8,cub_xs,
            keys_in,keys_out,vals_in,vals_out,n,0,28);
        cuda_xs_pack(keys_out,vals_out,xs,n);
        // The Xs source and pair_b sort buffers have now been consumed.
        auto* pair_a=static_cast<uint8_t*>(pool.ensure_pair_a());
        uint64_t* t1_meta=reinterpret_cast<uint64_t*>(pair_a);
        uint32_t* t1_mi=offset<uint32_t>(storage,size_t(n)*8);
        uint64_t* sorted_meta=reinterpret_cast<uint64_t*>(pair_b);
        uint32_t* sorted_mi=reinterpret_cast<uint32_t*>(storage);
        uint64_t const lmax=max_pairs_per_section(28,2);

        std::fprintf(stderr,"[cuda-only] T1 Direct24 index + match...\n");
        auto* coarse1=reinterpret_cast<uint64_t*>(scratch);
        auto* fine1=reinterpret_cast<uint64_t*>(scratch+17*sizeof(uint64_t));
        auto* direct1=reinterpret_cast<uint32_t*>(
            fine1+((size_t(16)<<18)+1));
        require(reinterpret_cast<uint8_t*>(direct1)+
                    ((size_t(16)<<24)+1)*sizeof(uint32_t)<=
                    scratch+p.sort_and_index_scratch_bytes,
                "CUDA-only: T1 Direct24 scratch overflow");
        cuda_t1_bucket_offsets(xs,n,24,16,coarse1);
        cuda_t1_fine_offsets(xs,coarse1,24,18,16,fine1);
        cuda_t1_direct24(xs,fine1,n,16,direct1);
        session.zero_device(pool.d_count(),sizeof(uint64_t));
        cuda_match_t1(keys,xs,coarse1,direct1,t0,4,28,2,4,
            (1u<<24)-1,2,28,t1_meta,t1_mi,pool.d_count(),
            cap,lmax,0,16);
        uint64_t const t1_count=counter(pool);
        require(t1_count!=0,"CUDA-only T1 produced zero pairs");
        std::fprintf(stderr,"[cuda-only] T1 pairs=%llu\n",
            static_cast<unsigned long long>(t1_count));

        // T1 metadata travels directly as 64-bit CUB radix values.
        // Xs is dead. Sorted keys overwrite storage[0..4*cap), while the
        // original T1 mi is in storage[8*n..], so the arrays do not alias.
        size_t cub_bytes=p.sort_and_index_scratch_bytes;
        cub_sort_pairs_u32_u64(scratch,cub_bytes,
            t1_mi,sorted_mi,t1_meta,sorted_meta,t1_count,0,28);

        // Candidate D T2 output: meta in pair A, mi in storage slot 1,
        // xbits in storage slot 2. The input sorted meta is in pair B.
        uint64_t* t2_meta=reinterpret_cast<uint64_t*>(pair_a);
        uint32_t* t2_mi=reinterpret_cast<uint32_t*>(storage)+cap;
        uint32_t* t2_xbits=reinterpret_cast<uint32_t*>(storage)+2*cap;
        auto* coarse23=reinterpret_cast<uint64_t*>(scratch);
        auto* fine23=reinterpret_cast<uint32_t*>(scratch+257*sizeof(uint64_t));
        auto* direct23=fine23+((size_t(256)<<16)+1);
        require(reinterpret_cast<uint8_t*>(direct23)+
                    ((size_t(256)<<20)+1)*sizeof(uint32_t)<=
                    scratch+p.sort_and_index_scratch_bytes,
                "CUDA-only: T2/T3 Direct20 scratch overflow");

        std::fprintf(stderr,"[cuda-only] T2 Direct20 index + match...\n");
        cuda_t23_bucket_offsets(sorted_mi,t1_count,20,256,coarse23);
        cuda_t23_fine_offsets(sorted_mi,coarse23,20,16,256,fine23);
        cuda_t23_direct20(sorted_mi,fine23,direct23,256);
        session.zero_device(pool.d_count(),sizeof(uint64_t));
        cuda_match_t2(keys,sorted_meta,coarse23,direct23,t0,64,28,2,
            (1u<<20)-1,6,28,14,t2_meta,t2_mi,t2_xbits,
            pool.d_count(),cap,lmax,0,256);
        uint64_t const t2_count=counter(pool);
        require(t2_count!=0,"CUDA-only T2 produced zero pairs");
        std::fprintf(stderr,"[cuda-only] T2 pairs=%llu\n",
            static_cast<unsigned long long>(t2_count));

        // T2 xbits occupies storage slot2. Park it in the now-dead
        // upper half of pair B before slot2 is used for sorted indices.
        uint32_t* parked_xbits=reinterpret_cast<uint32_t*>(pair_b)+cap;
        session.copy_d2d(parked_xbits,t2_xbits,
            size_t(t2_count)*sizeof(uint32_t));
        uint32_t* identity=reinterpret_cast<uint32_t*>(pair_b);
        uint32_t* t2_sorted_indices=reinterpret_cast<uint32_t*>(storage)+2*cap;
        cuda_init_u32_identity(identity,t2_count);
        cub_bytes=p.sort_and_index_scratch_bytes;
        cub_sort_pairs_u32_u32(scratch,cub_bytes,
            t2_mi,sorted_mi,identity,t2_sorted_indices,t2_count,0,28);
        // Pair B's parked xbits must be consumed BEFORE overwriting the
        // same pair B region with sorted T2 64-bit meta.
        uint32_t* sorted_xbits=reinterpret_cast<uint32_t*>(storage)+cap;
        cuda_gather_u32(parked_xbits,t2_sorted_indices,sorted_xbits,t2_count);
        cuda_gather_u64(t2_meta,t2_sorted_indices,sorted_meta,t2_count);

        std::fprintf(stderr,"[cuda-only] T3 Direct20 index + Feistel...\n");
        cuda_t23_bucket_offsets(sorted_mi,t2_count,20,256,coarse23);
        cuda_t23_fine_offsets(sorted_mi,coarse23,20,16,256,fine23);
        cuda_t23_direct20(sorted_mi,fine23,direct23,256);
        FeistelKey const feistel=make_feistel_key(plot_id.data(),28);
        uint64_t* fragments_in=reinterpret_cast<uint64_t*>(pair_a);
        session.zero_device(pool.d_count(),sizeof(uint64_t));
        cuda_match_t3(keys,feistel,sorted_meta,sorted_xbits,coarse23,
            direct23,t0,64,28,2,(1u<<20)-1,6,
            fragments_in,pool.d_count(),cap,lmax,0,256);
        uint64_t const t3_count=counter(pool);
        require(t3_count!=0,"CUDA-only T3 produced zero pairs");
        std::fprintf(stderr,"[cuda-only] T3 fragments=%llu\n",
            static_cast<unsigned long long>(t3_count));

        // Sorted T2 metadata dies after T3 match; pair B becomes sorted
        // fragment output, as in the verified hybrid Candidate D layout.
        uint64_t* fragments_out=reinterpret_cast<uint64_t*>(pair_b);
        cub_bytes=p.sort_and_index_scratch_bytes;
        cub_sort_keys_u64(scratch,cub_bytes,
            fragments_in,fragments_out,t3_count,0,56);
        std::fprintf(stderr,"[cuda-only] T3 sorted; hashing fragments...\n");
        auto const digest=digest_fragments(pool,fragments_out,t3_count);
        NativeCudaPipelineResult result{};
        result.t1_count=t1_count;
        result.t2_count=t2_count;
        result.t3_count=t3_count;
        result.sha256_fragments=digest;
        result.seconds=std::chrono::duration<double>(
            std::chrono::steady_clock::now()-started).count();
        session.release_device(table_memory);
        return result;
    } catch(...) {
        session.release_device(table_memory);
        throw;
    }
}
} // namespace pos2gpu
