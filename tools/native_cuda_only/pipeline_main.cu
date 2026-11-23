// CUDA-only experimental compute runner. Does not write a gplot.
// Usage: native_cuda_pipeline [--plan|--run] [--plot-id 64hex]
// --plan is default and avoids any multi-GiB VRAM allocation.
#include "gpu/NativeCudaStandalonePipeline.hpp"
#include "gpu/NativeCudaPool.hpp"
#include "gpu/NativeCudaRuntime.hpp"
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
unsigned hex_digit(char ch) {
    if(ch>='0'&&ch<='9')return unsigned(ch-'0');
    if(ch>='a'&&ch<='f')return unsigned(ch-'a'+10);
    if(ch>='A'&&ch<='F')return unsigned(ch-'A'+10);
    throw std::invalid_argument("invalid plot id hex character");
}
std::array<uint8_t,32> plot_id_from_hex(std::string_view hex) {
    if(hex.size()!=64)throw std::invalid_argument(
        "plot id must have exactly 64 hexadecimal characters");
    std::array<uint8_t,32> id{};
    for(size_t i=0;i<32;++i)id[i]=uint8_t(
        (hex_digit(hex[2*i])<<4)|hex_digit(hex[2*i+1]));
    return id;
}
// Upstream posCalculatePlotIdForIndex(group_id,0,0) hashes a packed
// 35-byte {group_id[32], big-endian plot_index[2], meta_group[1]} record.
// Fixed index/meta 0 matches an ordinary single-member test .gplot.
std::array<uint8_t,32> test_group_plot_id(
    std::array<uint8_t,32> const& group_id) {
    std::array<uint8_t,35> msg{};
    std::copy(group_id.begin(),group_id.end(),msg.begin());
    BCRYPT_ALG_HANDLE alg=nullptr;
    BCRYPT_HASH_HANDLE hash=nullptr;
    if(BCryptOpenAlgorithmProvider(&alg,BCRYPT_SHA256_ALGORITHM,nullptr,0)<0)
        throw std::runtime_error("group plot ID: failed to open SHA256");
    std::array<uint8_t,32> digest{};
    NTSTATUS st=BCryptCreateHash(alg,&hash,nullptr,0,nullptr,0,0);
    if(st>=0) st=BCryptHashData(hash,msg.data(),ULONG(msg.size()),0);
    if(st>=0) st=BCryptFinishHash(hash,digest.data(),ULONG(digest.size()),0);
    if(hash)(void)BCryptDestroyHash(hash);
    (void)BCryptCloseAlgorithmProvider(alg,0);
    if(st<0)throw std::runtime_error("group plot ID: BCrypt SHA256 failed");
    return digest;
}
void print_hex(uint8_t const* p,size_t len) {
    for(size_t i=0;i<len;++i)std::printf("%02x",unsigned(p[i]));
}
}
int main(int argc,char** argv) {
    try {
        bool run=false;
        bool explicit_plot_id=false, explicit_group_id=false;
        std::array<uint8_t,32> plot_id{};
        std::array<uint8_t,32> group_id{};
        // A constant, non-production test ID; can be replaced with --plot-id.
        for(unsigned i=0;i<32;++i)plot_id[i]=uint8_t((i*29+7)&255u);
        for(int i=1;i<argc;++i) {
            std::string_view const arg=argv[i];
            if(arg=="--plan")run=false;
            else if(arg=="--run")run=true;
            else if(arg=="--plot-id"&&i+1<argc) {
                plot_id=plot_id_from_hex(argv[++i]);
                explicit_plot_id=true;
            } else if(arg=="--group-id"&&i+1<argc) {
                group_id=plot_id_from_hex(argv[++i]);
                explicit_group_id=true;
            }
            else if(arg=="--help") {
                std::puts("native_cuda_pipeline [--plan|--run] [--plot-id 64hex|--group-id 64hex]");
                std::puts("  --plan  (default): only query memory and CUB workspace");
                std::puts("  --run   : run complete CUDA Xs->T1->T2->T3 compute");
                std::puts("            Output is counts+SHA256, NOT a .gplot file");
                std::puts("  --group-id computes test plot ID, index=0, meta_group=0");
                return 0;
            } else throw std::invalid_argument("unknown native CUDA pipeline option");
        }
        if(explicit_plot_id && explicit_group_id)
            throw std::invalid_argument("--plot-id and --group-id are mutually exclusive");
        if(explicit_group_id) {
            plot_id=test_group_plot_id(group_id);
            // CPU hash reference: SHA256(35 zero bytes) is
            // 0d5535e13cc9708d0ff0289af2fae27e564b6bcbcd9242f5140d96957744a517.
            bool all_zero=std::all_of(group_id.begin(),group_id.end(),
                [](uint8_t x){return x==0;});
            if(all_zero) {
                auto const expected=plot_id_from_hex(
                    "0d5535e13cc9708d0ff0289af2fae27e564b6bcbcd9242f5140d96957744a517");
                if(plot_id!=expected)
                    throw std::runtime_error("group plot ID SHA256 regression");
            }
        }
        int ordinal=-1;
        auto const available=pos2gpu::native_cuda_devices();
        for(auto const& d:available)
            if(d.name.find("RTX 4070 SUPER")!=std::string::npos)ordinal=d.ordinal;
        if(ordinal<0)throw std::runtime_error(
            "RTX 4070 SUPER not detected; refusing to run on another device");
        auto const p=pos2gpu::native_cuda_pool_plan(ordinal);
        std::printf("[cuda-only] device %d RTX 4070 SUPER; k=28 s=6\n",ordinal);
        std::printf("[cuda-only] Candidate D VRAM planned=%.3f GiB (excl. 128 MiB margin)\n",
            p.device_bytes()/double(1ull<<30));
        if(!run) {
            if(explicit_group_id) {
                std::printf("[cuda-only] group-derived plot ID: ");
                print_hex(plot_id.data(),plot_id.size());
                std::puts("");
            }
            std::puts("[cuda-only] PLAN OK (no large VRAM allocation)");
            std::puts("[cuda-only] Use --run to execute the full experimental GPU pipeline");
            return 0;
        }
        std::printf("[cuda-only] plot id: ");print_hex(plot_id.data(),plot_id.size());
        std::puts("");
        std::puts("[cuda-only] EXPERIMENTAL: native compute only, NOT a farmable .gplot");
        auto const result=pos2gpu::native_cuda_run_k28s6(plot_id,ordinal);
        std::printf("[cuda-only] T1=%llu T2=%llu T3=%llu seconds=%.3f\n",
            (unsigned long long)result.t1_count,
            (unsigned long long)result.t2_count,
            (unsigned long long)result.t3_count,result.seconds);
        std::printf("[cuda-only] sorted T3 fragment SHA256=");
        print_hex(result.sha256_fragments.data(),result.sha256_fragments.size());
        std::puts("");
        std::printf("[cuda-only] canonical unique T3 fragments=%llu SHA256=",
            static_cast<unsigned long long>(result.unique_t3_count));
        print_hex(result.sha256_unique_fragments.data(),
                  result.sha256_unique_fragments.size());
        std::puts("");
        std::puts("[cuda-only] COMPUTE OK (not a Full Proof validation)");
        return 0;
    } catch(std::exception const& e) {
        std::fprintf(stderr,"[cuda-only] FAILED: %s\n",e.what());
        return 1;
    }
}
