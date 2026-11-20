#include "gpu/SyclBackend.hpp"
#include "gpu/AesHashGpu.cuh"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <type_traits>
#include <new>
#include <numeric>

using namespace pos2gpu;

template<int B>
static inline AesState rotate_round(AesState s, const AesState& key,
                                    const uint32_t* t) {
    AesState out;
    #pragma unroll
    for (int c=0; c<4; ++c) {
        uint32_t a=t[(s.w[c]&255u)*B];
        uint32_t b=t[((s.w[(c+1)&3]>>8)&255u)*B];
        uint32_t d=t[((s.w[(c+2)&3]>>16)&255u)*B];
        uint32_t e=t[(s.w[(c+3)&3]>>24)*B];
        b=(b<<8)|(b>>24);
        d=(d<<16)|(d>>16);
        e=(e>>8)|(e<<24);
        out.w[c]=a^b^d^e^key.w[c];
    }
    return out;
}

template<int B>
static inline AesState rotate_rounds(AesState s, const AesHashKeys& keys,
                                     int rounds, const uint32_t* t) {
    #pragma unroll 2
    for(int r=0;r<rounds;++r) {
        s=rotate_round<B>(s,keys.round_key_1,t);
        s=rotate_round<B>(s,keys.round_key_2,t);
    }
    return s;
}


template<int B>
static inline AesState hash(AesState s, const AesHashKeys& keys,
                            int rounds, const uint32_t* t) {
    if constexpr(B==0) return run_rounds_smem(s,keys,rounds,t);
    else return rotate_rounds<B>(s,keys,rounds,t);
}

template<int B> class AesDivergenceKernel;
template<int B>
static void launch(sycl::queue& q, const uint32_t* tables,
                   const AesState* in, const uint8_t* counts,
                   AesState* out, size_t n, int rounds, AesHashKeys keys) {
    constexpr size_t threads=256;
    constexpr size_t words=(B==0?4:B)*256;
    q.submit([&](sycl::handler& h){
        sycl::local_accessor<uint32_t,1> local{sycl::range<1>{words},h};
        h.parallel_for<AesDivergenceKernel<B>>(
            sycl::nd_range<1>{((n+threads-1)/threads)*threads,threads},
            [=](sycl::nd_item<1> it){
                uint32_t* t=&local[0];
                size_t lid=it.get_local_id(0);
                #pragma unroll 1
                for(size_t j=lid;j<words;j+=threads){
                    if constexpr(B==0) t[j]=tables[j];
                    else t[j]=tables[j/B];
                }
                it.barrier(sycl::access::fence_space::local_space);
                if constexpr(B!=0) t+=(lid&(B-1));
                size_t i=it.get_global_id(0);
                if(i>=n) return;
                // Model T1: all lanes compute a target, then each lane
                // computes a runtime-dependent number of pairing hashes.
                // No binary search or global output atomics in this test.
                AesState input=in[i];
                AesState target=hash<B>(input,keys,rounds,t);
                AesState acc=target;
                int count=counts[i];
                #pragma unroll 1
                for(int j=0;j<count;++j){
                    AesState p;
                    p.w[0]=input.w[2]; p.w[1]=0;
                    p.w[2]=((target.w[0]^input.w[1])+0x9e3779b9u*(uint32_t(j)+1u))&0x0fffffffu;
                    p.w[3]=0;
                    AesState r=hash<B>(p,keys,rounds,t);
                    #pragma unroll
                    for(int k=0;k<4;++k) acc.w[k]^=r.w[k];
                }
                out[i]=acc;
            });
    }).wait_and_throw();
}

template<class T> T* alloc(sycl::queue& q,size_t n){
    auto* p=sycl::malloc_device<T>(n,q);
    if(!p) throw std::bad_alloc();
    return p;
}

int main(int argc,char** argv) try {
    size_t n=argc>1?std::strtoull(argv[1],nullptr,10):(4ull<<20);
    int rounds=argc>2?std::atoi(argv[2]):256;
    if(n<256 || n>(1ull<<24) || rounds<1 || rounds>256)
        throw std::runtime_error("invalid n or rounds");
    n=(n+255)&~size_t(255);
    auto devices=sycl::device::get_devices(sycl::info::device_type::gpu);
    auto it=std::find_if(devices.begin(),devices.end(),[](const sycl::device& d){
        return d.get_info<sycl::info::device::name>().find("4070 SUPER")!=std::string::npos;
    });
    if(it==devices.end()) throw std::runtime_error("RTX 4070 SUPER not found; refusing fallback");
    sycl::queue q{*it, [](sycl::exception_list es){for(auto& e:es) std::rethrow_exception(e);}};
    std::printf("device: %s; states=%zu; AES rounds=%d\n",it->get_info<sycl::info::device::name>().c_str(),n,rounds*2);
    std::mt19937_64 rng(0x50C0A53C0C0A);
    AesHashKeys keys;
    for(int k=0;k<4;++k){keys.round_key_1.w[k]=uint32_t(rng());keys.round_key_2.w[k]=uint32_t(rng());}
    std::vector<AesState> in(n), ref(n), got(n);
    for(auto& s:in){s.w[0]=1; s.w[1]=uint32_t(rng())&3; s.w[2]=uint32_t(rng())&0x0fffffffu; s.w[3]=0;}
    std::vector<uint8_t> fixed(n,1), random(n), grouped(n);
    std::poisson_distribution<int> poisson(1.0);
    for(auto& c:random) c=static_cast<uint8_t>(std::min(255,poisson(rng)));
    // Same exact count histogram as random, but group equal counts into warps.
    // Shuffle whole warp groups so large counts are distributed over blocks.
    auto sorted=random;
    std::sort(sorted.begin(),sorted.end());
    std::vector<size_t> order(n/32);
    std::iota(order.begin(),order.end(),size_t(0));
    std::shuffle(order.begin(),order.end(),rng);
    for(size_t w=0;w<order.size();++w)
        std::copy_n(sorted.begin()+order[w]*32,32,grouped.begin()+w*32);
    auto* di=alloc<AesState>(q,n);auto* dc=alloc<uint8_t>(q,n);auto* dout=alloc<AesState>(q,n);
    q.memcpy(di,in.data(),n*sizeof(AesState)).wait_and_throw();
    auto* tables=sycl_backend::aes_tables_device(q);
    bool ok=true;
    auto scenario=[&](const char* name,const std::vector<uint8_t>& counts){
        uint64_t pairs=0, issued=0;
        for(size_t w=0;w<n;w+=32){
            unsigned maxc=0;
            for(size_t l=0;l<32;++l){pairs+=counts[w+l];maxc=std::max(maxc,unsigned(counts[w+l]));}
            issued+=uint64_t(maxc)*32;
        }
        double util=issued?100.0*double(pairs)/double(issued):100.0;
        std::printf("\n%s: pairs=%llu, mean=%.4f, modeled pairing lane utilization=%.2f%%\n",
            name,(unsigned long long)pairs,double(pairs)/n,util);
        q.memcpy(dc,counts.data(),n*sizeof(uint8_t)).wait_and_throw();
        double base=0;
        auto run=[&](auto tag){
            constexpr int B=decltype(tag)::value;
            launch<B>(q,tables,di,dc,dout,n,rounds,keys);
            launch<B>(q,tables,di,dc,dout,n,rounds,keys);
            double best=1e30;
            for(int j=0;j<5;++j){
                auto t=std::chrono::steady_clock::now();
                launch<B>(q,tables,di,dc,dout,n,rounds,keys);
                double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();
                best=std::min(best,ms);
            }
            q.memcpy(got.data(),dout,n*sizeof(AesState)).wait_and_throw();
            size_t bad=0;
            if constexpr(B==0){base=best;ref=got;}
            else for(size_t i=0;i<n;++i) if(!std::equal(got[i].w,got[i].w+4,ref[i].w))++bad;
            ok &= bad==0;
            std::printf("%-12s %2d  %9.3f ms  %8.3f Ghash/s  %6.3fx  %s (%zu mismatches)\n",
                B==0?"T-table":"Rotate",B,best,double(n+pairs)/best/1e6,base/best,bad?"FAIL":"OK",bad);
        };
        run(std::integral_constant<int,0>{});
        run(std::integral_constant<int,8>{});
        run(std::integral_constant<int,16>{});
        run(std::integral_constant<int,32>{});
    };
    scenario("FIXED (one pair per lane)",fixed);
    scenario("RANDOM (Poisson mean=1)",random);
    scenario("GROUPED (same histogram, paired counts grouped by warp)",grouped);
    sycl::free(di,q);sycl::free(dc,q);sycl::free(dout,q);
    return ok?0:1;
} catch(const std::exception& e){std::fprintf(stderr,"ERROR: %s\n",e.what());return 2;}
