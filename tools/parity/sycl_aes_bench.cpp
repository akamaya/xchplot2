#include "gpu/SyclBackend.hpp"
#include "gpu/AesHashGpu.cuh"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <type_traits>
#include <new>

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

template<int B> class AesBenchKernel;
template<int B>
void launch(sycl::queue& q, const uint32_t* tables, const AesState* in,
            const AesState* kp, AesState* out, size_t n, int rounds) {
    constexpr size_t threads=256;
    constexpr size_t words=(B==0?4:B)*256;
    q.submit([&](sycl::handler& h){
        sycl::local_accessor<uint32_t,1> local{sycl::range<1>{words},h};
        h.parallel_for<AesBenchKernel<B>>(
            sycl::nd_range<1>{((n+threads-1)/threads)*threads,threads},
            [=](sycl::nd_item<1> it){
                uint32_t* t=&local[0];
                size_t lid=it.get_local_id(0);
                #pragma unroll 1
                for(size_t i=lid;i<words;i+=threads) {
                    if constexpr(B==0) t[i]=tables[i];
                    else t[i]=tables[i/B];
                }
                it.barrier(sycl::access::fence_space::local_space);
                size_t i=it.get_global_id(0);
                if(i>=n) return;
                AesHashKeys keys;
                keys.round_key_1=kp[(i/32)*2];
                keys.round_key_2=kp[(i/32)*2+1];
                if constexpr(B==0) out[i]=run_rounds_smem(in[i],keys,rounds,t);
                else out[i]=rotate_rounds<B>(in[i],keys,rounds,t+(lid&(B-1)));
            });
    }).wait_and_throw();
}

template<class T> T* alloc(sycl::queue& q,size_t n) {
    auto* p=sycl::malloc_device<T>(n,q);
    if(!p) throw std::bad_alloc();
    return p;
}

int main(int argc,char** argv) try {
    size_t n=argc>1?std::strtoull(argv[1],nullptr,10):(4ull<<20);
    int rounds=argc>2?std::atoi(argv[2]):256;
    if(!n || n>(1ull<<24) || rounds<1 || rounds>256) throw std::runtime_error("invalid n or rounds");
    n=(n+31)&~size_t(31);
    auto devices=sycl::device::get_devices(sycl::info::device_type::gpu);
    auto it=std::find_if(devices.begin(),devices.end(),[](const sycl::device& d){
        return d.get_info<sycl::info::device::name>().find("4070 SUPER")!=std::string::npos;
    });
    if(it==devices.end()) throw std::runtime_error("RTX 4070 SUPER not found; refusing fallback");
    sycl::queue q{*it, [](sycl::exception_list es){for(auto& e:es) std::rethrow_exception(e);}};
    std::printf("device: %s; states=%zu; AES rounds=%d\n",it->get_info<sycl::info::device::name>().c_str(),n,rounds*2);
    std::mt19937_64 rng(0x50C0A53C0C0A);
    std::vector<AesState> in(n), kp(n/16), ref(n), got(n);
    for(auto* v:{&in,&kp}) for(auto& s:*v) for(int j=0;j<4;++j) s.w[j]=uint32_t(rng());
    auto* di=alloc<AesState>(q,n); auto* dk=alloc<AesState>(q,kp.size()); auto* dout=alloc<AesState>(q,n);
    q.memcpy(di,in.data(),n*sizeof(AesState)).wait_and_throw();
    q.memcpy(dk,kp.data(),kp.size()*sizeof(AesState)).wait_and_throw();
    auto* tables=sycl_backend::aes_tables_device(q);
    double base=0; bool ok=true;
    auto run=[&](auto tag){
        constexpr int B=decltype(tag)::value;
        launch<B>(q,tables,di,dk,dout,n,rounds); // JIT warmup
        double best=1e30;
        for(int j=0;j<3;++j){
            auto t=std::chrono::steady_clock::now();
            launch<B>(q,tables,di,dk,dout,n,rounds);
            double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();
            best=std::min(best,ms);
        }
        q.memcpy(got.data(),dout,n*sizeof(AesState)).wait_and_throw();
        size_t bad=0;
        if constexpr(B==0) {base=best;ref=got;}
        else for(size_t i=0;i<n;++i) if(!std::equal(got[i].w,got[i].w+4,ref[i].w)) ++bad;
        ok &= bad==0;
        std::printf("%-12s %2d  %9.3f ms  %8.3f Ghash/s  %6.3fx  %s (%zu mismatches)\n",
            B==0?"T-table":"Rotate",B,best,double(n)/best/1e6,base/best,bad?"FAIL":"OK",bad);
    };
    run(std::integral_constant<int,0>{});
    run(std::integral_constant<int,1>{});
    run(std::integral_constant<int,4>{});
    run(std::integral_constant<int,8>{});
    run(std::integral_constant<int,16>{});
    run(std::integral_constant<int,32>{});
    sycl::free(di,q);sycl::free(dk,q);sycl::free(dout,q);
    return ok?0:1;
} catch(const std::exception& e){std::fprintf(stderr,"ERROR: %s\n",e.what());return 2;}
