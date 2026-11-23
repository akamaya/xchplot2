// Host-only arithmetic regression for exact AES last-word calculations.
// Build with the portable GPU headers; no SYCL installation required.
#include "gpu/AesLastRoundSycl.hpp"
#include "gpu/AesTables.inl"
#include <cstdint>
#include <cstdio>
#include <random>

using namespace pos2gpu;

static AesState full_reference(const AesState& s, const AesState& key)
{
    AesState out{};
    for (int w = 0; w < 4; ++w) {
        out.w[w] = aes_tables::T0[(s.w[w] & 255u)]
                 ^ aes_tables::T1[(s.w[(w+1)&3] >> 8) & 255u]
                 ^ aes_tables::T2[(s.w[(w+2)&3] >> 16) & 255u]
                 ^ aes_tables::T3[(s.w[(w+3)&3] >> 24)]
                 ^ key.w[w];
    }
    return out;
}

static bool test_one(const AesState& s, const AesState& key, const uint32_t* t, uint32_t bank)
{
    const AesState ref = full_reference(s, key);
    const uint32_t* lane = t + bank;
    if (aes_t0_last_word<0>(s, key, lane) != ref.w[0] ||
        aes_t0_last_word<1>(s, key, lane) != ref.w[1] ||
        aes_t0_last_word<2>(s, key, lane) != ref.w[2] ||
        aes_t0_last_word<3>(s, key, lane) != ref.w[3] ||
        aes_direct_last_word<0>(s, key, t, bank) != ref.w[0] ||
        aes_direct_last_word<1>(s, key, t, bank) != ref.w[1] ||
        aes_direct_last_word<2>(s, key, t, bank) != ref.w[2] ||
        aes_direct_last_word<3>(s, key, t, bank) != ref.w[3]) {
        return false;
    }
    return true;
}

int main()
{
    uint32_t t[256*16];
    for (int i = 0; i < 256; ++i)
        for (int j = 0; j < 16; ++j)
            t[i*16+j] = aes_tables::T0[i];
    std::mt19937_64 rng(0xA571A57120261123ULL);
    for (int sample = 0; sample < 10000; ++sample) {
        AesState s{}, key{};
        for (int i = 0; i < 4; ++i) {
            s.w[i] = static_cast<uint32_t>(rng());
            key.w[i] = static_cast<uint32_t>(rng());
        }
        if (sample == 0) s = {{0,0,0,0}};
        if (sample == 1) s = {{0xffffffffu,0xffffffffu,0xffffffffu,0xffffffffu}};
        const uint32_t bank = uint32_t(sample & 15);
        if (!test_one(s, key, t, bank)) {
            std::fprintf(stderr,"FAIL last round sample=%d bank=%u\n",sample,bank);
            return 1;
        }
        // Full chained AES hash (16 and 256 key pairs): explicit final-round
        // specialization must match an independent 4-table reference.
        if (sample < 200) {
            AesState k1{}, k2{};
            for (int i=0;i<4;++i) {k1.w[i]=uint32_t(rng());k2.w[i]=uint32_t(rng());}
            for (int pairs : {16,256}) {
                AesState full = s;
                AesState before = s;
                for (int p=0;p<pairs;++p) {
                    full = full_reference(full,k1);
                    full = full_reference(full,k2);
                }
                for (int p=0;p<pairs-1;++p) {
                    before = full_reference(before,k1);
                    before = full_reference(before,k2);
                }
                before = full_reference(before,k1);
                if (!test_one(before,k2,t,bank)) {
                    std::fprintf(stderr,"FAIL chain sample=%d pairs=%d\n",sample,pairs);
                    return 1;
                }
                if (aes_t0_last_word<0>(before,k2,t+bank)!=full.w[0] ||
                    aes_t0_last_word<3>(before,k2,t+bank)!=full.w[3]) {
                    std::fprintf(stderr,"FAIL chained hash sample=%d pairs=%d\n",sample,pairs);
                    return 1;
                }
            }
        }
    }
    std::puts("PASS: 10000 input/key states, all 4 output words, 16 bank replicas, and chained 32/512-round AES hashes");
    return 0;
}
