// Windows CUDA-only compatibility header for the pos2-chip Feistel key ABI.
// This is an independent key-schedule implementation, NOT a copy of the
// upstream source. It follows the public PoS2 specification's SHA-256 based
// 4-round derivation and preserves the FeistelCipher::FullRoundKey layout.
// Only this standalone CUDA target puts tools/native_cuda_only on the include
// path ahead of external pos2-chip headers. The hybrid CLI is untouched.
#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

class FeistelCipher {
public:
    struct RoundKey { uint32_t b, c, d; };
    struct FullRoundKey { RoundKey round[4]; };
    uint32_t k_;
    FullRoundKey round_key_{};

    explicit FeistelCipher(uint8_t const* plot_id, uint32_t k) : k_(k) {
        if(!plot_id || k!=28)throw std::invalid_argument(
            "CUDA-only Feistel key: expected plot id and k=28");
        BCRYPT_ALG_HANDLE algorithm=nullptr;
        NTSTATUS status=BCryptOpenAlgorithmProvider(
            &algorithm,BCRYPT_SHA256_ALGORITHM,nullptr,0);
        if(status<0)throw std::runtime_error("BCryptOpenAlgorithmProvider(SHA256) failed");
        uint8_t prev[32];
        std::memcpy(prev,plot_id,32);
        constexpr char salt[]="ChiaPos2Fesitel";
        static_assert(sizeof(salt)==16);
        try {
            for(int round=0;round<4;++round) {
                uint8_t message[48];
                std::memcpy(message,prev,32);
                std::memcpy(message+32,salt,15);
                message[47]=uint8_t(round+1);
                BCRYPT_HASH_HANDLE hash=nullptr;
                status=BCryptCreateHash(algorithm,&hash,nullptr,0,nullptr,0,0);
                if(status<0)throw std::runtime_error("BCryptCreateHash failed");
                status=BCryptHashData(hash,message,48,0);
                if(status>=0)status=BCryptFinishHash(hash,prev,32,0);
                (void)BCryptDestroyHash(hash);
                if(status<0)throw std::runtime_error("BCryptHashData/FinishHash failed");
                // Extract consecutive 28-bit words from the digest MSB-first.
                // For each i, take bits [28*i,28*(i+1)) from the SHA-256
                // digest, interpreting bit 0 as the high bit of byte 0.
                uint32_t words[3]{};
                for(int i=0;i<3;++i) {
                    for(int j=0;j<28;++j) {
                        int bit=i*28+j;
                        words[i]=(words[i]<<1) |
                            ((prev[bit>>3] >> (7-(bit&7)))&1u);
                    }
                }
                round_key_.round[round]={words[0],words[1],words[2]};
            }
        } catch(...) {
            (void)BCryptCloseAlgorithmProvider(algorithm,0);
            throw;
        }
        (void)BCryptCloseAlgorithmProvider(algorithm,0);
    }
};
static_assert(sizeof(FeistelCipher::RoundKey)==12);
static_assert(sizeof(FeistelCipher::FullRoundKey)==48);
