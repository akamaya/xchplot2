#pragma once
// Experimental partial final AESENC-compatible round for Chia PoS2.
// Both functions return one *exact* 32-bit output word, with MixColumns.
// They assume the caller already performed all preceding AES rounds.
#include "gpu/AesHashGpu.cuh"
#include "gpu/AesDirectAddressSycl.hpp"
#include <cstdint>

namespace pos2gpu {

template <int Word>
static inline uint32_t aes_t0_last_word(
    const AesState& s, const AesState& key, const uint32_t* t_lane)
{
    static_assert(Word >= 0 && Word < 4, "AES word out of range");
    const uint32_t a = t_lane[(s.w[Word] & 255u) * 16u];
    const uint32_t b = t_lane[((s.w[(Word + 1) & 3] >> 8) & 255u) * 16u];
    const uint32_t c = t_lane[((s.w[(Word + 2) & 3] >> 16) & 255u) * 16u];
    const uint32_t d = t_lane[(s.w[(Word + 3) & 3] >> 24) * 16u];
    return a ^ ((b << 8) | (b >> 24)) ^ ((c << 16) | (c >> 16))
             ^ ((d >> 8) | (d << 24)) ^ key.w[Word];
}

template <int Word>
static inline uint32_t aes_direct_last_word(
    const AesState& s, const AesState& key,
    const uint32_t* t0_base, uint32_t replica)
{
    static_assert(Word >= 0 && Word < 4, "AES word out of range");
    const uint8_t* base = reinterpret_cast<const uint8_t*>(t0_base);
    const uint32_t rb = replica << 2;
    const uint32_t a = aes_direct_addr_t0_load<0>(base, s.w[Word], rb);
    const uint32_t b = aes_direct_addr_t0_load<1>(base, s.w[(Word + 1) & 3], rb);
    const uint32_t c = aes_direct_addr_t0_load<2>(base, s.w[(Word + 2) & 3], rb);
    const uint32_t d = aes_direct_addr_t0_load<3>(base, s.w[(Word + 3) & 3], rb);
    return a ^ ((b << 8) | (b >> 24)) ^ ((c << 16) | (c >> 16))
             ^ ((d >> 8) | (d << 24)) ^ key.w[Word];
}

} // namespace pos2gpu
