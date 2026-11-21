#pragma once

// Shared16 Tezcan AES: calculate the T0 byte address from the original
// 32-bit state word instead of extracting a byte then scaling its index.
// Layout is unchanged: t[index * 16 + replica], replica in [0, 15].
// Only SYCL Tezcan paths include this header; native CUDA is unchanged.

#include <cstdint>

namespace pos2gpu {

template <int Byte>
static inline uint32_t aes_direct_addr_t0_load(
    const uint8_t* base, uint32_t word, uint32_t replica_bytes)
{
    static_assert(Byte >= 0 && Byte < 4, "AES byte index out of range");
    uint32_t shifted;
    if constexpr (Byte == 0) shifted = word << 6;
    else if constexpr (Byte == 1) shifted = word >> 2;
    else if constexpr (Byte == 2) shifted = word >> 10;
    else shifted = word >> 18;

    const uint32_t offset = (shifted & 0x3fc0u) | replica_bytes;
    return *reinterpret_cast<const uint32_t*>(base + offset);
}

static inline AesState aes_direct_addr_t0_round(
    AesState s, const AesState& key, const uint32_t* t, uint32_t replica)
{
    const uint8_t* base = reinterpret_cast<const uint8_t*>(t);
    const uint32_t replica_bytes = replica << 2;
    AesState out;
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        const uint32_t a = aes_direct_addr_t0_load<0>(
            base, s.w[c], replica_bytes);
        uint32_t b = aes_direct_addr_t0_load<1>(
            base, s.w[(c + 1) & 3], replica_bytes);
        uint32_t d = aes_direct_addr_t0_load<2>(
            base, s.w[(c + 2) & 3], replica_bytes);
        uint32_t e = aes_direct_addr_t0_load<3>(
            base, s.w[(c + 3) & 3], replica_bytes);
        b = (b << 8) | (b >> 24);
        d = (d << 16) | (d >> 16);
        e = (e >> 8) | (e << 24);
        out.w[c] = a ^ b ^ d ^ e ^ key.w[c];
    }
    return out;
}

} // namespace pos2gpu
