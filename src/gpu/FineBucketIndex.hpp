// FineBucketIndex.hpp — 32-bit absolute row offsets for T2/T3 fine buckets.
// Coarse bucket offsets stay uint64_t; only fine offsets are narrowed.
// The sorted input row count must fit in FineBucketOffset before GPU indexing.
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace pos2gpu {

using FineBucketOffset = std::uint32_t;
static_assert(sizeof(FineBucketOffset) == 4);

inline void validate_fine_bucket_input_count(std::uint64_t count)
{
    if (count > std::numeric_limits<FineBucketOffset>::max())
        throw std::invalid_argument("T2/T3 sorted input exceeds 32-bit fine bucket offset capacity");
}

struct FineBucketTempLayout {
    std::uint64_t fine_entries;
    std::size_t bucket_bytes;
    std::size_t fine_bytes;

    std::size_t total_bytes() const noexcept { return bucket_bytes + fine_bytes; }
};

// 64-bit coarse bucket array, then a packed 32-bit fine array.
// When enabled, the Direct suffix array follows the fine array directly.
inline FineBucketTempLayout fine_bucket_temp_layout(std::uint32_t num_buckets, int fine_bits)
{
    if (fine_bits < 1 || fine_bits > 16)
        throw std::invalid_argument("invalid T2/T3 fine bucket bit width");
    std::uint64_t const groups = std::uint64_t(num_buckets) << fine_bits;
    if (groups > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("fine bucket index exceeds 32-bit GPU launch count");
    return {groups + 1,
            sizeof(std::uint64_t) * (std::size_t(num_buckets) + 1),
            sizeof(FineBucketOffset) * std::size_t(groups + 1)};
}

inline FineBucketOffset* fine_bucket_offset_ptr(void* storage, std::size_t bucket_bytes) noexcept
{
    return reinterpret_cast<FineBucketOffset*>(static_cast<std::uint8_t*>(storage) + bucket_bytes);
}

inline FineBucketOffset const* fine_bucket_offset_ptr(void const* storage, std::size_t bucket_bytes) noexcept
{
    return reinterpret_cast<FineBucketOffset const*>(
        static_cast<std::uint8_t const*>(storage) + bucket_bytes);
}

} // namespace pos2gpu
