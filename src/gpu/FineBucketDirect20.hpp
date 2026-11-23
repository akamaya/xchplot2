// FineBucketDirect20.hpp — experimental full 20-bit exact T2/T3 lookup.
// Automatically enabled for the fixed k28/s6 T2/T3 Fine16 compact profile.
// Layout: existing bucket offsets + u32 Fine16 lower bounds +
//         u32 Direct20 lower bounds[256 * (1<<20) + 1].
// The first/last values for (bucket,target) are direct[key], direct[key+1].
// The index lives in reused match/sort scratch; no extra simultaneous T2/T3
// index buffers. The compact SYCL match kernels consume Direct20.
#pragma once

#include "gpu/FineBucketIndex.hpp"

#include <sycl/sycl.hpp>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace pos2gpu {

inline constexpr int kFineDirect20FineBits = 16;
inline constexpr int kFineDirect20TargetBits = 20;
inline constexpr uint32_t kFineDirect20NumBuckets = 256;
inline constexpr uint32_t kFineDirect20SuffixBits = 4;
inline constexpr uint32_t kFineDirect20SuffixCount = 16;

inline bool fine_direct20_enabled(int fine_bits, int target_bits, uint32_t num_buckets = 256)
{
    return fine_bits == kFineDirect20FineBits &&
           target_bits == kFineDirect20TargetBits &&
           num_buckets == kFineDirect20NumBuckets;
}

inline size_t fine_direct20_bytes(uint32_t num_buckets)
{
    if (num_buckets != kFineDirect20NumBuckets)
        throw std::invalid_argument("T2/T3 Direct20 requires 256 match buckets (k28 s6)");
    return (size_t(num_buckets) * (size_t{1} << kFineDirect20TargetBits) + 1)
        * sizeof(uint32_t);
}

inline FineBucketOffset* fine_direct20_ptr(FineBucketOffset* fine_offsets, uint32_t num_buckets)
{
    if (num_buckets != kFineDirect20NumBuckets)
        throw std::invalid_argument("T2/T3 Direct20 requires 256 match buckets");
    return fine_offsets + (uint64_t(num_buckets) << kFineDirect20FineBits) + 1;
}

inline FineBucketOffset const* fine_direct20_ptr(FineBucketOffset const* fine_offsets, uint32_t num_buckets)
{
    if (num_buckets != kFineDirect20NumBuckets)
        throw std::invalid_argument("T2/T3 Direct20 requires 256 match buckets");
    return fine_offsets + (uint64_t(num_buckets) << kFineDirect20FineBits) + 1;
}

// Build one independent group for every 16-bit Fine key; each group emits
// 16 absolute lower bounds for the four-bit suffix. Memory use:
//   Fine16 = 64 MiB + sentinel, Direct20 = 1024 MiB + sentinel.
// No extra scratch, atomics or host copy. Empty groups and duplicates are
// represented by repeating the same absolute offset. For heavily skewed
// groups we avoid an extremely long serial scan per work item by switching
// to bounded binary searches; both paths produce identical lower bounds.
inline void fine_direct20_build(
    uint32_t const* sorted_mi,
    FineBucketOffset const* fine_offsets,
    FineBucketOffset* direct,
    uint32_t num_buckets,
    sycl::queue& q)
{
    if (!sorted_mi || !fine_offsets || !direct)
        throw std::invalid_argument("T2/T3 Direct20: null input or output");
    if (num_buckets != kFineDirect20NumBuckets)
        throw std::invalid_argument("T2/T3 Direct20: invalid bucket count");
    constexpr uint32_t fine_mask = (1u << kFineDirect20FineBits) - 1u;
    constexpr uint32_t target_mask = (1u << kFineDirect20TargetBits) - 1u;
    size_t const groups = size_t(num_buckets) << kFineDirect20FineBits;
    q.parallel_for(sycl::range<1>{groups}, [=](sycl::id<1> item) {
        size_t const g = item[0];
        uint32_t const begin = fine_offsets[g];
        uint32_t const end = fine_offsets[g + 1];
        uint32_t const base_target = (uint32_t(g) & fine_mask) << kFineDirect20SuffixBits;
        size_t const base_out = g << kFineDirect20SuffixBits;
        if (end - begin <= 1024u) {
            uint32_t pos = begin;
            for (uint32_t suffix = 0; suffix < kFineDirect20SuffixCount; ++suffix) {
                uint32_t const target = base_target | suffix;
                while (pos < end && (sorted_mi[pos] & target_mask) < target)
                    ++pos;
                direct[base_out + suffix] = pos;
            }
        } else {
            for (uint32_t suffix = 0; suffix < kFineDirect20SuffixCount; ++suffix) {
                uint32_t const target = base_target | suffix;
                uint32_t lo = begin, hi = end;
                while (lo < hi) {
                    uint32_t const mid = lo + ((hi - lo) >> 1);
                    if ((sorted_mi[mid] & target_mask) < target) lo = mid + 1;
                    else hi = mid;
                }
                direct[base_out + suffix] = lo;
            }
        }
        if (g == groups - 1)
            direct[groups << kFineDirect20SuffixBits] = fine_offsets[groups];
    }).wait();
}

} // namespace pos2gpu
