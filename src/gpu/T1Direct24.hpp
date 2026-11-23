// T1Direct24.hpp — experimental full 24-bit exact-match index (SYCL path).
// Automatically enabled for the fixed k28/s6 T1 Fine18 compact profile.
// The index is a u32 lower-bound table of 16 * 2^24 + 1 elements.
// Reuses the T1 match-prepare scratch; never change the environment mid-process.
#pragma once

#include "gpu/XsCandidateGpu.hpp"

#include <sycl/sycl.hpp>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace pos2gpu {

inline constexpr int kT1Direct24FineBits = 18;
inline constexpr int kT1Direct24TargetBits = 24;
inline constexpr uint32_t kT1Direct24SuffixBits = 6;
inline constexpr uint32_t kT1Direct24NumBuckets = 16;

inline bool t1_direct24_enabled(int fine_bits, int target_bits, uint32_t num_buckets)
{
    return fine_bits == kT1Direct24FineBits &&
           target_bits == kT1Direct24TargetBits &&
           num_buckets == kT1Direct24NumBuckets;
}

inline size_t t1_direct24_bytes(uint32_t num_buckets)
{
    return (size_t(num_buckets) * (size_t{1} << kT1Direct24TargetBits) + 1)
        * sizeof(uint32_t);
}

// A group is one 18-bit fine bucket holding exactly 64 target values.
// For ordinary groups, advance a monotonic scan across their sorted entries
// and emit 64 lower bounds. This is O(n_rows + n_targets) across all groups,
// not 2^24 independent binary searches. For skewed groups, use bounded
// binary searches to avoid an exceptionally long single-work-item scan.
// The final direct offset is an extra sentinel: direct[16*2^24] = total.
inline void launch_t1_direct24_build(
    XsCandidateGpu const* d_sorted,
    uint64_t const* d_fine_offsets,
    uint64_t total,
    uint32_t num_buckets,
    uint32_t* d_direct_offsets,
    sycl::queue& q)
{
    if (!d_sorted || !d_fine_offsets || !d_direct_offsets)
        throw std::invalid_argument("T1 Direct24: null input/index pointer");
    if (num_buckets != kT1Direct24NumBuckets ||
        total > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("T1 Direct24: invalid buckets or >u32 row count");

    constexpr uint32_t fine_count = (1u << kT1Direct24FineBits);
    constexpr uint32_t suffix_count = (1u << kT1Direct24SuffixBits);
    constexpr uint32_t target_mask = (1u << kT1Direct24TargetBits) - 1u;
    uint64_t const groups = uint64_t(num_buckets) * fine_count;

    q.parallel_for(sycl::range<1>{static_cast<size_t>(groups)}, [=](sycl::id<1> item) {
        uint64_t const g = item[0];
        uint64_t const begin = d_fine_offsets[g];
        uint64_t const end = d_fine_offsets[g + 1];
        uint32_t const target_base =
            (static_cast<uint32_t>(g) & (fine_count - 1u)) << kT1Direct24SuffixBits;
        uint64_t const output_base = g * suffix_count;

        if (end - begin <= 1024u) {
            uint64_t pos = begin;
            for (uint32_t suffix = 0; suffix < suffix_count; ++suffix) {
                uint32_t const target = target_base + suffix;
                while (pos < end &&
                       (d_sorted[pos].match_info & target_mask) < target)
                    ++pos;
                d_direct_offsets[output_base + suffix] = static_cast<uint32_t>(pos);
            }
        } else {
            // Exact skew-safe fallback: at most 64 bounded lower_bounds.
            for (uint32_t suffix = 0; suffix < suffix_count; ++suffix) {
                uint32_t const target = target_base + suffix;
                uint64_t lo = begin, hi = end;
                while (lo < hi) {
                    uint64_t const mid = lo + ((hi - lo) >> 1);
                    if ((d_sorted[mid].match_info & target_mask) < target)
                        lo = mid + 1;
                    else
                        hi = mid;
                }
                d_direct_offsets[output_base + suffix] = static_cast<uint32_t>(lo);
            }
        }
        if (g == groups - 1)
            d_direct_offsets[groups * suffix_count] = static_cast<uint32_t>(total);
    }).wait();
}

} // namespace pos2gpu
