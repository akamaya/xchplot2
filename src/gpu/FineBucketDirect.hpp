// FineBucketDirect.hpp — experimental 16+4 exact target lookup for T2/T3.
// Retired 16+4 direct index: the fixed profile uses Direct20 instead.
// Layout: u32 fine_offsets[256*65536+1], then u8 suffix_start[256*2^20].
// Each 16-bit group stores 16 lower_bound relative offsets (0..254).
// A group containing >=255 rows uses 0xFF for all suffixes and falls back
// to the original exact lower_bound. No pair can be lost due to skew.
#pragma once

#include "gpu/MatchKernelCommon.cuh"
#include "gpu/FineBucketIndex.hpp"
#include <sycl/sycl.hpp>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

namespace pos2gpu {

inline bool fine_direct16_enabled(int fine_bits, int target_bits)
{
    (void)fine_bits;
    (void)target_bits;
    return false;
}

inline size_t fine_direct16_bytes(uint32_t num_buckets)
{
    return size_t(num_buckets) << 20; // 16 suffix offsets per each 16-bit group
}

inline uint8_t* fine_direct16_ptr(FineBucketOffset* fine_offsets, uint32_t num_buckets)
{
    return reinterpret_cast<uint8_t*>(fine_offsets + (uint64_t(num_buckets) << 16) + 1);
}

// One work item per 16-bit group, writing 16 consecutive bytes.
// Does not use atomics or allocate additional GPU scratch.
// This first implementation favors correctness/simplicity over build speed;
// profile the 'direct_build' phase before adopting it.
inline void fine_direct16_build(
    uint32_t const* sorted_mi, FineBucketOffset const* fine_offsets,
    uint8_t* direct, uint32_t num_buckets, sycl::queue& q)
{
    size_t const groups = size_t(num_buckets) << 16;
    q.parallel_for(sycl::range<1>{groups}, [=](sycl::id<1> item) {
        uint64_t const g = item[0];
        uint64_t const begin = fine_offsets[g];
        uint64_t const end = fine_offsets[g + 1];
        uint8_t* dst = direct + (g << 4);
        if (end - begin >= 255) {
            for (uint32_t suffix = 0; suffix < 16; ++suffix)
                dst[suffix] = uint8_t(255);
            return;
        }
        uint64_t pos = begin;
        for (uint32_t suffix = 0; suffix < 16; ++suffix) {
            while (pos < end && (sorted_mi[pos] & 15u) < suffix) ++pos;
            dst[suffix] = static_cast<uint8_t>(pos - begin);
        }
    }).wait();
}

struct FineDirect16Range {
    uint64_t first;
    uint64_t end;
};

POS2_DEVICE_INLINE FineDirect16Range fine_direct16_find(
    uint32_t const* sorted_mi, FineBucketOffset const* fine_offsets,
    uint8_t const* direct, uint64_t fine_idx,
    uint32_t target, uint32_t target_mask)
{
    uint64_t const group_lo = fine_offsets[fine_idx];
    uint64_t const group_hi = fine_offsets[fine_idx + 1];
    if (direct) {
        uint32_t const suffix = target & 15u;
        uint64_t const suffix_idx = (fine_idx << 4) | suffix;
        uint8_t const rel_lo = direct[suffix_idx];
        if (rel_lo != 255u) {
            uint64_t const first = group_lo + rel_lo;
            uint64_t const last = suffix == 15u
                ? group_hi : group_lo + direct[suffix_idx + 1];
            return {first, last};
        }
    }
    uint64_t const first = fine_bucket_lower_bound(
        sorted_mi, group_lo, group_hi, target, target_mask);
    return {first, group_hi};
}

} // namespace pos2gpu
