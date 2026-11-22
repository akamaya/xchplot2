#include "host/MatchScratchAlias.hpp"
#include "host/T2XbitsReuse.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>

using namespace pos2gpu;

int main()
{
    constexpr std::size_t MiB = 1024 * 1024;
    constexpr std::size_t n = std::size_t{1} << 28;
    constexpr std::size_t cap = n + (n >> 6);
    constexpr std::size_t storage = cap * 3 * sizeof(std::uint32_t);
    constexpr std::size_t pair_b = cap * (sizeof(std::uint64_t) + sizeof(std::uint32_t));
    static_assert(t2_xbits_reuse_layout_fits(cap, pair_b, storage));
    static_assert(!t2_xbits_reuse_layout_fits(cap, pair_b - 1, storage));
    static_assert(!t2_xbits_reuse_layout_fits(cap, pair_b, storage - 1));
    static_assert(t2_xbits_unsorted_offset(cap) == 2080 * MiB);
    static_assert(t2_xbits_sorted_offset(cap) == 1040 * MiB);
    static_assert(t3_match_live_prefix_bytes(cap, true) == 2080 * MiB);
    static_assert(t3_match_live_prefix_bytes(cap, false) == 1040 * MiB);
    // T2's unsorted xbits is disjoint from the T1 sorted-meta input.
    static_assert(t2_xbits_unsorted_offset(cap) + cap * sizeof(std::uint32_t) == pair_b);
    // T2 keys_out, sorted xbits, and T3 Fine scratch are disjoint.
    constexpr auto t2_idle = match_scratch_idle_suffix(storage, cap * sizeof(std::uint32_t));
    constexpr auto t3_idle = match_scratch_idle_suffix(storage, t3_match_live_prefix_bytes(cap, true));
    static_assert(t2_idle.available == 2080 * MiB);
    static_assert(t3_idle.offset == 2080 * MiB);
    static_assert(t3_idle.available == 1040 * MiB);
    static_assert(match_scratch_can_alias(true, t3_idle, 320 * MiB));
    static_assert(!match_scratch_can_alias(true, t3_idle, 1041 * MiB));
    constexpr auto t1_idle = match_scratch_idle_suffix(storage, n * 8);
    // The T3 alias is more restrictive than T2's; sizing must check both.
    static_assert(match_scratch_backing_bytes(68 * MiB, 16, 320 * MiB,
        320 * MiB, t1_idle, t2_idle, t3_idle, true) == 68 * MiB);
    static_assert(match_scratch_backing_bytes(68 * MiB, 16, 320 * MiB,
        1041 * MiB, t1_idle, t2_idle, t3_idle, true) == 1041 * MiB);
    static_assert(match_scratch_backing_bytes(68 * MiB, 16, 320 * MiB,
        320 * MiB, t1_idle, t2_idle, t3_idle, false) == 320 * MiB);
    // Higher k / oversized capacity never causes a wraparound fit.
    static_assert(!t2_xbits_reuse_layout_fits(
        (std::numeric_limits<std::size_t>::max)() / 12 + 1,
        pair_b, storage));
    assert(t2_xbits_reuse_layout_fits(cap, pair_b, storage));
    std::cout << "t2_xbits_reuse_test: PASS (pool footprints, T2/T3 lifetimes, alias fallback)\n";
}
