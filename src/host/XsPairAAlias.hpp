// XsPairAAlias.hpp — device-free Candidate C pool layout planner.
//
// When B is active, d_pair_a has 12*cap bytes. Its contents are not live
// before T1 match, so it can host Xs's temporary workspace. T2 match-info
// moves into d_storage[4*cap..8*cap) while T2 unsorted xbits move to
// d_pair_a[8*cap..12*cap). T2 sort borrows pair_b[0..4*cap) for its
// identity input, which dies before sorted T2 meta overwrites pair_b.
// Consequently d_pair_b needs only 8*cap, not 12*cap, bytes.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>

namespace pos2gpu {

inline bool xs_pair_a_alias_requested() noexcept
{
    return true;
}

constexpr bool xs_pair_a_alias_fits(bool requested, bool t2_reuse,
                                    std::size_t cap, std::size_t total_xs,
                                    std::size_t storage_bytes,
                                    std::size_t pair_a_bytes,
                                    std::size_t xs_temp_bytes) noexcept
{
    if (!requested || !t2_reuse) return false;
    if (cap > (std::numeric_limits<std::size_t>::max)() / 12 ||
        total_xs > (std::numeric_limits<std::size_t>::max)() / 12)
        return false;
    // Storage must fit the three u32[cap] sort slots AND the external
    // Xs keys_a after XsCandidateGpu[total_xs] (8 bytes each).
    return storage_bytes >= cap * 12 &&
           storage_bytes >= total_xs * 12 &&
           pair_a_bytes >= cap * 12 &&
           xs_temp_bytes <= pair_a_bytes;
}

constexpr std::size_t xs_pair_a_pair_b_bytes(
    std::size_t cap, std::size_t xs_temp_bytes,
    bool t2_reuse, bool xs_pair_a_alias) noexcept
{
    // With C active T2 sorted meta and T1 sorted meta need only 8*cap.
    // Without C, B's unsorted T2 xbits needs [8*cap..12*cap), and
    // the Xs temp also stays in pair_b.
    std::size_t const sorted_bytes =
        cap * ((t2_reuse && xs_pair_a_alias) ? 8 : 12);
    if (xs_pair_a_alias || sorted_bytes >= xs_temp_bytes)
        return sorted_bytes;
    return xs_temp_bytes;
}

constexpr std::size_t xs_pair_a_t2_match_live_prefix(
    std::size_t cap, bool xs_pair_a_alias) noexcept
{
    // C's T2 mi output occupies storage slot 1, so scratch starts at slot 2.
    return cap * (xs_pair_a_alias ? 8 : 4);
}

} // namespace pos2gpu
