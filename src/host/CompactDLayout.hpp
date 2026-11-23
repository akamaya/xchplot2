// CompactDLayout.hpp -- device-free sizing constraints for experimental
// Candidate D. Only the pooled pipeline uses this layout; streaming is intact.
#pragma once

#include <cstddef>
#include <cstdlib>
#include <limits>

namespace pos2gpu {

// Explicit opt-in. Leave A+B+C as the default until GPU parity and A/B pass.
inline bool compact_d_requested() noexcept
{
    return true;
}

constexpr bool compact_d_layout_fits(
    bool requested, bool t2_xbits_reuse, bool c_layout_requested,
    std::size_t total_xs, std::size_t cap, std::size_t storage_bytes,
    std::size_t pair_b_bytes, std::size_t xs_cub_bytes) noexcept
{
    constexpr auto MAX = (std::numeric_limits<std::size_t>::max)();
    if (!requested || !t2_xbits_reuse || !c_layout_requested ||
        cap > MAX / 12 || total_xs > MAX / 8) return false;
    // Xs: keys_a + vals_a in storage, keys_b + vals_b in pair_b,
    // CUB scratch directly after both storage input streams.
    if (xs_cub_bytes > MAX - total_xs * 8) return false;
    return pair_b_bytes >= total_xs * 8 &&
           storage_bytes >= total_xs * 8 + xs_cub_bytes &&
           // T1 match writes mi after Xs[total_xs], not into pair_a.
           storage_bytes >= total_xs * 8 + cap * 4 &&
           // T2 match simultaneously keeps sorted T1 keys, mi and xbits.
           storage_bytes >= cap * 12;
}

} // namespace pos2gpu
