// T2XbitsReuse.hpp — pure layout helpers for pooled T2 xbits relocation.
//
// During T2 Match, pair_b[0..8*cap) contains sorted T1 meta; the tail
// pair_b[8*cap..12*cap) can receive unsorted T2 xbits.
// After T2 radix sort, d_storage contains keys_out, vals_in, vals_out.
// vals_in is dead once CUB returns (the result is normalised to vals_out),
// and can receive sorted T2 xbits. During T3 Match, both keys_out and
// sorted xbits are live, leaving only vals_out as Fine index alias scratch.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>

namespace pos2gpu {

constexpr std::size_t t2_xbits_unsorted_offset(std::size_t cap) noexcept
{
    return cap * sizeof(std::uint64_t);
}

constexpr std::size_t t2_xbits_sorted_offset(std::size_t cap) noexcept
{
    return cap * sizeof(std::uint32_t);
}

constexpr std::size_t t3_match_live_prefix_bytes(std::size_t cap,
                                                  bool t2_reuse) noexcept
{
    return cap * sizeof(std::uint32_t) * (t2_reuse ? 2 : 1);
}

constexpr bool t2_xbits_reuse_layout_fits(std::size_t cap,
                                           std::size_t pair_b_bytes,
                                           std::size_t storage_bytes) noexcept
{
    // Avoid overflow in all later offset/footprint calculations.
    return cap <= (std::numeric_limits<std::size_t>::max)() / 12
        && pair_b_bytes >= cap * 12
        && storage_bytes >= cap * 12;
}

// OFF for same-executable A/B. Read once when the pool is constructed and
// keep that choice on the pool object for the lifetime of a plot/batch.
inline bool t2_xbits_reuse_enabled() noexcept
{
    char const* v = std::getenv("POS2GPU_T2_XBITS_REUSE");
    return !(v && v[0] == '0' && v[1] == '\0');
}

} // namespace pos2gpu
