// MatchScratchAlias.hpp — safe, device-free lifetime-based scratch planning.
// The pool's d_storage serves mutually exclusive phases:
//   T1 match: Xs candidates occupy its prefix.
//   T2/T3 match: sorted match keys occupy its first u32[cap] slot.
// The suffix of d_storage can hold the match-index scratch in those phases.
#pragma once

#include <cstddef>
#include <cstdlib>

namespace pos2gpu {

struct MatchScratchRegion {
    std::size_t offset = 0;
    std::size_t available = 0;
};

constexpr MatchScratchRegion match_scratch_idle_suffix(
    std::size_t storage_bytes, std::size_t live_prefix_bytes) noexcept
{
    if (live_prefix_bytes > storage_bytes) return {};
    return {live_prefix_bytes, storage_bytes - live_prefix_bytes};
}

constexpr bool match_scratch_can_alias(bool enabled,
                                        MatchScratchRegion region,
                                        std::size_t need) noexcept
{
    return enabled && need != 0 && need <= region.available;
}

// At least one byte keeps the legacy sort scratch pointer non-null on
// backends whose sort sizing returns zero; the match alias can then be zero.
constexpr std::size_t match_scratch_backing_bytes(
    std::size_t sort_bytes,
    std::size_t t1_need, std::size_t t2_need, std::size_t t3_need,
    MatchScratchRegion t1_region, MatchScratchRegion t23_region,
    bool enabled) noexcept
{
    std::size_t out = sort_bytes != 0 ? sort_bytes : std::size_t{1};
    if (!match_scratch_can_alias(enabled, t1_region, t1_need) && t1_need > out)
        out = t1_need;
    if (!match_scratch_can_alias(enabled, t23_region, t2_need) && t2_need > out)
        out = t2_need;
    if (!match_scratch_can_alias(enabled, t23_region, t3_need) && t3_need > out)
        out = t3_need;
    return out;
}

// Opt out for a controlled A/B from one executable. Read at pool construction
// and at phase dispatch; do not change this variable during a running plot.
inline bool match_scratch_alias_enabled() noexcept
{
    char const* v = std::getenv("POS2GPU_MATCH_INDEX_ALIAS");
    return !(v && v[0] == '0' && v[1] == '\0');
}

} // namespace pos2gpu
