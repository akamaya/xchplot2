#include "host/MatchScratchAlias.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>

using namespace pos2gpu;

int main()
{
    constexpr std::size_t MiB = 1024 * 1024;
    constexpr std::size_t N = std::size_t{1} << 28;
    constexpr std::size_t cap = N + (N >> 6);
    constexpr std::size_t storage = cap * 3 * sizeof(std::uint32_t);
    constexpr auto t1 = match_scratch_idle_suffix(storage, N * 8);
    constexpr auto t23 = match_scratch_idle_suffix(storage, cap * 4);
    static_assert(t1.offset == 2048 * MiB);
    static_assert(t1.available == 1072 * MiB);
    static_assert(t23.offset == 1040 * MiB);
    static_assert(t23.available == 2080 * MiB);
    static_assert(match_scratch_can_alias(true, t1, 320 * MiB));
    static_assert(match_scratch_can_alias(true, t23, 320 * MiB));
    static_assert(!match_scratch_can_alias(true, t1, 1073 * MiB));
    static_assert(!match_scratch_can_alias(false, t1, 320 * MiB));
    static_assert(!match_scratch_can_alias(true, t23, 0));
    static_assert(match_scratch_backing_bytes(8 * MiB, 64 * 1024, 320 * MiB,
                                             320 * MiB, t1, t23, true) == 8 * MiB);
    static_assert(match_scratch_backing_bytes(8 * MiB, 64 * 1024, 320 * MiB,
                                             320 * MiB, t1, t23, false) == 320 * MiB);
    static_assert(match_scratch_backing_bytes(8 * MiB, 1073 * MiB, 320 * MiB,
                                             320 * MiB, t1, t23, true) == 1073 * MiB);
    static_assert(match_scratch_backing_bytes(0, 0, 0, 0, t1, t23, true) == 1);
    static_assert(match_scratch_backing_bytes(8, 17, 64, 32,
        match_scratch_idle_suffix(10, 20),
        match_scratch_idle_suffix(100, 70), true) == 64);
    static_assert(match_scratch_idle_suffix(10, 20).available == 0);
    std::cout << "match_scratch_alias_test: PASS (pool alias/fallback/lifetime boundaries)\n";
}
