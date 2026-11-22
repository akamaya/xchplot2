#include "host/MatchScratchAlias.hpp"
#include "host/T2XbitsReuse.hpp"
#include "host/XsPairAAlias.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>

using namespace pos2gpu;

struct Span { std::size_t begin, end; };
constexpr bool separate(Span a, Span b) noexcept
{ return a.end <= b.begin || b.end <= a.begin; }

int main()
{
    constexpr std::size_t MiB = 1024 * 1024;
    constexpr std::size_t n = std::size_t{1} << 28;
    constexpr std::size_t cap = n + (n >> 6);
    constexpr std::size_t storage = cap * 12;
    constexpr std::size_t pair_a = cap * 12;
    constexpr std::size_t xs_temp = 3095 * MiB; // 3*N*u32 + <=23 MiB scratch

    static_assert(xs_pair_a_alias_fits(true, true, cap, n, storage, pair_a, xs_temp));
    static_assert(!xs_pair_a_alias_fits(false, true, cap, n, storage, pair_a, xs_temp));
    static_assert(!xs_pair_a_alias_fits(true, false, cap, n, storage, pair_a, xs_temp));
    static_assert(!xs_pair_a_alias_fits(true, true, cap, n, storage, pair_a, pair_a+1));
    static_assert(!xs_pair_a_alias_fits(true, true, cap, n, storage-1, pair_a, xs_temp));
    static_assert(!xs_pair_a_alias_fits(true, true, cap, n, storage, pair_a-1, xs_temp));
    static_assert(xs_pair_a_pair_b_bytes(cap, xs_temp, true, true) == 2080 * MiB);
    static_assert(xs_pair_a_pair_b_bytes(cap, xs_temp, true, false) == 3120 * MiB);
    static_assert(xs_pair_a_pair_b_bytes(cap, xs_temp, false, false) == 3120 * MiB);
    static_assert(xs_pair_a_t2_match_live_prefix(cap, true) == 2080 * MiB);
    static_assert(xs_pair_a_t2_match_live_prefix(cap, false) == 1040 * MiB);
    static_assert(t2_xbits_reuse_layout_fits(cap, 2080 * MiB, storage, true));
    static_assert(!t2_xbits_reuse_layout_fits(cap, 2080 * MiB, storage, false));

    // T2 Match: d_storage keys_in (slot 0), T2 mi output (slot 1),
    // Fine scratch (slot 2) have disjoint ranges.
    constexpr Span st_keys{0, 4*cap}, st_mi{4*cap, 8*cap};
    constexpr Span st_fine{8*cap, storage};
    static_assert(separate(st_keys, st_mi) && separate(st_mi, st_fine));
    // T2 Match: pair_a meta+xbits are disjoint; pair_b T1 sorted-meta
    // input has all 8*cap bytes, and is not written in this phase.
    constexpr Span a_meta{0, 8*cap}, a_xbits{8*cap, 12*cap};
    static_assert(separate(a_meta, a_xbits));
    // T2 Sort: pair_b[0..4*cap) identity input dies after sort;
    // d_storage's vals_out/indices (slot 2) survive the permutation.
    constexpr Span b_vals_in{0, 4*cap}, b_sorted_meta{0, 8*cap};
    constexpr Span st_vals_out{8*cap, 12*cap};
    static_assert(separate(st_vals_out, st_mi));
    static_assert(separate(st_vals_out, st_keys));
    static_assert(b_vals_in.end <= b_sorted_meta.end); // sequential reuse
    // Permute: T2 mi (slot 1) is dead and becomes sorted xbits.
    // Both are read/written only in separate phases, not simultaneously.
    constexpr auto t2_idle = match_scratch_idle_suffix(storage,
        xs_pair_a_t2_match_live_prefix(cap, true));
    static_assert(t2_idle.offset == 2080 * MiB);
    static_assert(t2_idle.available == 1040 * MiB);
    static_assert(match_scratch_can_alias(true, t2_idle, 320*MiB));
    static_assert(!match_scratch_can_alias(true, t2_idle, 1041*MiB));
    // Legacy B path and C path both retain stable A scratch sizing.
    constexpr auto t1_idle = match_scratch_idle_suffix(storage, n*8);
    constexpr auto t3_idle = match_scratch_idle_suffix(storage,
        t3_match_live_prefix_bytes(cap, true));
    static_assert(match_scratch_backing_bytes(68*MiB, 16, 320*MiB,
        320*MiB, t1_idle, t2_idle, t3_idle, true) == 68*MiB);
    assert(xs_pair_a_pair_b_bytes(cap, xs_temp, true, true) == 2080*MiB);
    std::cout << "xs_pair_a_alias_test: PASS (Xs/T2/T3 lifetime and size planning)\n";
}
