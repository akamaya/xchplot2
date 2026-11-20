// MatchDiag.hpp: optional, GPU-side candidate histogram for T1/T2 local compaction.
// Diagnostic probes run separately. Nothing is allocated when the mode is off.
#pragma once

#include <sycl/sycl.hpp>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace pos2gpu {

struct MatchDiagOptions {
    bool enabled = false;
    unsigned stride = 128; // sample every 128th original workgroup in each bucket

    static MatchDiagOptions read() {
        MatchDiagOptions d;
        const char* e = std::getenv("POS2GPU_MATCH_DIAG");
        d.enabled = e && e[0] == '1';
        if (!d.enabled) return d;
        if (const char* s = std::getenv("POS2GPU_MATCH_DIAG_STRIDE"); s && *s) {
            errno = 0;
            char* end = nullptr;
            unsigned long v = std::strtoul(s, &end, 10);
            if (errno || s == end || *end || v < 1 || v > 65536)
                throw std::invalid_argument("POS2GPU_MATCH_DIAG_STRIDE must be 1..65536");
            d.stride = static_cast<unsigned>(v);
        }
        return d;
    }
};

struct MatchDiagStats {
    unsigned long long fallback_groups; // exact, from ALL production compact groups
    unsigned long long sampled_groups;
    unsigned long long sampled_candidates;
    unsigned long long max_group_candidates;
    unsigned long long max_lane_candidates;
    unsigned long long bins[8];
};

inline void record_match_diag_group(MatchDiagStats* stats,
                                    unsigned long long group_candidates,
                                    unsigned long long max_lane)
{
    using Atomic = sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device>;
    Atomic{stats->sampled_groups}.fetch_add(1ULL);
    Atomic{stats->sampled_candidates}.fetch_add(group_candidates);
    Atomic{stats->max_group_candidates}.fetch_max(group_candidates);
    Atomic{stats->max_lane_candidates}.fetch_max(max_lane);
    unsigned bin = group_candidates == 0 ? 0 :
                   group_candidates <= 128 ? 1 :
                   group_candidates <= 256 ? 2 :
                   group_candidates <= 384 ? 3 :
                   group_candidates <= 512 ? 4 :
                   group_candidates <= 768 ? 5 :
                   group_candidates <= 1024 ? 6 : 7;
    Atomic{stats->bins[bin]}.fetch_add(1ULL);
}

inline void print_match_diag_stats(int table_id, unsigned stride,
                                   double aes_ms, double aes_search_ms, double pair_ms,
                                   double fused_ms, MatchDiagStats const& s,
                                   unsigned long long checksum)
{
    std::fprintf(stderr,
        "[match-diag] T%d stride=%u aes_target=%.3fms aes_target_search=%.3fms "
        "search_delta=%.3fms aes_search_pair=%.3fms pair_delta=%.3fms "
        "production_fused=%.3fms checksum=%llu\n",
        table_id, stride, aes_ms, aes_search_ms, aes_search_ms - aes_ms,
        pair_ms, pair_ms - aes_search_ms, fused_ms, checksum);
    std::fprintf(stderr,
        "[match-diag] T%d sampled_groups=%llu sampled_candidates=%llu "
        "mean_candidates_per_group=%.3f max_group_candidates(sample)=%llu "
        "max_lane_candidates(sample)=%llu fallback_over_512_groups(EXACT)=%llu\n",
        table_id, s.sampled_groups, s.sampled_candidates,
        s.sampled_groups ? double(s.sampled_candidates)/double(s.sampled_groups) : 0.0,
        s.max_group_candidates, s.max_lane_candidates, s.fallback_groups);
    std::fprintf(stderr,
        "[match-diag] T%d histogram(sample): 0=%llu 1..128=%llu 129..256=%llu "
        "257..384=%llu 385..512=%llu 513..768=%llu 769..1024=%llu >1024=%llu\n",
        table_id, s.bins[0], s.bins[1], s.bins[2], s.bins[3],
        s.bins[4], s.bins[5], s.bins[6], s.bins[7]);
}

} // namespace pos2gpu
