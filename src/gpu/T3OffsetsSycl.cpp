// T3OffsetsSycl.cpp — SYCL implementation of T3's match kernel. Mirrors
// the CUDA path; FeistelKey (40 B) is captured by value in the parallel_for
// lambda instead of going through CUDA constant memory. AdaptiveCpp's
// SSCP backend handles the capture via the kernel-arg mechanism, which is
// fine at this size — if local-memory spills ever bite, switch to a USM
// upload analogous to the CUDA cudaMemcpyToSymbolAsync path.

#include "gpu/MatchKernelCommon.cuh"
#include "gpu/FineBucketDirect20.hpp"
#include "gpu/MatchDiag.hpp"
#include "gpu/SyclBackend.hpp"
#ifdef XCHPLOT2_NATIVE_CUDA_MATCH
#include "gpu/NativeCudaMatch.hpp"
#endif
#include "gpu/T3Offsets.cuh"
#include "host/PoolSizing.hpp"

#include "gpu/AesDirectAddressSycl.hpp"
#include "gpu/AesLastRoundSycl.hpp"
#include <sycl/sycl.hpp>
#include <chrono>
#include <new>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace pos2gpu {

// Experimental T3 Tezcan AES. 32 equivalent T0 copies are interleaved in
// work-group local memory; distinct local lanes use different bank replicas.
static inline AesState t3c_tezcan_round(
    AesState s, const AesState& key, const uint32_t* t, uint32_t bank)
{
    // Byte-offset address arithmetic: same T0 values and shared layout.
    return aes_direct_addr_t0_round(s, key, t, bank);
}

static inline AesState t3c_tezcan_rounds(
    AesState state, const AesHashKeys& keys, int rounds,
    const uint32_t* t, uint32_t bank)
{
    #pragma unroll 2
    for (int r = 0; r < rounds; ++r) {
        state = t3c_tezcan_round(state, keys.round_key_1, t, bank);
        state = t3c_tezcan_round(state, keys.round_key_2, t, bank);
    }
    return state;
}

static inline uint32_t t3c_tezcan_target(
    const AesHashKeys& keys, uint32_t table_id, uint32_t match_key,
    uint64_t meta, const uint32_t* t, int extra, uint32_t bank)
{
    AesState s = set_int_vec_i128(
        int32_t(meta >> 32), int32_t(meta),
        int32_t(match_key), int32_t(table_id));
    return t3c_tezcan_rounds(
        s, keys, kAesMatchingTargetRounds << extra, t, bank).w[0];
}

static inline Result128 t3c_tezcan_pairing(
    const AesHashKeys& keys, uint64_t l, uint64_t r,
    const uint32_t* t, int extra, uint32_t bank)
{
    AesState s = set_int_vec_i128(
        int32_t(r >> 32), int32_t(r),
        int32_t(l >> 32), int32_t(l));
    s = t3c_tezcan_rounds(s, keys, kAesPairingRounds << extra, t, bank);
    return Result128{{s.w[0], s.w[1], s.w[2], s.w[3]}};
}


// Experiment A: T3 consumes only word 3 for the acceptance test.
static inline uint32_t t3c_tezcan_target_last(
    const AesHashKeys& keys, uint32_t table_id, uint32_t match_key,
    uint64_t meta, const uint32_t* t, int extra, uint32_t bank)
{
    AesState s = set_int_vec_i128(int32_t(meta >> 32), int32_t(meta),
                                  int32_t(match_key), int32_t(table_id));
    const int pairs = kAesMatchingTargetRounds << extra;
    #pragma unroll 2
    for (int i = 0; i < pairs - 1; ++i) {
        s = t3c_tezcan_round(s, keys.round_key_1, t, bank);
        s = t3c_tezcan_round(s, keys.round_key_2, t, bank);
    }
    s = t3c_tezcan_round(s, keys.round_key_1, t, bank);
    return aes_direct_last_word<0>(s, keys.round_key_2, t, bank);
}

static inline Result128 t3c_tezcan_pairing_last(
    const AesHashKeys& keys, uint64_t l, uint64_t r,
    const uint32_t* t, int extra, uint32_t bank)
{
    AesState s = set_int_vec_i128(int32_t(r >> 32), int32_t(r),
                                  int32_t(l >> 32), int32_t(l));
    const int pairs = kAesPairingRounds << extra;
    #pragma unroll 2
    for (int i = 0; i < pairs - 1; ++i) {
        s = t3c_tezcan_round(s, keys.round_key_1, t, bank);
        s = t3c_tezcan_round(s, keys.round_key_2, t, bank);
    }
    s = t3c_tezcan_round(s, keys.round_key_1, t, bank);
    Result128 res{{0u, 0u, 0u, 0u}};
    res.r[3] = aes_direct_last_word<3>(s, keys.round_key_2, t, bank);
    return res;
}


namespace {

// T2 and T3 share one candidate buffer; they are never live at the same
// time. The test accepts one in 2^strength candidates, so shared sizing
// keeps 50% headroom above that candidate:output ratio. At strength two
// the multiplier is 6; it grows with the match-key count to preserve
// per-bucket capacity at higher strengths. See
// sycl_backend::acquire_twophase_scratch in SyclBackend.hpp.
//
// Under-sizing is safe, not fatal: a bucket that overruns cand_cap sets the
// sticky overflow flag, run_t3_match_twophase returns false, and the caller
// rewinds the output counter and redoes the range through the single-kernel
// path, which is correct for any skew. That makes this a pure
// speed/memory dial — the failure mode is "slower", never "wrong".

// ---- Phase 2 prototype: two-phase T3 match ----------------------------
//
// ncu on the single-kernel launch_t3_match_all_buckets showed ~61%
// Speed-of-Light, latency-bound, with only ~12.5 of 32 warp lanes active:
// each thread runs one l-entry through a *variable-length* inner r-loop,
// and the expensive pairing AES (32 rounds) + Feistel sit inside that
// divergent loop. This splits the work in two:
//
//   Phase A — enumerate: one thread per l-entry. Does matching_target
//     AES + binary search + the r-loop, but only EMITS {l, r} candidate
//     index pairs (8 B each) to a scratch buffer. No pairing/Feistel.
//     Still divergent in the r-loop, but the loop body is now a cheap
//     atomic store, not 32 AES rounds.
//   Phase B — process: one thread per candidate pair. Re-gathers
//     meta/xbits, runs the pairing AES + test + Feistel + output write.
//     Fully convergent — every thread does exactly one pairing.
//
// Historical two-phase variant (removed from active dispatch). One bucket at a
// time so the candidate scratch stays small (~1 GB at k=28) and the
// path works regardless of card size. Output order still races on the
// atomic, which is fine — T3 output is sorted by proof_fragment
// afterward.
//
// Returns false when a bucket overflowed the candidate scratch (skewed
// distributions at small k / high strength) — the caller restores the
// output counter and re-runs the range through the single-kernel path,
// which is correct for any skew.


// Two-phase variant of launch_t3_match_section_pair (streaming-tier
// Minimal). Same structure as run_t3_match_twophase but reads meta_l
// from a section-sliced buffer (caller guarantees bucket range fits in
// one section_l). xbits and mi remain full-cap (per the section_pair
// contract). Shares the same acquire_twophase_scratch per-queue cache.
// Returns false on candidate-scratch overflow (see run_t3_match_twophase).




} // namespace

// Diagnostics only: sample the SAME T3 compact workgroups for four stages:
//   AES target, AES+search/scan, +pairing AES, +accepted-pair Feistel.
// No mutation of production match output. Stage differences are GPU kernel
// runtime estimates (not cycle-accurate isolated instruction costs).
// The separate scan is deliberately uncapped; this is the actual sampled
// candidate distribution, not the production path's 513-count saturation.
template<bool WithSearch, bool CollectStats,
         bool WithPairing = false, bool WithFeistel = false>
static void launch_t3_match_diag_probe(
    AesHashKeys keys,
    FeistelKey fk,
    uint64_t const* d_sorted_meta,
    uint32_t const* d_sorted_xbits,
    uint32_t const* d_sorted_mi,
    uint64_t const* d_offsets,
    FineBucketOffset const* d_fine_offsets,
    uint32_t num_match_keys,
    int k,
    int num_section_bits,
    int num_match_target_bits,
    int fine_bits,
    uint32_t target_mask,
    int num_test_bits,
    bool use_tezcan,
    uint64_t l_count_max,
    uint32_t bucket_begin,
    uint32_t bucket_end,
    uint32_t group_stride,
    uint32_t* d_digest,
    MatchDiagStats* d_stats,
    sycl::queue& q)
{
    constexpr size_t threads = 256;
    size_t const n_buckets = size_t(bucket_end - bucket_begin);
    size_t const original_groups = size_t((l_count_max + threads - 1) / threads);
    size_t const sampled_groups = (original_groups + group_stride - 1) / group_stride;
    if (!sampled_groups || !n_buckets) return;
    uint32_t* d_aes_tables = sycl_backend::aes_tables_device(q);
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<sycl::vec<uint32_t, 4>, 1> sT_local{
            sycl::range<1>{1024}, h};
        h.parallel_for(
            sycl::nd_range<2>{sycl::range<2>{n_buckets, sampled_groups * threads},
                              sycl::range<2>{1, threads}},
            [=, keys_copy = keys, fk_copy = fk](sycl::nd_item<2> it) {
                uint32_t* sT = reinterpret_cast<uint32_t*>(&sT_local[0]);
                uint32_t local_id = uint32_t(it.get_local_id(1));
                // V4INIT: 16x replicated T0, 128-bit stores on the Tezcan path.
                if (use_tezcan) {
                    #pragma unroll 1
                    for (uint32_t j = local_id; j < 1024; j += uint32_t(threads)) {
                        sT_local[j] = sycl::vec<uint32_t, 4>(d_aes_tables[j / 4]);
                    }
                } else {
                    #pragma unroll 1
                    for (uint32_t i = local_id; i < 4 * 256; i += uint32_t(threads)) {
                        sT[i] = d_aes_tables[i];
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);

                uint32_t bucket_id = bucket_begin + uint32_t(it.get_group(0));
                uint32_t section_l = bucket_id / num_match_keys;
                uint32_t match_key_r = bucket_id % num_match_keys;
                uint32_t section_r = pos2gpu::matching_section_r(section_l, num_section_bits);
                uint32_t r_bucket = section_r * num_match_keys + match_key_r;
                uint64_t l_start = d_offsets[section_l * num_match_keys];
                uint64_t l_end = d_offsets[(section_l + 1) * num_match_keys];
                uint64_t l = l_start + uint64_t(it.get_group(1)) *
                              uint64_t(group_stride) * uint64_t(threads) + local_id;

                uint32_t target_l = 0;
                uint64_t first = 0;
                uint32_t count = 0;
                uint32_t pair_digest = 0;
                if (l < l_end) {
                    uint64_t meta_l = d_sorted_meta[l];
                    target_l = (use_tezcan
                        ? t3c_tezcan_target(keys_copy, 3u, match_key_r, meta_l,
                                           sT, 0, local_id & 15u)
                        : pos2gpu::matching_target_smem(
                            keys_copy, 3u, match_key_r, meta_l, sT, 0)) & target_mask;
                    if constexpr (WithSearch) {
                        uint32_t shift = uint32_t(num_match_target_bits - fine_bits);
                        uint32_t fine_key = target_l >> shift;
                        uint64_t fine_idx = (uint64_t(r_bucket) << fine_bits) | fine_key;
                        uint64_t fine_hi = d_fine_offsets[fine_idx + 1];
                        first = pos2gpu::fine_bucket_lower_bound(
                            d_sorted_mi, d_fine_offsets[fine_idx], fine_hi,
                            target_l, target_mask);
                        for (uint64_t r = first; r < fine_hi; ++r) {
                            if ((d_sorted_mi[r] & target_mask) != target_l) break;
                            ++count;
                        }
                        if constexpr (WithPairing) {
                            for (uint64_t r = first; r < fine_hi; ++r) {
                                if ((d_sorted_mi[r] & target_mask) != target_l) break;
                                uint64_t meta_r = d_sorted_meta[r];
                                Result128 res = use_tezcan
                                    ? t3c_tezcan_pairing(keys_copy, meta_l, meta_r,
                                                         sT, 0, local_id & 15u)
                                    : pos2gpu::pairing_smem(
                                        keys_copy, meta_l, meta_r, sT, 0);
                                // Consume every pairing result even when no Feistel is used.
                                pair_digest ^= res.r[0] ^ res.r[3];
                                if constexpr (WithFeistel) {
                                    uint32_t test_mask = num_test_bits >= 32
                                        ? 0xFFFFFFFFu : ((1u << num_test_bits) - 1u);
                                    if ((res.r[3] & test_mask) == 0) {
                                        uint32_t xb_l = d_sorted_xbits[l];
                                        uint32_t xb_r = d_sorted_xbits[r];
                                        uint64_t all_x_bits = (uint64_t(xb_l) << k) | uint64_t(xb_r);
                                        uint64_t fragment = pos2gpu::feistel_encrypt(
                                            fk_copy, all_x_bits);
                                        pair_digest ^= uint32_t(fragment) ^ uint32_t(fragment >> 32);
                                    }
                                }
                            }
                        }
                    }
                }
                // Prevent the compiler from deleting AES/search/pairing/Feistel
                // as unused work. One output slot per sampled workgroup.
                uint32_t digest = target_l;
                if constexpr (WithSearch)
                    digest ^= uint32_t(first) ^ (count * 0x9e3779b9u);
                if constexpr (WithPairing)
                    digest ^= pair_digest;
                uint32_t group_digest = sycl::reduce_over_group(
                    it.get_group(), digest, sycl::plus<uint32_t>{});
                if (local_id == 0) {
                    d_digest[size_t(it.get_group(0)) * sampled_groups +
                             size_t(it.get_group(1))] = group_digest;
                }
                if constexpr (CollectStats) {
                    unsigned long long group_total = sycl::reduce_over_group(
                        it.get_group(), static_cast<unsigned long long>(count),
                        sycl::plus<unsigned long long>{});
                    unsigned long long group_max_lane = sycl::reduce_over_group(
                        it.get_group(), static_cast<unsigned long long>(count),
                        sycl::maximum<unsigned long long>{});
                    if (local_id == 0)
                        record_match_diag_group(d_stats, group_total, group_max_lane);
                }
            });
    }).wait();
}

static void launch_t3_match_all_buckets_compact(
    AesHashKeys keys,
    FeistelKey fk,
    uint64_t const* d_sorted_meta,
    uint32_t const* d_sorted_xbits,
    uint32_t const* d_sorted_mi,
    uint64_t const* d_offsets,
    FineBucketOffset const* d_fine_offsets,
    uint32_t num_match_keys,
    uint32_t num_buckets,
    int k,
    int num_section_bits,
    int num_match_target_bits,
    int fine_bits,
    uint32_t target_mask,
    int num_test_bits,
    T3PairingGpu* d_out_pairings,
    uint64_t* d_out_count,
    uint64_t out_capacity,
    uint64_t l_count_max,
    uint32_t bucket_begin,
    uint32_t bucket_end,
    sycl::queue& q)
{
    // Compact matching always uses the full-key Direct20 offsets.
    if (!pos2gpu::fine_direct20_enabled(fine_bits, num_match_target_bits, num_buckets))
        throw std::invalid_argument("T2/T3 Direct20 requires k28/s6 layout");
    FineBucketOffset const* d_direct20 =
        pos2gpu::fine_direct20_ptr(d_fine_offsets, num_buckets);

    (void)num_buckets;  // only the [begin, end) sub-range is iterated
    if (bucket_end <= bucket_begin) return;
    uint32_t const num_buckets_in_range = bucket_end - bucket_begin;

    uint32_t* d_aes_tables = sycl_backend::aes_tables_device(q);

    constexpr size_t threads = 256;
    uint64_t blocks_x_u64    = (l_count_max + threads - 1) / threads;
    size_t   const blocks_x  = static_cast<size_t>(blocks_x_u64);

    constexpr bool use_tezcan = true;

    // Experiment A: independent of the Direct24/Direct20 index selection.
    // Read on host and capture a uniform flag; OFF leaves original behavior.
    constexpr bool use_last_round_partial = true;

    auto* d_out_count_ull =
        reinterpret_cast<unsigned long long*>(d_out_count);

    MatchDiagOptions const diag = MatchDiagOptions::read();
    MatchDiagStats* d_match_diag_stats = nullptr;
    if (diag.enabled) {
        d_match_diag_stats = sycl::malloc_device<MatchDiagStats>(1, q);
        if (!d_match_diag_stats) throw std::bad_alloc();
        q.memset(d_match_diag_stats, 0, sizeof(MatchDiagStats)).wait();
    }
    auto const production_start = std::chrono::steady_clock::now();

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<sycl::vec<uint32_t, 4>, 1> sT_local{
            sycl::range<1>{1024}, h};
        // At most 300 candidate pairs. Local memory: 32 KiB AES +
        // ~4.7 KiB pair metadata + ~2.3 KiB xbits + a counter.
        constexpr uint32_t pair_cap = 330;
        sycl::local_accessor<uint64_t, 1> pairs_l{sycl::range<1>{pair_cap}, h};
        sycl::local_accessor<uint64_t, 1> pairs_r{sycl::range<1>{pair_cap}, h};
        sycl::local_accessor<uint32_t, 1> pairs_xl{sycl::range<1>{pair_cap}, h};
        sycl::local_accessor<uint32_t, 1> pairs_xr{sycl::range<1>{pair_cap}, h};
        sycl::local_accessor<uint32_t, 1> total_local{sycl::range<1>{1}, h};

        h.parallel_for(
            sycl::nd_range<2>{
                sycl::range<2>{ static_cast<size_t>(num_buckets_in_range),
                                blocks_x * threads },
                sycl::range<2>{ 1, threads }
            },
            [=, keys_copy = keys, fk_copy = fk](sycl::nd_item<2> it) {
                uint32_t* sT = reinterpret_cast<uint32_t*>(&sT_local[0]);
                size_t local_id = it.get_local_id(1);
                // V4INIT: 16x replicated T0, 128-bit stores on the Tezcan path.
                if (use_tezcan) {
                    #pragma unroll 1
                    for (size_t j = local_id; j < 1024; j += threads) {
                        sT_local[j] = sycl::vec<uint32_t, 4>(d_aes_tables[j / 4]);
                    }
                } else {
                    #pragma unroll 1
                    for (size_t i = local_id; i < 4 * 256; i += threads) {
                        sT[i] = d_aes_tables[i];
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);

                uint32_t bucket_id   = bucket_begin + static_cast<uint32_t>(it.get_group(0));
                uint32_t section_l   = bucket_id / num_match_keys;
                uint32_t match_key_r = bucket_id % num_match_keys;

                uint32_t section_r = pos2gpu::matching_section_r(section_l, num_section_bits);

                uint64_t l_start = d_offsets[section_l * num_match_keys];
                uint64_t l_end   = d_offsets[(section_l + 1) * num_match_keys];
                uint32_t r_bucket = section_r * num_match_keys + match_key_r;

                uint64_t l = l_start
                           + it.get_group(1) * uint64_t(threads)
                           + local_id;
                uint64_t meta_l = 0, first = 0, fine_hi = 0;
                uint32_t xb_l = 0, target_l = 0, count = 0;
                if (l < l_end) {
                    meta_l = d_sorted_meta[l];
                    xb_l = d_sorted_xbits[l];
                    target_l = (use_tezcan
                        ? (use_last_round_partial ? t3c_tezcan_target_last(keys_copy, 3u, match_key_r, meta_l, sT, 0, static_cast<uint32_t>(local_id) & 15u) : t3c_tezcan_target(keys_copy, 3u, match_key_r, meta_l, sT, 0, static_cast<uint32_t>(local_id) & 15u))
                        : pos2gpu::matching_target_smem(
                            keys_copy, 3u, match_key_r, meta_l, sT, 0)) & target_mask;
                    uint64_t const exact_key =
                        (uint64_t(r_bucket) << 20) | target_l;
                    first = d_direct20[exact_key];
                    fine_hi = d_direct20[exact_key + 1];
                    for (uint64_t r = first; r < fine_hi; ++r) {
                        if (++count > pair_cap) break;
                    }
                }

                uint32_t const test_mask = (num_test_bits >= 32)
                    ? 0xFFFFFFFFu : ((1u << num_test_bits) - 1u);
                auto emit = [&](uint64_t ml, uint64_t mr, uint32_t xl, uint32_t xr) {
                    pos2gpu::Result128 res = use_tezcan
                        ? (use_last_round_partial ? t3c_tezcan_pairing_last(keys_copy, ml, mr, sT, 0, static_cast<uint32_t>(local_id) & 15u) : t3c_tezcan_pairing(keys_copy, ml, mr, sT, 0, static_cast<uint32_t>(local_id) & 15u))
                        : pos2gpu::pairing_smem(keys_copy, ml, mr, sT, 0);
                    if ((res.r[3] & test_mask) != 0) return;
                    uint64_t all_x_bits = (uint64_t(xl) << k) | uint64_t(xr);
                    uint64_t fragment = pos2gpu::feistel_encrypt(fk_copy, all_x_bits);
                    sycl::atomic_ref<unsigned long long,
                                     sycl::memory_order::relaxed,
                                     sycl::memory_scope::device>
                        out_count_atomic{*d_out_count_ull};
                    unsigned long long out_idx = out_count_atomic.fetch_add(1ULL);
                    if (out_idx >= out_capacity) return;
                    T3PairingGpu p;
                    p.proof_fragment = fragment;
                    d_out_pairings[out_idx] = p;
                };

                uint32_t offset = sycl::exclusive_scan_over_group(
                    it.get_group(), count, sycl::plus<uint32_t>{});
                if (local_id == threads - 1) total_local[0] = offset + count;
                it.barrier(sycl::access::fence_space::local_space);
                uint32_t n = total_local[0];
                if (n > pair_cap) {
                    if (d_match_diag_stats && local_id == 0) {
                        using Atomic = sycl::atomic_ref<unsigned long long,
                            sycl::memory_order::relaxed, sycl::memory_scope::device>;
                        Atomic{d_match_diag_stats->fallback_groups}.fetch_add(1ULL);
                    }
                    // Uniform group-wide fallback: reprocess ALL matches.
                    // Count may be truncated at pair_cap+1 during enumeration.
                    if (l < l_end) {
                        for (uint64_t r = first; r < fine_hi; ++r) {
                            emit(meta_l, d_sorted_meta[r], xb_l, d_sorted_xbits[r]);
                        }
                    }
                    return;
                }
                for (uint32_t j = 0; j < count; ++j) {
                    uint32_t ci = offset + j;
                    pairs_l[ci] = meta_l;
                    pairs_r[ci] = d_sorted_meta[first + j];
                    pairs_xl[ci] = xb_l;
                    pairs_xr[ci] = d_sorted_xbits[first + j];
                }
                it.barrier(sycl::access::fence_space::local_space);
                for (uint32_t ci = static_cast<uint32_t>(local_id);
                     ci < n; ci += static_cast<uint32_t>(threads)) {
                    emit(pairs_l[ci], pairs_r[ci], pairs_xl[ci], pairs_xr[ci]);
                }
            });
    }).wait();
    auto const production_end = std::chrono::steady_clock::now();
    if (diag.enabled) {
        using Clock = std::chrono::steady_clock;
        MatchDiagStats host_stats{};
        uint32_t first_digest = 0;
        size_t const sampled_groups =
            ((size_t(l_count_max + 255) / 256) + diag.stride - 1) / diag.stride;
        size_t const total_sampled_groups =
            size_t(bucket_end - bucket_begin) * sampled_groups;
        auto* d_digest = sycl::malloc_device<uint32_t>(total_sampled_groups, q);
        if (!d_digest) {
            sycl::free(d_match_diag_stats, q);
            throw std::bad_alloc();
        }
        auto run_stage = [&](int stage) {
            if (stage == 0)
                launch_t3_match_diag_probe<false, false>(
                    keys, fk, d_sorted_meta, d_sorted_xbits, d_sorted_mi,
                    d_offsets, d_fine_offsets, num_match_keys, k,
                    num_section_bits, num_match_target_bits, fine_bits,
                    target_mask, num_test_bits, use_tezcan, l_count_max,
                    bucket_begin, bucket_end, diag.stride, d_digest,
                    d_match_diag_stats, q);
            else if (stage == 1)
                launch_t3_match_diag_probe<true, false>(
                    keys, fk, d_sorted_meta, d_sorted_xbits, d_sorted_mi,
                    d_offsets, d_fine_offsets, num_match_keys, k,
                    num_section_bits, num_match_target_bits, fine_bits,
                    target_mask, num_test_bits, use_tezcan, l_count_max,
                    bucket_begin, bucket_end, diag.stride, d_digest,
                    d_match_diag_stats, q);
            else if (stage == 2)
                launch_t3_match_diag_probe<true, false, true>(
                    keys, fk, d_sorted_meta, d_sorted_xbits, d_sorted_mi,
                    d_offsets, d_fine_offsets, num_match_keys, k,
                    num_section_bits, num_match_target_bits, fine_bits,
                    target_mask, num_test_bits, use_tezcan, l_count_max,
                    bucket_begin, bucket_end, diag.stride, d_digest,
                    d_match_diag_stats, q);
            else
                launch_t3_match_diag_probe<true, false, true, true>(
                    keys, fk, d_sorted_meta, d_sorted_xbits, d_sorted_mi,
                    d_offsets, d_fine_offsets, num_match_keys, k,
                    num_section_bits, num_match_target_bits, fine_bits,
                    target_mask, num_test_bits, use_tezcan, l_count_max,
                    bucket_begin, bucket_end, diag.stride, d_digest,
                    d_match_diag_stats, q);
        };
        double stage_ms[4] = {0, 0, 0, 0};
        // JIT warmup of all variants, then alternating traversal to reduce
        // order effects. Timers include kernel-launch overhead, consistently.
        for (int stage = 0; stage < 4; ++stage) run_stage(stage);
        for (int repeat = 0; repeat < 2; ++repeat) {
            for (int step = 0; step < 4; ++step) {
                int stage = repeat == 0 ? step : 3 - step;
                auto begin = Clock::now();
                run_stage(stage);
                auto end = Clock::now();
                stage_ms[stage] +=
                    std::chrono::duration<double, std::milli>(end - begin).count() / 2.0;
            }
        }
        // Full, uncapped candidate histogram on sampled workgroups.
        launch_t3_match_diag_probe<true, true>(
            keys, fk, d_sorted_meta, d_sorted_xbits, d_sorted_mi,
            d_offsets, d_fine_offsets, num_match_keys, k,
            num_section_bits, num_match_target_bits, fine_bits,
            target_mask, num_test_bits, use_tezcan, l_count_max,
            bucket_begin, bucket_end, diag.stride, d_digest,
            d_match_diag_stats, q);
        q.memcpy(&first_digest, d_digest, sizeof(first_digest)).wait();
        q.memcpy(&host_stats, d_match_diag_stats, sizeof(host_stats)).wait();
        double fused_ms = std::chrono::duration<double, std::milli>(
            production_end - production_start).count();
        print_match_diag_stats(3, diag.stride, stage_ms[0], stage_ms[1],
                               stage_ms[2], fused_ms, host_stats, first_digest);
        std::fprintf(stderr,
            "[match-diag] T3 aes_search_pair_feistel=%.3fms "
            "feistel_delta=%.3fms (accepted candidates only)\n",
            stage_ms[3], stage_ms[3] - stage_ms[2]);
        sycl::free(d_digest, q);
        sycl::free(d_match_diag_stats, q);
    }
}

void launch_t3_match_all_buckets(
    AesHashKeys keys,
    FeistelKey fk,
    uint64_t const* d_sorted_meta,
    uint32_t const* d_sorted_xbits,
    uint32_t const* d_sorted_mi,
    uint64_t const* d_offsets,
    FineBucketOffset const* d_fine_offsets,
    uint32_t num_match_keys,
    uint32_t num_buckets,
    int k,
    int num_section_bits,
    int num_match_target_bits,
    int fine_bits,
    uint32_t target_mask,
    int num_test_bits,
    T3PairingGpu* d_out_pairings,
    uint64_t* d_out_count,
    uint64_t out_capacity,
    uint64_t l_count_max,
    uint32_t bucket_begin,
    uint32_t bucket_end,
    sycl::queue& q)
{
#if defined(XCHPLOT2_NATIVE_CUDA_MATCH)
    if (q.get_device().get_backend() == sycl::backend::cuda) {
        // Keep nvcc away from SYCL headers, exactly like SortSyclCub.
        // The pointer comes from the existing 4-table SYCL upload; T0 is its
        // first 256 uint32 values.  Wait before launching on CUDA stream 0.
        uint32_t const* d_t0 = sycl_backend::aes_tables_device(q);
        q.wait();
        if (!fine_direct20_enabled(fine_bits, num_match_target_bits, num_buckets))
            throw std::invalid_argument("native CUDA T3 requires k28/s6 Direct20");
        auto const* direct = fine_direct20_ptr(d_fine_offsets, num_buckets);
        cuda_match_t3(keys, fk, d_sorted_meta, d_sorted_xbits, d_offsets,
            direct, d_t0, num_match_keys, k, num_section_bits, target_mask,
            num_test_bits, reinterpret_cast<uint64_t*>(d_out_pairings),
            d_out_count, out_capacity, l_count_max, bucket_begin, bucket_end);
        return;
    }
#endif
    launch_t3_match_all_buckets_compact(
            keys, fk, d_sorted_meta, d_sorted_xbits, d_sorted_mi,
            d_offsets, d_fine_offsets, num_match_keys, num_buckets, k,
            num_section_bits, num_match_target_bits, fine_bits, target_mask,
            num_test_bits, d_out_pairings, d_out_count, out_capacity,
            l_count_max, bucket_begin, bucket_end, q);
}

static void launch_t3_match_section_pair_compact(
    AesHashKeys keys,
    FeistelKey fk,
    uint64_t const* d_meta_l_slice,
    uint64_t section_l_row_start,
    uint64_t const* d_meta_r_slice,
    uint64_t section_r_row_start,
    uint32_t const* d_sorted_xbits,
    uint32_t const* d_sorted_mi,
    uint64_t const* d_offsets,
    FineBucketOffset const* d_fine_offsets,
    uint32_t num_match_keys,
    uint32_t num_buckets,
    int k,
    int num_section_bits,
    int num_match_target_bits,
    int fine_bits,
    uint32_t target_mask,
    int num_test_bits,
    T3PairingGpu* d_out_pairings,
    uint64_t* d_out_count,
    uint64_t out_capacity,
    uint64_t l_count_max,
    uint32_t bucket_begin,
    uint32_t bucket_end,
    sycl::queue& q)
{
    (void)num_buckets;
    if (bucket_end <= bucket_begin) return;
    uint32_t const num_buckets_in_range = bucket_end - bucket_begin;

    uint32_t* d_aes_tables = sycl_backend::aes_tables_device(q);

    constexpr size_t threads = 256;
    uint64_t blocks_x_u64    = (l_count_max + threads - 1) / threads;
    size_t   const blocks_x  = static_cast<size_t>(blocks_x_u64);

    constexpr bool use_tezcan = true;

    auto* d_out_count_ull =
        reinterpret_cast<unsigned long long*>(d_out_count);

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<sycl::vec<uint32_t, 4>, 1> sT_local{
            sycl::range<1>{1024}, h};
        // At most 300 candidate pairs. Local memory: 32 KiB AES +
        // ~4.7 KiB pair metadata + ~2.3 KiB xbits + a counter.
        constexpr uint32_t pair_cap = 330;
        sycl::local_accessor<uint64_t, 1> pairs_l{sycl::range<1>{pair_cap}, h};
        sycl::local_accessor<uint64_t, 1> pairs_r{sycl::range<1>{pair_cap}, h};
        sycl::local_accessor<uint32_t, 1> pairs_xl{sycl::range<1>{pair_cap}, h};
        sycl::local_accessor<uint32_t, 1> pairs_xr{sycl::range<1>{pair_cap}, h};
        sycl::local_accessor<uint32_t, 1> total_local{sycl::range<1>{1}, h};

        h.parallel_for(
            sycl::nd_range<2>{
                sycl::range<2>{ static_cast<size_t>(num_buckets_in_range),
                                blocks_x * threads },
                sycl::range<2>{ 1, threads }
            },
            [=, keys_copy = keys, fk_copy = fk](sycl::nd_item<2> it) {
                uint32_t* sT = reinterpret_cast<uint32_t*>(&sT_local[0]);
                size_t local_id = it.get_local_id(1);
                // V4INIT: 16x replicated T0, 128-bit stores on the Tezcan path.
                if (use_tezcan) {
                    #pragma unroll 1
                    for (size_t j = local_id; j < 1024; j += threads) {
                        sT_local[j] = sycl::vec<uint32_t, 4>(d_aes_tables[j / 4]);
                    }
                } else {
                    #pragma unroll 1
                    for (size_t i = local_id; i < 4 * 256; i += threads) {
                        sT[i] = d_aes_tables[i];
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);

                uint32_t bucket_id   = bucket_begin + static_cast<uint32_t>(it.get_group(0));
                uint32_t section_l   = bucket_id / num_match_keys;
                uint32_t match_key_r = bucket_id % num_match_keys;

                uint32_t section_r = pos2gpu::matching_section_r(section_l, num_section_bits);

                uint64_t l_start = d_offsets[section_l * num_match_keys];
                uint64_t l_end   = d_offsets[(section_l + 1) * num_match_keys];
                uint32_t r_bucket = section_r * num_match_keys + match_key_r;

                uint64_t l = l_start
                           + it.get_group(1) * uint64_t(threads)
                           + local_id;
                uint64_t meta_l = 0, first = 0, fine_hi = 0;
                uint32_t xb_l = 0, target_l = 0, count = 0;
                if (l < l_end) {
                    meta_l = d_meta_l_slice[l - section_l_row_start];
                    xb_l = d_sorted_xbits[l];
                    target_l = (use_tezcan
                        ? t3c_tezcan_target(keys_copy, 3u, match_key_r, meta_l,
                                           sT, 0, static_cast<uint32_t>(local_id) & 15u)
                        : pos2gpu::matching_target_smem(
                            keys_copy, 3u, match_key_r, meta_l, sT, 0)) & target_mask;
                    uint32_t fine_shift = static_cast<uint32_t>(num_match_target_bits - fine_bits);
                    uint32_t fine_key = target_l >> fine_shift;
                    uint64_t fine_idx = (uint64_t(r_bucket) << fine_bits) | fine_key;
                    fine_hi = d_fine_offsets[fine_idx + 1];
                    first = pos2gpu::fine_bucket_lower_bound(
                        d_sorted_mi, d_fine_offsets[fine_idx], fine_hi,
                        target_l, target_mask);
                    for (uint64_t r = first; r < fine_hi; ++r) {
                        if ((d_sorted_mi[r] & target_mask) != target_l) break;
                        if (++count > pair_cap) break;
                    }
                }

                uint32_t const test_mask = (num_test_bits >= 32)
                    ? 0xFFFFFFFFu : ((1u << num_test_bits) - 1u);
                auto emit = [&](uint64_t ml, uint64_t mr, uint32_t xl, uint32_t xr) {
                    pos2gpu::Result128 res = use_tezcan
                        ? t3c_tezcan_pairing(keys_copy, ml, mr, sT, 0,
                                             static_cast<uint32_t>(local_id) & 15u)
                        : pos2gpu::pairing_smem(keys_copy, ml, mr, sT, 0);
                    if ((res.r[3] & test_mask) != 0) return;
                    uint64_t all_x_bits = (uint64_t(xl) << k) | uint64_t(xr);
                    uint64_t fragment = pos2gpu::feistel_encrypt(fk_copy, all_x_bits);
                    sycl::atomic_ref<unsigned long long,
                                     sycl::memory_order::relaxed,
                                     sycl::memory_scope::device>
                        out_count_atomic{*d_out_count_ull};
                    unsigned long long out_idx = out_count_atomic.fetch_add(1ULL);
                    if (out_idx >= out_capacity) return;
                    T3PairingGpu p;
                    p.proof_fragment = fragment;
                    d_out_pairings[out_idx] = p;
                };

                uint32_t offset = sycl::exclusive_scan_over_group(
                    it.get_group(), count, sycl::plus<uint32_t>{});
                if (local_id == threads - 1) total_local[0] = offset + count;
                it.barrier(sycl::access::fence_space::local_space);
                uint32_t n = total_local[0];
                if (n > pair_cap) {
                    // Uniform group-wide fallback: reprocess ALL matches.
                    // Count may be truncated at pair_cap+1 during enumeration.
                    if (l < l_end) {
                        for (uint64_t r = first; r < fine_hi; ++r) {
                            if ((d_sorted_mi[r] & target_mask) != target_l) break;
                            emit(meta_l, d_meta_r_slice[r - section_r_row_start], xb_l, d_sorted_xbits[r]);
                        }
                    }
                    return;
                }
                for (uint32_t j = 0; j < count; ++j) {
                    uint32_t ci = offset + j;
                    pairs_l[ci] = meta_l;
                    pairs_r[ci] = d_meta_r_slice[first + j - section_r_row_start];
                    pairs_xl[ci] = xb_l;
                    pairs_xr[ci] = d_sorted_xbits[first + j];
                }
                it.barrier(sycl::access::fence_space::local_space);
                for (uint32_t ci = static_cast<uint32_t>(local_id);
                     ci < n; ci += static_cast<uint32_t>(threads)) {
                    emit(pairs_l[ci], pairs_r[ci], pairs_xl[ci], pairs_xr[ci]);
                }
            });
    }).wait();
}

void launch_t3_match_section_pair(
    AesHashKeys keys,
    FeistelKey fk,
    uint64_t const* d_meta_l_slice,
    uint64_t section_l_row_start,
    uint64_t const* d_meta_r_slice,
    uint64_t section_r_row_start,
    uint32_t const* d_sorted_xbits,
    uint32_t const* d_sorted_mi,
    uint64_t const* d_offsets,
    FineBucketOffset const* d_fine_offsets,
    uint32_t num_match_keys,
    uint32_t num_buckets,
    int k,
    int num_section_bits,
    int num_match_target_bits,
    int fine_bits,
    uint32_t target_mask,
    int num_test_bits,
    T3PairingGpu* d_out_pairings,
    uint64_t* d_out_count,
    uint64_t out_capacity,
    uint64_t l_count_max,
    uint32_t bucket_begin,
    uint32_t bucket_end,
    sycl::queue& q)
{
    launch_t3_match_section_pair_compact(
            keys, fk, d_meta_l_slice, section_l_row_start,
            d_meta_r_slice, section_r_row_start, d_sorted_xbits, d_sorted_mi,
            d_offsets, d_fine_offsets, num_match_keys, num_buckets, k,
            num_section_bits, num_match_target_bits, fine_bits, target_mask,
            num_test_bits, d_out_pairings, d_out_count, out_capacity,
            l_count_max, bucket_begin, bucket_end, q);
}

static void launch_t3_match_section_pair_split_compact(
    AesHashKeys keys,
    FeistelKey fk,
    uint64_t const* d_meta_l_slice,
    uint32_t const* d_xbits_l_slice,
    uint64_t section_l_row_start,
    uint64_t const* d_meta_r_slice,
    uint32_t const* d_xbits_r_slice,
    uint32_t const* d_mi_r_slice,
    uint64_t section_r_row_start,
    uint64_t const* d_offsets,
    FineBucketOffset const* d_fine_offsets,
    uint32_t num_match_keys,
    uint32_t num_buckets,
    int k,
    int num_section_bits,
    int num_match_target_bits,
    int fine_bits,
    uint32_t target_mask,
    int num_test_bits,
    T3PairingGpu* d_out_pairings,
    uint64_t* d_out_count,
    uint64_t out_capacity,
    uint64_t l_count_max,
    uint32_t bucket_begin,
    uint32_t bucket_end,
    sycl::queue& q)
{
    (void)num_buckets;
    if (bucket_end <= bucket_begin) return;
    uint32_t const num_buckets_in_range = bucket_end - bucket_begin;

    uint32_t* d_aes_tables = sycl_backend::aes_tables_device(q);

    constexpr size_t threads = 256;
    uint64_t blocks_x_u64    = (l_count_max + threads - 1) / threads;
    size_t   const blocks_x  = static_cast<size_t>(blocks_x_u64);

    constexpr bool use_tezcan = true;

    auto* d_out_count_ull =
        reinterpret_cast<unsigned long long*>(d_out_count);

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<sycl::vec<uint32_t, 4>, 1> sT_local{
            sycl::range<1>{1024}, h};
        // At most 300 candidate pairs. Local memory: 32 KiB AES +
        // ~4.7 KiB pair metadata + ~2.3 KiB xbits + a counter.
        constexpr uint32_t pair_cap = 330;
        sycl::local_accessor<uint64_t, 1> pairs_l{sycl::range<1>{pair_cap}, h};
        sycl::local_accessor<uint64_t, 1> pairs_r{sycl::range<1>{pair_cap}, h};
        sycl::local_accessor<uint32_t, 1> pairs_xl{sycl::range<1>{pair_cap}, h};
        sycl::local_accessor<uint32_t, 1> pairs_xr{sycl::range<1>{pair_cap}, h};
        sycl::local_accessor<uint32_t, 1> total_local{sycl::range<1>{1}, h};

        h.parallel_for(
            sycl::nd_range<2>{
                sycl::range<2>{ static_cast<size_t>(num_buckets_in_range),
                                blocks_x * threads },
                sycl::range<2>{ 1, threads }
            },
            [=, keys_copy = keys, fk_copy = fk](sycl::nd_item<2> it) {
                uint32_t* sT = reinterpret_cast<uint32_t*>(&sT_local[0]);
                size_t local_id = it.get_local_id(1);
                // V4INIT: 16x replicated T0, 128-bit stores on the Tezcan path.
                if (use_tezcan) {
                    #pragma unroll 1
                    for (size_t j = local_id; j < 1024; j += threads) {
                        sT_local[j] = sycl::vec<uint32_t, 4>(d_aes_tables[j / 4]);
                    }
                } else {
                    #pragma unroll 1
                    for (size_t i = local_id; i < 4 * 256; i += threads) {
                        sT[i] = d_aes_tables[i];
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);

                uint32_t bucket_id   = bucket_begin + static_cast<uint32_t>(it.get_group(0));
                uint32_t section_l   = bucket_id / num_match_keys;
                uint32_t match_key_r = bucket_id % num_match_keys;

                uint32_t section_r = pos2gpu::matching_section_r(section_l, num_section_bits);

                uint64_t l_start = d_offsets[section_l * num_match_keys];
                uint64_t l_end   = d_offsets[(section_l + 1) * num_match_keys];
                uint32_t r_bucket = section_r * num_match_keys + match_key_r;

                uint64_t l = l_start
                           + it.get_group(1) * uint64_t(threads)
                           + local_id;
                uint64_t meta_l = 0, first = 0, fine_hi = 0;
                uint32_t xb_l = 0, target_l = 0, count = 0;
                if (l < l_end) {
                    uint64_t l_off = l - section_l_row_start;
                    meta_l = d_meta_l_slice[l_off];
                    xb_l = d_xbits_l_slice[l_off];
                    target_l = (use_tezcan
                        ? t3c_tezcan_target(keys_copy, 3u, match_key_r, meta_l,
                                           sT, 0, static_cast<uint32_t>(local_id) & 15u)
                        : pos2gpu::matching_target_smem(
                            keys_copy, 3u, match_key_r, meta_l, sT, 0)) & target_mask;
                    uint32_t fine_shift = static_cast<uint32_t>(num_match_target_bits - fine_bits);
                    uint32_t fine_key = target_l >> fine_shift;
                    uint64_t fine_idx = (uint64_t(r_bucket) << fine_bits) | fine_key;
                    uint64_t lo = d_fine_offsets[fine_idx];
                    fine_hi = d_fine_offsets[fine_idx + 1];
                    uint64_t hi = fine_hi;
                    while (lo < hi) {
                        uint64_t mid = lo + ((hi - lo) >> 1);
                        uint32_t t = d_mi_r_slice[mid - section_r_row_start] & target_mask;
                        if (t < target_l) lo = mid + 1;
                        else hi = mid;
                    }
                    first = lo;
                    for (uint64_t r = first; r < fine_hi; ++r) {
                        if ((d_mi_r_slice[r - section_r_row_start] & target_mask) != target_l) break;
                        if (++count > pair_cap) break;
                    }
                }

                uint32_t const test_mask = (num_test_bits >= 32)
                    ? 0xFFFFFFFFu : ((1u << num_test_bits) - 1u);
                auto emit = [&](uint64_t ml, uint64_t mr, uint32_t xl, uint32_t xr) {
                    pos2gpu::Result128 res = use_tezcan
                        ? t3c_tezcan_pairing(keys_copy, ml, mr, sT, 0,
                                             static_cast<uint32_t>(local_id) & 15u)
                        : pos2gpu::pairing_smem(keys_copy, ml, mr, sT, 0);
                    if ((res.r[3] & test_mask) != 0) return;
                    uint64_t all_x_bits = (uint64_t(xl) << k) | uint64_t(xr);
                    uint64_t fragment = pos2gpu::feistel_encrypt(fk_copy, all_x_bits);
                    sycl::atomic_ref<unsigned long long,
                                     sycl::memory_order::relaxed,
                                     sycl::memory_scope::device>
                        out_count_atomic{*d_out_count_ull};
                    unsigned long long out_idx = out_count_atomic.fetch_add(1ULL);
                    if (out_idx >= out_capacity) return;
                    T3PairingGpu p;
                    p.proof_fragment = fragment;
                    d_out_pairings[out_idx] = p;
                };

                uint32_t offset = sycl::exclusive_scan_over_group(
                    it.get_group(), count, sycl::plus<uint32_t>{});
                if (local_id == threads - 1) total_local[0] = offset + count;
                it.barrier(sycl::access::fence_space::local_space);
                uint32_t n = total_local[0];
                if (n > pair_cap) {
                    // Uniform group-wide fallback: reprocess ALL matches.
                    // Count may be truncated at pair_cap+1 during enumeration.
                    if (l < l_end) {
                        for (uint64_t r = first; r < fine_hi; ++r) {
                            if ((d_mi_r_slice[r - section_r_row_start] & target_mask) != target_l) break;
                            emit(meta_l, d_meta_r_slice[r - section_r_row_start], xb_l, d_xbits_r_slice[r - section_r_row_start]);
                        }
                    }
                    return;
                }
                for (uint32_t j = 0; j < count; ++j) {
                    uint32_t ci = offset + j;
                    pairs_l[ci] = meta_l;
                    pairs_r[ci] = d_meta_r_slice[first + j - section_r_row_start];
                    pairs_xl[ci] = xb_l;
                    pairs_xr[ci] = d_xbits_r_slice[first + j - section_r_row_start];
                }
                it.barrier(sycl::access::fence_space::local_space);
                for (uint32_t ci = static_cast<uint32_t>(local_id);
                     ci < n; ci += static_cast<uint32_t>(threads)) {
                    emit(pairs_l[ci], pairs_r[ci], pairs_xl[ci], pairs_xr[ci]);
                }
            });
    }).wait();
}

void launch_t3_match_section_pair_split(
    AesHashKeys keys,
    FeistelKey fk,
    uint64_t const* d_meta_l_slice,
    uint32_t const* d_xbits_l_slice,
    uint64_t section_l_row_start,
    uint64_t const* d_meta_r_slice,
    uint32_t const* d_xbits_r_slice,
    uint32_t const* d_mi_r_slice,
    uint64_t section_r_row_start,
    uint64_t const* d_offsets,
    FineBucketOffset const* d_fine_offsets,
    uint32_t num_match_keys,
    uint32_t num_buckets,
    int k,
    int num_section_bits,
    int num_match_target_bits,
    int fine_bits,
    uint32_t target_mask,
    int num_test_bits,
    T3PairingGpu* d_out_pairings,
    uint64_t* d_out_count,
    uint64_t out_capacity,
    uint64_t l_count_max,
    uint32_t bucket_begin,
    uint32_t bucket_end,
    sycl::queue& q)
{
    launch_t3_match_section_pair_split_compact(
            keys, fk, d_meta_l_slice, d_xbits_l_slice, section_l_row_start,
            d_meta_r_slice, d_xbits_r_slice, d_mi_r_slice, section_r_row_start,
            d_offsets, d_fine_offsets, num_match_keys, num_buckets, k,
            num_section_bits, num_match_target_bits, fine_bits, target_mask,
            num_test_bits, d_out_pairings, d_out_count, out_capacity,
            l_count_max, bucket_begin, bucket_end, q);
}

} // namespace pos2gpu
