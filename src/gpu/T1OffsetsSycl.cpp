// T1OffsetsSycl.cpp — SYCL/AdaptiveCpp implementation of
// launch_compute_bucket_offsets, selected when XCHPLOT2_BACKEND=sycl.
//
// Same algorithm and output layout as T1OffsetsCuda.cu. The SYCL queue
// uses AdaptiveCpp's CUDA backend (gpu_selector picks the RTX 4090 in
// our test bench), which uses libcuda directly and shares the primary
// CUDA context with the rest of the pipeline — so raw CUDA device
// pointers from cudaMalloc are valid USM device pointers in the SYCL
// kernel without any copy or remap.
//
// Synchronisation: the function syncs `stream` before launching SYCL
// (so prior CUDA writes to d_sorted are visible) and waits for the
// SYCL queue after (so subsequent CUDA reads of d_offsets see the
// SYCL writes). Two extra host syncs vs. the pure-CUDA path; not
// perf-relevant for slice 2.

#include "gpu/MatchKernelCommon.cuh"
#include "gpu/MatchDiag.hpp"
#include "gpu/SyclBackend.hpp"
#include "gpu/T1Offsets.cuh"
#include "gpu/T1Direct24.hpp"
#include "gpu/AesLastRoundSycl.hpp"

#include <sycl/sycl.hpp>
#include <chrono>
#include <new>
#include <cstdlib>
#include <stdexcept>

namespace pos2gpu {

static inline AesState tezcan_round(
    AesState s, const AesState& key,
    const uint32_t* t)
{
    AesState out;
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        uint32_t a = t[(s.w[c] & 255u) * 16];
        uint32_t b = t[((s.w[(c+1)&3] >> 8) & 255u) * 16];
        uint32_t d = t[((s.w[(c+2)&3] >> 16) & 255u) * 16];
        uint32_t e = t[(s.w[(c+3)&3] >> 24) * 16];

        b = (b << 8) | (b >> 24);
        d = (d << 16) | (d >> 16);
        e = (e >> 8) | (e << 24);

        out.w[c] = a ^ b ^ d ^ e ^ key.w[c];
    }
    return out;
}

static inline AesState tezcan_rounds(
    AesState state, const AesHashKeys& keys,
    int rounds, const uint32_t* t)
{
    #pragma unroll 2
    for (int r = 0; r < rounds; ++r) {
        state = tezcan_round(state, keys.round_key_1, t);
        state = tezcan_round(state, keys.round_key_2, t);
    }
    return state;
}

static inline uint32_t tezcan_target(
    const AesHashKeys& keys,
    uint32_t table_id, uint32_t match_key,
    uint64_t meta, const uint32_t* t, int extra)
{
    AesState s = set_int_vec_i128(
        int32_t(meta >> 32), int32_t(meta),
        int32_t(match_key), int32_t(table_id));
    return tezcan_rounds(
        s, keys, kAesMatchingTargetRounds << extra, t).w[0];
}

static inline Result128 tezcan_pairing(
    const AesHashKeys& keys,
    uint64_t l, uint64_t r,
    const uint32_t* t, int extra)
{
    AesState s = set_int_vec_i128(
        int32_t(r >> 32), int32_t(r),
        int32_t(l >> 32), int32_t(l));
    s = tezcan_rounds(
        s, keys, kAesPairingRounds << extra, t);
    return Result128{{s.w[0], s.w[1], s.w[2], s.w[3]}};
}


// Experiment A: only the final AESENC round has fewer required words.
static inline uint32_t tezcan_target_last(
    const AesHashKeys& keys, uint32_t table_id, uint32_t match_key,
    uint64_t meta, const uint32_t* t, int extra)
{
    AesState s = set_int_vec_i128(int32_t(meta >> 32), int32_t(meta),
                                  int32_t(match_key), int32_t(table_id));
    const int pairs = kAesMatchingTargetRounds << extra;
    #pragma unroll 2
    for (int r = 0; r < pairs - 1; ++r) {
        s = tezcan_round(s, keys.round_key_1, t);
        s = tezcan_round(s, keys.round_key_2, t);
    }
    s = tezcan_round(s, keys.round_key_1, t);
    return aes_t0_last_word<0>(s, keys.round_key_2, t);
}

static inline Result128 tezcan_pairing_last(
    const AesHashKeys& keys, uint64_t l, uint64_t r,
    const uint32_t* t, int extra, uint32_t test_mask)
{
    AesState s = set_int_vec_i128(int32_t(r >> 32), int32_t(r),
                                  int32_t(l >> 32), int32_t(l));
    const int pairs = kAesPairingRounds << extra;
    #pragma unroll 2
    for (int i = 0; i < pairs - 1; ++i) {
        s = tezcan_round(s, keys.round_key_1, t);
        s = tezcan_round(s, keys.round_key_2, t);
    }
    s = tezcan_round(s, keys.round_key_1, t);
    Result128 res{{0u, 0u, 0u, 0u}};
    res.r[3] = aes_t0_last_word<3>(s, keys.round_key_2, t);
    // T1 requires match_info (word 0) only for accepted pairs.
    if ((res.r[3] & test_mask) == 0)
        res.r[0] = aes_t0_last_word<0>(s, keys.round_key_2, t);
    return res;
}


void launch_compute_bucket_offsets(
    XsCandidateGpu const* d_sorted,
    uint64_t total,
    int num_match_target_bits,
    uint32_t num_buckets,
    uint64_t* d_offsets,
    sycl::queue& q)
{
    constexpr size_t threads   = 256;
    size_t   const out_count   = static_cast<size_t>(num_buckets) + 1;
    size_t   const groups      = (out_count + threads - 1) / threads;

    q.parallel_for(
        sycl::nd_range<1>{ groups * threads, threads },
        [=](sycl::nd_item<1> it) {
            uint32_t b = static_cast<uint32_t>(it.get_global_id(0));
            if (b > num_buckets) return;
            if (b == num_buckets) { d_offsets[num_buckets] = total; return; }

            uint32_t bucket_shift = static_cast<uint32_t>(num_match_target_bits);
            uint64_t lo = 0, hi = total;
            while (lo < hi) {
                uint64_t mid = lo + ((hi - lo) >> 1);
                uint32_t v   = d_sorted[mid].match_info >> bucket_shift;
                if (v < b) lo = mid + 1;
                else       hi = mid;
            }
            d_offsets[b] = lo;
        }).wait();
}

void launch_compute_fine_bucket_offsets(
    XsCandidateGpu const* d_sorted,
    uint64_t const* d_bucket_offsets,
    int num_match_target_bits,
    int fine_bits,
    uint32_t num_buckets,
    uint64_t* d_fine_offsets,
    sycl::queue& q)
{
    constexpr size_t threads      = 256;
    uint32_t const   fine_count   = 1u << fine_bits;
    uint32_t const   total        = num_buckets * fine_count;
    size_t   const   groups       = (total + threads - 1) / threads;

    q.parallel_for(
        sycl::nd_range<1>{ groups * threads, threads },
        [=](sycl::nd_item<1> it) {
            uint32_t tid = static_cast<uint32_t>(it.get_global_id(0));
            if (tid >= total) return;

            uint32_t r_bucket = tid / fine_count;
            uint32_t fine_key = tid % fine_count;

            uint64_t r_start = d_bucket_offsets[r_bucket];
            uint64_t r_end   = d_bucket_offsets[r_bucket + 1];

            uint32_t target_mask = (num_match_target_bits >= 32)
                                    ? 0xFFFFFFFFu
                                    : ((1u << num_match_target_bits) - 1u);
            uint32_t shift = static_cast<uint32_t>(num_match_target_bits - fine_bits);

            uint64_t lo = r_start, hi = r_end;
            while (lo < hi) {
                uint64_t mid = lo + ((hi - lo) >> 1);
                uint32_t t   = (d_sorted[mid].match_info & target_mask) >> shift;
                if (t < fine_key) lo = mid + 1;
                else              hi = mid;
            }
            d_fine_offsets[tid] = lo;

            if (tid == total - 1) {
                d_fine_offsets[total] = d_bucket_offsets[num_buckets];
            }
        }).wait();
}

// Diagnostic-only stage probes. The original production kernel below is
// unchanged except for a rare-branch fallback counter. These kernels reuse
// the same AES implementation, search indices and group (1,256) geometry.
// Sampling changes the amount of work but NOT which two probes are compared.
// Target+search MINUS target-only includes search and scan, plus minor
// checksum dependencies; this is not a hardware instruction-level attribution.
template<bool WithSearch, bool CollectStats, bool WithPairing = false>
static void launch_t1_match_diag_probe(
    AesHashKeys keys,
    XsCandidateGpu const* d_sorted_xs,
    uint64_t const* d_offsets,
    uint64_t const* d_fine_offsets,
    uint32_t num_match_keys,
    int num_section_bits,
    int num_match_target_bits,
    int fine_bits,
    uint32_t target_mask,
    int extra_rounds_bits,
    uint64_t l_count_max,
    uint32_t bucket_begin, uint32_t bucket_end,
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
            [=, keys_copy = keys](sycl::nd_item<2> it) {
                uint32_t* sT = reinterpret_cast<uint32_t*>(&sT_local[0]);
                uint32_t local_id = uint32_t(it.get_local_id(1));
                // V4INIT: 16x replicated T0, 128-bit stores.
                #pragma unroll 1
                for (uint32_t j = local_id; j < 1024; j += uint32_t(threads)) {
                    sT_local[j] = sycl::vec<uint32_t, 4>(d_aes_tables[j / 4]);
                }
                it.barrier(sycl::access::fence_space::local_space);
                sT += (local_id & 15u);

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
                    uint32_t x_l = d_sorted_xs[l].x;
                    target_l = pos2gpu::tezcan_target(keys_copy, 1u, match_key_r, uint64_t(x_l), sT, extra_rounds_bits) & target_mask;
                    if constexpr (WithSearch) {
                        uint32_t fine_shift = uint32_t(num_match_target_bits - fine_bits);
                        uint32_t fine_key = target_l >> fine_shift;
                        uint64_t fine_idx = (uint64_t(r_bucket) << fine_bits) | fine_key;
                        uint64_t fine_hi = d_fine_offsets[fine_idx + 1];
                        uint64_t lo = d_fine_offsets[fine_idx], hi = fine_hi;
                        while (lo < hi) {
                            uint64_t mid = lo + ((hi - lo) >> 1);
                            uint32_t mid_target = d_sorted_xs[mid].match_info & target_mask;
                            if (mid_target < target_l) lo = mid + 1;
                            else hi = mid;
                        }
                        first = lo;
                        for (uint64_t r = first; r < fine_hi; ++r) {
                            if ((d_sorted_xs[r].match_info & target_mask) != target_l) break;
                            ++count;
                        }
                        if constexpr (WithPairing) {
                            for (uint64_t r = first; r < fine_hi; ++r) {
                                if ((d_sorted_xs[r].match_info & target_mask) != target_l) break;
                                uint32_t x_r = d_sorted_xs[r].x;
                                Result128 pair_res = pos2gpu::tezcan_pairing(
                                    keys_copy, uint64_t(x_l), uint64_t(x_r), sT, extra_rounds_bits);
                                pair_digest ^= pair_res.r[0] ^ pair_res.r[3];
                            }
                        }
                    }
                }
                // Consume results from every lane, so neither compiler nor
                // backend can eliminate the AES/search work as dead code.
                uint32_t digest = target_l;
                if constexpr (WithSearch) digest ^= uint32_t(first) ^ (count * 0x9e3779b9u);
                if constexpr (WithPairing) digest ^= pair_digest;
                uint32_t group_digest = sycl::reduce_over_group(
                    it.get_group(), digest, sycl::plus<uint32_t>{});
                if (local_id == 0) {
                    // Unique output per sampled workgroup: no serialized
                    // global atomic that could dominate the AES-only timer.
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

static void launch_t1_match_all_buckets_compact(
    AesHashKeys keys,
    XsCandidateGpu const* d_sorted_xs,
    uint64_t const* d_offsets,
    uint64_t const* d_fine_offsets,
    uint32_t num_match_keys,
    uint32_t num_buckets,
    int k,
    int num_section_bits,
    int num_match_target_bits,
    int fine_bits,
    int extra_rounds_bits,
    uint32_t target_mask,
    int num_test_bits,
    int num_match_info_bits,
    uint64_t* d_out_meta,
    uint32_t* d_out_mi,
    uint64_t* d_out_count,
    uint64_t out_capacity,
    uint64_t l_count_max,
    uint32_t bucket_begin,
    uint32_t bucket_end,
    sycl::queue& q)
{
    bool const use_direct24 = t1_direct24_enabled(
        fine_bits, num_match_target_bits, num_buckets);
    uint32_t const* d_direct24 = use_direct24
        ? reinterpret_cast<uint32_t const*>(
              d_fine_offsets + (uint64_t(num_buckets) << fine_bits) + 1)
        : nullptr;
    if (bucket_end <= bucket_begin) return;
    uint32_t const num_buckets_in_range = bucket_end - bucket_begin;

    uint32_t* d_aes_tables = sycl_backend::aes_tables_device(q);

    constexpr size_t threads = 256;
    uint64_t blocks_x_u64    = (l_count_max + threads - 1) / threads;
    size_t   const blocks_x  = static_cast<size_t>(blocks_x_u64);

    // Experiment A: independent of the Direct24/Direct20 index selection.
    // Read on host and capture a uniform flag; OFF leaves original behavior.
    bool const use_last_round_partial = [] {
        const char* v = std::getenv("POS2GPU_AES_LAST_ROUND_PARTIAL");
        return v && v[0] == '1' && v[1] == '\0';
    }();

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
        // Up to 512 pairs in each 256-lane group. Larger groups use the
        // original divergent pairing loop; no candidate is dropped.
        constexpr uint32_t pair_cap = 512;
        sycl::local_accessor<uint32_t, 1> pairs_l{sycl::range<1>{pair_cap}, h};
        sycl::local_accessor<uint32_t, 1> pairs_r{sycl::range<1>{pair_cap}, h};
        sycl::local_accessor<uint32_t, 1> total_local{sycl::range<1>{1}, h};

        h.parallel_for(
            sycl::nd_range<2>{
                sycl::range<2>{ static_cast<size_t>(num_buckets_in_range),
                                blocks_x * threads },
                sycl::range<2>{ 1, threads }
            },
            [=, keys_copy = keys](sycl::nd_item<2> it) {
                // Cooperative load of AES T-tables into local memory.
                uint32_t* sT = reinterpret_cast<uint32_t*>(&sT_local[0]);
                size_t local_id = it.get_local_id(1);
                // V4INIT: 16x replicated T0, 128-bit stores.
                #pragma unroll 1
                for (size_t j = local_id; j < 1024; j += threads) {
                    sT_local[j] = sycl::vec<uint32_t, 4>(d_aes_tables[j / 4]);
                }
                it.barrier(sycl::access::fence_space::local_space);
                sT += (local_id & 15);

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
                // All lanes, including the padding lanes, must reach every
                // group collective and barrier. Only valid lanes search.
                uint32_t x_l = 0;
                uint64_t first = 0;
                uint32_t count = 0;
                if (l < l_end) {
                    x_l = d_sorted_xs[l].x;
                    uint32_t target_l = (use_last_round_partial ? pos2gpu::tezcan_target_last(keys_copy, 1u, match_key_r, uint64_t(x_l), sT, extra_rounds_bits) : pos2gpu::tezcan_target(keys_copy, 1u, match_key_r, uint64_t(x_l), sT, extra_rounds_bits)) & target_mask;
                    if (use_direct24) {
                        uint64_t const key =
                            (uint64_t(r_bucket) << 24) | target_l;
                        first = d_direct24[key];
                        count = d_direct24[key + 1] - static_cast<uint32_t>(first);
                    } else {
                        uint32_t fine_shift = static_cast<uint32_t>(num_match_target_bits - fine_bits);
                        uint32_t fine_key = target_l >> fine_shift;
                        uint64_t fine_idx = (uint64_t(r_bucket) << fine_bits) | fine_key;
                        uint64_t lo = d_fine_offsets[fine_idx];
                        uint64_t fine_hi = d_fine_offsets[fine_idx + 1];
                        uint64_t hi = fine_hi;
                        while (lo < hi) {
                            uint64_t mid = lo + ((hi - lo) >> 1);
                            uint32_t target_mid = d_sorted_xs[mid].match_info & target_mask;
                            if (target_mid < target_l) lo = mid + 1;
                            else hi = mid;
                        }
                        first = lo;
                        for (uint64_t r = first; r < fine_hi; ++r) {
                            if ((d_sorted_xs[r].match_info & target_mask) != target_l) break;
                            ++count;
                        }
                    }
                }

                uint32_t test_mask = (num_test_bits >= 32) ? 0xFFFFFFFFu
                                                           : ((1u << num_test_bits) - 1u);
                uint32_t info_mask = (num_match_info_bits >= 32) ? 0xFFFFFFFFu
                                                                : ((1u << num_match_info_bits) - 1u);
                auto emit = [&](uint32_t a, uint32_t b) {
                    Result128 res = (use_last_round_partial ? pos2gpu::tezcan_pairing_last(keys_copy, uint64_t(a), uint64_t(b), sT, extra_rounds_bits, test_mask) : pos2gpu::tezcan_pairing(keys_copy, uint64_t(a), uint64_t(b), sT, extra_rounds_bits));
                    if ((res.r[3] & test_mask) != 0) return;
                    sycl::atomic_ref<unsigned long long,
                                     sycl::memory_order::relaxed,
                                     sycl::memory_scope::device>
                        out_count_atomic{*d_out_count_ull};
                    unsigned long long out_idx = out_count_atomic.fetch_add(1ULL);
                    if (out_idx >= out_capacity) return;
                    d_out_meta[out_idx] = (uint64_t(a) << k) | uint64_t(b);
                    d_out_mi[out_idx] = res.r[0] & info_mask;
                };

                // Saturating each contribution at cap+1 keeps the scan
                // in 32-bit range even for pathological distributions.
                // Preserve the actual count for the overflow fallback.
                uint32_t scan_count = count > pair_cap ? pair_cap + 1 : count;
                uint32_t offset = sycl::exclusive_scan_over_group(
                    it.get_group(), scan_count, sycl::plus<uint32_t>{});
                if (local_id == threads - 1) {
                    total_local[0] = offset + scan_count;
                }
                it.barrier(sycl::access::fence_space::local_space);
                uint32_t total_pairs = total_local[0];
                if (total_pairs > pair_cap) {
                    if (d_match_diag_stats && local_id == 0) {
                        using Atomic = sycl::atomic_ref<unsigned long long,
                            sycl::memory_order::relaxed, sycl::memory_scope::device>;
                        Atomic{d_match_diag_stats->fallback_groups}.fetch_add(1ULL);
                    }
                    // Uniform group branch, exact fallback. No local pair writes.
                    for (uint32_t j = 0; j < count; ++j)
                        emit(x_l, d_sorted_xs[first + j].x);
                    return;
                }
                for (uint32_t j = 0; j < count; ++j) {
                    uint32_t idx = uint32_t(offset) + j;
                    pairs_l[idx] = x_l;
                    pairs_r[idx] = d_sorted_xs[first + j].x;
                }
                it.barrier(sycl::access::fence_space::local_space);

                // Each lane now processes a contiguous candidate slot.
                // All full warps perform the long pairing AES in lockstep.
                for (uint32_t idx = uint32_t(local_id); idx < total_pairs; idx += uint32_t(threads))
                    emit(pairs_l[idx], pairs_r[idx]);
            });
    }).wait();
    auto const production_end = std::chrono::steady_clock::now();
    if (diag.enabled) {
        MatchDiagStats host_stats{};
        uint32_t first_digest = 0;
        using Clock = std::chrono::steady_clock;
        q.memcpy(&host_stats, d_match_diag_stats, sizeof(host_stats)).wait();
        // Do not include the sampling probes in production_fused_ms.
        double fused_ms = std::chrono::duration<double, std::milli>(
            production_end - production_start).count();
        size_t const sampled_groups =
            ((size_t(l_count_max + 255) / 256) + diag.stride - 1) / diag.stride;
        size_t const total_sampled_groups =
            size_t(bucket_end - bucket_begin) * sampled_groups;
        auto* d_digest = sycl::malloc_device<uint32_t>(total_sampled_groups, q);
        if (!d_digest) {
            sycl::free(d_match_diag_stats, q);
            throw std::bad_alloc();
        }
        double aes_ms = 0.0, aes_search_ms = 0.0, pair_ms = 0.0;
        // One warmup + two AB rounds, all on the SAME sampled workgroups.
        // Disable any other concurrent plotting before comparing the delta.
        for (int iteration = -1; iteration < 2; ++iteration) {
            auto a = Clock::now();
            launch_t1_match_diag_probe<false, false>(
                keys, d_sorted_xs, d_offsets, d_fine_offsets,
                num_match_keys, num_section_bits, num_match_target_bits,
                fine_bits, target_mask, extra_rounds_bits, l_count_max,
                bucket_begin, bucket_end, diag.stride, d_digest, d_match_diag_stats, q);
            auto b = Clock::now();
            auto bstart = Clock::now();
            launch_t1_match_diag_probe<true, false>(
                keys, d_sorted_xs, d_offsets, d_fine_offsets,
                num_match_keys, num_section_bits, num_match_target_bits,
                fine_bits, target_mask, extra_rounds_bits, l_count_max,
                bucket_begin, bucket_end, diag.stride, d_digest, d_match_diag_stats, q);
            auto c = Clock::now();
            auto pstart = Clock::now();
            launch_t1_match_diag_probe<true, false, true>(
                keys, d_sorted_xs, d_offsets, d_fine_offsets,
                num_match_keys, num_section_bits, num_match_target_bits,
                fine_bits, target_mask, extra_rounds_bits, l_count_max,
                bucket_begin, bucket_end, diag.stride, d_digest, d_match_diag_stats, q);
            auto pend = Clock::now();
            if (iteration >= 0) {
                aes_ms += std::chrono::duration<double, std::milli>(b - a).count() / 2.0;
                // No global-atomic contention; difference is still an estimate.
                aes_search_ms += std::chrono::duration<double, std::milli>(c - bstart).count() / 2.0;
                pair_ms += std::chrono::duration<double, std::milli>(pend - pstart).count() / 2.0;
            }
        }
        launch_t1_match_diag_probe<true, true>(
            keys, d_sorted_xs, d_offsets, d_fine_offsets,
                num_match_keys, num_section_bits, num_match_target_bits,
                fine_bits, target_mask, extra_rounds_bits, l_count_max,
                bucket_begin, bucket_end, diag.stride, d_digest, d_match_diag_stats, q);
        q.memcpy(&first_digest, d_digest, sizeof(first_digest)).wait();
        q.memcpy(&host_stats, d_match_diag_stats, sizeof(host_stats)).wait();
        print_match_diag_stats(1, diag.stride, aes_ms,
                               aes_search_ms, pair_ms, fused_ms, host_stats, first_digest);
        sycl::free(d_digest, q);
        sycl::free(d_match_diag_stats, q);
    }
}

void launch_t1_match_all_buckets(
    AesHashKeys keys,
    XsCandidateGpu const* d_sorted_xs,
    uint64_t const* d_offsets,
    uint64_t const* d_fine_offsets,
    uint32_t num_match_keys,
    uint32_t num_buckets,
    int k,
    int num_section_bits,
    int num_match_target_bits,
    int fine_bits,
    int extra_rounds_bits,
    uint32_t target_mask,
    int num_test_bits,
    int num_match_info_bits,
    uint64_t* d_out_meta,
    uint32_t* d_out_mi,
    uint64_t* d_out_count,
    uint64_t out_capacity,
    uint64_t l_count_max,
    uint32_t bucket_begin,
    uint32_t bucket_end,
    sycl::queue& q)
{
    // A and B both specialize T1 target; measure them separately.
    if (const char* a = std::getenv("POS2GPU_AES_LAST_ROUND_PARTIAL");
        a && a[0] == '1') {
        if (const char* b = std::getenv("POS2GPU_T1_FIRST_ROUND_FOLD");
            b && b[0] == '1')
            throw std::invalid_argument("Experiment A and B cannot both be ON for T1");
    }
    // Experimental path: default OFF, unchanged fused kernel otherwise.
    if (const char* mode = std::getenv("POS2GPU_T1_COMPACT"); mode && mode[0] == '1') {
        launch_t1_match_all_buckets_compact(
            keys, d_sorted_xs, d_offsets, d_fine_offsets,
            num_match_keys, num_buckets, k, num_section_bits,
            num_match_target_bits, fine_bits, extra_rounds_bits, target_mask,
            num_test_bits, num_match_info_bits, d_out_meta, d_out_mi,
            d_out_count, out_capacity, l_count_max, bucket_begin, bucket_end, q);
        return;
    }
    (void)num_buckets;
    if (bucket_end <= bucket_begin) return;
    uint32_t const num_buckets_in_range = bucket_end - bucket_begin;

    uint32_t* d_aes_tables = sycl_backend::aes_tables_device(q);

    constexpr size_t threads = 256;
    uint64_t blocks_x_u64    = (l_count_max + threads - 1) / threads;
    size_t   const blocks_x  = static_cast<size_t>(blocks_x_u64);

    // Experiment A: independent of the Direct24/Direct20 index selection.
    // Read on host and capture a uniform flag; OFF leaves original behavior.
    bool const use_last_round_partial = [] {
        const char* v = std::getenv("POS2GPU_AES_LAST_ROUND_PARTIAL");
        return v && v[0] == '1' && v[1] == '\0';
    }();

    auto* d_out_count_ull =
        reinterpret_cast<unsigned long long*>(d_out_count);

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<sycl::vec<uint32_t, 4>, 1> sT_local{
            sycl::range<1>{1024}, h};

        h.parallel_for(
            sycl::nd_range<2>{
                sycl::range<2>{ static_cast<size_t>(num_buckets_in_range),
                                blocks_x * threads },
                sycl::range<2>{ 1, threads }
            },
            [=, keys_copy = keys](sycl::nd_item<2> it) {
                // Cooperative load of AES T-tables into local memory.
                uint32_t* sT = reinterpret_cast<uint32_t*>(&sT_local[0]);
                size_t local_id = it.get_local_id(1);
                // V4INIT: 16x replicated T0, 128-bit stores.
                #pragma unroll 1
                for (size_t j = local_id; j < 1024; j += threads) {
                    sT_local[j] = sycl::vec<uint32_t, 4>(d_aes_tables[j / 4]);
                }
                it.barrier(sycl::access::fence_space::local_space);
                sT += (local_id & 15);

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
                if (l >= l_end) return;

                uint32_t x_l = d_sorted_xs[l].x;

                uint32_t target_l = (use_last_round_partial ? pos2gpu::tezcan_target_last(keys_copy, 1u, match_key_r, uint64_t(x_l), sT, extra_rounds_bits) : pos2gpu::tezcan_target(keys_copy, 1u, match_key_r, uint64_t(x_l), sT, extra_rounds_bits))
                                  & target_mask;

                uint32_t fine_shift = static_cast<uint32_t>(num_match_target_bits - fine_bits);
                uint32_t fine_key   = target_l >> fine_shift;
                uint64_t fine_idx   = (uint64_t(r_bucket) << fine_bits) | fine_key;
                uint64_t lo         = d_fine_offsets[fine_idx];
                uint64_t fine_hi    = d_fine_offsets[fine_idx + 1];
                uint64_t hi         = fine_hi;

                while (lo < hi) {
                    uint64_t mid = lo + ((hi - lo) >> 1);
                    uint32_t target_mid = d_sorted_xs[mid].match_info & target_mask;
                    if (target_mid < target_l) lo = mid + 1;
                    else                       hi = mid;
                }

                uint32_t test_mask = (num_test_bits >= 32) ? 0xFFFFFFFFu
                                                            : ((1u << num_test_bits) - 1u);
                uint32_t info_mask = (num_match_info_bits >= 32) ? 0xFFFFFFFFu
                                                                 : ((1u << num_match_info_bits) - 1u);

                for (uint64_t r = lo; r < fine_hi; ++r) {
                    uint32_t target_r = d_sorted_xs[r].match_info & target_mask;
                    if (target_r != target_l) break;

                    uint32_t x_r = d_sorted_xs[r].x;
                    pos2gpu::Result128 res = (use_last_round_partial ? pos2gpu::tezcan_pairing_last(keys_copy, uint64_t(x_l), uint64_t(x_r), sT, extra_rounds_bits, test_mask) : pos2gpu::tezcan_pairing(keys_copy, uint64_t(x_l), uint64_t(x_r), sT, extra_rounds_bits));

                    uint32_t test_result = res.r[3] & test_mask;
                    if (test_result != 0) continue;

                    uint32_t match_info_result = res.r[0] & info_mask;

                    sycl::atomic_ref<unsigned long long,
                                     sycl::memory_order::relaxed,
                                     sycl::memory_scope::device>
                        out_count_atomic{ *d_out_count_ull };
                    unsigned long long out_idx = out_count_atomic.fetch_add(1ULL);
                    if (out_idx >= out_capacity) return;

                    uint64_t meta = (uint64_t(x_l) << k) | uint64_t(x_r);
                    d_out_meta[out_idx] = meta;
                    d_out_mi  [out_idx] = match_info_result;
                }
            });
    }).wait();
}

} // namespace pos2gpu
