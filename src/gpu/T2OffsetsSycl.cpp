// T2OffsetsSycl.cpp — SYCL implementation of T2's three backend-dispatched
// kernels. Pattern mirrors T1OffsetsSycl.cpp; reuses the shared SYCL
// queue + AES-table USM buffer from SyclBackend.hpp.

#include "gpu/MatchKernelCommon.cuh"
#include "gpu/FineBucketDirect.hpp"
#include "gpu/MatchDiag.hpp"
#include "gpu/SyclBackend.hpp"
#include "gpu/T2Offsets.cuh"
#include "host/PoolSizing.hpp"

#include "gpu/AesDirectAddressSycl.hpp"
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

// T2 experimental Tezcan AES (same 16x-replicated T0 mapping as T1).
// Device helpers stay local to this translation unit.
static inline AesState t2c_tezcan_round(
    AesState s, const AesState& key, const uint32_t* t, uint32_t bank)
{
    // Byte-offset address arithmetic: same T0 values and shared layout.
    return aes_direct_addr_t0_round(s, key, t, bank);
}

static inline AesState t2c_tezcan_rounds(
    AesState state, const AesHashKeys& keys, int rounds, const uint32_t* t, uint32_t bank = 0)
{
    #pragma unroll 2
    for (int r = 0; r < rounds; ++r) {
        state = t2c_tezcan_round(state, keys.round_key_1, t, bank);
        state = t2c_tezcan_round(state, keys.round_key_2, t, bank);
    }
    return state;
}

static inline uint32_t t2c_tezcan_target(
    const AesHashKeys& keys, uint32_t table_id, uint32_t match_key,
    uint64_t meta, const uint32_t* t, int extra, uint32_t bank = 0)
{
    AesState s = set_int_vec_i128(
        int32_t(meta >> 32), int32_t(meta),
        int32_t(match_key), int32_t(table_id));
    return t2c_tezcan_rounds(
        s, keys, kAesMatchingTargetRounds << extra, t, bank).w[0];
}

static inline Result128 t2c_tezcan_pairing(
    const AesHashKeys& keys, uint64_t l, uint64_t r,
    const uint32_t* t, int extra, uint32_t bank = 0)
{
    AesState s = set_int_vec_i128(
        int32_t(r >> 32), int32_t(r),
        int32_t(l >> 32), int32_t(l));
    s = t2c_tezcan_rounds(s, keys, kAesPairingRounds << extra, t, bank);
    return Result128{{s.w[0], s.w[1], s.w[2], s.w[3]}};
}


void launch_t2_compute_bucket_offsets(
    uint32_t const* d_sorted_mi,
    uint64_t total,
    int num_match_target_bits,
    uint32_t num_buckets,
    uint64_t* d_offsets,
    sycl::queue& q)
{
    constexpr size_t threads = 256;
    size_t   const out_count = static_cast<size_t>(num_buckets) + 1;
    size_t   const groups    = (out_count + threads - 1) / threads;

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
                uint32_t v   = d_sorted_mi[mid] >> bucket_shift;
                if (v < b) lo = mid + 1;
                else       hi = mid;
            }
            d_offsets[b] = lo;
        }).wait();
}

void launch_t2_compute_fine_bucket_offsets(
    uint32_t const* d_sorted_mi,
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
                uint32_t t   = (d_sorted_mi[mid] & target_mask) >> shift;
                if (t < fine_key) lo = mid + 1;
                else              hi = mid;
            }
            d_fine_offsets[tid] = lo;

            if (tid == total - 1) {
                d_fine_offsets[total] = d_bucket_offsets[num_buckets];
            }
        }).wait();
}

namespace {

// Candidate scratch comes from the ONE buffer that T2 and T3 share (see
// sycl_backend::acquire_twophase_scratch in SyclBackend.hpp — they are never
// live at the same time, so the peak is the max of the two, not their sum).
// The test accepts one in 2^strength candidates, so the shared sizing rule
// keeps 50% headroom above that candidate:output ratio. At strength two this
// is 6 x the average per-bucket output; higher strengths need a proportionally
// larger multiplier because they divide the same output among more buckets.

// ---- Phase 2 prototype: two-phase T2 match ----------------------------
//
// Mirrors run_t3_match_twophase (T3OffsetsSycl.cpp). The single-kernel
// launch_t2_match_all_buckets has the same divergent shape as T3 — one
// thread per l-entry with a variable-length inner r-loop and the 32-round
// pairing AES inside it. This splits it:
//   Phase A — enumerate {u32 l, u32 r} candidate index pairs
//     (matching_target AES + binary search + r-loop). No pairing.
//   Phase B — process the flat pair list convergently: one thread per
//     candidate does exactly one pairing AES + test + the T2 output
//     (meta / match_info / x_bits).
// Tiled one bucket at a time so the candidate scratch stays bounded.
// Default-on (POS2GPU_T2_TWOPHASE=0 disables).
//
// Returns false when a bucket overflowed the candidate scratch — the
// caller rewinds the output counter and re-runs the range through the
// single-kernel path, which tolerates any skew.
[[nodiscard]] bool run_t2_match_twophase(
    AesHashKeys const& keys,
    uint64_t const* d_sorted_meta,
    uint32_t const* d_sorted_mi,
    uint64_t const* d_offsets,
    uint64_t const* d_fine_offsets,
    uint32_t num_match_keys,
    int k,
    int num_section_bits,
    int num_match_target_bits,
    int fine_bits,
    uint32_t target_mask,
    int num_test_bits,
    int num_match_info_bits,
    int half_k,
    uint64_t* d_out_meta,
    uint32_t* d_out_mi,
    uint32_t* d_out_xbits,
    uint64_t* d_out_count,
    uint64_t out_capacity,
    uint64_t l_count_max,
    uint32_t bucket_begin,
    uint32_t bucket_end,
    uint32_t* d_aes_tables,
    sycl::queue& q)
{
    uint32_t const num_buckets_in_range = bucket_end - bucket_begin;
    constexpr size_t threads = 256;
    size_t const blocks_x = static_cast<size_t>(
        (l_count_max + threads - 1) / threads);

    bool const dbg = [] {
        char const* v = std::getenv("POS2GPU_T2_TWOPHASE_DEBUG");
        return v && v[0] == '1';
    }();

    // Per-bucket candidate scratch: {u32 l, u32 r}. The test filter passes
    // one in num_match_keys candidates. Shared sizing keeps headroom above
    // that ratio at every strength; overflow still rewinds and retries the
    // allocation-free path. T2 and T3 reuse the same per-queue allocation.
    uint64_t const cand_cap = twophase_candidate_capacity(out_capacity, num_buckets_in_range, num_match_keys);
    auto* scratch_p = sycl_backend::acquire_twophase_scratch(q, cand_cap);
    if (!scratch_p) return false;   // over budget or OOM → single-kernel path
    auto& scratch = *scratch_p;
    uint32_t* d_cand_l     = scratch.d_cand_l;
    uint32_t* d_cand_r     = scratch.d_cand_r;
    uint64_t* d_cand_count = scratch.d_cand_count;
    uint32_t* d_overflow   = scratch.d_overflow;
    auto* d_cand_count_ull =
        reinterpret_cast<unsigned long long*>(d_cand_count);
    auto* d_out_count_ull =
        reinterpret_cast<unsigned long long*>(d_out_count);

    // Sticky overflow flag, checked once after the bucket loop —
    // silent candidate truncation would produce a plot that farms
    // fewer proofs with no diagnostic.
    q.memset(d_overflow, 0, sizeof(uint32_t)).wait();

    // Production v2: per-bucket ops chain through SYCL event deps —
    // memset → Phase A → Phase B → next bucket — with zero host sync per
    // bucket. See run_t3_match_twophase for the full rationale; this
    // mirrors that design.
    size_t const b_blocks_max = static_cast<size_t>(
        (cand_cap + threads - 1) / threads);

    sycl::event prev_event;
    bool have_prev = false;

    for (uint32_t b = bucket_begin; b < bucket_end; ++b) {
        auto e_memset = q.submit([&](sycl::handler& h) {
            if (have_prev) h.depends_on(prev_event);
            h.memset(d_cand_count, 0, sizeof(uint64_t));
        });

        // ---- Phase A: enumerate candidate (l, r) index pairs ----
        auto e_phase_a = q.submit([&](sycl::handler& h) {
            h.depends_on(e_memset);
            sycl::local_accessor<uint32_t, 1> sT_local{
                sycl::range<1>{4 * 256}, h};
            h.parallel_for(
                sycl::nd_range<1>{ blocks_x * threads, threads },
                [=, keys_copy = keys](sycl::nd_item<1> it) {
                    uint32_t* sT = &sT_local[0];
                    size_t local_id = it.get_local_id(0);
                    #pragma unroll 1
                    for (size_t i = local_id; i < 4 * 256; i += threads) {
                        sT[i] = d_aes_tables[i];
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    uint32_t bucket_id   = b;
                    uint32_t section_l   = bucket_id / num_match_keys;
                    uint32_t match_key_r = bucket_id % num_match_keys;

                    uint32_t section_r = pos2gpu::matching_section_r(section_l, num_section_bits);

                    uint64_t l_start = d_offsets[section_l * num_match_keys];
                    uint64_t l_end   = d_offsets[(section_l + 1) * num_match_keys];
                    uint32_t r_bucket = section_r * num_match_keys + match_key_r;

                    uint64_t l = l_start + it.get_global_id(0);
                    if (l >= l_end) return;

                    uint64_t meta_l = d_sorted_meta[l];
                    uint32_t target_l = pos2gpu::matching_target_smem(
                                            keys_copy, 2u, match_key_r, meta_l, sT, 0)
                                      & target_mask;

                    uint32_t fine_shift = static_cast<uint32_t>(num_match_target_bits - fine_bits);
                    uint32_t fine_key   = target_l >> fine_shift;
                    uint64_t fine_idx   = (uint64_t(r_bucket) << fine_bits) | fine_key;
                    uint64_t fine_hi    = d_fine_offsets[fine_idx + 1];
                    uint64_t lo         = pos2gpu::fine_bucket_lower_bound(
                        d_sorted_mi, d_fine_offsets[fine_idx], fine_hi,
                        target_l, target_mask);

                    for (uint64_t r = lo; r < fine_hi; ++r) {
                        uint32_t target_r = d_sorted_mi[r] & target_mask;
                        if (target_r != target_l) break;

                        sycl::atomic_ref<unsigned long long,
                                         sycl::memory_order::relaxed,
                                         sycl::memory_scope::device>
                            cand_atomic{ *d_cand_count_ull };
                        unsigned long long ci = cand_atomic.fetch_add(1ULL);
                        if (ci >= cand_cap) {
                            sycl::atomic_ref<uint32_t,
                                             sycl::memory_order::relaxed,
                                             sycl::memory_scope::device>
                                ovf{ *d_overflow };
                            ovf.store(1u);
                            return;
                        }
                        d_cand_l[ci] = static_cast<uint32_t>(l);
                        d_cand_r[ci] = static_cast<uint32_t>(r);
                    }
                });
        });

        // ---- Phase B: process this bucket's candidates convergently ----
        // cand_cap-sized grid; threads read the actual cand_n from the
        // device counter (Phase A's writes are visible via e_phase_a).
        auto e_phase_b = q.submit([&](sycl::handler& h) {
            h.depends_on(e_phase_a);
            sycl::local_accessor<uint32_t, 1> sT_local{
                sycl::range<1>{4 * 256}, h};
            h.parallel_for(
                sycl::nd_range<1>{ b_blocks_max * threads, threads },
                [=, keys_copy = keys](sycl::nd_item<1> it) {
                    uint32_t* sT = &sT_local[0];
                    size_t local_id = it.get_local_id(0);
                    #pragma unroll 1
                    for (size_t i = local_id; i < 4 * 256; i += threads) {
                        sT[i] = d_aes_tables[i];
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    uint64_t idx = it.get_global_id(0);
                    uint64_t cand_n = *d_cand_count;
                    if (cand_n > cand_cap) cand_n = cand_cap;
                    if (idx >= cand_n) return;

                    uint32_t l = d_cand_l[idx];
                    uint32_t r = d_cand_r[idx];

                    uint64_t meta_l = d_sorted_meta[l];
                    uint64_t meta_r = d_sorted_meta[r];

                    uint32_t test_mask = (num_test_bits >= 32) ? 0xFFFFFFFFu
                                                                : ((1u << num_test_bits) - 1u);
                    uint32_t info_mask = (num_match_info_bits >= 32) ? 0xFFFFFFFFu
                                                                     : ((1u << num_match_info_bits) - 1u);
                    int meta_bits = 2 * k;

                    pos2gpu::Result128 res = pos2gpu::pairing_smem(
                        keys_copy, meta_l, meta_r, sT, 0);

                    uint32_t test_result = res.r[3] & test_mask;
                    if (test_result != 0) return;

                    uint32_t match_info_result = res.r[0] & info_mask;
                    uint64_t meta_result_full = uint64_t(res.r[1]) | (uint64_t(res.r[2]) << 32);
                    uint64_t meta_result = (meta_bits == 64)
                                            ? meta_result_full
                                            : (meta_result_full & ((1ULL << meta_bits) - 1ULL));

                    uint32_t x_bits_l = static_cast<uint32_t>((meta_l >> k) >> half_k);
                    uint32_t x_bits_r = static_cast<uint32_t>((meta_r >> k) >> half_k);
                    uint32_t x_bits   = (x_bits_l << half_k) | x_bits_r;

                    sycl::atomic_ref<unsigned long long,
                                     sycl::memory_order::relaxed,
                                     sycl::memory_scope::device>
                        out_count_atomic{ *d_out_count_ull };
                    unsigned long long out_idx = out_count_atomic.fetch_add(1ULL);
                    if (out_idx >= out_capacity) return;

                    d_out_meta [out_idx] = meta_result;
                    d_out_mi   [out_idx] = match_info_result;
                    d_out_xbits[out_idx] = x_bits;
                });
        });

        // Optional debug print — adds one host sync per bucket.
        if (dbg) {
            uint64_t cand_n_dbg = 0;
            q.submit([&](sycl::handler& h) {
                h.depends_on(e_phase_a);
                h.memcpy(&cand_n_dbg, d_cand_count, sizeof(uint64_t));
            }).wait();
            std::fprintf(stderr,
                "[t2-twophase] bucket %u: %llu candidates (cap %llu)%s\n",
                b, static_cast<unsigned long long>(cand_n_dbg),
                static_cast<unsigned long long>(cand_cap),
                cand_n_dbg > cand_cap ? "  *** OVERFLOW ***" : "");
        }

        prev_event = e_phase_b;
        have_prev  = true;
    }

    // Single drain — the caller expects synchronous completion. After
    // the wait the cached scratch is fully drained for the next call;
    // no free — the per-queue cache owns the buffers.
    q.wait();

    uint32_t overflowed = 0;
    q.memcpy(&overflowed, d_overflow, sizeof(uint32_t)).wait();
    return overflowed == 0;
}

bool t2_twophase_enabled()
{
    // Default ON — same divergence rationale as T3 (see
    // t3_twophase_enabled in T3OffsetsSycl.cpp). POS2GPU_T2_TWOPHASE=0
    // falls back to the single-kernel path.
    static bool const enabled = [] {
        char const* v = std::getenv("POS2GPU_T2_TWOPHASE");
        return !(v && v[0] == '0');
    }();
    return enabled;
}

} // namespace

// Diagnostic-only stage probes. The original production kernel below is
// unchanged except for a rare-branch fallback counter. These kernels reuse
// the same AES implementation, search indices and group (1,256) geometry.
// Sampling changes the amount of work but NOT which two probes are compared.
// Target+search MINUS target-only includes search and scan, plus minor
// checksum dependencies; this is not a hardware instruction-level attribution.
template<bool WithSearch, bool CollectStats, bool WithPairing = false>
static void launch_t2_match_diag_probe(
    AesHashKeys keys,
    uint64_t const* d_sorted_meta,
    uint32_t const* d_sorted_mi,
    uint64_t const* d_offsets,
    uint64_t const* d_fine_offsets,
    uint32_t num_match_keys,
    int num_section_bits,
    int num_match_target_bits,
    int fine_bits,
    uint32_t target_mask,
    bool use_tezcan,
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
                // T2 regular AES uses the unreplicated first 1024 entries.

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
                        ? t2c_tezcan_target(keys_copy, 2u, match_key_r, meta_l, sT, 0, local_id & 15u)
                        : pos2gpu::matching_target_smem(keys_copy, 2u, match_key_r, meta_l, sT, 0)) & target_mask;
                    if constexpr (WithSearch) {
                        uint32_t fine_shift = uint32_t(num_match_target_bits - fine_bits);
                        uint32_t fine_key = target_l >> fine_shift;
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
                                Result128 pair_res = use_tezcan
                                    ? t2c_tezcan_pairing(keys_copy, meta_l, meta_r, sT, 0, local_id & 15u)
                                    : pos2gpu::pairing_smem(keys_copy, meta_l, meta_r, sT, 0);
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

static void launch_t2_match_all_buckets_compact(
    AesHashKeys keys,
    uint64_t const* d_sorted_meta,
    uint32_t const* d_sorted_mi,
    uint64_t const* d_offsets,
    uint64_t const* d_fine_offsets,
    uint32_t num_match_keys,
    uint32_t num_buckets,
    int k,
    int num_section_bits,
    int num_match_target_bits,
    int fine_bits,
    uint32_t target_mask,
    int num_test_bits,
    int num_match_info_bits,
    int half_k,
    uint64_t* d_out_meta,
    uint32_t* d_out_mi,
    uint32_t* d_out_xbits,
    uint64_t* d_out_count,
    uint64_t out_capacity,
    uint64_t l_count_max,
    uint32_t bucket_begin,
    uint32_t bucket_end,
    sycl::queue& q)
{
    // Only this SYCL all-buckets COMPACT kernel uses the compressed suffix index.
    uint8_t const* d_direct_offsets = pos2gpu::fine_direct16_enabled(
        fine_bits, num_match_target_bits)
        ? pos2gpu::fine_direct16_ptr(const_cast<uint64_t*>(d_fine_offsets), num_buckets)
        : nullptr;

    (void)num_buckets; // only the [begin, end) sub-range is iterated
    if (bucket_end <= bucket_begin) return;
    uint32_t const num_buckets_in_range = bucket_end - bucket_begin;

    uint32_t* d_aes_tables = sycl_backend::aes_tables_device(q);

    constexpr size_t threads = 256;
    uint64_t blocks_x_u64    = (l_count_max + threads - 1) / threads;
    size_t   const blocks_x  = static_cast<size_t>(blocks_x_u64);

    // Set on the host; do not read environment variables on device.
    bool const use_tezcan = [] {
        const char* v = std::getenv("POS2GPU_T2_TEZCAN");
        return v && v[0] == '1';
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
        // 512 candidate pairs of 2x u64 = 8 KiB plus 16 KiB AES table.
        constexpr uint32_t pair_cap = 480;
        sycl::local_accessor<uint64_t, 1> pairs_l{sycl::range<1>{pair_cap}, h};
        sycl::local_accessor<uint64_t, 1> pairs_r{sycl::range<1>{pair_cap}, h};
        sycl::local_accessor<uint32_t, 1> total_local{sycl::range<1>{1}, h};

        h.parallel_for(
            sycl::nd_range<2>{
                sycl::range<2>{ static_cast<size_t>(num_buckets_in_range),
                                blocks_x * threads },
                sycl::range<2>{ 1, threads }
            },
            [=, keys_copy = keys](sycl::nd_item<2> it) {
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
                // Every lane must participate in scan/barriers, including
                // padded l lanes in the last block.
                uint64_t meta_l = 0;
                uint64_t first = 0, fine_hi = 0;
                uint32_t target_l = 0, count = 0;
                if (l < l_end) {
                    meta_l = d_sorted_meta[l];
                    target_l = (use_tezcan
                        ? t2c_tezcan_target(keys_copy, 2u, match_key_r, meta_l, sT, 0, static_cast<uint32_t>(local_id) & 15u)
                        : pos2gpu::matching_target_smem(keys_copy, 2u, match_key_r, meta_l, sT, 0))
                        & target_mask;
                    uint32_t shift = uint32_t(num_match_target_bits - fine_bits);
                    uint32_t fine_key = target_l >> shift;
                    uint64_t fi = (uint64_t(r_bucket) << fine_bits) | fine_key;
                    auto const range = pos2gpu::fine_direct16_find(
                        d_sorted_mi, d_fine_offsets, d_direct_offsets,
                        fi, target_l, target_mask);
                    first = range.first;
                    fine_hi = range.end;
                    for (uint64_t r = first; r < fine_hi; ++r) {
                        if ((d_sorted_mi[r] & target_mask) != target_l) break;
                        if (++count > pair_cap) break;
                    }
                }

                uint32_t const test_mask = (num_test_bits >= 32) ? 0xFFFFFFFFu
                    : ((1u << num_test_bits) - 1u);
                uint32_t const info_mask = (num_match_info_bits >= 32) ? 0xFFFFFFFFu
                    : ((1u << num_match_info_bits) - 1u);
                int const meta_bits = 2 * k;

                // Preserve all three output fields and the original atomic
                // allocation semantics; candidate order is not guaranteed.
                auto emit = [&](uint64_t ml, uint64_t mr) {
                    Result128 res = use_tezcan
                        ? t2c_tezcan_pairing(keys_copy, ml, mr, sT, 0, static_cast<uint32_t>(local_id) & 15u)
                        : pos2gpu::pairing_smem(keys_copy, ml, mr, sT, 0);
                    if ((res.r[3] & test_mask) != 0) return;
                    uint32_t mi = res.r[0] & info_mask;
                    uint64_t full_meta = uint64_t(res.r[1]) | (uint64_t(res.r[2]) << 32);
                    uint64_t out_meta = (meta_bits == 64)
                        ? full_meta : (full_meta & ((1ULL << meta_bits) - 1ULL));
                    uint32_t xl = static_cast<uint32_t>((ml >> k) >> half_k);
                    uint32_t xr = static_cast<uint32_t>((mr >> k) >> half_k);
                    uint32_t xb = (xl << half_k) | xr;
                    sycl::atomic_ref<unsigned long long,
                                     sycl::memory_order::relaxed,
                                     sycl::memory_scope::device>
                        counter{*d_out_count_ull};
                    unsigned long long oi = counter.fetch_add(1ULL);
                    if (oi >= out_capacity) return;
                    d_out_meta[oi] = out_meta;
                    d_out_mi[oi] = mi;
                    d_out_xbits[oi] = xb;
                };

                // Bound scan count to 513. Any overflow is handled by a
                // group-uniform exact fallback, not by dropping pairs.
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
                    // All lanes enter this branch together. Padded lanes do
                    // nothing, and no lane needs a later barrier.
                    if (l < l_end) {
                        for (uint64_t r = first; r < fine_hi; ++r) {
                            if ((d_sorted_mi[r] & target_mask) != target_l) break;
                            emit(meta_l, d_sorted_meta[r]);
                        }
                    }
                    return;
                }

                for (uint32_t j = 0; j < count; ++j) {
                    uint32_t ci = offset + j;
                    pairs_l[ci] = meta_l;
                    pairs_r[ci] = d_sorted_meta[first + j];
                }
                it.barrier(sycl::access::fence_space::local_space);
                for (uint32_t ci = uint32_t(local_id); ci < n; ci += uint32_t(threads))
                    emit(pairs_l[ci], pairs_r[ci]);
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
            launch_t2_match_diag_probe<false, false>(
                keys, d_sorted_meta, d_sorted_mi, d_offsets, d_fine_offsets,
                num_match_keys, num_section_bits, num_match_target_bits,
                fine_bits, target_mask, use_tezcan, l_count_max,
                bucket_begin, bucket_end, diag.stride, d_digest, d_match_diag_stats, q);
            auto b = Clock::now();
            auto bstart = Clock::now();
            launch_t2_match_diag_probe<true, false>(
                keys, d_sorted_meta, d_sorted_mi, d_offsets, d_fine_offsets,
                num_match_keys, num_section_bits, num_match_target_bits,
                fine_bits, target_mask, use_tezcan, l_count_max,
                bucket_begin, bucket_end, diag.stride, d_digest, d_match_diag_stats, q);
            auto c = Clock::now();
            auto pstart = Clock::now();
            launch_t2_match_diag_probe<true, false, true>(
                keys, d_sorted_meta, d_sorted_mi, d_offsets, d_fine_offsets,
                num_match_keys, num_section_bits, num_match_target_bits,
                fine_bits, target_mask, use_tezcan, l_count_max,
                bucket_begin, bucket_end, diag.stride, d_digest, d_match_diag_stats, q);
            auto pend = Clock::now();
            if (iteration >= 0) {
                aes_ms += std::chrono::duration<double, std::milli>(b - a).count() / 2.0;
                // No global-atomic contention; difference is still an estimate.
                aes_search_ms += std::chrono::duration<double, std::milli>(c - bstart).count() / 2.0;
                pair_ms += std::chrono::duration<double, std::milli>(pend - pstart).count() / 2.0;
            }
        }
        launch_t2_match_diag_probe<true, true>(
            keys, d_sorted_meta, d_sorted_mi, d_offsets, d_fine_offsets,
                num_match_keys, num_section_bits, num_match_target_bits,
                fine_bits, target_mask, use_tezcan, l_count_max,
                bucket_begin, bucket_end, diag.stride, d_digest, d_match_diag_stats, q);
        q.memcpy(&first_digest, d_digest, sizeof(first_digest)).wait();
        q.memcpy(&host_stats, d_match_diag_stats, sizeof(host_stats)).wait();
        print_match_diag_stats(2, diag.stride, aes_ms,
                               aes_search_ms, pair_ms, fused_ms, host_stats, first_digest);
        sycl::free(d_digest, q);
        sycl::free(d_match_diag_stats, q);
    }
}

static void launch_t2_match_all_buckets_tezcan(
    AesHashKeys keys,
    uint64_t const* d_sorted_meta,
    uint32_t const* d_sorted_mi,
    uint64_t const* d_offsets,
    uint64_t const* d_fine_offsets,
    uint32_t num_match_keys,
    uint32_t num_buckets,
    int k,
    int num_section_bits,
    int num_match_target_bits,
    int fine_bits,
    uint32_t target_mask,
    int num_test_bits,
    int num_match_info_bits,
    int half_k,
    uint64_t* d_out_meta,
    uint32_t* d_out_mi,
    uint32_t* d_out_xbits,
    uint64_t* d_out_count,
    uint64_t out_capacity,
    uint64_t l_count_max,
    uint32_t bucket_begin,
    uint32_t bucket_end,
    sycl::queue& q)
{
    (void)num_buckets; // only the [begin, end) sub-range is iterated
    if (bucket_end <= bucket_begin) return;
    uint32_t const num_buckets_in_range = bucket_end - bucket_begin;

    uint32_t* d_aes_tables = sycl_backend::aes_tables_device(q);

    constexpr size_t threads = 256;
    uint64_t blocks_x_u64    = (l_count_max + threads - 1) / threads;
    size_t   const blocks_x  = static_cast<size_t>(blocks_x_u64);

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
                uint32_t* sT = reinterpret_cast<uint32_t*>(&sT_local[0]);
                size_t local_id = it.get_local_id(1);
                // Tezcan 16-way bank-replicated T0. All 256 T0 entries are
                // available as 16 equivalent copies in shared memory.
                // V4INIT: 16x replicated T0, 128-bit stores.
                #pragma unroll 1
                for (size_t j = local_id; j < 1024; j += threads) {
                    sT_local[j] = sycl::vec<uint32_t, 4>(d_aes_tables[j / 4]);
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
                if (l >= l_end) return;

                uint64_t meta_l = d_sorted_meta[l];

                uint32_t target_l = t2c_tezcan_target(
                                        keys_copy, 2u, match_key_r, meta_l, sT, 0, static_cast<uint32_t>(local_id) & 15u)
                                  & target_mask;

                uint32_t fine_shift = static_cast<uint32_t>(num_match_target_bits - fine_bits);
                uint32_t fine_key   = target_l >> fine_shift;
                uint64_t fine_idx   = (uint64_t(r_bucket) << fine_bits) | fine_key;
                uint64_t fine_hi    = d_fine_offsets[fine_idx + 1];
                uint64_t lo         = pos2gpu::fine_bucket_lower_bound(
                    d_sorted_mi, d_fine_offsets[fine_idx], fine_hi,
                    target_l, target_mask);

                uint32_t test_mask = (num_test_bits >= 32) ? 0xFFFFFFFFu
                                                            : ((1u << num_test_bits) - 1u);
                uint32_t info_mask = (num_match_info_bits >= 32) ? 0xFFFFFFFFu
                                                                 : ((1u << num_match_info_bits) - 1u);
                int meta_bits = 2 * k;

                for (uint64_t r = lo; r < fine_hi; ++r) {
                    uint32_t target_r = d_sorted_mi[r] & target_mask;
                    if (target_r != target_l) break;

                    uint64_t meta_r = d_sorted_meta[r];

                    pos2gpu::Result128 res = t2c_tezcan_pairing(
                        keys_copy, meta_l, meta_r, sT, 0, static_cast<uint32_t>(local_id) & 15u);

                    uint32_t test_result = res.r[3] & test_mask;
                    if (test_result != 0) continue;

                    uint32_t match_info_result = res.r[0] & info_mask;
                    uint64_t meta_result_full = uint64_t(res.r[1]) | (uint64_t(res.r[2]) << 32);
                    uint64_t meta_result = (meta_bits == 64)
                                            ? meta_result_full
                                            : (meta_result_full & ((1ULL << meta_bits) - 1ULL));

                    uint32_t x_bits_l = static_cast<uint32_t>((meta_l >> k) >> half_k);
                    uint32_t x_bits_r = static_cast<uint32_t>((meta_r >> k) >> half_k);
                    uint32_t x_bits   = (x_bits_l << half_k) | x_bits_r;

                    sycl::atomic_ref<unsigned long long,
                                     sycl::memory_order::relaxed,
                                     sycl::memory_scope::device>
                        out_count_atomic{ *d_out_count_ull };
                    unsigned long long out_idx = out_count_atomic.fetch_add(1ULL);
                    if (out_idx >= out_capacity) return;

                    d_out_meta [out_idx] = meta_result;
                    d_out_mi   [out_idx] = match_info_result;
                    d_out_xbits[out_idx] = x_bits;
                }
            });
    }).wait();
}

void launch_t2_match_all_buckets(
    AesHashKeys keys,
    uint64_t const* d_sorted_meta,
    uint32_t const* d_sorted_mi,
    uint64_t const* d_offsets,
    uint64_t const* d_fine_offsets,
    uint32_t num_match_keys,
    uint32_t num_buckets,
    int k,
    int num_section_bits,
    int num_match_target_bits,
    int fine_bits,
    uint32_t target_mask,
    int num_test_bits,
    int num_match_info_bits,
    int half_k,
    uint64_t* d_out_meta,
    uint32_t* d_out_mi,
    uint32_t* d_out_xbits,
    uint64_t* d_out_count,
    uint64_t out_capacity,
    uint64_t l_count_max,
    uint32_t bucket_begin,
    uint32_t bucket_end,
    sycl::queue& q)
{
    // Tezcan-only experiment. Local compaction has precedence when ON.
    // This branch intentionally bypasses T2's two-phase attempt so the
    // original fused kernel and Tezcan fused kernel can be compared fairly.
    if (const char* tv = std::getenv("POS2GPU_T2_TEZCAN"); tv && tv[0] == '1') {
        const char* cv = std::getenv("POS2GPU_T2_COMPACT");
        if (!(cv && cv[0] == '1')) {
            launch_t2_match_all_buckets_tezcan(
                keys, d_sorted_meta, d_sorted_mi, d_offsets,
            d_fine_offsets, num_match_keys, num_buckets, k,
            num_section_bits, num_match_target_bits, fine_bits,
            target_mask, num_test_bits, num_match_info_bits, half_k,
            d_out_meta, d_out_mi, d_out_xbits, d_out_count, out_capacity,
            l_count_max, bucket_begin, bucket_end, q);
            return;
        }
    }
    // Optional work-group local compaction. Original implementations
    // (including two-phase) remain unchanged when disabled.
    if (const char* v = std::getenv("POS2GPU_T2_COMPACT"); v && v[0] == '1') {
        launch_t2_match_all_buckets_compact(
            keys, d_sorted_meta, d_sorted_mi, d_offsets,
            d_fine_offsets, num_match_keys, num_buckets, k,
            num_section_bits, num_match_target_bits, fine_bits,
            target_mask, num_test_bits, num_match_info_bits, half_k,
            d_out_meta, d_out_mi, d_out_xbits, d_out_count, out_capacity,
            l_count_max, bucket_begin, bucket_end, q);
        return;
    }
    (void)num_buckets; // only the [begin, end) sub-range is iterated
    if (bucket_end <= bucket_begin) return;
    uint32_t const num_buckets_in_range = bucket_end - bucket_begin;

    uint32_t* d_aes_tables = sycl_backend::aes_tables_device(q);

    // Two-phase split, on by default (see run_t2_match_twophase).
    // Gated on out_capacity <= UINT32_MAX: the candidate scratch stores
    // row indices as u32, which wraps at k = 32 stream capacities. On
    // candidate-scratch overflow (skewed buckets), rewind the output
    // counter and redo the range through the single-kernel path.
    if (t2_twophase_enabled() && out_capacity <= UINT32_MAX) {
        uint64_t saved_count = 0;
        q.memcpy(&saved_count, d_out_count, sizeof(uint64_t)).wait();
        if (run_t2_match_twophase(
                keys, d_sorted_meta, d_sorted_mi, d_offsets, d_fine_offsets,
                num_match_keys, k, num_section_bits, num_match_target_bits,
                fine_bits, target_mask, num_test_bits, num_match_info_bits, half_k,
                d_out_meta, d_out_mi, d_out_xbits, d_out_count, out_capacity,
                l_count_max, bucket_begin, bucket_end, d_aes_tables, q)) {
            return;
        }
        q.memcpy(d_out_count, &saved_count, sizeof(uint64_t)).wait();
    }

    constexpr size_t threads = 256;
    uint64_t blocks_x_u64    = (l_count_max + threads - 1) / threads;
    size_t   const blocks_x  = static_cast<size_t>(blocks_x_u64);

    auto* d_out_count_ull =
        reinterpret_cast<unsigned long long*>(d_out_count);

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> sT_local{
            sycl::range<1>{4 * 256}, h};

        h.parallel_for(
            sycl::nd_range<2>{
                sycl::range<2>{ static_cast<size_t>(num_buckets_in_range),
                                blocks_x * threads },
                sycl::range<2>{ 1, threads }
            },
            [=, keys_copy = keys](sycl::nd_item<2> it) {
                uint32_t* sT = &sT_local[0];
                size_t local_id = it.get_local_id(1);
                #pragma unroll 1
                for (size_t i = local_id; i < 4 * 256; i += threads) {
                    sT[i] = d_aes_tables[i];
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
                if (l >= l_end) return;

                uint64_t meta_l = d_sorted_meta[l];

                uint32_t target_l = pos2gpu::matching_target_smem(
                                        keys_copy, 2u, match_key_r, meta_l, sT, 0)
                                  & target_mask;

                uint32_t fine_shift = static_cast<uint32_t>(num_match_target_bits - fine_bits);
                uint32_t fine_key   = target_l >> fine_shift;
                uint64_t fine_idx   = (uint64_t(r_bucket) << fine_bits) | fine_key;
                uint64_t fine_hi    = d_fine_offsets[fine_idx + 1];
                uint64_t lo         = pos2gpu::fine_bucket_lower_bound(
                    d_sorted_mi, d_fine_offsets[fine_idx], fine_hi,
                    target_l, target_mask);

                uint32_t test_mask = (num_test_bits >= 32) ? 0xFFFFFFFFu
                                                            : ((1u << num_test_bits) - 1u);
                uint32_t info_mask = (num_match_info_bits >= 32) ? 0xFFFFFFFFu
                                                                 : ((1u << num_match_info_bits) - 1u);
                int meta_bits = 2 * k;

                for (uint64_t r = lo; r < fine_hi; ++r) {
                    uint32_t target_r = d_sorted_mi[r] & target_mask;
                    if (target_r != target_l) break;

                    uint64_t meta_r = d_sorted_meta[r];

                    pos2gpu::Result128 res = pos2gpu::pairing_smem(
                        keys_copy, meta_l, meta_r, sT, 0);

                    uint32_t test_result = res.r[3] & test_mask;
                    if (test_result != 0) continue;

                    uint32_t match_info_result = res.r[0] & info_mask;
                    uint64_t meta_result_full = uint64_t(res.r[1]) | (uint64_t(res.r[2]) << 32);
                    uint64_t meta_result = (meta_bits == 64)
                                            ? meta_result_full
                                            : (meta_result_full & ((1ULL << meta_bits) - 1ULL));

                    uint32_t x_bits_l = static_cast<uint32_t>((meta_l >> k) >> half_k);
                    uint32_t x_bits_r = static_cast<uint32_t>((meta_r >> k) >> half_k);
                    uint32_t x_bits   = (x_bits_l << half_k) | x_bits_r;

                    sycl::atomic_ref<unsigned long long,
                                     sycl::memory_order::relaxed,
                                     sycl::memory_scope::device>
                        out_count_atomic{ *d_out_count_ull };
                    unsigned long long out_idx = out_count_atomic.fetch_add(1ULL);
                    if (out_idx >= out_capacity) return;

                    d_out_meta [out_idx] = meta_result;
                    d_out_mi   [out_idx] = match_info_result;
                    d_out_xbits[out_idx] = x_bits;
                }
            });
    }).wait();
}

static void launch_t2_match_section_pair_split_compact(
    AesHashKeys keys,
    uint64_t const* d_meta_l_slice,
    uint64_t section_l_row_start,
    uint64_t const* d_meta_r_slice,
    uint32_t const* d_mi_r_slice,
    uint64_t section_r_row_start,
    uint64_t const* d_offsets,
    uint64_t const* d_fine_offsets,
    uint32_t num_match_keys,
    uint32_t num_buckets,
    int k,
    int num_section_bits,
    int num_match_target_bits,
    int fine_bits,
    uint32_t target_mask,
    int num_test_bits,
    int num_match_info_bits,
    int half_k,
    uint64_t* d_out_meta,
    uint32_t* d_out_mi,
    uint32_t* d_out_xbits,
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

    // Set on the host; do not read environment variables on device.
    bool const use_tezcan = [] {
        const char* v = std::getenv("POS2GPU_T2_TEZCAN");
        return v && v[0] == '1';
    }();

    auto* d_out_count_ull =
        reinterpret_cast<unsigned long long*>(d_out_count);

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<sycl::vec<uint32_t, 4>, 1> sT_local{
            sycl::range<1>{1024}, h};
        // 512 candidate pairs of 2x u64 = 8 KiB plus 16 KiB AES table.
        constexpr uint32_t pair_cap = 480;
        sycl::local_accessor<uint64_t, 1> pairs_l{sycl::range<1>{pair_cap}, h};
        sycl::local_accessor<uint64_t, 1> pairs_r{sycl::range<1>{pair_cap}, h};
        sycl::local_accessor<uint32_t, 1> total_local{sycl::range<1>{1}, h};

        h.parallel_for(
            sycl::nd_range<2>{
                sycl::range<2>{ static_cast<size_t>(num_buckets_in_range),
                                blocks_x * threads },
                sycl::range<2>{ 1, threads }
            },
            [=, keys_copy = keys](sycl::nd_item<2> it) {
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
                // Sliced path used by low-VRAM tiers. The indices in
                // offsets/fine_offsets are global; slice accesses must
                // subtract the appropriate section row start.
                uint64_t meta_l = 0;
                uint64_t first = 0, fine_hi = 0;
                uint32_t target_l = 0, count = 0;
                if (l < l_end) {
                    meta_l = d_meta_l_slice[l - section_l_row_start];
                    target_l = (use_tezcan
                        ? t2c_tezcan_target(keys_copy, 2u, match_key_r, meta_l, sT, 0, static_cast<uint32_t>(local_id) & 15u)
                        : pos2gpu::matching_target_smem(keys_copy, 2u, match_key_r, meta_l, sT, 0))
                        & target_mask;
                    uint32_t shift = uint32_t(num_match_target_bits - fine_bits);
                    uint32_t fine_key = target_l >> shift;
                    uint64_t fi = (uint64_t(r_bucket) << fine_bits) | fine_key;
                    fine_hi = d_fine_offsets[fi + 1];
                    uint64_t lo = d_fine_offsets[fi], hi = fine_hi;
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

                uint32_t const test_mask = (num_test_bits >= 32) ? 0xFFFFFFFFu
                    : ((1u << num_test_bits) - 1u);
                uint32_t const info_mask = (num_match_info_bits >= 32) ? 0xFFFFFFFFu
                    : ((1u << num_match_info_bits) - 1u);
                int const meta_bits = 2 * k;

                // Preserve all three output fields and the original atomic
                // allocation semantics; candidate order is not guaranteed.
                auto emit = [&](uint64_t ml, uint64_t mr) {
                    Result128 res = use_tezcan
                        ? t2c_tezcan_pairing(keys_copy, ml, mr, sT, 0, static_cast<uint32_t>(local_id) & 15u)
                        : pos2gpu::pairing_smem(keys_copy, ml, mr, sT, 0);
                    if ((res.r[3] & test_mask) != 0) return;
                    uint32_t mi = res.r[0] & info_mask;
                    uint64_t full_meta = uint64_t(res.r[1]) | (uint64_t(res.r[2]) << 32);
                    uint64_t out_meta = (meta_bits == 64)
                        ? full_meta : (full_meta & ((1ULL << meta_bits) - 1ULL));
                    uint32_t xl = static_cast<uint32_t>((ml >> k) >> half_k);
                    uint32_t xr = static_cast<uint32_t>((mr >> k) >> half_k);
                    uint32_t xb = (xl << half_k) | xr;
                    sycl::atomic_ref<unsigned long long,
                                     sycl::memory_order::relaxed,
                                     sycl::memory_scope::device>
                        counter{*d_out_count_ull};
                    unsigned long long oi = counter.fetch_add(1ULL);
                    if (oi >= out_capacity) return;
                    d_out_meta[oi] = out_meta;
                    d_out_mi[oi] = mi;
                    d_out_xbits[oi] = xb;
                };

                // Bound scan count to 513. Any overflow is handled by a
                // group-uniform exact fallback, not by dropping pairs.
                uint32_t offset = sycl::exclusive_scan_over_group(
                    it.get_group(), count, sycl::plus<uint32_t>{});
                if (local_id == threads - 1) total_local[0] = offset + count;
                it.barrier(sycl::access::fence_space::local_space);
                uint32_t n = total_local[0];
                if (n > pair_cap) {
                    // All lanes enter this branch together. Padded lanes do
                    // nothing, and no lane needs a later barrier.
                    if (l < l_end) {
                        for (uint64_t r = first; r < fine_hi; ++r) {
                            if ((d_mi_r_slice[r - section_r_row_start] & target_mask) != target_l) break;
                            emit(meta_l, d_meta_r_slice[r - section_r_row_start]);
                        }
                    }
                    return;
                }

                for (uint32_t j = 0; j < count; ++j) {
                    uint32_t ci = offset + j;
                    pairs_l[ci] = meta_l;
                    pairs_r[ci] = d_meta_r_slice[first + j - section_r_row_start];
                }
                it.barrier(sycl::access::fence_space::local_space);
                for (uint32_t ci = uint32_t(local_id); ci < n; ci += uint32_t(threads))
                    emit(pairs_l[ci], pairs_r[ci]);
            });
    }).wait();
}

static void launch_t2_match_section_pair_split_tezcan(
    AesHashKeys keys,
    uint64_t const* d_meta_l_slice,
    uint64_t section_l_row_start,
    uint64_t const* d_meta_r_slice,
    uint32_t const* d_mi_r_slice,
    uint64_t section_r_row_start,
    uint64_t const* d_offsets,
    uint64_t const* d_fine_offsets,
    uint32_t num_match_keys,
    uint32_t num_buckets,
    int k,
    int num_section_bits,
    int num_match_target_bits,
    int fine_bits,
    uint32_t target_mask,
    int num_test_bits,
    int num_match_info_bits,
    int half_k,
    uint64_t* d_out_meta,
    uint32_t* d_out_mi,
    uint32_t* d_out_xbits,
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
                uint32_t* sT = reinterpret_cast<uint32_t*>(&sT_local[0]);
                size_t local_id = it.get_local_id(1);
                // Tezcan 16-way bank-replicated T0. All 256 T0 entries are
                // available as 16 equivalent copies in shared memory.
                // V4INIT: 16x replicated T0, 128-bit stores.
                #pragma unroll 1
                for (size_t j = local_id; j < 1024; j += threads) {
                    sT_local[j] = sycl::vec<uint32_t, 4>(d_aes_tables[j / 4]);
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
                if (l >= l_end) return;

                // Sliced read (l-side): l ∈ [section_l_row_start, section_l_row_end).
                uint64_t meta_l = d_meta_l_slice[l - section_l_row_start];

                uint32_t target_l = t2c_tezcan_target(
                                        keys_copy, 2u, match_key_r, meta_l, sT, 0, static_cast<uint32_t>(local_id) & 15u)
                                  & target_mask;

                uint32_t fine_shift = static_cast<uint32_t>(num_match_target_bits - fine_bits);
                uint32_t fine_key   = target_l >> fine_shift;
                uint64_t fine_idx   = (uint64_t(r_bucket) << fine_bits) | fine_key;
                uint64_t lo         = d_fine_offsets[fine_idx];
                uint64_t fine_hi    = d_fine_offsets[fine_idx + 1];
                uint64_t hi         = fine_hi;

                while (lo < hi) {
                    uint64_t mid = lo + ((hi - lo) >> 1);
                    // Sliced read (r-side): r ∈ [section_r_row_start, section_r_row_end).
                    uint32_t target_mid = d_mi_r_slice[mid - section_r_row_start] & target_mask;
                    if (target_mid < target_l) lo = mid + 1;
                    else                       hi = mid;
                }

                uint32_t test_mask = (num_test_bits >= 32) ? 0xFFFFFFFFu
                                                            : ((1u << num_test_bits) - 1u);
                uint32_t info_mask = (num_match_info_bits >= 32) ? 0xFFFFFFFFu
                                                                 : ((1u << num_match_info_bits) - 1u);
                int meta_bits = 2 * k;

                for (uint64_t r = lo; r < fine_hi; ++r) {
                    uint64_t r_off    = r - section_r_row_start;
                    uint32_t target_r = d_mi_r_slice[r_off] & target_mask;
                    if (target_r != target_l) break;

                    uint64_t meta_r = d_meta_r_slice[r_off];

                    pos2gpu::Result128 res = t2c_tezcan_pairing(
                        keys_copy, meta_l, meta_r, sT, 0, static_cast<uint32_t>(local_id) & 15u);

                    uint32_t test_result = res.r[3] & test_mask;
                    if (test_result != 0) continue;

                    uint32_t match_info_result = res.r[0] & info_mask;
                    uint64_t meta_result_full = uint64_t(res.r[1]) | (uint64_t(res.r[2]) << 32);
                    uint64_t meta_result = (meta_bits == 64)
                                            ? meta_result_full
                                            : (meta_result_full & ((1ULL << meta_bits) - 1ULL));

                    uint32_t x_bits_l = static_cast<uint32_t>((meta_l >> k) >> half_k);
                    uint32_t x_bits_r = static_cast<uint32_t>((meta_r >> k) >> half_k);
                    uint32_t x_bits   = (x_bits_l << half_k) | x_bits_r;

                    sycl::atomic_ref<unsigned long long,
                                     sycl::memory_order::relaxed,
                                     sycl::memory_scope::device>
                        out_count_atomic{ *d_out_count_ull };
                    unsigned long long out_idx = out_count_atomic.fetch_add(1ULL);
                    if (out_idx >= out_capacity) return;

                    d_out_meta [out_idx] = meta_result;
                    d_out_mi   [out_idx] = match_info_result;
                    d_out_xbits[out_idx] = x_bits;
                }
            });
    }).wait();
}

void launch_t2_match_section_pair_split(
    AesHashKeys keys,
    uint64_t const* d_meta_l_slice,
    uint64_t section_l_row_start,
    uint64_t const* d_meta_r_slice,
    uint32_t const* d_mi_r_slice,
    uint64_t section_r_row_start,
    uint64_t const* d_offsets,
    uint64_t const* d_fine_offsets,
    uint32_t num_match_keys,
    uint32_t num_buckets,
    int k,
    int num_section_bits,
    int num_match_target_bits,
    int fine_bits,
    uint32_t target_mask,
    int num_test_bits,
    int num_match_info_bits,
    int half_k,
    uint64_t* d_out_meta,
    uint32_t* d_out_mi,
    uint32_t* d_out_xbits,
    uint64_t* d_out_count,
    uint64_t out_capacity,
    uint64_t l_count_max,
    uint32_t bucket_begin,
    uint32_t bucket_end,
    sycl::queue& q)
{
    // Tezcan-only experiment. Local compaction has precedence when ON.
    // This branch intentionally bypasses T2's two-phase attempt so the
    // original fused kernel and Tezcan fused kernel can be compared fairly.
    if (const char* tv = std::getenv("POS2GPU_T2_TEZCAN"); tv && tv[0] == '1') {
        const char* cv = std::getenv("POS2GPU_T2_COMPACT");
        if (!(cv && cv[0] == '1')) {
            launch_t2_match_section_pair_split_tezcan(
                keys, d_meta_l_slice, section_l_row_start,
            d_meta_r_slice, d_mi_r_slice, section_r_row_start,
            d_offsets, d_fine_offsets, num_match_keys, num_buckets,
            k, num_section_bits, num_match_target_bits, fine_bits,
            target_mask, num_test_bits, num_match_info_bits, half_k,
            d_out_meta, d_out_mi, d_out_xbits, d_out_count, out_capacity,
            l_count_max, bucket_begin, bucket_end, q);
            return;
        }
    }
    // Optional work-group local compaction. Original implementations
    // (including two-phase) remain unchanged when disabled.
    if (const char* v = std::getenv("POS2GPU_T2_COMPACT"); v && v[0] == '1') {
        launch_t2_match_section_pair_split_compact(
            keys, d_meta_l_slice, section_l_row_start,
            d_meta_r_slice, d_mi_r_slice, section_r_row_start,
            d_offsets, d_fine_offsets, num_match_keys, num_buckets,
            k, num_section_bits, num_match_target_bits, fine_bits,
            target_mask, num_test_bits, num_match_info_bits, half_k,
            d_out_meta, d_out_mi, d_out_xbits, d_out_count, out_capacity,
            l_count_max, bucket_begin, bucket_end, q);
        return;
    }
    (void)num_buckets;
    if (bucket_end <= bucket_begin) return;
    uint32_t const num_buckets_in_range = bucket_end - bucket_begin;

    uint32_t* d_aes_tables = sycl_backend::aes_tables_device(q);

    constexpr size_t threads = 256;
    uint64_t blocks_x_u64    = (l_count_max + threads - 1) / threads;
    size_t   const blocks_x  = static_cast<size_t>(blocks_x_u64);

    auto* d_out_count_ull =
        reinterpret_cast<unsigned long long*>(d_out_count);

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> sT_local{
            sycl::range<1>{4 * 256}, h};

        h.parallel_for(
            sycl::nd_range<2>{
                sycl::range<2>{ static_cast<size_t>(num_buckets_in_range),
                                blocks_x * threads },
                sycl::range<2>{ 1, threads }
            },
            [=, keys_copy = keys](sycl::nd_item<2> it) {
                uint32_t* sT = &sT_local[0];
                size_t local_id = it.get_local_id(1);
                #pragma unroll 1
                for (size_t i = local_id; i < 4 * 256; i += threads) {
                    sT[i] = d_aes_tables[i];
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
                if (l >= l_end) return;

                // Sliced read (l-side): l ∈ [section_l_row_start, section_l_row_end).
                uint64_t meta_l = d_meta_l_slice[l - section_l_row_start];

                uint32_t target_l = pos2gpu::matching_target_smem(
                                        keys_copy, 2u, match_key_r, meta_l, sT, 0)
                                  & target_mask;

                uint32_t fine_shift = static_cast<uint32_t>(num_match_target_bits - fine_bits);
                uint32_t fine_key   = target_l >> fine_shift;
                uint64_t fine_idx   = (uint64_t(r_bucket) << fine_bits) | fine_key;
                uint64_t lo         = d_fine_offsets[fine_idx];
                uint64_t fine_hi    = d_fine_offsets[fine_idx + 1];
                uint64_t hi         = fine_hi;

                while (lo < hi) {
                    uint64_t mid = lo + ((hi - lo) >> 1);
                    // Sliced read (r-side): r ∈ [section_r_row_start, section_r_row_end).
                    uint32_t target_mid = d_mi_r_slice[mid - section_r_row_start] & target_mask;
                    if (target_mid < target_l) lo = mid + 1;
                    else                       hi = mid;
                }

                uint32_t test_mask = (num_test_bits >= 32) ? 0xFFFFFFFFu
                                                            : ((1u << num_test_bits) - 1u);
                uint32_t info_mask = (num_match_info_bits >= 32) ? 0xFFFFFFFFu
                                                                 : ((1u << num_match_info_bits) - 1u);
                int meta_bits = 2 * k;

                for (uint64_t r = lo; r < fine_hi; ++r) {
                    uint64_t r_off    = r - section_r_row_start;
                    uint32_t target_r = d_mi_r_slice[r_off] & target_mask;
                    if (target_r != target_l) break;

                    uint64_t meta_r = d_meta_r_slice[r_off];

                    pos2gpu::Result128 res = pos2gpu::pairing_smem(
                        keys_copy, meta_l, meta_r, sT, 0);

                    uint32_t test_result = res.r[3] & test_mask;
                    if (test_result != 0) continue;

                    uint32_t match_info_result = res.r[0] & info_mask;
                    uint64_t meta_result_full = uint64_t(res.r[1]) | (uint64_t(res.r[2]) << 32);
                    uint64_t meta_result = (meta_bits == 64)
                                            ? meta_result_full
                                            : (meta_result_full & ((1ULL << meta_bits) - 1ULL));

                    uint32_t x_bits_l = static_cast<uint32_t>((meta_l >> k) >> half_k);
                    uint32_t x_bits_r = static_cast<uint32_t>((meta_r >> k) >> half_k);
                    uint32_t x_bits   = (x_bits_l << half_k) | x_bits_r;

                    sycl::atomic_ref<unsigned long long,
                                     sycl::memory_order::relaxed,
                                     sycl::memory_scope::device>
                        out_count_atomic{ *d_out_count_ull };
                    unsigned long long out_idx = out_count_atomic.fetch_add(1ULL);
                    if (out_idx >= out_capacity) return;

                    d_out_meta [out_idx] = meta_result;
                    d_out_mi   [out_idx] = match_info_result;
                    d_out_xbits[out_idx] = x_bits;
                }
            });
    }).wait();
}

} // namespace pos2gpu
