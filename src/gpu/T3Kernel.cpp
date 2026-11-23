// T3Kernel.cu — port of pos2-chip Table3Constructor.
//
// Differences from T2:
//   - Input is T2Pairing { meta(64), match_info(32), x_bits(32) }.
//   - matching_target uses table_id=3 and meta=T2Pairing.meta (no extra rounds).
//   - pairing_t3 only consumes test_result; no match_info / meta extraction
//     from the AES output. AES rounds = AES_PAIRING_ROUNDS (16), no strength
//     bonus.
//   - Emit T3Pairing { proof_fragment = FeistelCipher.encrypt(all_x_bits) }
//     where all_x_bits = (l.x_bits << k) | r.x_bits.

#include "gpu/AesGpu.cuh"
#include "gpu/AesHashGpu.cuh"
#include "gpu/FineBucketConfig.hpp"
#include "gpu/FeistelCipherGpu.cuh"
#include "gpu/T2Offsets.cuh"
#include "gpu/T3Kernel.cuh"
#include "gpu/T3Offsets.cuh"
#include "gpu/FineBucketDirect20.hpp"
#ifdef XCHPLOT2_NATIVE_CUDA_MATCH
#include "gpu/NativeCudaIndex.hpp"
#endif
#include "gpu/FineBucketIndex.hpp"
#include "host/PoolSizing.hpp"

#include <climits>
#include <cstdint>

namespace pos2gpu {

// The CUDA __constant__ FeistelKey + its setup have moved to
// T3OffsetsCuda.cu, scoped to the wrapper that uses them. The SYCL
// path captures FeistelKey by value in the lambda instead.

T3MatchParams make_t3_params(int k, int strength)
{
    T3MatchParams p{};
    p.k                     = k;
    p.strength              = strength;
    p.num_section_bits      = (k < 28) ? 2 : (k - 26);
    p.num_match_key_bits    = strength;
    p.num_match_target_bits = k - p.num_section_bits - p.num_match_key_bits;
    return p;
}

// T3's three kernels (compute_bucket_offsets, compute_fine_bucket_offsets,
// match_all_buckets) have moved to the cross-backend path. The two offset
// kernels are bit-identical to T2's and reuse T2Offsets.cuh's wrappers; the
// match kernel — Feistel-encrypted output — has its own wrapper in
// T3Offsets.cuh. The previously-unused matching_section helper went with
// them.


namespace {

// Fixed profile uses Fine16 for T3 Direct20.

struct T3Derived {
    uint32_t num_sections;
    uint32_t num_match_keys;
    uint32_t num_buckets;
    int      fine_bits;
    uint64_t fine_entries;
    size_t   bucket_bytes;
    size_t   fine_bytes;
    size_t   temp_needed;
    uint32_t target_mask;
    int      num_test_bits;
    uint64_t l_count_max;
};

T3Derived derive_t3(T3MatchParams const& params)
{
    T3Derived d{};
    d.num_sections    = 1u << params.num_section_bits;
    d.num_match_keys  = 1u << params.num_match_key_bits;
    d.num_buckets     = d.num_sections * d.num_match_keys;
    d.fine_bits = fine_bucket_bits(3, params.num_match_target_bits);
    FineBucketTempLayout const layout = fine_bucket_temp_layout(d.num_buckets, d.fine_bits);
    d.fine_entries    = layout.fine_entries;
    d.bucket_bytes    = layout.bucket_bytes;
    d.fine_bytes      = layout.fine_bytes;
    d.temp_needed     = layout.total_bytes();
    // k28/s6 is the only supported plotting profile. Fail clearly on
    // unexpected layouts rather than silently choosing an obsolete index.
    if (!fine_direct20_enabled(d.fine_bits, params.num_match_target_bits, d.num_buckets))
        throw std::invalid_argument("T3 Direct20 requires k=28, strength=6 layout");
    d.temp_needed += fine_direct20_bytes(d.num_buckets);
    d.target_mask     = (params.num_match_target_bits >= 32)
                          ? 0xFFFFFFFFu
                          : ((1u << params.num_match_target_bits) - 1u);
    d.num_test_bits   = params.num_match_key_bits;
    d.l_count_max =
        static_cast<uint64_t>(max_pairs_per_section(params.k, params.num_section_bits));
    return d;
}

} // namespace

void launch_t3_match_prepare(
    uint8_t const* plot_id_bytes,
    T3MatchParams const& params,
    uint32_t const* d_sorted_mi,
    uint64_t t2_count,
    uint64_t* d_out_count,
    void* d_temp_storage,
    size_t* temp_bytes,
    sycl::queue& q)
{
    if (!plot_id_bytes || !temp_bytes) throw std::invalid_argument("invalid argument to launch wrapper");
    if (params.k < 18 || params.k > 28) throw std::invalid_argument("invalid argument to launch wrapper");
    if (params.strength < 2)            throw std::invalid_argument("invalid argument to launch wrapper");

    validate_fine_bucket_input_count(t2_count);
    T3Derived const d = derive_t3(params);

    if (d_temp_storage == nullptr) {
        *temp_bytes = d.temp_needed;
        return;
    }
    if (*temp_bytes < d.temp_needed) throw std::invalid_argument("invalid argument to launch wrapper");
    if (!d_sorted_mi || !d_out_count) throw std::invalid_argument("invalid argument to launch wrapper");

    auto* d_offsets      = reinterpret_cast<uint64_t*>(d_temp_storage);
    auto* d_fine_offsets = fine_bucket_offset_ptr(d_temp_storage, d.bucket_bytes);

    FineBucketTimer fine_timer(3, d.fine_bits);

    // T3 reuses T2's offset wrappers (identical layout + algorithm).
    launch_t2_compute_bucket_offsets(
        d_sorted_mi, t2_count,
        params.num_match_target_bits,
        d.num_buckets, d_offsets, q);
    fine_timer.mark("bucket_offsets", q);
    launch_t2_compute_fine_bucket_offsets(
        d_sorted_mi, d_offsets,
        params.num_match_target_bits, d.fine_bits,
        d.num_buckets, d_fine_offsets, q);
    fine_timer.mark("fine_offsets", q);
    auto* direct = fine_direct20_ptr(d_fine_offsets, d.num_buckets);
    FineBucketTimer direct_timer(3, d.fine_bits);
    #ifdef XCHPLOT2_NATIVE_CUDA_MATCH
    if (q.get_device().get_backend() == sycl::backend::cuda) {
        q.wait();
        cuda_t23_direct20(d_sorted_mi, d_fine_offsets, direct, d.num_buckets);
    } else
#endif
    {
        fine_direct20_build(d_sorted_mi, d_fine_offsets, direct, d.num_buckets, q);
    }
    direct_timer.mark("direct20_build", q);
    q.memset(d_out_count, 0, sizeof(uint64_t)).wait();
}

void launch_t3_match_range(
    uint8_t const* plot_id_bytes,
    T3MatchParams const& params,
    uint64_t const* d_sorted_meta,
    uint32_t const* d_sorted_xbits,
    uint32_t const* d_sorted_mi,
    uint64_t t2_count,
    T3PairingGpu* d_out_pairings,
    uint64_t* d_out_count,
    uint64_t capacity,
    void const* d_temp_storage,
    uint32_t bucket_begin,
    uint32_t bucket_end,
    sycl::queue& q)
{
    validate_fine_bucket_input_count(t2_count);
    if (!plot_id_bytes) throw std::invalid_argument("invalid argument to launch wrapper");
    if (params.k < 18 || params.k > 28) throw std::invalid_argument("invalid argument to launch wrapper");
    if (params.strength < 2)            throw std::invalid_argument("invalid argument to launch wrapper");
    if (!d_temp_storage)                throw std::invalid_argument("invalid argument to launch wrapper");
    if (!d_sorted_meta || !d_sorted_xbits || !d_sorted_mi
        || !d_out_pairings || !d_out_count) throw std::invalid_argument("invalid argument to launch wrapper");

    T3Derived const d = derive_t3(params);

    if (bucket_end > d.num_buckets) throw std::invalid_argument("invalid argument to launch wrapper");
    if (bucket_end <= bucket_begin) return;

    constexpr int kThreads = 256;
    uint64_t const blocks_x_u64 = (d.l_count_max + kThreads - 1) / kThreads;
    if (blocks_x_u64 > UINT_MAX) throw std::invalid_argument("invalid argument to launch wrapper");

    auto const* d_offsets      = reinterpret_cast<uint64_t const*>(d_temp_storage);
    auto const* d_fine_offsets = fine_bucket_offset_ptr(d_temp_storage, d.bucket_bytes);

    AesHashKeys keys = make_keys(plot_id_bytes);
    FeistelKey  fk   = make_feistel_key(plot_id_bytes, params.k);

    FineBucketTimer fine_timer(3, d.fine_bits);
    launch_t3_match_all_buckets(
        keys, fk,
        d_sorted_meta, d_sorted_xbits, d_sorted_mi,
        const_cast<uint64_t*>(d_offsets),
        const_cast<FineBucketOffset*>(d_fine_offsets),
        d.num_match_keys, d.num_buckets,
        params.k, params.num_section_bits,
        params.num_match_target_bits, d.fine_bits,
        d.target_mask, d.num_test_bits,
        d_out_pairings, d_out_count,
        capacity, d.l_count_max,
        bucket_begin, bucket_end,
        q);
    fine_timer.mark("match_fused", q);
}

void launch_t3_match_section_pair_range(
    uint8_t const* plot_id_bytes,
    T3MatchParams const& params,
    uint64_t const* d_meta_l_slice,
    uint64_t section_l_row_start,
    uint64_t const* d_meta_r_slice,
    uint64_t section_r_row_start,
    uint32_t const* d_sorted_xbits,
    uint32_t const* d_sorted_mi,
    uint64_t t2_count,
    T3PairingGpu* d_out_pairings,
    uint64_t* d_out_count,
    uint64_t capacity,
    void const* d_temp_storage,
    uint32_t bucket_begin,
    uint32_t bucket_end,
    sycl::queue& q)
{
    validate_fine_bucket_input_count(t2_count);
    if (!plot_id_bytes) throw std::invalid_argument("invalid argument to launch wrapper");
    if (params.k < 18 || params.k > 28) throw std::invalid_argument("invalid argument to launch wrapper");
    if (params.strength < 2)            throw std::invalid_argument("invalid argument to launch wrapper");
    if (!d_temp_storage)                throw std::invalid_argument("invalid argument to launch wrapper");
    if (!d_meta_l_slice || !d_meta_r_slice
        || !d_sorted_xbits || !d_sorted_mi
        || !d_out_pairings || !d_out_count) throw std::invalid_argument("invalid argument to launch wrapper");

    T3Derived const d = derive_t3(params);

    if (bucket_end > d.num_buckets) throw std::invalid_argument("invalid argument to launch wrapper");
    if (bucket_end <= bucket_begin) return;

    constexpr int kThreads = 256;
    uint64_t const blocks_x_u64 = (d.l_count_max + kThreads - 1) / kThreads;
    if (blocks_x_u64 > UINT_MAX) throw std::invalid_argument("invalid argument to launch wrapper");

    auto const* d_offsets      = reinterpret_cast<uint64_t const*>(d_temp_storage);
    auto const* d_fine_offsets = fine_bucket_offset_ptr(d_temp_storage, d.bucket_bytes);

    AesHashKeys keys = make_keys(plot_id_bytes);
    FeistelKey  fk   = make_feistel_key(plot_id_bytes, params.k);

    FineBucketTimer fine_timer(3, d.fine_bits);
    launch_t3_match_section_pair(
        keys, fk,
        d_meta_l_slice, section_l_row_start,
        d_meta_r_slice, section_r_row_start,
        d_sorted_xbits, d_sorted_mi,
        const_cast<uint64_t*>(d_offsets),
        const_cast<FineBucketOffset*>(d_fine_offsets),
        d.num_match_keys, d.num_buckets,
        params.k, params.num_section_bits,
        params.num_match_target_bits, d.fine_bits,
        d.target_mask, d.num_test_bits,
        d_out_pairings, d_out_count,
        capacity, d.l_count_max,
        bucket_begin, bucket_end,
        q);
    fine_timer.mark("match_fused", q);
}

void launch_t3_match_section_pair_split_range(
    uint8_t const* plot_id_bytes,
    T3MatchParams const& params,
    uint64_t const* d_meta_l_slice,
    uint32_t const* d_xbits_l_slice,
    uint64_t section_l_row_start,
    uint64_t const* d_meta_r_slice,
    uint32_t const* d_xbits_r_slice,
    uint32_t const* d_mi_r_slice,
    uint64_t section_r_row_start,
    T3PairingGpu* d_out_pairings,
    uint64_t* d_out_count,
    uint64_t capacity,
    void const* d_temp_storage,
    uint32_t bucket_begin,
    uint32_t bucket_end,
    sycl::queue& q)
{
    if (!plot_id_bytes) throw std::invalid_argument("invalid argument to launch wrapper");
    if (params.k < 18 || params.k > 28) throw std::invalid_argument("invalid argument to launch wrapper");
    if (params.strength < 2)            throw std::invalid_argument("invalid argument to launch wrapper");
    if (!d_temp_storage)                throw std::invalid_argument("invalid argument to launch wrapper");
    if (!d_meta_l_slice || !d_xbits_l_slice
        || !d_meta_r_slice || !d_xbits_r_slice || !d_mi_r_slice
        || !d_out_pairings || !d_out_count) throw std::invalid_argument("invalid argument to launch wrapper");

    T3Derived const d = derive_t3(params);

    if (bucket_end > d.num_buckets) throw std::invalid_argument("invalid argument to launch wrapper");
    if (bucket_end <= bucket_begin) return;

    constexpr int kThreads = 256;
    uint64_t const blocks_x_u64 = (d.l_count_max + kThreads - 1) / kThreads;
    if (blocks_x_u64 > UINT_MAX) throw std::invalid_argument("invalid argument to launch wrapper");

    auto const* d_offsets      = reinterpret_cast<uint64_t const*>(d_temp_storage);
    auto const* d_fine_offsets = fine_bucket_offset_ptr(d_temp_storage, d.bucket_bytes);

    AesHashKeys keys = make_keys(plot_id_bytes);
    FeistelKey  fk   = make_feistel_key(plot_id_bytes, params.k);

    FineBucketTimer fine_timer(3, d.fine_bits);
    launch_t3_match_section_pair_split(
        keys, fk,
        d_meta_l_slice, d_xbits_l_slice, section_l_row_start,
        d_meta_r_slice, d_xbits_r_slice, d_mi_r_slice, section_r_row_start,
        const_cast<uint64_t*>(d_offsets),
        const_cast<FineBucketOffset*>(d_fine_offsets),
        d.num_match_keys, d.num_buckets,
        params.k, params.num_section_bits,
        params.num_match_target_bits, d.fine_bits,
        d.target_mask, d.num_test_bits,
        d_out_pairings, d_out_count,
        capacity, d.l_count_max,
        bucket_begin, bucket_end,
        q);
    fine_timer.mark("match_fused", q);
}

void launch_t3_match(
    uint8_t const* plot_id_bytes,
    T3MatchParams const& params,
    uint64_t const* d_sorted_meta,
    uint32_t const* d_sorted_xbits,
    uint32_t const* d_sorted_mi,
    uint64_t t2_count,
    T3PairingGpu* d_out_pairings,
    uint64_t* d_out_count,
    uint64_t capacity,
    void* d_temp_storage,
    size_t* temp_bytes,
    sycl::queue& q)
{
    // Single-shot wrapper: prepare + one full-range match. Preserves the
    // original API for pool path, test mode, and parity-test callers.
    launch_t3_match_prepare(
        plot_id_bytes, params, d_sorted_mi, t2_count,
        d_out_count, d_temp_storage, temp_bytes, q);
    if (d_temp_storage == nullptr) return;  // size-query path

    T3Derived const d = derive_t3(params);
    launch_t3_match_range(
        plot_id_bytes, params,
        d_sorted_meta, d_sorted_xbits, d_sorted_mi, t2_count,
        d_out_pairings, d_out_count,
        capacity, d_temp_storage,
        /*bucket_begin=*/0, /*bucket_end=*/d.num_buckets, q);
}

} // namespace pos2gpu
