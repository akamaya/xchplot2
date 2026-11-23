#pragma once
// Native CUDA T1/T2/T3 Compact matcher ABI.
// Keep nvcc translation units free of SYCL/AdaptiveCpp headers.  The
// SYCL-to-CUDA fence belongs in the T*OffsetsSycl.cpp adapter.
#include "gpu/AesHashGpu.cuh"
#include "gpu/XsCandidateGpu.hpp"
#include "gpu/FeistelCipherGpu.cuh"
#include <cstdint>

namespace pos2gpu {

void cuda_match_t1(AesHashKeys keys, XsCandidateGpu const* xs,
    uint64_t const* bucket_offsets, uint32_t const* direct24, uint32_t const* t0,
    uint32_t num_match_keys, int k, int section_bits, int extra_rounds_bits,
    uint32_t target_mask, int test_bits, int info_bits,
    uint64_t* out_meta, uint32_t* out_mi, uint64_t* out_count,
    uint64_t capacity, uint64_t l_count_max, uint32_t bucket_begin, uint32_t bucket_end);

void cuda_match_t2(AesHashKeys keys, uint64_t const* meta,
    uint64_t const* bucket_offsets, uint32_t const* direct20, uint32_t const* t0,
    uint32_t num_match_keys, int k, int section_bits, uint32_t target_mask,
    int test_bits, int info_bits, int half_k,
    uint64_t* out_meta, uint32_t* out_mi, uint32_t* out_xbits,
    uint64_t* out_count, uint64_t capacity, uint64_t l_count_max,
    uint32_t bucket_begin, uint32_t bucket_end);

void cuda_match_t3(AesHashKeys keys, FeistelKey fk, uint64_t const* meta,
    uint32_t const* xbits, uint64_t const* bucket_offsets,
    uint32_t const* direct20, uint32_t const* t0, uint32_t num_match_keys,
    int k, int section_bits, uint32_t target_mask, int test_bits,
    uint64_t* out_fragments, uint64_t* out_count, uint64_t capacity,
    uint64_t l_count_max, uint32_t bucket_begin, uint32_t bucket_end);

} // namespace pos2gpu
