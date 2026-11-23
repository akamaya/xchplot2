#pragma once
// Native CUDA histogram and atomic partition for two-level sort.
// The segmented inner radix sort continues to use existing CUDA CUB.
#include <cstdint>
namespace pos2gpu {
void cuda_two_level_histogram(uint32_t const* keys,uint32_t* histogram,
    uint64_t count,int high_bit_offset,int num_top_bits);
void cuda_two_level_partition(uint32_t const* keys,uint32_t const* vals,
    uint32_t* part_keys,uint32_t* part_vals,uint32_t* cursors,
    uint64_t count,int high_bit_offset,int num_top_bits);
}
