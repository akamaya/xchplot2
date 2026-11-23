#pragma once
// Native CUDA replacements for device-only pipeline primitives.
#include <cstdint>
namespace pos2gpu {
void cuda_init_u32_identity(uint32_t*,uint64_t);
void cuda_gather_u64(uint64_t const*,uint32_t const*,uint64_t*,uint64_t);
void cuda_gather_u32(uint32_t const*,uint32_t const*,uint32_t*,uint64_t);
void cuda_permute_t2(uint64_t const*,uint32_t const*,uint32_t const*,uint64_t*,uint32_t*,uint64_t);
void cuda_inverse_u32(uint32_t const*,uint32_t*,uint64_t);
void cuda_scatter_u64(uint64_t const*,uint32_t const*,uint64_t*,uint64_t);
void cuda_scatter_u32(uint32_t const*,uint32_t const*,uint32_t*,uint64_t);
void cuda_permute_t2_scatter(uint64_t const*,uint32_t const*,uint32_t const*,uint64_t*,uint32_t*,uint64_t);
void cuda_merge_pairs_stable_2way_u32_u32(uint32_t const*,uint32_t const*,uint64_t,
    uint32_t const*,uint32_t const*,uint64_t,uint32_t*,uint32_t*,uint64_t);
}
