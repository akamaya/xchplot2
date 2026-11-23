#pragma once
// Native CUDA peer transport interface. This header does not include SYCL.
#include <cstdint>
namespace pos2gpu::detail {
void cuda_peer_count_u32(uint32_t const*,uint64_t,uint32_t,int,int,uint32_t*);
void cuda_peer_count_u64(uint64_t const*,uint64_t,uint32_t,int,int,uint32_t*);
void cuda_peer_scatter_u32_u32(uint32_t const*,uint32_t const*,uint64_t,uint32_t,int,int,uint32_t const*,uint32_t*,uint32_t*,uint32_t*);
void cuda_peer_scatter_u32_u64(uint32_t const*,uint64_t const*,uint64_t,uint32_t,int,int,uint32_t const*,uint32_t*,uint32_t*,uint64_t*);
void cuda_peer_scatter_u32_u64u32(uint32_t const*,uint64_t const*,uint32_t const*,uint64_t,uint32_t,int,int,uint32_t const*,uint32_t*,uint32_t*,uint64_t*,uint32_t*);
void cuda_peer_scatter_u64_keys(uint64_t const*,uint64_t,uint32_t,int,int,uint32_t const*,uint32_t*,uint64_t*);
}
