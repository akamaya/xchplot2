#pragma once
// nvcc-only exact-index creation ABI, no SYCL headers.
#include "gpu/XsCandidateGpu.hpp"
#include <cstdint>
namespace pos2gpu {
void cuda_t1_bucket_offsets(XsCandidateGpu const*,uint64_t,int,uint32_t,uint64_t*);
void cuda_t1_fine_offsets(XsCandidateGpu const*,uint64_t const*,int,int,uint32_t,uint64_t*);
void cuda_t1_direct24(XsCandidateGpu const*,uint64_t const*,uint64_t,uint32_t,uint32_t*);
void cuda_t23_bucket_offsets(uint32_t const*,uint64_t,int,uint32_t,uint64_t*);
void cuda_t23_fine_offsets(uint32_t const*,uint64_t const*,int,int,uint32_t,uint32_t*);
void cuda_t23_direct20(uint32_t const*,uint32_t const*,uint32_t*,uint32_t);
}
