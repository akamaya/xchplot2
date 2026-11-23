#pragma once
// Native CUDA Xs gen + pack, no SYCL headers.
#include "gpu/AesHashGpu.cuh"
#include "gpu/XsCandidateGpu.hpp"
#include <cstdint>
namespace pos2gpu {
void cuda_xs_generate(AesHashKeys,uint32_t*,uint32_t*,uint64_t,uint64_t,int,uint32_t,uint32_t const*);
void cuda_xs_pack(uint32_t const*,uint32_t const*,XsCandidateGpu*,uint64_t);
}
