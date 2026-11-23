#pragma once
// Experimental CUDA-only Xs -> T1 -> T2 -> T3 compute path.
// Does NOT construct a gplot file; it returns counts and a digest of sorted
// T3 fragments. The production plot-writing CLI is intentionally untouched.
#include <array>
#include <cstdint>

namespace pos2gpu {
struct NativeCudaPipelineResult {
    uint64_t t1_count=0;
    uint64_t t2_count=0;
    uint64_t t3_count=0;
    std::array<uint8_t,32> sha256_fragments{};
    double seconds=0;
};
NativeCudaPipelineResult native_cuda_run_k28s6(
    std::array<uint8_t,32> const& plot_id, int cuda_ordinal);
} // namespace pos2gpu
