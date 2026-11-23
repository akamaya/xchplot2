#pragma once
// Experimental CUDA-only Xs -> T1 -> T2 -> T3 compute path.
// Does NOT construct a gplot file; it returns counts and a digest of sorted
// T3 fragments. The production plot-writing CLI is intentionally untouched.
#include <array>
#include <cstdint>
#include <vector>

namespace pos2gpu {
struct NativeCudaPipelineResult {
    uint64_t t1_count=0;
    uint64_t t2_count=0;
    uint64_t t3_count=0;
    uint64_t unique_t3_count=0;
    std::array<uint8_t,32> sha256_fragments{};
    std::array<uint8_t,32> sha256_unique_fragments{};
    double seconds=0;
};
// Scan CUDA devices without exposing CUDA Runtime types in host headers.
int native_cuda_rtx4070_super_ordinal();

NativeCudaPipelineResult native_cuda_run_k28s6(
    std::array<uint8_t,32> const& plot_id, int cuda_ordinal,
    // Optional: return sorted 56-bit T3 fragments to the existing
    // CPU .gplot writer. Default is compute/digest-only with no
    // multi-GiB host allocation.
    std::vector<uint64_t>* out_sorted_fragments = nullptr,
    // Default preserves standalone diagnostic digests. Set false on
    // the normal test-plot writer bridge: it already validates
    // sorted fragments and can be Full Proof checked on disk.
    bool compute_digests = true);
} // namespace pos2gpu
