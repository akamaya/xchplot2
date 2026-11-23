// MultiGpuPipelineParallel.cpp — Phase 2.1c/d orchestrator. See header.

#include "host/MultiGpuPipelineParallel.hpp"

#include "gpu/SyclBackend.hpp"
#include "host/GpuBufferPool.hpp"   // host_pinned_reserve_check
#include "host/HostPinnedPool.hpp"
#include "host/Cancel.hpp"
#include "host/VramBudget.hpp"
#include "host/PoolSizing.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>

namespace pos2gpu {

namespace {

std::uint64_t default_vram_for_device(int id)
{
    auto const& devs = sycl_backend::usable_gpu_devices();
    if (id < 0 || static_cast<std::size_t>(id) >= devs.size()) {
        throw std::runtime_error(
            "select_pipeline_devices: device id " + std::to_string(id) +
            " out of range (have " + std::to_string(devs.size()) + " GPUs)");
    }
    return devs[id].get_info<sycl::info::device::global_mem_size>();
}

} // namespace

namespace {

// Heaviness order for N stages — the stage at index 0 of the returned
// vector is the heaviest (gets the largest-VRAM device); index N-1 is
// the lightest. Estimates without per-stage VRAM-peak measurement:
//   N=2: stage 0 (Xs+T1+T2) > stage 1 (T3+Frag) — Xs candidate stream
//        + CUB radix scratch dominate stage 0 in every tier.
//   N=3: stage 1 (T2 match+sort) > stage 0 (Xs+T1) > stage 2 (T3+Frag)
//        — T2 holds cap-sized device output AND CUB sort scratch
//        concurrently; Xs+T1 alone is lighter; T3+Frag is the
//        smallest.
// Update once per-stage VRAM peaks are measured.
std::vector<int> stage_heaviness_order(int N)
{
    if (N == 2) return {0, 1};
    if (N == 3) return {1, 0, 2};
    throw std::runtime_error(
        "select_pipeline_devices: heaviness order undefined for N=" +
        std::to_string(N) + " (only 2 and 3 are implemented)");
}

} // namespace

PipelineDeviceAssignment select_pipeline_devices(
    std::vector<int> const&                  device_ids,
    std::function<std::uint64_t(int)> const& vram_for_device)
{
    int const N = static_cast<int>(device_ids.size());
    if (N != 2 && N != 3) {
        throw std::runtime_error(
            "select_pipeline_devices: device_ids.size() must be 2 or 3 (got " +
            std::to_string(N) + ")");
    }
    for (int id : device_ids) {
        if (id < 0) throw std::runtime_error(
            "select_pipeline_devices: device ids must be non-negative");
    }

    auto vram_lookup = [&](int idx) { return vram_for_device(device_ids[idx]); };

    PipelineDeviceAssignment out;
    out.dev_ids.assign(N, -1);
    out.dev_vram_bytes.assign(N, 0);

    // Uniform-VRAM rig: keep caller's order (no reason to shuffle
    // identical cards, and the user's slot-order preference may matter).
    // Matches the 2-stage tie-keeps-order behaviour.
    bool all_vram_equal = true;
    auto const v0 = vram_lookup(0);
    for (int i = 1; i < N; ++i) {
        if (vram_lookup(i) != v0) { all_vram_equal = false; break; }
    }
    if (all_vram_equal) {
        for (int i = 0; i < N; ++i) {
            out.dev_ids[i]        = device_ids[i];
            out.dev_vram_bytes[i] = v0;
        }
        out.reordered = false;
    } else {
        // Heterogeneous: sort device indices VRAM-descending and
        // assign biggest-VRAM device to heaviest stage.
        std::vector<int> vram_rank(N);
        for (int i = 0; i < N; ++i) vram_rank[i] = i;
        std::stable_sort(vram_rank.begin(), vram_rank.end(),
                         [&](int a, int b) { return vram_lookup(a) > vram_lookup(b); });

        std::vector<int> const heaviness = stage_heaviness_order(N);
        for (int r = 0; r < N; ++r) {
            int const stage = heaviness[r];
            int const src   = vram_rank[r];
            out.dev_ids[stage]        = device_ids[src];
            out.dev_vram_bytes[stage] = vram_lookup(src);
        }
        for (int i = 0; i < N; ++i) {
            if (out.dev_ids[i] != device_ids[i]) { out.reordered = true; break; }
        }
    }

    // Backward-compat scalar fields for N=2 callers + existing parity
    // tests. Left default-zero for N>2; new code reads dev_ids[].
    if (N == 2) {
        out.dev_first             = out.dev_ids[0];
        out.dev_second            = out.dev_ids[1];
        out.dev_first_vram_bytes  = out.dev_vram_bytes[0];
        out.dev_second_vram_bytes = out.dev_vram_bytes[1];
    }
    return out;
}

PipelineDeviceAssignment select_pipeline_devices(std::vector<int> const& device_ids)
{
    return select_pipeline_devices(device_ids, default_vram_for_device);
}

PipelineParallelSplitResult run_pipeline_parallel_split(
    GpuPipelineConfig const&, std::vector<int> const&,
    std::vector<PipelineStageTier> const&)
{
    throw std::runtime_error(
        "pipeline-plot disabled: streaming GPU stages have been removed");
}

std::vector<PipelineParallelSplitResult> run_pipeline_parallel_batch(
    std::vector<GpuPipelineConfig> const&, std::vector<int> const&,
    int, std::vector<PipelineStageTier> const&,
    PipelineBatchPlotCallback, PipelineReadyCallback,
    std::vector<PipelineStageStats>*)
{
    throw std::runtime_error(
        "pipeline-plot disabled: streaming GPU stages have been removed");
}

} // namespace pos2gpu
