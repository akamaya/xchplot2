// GpuPipeline.cu — orchestrates Xs → T1 → T2 → T3 on the device, with
// CUB radix sort between phases (each phase consumes sorted-by-match_info
// input). Final T3 output is sorted by proof_fragment (low 2k bits) to
// match pos2-chip Table3Constructor::post_construct_span.
//
// Two overloads live here:
//   run_gpu_pipeline(cfg)       — transient pool, one-shot.
//   run_gpu_pipeline(cfg, pool) — shared pool, batch-friendly. This is the
//                                 real implementation; the one-shot form
//                                 just wraps it in a temporary pool.

#include "host/GpuPipeline.hpp"
#include "host/GpuBufferPool.hpp"
#include "host/T2XbitsReuse.hpp"
#include "host/XsPairAAlias.hpp"
#include "host/VramBudget.hpp"
#include "host/PoolSizing.hpp"
#include "host/HostGuard.hpp"      // pinned-host redzones — see header

#include "gpu/AesGpu.cuh"
#include "gpu/FineBucketConfig.hpp"
#include "gpu/XsKernel.cuh"
#include "gpu/XsKernels.cuh"   // launch_xs_gen / launch_xs_pack (stage 4e)
#include "gpu/T1Kernel.cuh"
#include "gpu/T2Kernel.cuh"
#include "gpu/T3Kernel.cuh"
#include "gpu/PipelineKernels.cuh"
#include "gpu/Sort.cuh"
#include "gpu/SyclBackend.hpp"

#include <sycl/sycl.hpp>


#include <algorithm>
#include <chrono>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace pos2gpu {

namespace {


// Pooled path: the match-count validator below is shared by T1.
// Sanity-check t1_count after T1 match. Healthy plots produce ~2^k
// entries; anything below total_xs/64 (= 2^(k-6)) — let alone literal
// zero — points at kernel correctness on the device, not a VRAM
// shortfall. Catching this here surfaces a clear diagnostic instead of
// letting downstream sort-scratch alloc fail with the misleading
// "Card likely too small" message. Two AMD/HIP cases produce 0 T1
// matches at k=28: the gfx1013/RDNA1 community spoof on a W5700, and
// AdaptiveCpp's generic SSCP JIT on the same RDNA1 silicon (the JIT
// path is theoretically more compatible than the AOT spoof but has
// been observed to miscompile the matcher). Only the OOM further down
// was visible before this check.
inline void validate_t1_count(uint64_t t1_count, int k)
{
    uint64_t const min_plausible = (1ULL << k) >> 6;
    if (t1_count >= min_plausible) return;

    // Did the kernels run at all? A device that faulted mid-run reaches this
    // check with the same t1_count == 0 as a device that miscompiled the
    // matcher, and the advice for the two is opposite: one is a driver/device
    // problem the parity tests cannot reproduce, the other is exactly what
    // they exist to catch. The async error count decides it.
    if (unsigned const n = async_error_count(); n > 0) {
        std::string const first = first_async_error();
        throw std::runtime_error(
            "T1 match produced " + std::to_string(t1_count) + " entries, and " +
            std::to_string(n) + " asynchronous backend error(s) were reported "
            "during the run. The kernels did not compute a wrong answer — they "
            "did not execute. This is a driver or device fault, NOT a codegen "
            "or parity problem, and the parity tests will not reproduce it.\n"
            "  First backend error: " + first + "\n"
            "  Reading Level Zero codes: 0x70000001 (ze:1879048193) is "
            "DEVICE_LOST and is the ROOT event. The 0x70000003 "
            "(ze:1879048195) OUT_OF_DEVICE_MEMORY storm that follows it is a "
            "dead context reporting on every subsequent call — it is not a "
            "memory shortage, and chasing it wastes time.\n"
            "  DEVICE_LOST part-way through a run usually means a kernel "
            "outran the GPU's job/preemption timeout and the driver reset the "
            "engine. Check `dmesg` for xe/i915 reset or hangcheck lines. On "
            "the Xe driver the limit is "
            "/sys/class/drm/card*/device/tile0/gt0/engines/*/job_timeout_ms "
            "(and preempt_timeout_us); i915 uses "
            "i915.enable_hangcheck / preempt_timeout_ms. Re-running at a "
            "smaller k shortens every kernel and is the fastest way to confirm "
            "the diagnosis: if a small k completes and k=28 does not, it is "
            "the timeout, not the code.");
    }

    throw std::runtime_error(
        "T1 match produced " + std::to_string(t1_count) + " entries "
        "(expected ~2^" + std::to_string(k) + " = " +
        std::to_string(1ULL << k) + " for k=" + std::to_string(k) +
        "). No asynchronous backend errors were reported, so the kernels ran "
        "and returned wrong results — a device codegen issue, not a VRAM "
        "shortfall. On AMD/HIP this most often means the "
        "AdaptiveCpp target produced wrong output for the actual gfx "
        "ISA — either the gfx1013/RDNA1 community AOT spoof or the "
        "generic SSCP JIT path on an unvalidated card. Build the "
        "parity tests via cmake and verify on this device: "
        "sycl_g_x_parity, sycl_sort_parity, sycl_bucket_offsets_parity, "
        "sycl_t1_parity. The first three exercise individual kernels at "
        "small N; sycl_t1_parity runs the full T1 matcher against the "
        "pos2-chip CPU reference and is the closest reproducer of the "
        "k=28 failure. README's 'Community-tested, not parity-validated' "
        "caveat applies.");
}

} // namespace

namespace {

void drain_fragment_sink(sycl::queue& q, uint64_t const* source, uint64_t count,
                         bool source_on_host, GpuFragmentSink const& sink)
{
    if (source_on_host)
    {
        sink.drain(count, [&](std::span<uint64_t> tile, uint64_t offset)
        {
            std::memcpy(tile.data(), source + offset, tile.size_bytes());
        });
        return;
    }
    // The queue is out of order. Finish the source's final sort once, then
    // wait only on each transfer's event while CPU consumption uses the other
    // half of the same pinned allocation. At most one D2H copy is in flight,
    // preserving the Level Zero shared-copy boundary used by spill staging.
    q.wait_and_throw();
    sink.drain_overlapped(count, [&](std::span<uint64_t> tile, uint64_t offset)
    {
        return q.memcpy(tile.data(), source + offset, tile.size_bytes());
    }, [&] { q.wait_and_throw(); });
}

GpuPipelineResult run_gpu_pipeline_impl(GpuPipelineConfig const& cfg,
                                        GpuBufferPool& pool,
                                        int pinned_index,
                                        GpuFragmentSink const* sink,
                                        GpuDeviceFragmentOutput const* device_output)
{

    sycl::queue& q = sycl_backend::queue();
    // Grant the T2/T3 two-phase match its VRAM budget for this queue before
    // any match runs. Nothing else bounds that allocation, and it is not
    // covered by the pool sizing — see SyclBackend.hpp.
    sycl_backend::set_twophase_budget(q, cfg.twophase_budget_bytes);
    if (cfg.testnet) throw std::invalid_argument("PoS2 1.0 removes testnet-specific plots");
    if (cfg.k < 18 || cfg.k > 28 || (cfg.k & 1) != 0) {
        throw std::runtime_error("k must be even in [18, 28]");
    }
    if (cfg.strength < 2) {
        throw std::runtime_error("strength must be >= 2");
    }
    if (pool.k != cfg.k || pool.strength != cfg.strength
        || pool.testnet != cfg.testnet)
    {
        throw std::runtime_error(
            "GpuBufferPool was sized for different (k, strength, testnet)");
    }
    if (pinned_index < 0 || pinned_index >= GpuBufferPool::kNumPinnedBuffers) {
        throw std::runtime_error(
            "pinned_index must be in [0, GpuBufferPool::kNumPinnedBuffers)");
    }

    uint64_t const total_xs = pool.total_xs;
    uint64_t const cap      = pool.cap;

    constexpr int kThreads = 256;
    auto blocks = [&](uint64_t n) {
        return unsigned((n + kThreads - 1) / kThreads);
    };

    // ---- pool aliases ----
    // d_pair_a carries the "current phase match output": T1, then T2, then T3.
    // d_pair_b carries the "current phase sort output": sorted T1, sorted T2,
    // then final uint64_t fragments. Each subsequent phase's output overwrites
    // the previous (consumed) contents in the same slot.
    XsCandidateGpu* d_xs             = static_cast<XsCandidateGpu*>(pool.d_storage);
    // d_pair_a-derived aliases (d_t1_meta, d_t1_mi, d_t2_meta, d_t2_mi,
    // d_t2_xbits, d_t3) are NOT declared here. They're declared inside
    // the Xs phase block below, right after pool.ensure_pair_a()
    // performs the lazy malloc_device for d_pair_a. Deferring that
    // alloc until after Xs gen has been submitted to the queue lets
    // the ~400-500 ms CPU-side malloc_device overlap with Xs's
    // ~750 ms GPU execution — saves ~400-500 ms off first-plot wall;
    // batch plots 2+ hit ensure_pair_a's cached-pointer fast path
    // so the alloc cost is paid exactly once per pool.
    //
    // d_pair_b-derived aliases stay up here because d_pair_b is
    // eager-allocated by the pool ctor: Xs gen needs it as scratch
    // from the start of the pipeline.
    uint64_t*       d_t1_meta_sorted  = static_cast<uint64_t*>      (pool.d_pair_b);
    uint64_t*       d_t2_meta_sorted  = static_cast<uint64_t*>      (pool.d_pair_b);
    uint32_t*       d_t2_xbits_sorted = reinterpret_cast<uint32_t*>(
        static_cast<uint8_t*>(pool.d_pair_b) + pool.cap * sizeof(uint64_t));
    uint64_t*       d_frags_out       = static_cast<uint64_t*>      (pool.d_pair_b);

    uint64_t*       d_count        = pool.d_counter;
    // Xs phase needs ~3.22 GB scratch at k=28 in split-keys_a mode
    // (3 × total_xs × u32 + cub). Without C, idle d_pair_b hosts the
    // scratch. With C, d_pair_a hosts Xs scratch before T1 match, and
    // pair_b shrinks to 8*cap.
    //
    // Split-keys_a: the Xs sort's keys_a (total_xs · u32 = 1 GiB at
    // k=28) lives in d_storage's tail — bytes [total_xs·8, storage_bytes)
    // which is idle during Xs gen+sort. The final pack phase writes
    // d_storage[0..total_xs·8) only, leaving keys_a's memory region
    // undisturbed (and its contents unread after the sort anyway, so
    // the overlap on T1/T2/T3-sort aliases in d_storage after pack is
    // a pure write-without-read of stale bytes). Saves ~1 GiB off the
    // pair_b xs-scratch region — see GpuBufferPool.cpp for sizing.
    void* const d_xs_split_keys_a = static_cast<uint8_t*>(pool.d_storage)
                                    + pool.total_xs * sizeof(XsCandidateGpu);
    // C: Xs scratch borrows pair_a before T1 writes match output.
    // pair_a is allocated here rather than after launching Xs; the size
    // planner enables C only when the full split-Xs scratch fits.
    void*           d_xs_temp      = pool.compact_d ? nullptr
        : pool.xs_pair_a_alias ? pool.ensure_pair_a() : pool.d_pair_b;
    void*           d_sort_scratch = pool.d_sort_scratch;
    // Lazy pinned-host alloc: skips ~600 ms × (kNumPinnedBuffers-1)
    // on single-plot runs (only slot 0 gets allocated). See
    // GpuBufferPool::ensure_pinned header comment for rationale.
    uint64_t*       h_pinned_t3    = sink || device_output ? nullptr : pool.ensure_pinned(pinned_index);
    // T1/T2/T3 exact indexes and radix sorting take turns using the
    // same dedicated scratch allocation. Nothing aliases live d_storage.
    auto select_match_temp = [&](size_t required) -> void* {
        if (required > pool.sort_scratch_bytes)
            throw std::runtime_error("Match index exceeds pooled scratch");
        return d_sort_scratch;
    };

    // Sort key/val arrays alias d_storage. Safe because Xs is fully consumed
    // by T1 match (stream-synchronised) before we enter T1 sort.
    //
    // Only three slots live here — keys_out, vals_in, vals_out. The
    // sort's keys_input is always the SoA match-info stream from
    // d_pair_a (d_t1_mi / d_t2_mi), so the fourth slot that would
    // have hosted "d_keys_in" is neither allocated nor used. See
    // GpuBufferPool.cpp for the matching storage_bytes shrink.
    auto     storage_u32 = static_cast<uint32_t*>(pool.d_storage);
    uint32_t* d_keys_out = storage_u32 + 0 * cap;
    uint32_t* d_vals_in  = storage_u32 + 1 * cap;
    uint32_t* d_vals_out = storage_u32 + 2 * cap;
    // T2 sort consumes vals_in before the permute. When B is enabled,
    // reuse that dead slot as the sorted T2 xbits which T3 later reads.
    if (pool.t2_xbits_reuse)
        d_t2_xbits_sorted = d_vals_in;

    // ---- per-phase wall-time profiling ----
    // Enabled when either cfg.profile is set (xchplot2 -P / --profile) or
    // POS2GPU_PHASE_TIMING=1 is in the env. Each phase's wall is measured
    // around q.wait()s so launches actually drain to the device before the
    // next start sample — adds a sync point but gives an honest breakdown.
    // When disabled, begin/end/report are early-out and add ~zero cost.
    bool const phase_timing = cfg.profile || [] {
        char const* v = std::getenv("POS2GPU_PHASE_TIMING");
        return v && v[0] == '1';
    }();
    using phase_clock = std::chrono::steady_clock;
    std::vector<std::pair<char const*, phase_clock::time_point>> phase_starts;
    std::vector<std::pair<char const*, double>>                  phase_records;
    // XCHPLOT2_HOST_GUARD: track the current phase label whether or not
    // phase timing is on, so a redzone report names the phase that broke
    // the buffer and not just the buffer. See HostGuard.hpp.
    char const* guard_phase = "(pipeline start)";
    auto begin_phase = [&](char const* label) -> int {
        if (pos2gpu::host_guard_on()) guard_phase = label;
        if (!phase_timing) return -1;
        q.wait();
        phase_starts.emplace_back(label, phase_clock::now());
        return static_cast<int>(phase_starts.size() - 1);
    };
    auto end_phase = [&](int idx) {
        // Drain before checking: a kernel still in flight has not done its
        // damage yet, and a guard that samples too early reads clean.
        if (pos2gpu::host_guard_on()) {
            q.wait();
            pos2gpu::host_guard_check(guard_phase);
        }
        if (idx < 0) return;
        q.wait();
        auto const t1 = phase_clock::now();
        auto const& [name, t0] = phase_starts[idx];
        double const ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        phase_records.emplace_back(name, ms);
    };
    auto report_phases = [&]() {
        if (!phase_timing || phase_records.empty()) return;
        double total = 0.0;
        for (auto const& [_n, ms] : phase_records) total += ms;
        std::fprintf(stderr, "[phase-timing]");
        for (auto const& [name, ms] : phase_records) {
            std::fprintf(stderr, " %s=%.1fms(%.0f%%)",
                name, ms, total > 0.0 ? 100.0 * ms / total : 0.0);
        }
        std::fprintf(stderr, " total=%.1fms\n", total);
    };

    // ---------- Phase Xs ----------
    size_t xs_temp_bytes = 0;
    launch_construct_xs(cfg.plot_id.data(), cfg.k, cfg.testnet,
                              nullptr, nullptr, &xs_temp_bytes, q,
                              d_xs_split_keys_a);
    int p_xs = begin_phase("Xs gen+sort");
    // Xs phase events stubbed in slice 17b — pass nullptr for the (no-op)
    // profiling event slots. The launch_construct_xs_profiled signature still
    // accepts cudaEvent_t for API compatibility but ignores the values.
    if (pool.compact_d) {
        launch_construct_xs_compact_d(
            cfg.plot_id.data(), cfg.k, cfg.testnet, d_xs,
            pool.d_storage, pool.storage_bytes, pool.d_pair_b,
            pool.pair_b_bytes, q);
    } else {
        launch_construct_xs_profiled(cfg.plot_id.data(), cfg.k, cfg.testnet,
                                           d_xs, d_xs_temp, &xs_temp_bytes,
                                           nullptr, nullptr, q,
                                           d_xs_split_keys_a);
    }
    // Both C and D require the Xs pack to finish before T1 match.
    // The CUB wrapper only synchronizes the radix sort BEFORE the SYCL
    // pack kernel; an out-of-order SYCL queue must be explicitly drained.
    if (pool.xs_pair_a_alias) q.wait_and_throw();
    // Overlap d_pair_a's lazy malloc_device (~400-500 ms for 4.36 GB at
    // k=28) with Xs gen's GPU execution. In production
    // (POS2GPU_PHASE_TIMING unset), launch_construct_xs_profiled returns
    // immediately with the kernel in-flight on the queue; this CPU-side
    // alloc then runs in parallel and its wall is hidden behind Xs's
    // ~750 ms GPU work. In phase_timing mode xs-timing's internal
    // q.waits serialise Xs first, then this alloc pays full wall — a
    // diagnostic-mode trade-off.
    void* const d_pair_a_raw = pool.ensure_pair_a();
    end_phase(p_xs);

    // d_pair_a-derived aliases, now that the lazy alloc has resolved.
    // Same layout as the old eager version — just computed from the
    // local d_pair_a_raw instead of pool.d_pair_a so there's no
    // confusion about when the pointer became valid.
    //
    // T1 match output is SoA, carved out of d_pair_a. Layout: meta[cap]
    // (cap·8 B) then mi[cap] (cap·4 B). Total cap·12 B, fits in d_pair_a's
    // cap·16 B budget.
    uint64_t*     d_t1_meta = static_cast<uint64_t*>(d_pair_a_raw);
    uint32_t*     d_t1_mi = pool.compact_d
        ? reinterpret_cast<uint32_t*>(
            static_cast<uint8_t*>(pool.d_storage) +
            total_xs * sizeof(XsCandidateGpu))
        : reinterpret_cast<uint32_t*>(
            static_cast<uint8_t*>(d_pair_a_raw) + pool.cap * sizeof(uint64_t));
    // T2 match output is SoA, carved out of d_pair_a. Layout: meta[cap]
    // (cap·8 B), then mi[cap] (cap·4 B), then xbits[cap] (cap·4 B). Total
    // cap·16 B, matching d_pair_a's size.
    uint64_t*     d_t2_meta  = static_cast<uint64_t*>(d_pair_a_raw);
    // C: store T2 match_info in storage's idle second u32 slot,
    // leaving pair_a's last u32 slot for unsorted xbits.
    uint32_t*     d_t2_mi    = pool.xs_pair_a_alias
        ? d_vals_in
        : reinterpret_cast<uint32_t*>(
            static_cast<uint8_t*>(d_pair_a_raw) + pool.cap * sizeof(uint64_t));
    uint32_t*     d_t2_xbits = pool.compact_d
        ? d_vals_out  // storage slot 2; parked in pair_b after T2 match
        : pool.xs_pair_a_alias
        ? reinterpret_cast<uint32_t*>(
            static_cast<uint8_t*>(d_pair_a_raw) +
            pool.cap * sizeof(uint64_t))
        : pool.t2_xbits_reuse
            ? reinterpret_cast<uint32_t*>(
                static_cast<uint8_t*>(pool.d_pair_b) +
                t2_xbits_unsorted_offset(static_cast<size_t>(cap)))
            : reinterpret_cast<uint32_t*>(
                static_cast<uint8_t*>(d_pair_a_raw) +
                pool.cap * (sizeof(uint64_t) + sizeof(uint32_t)));
    T3PairingGpu* d_t3       = static_cast<T3PairingGpu*>(d_pair_a_raw);

    // ---------- Phase T1 ----------
    auto t1p = make_t1_params(cfg.k, cfg.strength);
    size_t t1_temp_bytes = 0;
    launch_t1_match(cfg.plot_id.data(), t1p, d_xs, total_xs,
                          nullptr, nullptr, d_count, cap,
                          nullptr, &t1_temp_bytes, q);
    void* const t1_match_temp = select_match_temp(t1_temp_bytes);
    // The queue is out-of-order: wait so the counter zeroing can't be
    // scheduled after the match kernel that increments it.
    q.memset(d_count, 0, sizeof(uint64_t)).wait();
    int p_t1 = begin_phase("T1 match");
    launch_t1_match(cfg.plot_id.data(), t1p, d_xs, total_xs,
                          d_t1_meta, d_t1_mi, d_count, cap,
                          t1_match_temp, &t1_temp_bytes, q);
    end_phase(p_t1);

    // No explicit sync: the next cudaMemcpy (non-async, default stream)
    // implicitly drains prior stream work before the host reads t1_count.
    uint64_t t1_count = 0;
    q.memcpy(&t1_count, d_count, sizeof(uint64_t)).wait();
    if (t1_count > cap) throw std::runtime_error("T1 overflow");
    validate_t1_count(t1_count, cfg.k);


    // Sort T1 by match_info (low k bits). d_storage is now repurposed
    // as (keys_in, keys_out, vals_in, vals_out), Xs having been fully
    // consumed by T1 match above. T1 match emits match_info in a SoA
    // stream (d_t1_mi), so we feed that directly to CUB as the sort key
    // input rather than extracting from a packed struct.
    int p_t1_sort = begin_phase("T1 sort");
    {
        size_t sort_bytes = pool.sort_scratch_bytes;
        if (pool.compact_d) {
            // No cap-sized index arrays: carry T1's u64 meta directly.
            // Input: storage tail + pair_a; output: storage slot 0 + pair_b.
            launch_sort_pairs_u32_u64(
                d_sort_scratch, sort_bytes, d_t1_mi, d_keys_out,
                d_t1_meta, d_t1_meta_sorted, t1_count, 0, cfg.k, q);
        } else {
            launch_init_u32_identity(d_vals_in, t1_count, q);
            launch_sort_pairs_u32_u32(
                d_sort_scratch, sort_bytes,
                d_t1_mi, d_keys_out, d_vals_in, d_vals_out,
                t1_count, /*begin_bit=*/0, /*end_bit=*/cfg.k, q);
            launch_gather_u64(d_t1_meta, d_vals_out, d_t1_meta_sorted, t1_count, q);
        }
    }
    end_phase(p_t1_sort);

    // ---------- Phase T2 ----------
    // Sorted T1 = (d_t1_meta_sorted: uint64 meta, d_keys_out: uint32 match_info).
    // No AoS struct anymore — saves 33 % of sorted-T1 bandwidth on both the
    // permute write and the match-kernel hot path.
    auto t2p = make_t2_params(cfg.k, cfg.strength);
    size_t t2_temp_bytes = 0;
    launch_t2_match(cfg.plot_id.data(), t2p, nullptr, nullptr, t1_count,
                          nullptr, nullptr, nullptr, d_count, cap,
                          nullptr, &t2_temp_bytes, q);
    void* const t2_match_temp = select_match_temp(t2_temp_bytes);
    q.memset(d_count, 0, sizeof(uint64_t)).wait();
    int p_t2 = begin_phase("T2 match");
    launch_t2_match(cfg.plot_id.data(), t2p, d_t1_meta_sorted, d_keys_out, t1_count,
                          d_t2_meta, d_t2_mi, d_t2_xbits, d_count, cap,
                          t2_match_temp, &t2_temp_bytes, q);
    end_phase(p_t2);

    uint64_t t2_count = 0;
    q.memcpy(&t2_count, d_count, sizeof(uint64_t)).wait();
    if (t2_count > cap) throw std::runtime_error("T2 overflow");

    int p_t2_sort = begin_phase("T2 sort");
    {
        // T2 match emitted match_info as a SoA stream (d_t2_mi) — feed
        // it straight into CUB as the sort key input rather than
        // re-extracting from a packed struct. vals_in just needs a
        // 0..n-1 identity fill.
        // C: d_vals_in holds T2 mi, so it cannot also hold the
        // identity sequence. The sorted T1 metadata in pair_b is dead
        // after T2 match: borrow its first 4*cap bytes for identity,
        // then overwrite with sorted T2 meta after CUB has completed.
        uint32_t* t2_vals_in = pool.xs_pair_a_alias
            ? static_cast<uint32_t*>(pool.d_pair_b) : d_vals_in;
        if (pool.compact_d) {
            // The T2 outputs occupy all 3 storage slots at match exit.
            // Park xbits in pair_b's upper half now that sorted T1 meta
            // is dead. Free storage slot 2 for CUB's sorted indices.
            uint32_t* parked_xbits = static_cast<uint32_t*>(pool.d_pair_b) + cap;
            q.memcpy(parked_xbits, d_t2_xbits,
                     t2_count * sizeof(uint32_t)).wait_and_throw();
            launch_init_u32_identity(t2_vals_in, t2_count, q);
            size_t sort_bytes = pool.sort_scratch_bytes;
            launch_sort_pairs_u32_u32(
                d_sort_scratch, sort_bytes, d_t2_mi, d_keys_out,
                t2_vals_in, d_vals_out, t2_count, 0, cfg.k, q);
            // Gather xbits before overwriting pair_b with sorted meta;
            // T2 mi in storage slot 1 is dead after the radix sort.
            launch_gather_u32(parked_xbits, d_vals_out,
                              d_t2_xbits_sorted, t2_count, q);
            launch_gather_u64(d_t2_meta, d_vals_out,
                              d_t2_meta_sorted, t2_count, q);
        } else {
            launch_init_u32_identity(t2_vals_in, t2_count, q);
            size_t sort_bytes = pool.sort_scratch_bytes;
            launch_sort_pairs_u32_u32(
                d_sort_scratch, sort_bytes,
                d_t2_mi, d_keys_out, t2_vals_in, d_vals_out,
                t2_count, 0, cfg.k, q);
            launch_permute_t2(d_t2_meta, d_t2_xbits, d_vals_out,
                              d_t2_meta_sorted, d_t2_xbits_sorted, t2_count, q);
        }
    }
    end_phase(p_t2_sort);

    // ---------- Phase T3 ----------
    // d_keys_out now holds the T2 sorted match_info (T1's was overwritten by
    // the T2 sort above) — pass as the slim stream for binary search in T3.
    auto t3p = make_t3_params(cfg.k, cfg.strength);
    size_t t3_temp_bytes = 0;
    launch_t3_match(cfg.plot_id.data(), t3p,
                          d_t2_meta_sorted, d_t2_xbits_sorted,
                          nullptr, t2_count,
                          d_t3, d_count, cap,
                          nullptr, &t3_temp_bytes, q);
    void* const t3_match_temp = select_match_temp(t3_temp_bytes);
    q.memset(d_count, 0, sizeof(uint64_t)).wait();
    int p_t3 = begin_phase("T3 match + Feistel");
    launch_t3_match(cfg.plot_id.data(), t3p,
                          d_t2_meta_sorted, d_t2_xbits_sorted,
                          d_keys_out, t2_count,
                          d_t3, d_count, cap,
                          t3_match_temp, &t3_temp_bytes, q);
    end_phase(p_t3);

    uint64_t t3_count = 0;
    q.memcpy(&t3_count, d_count, sizeof(uint64_t)).wait();
    if (t3_count > cap) throw std::runtime_error("T3 overflow");

    // Sort T3 by proof_fragment (low 2k bits). T3PairingGpu is just a
    // uint64_t, so reinterpret the d_pair_a slot directly.
    uint64_t* d_frags_in = reinterpret_cast<uint64_t*>(d_t3);
    // This gate runs after the next member's match work, so its generation can
    // overlap the prior member's CPU drain. The independent destination cannot
    // alias pool tables: the next Xs phase immediately reuses d_pair_b.
    if (device_output) d_frags_out = device_output->destination(t3_count).data();
    int p_t3_sort = begin_phase("T3 sort");
    {
        size_t sort_bytes = pool.sort_scratch_bytes;
        launch_sort_keys_u64(
            d_sort_scratch, sort_bytes,
            d_frags_in, d_frags_out,
            t3_count, /*begin_bit=*/0, /*end_bit=*/2 * cfg.k, q);
    }
    end_phase(p_t3_sort);

    // ---------- D2H ----------
    int p_d2h = begin_phase("D2H copy T3 fragments (pinned)");
    GpuPipelineResult result;
    result.t1_count = t1_count;
    result.t2_count = t2_count;
    result.t3_count = t3_count;

    if (device_output) {
        // The caller may submit copies on a distinct queue in this context.
        // Establish source readiness before returning the device destination.
        q.wait_and_throw();
    } else if (sink) {
        drain_fragment_sink(q, d_frags_out, t3_count, false, *sink);
    } else if (t3_count > 0) {
        q.memcpy(h_pinned_t3, d_frags_out, sizeof(uint64_t) * t3_count);
        q.wait();
    }
    end_phase(p_d2h);

    if (!sink && !device_output && t3_count > 0) {
        // Borrow: caller (batch producer) promises to finish consuming this
        // pinned slot before reusing it for another plot.
        result.external_fragments_ptr   = h_pinned_t3;
        result.external_fragments_count = t3_count;
    }

    // Xs gen / sort per-phase timings stubbed in slice 17b — see profiling
    // notes above.

    // Release d_pair_a so it isn't held between plots in a batch run.
    // At ~5 ms/alloc on amdgcn (sycl::malloc_device effectively just
    // reserves virtual address space), the per-plot realloc cost is
    // below noise, but freeing 4.36 GB during the inter-plot gap means
    // the pool path is viable on cards with ~7-8 GiB free that would
    // otherwise hit InsufficientVramError and fall back to streaming.
    // The final q.wait() inside the D2H block above has already drained
    // T3 sort so the buffer is safe to free.
    // Candidate C must allocate pair_a before Xs starts. Keep it across
    // batch plots to avoid a malloc_device cost ahead of every Xs launch.
    // With C off, preserve the existing inter-plot release policy.
    if (!pool.xs_pair_a_alias) pool.release_pair_a();

    report_phases();
    return result;
}

} // namespace

GpuPipelineResult run_gpu_pipeline(GpuPipelineConfig const& cfg,
                                   GpuBufferPool& pool,
                                   int pinned_index)
{
    return run_gpu_pipeline_impl(cfg, pool, pinned_index, nullptr, nullptr);
}

GpuPipelineResult run_gpu_pipeline(GpuPipelineConfig const& cfg,
                                   GpuBufferPool& pool,
                                   GpuFragmentSink const& sink)
{
    sink.validate();
    return run_gpu_pipeline_impl(cfg, pool, 0, &sink, nullptr);
}

GpuPipelineResult run_gpu_pipeline(GpuPipelineConfig const& cfg,
                                   GpuBufferPool& pool,
                                   GpuDeviceFragmentOutput const& output)
{
    output.validate();
    return run_gpu_pipeline_impl(cfg, pool, 0, nullptr, &output);
}

GpuPipelineResult run_gpu_pipeline(GpuPipelineConfig const& cfg)
{
    // Fixed k28/s6 profile: the pooled pipeline is the only execution path.
    // Let InsufficientVramError propagate with the required/free VRAM figures.
    if (char const* force = std::getenv("XCHPLOT2_STREAMING");
        force && force[0] == '1')
        throw std::invalid_argument(
            "XCHPLOT2_STREAMING is unsupported: streaming pipeline was removed");
    GpuBufferPool pool(cfg.k, cfg.strength, cfg.testnet);
    GpuPipelineResult r = run_gpu_pipeline(cfg, pool, /*pinned_index=*/0);
    if (r.external_fragments_ptr && r.external_fragments_count > 0) {
        r.t3_fragments_storage.resize(r.external_fragments_count);
        std::memcpy(r.t3_fragments_storage.data(), r.external_fragments_ptr,
                    sizeof(uint64_t) * r.external_fragments_count);
    }
    r.external_fragments_ptr = nullptr;
    r.external_fragments_count = 0;
    return r;
}

// Deleted streaming engine. These compatibility entry points are retained
// temporarily for separately-built legacy tests and tools. They never allocate
// device buffers, try a lower-VRAM tier, or produce plots.
namespace {
[[noreturn]] void streaming_removed()
{
    throw std::runtime_error(
        "GPU streaming pipeline removed: k28/s6 requires the pooled GPU "
        "layout; insufficient VRAM is a fatal error");
}
} // namespace

GpuPipelineResult run_gpu_pipeline_streaming(GpuPipelineConfig const&)
{
    streaming_removed();
}
GpuPipelineResult run_gpu_pipeline_streaming(GpuPipelineConfig const&,
                                             uint64_t*, size_t)
{
    streaming_removed();
}
GpuPipelineResult run_gpu_pipeline_streaming(GpuPipelineConfig const&,
                                             uint64_t*, size_t,
                                             StreamingPinnedScratch const&)
{
    streaming_removed();
}
GpuPipelineResult run_gpu_pipeline_streaming(GpuPipelineConfig const&,
                                             GpuFragmentSink const&,
                                             StreamingPinnedScratch const&)
{
    streaming_removed();
}

uint64_t* streaming_alloc_pinned_uint64(size_t count)
{
    // Throws rather than returning null: the callers treat null as "allocation
    // failed, fall back", but a host that is out of RAM has nothing to fall
    // back TO — every lower tier wants more host memory, not less. Failing
    // here with the reason beats failing later without one.
    size_t const bytes = count * sizeof(uint64_t);
    host_pinned_reserve_check(bytes, "streaming pinned u64");
    // See s_malloc_host_raw for the redzone rationale. These are the D2H
    // drain slots and the caller-provided scratch tables — the buffers
    // device kernels write into through a cursor, so the ones most worth
    // guarding.
    void* p = sycl::malloc_host(bytes + 2 * pos2gpu::host_guard_pad(),
                                sycl_backend::queue());
    if (!p) return nullptr;
    return static_cast<uint64_t*>(
        pos2gpu::host_guard_arm(p, bytes, "streaming pinned u64"));
}

uint32_t* streaming_alloc_pinned_uint32(size_t count)
{
    size_t const bytes = count * sizeof(uint32_t);
    host_pinned_reserve_check(bytes, "streaming pinned u32");
    void* p = sycl::malloc_host(bytes + 2 * pos2gpu::host_guard_pad(),
                                sycl_backend::queue());
    if (!p) return nullptr;  // nullptr on failure
    return static_cast<uint32_t*>(
        pos2gpu::host_guard_arm(p, bytes, "streaming pinned u32"));
}

uint64_t twophase_bytes_held()
{
    return sycl_backend::twophase_bytes_held(sycl_backend::queue());
}

void streaming_free_pinned_uint32(uint32_t* ptr)
{
    if (ptr) sycl::free(pos2gpu::host_guard_disarm(ptr, "free pinned u32"),
                        sycl_backend::queue());
}

void streaming_free_pinned_uint64(uint64_t* ptr)
{
    if (ptr) sycl::free(pos2gpu::host_guard_disarm(ptr, "free pinned u64"),
                        sycl_backend::queue());
}

void streaming_host_guard_check(char const* where)
{
    pos2gpu::host_guard_check(where);
}

void bind_current_device(int device_id)
{
    sycl_backend::set_current_device_id(device_id);
}

int gpu_device_count()
{
    try {
        return sycl_backend::get_gpu_device_count();
    } catch (...) {
        return 0;
    }
}

} // namespace pos2gpu
