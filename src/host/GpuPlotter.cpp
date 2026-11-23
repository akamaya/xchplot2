// GpuPlotter.cpp — orchestrates per-phase CPU/GPU strategy. For any
// phase not yet implemented on GPU, dispatches to the pos2-chip CPU
// reference (via run_cpu_plotter_to_fragments in PlotFileWriterParallel.cpp).
// We then call write_plot_file_parallel for the FSE compression and
// serialization, regardless of which path produced the fragments.
//
// Deliberately includes NO pos2-chip headers — those all live behind
// PlotFileWriterParallel.{hpp,cpp} so that one .cpp is the sole TU
// pulling soft_aes.hpp into the link.

#include "host/GpuPlotter.hpp"
#include "host/BatchPlotter.hpp"
#include "host/GpuPipeline.hpp"
#include "host/PlotFileWriterParallel.hpp"
#if defined(_WIN32) && defined(XCHPLOT2_NATIVE_CUDA_MATCH)
#include "gpu/NativeCudaStandalonePipeline.hpp"
#endif

#include <filesystem>
#include <chrono>
#include <algorithm>
#include <optional>
#include <cstdio>
#include <cstdint>
#include <array>
#include <iostream>
#include <span>
#include <memory>
#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <thread>
#include <utility>
#include <stdexcept>
#include <vector>

namespace pos2gpu {

namespace {

void warn_if_gpu_requested_but_unimplemented(GpuPlotOptions const& o)
{
    auto warn = [&](char const* phase) {
        std::cerr << "[xchplot2] WARNING: " << phase
                  << " GPU path not implemented yet — falling back to CPU.\n";
    };
    if (o.t1 == PhaseStrategy::Gpu) warn("T1");
    if (o.t2 == PhaseStrategy::Gpu) warn("T2");
    if (o.t3 == PhaseStrategy::Gpu) warn("T3");
}

} // namespace

std::string plot_to_file(GpuPlotOptions const& opts, std::string const& output_dir)
{
    auto const plot_started=std::chrono::steady_clock::now();
    if (opts.k < 18 || opts.k > 28 || (opts.k & 1) != 0) {
        throw std::runtime_error("k must be even and in [18, 28]");
    }
    if (opts.strength < 2 || opts.strength > 63) {
        throw std::runtime_error("strength must be in [2, 63]");
    }
    if (opts.testnet)
        throw std::invalid_argument("PoS2 1.0 removes testnet-specific plots; omit --testnet");
    if (!opts.raw && opts.plot_index != 0)
        throw std::invalid_argument("single-plot groups must start at plot index 0; use --raw for group members");
    // The independent native CUDA path is deliberately limited to
    // public, non-farmable test plots. Never accept a caller-provided
    // secret-bearing memo (including a farmer master SK) in this path.
    // Validate before GPU work or creating any output directory.
    if (opts.native_cuda_test) {
        if (!opts.memo.empty())
            throw std::invalid_argument(
                "native CUDA test is test-only: secret-bearing --memo is not supported");
        if (opts.k != 28 || opts.strength != 6)
            throw std::invalid_argument("native CUDA test supports only k28/s6");
    }


    // Build output filename. Caller may override via opts.out_name (used
    // by integrations). Default includes the group identity and parameters.
    std::string filename;
    if (!opts.out_name.empty()) {
        filename = opts.out_name;
    } else {
        std::string group_id_hex;
        group_id_hex.reserve(64);
        static char const hex[] = "0123456789abcdef";
        for (uint8_t b : opts.group_id) {
            group_id_hex += hex[b >> 4];
            group_id_hex += hex[b & 0xF];
        }
        filename = "plot_" + std::to_string(opts.k)
                 + "_" + std::to_string(opts.strength)
                 + "_" + std::to_string(opts.plot_index)
                 + "_" + std::to_string(opts.meta_group)
                 + "_" + group_id_hex
                 + (opts.raw ? ".plot2" : ".gplot");
    }

    std::filesystem::create_directories(output_dir);
    auto full_path = std::filesystem::path(output_dir) / filename;
    if (opts.native_cuda_test && std::filesystem::exists(full_path))
        throw std::runtime_error("native CUDA test refuses to overwrite existing .gplot: " +
                                 full_path.string());

    // Memo: caller-supplied bytes if present (real farmable plots), else
    // a 112-byte stub (test plots only — harvester will reject).
    std::vector<uint8_t> memo_bytes;
    if (!opts.memo.empty()) {
        memo_bytes = opts.memo;
    } else {
        memo_bytes.assign(32 + 48 + 32, 0);
    }
    if (memo_bytes.size() > 255) {
        throw std::runtime_error("memo too long (max 255 bytes; PlotFile uses uint8 length)");
    }
    BatchEntry entry;
    entry.k = opts.k;
    entry.strength = opts.strength;
    entry.plot_index = opts.plot_index;
    entry.meta_group = opts.meta_group;
    entry.raw = opts.raw;
    entry.group_id = opts.group_id;
    entry.plot_id = opts.plot_id;
    entry.memo = std::move(memo_bytes);
    entry.out_dir = output_dir;
    entry.out_name = filename;
    validate_batch_entry(entry);

    bool const all_gpu = (opts.t1 == PhaseStrategy::Gpu)
                      && (opts.t2 == PhaseStrategy::Gpu)
                      && (opts.t3 == PhaseStrategy::Gpu);

    if (opts.native_cuda_test && !all_gpu)
        throw std::invalid_argument("native CUDA test requires full GPU T1/T2/T3");
    if (!all_gpu) {
        warn_if_gpu_requested_but_unimplemented(opts);
    }

    // Either path produces a vector of ProofFragments we pass to the writer
    // as a span. GPU path owns via GpuPipelineResult::t3_fragments_storage;
    // CPU path owns via cpu_fragments below.
    GpuPipelineResult pr;
    std::vector<uint64_t> cpu_fragments;
    std::unique_ptr<NativeCudaPlotSession> native_session;
    std::span<uint64_t const> fragments;
    if (all_gpu && opts.native_cuda_test) {
#if defined(_WIN32) && defined(XCHPLOT2_NATIVE_CUDA_MATCH)
        // Deliberately diagnostic-only: the production pipeline and batch
        // worker still use the previously verified CUDA/SYCL hybrid.
        // Only the GPU computation is CUDA-only here. The existing,
        // well-tested CPU FSE/.gplot writer runs immediately afterward.
        int const ordinal = native_cuda_rtx4070_super_ordinal();
        // Original batch-mode contract: Writer borrows a pinned T3 output
        // slot from a persistent CUDA pool until file publication completes.
        native_session=std::make_unique<NativeCudaPlotSession>(ordinal);
        auto const frame=native_session->generate(opts.plot_id);
        fragments=frame.fragments;
        std::cerr << "[cuda-native-test] GPU T1=" << frame.stats.t1_count
                  << " T2=" << frame.stats.t2_count
                  << " T3=" << frame.stats.t3_count
                  << " compute+capture=" << frame.stats.seconds << "s"
                  << " transfer=pinned-direct\n";
#else
        throw std::runtime_error("native CUDA test requires Windows CUDA-only core build");
#endif
    } else if (all_gpu) {
        // Run the full pipeline on GPU; hand the sorted T3 fragments to
        // the CPU PlotFile writer for FSE compression and serialization.
        GpuPipelineConfig cfg;
        cfg.plot_id  = opts.plot_id;
        cfg.k        = opts.k;
        cfg.strength = opts.strength;
        cfg.testnet  = opts.testnet;
        cfg.profile  = opts.profile;
        pr = run_gpu_pipeline(cfg);
        fragments = pr.fragments();
        if (opts.verbose) {
            std::cerr << "[xchplot2] T1=" << pr.t1_count
                      << " T2=" << pr.t2_count
                      << " T3=" << pr.t3_count
                      << " (all on GPU)\n";
        }
    } else {
        cpu_fragments = run_cpu_plotter_to_fragments(
            opts.plot_id.data(),
            static_cast<uint8_t>(opts.k),
            static_cast<uint8_t>(opts.strength),
            opts.testnet ? uint8_t{1} : uint8_t{0},
            opts.verbose);
        fragments = std::span<uint64_t const>(cpu_fragments.data(),
                                              cpu_fragments.size());
    }

    // End-to-end plot duration is the performance metric: CUDA compute,
    // T3 D2H capture, FSE compression, NAS write, and publication.
    auto const writer_start=std::chrono::steady_clock::now();
    write_plot_file_parallel(full_path.string(), fragments, entry);
    if (opts.native_cuda_test) {
        auto const done=std::chrono::steady_clock::now();
        std::fprintf(stderr,"[plot-timing] writer=%.3f s full_plot=%.3f s\n",
            std::chrono::duration<double>(done-writer_start).count(),
            std::chrono::duration<double>(done-plot_started).count());
    }

    return full_path.string();
}


void plot_native_cuda_series(
    GpuPlotOptions const& opts, std::string const& output_dir, unsigned count)
{
#if defined(_WIN32) && defined(XCHPLOT2_NATIVE_CUDA_MATCH)
    // The test batch deliberately has NO secret memo input. Unlike the
    // generic farming batch it only writes dummy-memo single-member groups.
    if (!opts.native_cuda_test || !opts.memo.empty() || opts.raw ||
        opts.k!=28 || opts.strength!=6 || opts.plot_index!=0 ||
        opts.testnet || count<2 || count>100)
        throw std::invalid_argument(
            "native CUDA test batch requires k28/s6, 2..100 plots, "
            "index=0, no raw mode and NO private-key memo");
    if(!opts.out_name.empty() &&
       (std::filesystem::path(opts.out_name).has_parent_path() ||
        std::filesystem::path(opts.out_name).extension()!=".gplot"))
        throw std::invalid_argument(
            "native CUDA batch filename prefix must be a .gplot basename");

    struct Job {
        BatchEntry entry;
        std::string path;
    };
    std::vector<Job> jobs;
    jobs.reserve(count);
    std::filesystem::create_directories(output_dir);
    for(unsigned i=0;i<count;++i) {
        auto group_id=opts.group_id;
        // Deterministic, public per-member identity. Plot 0 retains the
        // seed group ID; plot i XORs a big-endian 64-bit index into its tail.
        // Each group is a single-member group (plot_index remains 0).
        for(unsigned byte=0;byte<8;++byte)
            group_id[31-byte]^=uint8_t((uint64_t(i)>>(byte*8))&255u);
        BatchEntry entry;
        entry.k=28;
        entry.strength=6;
        entry.plot_index=0;
        entry.meta_group=opts.meta_group;
        entry.raw=false;
        entry.group_id=group_id;
        entry.plot_id=plot_id_for_group(group_id,0,uint8_t(opts.meta_group));
        entry.memo.assign(112,0);
        entry.out_dir=output_dir;
        std::string const stem=opts.out_name.empty()
            ? std::string("native_cuda_test")
            : std::filesystem::path(opts.out_name).stem().string();
        entry.out_name=stem+"_"+std::to_string(i)+".gplot";
        validate_batch_entry(entry);
        auto path=(std::filesystem::path(output_dir)/entry.out_name).string();
        if(std::filesystem::exists(path))
            throw std::runtime_error("native CUDA batch refuses existing plot: "+path);
        jobs.push_back(Job{std::move(entry),std::move(path)});
    }

    using Clock=std::chrono::steady_clock;
    auto const started=Clock::now();
    // Original batch contract: construct the shared FSE pool on the
    // producer thread before the consumer starts (thread-priority safety).
    warm_writer_pool();
    NativeCudaPlotSession session(native_cuda_rtx4070_super_ordinal());
    struct Ready {
        unsigned index=0;
        int slot=0;
        NativeCudaPlotFrame frame{};
    };
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<Ready> ready;
    bool busy[2]{false,false};
    bool producer_closed=false;
    std::exception_ptr writer_error;
    unsigned completed=0;
    std::thread consumer([&] {
        try {
            for(;;) {
                Ready item;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    cv.wait(lock,[&] { return producer_closed || !ready.empty(); });
                    if(ready.empty())break;
                    item=std::move(ready.front());
                    ready.pop_front();
                }
                auto const wstart=Clock::now();
                write_plot_file_parallel(
                    jobs[item.index].path,item.frame.fragments,
                    jobs[item.index].entry);
                double const write_seconds=std::chrono::duration<double>(
                    Clock::now()-wstart).count();
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    busy[item.slot]=false; // the consumer no longer reads it
                    ++completed;
                }
                cv.notify_all();
                std::fprintf(stderr,
                    "[cuda-batch] wrote %u/%u, writer=%.3f s: %s\n",
                    item.index+1,count,write_seconds,jobs[item.index].path.c_str());
            }
        } catch(...) {
            {
                std::lock_guard<std::mutex> lock(mutex);
                writer_error=std::current_exception();
                producer_closed=true;
            }
            cv.notify_all();
        }
    });

    std::exception_ptr producer_error;
    try {
        for(unsigned i=0;i<count;++i) {
            int const slot=int(i&1u);
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock,[&] { return !busy[slot] || bool(writer_error); });
                if(writer_error)std::rethrow_exception(writer_error);
                busy[slot]=true;
            }
            auto const gpu_start=Clock::now();
            auto frame=session.generate(jobs[i].entry.plot_id,slot);
            std::fprintf(stderr,
                "[cuda-batch] GPU %u/%u T1=%llu T2=%llu T3=%llu elapsed=%.3f s\n",
                i+1,count,
                static_cast<unsigned long long>(frame.stats.t1_count),
                static_cast<unsigned long long>(frame.stats.t2_count),
                static_cast<unsigned long long>(frame.stats.t3_count),
                std::chrono::duration<double>(Clock::now()-gpu_start).count());
            {
                std::lock_guard<std::mutex> lock(mutex);
                if(writer_error)std::rethrow_exception(writer_error);
                ready.push_back(Ready{i,slot,std::move(frame)});
            }
            cv.notify_all();
        }
    } catch(...) {
        producer_error=std::current_exception();
    }
    {
        std::lock_guard<std::mutex> lock(mutex);
        producer_closed=true;
    }
    cv.notify_all();
    consumer.join(); // never free pinned slots while writer may read them
    if(producer_error)std::rethrow_exception(producer_error);
    if(writer_error)std::rethrow_exception(writer_error);
    if(completed!=count)
        throw std::runtime_error("native CUDA batch did not publish all plots");
    auto const elapsed=std::chrono::duration<double>(
        Clock::now()-started).count();
    std::fprintf(stderr,
        "[cuda-batch] complete=%u wall=%.3f s throughput=%.3f s/plot "
        "(including startup and final writer drain)\n",
        completed,elapsed,elapsed/double(completed));
#else
    (void)opts;(void)output_dir;(void)count;
    throw std::runtime_error("native CUDA test batch requires Windows CUDA build");
#endif
}

} // namespace pos2gpu
