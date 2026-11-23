// FineBucketConfig.hpp - experimental per-table fine-index sizing + timing.
// Default is 8 bits to preserve existing behavior.
#pragma once

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace pos2gpu {

inline int fine_bucket_bits(int table_id, int target_bits)
{
    if (table_id < 1 || table_id > 3)
        throw std::invalid_argument("invalid fine bucket table id");
    int const bits = table_id == 1 && target_bits == 24 ? 18
                   : (table_id == 2 || table_id == 3) && target_bits == 20 ? 16
                   : 8; // non-28 shape for test/compatibility code only
    if (bits >= target_bits)
        throw std::invalid_argument("fine index bits exceed target bits");
    return bits;
}

// This times host wall intervals while synchronizing the GPU queue.
// It does NOT break the fused matching kernel into AES/search phases.
class FineBucketTimer {
public:
    FineBucketTimer(int table_id, int bits) : table_(table_id), bits_(bits),
        enabled_([] { const char* v = std::getenv("POS2GPU_FINE_TIMING");
                       return v && v[0] == '1'; }()),
        last_(std::chrono::steady_clock::now()) {}

    template <typename Queue>
    void mark(const char* label, Queue& q)
    {
        if (!enabled_) return;
        q.wait();
        auto const now = std::chrono::steady_clock::now();
        double const ms = std::chrono::duration<double, std::milli>(now - last_).count();
        std::fprintf(stderr, "[fine-timing] T%d bits=%d %s=%.3f ms\n", table_, bits_, label, ms);
        last_ = now;
    }
private:
    int table_;
    int bits_;
    bool enabled_;
    std::chrono::steady_clock::time_point last_;
};

} // namespace pos2gpu
