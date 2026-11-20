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
    const char* name;
    int max_bits;
    switch (table_id) {
    case 1: name = "POS2GPU_T1_FINE_BITS"; max_bits = 18; break;
    case 2: name = "POS2GPU_T2_FINE_BITS"; max_bits = 16; break;
    case 3: name = "POS2GPU_T3_FINE_BITS"; max_bits = 16; break;
    default: throw std::invalid_argument("invalid fine bucket table id");
    }

    int bits = 8;
    if (const char* value = std::getenv(name); value && value[0] != '\0') {
        char* end = nullptr;
        errno = 0;
        long const parsed = std::strtol(value, &end, 10);
        if (errno != 0 || end == value || *end != '\0' || parsed < 1 || parsed > max_bits)
            throw std::invalid_argument(std::string(name) + " must be in [1, " +
                                        std::to_string(max_bits) + "]");
        bits = static_cast<int>(parsed);
    }
    if (bits >= target_bits)
        throw std::invalid_argument(std::string(name) + " must be less than num_match_target_bits (" +
                                    std::to_string(target_bits) + ")");
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
