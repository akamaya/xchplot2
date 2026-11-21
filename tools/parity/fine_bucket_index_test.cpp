#include "gpu/FineBucketIndex.hpp"
#include "host/PoolSizing.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
void expect(bool okay)
{
    if (!okay) throw std::runtime_error("FineBucketIndex host test failed");
}
}

int main()
{
    using namespace pos2gpu;
    static_assert(sizeof(FineBucketOffset) == 4);
    expect(match_phase_capacity(28, 2) == 272629760ULL);
    validate_fine_bucket_input_count(match_phase_capacity(28, 2));
    validate_fine_bucket_input_count(std::numeric_limits<FineBucketOffset>::max());
    bool threw = false;
    try { validate_fine_bucket_input_count(std::uint64_t{1} << 32); }
    catch (std::invalid_argument const&) { threw = true; }
    expect(threw);

    auto const layout = fine_bucket_temp_layout(256, 16);
    expect(layout.fine_entries == 256ULL * 65536 + 1);
    expect(layout.bucket_bytes == 257 * sizeof(std::uint64_t));
    expect(layout.fine_bytes == (256ULL * 65536 + 1) * sizeof(FineBucketOffset));
    expect(layout.fine_bytes == 67108868ULL);
    expect(layout.total_bytes() == layout.bucket_bytes + layout.fine_bytes);
    // 256 MiB suffix array + 64 MiB fine array = 320 MiB (+ sentinel).
    expect(layout.fine_bytes + (256ULL << 20) == (320ULL << 20) + 4);

    auto const small = fine_bucket_temp_layout(4, 8);
    std::vector<std::uint64_t> buffer((small.total_bytes() + 7) / 8);
    auto* fine = fine_bucket_offset_ptr(buffer.data(), small.bucket_bytes);
    auto const* const_fine = fine_bucket_offset_ptr(
        static_cast<void const*>(buffer.data()), small.bucket_bytes);
    expect(fine == const_fine);
    fine[0] = 17;
    fine[small.fine_entries - 1] = 123;
    expect(const_fine[0] == 17 && const_fine[small.fine_entries - 1] == 123);

    for (auto const& [b, bits] : std::array<std::pair<unsigned, int>, 2>{{
             {65536u, 16}, {1u, 0}}}) {
        threw = false;
        try { (void)fine_bucket_temp_layout(b, bits); }
        catch (std::invalid_argument const&) { threw = true; }
        expect(threw);
    }
    return 0;
}
