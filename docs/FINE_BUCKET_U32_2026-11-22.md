# T2/T3 Fine Bucket u32 offset experiment (2026-11-22)

Suggested branch: `fine-bucket-u32-20261122`, based on `main`.

## Implementation
- Fine offsets are **32-bit absolute sorted-row indices**. The coarse bucket offsets, metadata, match counters and output capacity remain 64-bit.
- T2/T3 prepare, full, compact, two-phase and sliced matching use the same new type.
- `POS2GPU_FINE_DIRECT=1` places its byte suffix array immediately after the **shorter** fine array.
- Input row count is checked against `UINT32_MAX` at prepare and full-match boundaries. Fine-entry indexing is checked separately.
- The existing exact target lower_bound fallback is retained for skewed groups; the suffix array remains a separate array (not interleaved).

## Estimated memory, k28 strength6 and fine_bits16
- Coarse buckets: unchanged (`(256 + 1) * 8` bytes).
- Fine starts: `(256 * 65536 + 1) * 8` -> `(256 * 65536 + 1) * 4` bytes.
- Saving: **67,108,868 bytes (~64 MiB)** per live fine index.
- Direct suffix bytes: `256 * 2^20 = 256 MiB` unchanged.
- Fine + suffix: approximately **384 MiB -> 320 MiB** (excluding coarse buckets).

## Validation commands (PowerShell, repository root)
1. `cmake -S . -B build/host-u32 -DXCHPLOT2_HOST_TESTS_ONLY=ON`
2. `cmake --build build/host-u32 --config Release --target fine_bucket_index_test`
3. `ctest --test-dir build/host-u32 -C Release -R fine_bucket_index_test --output-on-failure`
4. Build the Windows SYCL GPU target with the project's normal toolchain and run baseline versus this branch with the same settings (k28/s6); compare plot hashes / verify `--full`.
5. Time T2/T3 fine build, Direct build and match with `POS2GPU_FINE_TIMING=1`, with and without `POS2GPU_FINE_DIRECT=1`.

Results are recorded below; VRAM reduction alone does not establish a throughput improvement.
## Measured results (2026-11-22; matched Windows toolchain)

**Conclusion: essentially unchanged throughput; ~64 MiB less memory per live T2/T3 fine index.**
The u32 index is primarily a memory-footprint optimization, not a confirmed speedup.

Conditions: Windows 11, NVIDIA GeForce RTX 4070 SUPER (gpu0, 12 GiB),
k=28, strength=6, one worker; Compact and Tezcan enabled for T2/T3,
two-phase disabled, fine bits T1/T2/T3=8/16/16, Direct enabled,
phase timing enabled. Both variants were built using the rebuilt local
LLVM/AdaptiveCpp toolchain and benchmarked with one excluded warmup plus
10 measured plots, with the same NAS output destination. The 64-bit
baseline was built from the separate `C:\dev\xchplot2-main` worktree.

| Metric (10 measured plots; warmup excluded) | 64-bit fine index | 32-bit fine index | Delta |
| --- | ---: | ---: | ---: |
| T1 match mean | 7,842.50 ms | 7,841.15 ms | -1.35 ms (-0.02%) |
| T2 match mean | 8,387.90 ms | 8,360.76 ms | -27.14 ms (-0.32%) |
| T3 match + Feistel mean | 8,673.04 ms | 8,649.79 ms | -23.25 ms (-0.27%) |
| GPU phase total mean | 26,140.72 ms | 26,092.02 ms | -48.70 ms (-0.19%) |
| Benchmark steady-state | 26.28 s/plot | 26.21 s/plot | -0.07 s (-0.27%) |
| End-to-end reported | 0.125 TiB/hour | 0.125 TiB/hour | unchanged at displayed precision |

The 64-bit baseline emitted a first-run JIT cache warning; the warmup
was excluded from the measured phase averages. A ~0.27% end-to-end
change from these non-interleaved runs is too small to establish a
reproducible speedup without additional controlled repetitions.

### Memory model

At 256 buckets, fine_bits=16 and Direct suffix indexing enabled:

- 64-bit fine starts: `(256 * 65536 + 1) * 8 = 134,217,736` bytes.
- 32-bit fine starts: `(256 * 65536 + 1) * 4 = 67,108,868` bytes.
- Saving: **67,108,868 bytes = 64 MiB + 4 bytes**, per simultaneously live fine-index allocation.
- Suffix table: `256 * 2^20 = 268,435,456` bytes (unchanged).
- Combined fine + suffix: approximately **384 MiB -> 320 MiB**.

These sizes are derived from the buffer layout, not a direct measurement
of device-wide peak VRAM. The implementation leaves coarse bucket
indices at 64 bit. T2/T3 reuse matching workspace, so do not add both
per-table savings when estimating concurrent peak VRAM reduction.

### Correctness evidence

- Windows SYCL GPU build completed; k28 / strength 6 benchmark plotted successfully.
- Host-only `fine_bucket_index_test`: **1/1 PASS**.
- Two retained u32 bench plots each passed `verify --full --trials 100`:
  **92/92** and **104/104** full proofs; **196/196 total** across two samples.
- A sampled full-proof verification is not exhaustive CPU/GPU byte parity.
  Broader T2/T3 CPU/GPU parity and other GPU/memory tiers are not
  recorded as passing specifically for this u32 change.

**Disposition:** Keep the u32 change as a memory reduction with effectively
neutral measured performance; do not claim a confirmed speedup.
