# Fine Bucket match-index scratch aliasing (Candidate A, 2026-11-23)

## Goal

Reduce device VRAM in the **pooled** GPU pipeline without another
kernel, device copy, or change to T1/T2/T3 matching algorithms. This change
builds on the T2/T3 32-bit fine-offset experiment from 2026-11-22.
It does **not** change the streaming pipeline.

## Implementation

The pool previously sized `d_sort_scratch` to the maximum of CUB sorting
scratch and the T1/T2/T3 Fine Bucket index (including 16+4 direct suffix).
That meant ~320 MiB of the T2/T3 index remained separately allocated even
though the sort and match phases do not overlap.

The match temp is now aliased to `d_storage` while its previous data is idle:

| Stage | Live prefix in d_storage | Borrowable suffix (k28) |
| --- | --- | ---: |
| T1 Match | XsCandidateGpu[2^k] | 1,072 MiB |
| T2 Match | sorted T1 match-info keys `uint32[cap]` | 2,080 MiB |
| T3 Match | sorted T2 match-info keys `uint32[cap]` | 2,080 MiB |

The original `d_sort_scratch` remains for CUB sorting and as an automatic
fallback whenever the requested match-index size does not fit the available
suffix (including non-k28 settings). Sizing and runtime dispatch use the same
`MatchScratchAlias.hpp` helper; there is no out-of-bounds alias.

`POS2GPU_MATCH_INDEX_ALIAS=0` disables aliasing and restores the original
scratch sizing and match storage, for same-binary A/B measurements. Unset or
`=1` enables it. Set this **before starting the process**, not mid-run.

With k28/s6, T2/T3 Fine16 Direct in the 32-bit version, the separate index
allocation previously needed ~320 MiB. Expected pool reduction is:

    old max(CUB scratch, match-index size)
      - new max(CUB scratch, any match-index that cannot alias)

Therefore actual savings depend on the CUB sort scratch requirement and the
runtime Fine Bits settings; use `POS2GPU_POOL_DEBUG=1` to see the declared
`saved=...` value. This is **not a measured peak VRAM result** yet.

## Baseline before Candidate A (same Windows toolchain)

- RTX 4070 SUPER 12 GiB, k=28, strength=6, one GPU worker.
- 32-bit Fine Bucket implementation: 26.21 s/plot steady-state (10 measured,
  one warmup), full proof verification 196/196 on two plots.
- 64-bit Fine Bucket implementation: 26.28 s/plot steady-state (10 measured,
  one warmup). The difference was too small to assert a speed change.

## Validation plan (original)

1. Host unit test: `cmake -S . -B build/host-alias -DXCHPLOT2_HOST_TESTS_ONLY=ON`,
   `cmake --build build/host-alias --target match_scratch_alias_test`,
   `ctest --test-dir build/host-alias -R '^match_scratch_alias_test$' --output-on-failure`.
2. Build the GPU executable: `cmake --build build/windows-sycl --target xchplot2 --parallel 2`.
3. Use the same previous Fine16 / Direct / Compact / Tezcan settings as the
   32-bit baseline. Confirm the **pool** path is used with
   `POS2GPU_POOL_DEBUG=1`. Check `match-index bytes`, the per-phase alias
   decisions, and the reported scratch savings.
4. Run `bench -k 28 -s 6 -n 1 --warmup 1 --keep --devices gpu0`, then verify
   the saved plots with `verify --full --trials 100`.
5. Benchmark `POS2GPU_MATCH_INDEX_ALIAS=0` and `=1` in **separate process
   launches**, both `-n 10 --warmup 1`, with matching environment settings.
   Prefer alternating baseline/new to account for clock/temperature drift.
   Record phase times, steady-state, and the `vram: peak` watchdog result.
6. Re-test a smaller `k` and non-Direct Fine settings to cover fallback.

No commit/push is performed by the installer. Measured GPU results are recorded below.

## Measured results (2026-11-23)

- GPU: RTX 4070 SUPER, k28, strength 6, Fine Direct 16+4, T1/T2/T3 = 8/16/16 bits.
- A/B: 10 measured plots each, 1 warmup, same executable and settings.
- Alias OFF: 26.21 s/plot; Alias ON: 26.25 s/plot (+0.04 s, +0.15%). Performance effectively unchanged.
- Dedicated scratch: 335546380 -> 71597567 bytes; saved 263948813 bytes (251.72 MiB). This is allocated pool capacity, not measured peak VRAM.
- Alias active for T1/T2/T3; full-proof validation: 183/183 PASS across 2 plots.
- Conclusion: Adopt Candidate A for memory savings without a meaningful performance regression.
