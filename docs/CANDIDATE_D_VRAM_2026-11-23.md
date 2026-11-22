# Candidate D: pooled GPU VRAM reuse (experimental, 2026-11-23)

Candidate D has been GPU-benchmarked on RTX 4070 SUPER and passed
**214/214 sampled full-proof validations**. See measured results below. It changes no streaming-tier code. The existing
A+B+C baseline remains the default (`POS2GPU_COMPACT_D` unset or `=0`).

## Goal and buffer layout (k28)

| Allocation | A+B+C | Candidate D |
| --- | ---: | ---: |
| d_storage | 3120 MiB | 3120 MiB |
| d_pair_a | 3120 MiB | 2080 MiB |
| d_pair_b | 2080 MiB | 2080 MiB |
| d_sort_scratch | ~68 MiB | ~320 MiB |
| **Total** | **~8388 MiB** | **~7600 MiB** |

The observed k28/s6 GPU pool reduction is **788.28 MiB** of allocated VRAM
(the amount can differ by configuration and backend). This is **pool allocation
accounting**, not observed device-wide peak VRAM. Host pinned buffers unchanged.

## Phases and alias safety

- **Xs**: storage[0..4N) holds keys_a, storage[4N..8N) vals_a,
  storage[8N..] CUB scratch; pair_b[0..4N) keys_b and pair_b[4N..8N) vals_b.
  After sort, packing writes XsCandidateGpu[0..8N) over dead storage inputs.
- **T1 match**: Xs input storage[0..8N); T1 meta in pair_a[0..8cap);
  T1 mi in storage[8N..8N+4cap); fine scratch can use the residual storage
  suffix or the dedicated fallback.
- **T1 sort**: direct stable `u32/u64` sort using mi and meta as input;
  sorted mi to storage slot0 and sorted T1 meta to pair_b. No index/gather.
- **T2 match**: sorted T1 keys in storage slot0 and meta in pair_b.
  Unsorted T2 meta in pair_a, mi in storage slot1, xbits in storage slot2.
  Fine scratch uses dedicated `d_sort_scratch` (no storage suffix free).
- **T2 sort**: copy unsorted xbits from storage slot2 to pair_b's upper half
  (one added D2D). Identity starts in pair_b lower half. CUB writes sorted
  keys to storage slot0 and sorted indices to storage slot2. Gather xbits
  first from pair_b into storage slot1, then meta from pair_a into pair_b.
- **T3 match/sort**: sorted keys in storage slot0, sorted xbits in slot1,
  sorted meta in pair_b; T3 outputs into pair_a. Unchanged T3 radix sort.

Only `POS2GPU_COMPACT_D=1` requests D; it also requires B enabled and C-layout
requested. When the layout cannot fit, it stays on A+B+C. Read
`[pool] Candidate D=1` to confirm. Set before process start.

## Validation checklist (Windows 11, gpu0 RTX 4070 SUPER)

1. Host test: build and run existing `xs_pair_a_alias_test` (extended with D
   static assertions). Do not assume host tests prove GPU parity.
2. Build GPU `xchplot2` with VS2022 Developer PowerShell, `-j 16`.
3. Run a kept plot with D on, then `verify --full --trials 100` on every kept
   `.gplot`. Repeat for at least two plots and compare with baseline.
4. Separate-process 10-plot A/B with one excluded warmup and identical A/B/C
   flags, varying only D=0/1. Compare `steady-state` and `phase-timing`.
5. Check `POS2GPU_POOL_DEBUG=1` logs and `vram: peak` measurements. Verify no
   streaming fallback and no backend device faults.
6. Reject if steady-state exceeds baseline by more than ~1% repeatedly,
   or if any full-proof validation fails. Never merge to main unverified.

The patch script never commits, resets, cleans, pushes, or overwrites existing
tracked changes. Test on a new local branch only.

## Measured results: RTX 4070 SUPER (2026-11-23)

Environment: Windows 11 Pro, NVIDIA RTX 4070 SUPER 12 GB (`gpu0`),
`k=28`, strength `6`, one GPU worker, Xs/T1/T2/T3 baseline A+B+C
active, Fine Bucket bits `8/16/16`, Fine Direct and Tezcan enabled.
`POS2GPU_PHASE_TIMING=0` in the throughput A/B test.
Each run: `bench -k 28 -s 6 -n 10 --warmup 1 --devices gpu0`
with the same NAS output path. Each recorded steady-state value excludes
one warmup plot, and the two runs differ only by `POS2GPU_COMPACT_D=0/1`.

### Allocated device pool (not peak device VRAM)

| Buffer | D=0 (bytes) | D=1 (bytes) | D=0 (MiB) | D=1 (MiB) |
| --- | ---: | ---: | ---: | ---: |
| `d_storage` | 3,271,557,120 | 3,271,557,120 | 3,120 | 3,120 |
| `d_pair_a` | 3,271,557,120 | 2,181,038,080 | 3,120 | 2,080 |
| `d_pair_b` | 2,181,038,080 | 2,181,038,080 | 2,080 | 2,080 |
| `d_sort_scratch` | 71,597,567 | 335,546,380 | 68.28 | 320.00 |
| `d_counter` | 8 | 8 | ~0 | ~0 |
| **Total** | **8,795,749,895** | **7,969,179,668** | **8,388.28** | **7,600.00** |

**Saving: 826,570,227 bytes = 788.28 MiB (9.40% of A+B+C pool).**
This is a summed allocation accounting figure, **not an observed
whole-device peak VRAM reduction**. Pinned host buffers are excluded.
Candidate D requires its T2 Fine index in the dedicated scratch:
`alias T1=1 T2=0 T3=1`; the increased scratch is included above.

### Performance and proof correctness

| Metric | D=0 | D=1 |
| --- | ---: | ---: |
| Measured plots | 10 | 10 |
| Warmup plots (excluded) | 1 | 1 |
| `steady-state` | **26.08 s/plot** | **26.08 s/plot** |
| `end-to-end` | 0.126 TiB/hour | 0.126 TiB/hour |

**No throughput regression observed at the benchmark's 0.01 s reporting
precision.** A small performance effect below measurement noise is not
excluded. The first two retained Candidate D plots also passed
`verify --full --trials 100` independently: **112 + 102 = 214 full proofs,
all passed**. The 10-plot benchmark was non-kept and is a throughput
measurement, not a proof-validation test.

### Decision

**Adopt Candidate D for the tested RTX 4070 SUPER `k28/s6` pool path.**
Retain `POS2GPU_COMPACT_D=0` as an explicit A+B+C fallback and avoid
assuming that the same savings or speed hold on other backends/`k` values
without testing. The implementation remains opt-in through
`POS2GPU_COMPACT_D=1`; enabling it by default requires a separate decision.
