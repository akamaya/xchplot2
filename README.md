# xchplot2

> **Refactor branch:** k=28 / strength=6 fixed for generation; verified SYCL GPU optimizations selected automatically. See [refactor notes](docs/VERIFIED_GPU_ALWAYS_ON_2026-11-23.md). Earlier experimental switch examples below are historical only.

<!-- BEGIN XCHPLOT2-FORK-OPTIMIZATION -->

> [!NOTE]
> **Experimental optimization fork.** This fork targets fast, correct Chia PoS2 plot generation on **native Windows 11 with a 12 GB VRAM NVIDIA GPU at k=28 / Strength 6**. Measurements below use an RTX 4070 SUPER with SYCL / AdaptiveCpp (CUDA backend). Based on [Jsewill/xchplot2](https://github.com/Jsewill/xchplot2); the original project documentation continues below.

## Windows 11 / 12 GB VRAM / Strength 6 optimization

**Goal:** Maximize practical k=28 / Strength 6 throughput within the available VRAM of a 12 GB GPU, without sacrificing proof correctness.

### NVIDIA native CUDA core migration branch (2026-11-23)

This branch is based on `refactor/always-on-verified-gpu-optimizations-20261123` (`f6646911`), **not** main. It ports the hot k28/s6 GPU stages (Xs AES, Direct24/Direct20 index builders, T1/T2/T3 Compact match, and gather/scatter utilities) to **nvcc CUDA**; CUB sorting was already native. The fixed AES final-round and VRAM reuse layout are retained.

**Verified CUDA core (`4fafeb2d`): Windows RTX 4070 SUPER build, 25.13 s/plot (10 measured), Full Proof 114/114 PASS.** Follow-up peer/two-level/runtime changes still await a fresh build. The Plot CLI remains a CUDA GPU-core / SYCL-host **hybrid**, not yet fully CUDA-only. Host orchestration and other secondary SYCL paths are not removed until replaced. The previous **25.18 s/plot** is the verified *SYCL* baseline, not a CUDA measurement.

See the [CUDA core port details, build and verification checklist](docs/NATIVE_CUDA_CORE_PORT_2026-11-23.md). A standalone [nvcc-only smoke test](tools/native_cuda_only/CMakeLists.txt) already passed native CUDA runtime, AES Xs **4096/4096**, auxiliary kernels and pinned-memory checks without AdaptiveCpp. The next, **unverified Stage 2**, adds native CUB-sort parity and a Candidate-D 12 GB GPU pool with shared Direct24/20 scratch sizing; the full Plot CLI still uses SYCL host orchestration. The native CUDA stages are **always selected when `XCHPLOT2_BUILD_CUDA=ON`** on this branch; there is no separate CUDA-core ON/OFF flag and no old runtime ON/OFF switches are restored.

### CUDA-only RTX 4070 SUPER test plotting — 実測・検証記録

Native CUDA Xs→T1→T2→T3 (Candidate D 8.172 GiB, native CUB)
produces dummy-memo, non-farmable `.gplot` files using the existing
CPU FSE Writer. Private-key `--memo` is forbidden. The standalone
CUDA GPU path is SYCL-free; the complete test CLI still links
the hybrid host runtime.

**RTX 4070 SUPER実機結果（2026-11-23）:**

| Mode | GPU + D2H | Writer | Complete plot | Proof check |
|---|---:|---:|---:|---|
| Native, pageable output | 25.4249 s | 5.472 s | 30.974 s | Not run |
| Native, pinned direct output | **24.9937 s** | **15.159 s** | **40.443 s** | **81/81 full proofs PASS** |
| Native, 3-plot pinned overlapped batch | 25.126 / 25.293 / 25.049 s | 5.579 / 5.587 / 5.843 s | **81.414 s total; 27.138 s/plot** incl. startup + drain | Batch files not yet verified |

An earlier native .gplot passed **109/109 full proofs**. The single
pinned Writer's 15.159 s included **11.105 s** in the combined
`fflush`/`_commit`/`fclose` stage; FSE compression took only
0.131 s. Three consecutive NAS outputs instead had Writer times
around 5.6–5.8 s. Thus the reason for sporadic slow SMB durability
is **not yet established**. An optional
`POS2GPU_WRITER_PHASES=1` reports individual `fflush`,
`fsync_commit` and `fclose` timings on **new builds**; that
split has not yet been tested on the user's host.

The original hybrid baseline **25.13 s/plot** is a 10-plot
steady-state result; the user also reported **26–27 s end-to-end
for a single plot**. The native 3-plot average includes startup
and final Writer drain, so do not compare these as equivalent
benchmarks. GPU and Writer overlap has been verified in the
3-plot batch, but the single-plot end-to-end regression remains open.

The generator does **not** perform full-T3 SHA comparisons or
deduplication diagnostics. Full Proof is checked separately after
file generation.

**Full experiment log, results, failures and reproduction commands:**
[CUDA-only Plot experiment results, 2026-11-23](docs/NATIVE_CUDA_PLOT_EXPERIMENT_RESULTS_2026-11-23.md).
Implementation details:
[native CUDA core port](docs/NATIVE_CUDA_CORE_PORT_2026-11-23.md).

### Measured progress

| Iteration | Steady-state time | What changed |
|---|---:|---|
| Initial SYCL baseline | 68.17 s/plot | Baseline implementation |
| T1 Tezcan AES only | 67.84 s/plot | Fast AES alone gave little end-to-end improvement |
| T1 Local Compaction | 58.32 s/plot | Work-group candidate compaction reduced warp divergence |
| T2/T3 two-phase disabled | 44.57 s/plot | Avoided costly scratch-overflow fallback/recomputation |
| T2 Local Compaction, Tezcan OFF | 39.72 s/plot | T2 candidate compaction |
| T2 Local Compaction + banked Tezcan | 36.75 s/plot | T2 match reduced to about 9.79 s |
| T3 Local Compaction, Tezcan OFF | 32.11 s/plot | T3 candidate compaction |
| **T3 Local Compaction + banked Tezcan** | **29.10 s/plot** | **T3 match + Feistel reduced to about 10.02 s** |
| Fine Bucket 8/8/8 (search baseline) | 28.74 s/plot | 10-plot baseline for subsequent Fine Bucket comparisons |
| Fine Bucket 8/14/14 (binary search) | 28.16 s/plot | Wider fine index reduces binary-search steps; 10 plots |
| Fine Bucket 8/16/16 (binary search) | 27.95 s/plot | 16-bit fine index; 10 plots |
| **Fine Bucket 8/16/16 + 16+4-bit direct lookup** | **27.27 s/plot** | **0.68 s (2.43% shorter time) vs 16-bit binary search; 10 plots** |
| T2/T3 compact pair capacity 480/330 | — (stage timing only) | T2 -313 ms (3.44%), T3 -293 ms (3.18%); ~0.61 s combined kernel reduction; T1 stays 512 |
| **AES V4INIT: 128-bit T0 initialization (T1–T3)** | **26.19 s/plot (best single observation)** | **Same-config A/B phase total -465.6 ms (-1.73%); 2 measured plots per mode; 114/114 full proofs PASS** |
| Fine Bucket T2/T3 index u64 -> u32 | 26.21 s/plot (10 plots) | 64 MiB smaller Fine16 Direct index; 64-bit reference 26.28 s/plot; speed difference within noise |
| **Candidate A: Fine index scratch aliases `d_storage`** | **26.25 s/plot (10 plots)** | **251.72 MiB** less allocated VRAM; OFF 26.21 / ON 26.25 s/plot; 183/183 full proofs PASS |
| **Candidate B: T2 xbits buffer reuse** | **26.28 s/plot (10 plots)** | **1,040 MiB** further saving; OFF 26.25 / ON 26.28 s/plot; 223/223 full proofs PASS |
| **Candidate C: Xs scratch aliases `d_pair_a`** | **26.15 s/plot (10 plots)** | **1,040 MiB** further saving; OFF 26.20 / ON 26.15 s/plot; 236/236 full proofs PASS |
| **Candidate D: pooled Xs/T1/T2 buffer remapping** | **26.08 s/plot (10 plots)** | **788.28 MiB** further saving; D OFF/ON 26.08 / 26.08 s/plot; 214/214 full proofs PASS |
| **T1 Direct24 exact match** | **25.96 s/plot (10 plots)** | Direct24 build ~18.8 ms, 89/89 full proofs PASS; pool 8,336 MiB |
| **T1 Direct24 + T2/T3 Direct20 exact match** | **25.56 s/plot (10 plots)** | ~0.40 s faster vs T1 Direct24-only; 87/87 full proofs PASS; pool 8,368 MiB |
| **AES Experiment A: partial AES final round (T1/T2/T3)** | **25.18 s/plot (10 plots)** | 25.53 to 25.18 s/plot (-0.35 s, -1.37%); T2 -164 ms, T3 -189 ms; 90/90 full proofs PASS; `POS2GPU_AES_LAST_ROUND_PARTIAL=1` |

**Earlier Candidate D 10-plot steady-state (2026-11-23):** **26.08 s/plot**
with Candidates A+B+C+D enabled, after one excluded warmup (RTX 4070 SUPER,
k28/s6). The same-build D OFF/ON comparison measured 26.08 / 26.08 s/plot;
no throughput regression was observed at the 0.01-second reporting precision.
Candidate C previously measured 26.15 s/plot in a separate batch; results
from different batches do not establish a speedup.

The Compact, Tezcan, Fine Bucket and two-phase selections are **opt-in experimental optimizations**, not a default-setting or world-record claim. **V4INIT itself is permanently enabled in the selected SYCL Tezcan paths** (no `POS2GPU_AES_VEC_INIT` switch). Intermediate rows are historical measurements from different runs; the Fine Bucket comparison used 10 measured plots per configuration, whereas the AES V4INIT phase A/B used two runs with one measured plot each after warmup.

<!-- BEGIN XCHPLOT2-VRAM-ABC-20261123 -->
### Fine Bucket 32-bit and GPU pool VRAM optimizations (2026-11-22 to 2026-11-23)

All measurements below used a single **RTX 4070 SUPER 12 GiB**, native Windows 11,
SYCL/AdaptiveCpp CUDA, `k=28`, strength 6, one warmup followed by **10 measured plots**
per A/B mode. The optimization flags were explicitly set for every run.
The A/B comparisons are separate experiment batches, not a single paired
four-step timing sequence; small timing differences do not establish a speedup.

| Step | Incremental reduction in GPU pool allocation | Cumulative reduction vs 32-bit baseline | 10-plot steady-state, optimization enabled |
|---|---:|---:|---:|
| 32-bit Fine Bucket index | 64 MiB vs 64-bit index | Baseline for A/B/C | 26.21 s/plot |
| Candidate A: reuse `d_storage` for Fine match index | 251.72 MiB | 251.72 MiB | 26.25 s/plot |
| Candidate B: relocate T2 xbits | 1,040 MiB | 1,291.72 MiB | 26.28 s/plot |
| **Candidate C: alias Xs scratch with `d_pair_a`** | **1,040 MiB** | **2,331.72 MiB (about 2.28 GiB)** | **26.15 s/plot** |

**Allocation accounting, not device-wide peak VRAM telemetry.** The 2,331.72 MiB
is the **A+B+C** reduction relative to the 32-bit-index baseline; the earlier
64 MiB index reduction is listed separately. GPU pool reductions apply to the
**pooled GPU pipeline** under the tested flags, not all streaming paths or GPUs.
At k28, B and C each remove `cap * 4 = 1,090,519,040 bytes (1,040 MiB)`;
A removes 263,948,813 bytes of separately allocated Fine scratch.

**Correctness checks:** A: 183/183, B: 223/223, C: 236/236 full proofs validated
(two retained `.gplot` files and 100 trials each per experiment).
These are sampled full-proof checks, not exhaustive CPU/GPU byte parity.
C also passed all three host tests: `xs_pair_a_alias_test`,
`t2_xbits_reuse_test`, and `match_scratch_alias_test`.

**Switches:** `POS2GPU_MATCH_INDEX_ALIAS=1` (A),
`POS2GPU_T2_XBITS_REUSE=1` (B),
`POS2GPU_XS_PAIR_A_ALIAS=1` (C). C requires B; missing required space causes
a safe fallback instead of claiming the C allocation saving. See the documents
linked under **Detailed optimization records** for lifetime and fallback rules.
<!-- END XCHPLOT2-VRAM-ABC-20261123 -->

### Candidate D: additional 788.28 MiB GPU pool VRAM saving (2026-11-23)

Candidate D (`POS2GPU_COMPACT_D=1`) reuses Xs/T1/T2 buffer lifetimes
in the pooled GPU pipeline. It changes `d_pair_a` from 3,120 to 2,080 MiB,
while shared sort/Fine scratch increases from 68.28 to 320 MiB.

| Metric | D OFF: A+B+C | D ON: A+B+C+D |
|---|---:|---:|
| **Total allocated GPU pool** | **8,388.28 MiB** | **7,600.00 MiB** |
| Added VRAM saving | -- | **788.28 MiB (9.40%)** |
| 10-plot steady-state | 26.08 s/plot | **26.08 s/plot** |
| Excluded warmup | 1 plot | 1 plot |

The comparison used **RTX 4070 SUPER 12 GB (`gpu0`)**, Windows 11,
**k=28, strength 6**, one GPU worker, Fine Bits 8/16/16, Fine Direct,
Compact and Tezcan ON, two-phase OFF. Two retained Candidate D plots
passed `verify --full --trials 100`: **214/214 full proofs validated**.
The 10-plot throughput runs were non-kept.

The 788.28 MiB improvement measures **sum of GPU pool allocations, not
observed whole-device peak VRAM**. No throughput regression was observed
at 0.01 s/plot report precision; smaller effects are not ruled out.
Candidate D remains **opt-in**; unset `POS2GPU_COMPACT_D` or set it to
`0` to use A+B+C. See the [Candidate D detailed test record](docs/CANDIDATE_D_VRAM_2026-11-23.md).


<!-- BEGIN XCHPLOT2-EXACT-INDEX-20261123 -->
### Full-key exact-match indexes: T1 Direct24 + T2/T3 Direct20 (2026-11-23)

Experimental **SYCL compact-match only** optimization for RTX 4070 SUPER
12 GB (`gpu0`) on Windows 11, `k=28`, strength 6. The indexes provide
absolute 32-bit start offsets per complete match target instead of per-target
binary search or 16+4-bit compressed lookup.

- **T1 Direct24:** 16 match buckets x 2^24 targets, plus one sentinel:
  1,024 MiB of `uint32_t` offsets. Fine18 offsets (32 MiB) are used
  **only to build** the Direct24 index, not for T1 matching. Enable with
  `POS2GPU_T1_DIRECT24=1`, `POS2GPU_T1_FINE_BITS=18` and
  `POS2GPU_T1_COMPACT=1`.
- **T2/T3 Direct20:** 256 match buckets x 2^20 targets, plus sentinel:
  1,024 MiB of offsets per phase. Fine16 (64 MiB) remains during index
  construction. The T2 and T3 indexes **share** the same phase-reused GPU
  sort/match scratch with T1; the three allocations do not accumulate.
  Enable with `POS2GPU_T23_DIRECT20=1`,
  `POS2GPU_T2_FINE_BITS=16`, `POS2GPU_T3_FINE_BITS=16`,
  `POS2GPU_T2_COMPACT=1`, `POS2GPU_T3_COMPACT=1`.

Both builders scan the sorted target values within Fine18/Fine16 groups,
handle empty keys and duplicates via repeated lower-bound offsets, and
retain a skew-safe bounded-binary-search fallback. The full-key range
is `[direct[(bucket<<bits)|target], direct[((bucket<<bits)|target)+1])`.
The prior search paths remain available with the new flags disabled.

#### Measured results (one excluded warmup, 10 measured plots per run)

| Index configuration | Steady-state | GPU pool allocation |
|---|---:|---:|
| Candidate D + T1 Fine8 + T2/T3 Fine16 direct 16+4 | 26.07 s/plot | 7,600 MiB |
| Candidate D + T1 Fine18 + T2/T3 direct 16+4 | 26.02 s/plot | 7,600 MiB |
| Candidate D + **T1 Direct24**, T2/T3 direct 16+4 | **25.96 s/plot** | **8,336 MiB** |
| Candidate D + **T1 Direct24 + T2/T3 Direct20** | **25.56 s/plot** | **8,368 MiB** |

The first two results come from a Fine8/Fine18/Fine8 comparison (26.07,
26.02, 26.09 s/plot). Each Direct24 and Direct20 measurement uses a
separate 10-plot run; **these are cross-run comparisons, not same-run
paired A/B tests**, and small differences may reflect run-to-run variability.
All measurements use Candidate D and the Tezcan/compact production flags
with two-phase matching disabled.

With T1 Direct24, the `direct24_build` stage stabilized at about **18.8 ms**
per plot after JIT warmup. T1 matching improved from about **7.78 s**
(Fine18) to **7.70 s** (Direct24). Adding T2/T3 Direct20 reduced T2's
`match_fused` from about **8.35 s** to **8.14 s** and T3's from
about **8.63 s** to **8.42 s**. T2/T3 Direct20 builds cost about
**10.0/9.8 ms**, versus about **6.8/6.7 ms** for 16+4-bit compressed
index builds. Full configuration versus T1-only Direct24 improved
the reported steady-state by about **0.40 s/plot (1.54%)**.

**VRAM accounting:** Candidate D base is 7,600 MiB of GPU pool allocations.
Direct24 increases pooled sort/match scratch from 320 to 1,056 MiB,
adding 736 MiB. Enabling T2/T3 Direct20 expands that shared scratch
to 1,088 MiB, adding **32 MiB** (not another 2 GiB). The final
pool allocation is **8,368 MiB (~8.17 GiB)**, consisting of
`d_storage=3,120 MiB`, `d_pair_a=2,080 MiB`, `d_pair_b=2,080 MiB`
and `d_sort_scratch=1,088 MiB`. This is **allocated pool memory,
not observed whole-device peak VRAM**; pinned host memory is separate.

**Retained-plot correctness:** T1 Direct24 alone: `verify --full --trials 100`,
**89/89 full proofs valid**. T1 Direct24 + T2/T3 Direct20:
`verify --full --trials 100`, **87/87 full proofs valid**.
Both runs returned `OK (full proof validation)`. These are sampled
correctness tests, not an exhaustive proof or a multi-device parity test.

**Scope:** Experimental opt-in; existing defaults unchanged.
The exact-index acceleration is implemented for the SYCL compact T1/T2/T3
paths and has not been validated for all plot sizes, strength settings,
backends, tiers or simultaneous workers. If memory admission fails on a
12 GB GPU, disable the new exact-index flags and retain Candidate D.

<!-- END XCHPLOT2-EXACT-INDEX-20261123 -->

### Latest AES V4INIT A/B — same configuration

`k=28`, Strength 6, RTX 4070 SUPER, one worker, Fine Bits **8/16/16**, T1/T2/T3 Compact ON, T2/T3 Tezcan ON, two-phase OFF, Fine Direct ON, pair capacities T1/T2/T3 **512/480/330**. Two OFF→ON pairs; each run used one excluded warmup and one measured plot. Values are arithmetic averages of the two measured `[phase-timing]` entries per mode.

| Stage | Previous scalar initialization | V4INIT (128-bit) | Improvement |
|---|---:|---:|---:|
| T1 match | 7,802.9 ms | 7,826.3 ms | -0.30% (no confirmed benefit) |
| **T2 match** | **8,685.3 ms** | **8,505.9 ms** | **2.07% shorter** |
| **T3 match + Feistel** | **9,066.3 ms** | **8,809.1 ms** | **2.84% shorter** |
| **All GPU phases** | **26,916.0 ms** | **26,450.4 ms** | **465.6 ms / 1.73% shorter** |

The benchmark's `steady-state` is affected by asynchronous output and the NAS; do not use the best single 26.19 s/plot as the paired performance effect. Nsight Compute verified that the initializer emits **`STS.128`**, reducing warp-level initialization store instructions from 131,072 to 32,768 (the number of transferred shared-memory wavefronts is unchanged). [Detailed V4INIT investigation](docs/AES_V4INIT_OPTIMIZATION_2026-11-21.md).

### Benchmark configuration

- **GPU:** NVIDIA GeForce RTX 4070 SUPER, 12 GB VRAM (`gpu0`); one GPU plotting worker
- **OS / backend:** Windows 11; SYCL / AdaptiveCpp targeting CUDA
- **Workload:** k=28, Strength 6; synthetic `.gplot` benchmark output
- **Measurement:** Current A/B/C memory experiments: one warmup + ten measured plots per mode; 2026-11-21 Fine Bucket result: one warmup + ten measured plots; AES V4INIT phase A/B: two runs per mode with one warmup + one measured plot each; early T1/T2/T3 iterations used one warmup + three measured plots.
- **Latest V4INIT phase stage times:** T1 match ~7.83 s; T2 match ~8.51 s; T3 match + Feistel ~8.81 s (two measured plots per mode). Earlier T1/T2/T3 baseline was ~7.8 / ~9.8 / ~10.0 s.
- **Historical peak device-wide VRAM use (before A/B/C):** 10,474 MiB, with 11,069 MiB free at benchmark start and a minimum of 595 MiB remaining. The 2.28 GiB A/B/C savings above are *pool allocation accounting*, not an updated peak-VRAM measurement.
- **Steady-state spread:** 28.98窶・9.21 s/plot, standard deviation 0.11 s in the fastest three-plot run

### Reproduce the current A+B+C measured configuration

On Windows 11 with the compiled `build/windows-sycl` GPU target, run the
following **single PowerShell command** from any directory. It explicitly
sets each benchmark flag so it does not depend on previous shell state.
The benchmark uses `gpu0` (RTX 4070 SUPER) and the existing NAS output directory.

```powershell
cd C:\dev\xchplot2; $env:POS2GPU_T1_COMPACT='1'; $env:POS2GPU_T2_COMPACT='1'; $env:POS2GPU_T3_COMPACT='1'; $env:POS2GPU_T2_TEZCAN='1'; $env:POS2GPU_T3_TEZCAN='1'; $env:POS2GPU_T2_TWOPHASE='0'; $env:POS2GPU_T3_TWOPHASE='0'; $env:POS2GPU_T1_FINE_BITS='8'; $env:POS2GPU_T2_FINE_BITS='16'; $env:POS2GPU_T3_FINE_BITS='16'; $env:POS2GPU_FINE_DIRECT='1'; $env:POS2GPU_MATCH_DIAG='0'; $env:POS2GPU_PHASE_TIMING='1'; $env:POS2GPU_POOL_DEBUG='1'; $env:POS2GPU_MATCH_INDEX_ALIAS='1'; $env:POS2GPU_T2_XBITS_REUSE='1'; $env:POS2GPU_XS_PAIR_A_ALIAS='1'; .\build\windows-sycl\tools\xchplot2\xchplot2.exe bench -k 28 -s 6 -n 10 --warmup 1 --devices gpu0 -o \\NAS-UGREEN\personal_folder\plots_test
```

Environment variables are per-process. These pooled buffer aliases are
experimental; they do **not** accelerate the streaming tier. `bench` builds
synthetic, non-farmable plots and deletes them unless `--keep` is specified.

### Correctness and verification

- **T2 CPU/GPU parity:** **30/30 PASS** at k=18, covering Strength 2窶・ and multiple plot IDs/seeds.
- **T2 single-GPU / sharded parity:** **6/6 PASS** at k=18/20/22.
- **T2-generated k=28 / Strength 6 plot:** `verify --trials 100 --full` validated **126/126 full proofs**.
- **T3 single-GPU / sharded parity:** **6/6 PASS** at k=18/20/22, comparing the full 64-bit proof fragments.
- **T3-optimized k=28 / Strength 6 plot:** `verify --trials 100 --full` validated **91/91 full proofs**.
- Earlier T1 parity tests matched 4,189,913 results, but the final T1 Local Compaction path has not been documented here as independently passing broad CPU-reference parity.
- **V4INIT + Fine Direct + Compact / Tezcan, k=28 / Strength 6:** one retained synthetic `.gplot`, `verify --full --trials 100`: **46 challenges with quality chains, 114 chains, 114/114 full proofs validated; OK**. This is a sample-based full-proof check, **not a same-input CPU/GPU byte-for-byte parity test**.

Candidate A/B/C also passed 183/183, 223/223 and 236/236 sampled full proofs, respectively, under the tested pooled k28/s6 configurations.

Full-proof checks sample random challenges; they do **not** exhaustively certify all possible proofs, GPU models, strength settings, or streaming tiers. The optimization branch remains experimental.

### Detailed optimization records

- [T1 investigation and benchmark history (Japanese)](docs/OPTIMIZATION_HISTORY_2026-11-20.md)
- [T1 offline graph report (HTML, Japanese)](docs/OPTIMIZATION_REPORT_2026-11-20.html)
- [T2 Local Compaction / Tezcan implementation and parity checks (Japanese)](docs/t2_gpu_optimization_2026-11-20.md)
- [T3 Local Compaction / Tezcan implementation and proof verification (Japanese)](docs/t3_gpu_optimization_2026-11-20.md)
- [AES experiments and production A/B (Shared16 direct-address adopted on T2/T3; rejected BitSlice / ILP / MIO / Hybrid / PairTable candidates)](docs/AES_EXPERIMENTS_2026-11-21.md)
- [AES Experiment A: final-round partial computation, k28/s6 benchmark, and 90/90 full-proof validation (2026-11-23)](docs/AES_LAST_ROUND_PARTIAL_2026-11-23.md)
- [Fine Bucket 16+4-bit direct lookup, 10-plot benchmarks (Japanese)](docs/FINE_BUCKET_OPTIMIZATION_2026-11-21.md)
- [T1/T2/T3 candidate-capacity tuning, occupancy analysis (Japanese)](docs/GPU_PAIR_CAP_T1_T3_2026-11-21.md)
- [AES V4INIT: research, SASS, 2-pair A/B and full-proof validation (Japanese)](docs/AES_V4INIT_OPTIMIZATION_2026-11-21.md)
- [Fine Bucket 32-bit index: 64 MiB reduction and 10-plot A/B](docs/FINE_BUCKET_U32_2026-11-22.md)
- [Candidate A: Fine match scratch alias, 251.72 MiB reduction](docs/FINE_BUCKET_SCRATCH_ALIAS_2026-11-23.md)
- [Candidate B: T2 xbits reuse, 1,040 MiB reduction](docs/T2_XBITS_REUSE_2026-11-23.md)
- [Candidate C: Xs scratch / pair_a alias, 1,040 MiB reduction](docs/XS_PAIR_A_ALIAS_2026-11-23.md)
- [Candidate D: additional VRAM saving and validated benchmark](docs/CANDIDATE_D_VRAM_2026-11-23.md)

The original project's usage, build instructions, hardware compatibility and performance comparisons are preserved below. This fork does not modify or imply endorsement by the upstream project.

---

<!-- BEGIN XCHPLOT2-FINE-BUCKET-DOC -->

### Fine Bucket search optimization (2026-11-21)

**Latest measured configuration:** `T1/T2/T3 Fine Bits = 8/16/16` with T2/T3 **16+4-bit compressed direct lookup** (`POS2GPU_FINE_DIRECT=1`, `COMPACT=1`) on Windows 11 / RTX 4070 SUPER, k=28 / Strength 6. The 10-plot measurement (plus one excluded warmup) achieved **27.27 s/plot** versus **27.95 s/plot** for 16-bit + binary search and **28.74 s/plot** for the 8-bit baseline. The saved 16-bit group offsets (128 MiB) and 4-bit suffix-start table (256 MiB) share the same **384 MiB** pool scratch between T2 and T3. A 255+-row group falls back to exact binary search; the default remains disabled.

The Fine Bucket implementation alone initially lacked a new full-proof validation result. A later **combined** Fine Direct + Compact/Tezcan + V4INIT k=28/Strength 6 plot passed **114/114 full proofs**. A dedicated same-input CPU/GPU byte-parity test for Fine Direct alone is still not recorded. Fine Direct applies only to the **SYCL T2/T3 all-buckets COMPACT** path, not every backend/tier.

See the complete design, reproducible PowerShell command, measurements, limits and verification checklist in [Fine Bucket 16+4 optimization record](docs/FINE_BUCKET_OPTIMIZATION_2026-11-21.md).

<!-- END XCHPLOT2-FINE-BUCKET-DOC -->

<!-- END XCHPLOT2-FORK-OPTIMIZATION -->

CPU and GPU plotter for Chia v2 proofs of space (CHIP-48). Produces ordinary
PoS2 1.0 `.gplot` files using the pinned
[pos2-chip](https://github.com/Chia-Network/pos2-chip) CPU reference.

This is the **`main` branch**, using SYCL/AdaptiveCpp with a CUB fast path
on NVIDIA. The [`cuda-only` branch](https://github.com/Jsewill/xchplot2/tree/cuda-only)
provides the native CUDA implementation for NVIDIA.

This build pins [pos2-chip 1.0.0](https://github.com/Chia-Network/pos2-chip/releases/tag/1.0.0).
The new Feistel cipher and group format are incompatible with earlier plots;
existing plots must be replotted and old job manifests cannot be resumed.
Farmer plot-group integration has merged upstream; release and activation
support must still be confirmed in the
[compatibility notes](contrib/pos2-pr118/README.md).
Only k28 plots are intended for farming; smaller supported sizes are for testing.

Linux and experimental native Windows builds include
[grouped plotting, drive filling and growth](REFERENCE.md#grouped-plotting-and-drive-filling).
Use `plot --group-members N` for several members per file, or `fill DIR...` to
plan and fill drives. `plan DIR...` previews the operation and can optionally
save a layout for later execution. Add `fill --replot` to
[replace eligible old plots](REFERENCE.md#replacing-an-existing-farm), using
configured farm directories and optional exclusions. Use the intended harvester's profile when plotting
on another machine. Repeat `fill DIR` to continue through any member, or add
options to change future work while keeping existing plots and reservations.
`pause`, `drain`, `resume` and `retire` control a job or selected destinations;
`status DIR --watch` follows saved progress.
`--discover-drives DIR` selects supported drives mounted beneath a directory
at any depth. See [drive filling](REFERENCE.md#grouped-plotting-and-drive-filling)
for recovery and policy changes. [Shared compute](REFERENCE.md#shared-compute)
lets enrolled machines contribute CPU/GPU capacity to multiple plotting hosts.
These features are available in `main` source builds; the v0.14.1 release
archives predate them. Follow the [installation guide](INSTALL.md) to build
from source.

[Quick start](#quick-start) · [Hardware](#hardware-compatibility) ·
[Build](#build) · [Commands](#use) · [Performance](#performance) ·
[Documentation](#documentation)

## Quick start

For prebuilt archives, follow [binary installation](INSTALL.md#binary-archives).

Install the [build dependencies](INSTALL.md#native-install) first.
For containers or Windows, follow [INSTALL.md](INSTALL.md).

```bash
cargo install --git https://github.com/Jsewill/xchplot2 --locked
xchplot2 devices

# Replace the key, contract address, and output directory with your values.
xchplot2 plot -k 28 -n 10 \
    -f <farmer-pk-hex> \
    -c <pool-contract-xch1-or-txch1> \
    -o /mnt/plots
```

Without shared compute, this command uses one GPU. Add `--devices gpu` to
use all GPUs, or `--devices all` to include CPU workers. See
[device selection](REFERENCE.md#devices-and-cpu-workers) for more options.

Keep the generated `xchplot2-job-*.tsv` manifest private: it contains plot
keys and lets you [resume an interrupted job](REFERENCE.md#plotting-and-recovery).

Each completed output path is printed to stdout. Check one output with:

```bash
xchplot2 verify /mnt/plots/NAME.gplot --full --trials 100
```

`verify --full` samples challenges and validates the resulting full proofs;
it does not check every byte. See [validation](CONTRIBUTING.md#building-and-running-tests)
for CPU-reference comparisons.

## Hardware compatibility

| Resource | Support and requirements |
|---|---|
| NVIDIA | Maxwell or newer via CUDA/CUB. Pre-Turing GPUs require a CUDA 12.x build. |
| AMD | AdaptiveCpp HIP with a compatible ROCm runtime and driver. RDNA1 needs the [target-selection guidance](INSTALL.md#amd-target-selection). |
| Intel | AdaptiveCpp Level Zero; see the [required settings and limitations](REFERENCE.md#troubleshooting). Use CPU plotting on integrated Intel GPUs for production. |
| VRAM | Tiny's base k=28 floor is 1,228 MiB free after context creation, including the default 128 MiB buffer. Backend sort scratch can raise it. |
| Host RAM | Depends on tier and worker count. Lower VRAM tiers generally use more host RAM; see [memory requirements](REFERENCE.md#memory-requirements). |
| CPU plotting | Opt in with `--devices cpu`, `--devices all`, or `--cpu`; uses pos2-chip's CPU plotter. |
| OS | Linux; WSL2 requires support from the GPU vendor. [Native Windows builds](INSTALL.md#windows) are experimental. macOS is unsupported. |

See the [installation guide](INSTALL.md) for drivers, runtime versions and
platform requirements. Memory needs depend on the backend, tier and workload;
use the [memory guide](REFERENCE.md#memory-requirements) when sizing a system.

## Build

See [INSTALL.md](INSTALL.md) for dependencies, containers, Cargo, CMake,
architecture selection, and Windows/WSL2. CMake also builds the parity and
host test binaries.

## Use

See [configuration and argument files](REFERENCE.md#configuration-and-argument-files)
for reusable options. `xchplot2 --help` prints command syntax.

### Storage planning and filling

| Command | Purpose and guide |
|---|---|
| [xchplot2 plan](REFERENCE.md#grouped-plotting-and-drive-filling) | Preview a farming layout; optionally save it for execution |
| [xchplot2 fill](REFERENCE.md#grouped-plotting-and-drive-filling) | Populate destination drives and resume interrupted work |

To replace an existing farm, use the separate
[inventory and migration workflow](REFERENCE.md#replacing-an-existing-farm).
Ordinary `fill` uses free space and does not remove existing plots. `fill --replot`
prepares and saves its migration before removing selected old plots. It permits
bounded early reclamation when needed; use `--reclaim-before-bytes 0` to require
replacement-first retirement. Complete farming resource bounds are required.
New reviews use bounded structural checks on replacement groups; optional
`sampled` or `strict` verification adds full proofs. Those later checks cannot
recover old plots already deleted under the allowance.

### Shared compute

| Command | Purpose and guide |
|---|---|
| [xchplot2 compute](REFERENCE.md#shared-compute) | Set up sharing, offer capacity, or run a dedicated broker |

Run `xchplot2 compute setup` on the first machine to start its broker, then
create an invitation for each helper. On a helper,
`xchplot2 compute setup INVITATION` enrolls it and starts its persistent
worker. Plotting jobs discover enrolled helpers automatically. For the
default ports and commands to allow connections through a firewall, see the
[network guide](REFERENCE.md#networks-and-dedicated-brokers).

### Plotting jobs

| Command | Purpose and guide |
|---|---|
| [xchplot2 plot](REFERENCE.md#plotting-and-recovery) | Create plots from farmer and pool keys |
| [xchplot2 batch](REFERENCE.md#batch-manifests) | Run or resume a saved plot manifest |
| [xchplot2 grow](REFERENCE.md#growing-an-existing-group) | Extend or recover an existing group file |

See [plotting and recovery](REFERENCE.md#plotting-and-recovery) for saved jobs
and resuming interrupted work.

### Checks and performance

| Command | Purpose and guide |
|---|---|
| [xchplot2 devices](REFERENCE.md#devices-and-cpu-workers) | List GPUs and CPU NUMA nodes |
| [xchplot2 bench](REFERENCE.md#benchmarking) | Measure throughput and estimate time to fill storage |
| [xchplot2 verify](REFERENCE.md#verification) | Check an existing plot, including full proofs with `--full` |

### Testing

| Command | Purpose and guide |
|---|---|
| [xchplot2 test](REFERENCE.md#single-test-plot) | Build a test plot from a group ID and memo |
| [xchplot2 parity-check](REFERENCE.md#parity-checks) | Run the built parity and host tests |

### Shell setup

| Command | Purpose and guide |
|---|---|
| [xchplot2 completions](REFERENCE.md#shell-completions) | Generate Bash, zsh, or fish completions |

## Performance

Measure the current build on your hardware and destination disk:

```bash
xchplot2 bench -k 28 -n 10 --warmup 2 --devices gpu0 -o /scratch
```

Use `--devices cpu` to compare CPU plotting. Throughput depends on the tier,
available memory, compression and storage; sharing a GPU or disk can change
the result. See the [benchmarking guide](REFERENCE.md#benchmarking) for options.

[Recorded benchmarks](BENCHMARKS.md) use the older, pre-1.0 cipher and raw
format. They are historical comparisons, not current `.gplot` performance
estimates.

## Documentation

| Guide | Contents |
|---|---|
| [Installation](INSTALL.md) | Dependencies, containers, Cargo/CMake, Windows and WSL2 |
| [Command reference](REFERENCE.md) | All commands, configuration, devices, memory, environment variables, troubleshooting |
| [Benchmark results](BENCHMARKS.md) | Dated measurements, methodology, and memory use |
| [Contributing](CONTRIBUTING.md) | Architecture, local tests, CI, and contribution conventions |
| [Compatibility and reference tools](contrib/pos2-pr118/README.md) | Upstream compatibility, independent checks, and the experimental reader extension |
| [Security](SECURITY.md) | Private key and manifest handling; vulnerability reporting |

## License

MIT — see [LICENSE](LICENSE) and [NOTICE](NOTICE) for third-party
attributions. Built with SI.

## Like this? Send a coin my way!

If you appreciate this, and want to give back, feel free.

xch1d80tfje65xy97fpxg7kl89wugnd6svlv5uag2qays0um5ay5sn0qz8vph8
