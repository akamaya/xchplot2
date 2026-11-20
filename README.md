# xchplot2

<!-- BEGIN XCHPLOT2-FORK-OPTIMIZATION -->

> [!NOTE]
> **Experimental optimization fork.** This fork targets fast, correct Chia PoS2 plot generation on **native Windows 11 with a 12 GB VRAM NVIDIA GPU at k=28 / Strength 6**. Measurements below use an RTX 4070 SUPER with SYCL / AdaptiveCpp (CUDA backend). Based on [Jsewill/xchplot2](https://github.com/Jsewill/xchplot2); the original project documentation continues below.

## Windows 11 / 12 GB VRAM / Strength 6 optimization

**Goal:** Maximize practical k=28 / Strength 6 throughput within the available VRAM of a 12 GB GPU, without sacrificing proof correctness.

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

**Best measured result (2026-11-20):** **29.10 s/plot**, down **57.3%** from the initial 68.17 s/plot (**2.34x throughput**). The measured end-to-end rate was **81.2 TiB/month** (30-day projection), up from 64.3 TiB/month after T2 optimization.

These are **opt-in experimental optimizations**, not default settings or a world-record claim. Intermediate rows include measurements from different runs; the T2 and T3 Tezcan ON/OFF comparisons used one warmup plus three measured plots in each configuration.

### Benchmark configuration

- **GPU:** NVIDIA GeForce RTX 4070 SUPER, 12 GB VRAM (`gpu0`); one GPU plotting worker
- **OS / backend:** Windows 11; SYCL / AdaptiveCpp targeting CUDA
- **Workload:** k=28, Strength 6; synthetic `.gplot` benchmark output
- **Measurement:** one warmup plot followed by three measured plots
- **Fastest measured stage times:** T1 match ~7.8 s; T2 match ~9.8 s; T3 match + Feistel ~10.0 s
- **Peak device-wide VRAM use:** 10,474 MiB, with 11,069 MiB free at benchmark start and a minimum of 595 MiB remaining
- **Steady-state spread:** 28.98窶・9.21 s/plot, standard deviation 0.11 s in the fastest three-plot run

### Reproduce the optimized configuration

On the `aes-optimize` branch, after building the Windows SYCL target, run the following in **PowerShell** from the repository root:

```powershell
$env:POS2GPU_T1_COMPACT='1'; $env:POS2GPU_T2_COMPACT='1'; $env:POS2GPU_T2_TEZCAN='1'; $env:POS2GPU_T2_TWOPHASE='0'; $env:POS2GPU_T3_COMPACT='1'; $env:POS2GPU_T3_TEZCAN='1'; $env:POS2GPU_T3_TWOPHASE='0'; $env:POS2GPU_PHASE_TIMING='1'
.\build\windows-sycl\tools\xchplot2\xchplot2.exe bench -k 28 -s 6 -n 3 --warmup 1 --devices gpu0 -o D:\plots_test
```

Change the output directory and device ID as needed. These environment variables affect the current PowerShell process and its child processes; they are **not** enabled by a Git commit. The benchmark creates **synthetic, non-farmable** plots and removes them on exit unless `--keep` is supplied.

### Correctness and verification

- **T2 CPU/GPU parity:** **30/30 PASS** at k=18, covering Strength 2窶・ and multiple plot IDs/seeds.
- **T2 single-GPU / sharded parity:** **6/6 PASS** at k=18/20/22.
- **T2-generated k=28 / Strength 6 plot:** `verify --trials 100 --full` validated **126/126 full proofs**.
- **T3 single-GPU / sharded parity:** **6/6 PASS** at k=18/20/22, comparing the full 64-bit proof fragments.
- **T3-optimized k=28 / Strength 6 plot:** `verify --trials 100 --full` validated **91/91 full proofs**.
- Earlier T1 parity tests matched 4,189,913 results, but the final T1 Local Compaction path has not been documented here as independently passing broad CPU-reference parity.

Full-proof checks sample random challenges; they do **not** exhaustively certify all possible proofs, GPU models, strength settings, or streaming tiers. The optimization branch remains experimental.

### Detailed optimization records

- [T1 investigation and benchmark history (Japanese)](docs/OPTIMIZATION_HISTORY_2026-11-20.md)
- [T1 offline graph report (HTML, Japanese)](docs/OPTIMIZATION_REPORT_2026-11-20.html)
- [T2 Local Compaction / Tezcan implementation and parity checks (Japanese)](docs/t2_gpu_optimization_2026-11-20.md)
- [T3 Local Compaction / Tezcan implementation and proof verification (Japanese)](docs/t3_gpu_optimization_2026-11-20.md)

The original project's usage, build instructions, hardware compatibility and performance comparisons are preserved below. This fork does not modify or imply endorsement by the upstream project.

---

<!-- END XCHPLOT2-FORK-OPTIMIZATION -->


GPU plotter for Chia v2 proofs of space (CHIP-48). Produces single-member
`.gplot` files for PoS2 1.0, byte-identical to the pinned
[pos2-chip](https://github.com/Chia-Network/pos2-chip) CPU reference.

This is the **`main` branch**, using SYCL/AdaptiveCpp with a CUB fast path
on NVIDIA. The [`cuda-only` branch](https://github.com/Jsewill/xchplot2/tree/cuda-only)
provides the native CUDA implementation for NVIDIA.

This build pins [pos2-chip 1.0.0](https://github.com/Chia-Network/pos2-chip/releases/tag/1.0.0).
The new Feistel cipher and group format are incompatible with earlier plots;
existing plots must be replotted and old job manifests cannot be resumed.
Farmer integration is still in progress upstream; see the
[compatibility notes](contrib/pos2-pr118/README.md).

An opt-in [grouped-job runner](contrib/pos2-pr118/README.md#experimental-grouped-jobs)
supports experimental multi-member groups, shared keys, and resumable assembly.
It requires a separate build; normal commands produce independent single-member groups.

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

Each completed output path is printed to stdout. Check one output with:

```bash
xchplot2 verify /mnt/plots/NAME.gplot --full --trials 100
```

`verify --full` samples challenges and validates the resulting full proofs;
it does not check every byte. See [validation](CONTRIBUTING.md#building-and-running-tests)
for CPU-reference comparisons.

## Hardware compatibility

| Resource | Requirement or tested scope |
|---|---|
| NVIDIA | Maxwell or newer via CUDA/CUB. Pre-Turing GPUs require a CUDA 12.x build. RTX 4090 is in the current hardware benchmark set. |
| AMD | AdaptiveCpp HIP; RX 6700 XT (`gfx1031`) is in the current hardware benchmark set. RDNA1 needs the [installation-path guidance](INSTALL.md#amd-target-selection). |
| Intel | Arc B580 with AdaptiveCpp Level Zero; see the [runtime workaround](REFERENCE.md#troubleshooting). |
| VRAM | Tiny's base k=28 floor is 1,228 MiB free after context creation, including the default 128 MiB buffer. Backend sort scratch can raise it. |
| Host RAM | Depends on tier and worker count. Lower VRAM tiers generally use more host RAM; see [memory requirements](REFERENCE.md#memory-requirements). |
| CPU plotting | Opt in with `--devices cpu`, `--devices all`, or `--cpu`; uses pos2-chip's CPU plotter. |
| OS | Linux is tested. WSL2 requires support from the GPU vendor. [Native Windows SYCL/NVIDIA](INSTALL.md#windows) is experimental, with GPU qualification pending. macOS is unsupported. |

The [benchmark report](BENCHMARKS.md) records tested hardware and toolchains.
Build checks and GPU hardware checks are described separately in
[CONTRIBUTING.md](CONTRIBUTING.md). Physical low-capacity cards are outside
the latest benchmark set; a software VRAM cap does not certify another card.

## Build

See [INSTALL.md](INSTALL.md) for dependencies, containers, Cargo, CMake,
architecture selection, and Windows/WSL2. CMake also builds the parity and
host test binaries.

## Use

| Command | Purpose and guide |
|---|---|
| [xchplot2 plot](REFERENCE.md#plotting-and-recovery) | Create plots from farmer and pool keys |
| [xchplot2 batch](REFERENCE.md#batch-manifests) | Run or resume a saved plot manifest |
| [xchplot2 bench](REFERENCE.md#benchmarking) | Measure throughput and estimate time to fill storage |
| [xchplot2 devices](REFERENCE.md#devices-and-cpu-workers) | List GPUs and CPU NUMA nodes |
| [xchplot2 verify](REFERENCE.md#verification) | Check an existing plot, including full proofs with `--full` |
| [xchplot2 test](REFERENCE.md#single-test-plot) | Build a test plot from a group ID and memo |
| [xchplot2 parity-check](REFERENCE.md#parity-checks) | Run the built parity and host tests |
| [xchplot2 completions](REFERENCE.md#shell-completions) | Generate Bash, zsh, or fish completions |

See [configuration and argument files](REFERENCE.md#configuration-and-argument-files)
for reusable options. `xchplot2 --help` prints command syntax.

Ordinary plotting uses one GPU. Select all GPUs with `--devices gpu`; add
CPU workers with `--cpu`, or select both with `--devices all`. Each GPU
chooses a tier from its own free VRAM. See the
[device reference](REFERENCE.md#devices-and-cpu-workers).

Before starting, `plot` saves an `xchplot2-job-*.tsv` manifest in the output
directory. It contains private plot keys; keep it private and retain it to
recover the job. Repeat the original `plot` command with `--resume`, or
resume directly from the saved manifest:

```bash
xchplot2 batch /path/to/job.tsv --resume
```

Resume validates existing files before skipping them. See
[plotting and recovery](REFERENCE.md#plotting-and-recovery) for identity
matching, manifest selection, progress, stdout, and exit status.

If host RAM is short, use `--temp-dir` to select a real disk for automatic
spill, or `--max-host-ram` to set a budget. Available reductions differ by
tier and branch; see [host RAM and disk-offload](REFERENCE.md#host-ram-and-disk-offload).

## Performance

These figures use the pre-1.0 cipher and raw plot format.

Measured September 9, 2026, at k=28, strength=2, using real file writes,
FSE compression, and durability barriers. Times are mean completion intervals
over ten measured plots after two warmups, using one GPU per host. Each
non-auto tier was forced.

| Tier (`main`) | RTX 4090, CUDA/CUB | RX 6700 XT, AdaptiveCpp HIP | Arc B580, AdaptiveCpp Level Zero |
|---|---:|---:|---:|
| Auto (pool) | 2.50 s | 9.63 s | 13.75 s |
| Plain | 3.91 s; 2.66 s repeat | 9.76 s | 14.17 s |
| Compact | 3.84 s | 10.51 s | — |
| Minimal | 16.40 s | 22.19 s | — |
| Tiny | 27.74 s | 29.77 s | — |
| Pinned | 27.42 s | 29.73 s | — |

— means not benchmarked. See the [benchmark configurations](BENCHMARKS.md#hardware-and-builds).

The two RTX 4090 Plain timings differ for an undetermined reason.

The native `cuda-only` auto path measured **2.20 s/plot** on the same RTX
4090, with its optional D2H/Xs overlap enabled. These runs do not isolate the
cause of the difference between the native and SYCL runtimes.

The [benchmarks](BENCHMARKS.md) include configurations, variability, and
memory use. Forced tiers can use additional match scratch on these roomy GPUs;
their timings do not predict performance on a card restricted to a tier's
minimum VRAM.

Multi-GPU throughput also depends on shared PCIe bandwidth, CPU compression,
and storage. This benchmark set uses one GPU per host.

## Documentation

| Guide | Contents |
|---|---|
| [Installation](INSTALL.md) | Dependencies, containers, Cargo/CMake, Windows and WSL2 |
| [Command reference](REFERENCE.md) | All commands, configuration, devices, memory, environment variables, troubleshooting |
| [Benchmark results](BENCHMARKS.md) | Dated measurements, methodology, and memory use |
| [Contributing](CONTRIBUTING.md) | Architecture, local tests, CI, and contribution conventions |
| [Experimental grouping](contrib/pos2-pr118/README.md) | Grouped jobs, resume, memory limits, and upstream compatibility |
| [Security](SECURITY.md) | Private key and manifest handling; vulnerability reporting |

## License

MIT — see [LICENSE](LICENSE) and [NOTICE](NOTICE) for third-party
attributions. Built collaboratively with
[Claude](https://claude.ai/code).

## Like this? Send a coin my way!

If you appreciate this, and want to give back, feel free.

xch1d80tfje65xy97fpxg7kl89wugnd6svlv5uag2qays0um5ay5sn0qz8vph8
