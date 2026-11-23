# RTX 4070 SUPER k28/s6 — CUDA core migration (2026-11-23)

## Status and provenance

- **Base**: refactor/always-on-verified-gpu-optimizations-20261123 at `f6646911faf1a793532abbd06cf55e2696bbe046` (not main).
- **Work branch**: `cuda/nvidia-12gb-k28s6`.
- **No legacy runtime ON/OFF switches restored**. Verified Compact, Direct24/Direct20, Tezcan and AES Experiment A remain the fixed k28/s6 production profile.
- **CUDA core commit `4fafeb2d` passed the user's Windows RTX 4070 SUPER build**, 11 plots (1 warmup + 10 measured), **25.13 s/plot** steady state, and retained-plot Full Proof validation **114/114 PASS** (100 trials). Historical SYCL baseline: **25.18 s/plot**; the 0.05 s difference is not significant.
- **Later CUDA-only migration work (peer transport, two-level sort, runtime facade, standalone smoke) is not yet rebuilt or validated on hardware.** Never attribute the earlier 25.13 s figure to these later changes.
- **This is a CUDA GPU-core port, NOT YET a pure CUDA-only build**: AdaptiveCpp still builds the host orchestration, buffer/queue management and remaining non-hot-path implementations.

## GPU work moved to native CUDA

| Stage | CUDA implementation | Integration |
|---|---|---|
| Xs AES generation and pack | `NativeCudaXs.cu` | `XsKernelsSycl.cpp` dispatches for CUDA queues |
| Stable radix sort | Existing `SortCuda.cu` (CUB) | Existing `SortSyclCub.cpp` bridge |
| T1/T2/T3 coarse bucket and Fine18/Fine16 indices | `NativeCudaIndex.cu` | `T1OffsetsSycl.cpp`, `T2OffsetsSycl.cpp` |
| T1 Direct24, T2/T3 Direct20 full-key exact indices | `NativeCudaIndex.cu` | `T1Kernel.cpp`, `T2Kernel.cpp`, `T3Kernel.cpp` |
| T1 Compact pairing, 512 AES rounds | `NativeCudaMatch.cu` | `T1OffsetsSycl.cpp` |
| T2 Compact pairing, 32 AES rounds | `NativeCudaMatch.cu` | `T2OffsetsSycl.cpp` |
| T3 Compact pairing + Feistel | `NativeCudaMatch.cu` | `T3OffsetsSycl.cpp` |
| Identity, gather, scatter, inverse, T2 permutation, stable two-way merge | `NativeCudaPipeline.cu` | `PipelineKernelsSycl.cpp` |

Key implementation details:

- Direct24/Direct20 index layouts, absolute u32 row offsets, empty buckets, duplicates and last sentinel match the SYCL versions.
- AES uses the shared16 V4INIT replicated T0 layout, original alternating keys, and the exact partial *final* AESENC round from Experiment A. T1 pairs=256 (512 rounds), T2/T3 pairs=16 (32 rounds). No first-round-fold Experiment B.
- CUB BlockScan compacts candidates in 256-thread blocks. T1/T2/T3 caps 512/480/330 match the verified implementation. Overflow processes **all** matching pairs rather than dropping candidates. T3 uses the unchanged four-round Feistel.
- The CUDA runtime selects the actual GPU owning the AdaptiveCpp CUDA-USM input pointer, rather than assuming the CUDA runtime device index matches a SYCL index. This matters on multi-NVIDIA-GPU hosts.
- Existing `sycl::queue` submissions are drained before native default-stream CUDA launches; native entry points check launch and synchronize before returning. This is deliberately conservative during initial validation.

## Build (VS2022 Developer PowerShell, Windows 11)

Build in a **new** build directory under `C:\dev\xchplot2`, leaving the existing SYCL build environment intact. Use CUDA Toolkit 12.9 and the already-installed AdaptiveCpp and OpenSSL toolchain.

```powershell
cd C:\dev\xchplot2; $env:ACPP_PREFIX=(Resolve-Path .\build\windows-toolchain\install).Path; $env:PATH="$env:ACPP_PREFIX\bin;$env:CUDA_PATH\bin;$env:PATH"; $openssl=(Resolve-Path .\build\windows-openssl\install).Path; cmake -S . -B .\build\windows-cuda-core -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_C_COMPILER=$env:ACPP_PREFIX/bin/clang.exe" "-DCMAKE_CXX_COMPILER=$env:ACPP_PREFIX/bin/clang++.exe" "-DOPENSSL_ROOT_DIR:PATH=$openssl" -DOPENSSL_USE_STATIC_LIBS=ON '-DACPP_TARGETS=generic;omp' -DXCHPLOT2_BUILD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
```

```powershell
cd C:\dev\xchplot2; cmake --build .\build\windows-cuda-core --target xchplot2 -j 16
```

The project may require the appropriate VS2022 MSVC/NVCC host toolchain and its usual CMake dependencies. **Build success has not been established in this authoring environment.**

## Hardware validation checklist

1. Build with the commands above. Stop on the first nvcc/host-link diagnostic; do not silently fall back to the old SYCL binary.
2. Confirm `xchplot2.exe devices` lists RTX 4070 SUPER as `gpu0`. Avoid accidentally benchmarking RTX 5070.
3. Generate a retained k28/s6 .gplot on `\\NAS-UGREEN\personal_folder\plots_test` using the newly built binary; check exit code.
4. `verify <file> --full --trials 100` should report `OK (full proof validation)`. This sampled test does not prove all possible inputs.
5. Measure `bench -n 10 --warmup 1 --devices gpu0` (k=28/s=6 are fixed and no longer accept `-k`/`-s`). Compare with the **validated CUDA 25.13 s/plot** and historical SYCL 25.18 s/plot. **No new OFF remeasurement is needed** unless the toolchain/environment changes.
6. If possible, test one overflow-heavy synthetic case and independent exact-index parity. The build is not yet certified.

## CUDA-only Stage 2: Candidate D pool and direct CUB checks (unverified)

After the 4096/4096 pure-CUDA Xs parity and host-memory/peer/gather tests passed
on the RTX 4070 SUPER, the standalone target was extended:

- `src/gpu/NativeCudaPool.hpp/.cpp` implements a **SYCL-free**
  `NativeCudaPoolPlan` / `NativeCudaPool` with the exact k28/s6
  Candidate-D allocation/alias footprint of the validated hybrid pool:
  `d_storage`, lazy `d_pair_a`, `d_pair_b`, shared
  Direct24/Direct20/CUB scratch, 64-bit counter, and lazy pinned T3 output.
  It uses the shared device-free `PoolSizing.hpp` and
  `CompactDLayout.hpp` helpers rather than maintaining a different cap
  formula. It checks `cudaMemGetInfo` and preserves a 128 MiB safety margin.
- CUB `SortPairs(u32,u32)`, `SortPairs(u32,u64)`, and `SortKeys(u64)`
  are linked directly from the existing `SortCuda.cu`, bypassing
  `SortSyclCub.cpp`. The standalone smoke now verifies their exact
  ordering and value correspondence against CPU references on 4096 keys.
- The standalone smoke calls `native_cuda_pool_plan()` to check
  Candidate-D dimensions, exact-match index scratch, and reported VRAM
  footprint, **without allocating the full 8+ GiB pool**.

This new work **has not been rebuilt or run on hardware yet**. The existing
verified CUDA/SYCL hybrid Plot executable is unchanged by these independent
CUDA-only pool/test sources. The remaining full native Plot work is to
orchestrate Xs/T1/T2/T3 through the native APIs while retaining the precise
input/output alias lifetimes, native Feistel key schedule, and the existing
Plot writer/verification code. The pool is intentionally not connected to
the production CLI until that pipeline is in place.

### Stage 2 incremental compile and check (VS2022 Developer PowerShell)

```powershell
cd C:\dev\xchplot2; git pull --ff-only origin cuda/nvidia-12gb-k28s6; if ($LASTEXITCODE -ne 0) { throw 'git pull failed' }; cmake --build .\build\windows-native-cuda-only-nvcc -j 16; if ($LASTEXITCODE -ne 0) { throw 'nvcc build failed' }; .\build\windows-native-cuda-only-nvcc\native_cuda_smoke.exe; if ($LASTEXITCODE -ne 0) { throw 'CUDA smoke failed' }
```


## CUDA-only Stage 3: independent complete GPU compute (UNVERIFIED)

**Source-only milestone.** Stage 2 standalone tests have passed on RTX 4070
SUPER, including CPU/GPU AES 4096/4096, CUDA CUB u32/u32 + u32/u64 + u64
4096-entry parity and Candidate-D 8.172 GiB allocation **planning**.
Stage 3 introduces a *separate* `native_cuda_pipeline.exe` without changing the
already-verified production hybrid `xchplot2.exe`.

### Implementation

- `NativeCudaStandalonePipeline.cpp/.hpp`: the native Windows host
  orchestrator runs Xs AES generation + CUB radix sort + Xs pack;
  builds the T1 Direct24 index and T1 match; sorts T1 (u32/u64 CUB);
  builds the T2 Direct20 index + T2 match; parks T2 xbits in reused
  pair B memory while sorting T2; builds T3 Direct20 + Feistel match;
  sorts T3 fragments (u64 CUB). All NVIDIA device execution uses
  native CUDA; no SYCL queue, AdaptiveCpp allocation or runtime.
- `tools/native_cuda_only/pos/FeistelCipher.hpp`: a local, independent
  Windows CNG/BCrypt SHA-256 implementation of the PoS2 4-round key
  schedule. The `FullRoundKey` ABI matches pos2-chip's header; this
  private shim is included **only** by the standalone CUDA targets.
  The regular hybrid CLI keeps using the external pos2-chip header.
- `native_cuda_smoke` now tests four complete Feistel SHA-256 key
  vectors against independent reference words. A Feistel test PASS is
  required before trusting the new T3 CUDA-only path.
- `pipeline_main.cu` offers `--plan` (default, no multi-GiB
  allocation) and `--run` (opt-in full k28/s6 calculation on RTX 4070
  SUPER). A custom plot ID can be passed as `--plot-id 64hex`.
  The fixed demo plot ID is NOT intended for farming.
- The result contains T1/T2/T3 counts and SHA-256 of the **sorted
  64-bit T3 fragments**. It reads T3 output through a small
  pinned-host tile so full output fits the existing 128 GiB host RAM
  without allocating a second huge pinned buffer.

**Boundary: no `.gplot` writer and no Full Proof verifier is linked to this
standalone binary.** `COMPUTE OK` is not proof validation. Do not treat the
runner as a finished plotter or replace a validated farming plot with
these diagnostics. Integrating the on-disk plot writer remains separate.

**Hardware validation 2026-11-23 — RTX 4070 SUPER (CUDA ordinal 0):**
Stage 3 built and the independent CUDA-only compute runner completed
end-to-end without any SYCL/AdaptiveCpp runtime calls:

```text
[native-cuda-smoke] Windows CNG Feistel key schedule PASS
[native-cuda-smoke] Xs AES: 4096/4096 parity PASS
[native-cuda-smoke] CUDA CUB u32/u32 sort: 4096 PASS
[native-cuda-smoke] CUDA CUB u32/u64 sort: 4096 PASS
[native-cuda-smoke] CUDA CUB u64 sort: 4096 PASS
[native-cuda-smoke] Candidate D pool: 8.172 GiB VRAM ... planning PASS
[native-cuda-smoke] OK (no AdaptiveCpp/SYCL)
[cuda-only] plot id: 0724415e7b98b5d2ef0c294663809dbad7f4112e4b6885a2bfdcf91633506d8a
[cuda-only] T1 pairs=268396548
[cuda-only] T2 pairs=268356064
[cuda-only] T3 fragments=268266000
[cuda-only] T1=268396548 T2=268356064 T3=268266000 seconds=25.987
[cuda-only] sorted T3 fragment SHA256=ed2eb2c1849b9276423c32a85c15bb1e6b1f36420550e88b1fc59e0647f6a1f0
[cuda-only] COMPUTE OK (not a Full Proof validation)
```

The 25.987 s is one first full compute + T3 SHA-256 observation.
The old 25.13 s/plot was a 10-plot steady-state benchmark
that includes actual file writing with producer/consumer overlap:
**these timings are not directly comparable**. No claim of full
proof correctness is made by the native compute test. Identity-parity
against a reference plot with the **same** Plot ID and on-disk
full-proof checking are the next validation gates.

### Build and staged checks (VS2022 Developer PowerShell)

After `git pull --ff-only origin cuda/nvidia-12gb-k28s6`, rerun
CMake to register the new executable and compile with 16 jobs:

```powershell
cd C:\dev\xchplot2; cmake -S .\tools\native_cuda_only -B .\build\windows-native-cuda-only-nvcc -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89; if ($LASTEXITCODE -ne 0) { throw 'CUDA-only configure failed' }; cmake --build .\build\windows-native-cuda-only-nvcc -j 16; if ($LASTEXITCODE -ne 0) { throw 'CUDA-only build failed' }
```

Run the updated smoke, including Feistel CNG key vectors:

```powershell
cd C:\dev\xchplot2; .\build\windows-native-cuda-only-nvcc\native_cuda_smoke.exe; if ($LASTEXITCODE -ne 0) { throw 'CUDA-only smoke failed' }
```

No large VRAM allocation in this plan query:

```powershell
cd C:\dev\xchplot2; .\build\windows-native-cuda-only-nvcc\native_cuda_pipeline.exe --plan; if ($LASTEXITCODE -ne 0) { throw 'CUDA-only plan failed' }
```

Only after compilation, smoke and plan pass, opt-in to the full
compute-only RTX 4070 SUPER run:

```powershell
cd C:\dev\xchplot2; .\build\windows-native-cuda-only-nvcc\native_cuda_pipeline.exe --run; if ($LASTEXITCODE -ne 0) { throw 'CUDA-only full compute failed' }
```

On the first run, report the count/digest/first error to compare
against the reference pipeline; **avoid rerunning the old ON/OFF benchmarks**.
The compute-only diagnostic intentionally writes no plot or test files.

## Remaining native CUDA-only work

AdaptiveCpp/SYCL is still required for device discovery, CUDA-USM allocation/queues, pooled GPU host orchestration, and some distributed/sharded/streaming and fallback kernels. Those are **not** removed because doing so before porting the host layer would break plot creation. The CUDA-specific branch isolates the new code without altering main or the refactor branch. There is no claimed speedup until the k28/s6 hardware verification above.

## Follow-up migration: CUDA-only runtime foundation (unverified)

The verified first CUDA core commit `4fafeb2d` is preserved as a separate rollback boundary. Later work adds:

- `NativeCudaPeer.cu`: native peer/distributed radix-sort count and atomic scatter kernels (six entry points), dispatching through `SortDistributedPeer.cpp`.
- `NativeCudaTwoLevel.cu`: native top-bit histogram and partition; segmented inner sort already uses CUDA CUB.
- `NativeCudaRuntime.hpp/.cpp`: **SYCL-independent** host-side CUDA device enumeration, stream ownership, device/pinned allocations, synchronous H2D/D2H/D2D copies, memset and memory information. This is the basis for replacing the SYCL-dependent buffer pool/host pipeline later; it is **not yet wired into the plot CLI**.
- `tools/native_cuda_only/`: standalone CUDA-only CMake project and `native_cuda_smoke`, which requires **neither AdaptiveCpp nor SYCL**. It checks that the selected CUDA device is RTX 4070 SUPER, exercises the native runtime, checks Xs AES output against a separate CPU T-table reference over 4096 items, tests two-level histogram/partition, peer count/scatter and gather (multiset-parity where order is intentionally nondeterministic).
- The CMake CUDA build now selects the native CUDA core unconditionally; the separate `XCHPLOT2_NATIVE_CUDA_MATCH` user ON/OFF option was removed. This is a branch-specific policy, not a global main change.
- The bench help was corrected: `-k` and `-s` are unsupported in the fixed k28/s6 profile.

### Standalone CUDA-only check (VS2022 Developer PowerShell, one line each)

The first Windows standalone link attempt used AdaptiveCpp's `clang++.exe`
as the CXX linker and failed with missing `cudadevrt.lib` /
`cudart_static.lib` despite every CUDA object compiling. This is a
Windows clang++/CMake CUDA implicit-library search-path issue, **not**
a CUDA kernel compilation error. The standalone CMake project now
enables only the CUDA language and uses nvcc for the C++ host facade
and final executable link. A fresh build directory
(`build/windows-native-cuda-only-nvcc`) is required to avoid
reusing the old cached Clang CXX compiler. On Windows, run from
VS2022 Developer PowerShell so nvcc can find MSVC `cl.exe`.

```powershell
cd C:\dev\xchplot2; $env:PATH="$env:CUDA_PATH\bin;$env:PATH"; cmake -S .\tools\native_cuda_only -B .\build\windows-native-cuda-only-nvcc -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
```

```powershell
cd C:\dev\xchplot2; cmake --build .\build\windows-native-cuda-only-nvcc -j 16
```

```powershell
cd C:\dev\xchplot2; .\build\windows-native-cuda-only-nvcc\native_cuda_smoke.exe; if ($LASTEXITCODE -ne 0) { throw 'native CUDA-only smoke failed' }
```

This standalone target tests native GPU primitives and runtime independently. It is **not** a full Plot executable and does **not** validate the Feistel/Match kernels in isolation; the previously verified hybrid CLI covers those. A complete CUDA-only Plot CLI still requires the host pipeline, buffer pool, device/queue abstraction and residual SYCL kernels to be ported before AdaptiveCpp can be removed from the main CMake project.
