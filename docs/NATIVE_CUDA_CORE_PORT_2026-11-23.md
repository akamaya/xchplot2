# RTX 4070 SUPER k28/s6 — CUDA core migration (2026-11-23)

## Status and provenance

- **Base**: refactor/always-on-verified-gpu-optimizations-20261123 at `f6646911faf1a793532abbd06cf55e2696bbe046` (not main).
- **Work branch**: `cuda/nvidia-12gb-k28s6`.
- **No legacy runtime ON/OFF switches restored**. Verified Compact, Direct24/Direct20, Tezcan and AES Experiment A remain the fixed k28/s6 production profile.
- **Implementation committed, not yet nvcc-compiled or hardware-verified.** The existing SYCL 25.18 s/plot result is a historical baseline, **not** a CUDA result.
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
cd C:\dev\xchplot2; $env:ACPP_PREFIX=(Resolve-Path .\build\windows-toolchain\install).Path; $env:PATH="$env:ACPP_PREFIX\bin;$env:CUDA_PATH\bin;$env:PATH"; $openssl=(Resolve-Path .\build\windows-openssl\install).Path; cmake -S . -B .\build\windows-cuda-core -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_C_COMPILER=$env:ACPP_PREFIX/bin/clang.exe" "-DCMAKE_CXX_COMPILER=$env:ACPP_PREFIX/bin/clang++.exe" "-DOPENSSL_ROOT_DIR:PATH=$openssl" -DOPENSSL_USE_STATIC_LIBS=ON '-DACPP_TARGETS=generic;omp' -DXCHPLOT2_BUILD_CUDA=ON -DXCHPLOT2_NATIVE_CUDA_MATCH=ON -DCMAKE_CUDA_ARCHITECTURES=89
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
5. Measure `bench -k 28 -s 6 -n 10 --warmup 1 --devices gpu0`. Compare with the **historical SYCL steady-state of 25.18 s/plot**. **No new OFF remeasurement is needed** unless the toolchain/environment changes.
6. If possible, test one overflow-heavy synthetic case and independent exact-index parity. The build is not yet certified.

## Remaining native CUDA-only work

AdaptiveCpp/SYCL is still required for device discovery, CUDA-USM allocation/queues, pooled GPU host orchestration, and some distributed/sharded/streaming and fallback kernels. Those are **not** removed because doing so before porting the host layer would break plot creation. The CUDA-specific branch isolates the new code without altering main or the refactor branch. There is no claimed speedup until the k28/s6 hardware verification above.
