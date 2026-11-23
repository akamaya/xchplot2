cmake_minimum_required(VERSION 3.24)
# Standalone native CUDA validation: intentionally no AdaptiveCpp, SYCL,
# LLVM, HIP or OpenSSL dependency. This is NOT the full plot CLI.
# CUDA is the only CMake language here. Using CXX as well could select
# AdaptiveCpp's clang++ from PATH, whose GNU-like Windows linker mishandles
# CUDA's implicit cudadevrt.lib/cudart_static.lib search path.
# Nvcc uses the VS2022 MSVC toolchain to compile AND link.
project(xchplot2_native_cuda_smoke LANGUAGES CUDA)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()
find_package(CUDAToolkit REQUIRED)
get_filename_component(_x2_root "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
# NativeCudaRuntime.cpp contains ordinary C++, but it is compiled by nvcc
# to avoid the standalone build accidentally depending on clang++.
set_source_files_properties(
    "${_x2_root}/src/gpu/NativeCudaRuntime.cpp"
    "${_x2_root}/src/gpu/NativeCudaPool.cpp"
    PROPERTIES LANGUAGE CUDA)

add_executable(native_cuda_smoke
    smoke.cu
    "${_x2_root}/src/gpu/NativeCudaRuntime.cpp"
    "${_x2_root}/src/gpu/NativeCudaPool.cpp"
    "${_x2_root}/src/gpu/SortCuda.cu"
    "${_x2_root}/src/gpu/NativeCudaXs.cu"
    "${_x2_root}/src/gpu/NativeCudaIndex.cu"
    "${_x2_root}/src/gpu/NativeCudaPipeline.cu"
    "${_x2_root}/src/gpu/NativeCudaPeer.cu"
    "${_x2_root}/src/gpu/NativeCudaTwoLevel.cu")
target_include_directories(native_cuda_smoke PRIVATE "${CMAKE_CURRENT_LIST_DIR}" "${_x2_root}/src")
target_link_libraries(native_cuda_smoke PRIVATE CUDA::cudart bcrypt)
target_compile_options(native_cuda_smoke PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
set_target_properties(native_cuda_smoke PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    LINKER_LANGUAGE CUDA)
message(STATUS "Native CUDA standalone smoke: no AdaptiveCpp/SYCL dependency")

# Independent experimental k28/s6 compute pipeline. The normal .gplot
# writer is NOT linked here: native_cuda_pipeline only reports counts and
# SHA256 of sorted T3 fragments. Run --plan first; --run is opt-in and
# allocates the full 12-GB-card Candidate D working set.
set_source_files_properties(
    "${_x2_root}/src/gpu/NativeCudaStandalonePipeline.cpp"
    PROPERTIES LANGUAGE CUDA)
add_executable(native_cuda_pipeline
    pipeline_main.cu
    "${_x2_root}/src/gpu/NativeCudaStandalonePipeline.cpp"
    "${_x2_root}/src/gpu/NativeCudaMatch.cu"
    "${_x2_root}/src/gpu/NativeCudaRuntime.cpp"
    "${_x2_root}/src/gpu/NativeCudaPool.cpp"
    "${_x2_root}/src/gpu/SortCuda.cu"
    "${_x2_root}/src/gpu/NativeCudaXs.cu"
    "${_x2_root}/src/gpu/NativeCudaIndex.cu"
    "${_x2_root}/src/gpu/NativeCudaPipeline.cu")
# Use this project's private, Windows CNG-backed FeistelCipher key schedule.
# It has the same FullRoundKey layout as pos2-chip but does not require
# the external pos2-chip library or OpenSSL.
target_include_directories(native_cuda_pipeline PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}" "${_x2_root}/src")
target_link_libraries(native_cuda_pipeline PRIVATE CUDA::cudart bcrypt)
target_compile_options(native_cuda_pipeline PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
set_target_properties(native_cuda_pipeline PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF LINKER_LANGUAGE CUDA)
message(STATUS "Native CUDA compute runner: pure nvcc + Windows CNG (no SYCL)")
