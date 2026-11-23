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
target_include_directories(native_cuda_smoke PRIVATE "${_x2_root}/src")
target_link_libraries(native_cuda_smoke PRIVATE CUDA::cudart)
target_compile_options(native_cuda_smoke PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
set_target_properties(native_cuda_smoke PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    LINKER_LANGUAGE CUDA)
message(STATUS "Native CUDA standalone smoke: no AdaptiveCpp/SYCL dependency")
