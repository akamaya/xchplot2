cmake_minimum_required(VERSION 3.24)
# Standalone native CUDA validation: intentionally no AdaptiveCpp, SYCL,
# LLVM, HIP or OpenSSL dependency. This is NOT the full plot CLI.
project(xchplot2_native_cuda_smoke LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()
find_package(CUDAToolkit REQUIRED)
get_filename_component(_x2_root "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
add_executable(native_cuda_smoke
    smoke.cu
    "${_x2_root}/src/gpu/NativeCudaXs.cu"
    "${_x2_root}/src/gpu/NativeCudaIndex.cu"
    "${_x2_root}/src/gpu/NativeCudaPipeline.cu"
    "${_x2_root}/src/gpu/NativeCudaPeer.cu"
    "${_x2_root}/src/gpu/NativeCudaTwoLevel.cu")
target_include_directories(native_cuda_smoke PRIVATE "${_x2_root}/src")
target_link_libraries(native_cuda_smoke PRIVATE CUDA::cudart)
target_compile_options(native_cuda_smoke PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
set_target_properties(native_cuda_smoke PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF)
message(STATUS "Native CUDA standalone smoke: no AdaptiveCpp/SYCL dependency")
