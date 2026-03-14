cmake_minimum_required(VERSION 3.24)
project(high_performance_cuda_kernel_optimization LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)

option(ENABLE_FAST_MATH "Enable CUDA fast math optimizations" ON)

add_executable(cuda_kernels_bench
    src/main.cu
    src/kernels.cu
)

target_include_directories(cuda_kernels_bench PRIVATE include)
target_compile_options(cuda_kernels_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

if(NOT ENABLE_FAST_MATH)
    target_compile_options(cuda_kernels_bench PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-fmad=false>
    )
endif()
