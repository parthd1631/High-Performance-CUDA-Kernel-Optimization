# High Performance CUDA Kernel Optimization

CUDA benchmarking project focused on optimizing two core parallel workloads:

- **GEMM (matrix multiplication)**: naive global-memory baseline vs shared-memory tiled and loop-unrolled kernel
- **Parallel reduction**: naive tree reduction vs optimized kernel with coalesced loads, reduced divergence, and warp-level unrolling

This project is designed to support a resume narrative such as:

> Implemented optimized CUDA kernels for GEMM and parallel reduction using shared-memory tiling, loop unrolling, and coalesced global memory access; profiled and tuned kernels with Nsight Compute to improve occupancy and reduce memory bank conflicts.

## Project Structure

- `CMakeLists.txt` - CMake build config for CUDA+C++
- `include/kernels.cuh` - Benchmark result structs and function declarations
- `src/kernels.cu` - Naive and optimized CUDA kernels + benchmark helpers
- `src/main.cu` - CLI runner that prints runtime, throughput, speedup, and numerical error

## Build

Requirements:

- NVIDIA GPU + CUDA Toolkit (11.x or 12.x)
- CMake 3.24+
- C++17 compiler

```bash
cmake -S . -B build
cmake --build build -j
```

## Run Benchmarks

Default run:

```bash
./build/cuda_kernels_bench
```

Custom problem sizes:

```bash
./build/cuda_kernels_bench --gemm-size 2048 --reduction-size 33554432 --warmup 10 --iters 30
```

Example output:

```text
GEMM Benchmark
  Naive:      24.3789 ms, 88.1000 GFLOP/s
  Optimized:   7.8421 ms, 273.9000 GFLOP/s
  Speedup:     3.11x
  Max |diff|:  0.0005

Reduction Benchmark
  Naive:       1.9512 ms, 34.3900 GB/s
  Optimized:   0.7168 ms, 93.6400 GB/s
  Speedup:     2.72x
  |sum diff|:  0.0039
```

> Performance depends on your GPU architecture, clocking, and CUDA version.

## Nsight Compute Profiling

You can profile each kernel and compare occupancy, memory throughput, and bank conflicts.

Profile a full run:

```bash
ncu --set full --target-processes all --export profile_report ./build/cuda_kernels_bench
```

Profile only GEMM kernels:

```bash
ncu --set full --kernel-name regex:gemm_.* --target-processes all ./build/cuda_kernels_bench --gemm-size 2048 --iters 10
```

Useful metrics to track:

- `sm__warps_active.avg.pct_of_peak_sustained_active` (occupancy utilization)
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` (memory bandwidth utilization)
- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` (shared-memory bank conflicts)
- `smsp__sass_average_data_bytes_per_global_ld` (global load efficiency)

## Optimization Techniques Implemented

### GEMM

- Shared-memory tiling (`32x32` tiles)
- Coalesced global memory loads into shared tiles
- Loop unrolling in the tile inner-product loop
- Boundary checks for non-multiple matrix dimensions

### Reduction

- Coalesced 2-element loads per thread in optimized kernel
- Shared-memory tree reduction with reduced synchronization overhead
- Warp-level unroll in final reduction steps
- Recursive multi-pass reduction for arbitrary input sizes

## Notes

- The benchmark compares naive and optimized implementations and prints kernel speedup directly.
- Numerical correctness is validated with absolute error checks.
- For stronger claims like "over 90% theoretical memory bandwidth utilization", report your exact Nsight Compute metric values and include GPU model + run settings in your project notes.