#pragma once

#include <cuda_runtime.h>

struct GemmResult {
    float ms = 0.0f;
    float gflops = 0.0f;
    float max_abs_error = 0.0f;
};

struct ReductionResult {
    float ms = 0.0f;
    float bandwidth_gbps = 0.0f;
    float abs_error = 0.0f;
};

GemmResult benchmark_gemm_naive(int n, int warmup_iters, int timed_iters);
GemmResult benchmark_gemm_optimized(int n, int warmup_iters, int timed_iters);

ReductionResult benchmark_reduction_naive(int num_elements, int warmup_iters, int timed_iters);
ReductionResult benchmark_reduction_optimized(int num_elements, int warmup_iters, int timed_iters);
