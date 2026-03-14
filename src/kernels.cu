#include "kernels.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t err = (call);                                                                  \
        if (err != cudaSuccess) {                                                                  \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " -> "                \
                      << cudaGetErrorString(err) << std::endl;                                     \
            std::exit(EXIT_FAILURE);                                                                \
        }                                                                                          \
    } while (0)

namespace {

constexpr int GEMM_TILE = 32;
constexpr int REDUCE_BLOCK_SIZE = 256;

__global__ void gemm_naive_kernel(const float* a, const float* b, float* c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n || col >= n) {
        return;
    }

    float sum = 0.0f;
    for (int k = 0; k < n; ++k) {
        sum += a[row * n + k] * b[k * n + col];
    }
    c[row * n + col] = sum;
}

__global__ void gemm_tiled_optimized_kernel(const float* a, const float* b, float* c, int n) {
    __shared__ float a_tile[GEMM_TILE][GEMM_TILE];
    __shared__ float b_tile[GEMM_TILE][GEMM_TILE];

    int row = blockIdx.y * GEMM_TILE + threadIdx.y;
    int col = blockIdx.x * GEMM_TILE + threadIdx.x;

    float acc = 0.0f;
    int tiles = (n + GEMM_TILE - 1) / GEMM_TILE;

    for (int tile = 0; tile < tiles; ++tile) {
        int a_col = tile * GEMM_TILE + threadIdx.x;
        int b_row = tile * GEMM_TILE + threadIdx.y;

        a_tile[threadIdx.y][threadIdx.x] = (row < n && a_col < n) ? a[row * n + a_col] : 0.0f;
        b_tile[threadIdx.y][threadIdx.x] = (b_row < n && col < n) ? b[b_row * n + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < GEMM_TILE; ++k) {
            acc += a_tile[threadIdx.y][k] * b_tile[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        c[row * n + col] = acc;
    }
}

__global__ void reduce_naive_kernel(const float* in, float* out, int n) {
    __shared__ float shared[REDUCE_BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    shared[tid] = (idx < n) ? in[idx] : 0.0f;
    __syncthreads();

    // Baseline reduction with less efficient addressing and no warp-level unroll.
    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        if ((tid % (stride << 1)) == 0 && (tid + stride) < blockDim.x) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = shared[0];
    }
}

__global__ void reduce_optimized_kernel(const float* in, float* out, int n) {
    __shared__ float shared[REDUCE_BLOCK_SIZE];
    int tid = threadIdx.x;
    int global_idx = blockIdx.x * (blockDim.x * 2) + tid;

    float value = 0.0f;
    if (global_idx < n) {
        value = in[global_idx];
    }
    int second = global_idx + blockDim.x;
    if (second < n) {
        value += in[second];
    }
    shared[tid] = value;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid < 32) {
        volatile float* vmem = shared;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    if (tid == 0) {
        out[blockIdx.x] = shared[0];
    }
}

void fill_random(std::vector<float>& data, float min_value = -1.0f, float max_value = 1.0f) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(min_value, max_value);
    for (float& v : data) {
        v = dist(rng);
    }
}

float compute_max_abs_error(const std::vector<float>& a, const std::vector<float>& b) {
    float max_error = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        max_error = std::max(max_error, std::abs(a[i] - b[i]));
    }
    return max_error;
}

float run_reduction_recursive(float* d_input, int n, bool optimized) {
    float* d_curr_in = d_input;
    float* d_curr_out = nullptr;
    std::vector<float*> to_free;

    int curr_n = n;
    while (curr_n > 1) {
        int grid = optimized ? (curr_n + (REDUCE_BLOCK_SIZE * 2 - 1)) / (REDUCE_BLOCK_SIZE * 2)
                             : (curr_n + REDUCE_BLOCK_SIZE - 1) / REDUCE_BLOCK_SIZE;
        CUDA_CHECK(cudaMalloc(&d_curr_out, grid * sizeof(float)));
        to_free.push_back(d_curr_out);

        if (optimized) {
            reduce_optimized_kernel<<<grid, REDUCE_BLOCK_SIZE>>>(d_curr_in, d_curr_out, curr_n);
        } else {
            reduce_naive_kernel<<<grid, REDUCE_BLOCK_SIZE>>>(d_curr_in, d_curr_out, curr_n);
        }
        CUDA_CHECK(cudaGetLastError());

        d_curr_in = d_curr_out;
        curr_n = grid;
    }

    float sum = 0.0f;
    CUDA_CHECK(cudaMemcpy(&sum, d_curr_in, sizeof(float), cudaMemcpyDeviceToHost));
    for (float* ptr : to_free) {
        CUDA_CHECK(cudaFree(ptr));
    }
    return sum;
}

GemmResult benchmark_gemm_common(int n, int warmup_iters, int timed_iters, bool optimized) {
    const size_t bytes = static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(float);
    const size_t elements = static_cast<size_t>(n) * static_cast<size_t>(n);

    std::vector<float> h_a(elements), h_b(elements), h_out(elements), h_ref(elements);
    fill_random(h_a);
    fill_random(h_b);

    float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr, *d_ref = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_ref, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    dim3 tile_block(GEMM_TILE, GEMM_TILE);
    dim3 tile_grid((n + GEMM_TILE - 1) / GEMM_TILE, (n + GEMM_TILE - 1) / GEMM_TILE);

    for (int i = 0; i < warmup_iters; ++i) {
        if (optimized) {
            gemm_tiled_optimized_kernel<<<tile_grid, tile_block>>>(d_a, d_b, d_out, n);
        } else {
            gemm_naive_kernel<<<grid, block>>>(d_a, d_b, d_out, n);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < timed_iters; ++i) {
        if (optimized) {
            gemm_tiled_optimized_kernel<<<tile_grid, tile_block>>>(d_a, d_b, d_out, n);
        } else {
            gemm_naive_kernel<<<grid, block>>>(d_a, d_b, d_out, n);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / static_cast<float>(timed_iters);

    // Use naive GEMM as a numerical reference output.
    gemm_naive_kernel<<<grid, block>>>(d_a, d_b, d_ref, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ref.data(), d_ref, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_ref));

    float gflops = (2.0f * n * n * n) / (avg_ms * 1.0e6f);
    return GemmResult{
        avg_ms,
        gflops,
        compute_max_abs_error(h_out, h_ref),
    };
}

ReductionResult benchmark_reduction_common(int num_elements, int warmup_iters, int timed_iters, bool optimized) {
    const size_t bytes = static_cast<size_t>(num_elements) * sizeof(float);
    std::vector<float> h_in(num_elements);
    fill_random(h_in, 0.0f, 1.0f);

    double cpu_sum = 0.0;
    for (float v : h_in) {
        cpu_sum += static_cast<double>(v);
    }

    float* d_in = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    for (int i = 0; i < warmup_iters; ++i) {
        (void)run_reduction_recursive(d_in, num_elements, optimized);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    float gpu_sum = 0.0f;
    for (int i = 0; i < timed_iters; ++i) {
        gpu_sum = run_reduction_recursive(d_in, num_elements, optimized);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / static_cast<float>(timed_iters);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));

    // One-pass effective bandwidth estimate (input bytes / average kernel time).
    float bandwidth = (static_cast<float>(bytes) / 1.0e9f) / (avg_ms / 1000.0f);
    return ReductionResult{
        avg_ms,
        bandwidth,
        std::abs(gpu_sum - static_cast<float>(cpu_sum)),
    };
}

} // namespace

GemmResult benchmark_gemm_naive(int n, int warmup_iters, int timed_iters) {
    return benchmark_gemm_common(n, warmup_iters, timed_iters, false);
}

GemmResult benchmark_gemm_optimized(int n, int warmup_iters, int timed_iters) {
    return benchmark_gemm_common(n, warmup_iters, timed_iters, true);
}

ReductionResult benchmark_reduction_naive(int num_elements, int warmup_iters, int timed_iters) {
    return benchmark_reduction_common(num_elements, warmup_iters, timed_iters, false);
}

ReductionResult benchmark_reduction_optimized(int num_elements, int warmup_iters, int timed_iters) {
    return benchmark_reduction_common(num_elements, warmup_iters, timed_iters, true);
}
