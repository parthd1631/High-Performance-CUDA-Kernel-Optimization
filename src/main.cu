#include "kernels.cuh"

#include <cuda_runtime.h>

#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

struct Config {
    int gemm_size = 1024;
    int reduction_size = 1 << 24;
    int warmup_iters = 5;
    int timed_iters = 20;
};

void print_usage(const char* exe) {
    std::cout
        << "Usage: " << exe << " [options]\n"
        << "Options:\n"
        << "  --gemm-size <int>       Square matrix size for GEMM benchmark (default: 1024)\n"
        << "  --reduction-size <int>  Number of elements for reduction benchmark (default: 16777216)\n"
        << "  --warmup <int>          Warmup iterations (default: 5)\n"
        << "  --iters <int>           Timed iterations (default: 20)\n"
        << "  --help                  Show help\n";
}

Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--gemm-size" && i + 1 < argc) {
            cfg.gemm_size = std::stoi(argv[++i]);
        } else if (arg == "--reduction-size" && i + 1 < argc) {
            cfg.reduction_size = std::stoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            cfg.warmup_iters = std::stoi(argv[++i]);
        } else if (arg == "--iters" && i + 1 < argc) {
            cfg.timed_iters = std::stoi(argv[++i]);
        } else if (arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            std::cerr << "Unknown argument: " << arg << "\n\n";
            print_usage(argv[0]);
            std::exit(1);
        }
    }
    return cfg;
}

} // namespace

int main(int argc, char** argv) {
    Config cfg = parse_args(argc, argv);

    cudaDeviceProp prop{};
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        std::cerr << "Failed to query CUDA device: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    std::cout << "=== High Performance CUDA Kernel Optimization ===\n";
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "GEMM size: " << cfg.gemm_size << "x" << cfg.gemm_size << "\n";
    std::cout << "Reduction elements: " << cfg.reduction_size << "\n";
    std::cout << "Warmup iterations: " << cfg.warmup_iters << ", Timed iterations: " << cfg.timed_iters << "\n\n";

    std::cout << std::fixed << std::setprecision(4);

    GemmResult gemm_naive = benchmark_gemm_naive(cfg.gemm_size, cfg.warmup_iters, cfg.timed_iters);
    GemmResult gemm_opt = benchmark_gemm_optimized(cfg.gemm_size, cfg.warmup_iters, cfg.timed_iters);

    ReductionResult red_naive =
        benchmark_reduction_naive(cfg.reduction_size, cfg.warmup_iters, cfg.timed_iters);
    ReductionResult red_opt =
        benchmark_reduction_optimized(cfg.reduction_size, cfg.warmup_iters, cfg.timed_iters);

    std::cout << "GEMM Benchmark\n";
    std::cout << "  Naive:      " << gemm_naive.ms << " ms, " << gemm_naive.gflops << " GFLOP/s\n";
    std::cout << "  Optimized:  " << gemm_opt.ms << " ms, " << gemm_opt.gflops << " GFLOP/s\n";
    std::cout << "  Speedup:    " << (gemm_naive.ms / gemm_opt.ms) << "x\n";
    std::cout << "  Max |diff|: " << gemm_opt.max_abs_error << "\n\n";

    std::cout << "Reduction Benchmark\n";
    std::cout << "  Naive:      " << red_naive.ms << " ms, " << red_naive.bandwidth_gbps << " GB/s\n";
    std::cout << "  Optimized:  " << red_opt.ms << " ms, " << red_opt.bandwidth_gbps << " GB/s\n";
    std::cout << "  Speedup:    " << (red_naive.ms / red_opt.ms) << "x\n";
    std::cout << "  |sum diff|: " << red_opt.abs_error << "\n";

    return 0;
}
