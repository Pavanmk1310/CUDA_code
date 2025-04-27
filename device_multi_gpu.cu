#include <iostream>
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>

typedef long long int64;

const int64 mod = 12289;
const int64 root = 11;
const int N = 1024;

// Helper functions for modular arithmetic
__device__ int64 modmul(int64 a, int64 b) {
    return (a * b) % mod;
}

__device__ int64 modadd(int64 a, int64 b) {
    return (a + b) % mod;
}

__device__ int64 modsub(int64 a, int64 b) {
    return (a - b + mod) % mod;
}

__host__ __device__ int64 modpow(int64 base, int64 exp, int64 m) {
    int64 res = 1;
    base %= m;
    while (exp > 0) {
        if (exp & 1) res = res * base % m;
        base = base * base % m;
        exp >>= 1;
    }
    return res;
}

// Kernel for a single stage of NTT
__global__ void ntt_stage_kernel(int64* a, int n, int len, int64 root_power, bool invert) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int group_size = len / 2;
    
    // Calculate which group and position within the group this thread is processing
    int group = idx / group_size;
    int pos = idx % group_size;
    
    // Calculate actual index to process
    int i = group * len + pos;
    
    // Ensure we're within bounds
    if (i + group_size < n) {
        // Calculate twiddle factor
        int64 wlen = root_power;
        if (invert) wlen = modpow(wlen, mod - 2, mod);
        
        int64 w = modpow(wlen, pos, mod);
        
        // Perform butterfly operation
        int64 u = a[i];
        int64 v = modmul(a[i + group_size], w);
        a[i] = modadd(u, v);
        a[i + group_size] = modsub(u, v);
    }
}

// Kernel for pointwise multiplication
__global__ void pointwise_mult_kernel(int64* A, int64* B, int64* C, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        C[i] = modmul(A[i], B[i]);
    }
}

// Kernel for applying inverse NTT scaling factor
__global__ void scale_kernel(int64* a, int n, int64 n_inv) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        a[i] = modmul(a[i], n_inv);
    }
}

// Function to perform complete NTT
void perform_ntt(int64* d_data, int n, bool invert, cudaStream_t stream = 0) {
    int threads = 256;
    
    // Calculate the power of the primitive root for each stage
    int64 primitive_root = root;
    
    // For each stage of the NTT
    for (int len = 2; len <= n; len <<= 1) {
        int group_size = len / 2;
        int num_groups = n / len;
        int total_threads = num_groups * group_size;
        int blocks = (total_threads + threads - 1) / threads;
        
        // Calculate root power for this stage
        int64 root_power = modpow(primitive_root, (mod - 1) / len, mod);
        
        // Launch kernel for this stage
        ntt_stage_kernel<<<blocks, threads, 0, stream>>>(d_data, n, len, root_power, invert);
    }
    
    // Apply scaling factor for inverse NTT
    if (invert) {
        int blocks = (n + threads - 1) / threads;
        int64 n_inv = modpow(n, mod - 2, mod);
        scale_kernel<<<blocks, threads, 0, stream>>>(d_data, n, n_inv);
    }
}

int main() {
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count < 2) {
        std::cerr << "Need at least 2 GPUs!" << std::endl;
        return 1;
    }

    int64* h_A = new int64[N];
    int64* h_B = new int64[N];

    srand(42);
    for (int i = 0; i < N; ++i) {
        h_A[i] = rand() % mod;
        h_B[i] = rand() % mod;
    }

    int64 *d_A0, *d_B0, *d_C0;
    int64 *d_A1, *d_B1, *d_C1;
    int half_N = N / 2;

    // Create streams for asynchronous operations
    cudaStream_t stream0, stream1;
    
    // Create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // GPU 0 setup
    cudaSetDevice(0);
    cudaStreamCreate(&stream0);
    cudaMalloc(&d_A0, half_N * sizeof(int64));
    cudaMalloc(&d_B0, half_N * sizeof(int64));
    cudaMalloc(&d_C0, half_N * sizeof(int64));
    cudaMemcpyAsync(d_A0, h_A, half_N * sizeof(int64), cudaMemcpyHostToDevice, stream0);
    cudaMemcpyAsync(d_B0, h_B, half_N * sizeof(int64), cudaMemcpyHostToDevice, stream0);

    // GPU 1 setup
    cudaSetDevice(1);
    cudaStreamCreate(&stream1);
    cudaMalloc(&d_A1, half_N * sizeof(int64));
    cudaMalloc(&d_B1, half_N * sizeof(int64));
    cudaMalloc(&d_C1, half_N * sizeof(int64));
    cudaMemcpyAsync(d_A1, h_A + half_N, half_N * sizeof(int64), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_B1, h_B + half_N, half_N * sizeof(int64), cudaMemcpyHostToDevice, stream1);

    // Calculate kernel launch parameters
    int threads = 256;
    int blocks = (half_N + threads - 1) / threads;

    // Start timing
    cudaSetDevice(0);
    cudaEventRecord(start);

    // GPU 0: Forward NTT for both arrays
    cudaSetDevice(0);
    perform_ntt(d_A0, half_N, false, stream0);
    perform_ntt(d_B0, half_N, false, stream0);

    // GPU 1: Forward NTT for both arrays
    cudaSetDevice(1);
    perform_ntt(d_A1, half_N, false, stream1);
    perform_ntt(d_B1, half_N, false, stream1);

    // GPU 0: Pointwise multiplication
    cudaSetDevice(0);
    pointwise_mult_kernel<<<blocks, threads, 0, stream0>>>(d_A0, d_B0, d_C0, half_N);

    // GPU 1: Pointwise multiplication
    cudaSetDevice(1);
    pointwise_mult_kernel<<<blocks, threads, 0, stream1>>>(d_A1, d_B1, d_C1, half_N);

    // GPU 0: Inverse NTT
    cudaSetDevice(0);
    perform_ntt(d_C0, half_N, true, stream0);

    // GPU 1: Inverse NTT
    cudaSetDevice(1);
    perform_ntt(d_C1, half_N, true, stream1);

    // Stop timing
    cudaSetDevice(0);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy results back to host
    int64* result = new int64[N];
    
    cudaSetDevice(0);
    cudaMemcpyAsync(result, d_C0, half_N * sizeof(int64), cudaMemcpyDeviceToHost, stream0);
    cudaStreamSynchronize(stream0);
    
    cudaSetDevice(1);
    cudaMemcpyAsync(result + half_N, d_C1, half_N * sizeof(int64), cudaMemcpyDeviceToHost, stream1);
    cudaStreamSynchronize(stream1);

    // Print results
    std::cout << "Result (first 10 coeffs): ";
    for (int i = 0; i < 10; ++i)
        std::cout << result[i] << " ";
    std::cout << "...\n";

    std::cout << "Total GPU time (both GPUs): " << milliseconds << " ms\n";

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] result;

    cudaSetDevice(0);
    cudaFree(d_A0);
    cudaFree(d_B0);
    cudaFree(d_C0);
    cudaStreamDestroy(stream0);
    
    cudaSetDevice(1);
    cudaFree(d_A1);
    cudaFree(d_B1);
    cudaFree(d_C1);
    cudaStreamDestroy(stream1);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}