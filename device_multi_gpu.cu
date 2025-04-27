#include <iostream>
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>

typedef long long int64;

const int64 mod = 12289;
const int64 root = 11;
const int N = 1024;

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

__global__ void ntt_kernel(int64* a, int n, int64 root, bool invert) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    for (int len = 2; len <= n; len <<= 1) {
        int64 wlen = modpow(root, (mod - 1) / len, mod);
        if (invert) wlen = modpow(wlen, mod - 2, mod);

        for (int i = tid * len; i < n; i += gridDim.x * blockDim.x * len) {
            int64 w = 1;
            for (int j = 0; j < len / 2; ++j) {
                int64 u = a[i + j];
                int64 v = modmul(a[i + j + len / 2], w);
                a[i + j] = modadd(u, v);
                a[i + j + len / 2] = modsub(u, v);
                w = modmul(w, wlen);
            }
        }
        __syncthreads();
    }
    if (invert && tid < n) {
        int64 n_inv = modpow(n, mod - 2, mod);
        a[tid] = modmul(a[tid], n_inv);
    }
}

__global__ void pointwise_mult_kernel(int64* A, int64* B, int64* C, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        C[i] = modmul(A[i], B[i]);
    }
}

int main(int argc, char* argv[]) {
    // Default to using all available GPUs
    int num_gpus_to_use = 0;
    
    // Check if number of GPUs is specified as command line argument
    if (argc > 1) {
        num_gpus_to_use = atoi(argv[1]);
        if (num_gpus_to_use <= 0) {
            std::cerr << "Invalid number of GPUs specified. Using all available GPUs." << std::endl;
            num_gpus_to_use = 0;
        }
    }
    
    // Get the number of available GPUs
    int available_gpus;
    cudaGetDeviceCount(&available_gpus);
    
    if (available_gpus == 0) {
        std::cerr << "No CUDA-capable devices found!" << std::endl;
        return 1;
    }
    
    // If num_gpus_to_use is 0 or greater than available, use all available
    if (num_gpus_to_use == 0 || num_gpus_to_use > available_gpus) {
        num_gpus_to_use = available_gpus;
    }
    
    std::cout << "Using " << num_gpus_to_use << " out of " << available_gpus << " available GPUs." << std::endl;
    
    // Input data
    int64* h_A = new int64[N];
    int64* h_B = new int64[N];
    int64* h_C = new int64[N]; // Result array
    
    // Initialize input data
    srand(42);
    for (int i = 0; i < N; ++i) {
        h_A[i] = rand() % mod;
        h_B[i] = rand() % mod;
    }
    
    // Calculate chunk size per GPU
    int chunk_size = (N + num_gpus_to_use - 1) / num_gpus_to_use;
    
    // Arrays to store device pointers for each GPU
    std::vector<int64*> d_A(num_gpus_to_use);
    std::vector<int64*> d_B(num_gpus_to_use);
    std::vector<int64*> d_C(num_gpus_to_use);
    std::vector<cudaStream_t> streams(num_gpus_to_use);
    
    // For timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    // Allocate memory and copy data for each GPU
    for (int dev = 0; dev < num_gpus_to_use; dev++) {
        cudaSetDevice(dev);
        cudaStreamCreate(&streams[dev]);
        
        // Calculate actual chunk size for this GPU
        int start_idx = dev * chunk_size;
        int end_idx = std::min((dev + 1) * chunk_size, N);
        int actual_chunk_size = end_idx - start_idx;
        
        if (actual_chunk_size <= 0) continue; // Skip if this GPU gets no data
        
        // Allocate memory on this GPU
        cudaMalloc(&d_A[dev], actual_chunk_size * sizeof(int64));
        cudaMalloc(&d_B[dev], actual_chunk_size * sizeof(int64));
        cudaMalloc(&d_C[dev], actual_chunk_size * sizeof(int64));
        
        // Copy data to this GPU
        cudaMemcpyAsync(d_A[dev], h_A + start_idx, actual_chunk_size * sizeof(int64), 
                       cudaMemcpyHostToDevice, streams[dev]);
        cudaMemcpyAsync(d_B[dev], h_B + start_idx, actual_chunk_size * sizeof(int64), 
                       cudaMemcpyHostToDevice, streams[dev]);
    }
    
    // Launch kernels on each GPU
    for (int dev = 0; dev < num_gpus_to_use; dev++) {
        cudaSetDevice(dev);
        
        // Calculate actual chunk size for this GPU
        int start_idx = dev * chunk_size;
        int end_idx = std::min((dev + 1) * chunk_size, N);
        int actual_chunk_size = end_idx - start_idx;
        
        if (actual_chunk_size <= 0) continue; // Skip if this GPU gets no data
        
        int threads = 256;
        int blocks = (actual_chunk_size + threads - 1) / threads;
        
        // Forward NTT
        ntt_kernel<<<blocks, threads, 0, streams[dev]>>>(d_A[dev], actual_chunk_size, root, false);
        ntt_kernel<<<blocks, threads, 0, streams[dev]>>>(d_B[dev], actual_chunk_size, root, false);
        
        // Pointwise multiplication
        pointwise_mult_kernel<<<blocks, threads, 0, streams[dev]>>>(d_A[dev], d_B[dev], d_C[dev], actual_chunk_size);
        
        // Inverse NTT
        ntt_kernel<<<blocks, threads, 0, streams[dev]>>>(d_C[dev], actual_chunk_size, root, true);
    }
    
    // Synchronize all devices and copy results back
    for (int dev = 0; dev < num_gpus_to_use; dev++) {
        cudaSetDevice(dev);
        
        // Calculate actual chunk size for this GPU
        int start_idx = dev * chunk_size;
        int end_idx = std::min((dev + 1) * chunk_size, N);
        int actual_chunk_size = end_idx - start_idx;
        
        if (actual_chunk_size <= 0) continue; // Skip if this GPU gets no data
        
        // Ensure computation is complete
        cudaStreamSynchronize(streams[dev]);
        
        // Copy result back to host
        cudaMemcpyAsync(h_C + start_idx, d_C[dev], actual_chunk_size * sizeof(int64),
                       cudaMemcpyDeviceToHost, streams[dev]);
    }
    
    // Make sure all memory copies are complete
    for (int dev = 0; dev < num_gpus_to_use; dev++) {
        cudaSetDevice(dev);
        cudaStreamSynchronize(streams[dev]);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Print results
    std::cout << "Result (first 10 coeffs): ";
    for (int i = 0; i < std::min(10, N); ++i)
        std::cout << h_C[i] << " ";
    std::cout << "...\n";
    
    std::cout << "Total GPU time (across " << num_gpus_to_use << " GPUs): " << milliseconds << " ms\n";
    
    // Clean up GPU memory
    for (int dev = 0; dev < num_gpus_to_use; dev++) {
        cudaSetDevice(dev);
        if (d_A[dev]) cudaFree(d_A[dev]);
        if (d_B[dev]) cudaFree(d_B[dev]);
        if (d_C[dev]) cudaFree(d_C[dev]);
        cudaStreamDestroy(streams[dev]);
    }
    
    // Clean up host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
