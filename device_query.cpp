#include <iostream>
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>

typedef long long int64;

const int64 mod = 12289; // Realistic modulus used in lattice crypto
const int64 root = 11;   // Primitive root of unity mod 12289
const int N = 1024;      // Polynomial degree (power of 2)

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

int main() {
    int64* h_A = new int64[N];
    int64* h_B = new int64[N];
    
    srand(42);
    for (int i = 0; i < N; ++i) {
        h_A[i] = rand() % mod;
        h_B[i] = rand() % mod;
    }
    
    int64 *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(int64));
    cudaMalloc(&d_B, N * sizeof(int64));
    cudaMalloc(&d_C, N * sizeof(int64));
    
    cudaMemcpy(d_A, h_A, N * sizeof(int64), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(int64), cudaMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    // Setup CUDA events
    cudaEvent_t start, stop;
    float t_ntt_fwd, t_mult, t_ntt_inv, t_total;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    ntt_kernel<<<blocks, threads>>>(d_A, N, root, false);
    ntt_kernel<<<blocks, threads>>>(d_B, N, root, false);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t_ntt_fwd, start, stop);
    
    cudaEventRecord(start);
    pointwise_mult_kernel<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t_mult, start, stop);
    
    cudaEventRecord(start);
    ntt_kernel<<<blocks, threads>>>(d_C, N, root, true);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t_ntt_inv, start, stop);
    
    // Total time = sum of above
    t_total = t_ntt_fwd + t_mult + t_ntt_inv;
    
    // Copy back and display results
    int64* result = new int64[N];
    cudaMemcpy(result, d_C, N * sizeof(int64), cudaMemcpyDeviceToHost);
    
    std::cout << "Result (first 10 coeffs): ";
    for (int i = 0; i < 10; ++i)
        std::cout << result[i] << " ";
    std::cout << "...\n";
    
    std::cout << "Benchmark Results:\n";
    std::cout << " Forward NTT time : " << t_ntt_fwd << " ms\n";
    std::cout << " Pointwise mult  : " << t_mult << " ms\n";
    std::cout << " Inverse NTT time: " << t_ntt_inv << " ms\n";
    std::cout << " Total GPU time  : " << t_total << " ms\n";
    
    delete[] h_A;
    delete[] h_B;
    delete[] result;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}