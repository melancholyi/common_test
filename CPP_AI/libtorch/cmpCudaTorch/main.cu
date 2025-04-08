#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <chrono>
#include <cstdlib>
#include <matplotlibcpp.h>

using namespace std;
namespace plt = matplotlibcpp;
// CUDA Kernel for Matrix Multiplication
__global__ void matrixMul(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

float cuda_matrix_multiply(const vector<float>& h_A, const vector<float>& h_B, vector<float>& h_C, int N) {
    float *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(float);

    // Allocate device memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy data to device
    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);

    // Configure grid and block
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    // Timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrixMul<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    // cout << "CUDA time: " << milliseconds << " ms" << endl;

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

float libtorch_matrix_multiply(vector<float>& h_A, vector<float>& h_B, vector<float>& h_C, int N) {
    // Create tensors from host data
    torch::Tensor ta = torch::from_blob(h_A.data(), {N, N}).to(torch::kFloat32);
    torch::Tensor tb = torch::from_blob(h_B.data(), {N, N}).to(torch::kFloat32);

    
    // Move to GPU
    auto ta_gpu = ta.to(torch::kCUDA);
    auto tb_gpu = tb.to(torch::kCUDA);

    // Timing
    auto start = chrono::high_resolution_clock::now();

    // Perform matrix multiplication
    torch::Tensor tc_gpu = torch::matmul(ta_gpu, tb_gpu);

    // Move result back to CPU
    torch::Tensor tc = tc_gpu.to(torch::kCPU);

    // Copy result
    memcpy(h_C.data(), tc.data_ptr<float>(), N*N*sizeof(float));

    // Calculate elapsed time
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration = end - start;
    // cout << "LibTorch time: " << duration.count() << " ms" << endl;
    return duration.count();
}

int main(int argc, char* argv[]) {
    int NMax = 1; // 矩阵大小
    if (argc != 2) {
        std::cerr << "error input, default N = 1000" << std::endl;
        NMax = 1000; // 默认矩阵大小
    }else{
        NMax = std::atoi(argv[1]); // 将命令行参数转换为整数
        if (NMax <= 0) {
            std::cerr << "Matrix size must be a positive integer." << std::endl;
            return 1; // 如果矩阵大小无效，返回错误码1
        }
    }

    std::vector<float> x, timeCudaVec, timeTorchVec;
    for(int N = 1; N < NMax+1; N += 9){
        int size = N * N;
        vector<float> h_A(size), h_B(size), h_C_cuda(size), h_C_libtorch(size);
    
        // Initialize matrices with random values
        for (int i = 0; i < size; ++i) {
            h_A[i] = static_cast<float>(rand()) / RAND_MAX;
            h_B[i] = static_cast<float>(rand()) / RAND_MAX;
        }
    
        cout << "Matrix size: " << N << "x" << N << endl;
        // Run CUDA implementation
        float time_cuda = cuda_matrix_multiply(h_A, h_B, h_C_cuda, N);
    
        // Run LibTorch implementation
        float time_torch = libtorch_matrix_multiply(h_A, h_B, h_C_libtorch, N);
        x.push_back(N);
        timeCudaVec.push_back(time_cuda);
        timeTorchVec.push_back(time_torch);
    }

    
    plt::figure_size(1000, 600);
    plt::named_plot("timeCudaVec", x, timeCudaVec, "b-");
    plt::named_plot("timeTorchVec", x, timeTorchVec, "r--");
    plt::grid(true);
    plt::legend();
    plt::show();



    return 0;
}