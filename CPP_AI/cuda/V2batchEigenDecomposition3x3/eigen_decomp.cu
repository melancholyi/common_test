#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
#include <cmath>

// #define NUM_MATRICES 31*80*80
#define NUM_MATRICES (927000)
#define MATRIX_SIZE 3
#define BLOCK_SIZE 4096

#define IS_PRINT_RUNTIME false
#define IS_PRINT_ERROR false

// 生成随机对称矩阵
void generateRandomSymmetricMatrices(float* matrices, int numMatrices) {
    for (int i = 0; i < numMatrices; ++i) {
        float* m = matrices + i * 9;
        for (int j = 0; j < 9; ++j) {
            m[j] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        }
        // Symmetrize matrix
        for (int row = 0; row < 3; ++row) {
            for (int col = row + 1; col < 3; ++col) {
                float avg = (m[row*3 + col] + m[col*3 + row]) / 2.0f;
                m[row*3 + col] = avg;
                m[col*3 + row] = avg;
            }
        }
    }
}

// 交换特征值和特征向量
__device__ void swap(float* a, float* b) {
    float temp = *a;
    *a = *b;
    *b = temp;
}

// 3x3对称矩阵的Jacobi特征分解
__device__ void jacobi3x3(const float* A, float* eigVals, float* eigVecs) {
    float V[9] = {1.0f, 0.0f, 0.0f,
                  0.0f, 1.0f, 0.0f,
                  0.0f, 0.0f, 1.0f};
    float B[9];
    for (int i = 0; i < 9; ++i) B[i] = A[i];

    const int maxIter = 50;
    const float epsilon = 1e-6f;

    for (int iter = 0; iter < maxIter; ++iter) {
        // Find max off-diagonal element
        int p = 0, q = 1;
        float maxVal = fabsf(B[1]);
        if (fabsf(B[2]) > maxVal) { p = 0; q = 2; maxVal = fabsf(B[2]); }
        if (fabsf(B[5]) > maxVal) { p = 1; q = 2; maxVal = fabsf(B[5]); }

        if (maxVal < epsilon) break;

        // Compute rotation angle
        float theta = 0.5f * atan2f(2 * B[p*3 + q], B[q*3 + q] - B[p*3 + p]);
        float c = cosf(theta);
        float s = sinf(theta);

        // Apply rotation to B
        float Bpp = B[p*3 + p];
        float Bqq = B[q*3 + q];
        float Bpq = B[p*3 + q];

        B[p*3 + p] = c*c*Bpp + s*s*Bqq - 2*c*s*Bpq;
        B[q*3 + q] = s*s*Bpp + c*c*Bqq + 2*c*s*Bpq;
        B[p*3 + q] = 0.0f;
        B[q*3 + p] = 0.0f;

        for (int r = 0; r < 3; ++r) {
            if (r != p && r != q) {
                float Brp = B[r*3 + p];
                float Brq = B[r*3 + q];
                B[r*3 + p] = c*Brp - s*Brq;
                B[p*3 + r] = B[r*3 + p];
                B[r*3 + q] = s*Brp + c*Brq;
                B[q*3 + r] = B[r*3 + q];
            }
        }

        // Update eigenvectors
        for (int r = 0; r < 3; ++r) {
            float Vrp = V[r*3 + p];
            float Vrq = V[r*3 + q];
            V[r*3 + p] = c*Vrp - s*Vrq;
            V[r*3 + q] = s*Vrp + c*Vrq;
        }
    }

    // Extract eigenvalues
    eigVals[0] = B[0];
    eigVals[1] = B[4];
    eigVals[2] = B[8];

    // Sort eigenvalues and eigenvectors
    if (eigVals[0] < eigVals[1]) {
        swap(&eigVals[0], &eigVals[1]);
        for (int r = 0; r < 3; ++r) swap(&V[r*3 + 0], &V[r*3 + 1]);
    }
    if (eigVals[0] < eigVals[2]) {
        swap(&eigVals[0], &eigVals[2]);
        for (int r = 0; r < 3; ++r) swap(&V[r*3 + 0], &V[r*3 + 2]);
    }
    if (eigVals[1] < eigVals[2]) {
        swap(&eigVals[1], &eigVals[2]);
        for (int r = 0; r < 3; ++r) swap(&V[r*3 + 1], &V[r*3 + 2]);
    }

    // Store eigenvectors
    for (int i = 0; i < 9; ++i) eigVecs[i] = V[i];
}

__global__ void eigenDecompositionKernel(const float* matrices, float* eigenvalues, float* eigenvectors, int numMatrices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numMatrices) return;

    float matrix[9];
    for (int i = 0; i < 9; ++i) matrix[i] = matrices[idx*9 + i];

    float eigVals[3];
    float eigVecs[9];
    jacobi3x3(matrix, eigVals, eigVecs);

    for (int i = 0; i < 3; ++i) eigenvalues[idx*3 + i] = eigVals[i];
    for (int i = 0; i < 9; ++i) eigenvectors[idx*9 + i] = eigVecs[i];
}



std::tuple<double, double> testOnce(){
    srand(42); // 固定随机种子便于验证

    // 分配主机内存
    float* h_matrices = (float*)malloc(NUM_MATRICES * 9 * sizeof(float));
    float* h_eigenvalues = (float*)malloc(NUM_MATRICES * 3 * sizeof(float));
    float* h_eigenvectors = (float*)malloc(NUM_MATRICES * 9 * sizeof(float));

    // 生成随机对称矩阵
    generateRandomSymmetricMatrices(h_matrices, NUM_MATRICES);


    // 创建 CUDA 事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 记录起始时间
    cudaEventRecord(start);

    auto start_time_cpu = std::chrono::high_resolution_clock::now();

    // 分配设备内存
    float *d_matrices, *d_eigenvalues, *d_eigenvectors;
    cudaMalloc(&d_matrices, NUM_MATRICES * 9 * sizeof(float));
    cudaMalloc(&d_eigenvalues, NUM_MATRICES * 3 * sizeof(float));
    cudaMalloc(&d_eigenvectors, NUM_MATRICES * 9 * sizeof(float));

    // 拷贝数据到设备
    cudaMemcpy(d_matrices, h_matrices, NUM_MATRICES * 9 * sizeof(float), cudaMemcpyHostToDevice);

    // 启动核函数
    int blockSize = BLOCK_SIZE;
    int gridSize = (NUM_MATRICES + blockSize - 1) / blockSize;
    eigenDecompositionKernel<<<gridSize, blockSize>>>(d_matrices, d_eigenvalues, d_eigenvectors, NUM_MATRICES);

    // 拷贝回结果
    cudaMemcpy(h_eigenvalues, d_eigenvalues, NUM_MATRICES * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_eigenvectors, d_eigenvectors, NUM_MATRICES * 9 * sizeof(float), cudaMemcpyDeviceToHost);

    auto end_time_cpu = std::chrono::high_resolution_clock::now();
    auto duration_cpu = std::chrono::duration_cast<std::chrono::microseconds>(end_time_cpu - start_time_cpu);
    if(IS_PRINT_RUNTIME){
        printf("CPU 计算耗时: %.3f 毫秒\n", duration_cpu.count()/1000.0f);

    }
    

    // 记录结束时间
    cudaEventRecord(stop);

    // 等待核函数完成
    cudaEventSynchronize(stop);

    // 计算时间差
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    if(IS_PRINT_RUNTIME){
        printf("CUDA 计算耗时: %.3f 毫秒\n", milliseconds);
    }


    // 验证结果
    const int numTests = NUM_MATRICES;
    bool is_print_matrix = false;
    float all_error = 0.0f;
    for (int t = 0; t < numTests; ++t) {
        int idx = rand() % NUM_MATRICES;

        // 原始矩阵
        float* m = h_matrices + idx*9;


        // 特征值和特征向量
        float* vals = h_eigenvalues + idx*3;
        float* vecs = h_eigenvectors + idx*9;


        // 重构矩阵
        float reconstructed[9] = {0};
        for (int k = 0; k < 3; ++k) {
            float lambda = vals[k];
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    reconstructed[i*3 + j] += lambda * vecs[i*3 + k] * vecs[j*3 + k];
                }
            }
        }

        // 计算差异
        float diff = 0.0f;
        const float epsilon = 1e15f;
        for (int i = 0; i < 9; ++i) 
            diff += fabsf(reconstructed[i] - m[i]);
        auto diff_fourdot = (int)(diff * epsilon)/epsilon;
        all_error += diff_fourdot;

        if(is_print_matrix){
            printf("\n验证矩阵 %d:\n", idx);

            // print 原始矩阵
            printf("原始矩阵:\n");
            for (int i = 0; i < 3; ++i) {
                printf("[%8.4f %8.4f %8.4f]\n", m[i*3], m[i*3+1], m[i*3+2]);
            }
            
            // print 特征值和特征向量
            printf("\n特征值: %8.4f %8.4f %8.4f\n", vals[0], vals[1], vals[2]);
            printf("特征向量:\n");
            for (int i = 0; i < 3; ++i) {
                printf("[%8.4f %8.4f %8.4f]\n", vecs[i*3], vecs[i*3+1], vecs[i*3+2]);
            }

            // print 重构矩阵
            printf("\n重构矩阵:\n");
            for (int i = 0; i < 3; ++i) {
                printf("[%8.4f %8.4f %8.4f]\n", 
                        reconstructed[i*3], reconstructed[i*3+1], reconstructed[i*3+2]);
            }

            // print 差异
            printf("index: %d 总差异: %.6f\n", t, diff);
        }

    }

    if(IS_PRINT_ERROR){
        printf("总差异: %.6f\n", all_error);
        printf("average diff: %.6f\n", all_error / numTests);
    }

    // 清理内存
    free(h_matrices);
    free(h_eigenvalues);
    free(h_eigenvectors);
    cudaFree(d_matrices);
    cudaFree(d_eigenvalues);
    cudaFree(d_eigenvectors);

    return {milliseconds, all_error / numTests};
}

// int main() {
//     int count = 10000;
    
//     for(int i = 0 ; i < 10000; i++){
//         auto runtime = testOnce();
//     }

//     return 0;
// }


int main() {
    int count = 1000;
    std::vector<double> runtimes;

    // 收集运行时间
    double percent = 0;
    double all_error = 0;
    for (int i = 0; i < count; i++) {
        auto [runtime, avg_error] = testOnce();
        runtimes.push_back(runtime);

        percent = (i + 1) * 100.0 / count;
        if (i % (int)(count/100) == 0) {
            std::cout << "进度: " << percent << "%\r" << std::flush;
        }
        all_error += avg_error;
    }


    // 排序
    std::sort(runtimes.begin(), runtimes.end());

    // 去掉最大值和最小值
    if (count > 2) {
        runtimes.erase(runtimes.begin());
        runtimes.pop_back();
    }

    // 计算统计量
    if (!runtimes.empty()) {
        double sum = 0.0;
        std::map<double, int> freq_map;

        // 计算总和和频率
        for (double rt : runtimes) {
            sum += rt;
            freq_map[rt]++;
        }

        double mean = sum / runtimes.size();

        // 计算方差
        double variance = 0.0;
        for (double rt : runtimes) {
            variance += std::pow(rt - mean, 2);
        }
        variance /= runtimes.size();

        // 找最大值和最小值
        double max_val = runtimes.back();
        double min_val = runtimes.front();

        // 找众数
        double mode = runtimes[0];
        int max_count = 1;
        for (const auto& pair : freq_map) {
            if (pair.second > max_count) {
                max_count = pair.second;
                mode = pair.first;
            }
        }

        // 输出结果
        std::cout << "样本数量（去掉最大最小值后）: " << runtimes.size() << std::endl;
        std::cout << "最大值: " << max_val << " 毫秒" << std::endl;
        std::cout << "最小值: " << min_val << " 毫秒" << std::endl;
        std::cout << "均值: " << mean << " 毫秒" << std::endl;
        std::cout << "众数(保留整数): " << static_cast<int>(mode) << " 毫秒" << std::endl;
        std::cout << "方差: " << variance << std::endl;
        std::cout << "标准差: " << std::sqrt(variance) << std::endl;

        std::cout << "平均差异: " << all_error / count << std::endl;
    } else {
        std::cout << "没有有效的运行时间数据。" << std::endl;
    }

    /*
    927000 x 3 x 3 dim
    样本数量（去掉最大最小值后）: 998
    最大值: 24.702 毫秒
    最小值: 13.5277 毫秒
    均值: 14.7503 毫秒
    众数(保留整数): 14 毫秒
    方差: 0.444104
    标准差: 0.666411
    平均差异: -2.14579e-06
    */

    return 0;
}



