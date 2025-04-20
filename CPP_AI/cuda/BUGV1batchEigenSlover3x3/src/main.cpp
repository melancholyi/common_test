// #include "eig3x3.h"
// #include <iostream>
// #include <vector>
// #include <random>
// #include <chrono>
// #include <cuda_runtime.h>
// #include <Eigen/Dense>  // 新增Eigen头文件

// // 生成任意矩阵并对称化
// void generateMatrices(float* hostMatrices, int numMatrices) {
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::normal_distribution<float> dist(0.0f, 1.0f);
    
//     for(int i = 0; i < numMatrices; ++i) {
//         float* m = &hostMatrices[i * 9];
        
//         // 生成任意矩阵
//         for(int j = 0; j < 9; ++j) 
//             m[j] = dist(gen);
        
//         // 对称化处理：(M + M^T)/2
//         m[1] = (m[1] + m[3])/2;  // a12
//         m[2] = (m[2] + m[6])/2;  // a13
//         m[5] = (m[5] + m[7])/2;  // a23
        
//         // 保持对称性
//         m[3] = m[1];  // a21
//         m[6] = m[2];  // a31
//         m[7] = m[5];  // a32
//     }
// }

// // // 验证所有元素的重构误差
// // float verifyResult(
// //     const float* original,
// //     const float* eigenvalues,
// //     const float* eigenvectors,
// //     int numMatrices)
// // {
// //     float maxError = 0.0f;
// //     for(int i = 0; i < numMatrices; ++i) {
// //         const float* m = &original[i * 9];
// //         const float* vals = &eigenvalues[i * 3];
// //         const float* vecs = &eigenvectors[i * 9];
        
// //         // 重构对称矩阵
// //         float recon[9] = {0};
// //         for(int j = 0; j < 3; ++j) {
// //             for(int k = 0; k < 3; ++k) {
// //                 for(int l = 0; l < 3; ++l) {
// //                     recon[j*3 + k] += vecs[j*3 + l] * vals[l] * vecs[k*3 + l];
// //                 }
// //             }
// //         }
        
// //         // 计算全矩阵误差
// //         float error = 0.0f;
// //         for(int j = 0; j < 9; ++j) {
// //             error += fabs(recon[j] - m[j]);
// //         }
// //         maxError = fmaxf(maxError, error);
// //     }
// //     return maxError;
// // }

// float verifyResult(
//     const float* original,
//     const float* eigenvalues,
//     const float* eigenvectors,
//     int numMatrices)
// {
//     const int print_count = 10; // 控制打印数量
//     float maxError = 0.0f;
    
//     std::cout << "\n===== 前" << print_count << "个矩阵重构验证 =====" << std::endl;
//     std::cout.precision(4);
//     std::cout << std::scientific;

//     for(int i = 0; i < numMatrices; ++i) {
//         const float* m = &original[i * 9];
//         const float* vals = &eigenvalues[i * 3];
//         const float* vecs = &eigenvectors[i * 9];
        
//         // 重构矩阵
//         float recon[9] = {0};
//         for(int j = 0; j < 3; ++j) {
//             for(int k = 0; k < 3; ++k) {
//                 for(int l = 0; l < 3; ++l) {
//                     recon[j*3 + k] += vecs[j*3 + l] * vals[l] * vecs[k*3 + l];
//                 }
//             }
//         }

//         // 计算误差
//         float error = 0.0f;
//         for(int j = 0; j < 9; ++j) {
//             error += fabs(recon[j] - m[j]);
//         }
//         maxError = fmaxf(maxError, error);

//         // 打印前print_count个矩阵
//         if(i < print_count) {
//             std::cout << "\n矩阵 #" << i+1 << "/" << numMatrices 
//                      << " 总误差: " << error << std::endl;
            
//             // 打印原始矩阵
//             std::cout << "原始矩阵:\n";
//             for(int row=0; row<3; ++row){
//                 for(int col=0; col<3; ++col){
//                     std::cout << m[row*3 + col] << " ";
//                 }
//                 std::cout << "\n";
//             }

//             // 打印重构矩阵
//             std::cout << "重构矩阵:\n";
//             for(int row=0; row<3; ++row){
//                 for(int col=0; col<3; ++col){
//                     std::cout << recon[row*3 + col] << " ";
//                 }
//                 std::cout << "\n";
//             }

//             // 打印逐元素误差
//             std::cout << "元素绝对误差:\n";
//             for(int row=0; row<3; ++row){
//                 for(int col=0; col<3; ++col){
//                     const int idx = row*3 + col;
//                     std::cout << fabs(recon[idx] - m[idx]) << " ";
//                 }
//                 std::cout << "\n";
//             }
//             std::cout << "----------------------------\n";
//         }
//     }
//     return maxError;
// }



// // 新增：打印矩阵和特征分解结果
// void printComparison(const float* cpu_mat, const Eigen::Vector3f& cpu_vals, 
//     const Eigen::Matrix3f& cpu_vecs, const float* gpu_vals,
//     const float* gpu_vecs) {
//     std::cout.precision(4);
//     std::cout << std::scientific;

//     // 打印原始矩阵
//     std::cout << "\n原始矩阵:\n";
//     for(int i = 0; i < 3; ++i) {
//     for(int j = 0; j < 3; ++j)
//     std::cout << cpu_mat[i*3 + j] << " ";
//     std::cout << "\n";
//     }

//     // 打印CPU结果
//     std::cout << "\nEigen计算结果：";
//     std::cout << "\n特征值: " << cpu_vals.transpose();
//     std::cout << "\n特征向量:\n" << cpu_vecs << "\n";

//     // 打印GPU结果
//     std::cout << "GPU计算结果：";
//     std::cout << "\n特征值: [" << gpu_vals[0] << ", " 
//     << gpu_vals[1] << ", " << gpu_vals[2] << "]";
//     std::cout << "\n特征向量:\n";
//     for(int i = 0; i < 3; ++i) {
//     std::cout << "[" << gpu_vecs[i*3] << ", " 
//     << gpu_vecs[i*3+1] << ", "
//     << gpu_vecs[i*3+2] << "]\n";
//     }

//     // 计算误差
//     Eigen::Vector3f gpu_vals_vec(gpu_vals[0], gpu_vals[1], gpu_vals[2]);
//     Eigen::Matrix3f gpu_vecs_mat;
//     for(int i=0; i<3; ++i)
//     gpu_vecs_mat.col(i) = Eigen::Vector3f(gpu_vecs[i*3], gpu_vecs[i*3+1], gpu_vecs[i*3+2]);

//     float val_error = (cpu_vals - gpu_vals_vec).norm();
//     float vec_error = 0;
//     for(int i=0; i<3; ++i) {
//     float dot = fabs(cpu_vecs.col(i).dot(gpu_vecs_mat.col(i)));
//     vec_error += 1 - dot;  // 考虑方向可能相反
//     }

//     std::cout << "验证结果：\n"
//     << "特征值误差: " << val_error << "\n"
//     << "特征向量平均误差: " << vec_error/3 << "\n"
//     << "----------------------------------------\n";
// }

// int main() {
//     const int numMatrices = 972000;
//     const size_t matrixSizeBytes = 9 * sizeof(float);
    
//     // 主机内存分配
//     std::vector<float> hostMatrices(numMatrices * 9);
//     std::vector<float> hostEigenvalues(numMatrices * 3);
//     std::vector<float> hostEigenvectors(numMatrices * 9);
    
//     // 生成测试数据
//     generateMatrices(hostMatrices.data(), numMatrices);
    
//     // 设备内存分配
//     float *d_matrices, *d_eigenvalues, *d_eigenvectors;
//     cudaMalloc(&d_matrices, numMatrices * matrixSizeBytes);
//     cudaMalloc(&d_eigenvalues, numMatrices * 3 * sizeof(float));
//     cudaMalloc(&d_eigenvectors, numMatrices * 9 * sizeof(float));
    
//     // 拷贝数据到设备
//     cudaMemcpy(d_matrices, hostMatrices.data(), 
//               numMatrices * matrixSizeBytes, cudaMemcpyHostToDevice);
    
//     // 创建CUDA流并执行
//     cudaStream_t stream;
//     cudaStreamCreate(&stream);
    
//     auto start = std::chrono::high_resolution_clock::now();
//     batchedEigen3x3(d_matrices, d_eigenvalues, d_eigenvectors, numMatrices, stream);
//     cudaStreamSynchronize(stream);
//     auto end = std::chrono::high_resolution_clock::now();
    
//     // 获取结果
//     cudaMemcpy(hostEigenvalues.data(), d_eigenvalues,
//               numMatrices * 3 * sizeof(float), cudaMemcpyDeviceToHost);
//     cudaMemcpy(hostEigenvectors.data(), d_eigenvectors,
//               numMatrices * 9 * sizeof(float), cudaMemcpyDeviceToHost);
    
//     // 输出报告
//     std::cout << "===== 性能报告 =====\n"
//               << "矩阵数量: " << numMatrices << "\n"
//               << "存储方式: 完整9元素存储\n"
//               << "计算时间: " 
//               << std::chrono::duration<double>(end - start).count() * 1000
//               << " ms\n"
//               << "最大重构误差: " 
//               << verifyResult(hostMatrices.data(), hostEigenvalues.data(),
//                              hostEigenvectors.data(), 1000) << "\n";

//     // 新增：随机选择3个矩阵的索引
//     std::vector<int> sample_indices;
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_int_distribution<> dist(0, numMatrices-1);
    
//     for(int i=0; i<3; ++i) {
//         sample_indices.push_back(dist(gen));
//     }
//     // 新增：对比验证部分
//     std::cout << "\n===== 详细结果对比 (随机抽样3个矩阵) =====";
//     for(int idx : sample_indices) {
//         // 获取GPU计算结果
//         const float* gpu_vals = &hostEigenvalues[idx*3];
//         const float* gpu_vecs = &hostEigenvectors[idx*9];
        
//         // 获取原始矩阵
//         Eigen::Matrix3f mat;
//         const float* cpu_mat = &hostMatrices[idx*9];
//         for(int i=0; i<9; ++i) mat(i/3, i%3) = cpu_mat[i];

//         // 使用Eigen计算
//         Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(mat);
//         Eigen::Vector3f eigen_vals = solver.eigenvalues();
//         Eigen::Matrix3f eigen_vecs = solver.eigenvectors();

//         // 调整特征值顺序（按降序排列）
//         eigen_vals.reverse();
//         eigen_vecs = eigen_vecs.rowwise().reverse();

//         // 打印对比
//         printComparison(cpu_mat, eigen_vals, eigen_vecs, gpu_vals, gpu_vecs);
//     }






//     // 清理资源
//     cudaFree(d_matrices);
//     cudaFree(d_eigenvalues);
//     cudaFree(d_eigenvectors);
//     cudaStreamDestroy(stream);
    
//     return 0;
// }


#include "eig3x3.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>

// 生成随机对称矩阵
void generateSymmetricMatrices(float* hostMatrices, int numMatrices) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    // 只生成上三角元素（6个元素）
    for(int i = 0; i < numMatrices; ++i) {
        float* m = &hostMatrices[i * 6];
        m[0] = dist(gen); // a11
        m[1] = dist(gen); // a12
        m[2] = dist(gen); // a13
        m[3] = dist(gen); // a22
        m[4] = dist(gen); // a23
        m[5] = dist(gen); // a33
    }
}

// // 验证结果正确性
// float verifyResult(
//     const float* original,
//     const float* eigenvalues,
//     const float* eigenvectors,
//     int numMatrices)
// {
//     float maxError = 0.0f;
//     for(int i = 0; i < numMatrices; ++i) {
//         const float* m = &original[i * 6];
//         const float* vals = &eigenvalues[i * 3];
//         const float* vecs = &eigenvectors[i * 9];
        
//         // 重构矩阵: V * diag(λ) * V^T
//         float recon[9] = {0};
//         for(int j = 0; j < 3; ++j) {
//             for(int k = 0; k < 3; ++k) {
//                 for(int l = 0; l < 3; ++l) {
//                     recon[j*3 + k] += vecs[j*3 + l] * vals[l] * vecs[k*3 + l];
//                 }
//             }
//         }
        
//         // 计算与原矩阵的误差
//         float error = 0.0f;
//         error += fabs(recon[0] - m[0]); // a11
//         error += fabs(recon[1] - m[1]); // a12
//         error += fabs(recon[2] - m[2]); // a13
//         error += fabs(recon[4] - m[3]); // a22
//         error += fabs(recon[5] - m[4]); // a23
//         error += fabs(recon[8] - m[5]); // a33
//         maxError = std::max(maxError, error);
//     }
//     return maxError;
// }


float verifyResult(
    const float* original,
    const float* eigenvalues,
    const float* eigenvectors,
    int numMatrices)
{
    const int print_count = 10; // 控制打印数量
    float maxError = 0.0f;
    
    std::cout << "\n===== 前" << print_count << "个矩阵重构验证 =====" << std::endl;
    std::cout.precision(4);
    std::cout << std::scientific;

    for(int i = 0; i < numMatrices; ++i) {
        const float* m = &original[i * 9];
        const float* vals = &eigenvalues[i * 3];
        const float* vecs = &eigenvectors[i * 9];
        
        // 重构矩阵
        float recon[9] = {0};
        for(int j = 0; j < 3; ++j) {
            for(int k = 0; k < 3; ++k) {
                for(int l = 0; l < 3; ++l) {
                    recon[j*3 + k] += vecs[j*3 + l] * vals[l] * vecs[k*3 + l];
                }
            }
        }

        // 计算误差
        float error = 0.0f;
        for(int j = 0; j < 9; ++j) {
            error += fabs(recon[j] - m[j]);
        }
        maxError = fmaxf(maxError, error);

        // 打印前print_count个矩阵
        if(i < print_count) {
            std::cout << "\n矩阵 #" << i+1 << "/" << numMatrices 
                     << " 总误差: " << error << std::endl;
            
            // 打印原始矩阵
            std::cout << "原始矩阵:\n";
            for(int row=0; row<3; ++row){
                for(int col=0; col<3; ++col){
                    std::cout << m[row*3 + col] << " ";
                }
                std::cout << "\n";
            }

            // 打印重构矩阵
            std::cout << "重构矩阵:\n";
            for(int row=0; row<3; ++row){
                for(int col=0; col<3; ++col){
                    std::cout << recon[row*3 + col] << " ";
                }
                std::cout << "\n";
            }

            // 打印逐元素误差
            std::cout << "元素绝对误差:\n";
            for(int row=0; row<3; ++row){
                for(int col=0; col<3; ++col){
                    const int idx = row*3 + col;
                    std::cout << fabs(recon[idx] - m[idx]) << " ";
                }
                std::cout << "\n";
            }
            std::cout << "----------------------------\n";
        }
    }
    return maxError;
}

int main() {
    const int numMatrices = 972000;
    const size_t matrixSizeBytes = 6 * sizeof(float); // 上三角存储
    
    // 主机内存分配
    std::vector<float> hostMatrices(numMatrices * 6);
    std::vector<float> hostEigenvalues(numMatrices * 3);
    std::vector<float> hostEigenvectors(numMatrices * 9);
    
    // 生成测试数据
    generateSymmetricMatrices(hostMatrices.data(), numMatrices);
    
    // 设备内存分配
    float *d_matrices, *d_eigenvalues, *d_eigenvectors;
    cudaMalloc(&d_matrices, numMatrices * matrixSizeBytes);
    cudaMalloc(&d_eigenvalues, numMatrices * 3 * sizeof(float));
    cudaMalloc(&d_eigenvectors, numMatrices * 9 * sizeof(float));
    
    // 拷贝数据到设备
    cudaMemcpy(d_matrices, hostMatrices.data(), 
              numMatrices * matrixSizeBytes, cudaMemcpyHostToDevice);
    
    // 创建CUDA流
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // 执行计算并计时
    auto start = std::chrono::high_resolution_clock::now();
    
    batchedEigen3x3(d_matrices, d_eigenvalues, d_eigenvectors, 
                   numMatrices, stream);
    
    cudaStreamSynchronize(stream);
    auto end = std::chrono::high_resolution_clock::now();
    
    // 拷贝结果回主机
    cudaMemcpy(hostEigenvalues.data(), d_eigenvalues,
              numMatrices * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostEigenvectors.data(), d_eigenvectors,
              numMatrices * 9 * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 验证结果
    float maxError = verifyResult(
        hostMatrices.data(),
        hostEigenvalues.data(),
        hostEigenvectors.data(),
        std::min(numMatrices, 1000) // 抽样验证前1000个
    );
    
    // 输出结果
    std::cout << "===== 性能报告 =====\n"
              << "矩阵数量: " << numMatrices << "\n"
              << "计算时间: " 
              << std::chrono::duration<double>(end - start).count() * 1000
              << " ms\n"
              << "最大重构误差: " << maxError << "\n";
    
    // 清理资源
    cudaFree(d_matrices);
    cudaFree(d_eigenvalues);
    cudaFree(d_eigenvectors);
    cudaStreamDestroy(stream);
    
    return 0;
}