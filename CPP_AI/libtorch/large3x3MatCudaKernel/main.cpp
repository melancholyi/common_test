#include <torch/torch.h>
#include <iostream>
#include "matrix3x3eigendecomp.h"

// 定义函数来重建矩阵并比较差异
void reconstruct_matrices_and_compare(
    const torch::Tensor &original_matrices,
    const torch::Tensor &eigenvalues,
    const torch::Tensor &eigenvectors,
    torch::Tensor &differences
) {
    // 确保所有张量都在CPU上
    torch::Tensor original = original_matrices.to(torch::kCPU);
    torch::Tensor values = eigenvalues.to(torch::kCPU);
    torch::Tensor vectors = eigenvectors.to(torch::kCPU);

    // 重建矩阵
    torch::Tensor reconstructed = torch::zeros_like(original);

    for (int i = 0; i < values.size(0); i++) {
        auto v = vectors[i];
        auto d = torch::diag(values[i]);
        reconstructed[i] = v.matmul(d).matmul(v.transpose(0, 1));
    }

    // 计算差异
    differences = (original - reconstructed).abs().sum({1, 2});
}


// 定义函数来对比两种方法得到的特征值和特征向量
void compare_eigen_decomposition(
    const torch::Tensor &original_matrices,
    const torch::Tensor &original_eigenvalues,
    const torch::Tensor &original_eigenvectors
) {
    const int numSamples = 10; // 随机抽取的矩阵数量
    auto indices = torch::randperm(original_matrices.size(0)).slice(0, 0, numSamples);

    torch::Tensor sampled_matrices = original_matrices.index_select(0, indices);
    torch::Tensor sampled_eigenvalues = original_eigenvalues.index_select(0, indices);
    torch::Tensor sampled_eigenvectors = original_eigenvectors.index_select(0, indices);

    // 使用 torch::eigh 计算特征分解
    auto [eigh_eigenvalues, eigh_eigenvectors] = torch::linalg::eigh(sampled_matrices, "L");
    eigh_eigenvalues = eigh_eigenvalues.index({torch::indexing::Slice(), torch::tensor({2, 1, 0})});
    eigh_eigenvectors = eigh_eigenvectors.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::tensor({2, 1, 0})});

    // 对比特征值
    std::cout << "========== Compare Eigenvalues: ==========" << std::endl;
    for (int i = 0; i < numSamples; i++) {
        std::cout << "==========Sample " << i + 1 << " Eigenvalues:" << std::endl;
        std::cout << "Original: \n" << sampled_eigenvalues[i] << std::endl;
        std::cout << "torch::eigh: \n" << eigh_eigenvalues[i] << std::endl;
        std::cout << "Difference: \n" << (sampled_eigenvalues[i] - eigh_eigenvalues[i]).abs() << std::endl << std::endl;
    }

    // 对比特征向量
    std::cout << "========== Compare Eigenvectors: ==========" << std::endl;
    for (int i = 0; i < numSamples; i++) {
        std::cout << "==========Sample " << i + 1 << " Eigenvectors:" << std::endl;
        std::cout << "Original: \n" << sampled_eigenvectors[i] << std::endl;
        std::cout << "torch::eigh: \n" << eigh_eigenvectors[i] << std::endl;
        std::cout << "Difference: \n" << (sampled_eigenvectors[i] - eigh_eigenvectors[i]).abs() << std::endl << std::endl;
    }
}

int main() {
    // 定义张量大小
    const int numMatrices = 92700;
    const int matrixSize = 3;

    // 随机生成对称矩阵
    auto options = torch::TensorOptions().dtype(torch::kF32).device(torch::kCPU);
    torch::Tensor matrices = torch::randn({numMatrices, matrixSize, matrixSize}, options);
    matrices = 0.5 * (matrices + matrices.transpose(1, 2)); // 使其对称



    //------------------------------------------
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    auto time_start_cpu = std::chrono::high_resolution_clock::now();
    cudaEventRecord(start);

    // 确保张量是连续的
    matrices = matrices.contiguous();
    // 创建设备张量
    torch::Tensor d_matrices = matrices.to(torch::kCUDA);
    torch::Tensor d_eigenvalues = torch::empty({numMatrices, 3}, options.dtype(torch::kF32).device(torch::kCUDA));
    torch::Tensor d_eigenvectors = torch::empty({numMatrices, 3, 3}, options.dtype(torch::kF32).device(torch::kCUDA));
    // 执行特征分解
    eigenDecompositionLauncher(d_matrices, d_eigenvalues, d_eigenvectors);

    /*
    // //NOTE: raw_ptr version
    // // 获取原始指针
    // const float* d_matrices_ptr = d_matrices.data_ptr<float>();
    // float* d_eigenvalues_ptr = d_eigenvalues.data_ptr<float>();
    // float* d_eigenvectors_ptr = d_eigenvectors.data_ptr<float>();

    // // 调用 CUDA 核函数
    // eigenDecompositionLauncher(d_matrices_ptr, d_eigenvalues_ptr, d_eigenvectors_ptr, numMatrices);

    // // 检查 CUDA 错误
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    //     return -1;
    // }
    */
    // 将结果传输回主机
    torch::Tensor eigenvalues = d_eigenvalues.to(torch::kCPU);
    torch::Tensor eigenvectors = d_eigenvectors.to(torch::kCPU);



    // 记录结束时间
    cudaEventRecord(stop);

    // 等待核函数完成
    cudaEventSynchronize(stop);
    // 计算时间差
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    auto time_end_cpu = std::chrono::high_resolution_clock::now();
    auto duration_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_cpu - time_start_cpu);


    // // 打印部分结果
    // std::cout << "First few eigenvalues:" << std::endl;
    // std::cout << eigenvalues.slice(0, 0, 5) << std::endl;

    // std::cout << "First few eigenvectors:" << std::endl;
    // std::cout << eigenvectors.slice(0, 0, 5) << std::endl;

    // 重建矩阵并计算差异
    torch::Tensor differences = torch::empty({numMatrices}, options.dtype(torch::kF32));
    reconstruct_matrices_and_compare(matrices, eigenvalues, eigenvectors, differences);

    // 累积打印差异
    double total_diff = differences.sum().item<float>();


    compare_eigen_decomposition(matrices, eigenvalues, eigenvectors);


    std::cout << "Total difference between original and reconstructed matrices: " << total_diff << std::endl;
    std::cout << "CUDA kernel execution time: " << milliseconds << " ms" << std::endl;
    std::cout << "CPU execution time: " << duration_cpu.count() << " ms" << std::endl;

    return 0;
}