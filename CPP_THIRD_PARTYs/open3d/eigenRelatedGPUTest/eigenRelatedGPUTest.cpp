/*
 * @Author: chasey && melancholycy@gmail.com
 * @Date: 2025-03-23 13:21:48
 * @LastEditTime: 2025-03-24 03:39:11
 * @FilePath: /test/CPP_THIRD_PARTYs/open3d/eigenRelatedGPUTest/eigenRelatedGPUTest.cpp
 * @Description: 
 * @Reference: 
 * Copyright (c) 2025 by chasey && melancholycy@gmail.com, All Rights Reserved. 
 */
#include <Eigen/Dense>
#include <Eigen/Core>
#include <open3d/Open3D.h>
#include <iostream>

int main() {
    // 创建一个随机的矩阵
    std::vector<float> matrix_data = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f};

    open3d::core::Tensor matrix = open3d::core::Tensor(matrix_data, {3, 3}, open3d::core::Float32);

    // 执行SVD分解
    auto svd_result = matrix.SVD();

    // 获取SVD分解的结果
    // 使用std::get提取三个矩阵
    open3d::core::Tensor U = std::get<0>(svd_result);
    open3d::core::Tensor Sigma = std::get<1>(svd_result);
    open3d::core::Tensor Vt = std::get<2>(svd_result);

    // 打印结果
    std::cout << "Original Matrix:\n" << matrix.ToString() << std::endl;
    std::cout << "U:\n" << U.ToString() << std::endl;
    std::cout << "Sigma:\n" << Sigma.ToString() << std::endl;
    std::cout << "Vt:\n" << Vt.ToString() << std::endl;


    /////////////////////////////////////////////////////////////////////////////////////////////////////
    // 将 std::vector 转换为 Eigen::Matrix3d
    Eigen::Matrix3d matrix_eig;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            matrix_eig(i, j) = matrix_data[i * 3 + j];
        }
    }

    // 输出矩阵
    std::cout << "matrix_eig:\n" << matrix_eig << std::endl;

    // 使用 Eigen 的 EigenSolver 计算特征值和特征向量
    Eigen::EigenSolver<Eigen::Matrix3d> solver(matrix_eig);

    solver.compute(matrix_eig);
    // 获取特征值
    auto eigenvalues = solver.eigenvalues();
    std::cout << "Eigenvalues:\n" << eigenvalues << std::endl;

    // 获取特征向量
    auto eigenvectors = solver.eigenvectors();
    std::cout << "Eigenvectors:\n" << eigenvectors << std::endl;


    return 0;
}