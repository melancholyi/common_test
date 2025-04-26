/*
 * @Author: chasey && melancholycy@gmail.com
 * @Date: 2025-04-19 03:47:10
 * @LastEditTime: 2025-04-21 10:30:49
 * @FilePath: /test/CPP_THIRD_PARTYs/eigen/commonUsage.cpp
 * @Description: 
 * @Reference: 
 * Copyright (c) 2025 by chasey && melancholycy@gmail.com, All Rights Reserved. 
 */
#include <iostream>
#include <Eigen/Dense>
#include <chrono>
#include <vector>
#include <iostream>
#include <random>

//==========PART:1 Eigen Decomposition==========
void eigenDecomposition() {
    // Create a 3x3 covariance matrix (symmetric matrix)
    Eigen::Matrix3d covMatrix;
    // covMatrix << 1.0, 2.0, 3.0,
    //              4.0, 5.0, 6.0,
    //              7.0, 8.0, 10.0;
    // covMatrix << 5.0, 0.0, 0.0,
    //             0.0, 10.0, 0.0,
    //             0.0, 0.0, 1.0;


    covMatrix << -0.6619, 0.6361, -0.8202,
                 0.6361, 0.1741, 0.3069,
                 -0.8202, 0.3069, -0.2102;
    
    // Perform eigen decomposition
    Eigen::EigenSolver<Eigen::Matrix3d> eigenSolver(covMatrix);
    
    // Get eigenvalues and eigenvectors
    Eigen::Vector3d eigenvalues = eigenSolver.eigenvalues().real();
    Eigen::Matrix3d eigenvectors = eigenSolver.eigenvectors().real();
    
    // Output results
    std::cout << "covMatrix:\n" << covMatrix << "\n\n";
    std::cout << "Eigenvalues:\n" << eigenvalues << "\n\n";
    std::cout << "Eigenvectors:\n" << eigenvectors << std::endl;
    /*
    ==========PART: 1 Eigen Decomposition==========
    covMatrix:
    1 2 3
    4 5 6
    7 8 9

    Eigenvalues:
    16.1168
    -1.11684
    1.6795e-15

    Eigenvectors:
    0.231971   0.78583  0.408248
    0.525322 0.0867513 -0.816497
    0.818673 -0.612328  0.408248
    */
}

//====================PART:2 Batch Eigen Slover====================
void batchEigenDecomposition(){
    using namespace Eigen;
    using namespace std;

    // 定义 3x3 的协方差矩阵 A
    Matrix3d A;
    A << 1, 2, 3,
        4, 5, 6,
        7, 8, 9;

    // 使用 Eigen 进行矩阵 A 的特征分解
    cout << "矩阵 A:" << endl << A << endl;

    // 特征分解
    Eigen::SelfAdjointEigenSolver<Matrix3d> eigensolverA(A);
    if (eigensolverA.info() != Eigen::Success) {
        cerr << "特征分解失败!" << endl;
    }

    cout << "矩阵 A 的特征值:" << endl << eigensolverA.eigenvalues() << endl;
    cout << "矩阵 A 的特征向量:" << endl << eigensolverA.eigenvectors() << endl;

    // 构造 12x12 的对角矩阵 diag(A, A, A, A)
    MatrixXd diag_A_A_A_A(12, 12);
    diag_A_A_A_A.setZero();

    // 将 4 个 A 放置在对角线上
    diag_A_A_A_A.block<3, 3>(0, 0) = A;
    diag_A_A_A_A.block<3, 3>(3, 3) = A;
    diag_A_A_A_A.block<3, 3>(6, 6) = A;
    diag_A_A_A_A.block<3, 3>(9, 9) = A;

    cout << "12x12 对角矩阵 diag(A,A,A,A):" << endl << diag_A_A_A_A << endl;

    // 对 12x12 对角矩阵进行特征分解
    Eigen::EigenSolver<MatrixXd> eigensolver12x12(diag_A_A_A_A);
    if (eigensolver12x12.info() != Eigen::Success) {
        cerr << "特征分解失败!" << endl;
    }

    cout << "12x12 对角矩阵的特征值:" << endl << eigensolver12x12.eigenvalues() << endl;
    cout << "12x12 对角矩阵的特征向量:" << endl << eigensolver12x12.eigenvectors() << endl;
}

//====================PART: 3 large amount 3x3Matrix's comparision
Eigen::Matrix3d generateRandomCovarianceMatrix(int size=3) {
    // 随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);

    // 生成随机矩阵
    Eigen::Matrix3d mat;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            mat(i, j) = dist(gen);
        }
    }

    // 使其成为对称矩阵
    Eigen::MatrixXd symmetricMat = (mat + mat.transpose()) / 2.0;

    // 添加对角元素使矩阵为正定（协方差矩阵通常为正定）
    symmetricMat.diagonal().array() += 1.0;

    return symmetricMat;
}
void largeMatrix3X3EigenDecompistion(){
    const int numMatrices = 91*76*76;
    std::cout << "numMatrices: " << numMatrices << std::endl;
    const int matrixSize = 3;

    // 生成随机协方差矩阵
    std::vector<Eigen::Matrix3d> covarianceMatrices;
    for (int i = 0; i < numMatrices; ++i) {
        covarianceMatrices.push_back(generateRandomCovarianceMatrix(matrixSize));
    }

    // 使用 Eigen::SelfAdjointEigenSolver 的性能测试
    {
        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver;
        for (const auto& mat : covarianceMatrices) {
            eigensolver.compute(mat);
            auto evals = eigensolver.eigenvalues();
            auto evecs = eigensolver.eigenvectors();
            // 可以访问 eigensolver.eigenvalues() 和 eigensolver.eigenvectors()
        }

        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        std::cout << "Eigen::SelfAdjointEigenSolver 运行时间： " << elapsed.count()*1000 << " ms" << std::endl;
    }

    // 使用 Eigen::EigenSolver 的性能测试
    {
        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver;
        for (const auto& mat : covarianceMatrices) {
            // Eigen::EigenSolver<Eigen::MatrixXd> eigensolver(mat);
            eigensolver.compute(mat);
            auto evals = eigensolver.eigenvalues();
            auto evecs = eigensolver.eigenvectors();
            // 可以访问 eigensolver.eigenvalues() 和 eigensolver.eigenvectors()
        }

        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        std::cout << "Eigen::EigenSolver 运行时间： " << elapsed.count()*1000 << " ms" << std::endl;
    }
}

int main() {
    //==========PART:1 Eigen Decomposition==========
    std::cout << "==========PART: 1 Eigen Decomposition==========" << std::endl;
    eigenDecomposition();

    // //====================PART:2 Eigen SVD====================
    // std::cout << "==========PART: 2 batch Eigen Decomposition==========" << std::endl;
    // batchEigenDecomposition();

    // //====================PART: 3 largeMatrix3X3EigenDecompistion====================
    // largeMatrix3X3EigenDecompistion();

    
    return 0;
}