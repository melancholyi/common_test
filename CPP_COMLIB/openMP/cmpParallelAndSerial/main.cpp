/*
 * @Author: chasey melancholycy@gmail.com
 * @Date: 2025-03-11 11:14:35
 * @FilePath: /mesh_planner/test/cpp/openMP/main.cpp
 * @Description: 
 * 
 * Copyright (c) 2025 by chasey (melancholycy@gmail.com), All Rights Reserved. 
 */
#include <iostream>
#include <vector>
#include <chrono>
// #include <omp.h>

// 非并行版本
void nonParallelVersion(int rows, int cols, std::vector<std::vector<int>>& matrix) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = i * j; // 示例操作
        }
    }
}

// 并行版本（使用OpenMP）
void parallelVersion(int rows, int cols, std::vector<std::vector<int>>& matrix) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i].push_back(i * j);
        }
    }

#pragma omp parallel for collapse(2)
for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
        matrix[i].push_back(i + j);
    }
}
}

int main() {
    const int rows = 10000;
    const int cols = 10000;

    // 初始化矩阵
    std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols, 0));

    // 测试非并行版本
    auto start = std::chrono::high_resolution_clock::now();
    nonParallelVersion(rows, cols, matrix);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> nonParallelTime = end - start;
    std::cout << "Non-parallel version took " << nonParallelTime.count() * 1000 << " ms." << std::endl;

    // 测试并行版本
    start = std::chrono::high_resolution_clock::now();
    parallelVersion(rows, cols, matrix);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> parallelTime = end - start;
    std::cout << "Parallel version took " << parallelTime.count() * 1000 << " ms1." << std::endl;


    std::vector<int> vec;
    vec.reserve(10);
    vec.push_back(1);
    std::cout << vec.size() << std::endl;;

    return 0;
}