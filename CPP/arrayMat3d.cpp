/*
 * @Author: chasey melancholycy@gmail.com
 * @Date: 2025-02-12 12:45:42
 * @FilePath: /mesh_planner/test/cpp/arrayMat3d.cpp
 * @Description: 
 * 
 * Copyright (c) 2025 by chasey (melancholycy@gmail.com), All Rights Reserved. 
 */#include <iostream>
#include <array>
#include <Eigen/Core>

// 将 Eigen::Matrix3d 转换为 std::array<double, 9>
inline std::array<double, 9> covMat2array(const Eigen::Matrix3d& covMat) {
    std::array<double, 9> arr;
    Eigen::Map<Eigen::Matrix3d>(arr.data()) = covMat;
    return arr;
}

// 将 std::array<double, 9> 转换为 Eigen::Matrix3d
inline Eigen::Matrix3d array2covMat(const std::array<double, 9>& array) {
    Eigen::Map<const Eigen::Matrix3d> mat(array.data());
    return mat;
}

int main() {
    // 创建一个 Matrix3d 对象
    Eigen::Matrix3d eigMat;
    eigMat << 1, 2, 3,
              4, 5, 6,
              7, 8, 9;

    std::cout << "eigMat:\n" << eigMat << std::endl;

    // 使用 covMat2array 函数将 Matrix3d 转换为 std::array
    std::array<double, 9> arr = covMat2array(eigMat);

    // 打印 std::array 的内容
    std::cout << "std::array from Matrix3d:" << std::endl;
    for (const auto& val : arr) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // 使用 array2covMat 函数将 std::array 转换回 Matrix3d
    Eigen::Matrix3d newMat = array2covMat(arr);

    // 打印新的 Matrix3d
    std::cout << "New Matrix3d from std::array:\n" << newMat << std::endl;

    return 0;
}