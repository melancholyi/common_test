cmake_minimum_required(VERSION 3.10)

# 项目名称
project(eigenRelatedGPUTest)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 查找 package
find_package(Open3D REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 REQUIRED)

# 包含 Open3D 和 Eigen 的头文件目录
include_directories(${Open3D_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})

# 设置源文件和目标文件
add_executable(eigenRelatedGPUTest eigenRelatedGPUTest.cpp)

# 链接 Open3D 和 Eigen 库
target_link_libraries(eigenRelatedGPUTest PRIVATE Eigen3::Eigen)
target_link_libraries(eigenRelatedGPUTest PRIVATE Open3D::Open3D)
target_link_libraries(eigenRelatedGPUTest PRIVATE CUDA::cudart)
