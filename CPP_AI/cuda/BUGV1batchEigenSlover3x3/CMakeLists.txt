cmake_minimum_required(VERSION 3.12)
project(eigen3x3)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找CUDA
find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)

# 包含目录
include_directories(include)

# 编译CUDA部分
cuda_add_library(eigen3x3_cuda STATIC
    src/eig3x3.cu
)

# 主程序
add_executable(eigen3x3_test
    src/main.cpp
)

# 链接库
target_link_libraries(eigen3x3_test
    eigen3x3_cuda
    ${CUDA_LIBRARIES}
    Eigen3::Eigen
)

# 优化选项
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(eigen3x3_cuda PRIVATE -O3 -Xptxas -O3)
    target_compile_options(eigen3x3_test PRIVATE -O3)
endif()