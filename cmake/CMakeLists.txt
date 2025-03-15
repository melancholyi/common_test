cmake_minimum_required(VERSION 3.18)  # 确保CMake版本支持CUDA相关功能
project(CheckCUDAVersion)

# 启用CUDA语言支持
enable_language(CUDA)

# 查找CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# 获取CUDA Toolkit版本
message(STATUS "CUDA Toolkit Version: ${CUDAToolkit_VERSION}")

# 如果需要，可以进一步处理版本信息
if(CUDAToolkit_VERSION VERSION_LESS "11.0")
    message(WARNING "Your CUDA Toolkit version is less than 11.0")
else()
    message(STATUS "Your CUDA Toolkit version is ${CUDAToolkit_VERSION}, which is supported.")
endif()