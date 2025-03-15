cmake_minimum_required(VERSION 3.10)
project(RandomPointCloudExample)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 查找 Open3D 库
find_package(Open3D REQUIRED)
find_package(CUDAToolkit REQUIRED)

# 添加可执行文件
add_executable(open3dgpu open3dgpu.cpp)


# 链接 Open3D 库
target_link_libraries(open3dgpu PRIVATE Open3D::Open3D)
target_link_libraries(open3dgpu PRIVATE CUDA::cudart)