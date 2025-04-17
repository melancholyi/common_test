cmake_minimum_required(VERSION 3.18)
project(constructSe2Travmap LANGUAGES CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 找到 Eigen3
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# 找到 PyTorch
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})

# 添加可执行文件
add_executable(constructSe2Travmap constructSe2Travmap.cpp)

# 链接库
target_link_libraries(constructSe2Travmap ${TORCH_LIBRARIES} Eigen3::Eigen)

# 如果需要 CUDA 支持，确保链接 CUDA 工具包
if(TORCH_CUDA_AVAILABLE)
    find_package(CUDA REQUIRED)
    target_link_libraries(constructSe2Travmap ${CUDA_LIBRARIES})
endif()

# 处理 Torch 的动态链接问题
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${TORCH_LINK_FLAGS}")