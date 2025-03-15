# 设置最小 CMake 版本
cmake_minimum_required(VERSION 3.10)

# 设置项目名称和支持的语言
project(BTree CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 查找 Boost 库
find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)

# Include the directory where the header file is located
include_directories(${CMAKE_SOURCE_DIR}/RTree)


# 包含 Boost 的头文件路径
include_directories(${Boost_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})

# 添加可执行文件 and 链接 Boost 库
add_executable(rtree_single rtree_single.cpp)  # 假设你的源代码文件名为 main.cpp
target_link_libraries(rtree_single ${Boost_LIBRARIES} ${EIGEN3_LIBRARIES})


# 添加可执行文件 and 链接 Boost 库
add_executable(rtree_customnode rtree_customnode.cpp)  # 假设你的源代码文件名为 main.cpp
target_link_libraries(rtree_customnode ${Boost_LIBRARIES} ${EIGEN3_LIBRARIES})
