# 设置 CMake 的最低版本要求
cmake_minimum_required(VERSION 3.10)

# 设置项目名称
project(fixedCapacityHeap)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 添加可执行文件
add_executable(${PROJECT_NAME} main.cpp)

# 如果需要链接 Boost 或其他库，可以在这里添加
# find_package(Boost REQUIRED)
# target_link_libraries(${PROJECT_NAME} Boost::boost)