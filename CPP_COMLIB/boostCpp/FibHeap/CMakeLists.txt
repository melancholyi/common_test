# 设置最小 CMake 版本
cmake_minimum_required(VERSION 3.10)

# 设置项目名称和支持的语言
project(FibonacciHeapExample CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 查找 Boost 库
find_package(Boost REQUIRED)

# 包含 Boost 的头文件路径
include_directories(${Boost_INCLUDE_DIRS})

# 添加可执行文件
add_executable(FibonacciHeapExample main.cpp)  # 假设你的源代码文件名为 main.cpp

# 添加可执行文件
add_executable(compareBoostStdHeap compareBoostStdHeap.cpp)  # 假设你的源代码文件名为 main.cpp


# 链接 Boost 库
target_link_libraries(FibonacciHeapExample ${Boost_LIBRARIES})
target_link_libraries(compareBoostStdHeap ${Boost_LIBRARIES})
