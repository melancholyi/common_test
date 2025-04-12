cmake_minimum_required(VERSION 3.10)
project(openMPHelloworld)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3")

# 添加可执行文件
add_executable(singleLoop singleLoop.cpp)
add_executable(doubleLoop doubleLoop.cpp)
add_executable(doubleLoopOther doubleLoopOther.cpp)

# 查找OpenMP库
find_package(OpenMP REQUIRED)

# 链接OpenMP库
target_link_libraries(singleLoop PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(doubleLoop PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(doubleLoopOther PRIVATE OpenMP::OpenMP_CXX)


