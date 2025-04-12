cmake_minimum_required(VERSION 3.10)
project(errorUnorderedMap)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3")

# 添加可执行文件
add_executable(${PROJECT_NAME} ${PROJECT_NAME}.cpp)

# 查找OpenMP库
find_package(OpenMP REQUIRED)

# 链接OpenMP库
target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)


