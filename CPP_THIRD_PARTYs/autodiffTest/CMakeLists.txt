# 指定 CMake 的最小版本
cmake_minimum_required(VERSION 3.16)

# 设置项目名称和版本
project(MyProject VERSION 1.0)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3")

# 添加源文件
add_executable(autodiffTest src/main.cpp)

# 包含 autodiff 的头文件目录
target_include_directories(autodiffTest PRIVATE third_party/autodiff)

# 如果 autodiff 是通过系统安装的，可以使用 find_package
# find_package(autodiff REQUIRED)
# target_link_libraries(MyExecutable autodiff::autodiff)