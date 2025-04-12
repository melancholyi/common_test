cmake_minimum_required(VERSION 3.10)
project(gridUpdate)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加源文件
add_executable(${PROJECT_NAME} gridUpdate.cpp)

# 如果需要链接其他库（如 OpenCV 或 PCL），可以在这里添加
# target_link_libraries(${PROJECT_NAME} PRIVATE some_library)