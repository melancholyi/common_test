cmake_minimum_required(VERSION 3.10)
project(ikdtreeTest)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-std=c++14 -pthread -O3") 

# 查找 Boost 库
find_package(Boost REQUIRED)
find_package(PCL REQUIRED)

# 包含 Boost 头文件
include_directories(${Boost_INCLUDE_DIRS} ikd-Tree ikdtree ${PCL_INCLUDE_DIRS})

# 添加可执行文件
add_executable(${PROJECT_NAME} main.cpp ikd-Tree/ikd_Tree.cpp ikdtree/ikd_tree.cpp)
add_executable(ikdtree_kdtree ikdtree_kdtree.cpp ikd-Tree/ikd_Tree.cpp ikdtree/ikd_tree.cpp)

# 链接 Boost 库
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES}    ${PCL_LIBRARIES})
target_link_libraries(ikdtree_kdtree ${Boost_LIBRARIES}    ${PCL_LIBRARIES})