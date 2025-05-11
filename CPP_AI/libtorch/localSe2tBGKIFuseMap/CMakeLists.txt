# cmake_minimum_required(VERSION 3.21)  # 设置CMake的最低版本
# set(CMAKE_CXX_STANDARD 17)  # 设置C++标准
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# project(LocalTensorBufferProject)  # 定义项目名称

# find_package(Eigen3 REQUIRED)
# find_package(Torch REQUIRED)



# # 添加包含目录
# include_directories(include)  # 添加项目的include目录

# # 定义源文件和头文件
# set(SOURCES
#   main.cpp  # 主函数文件
#   src/local_se2t_bgkimap.cpp  # LocalTensorBuffer类的实现文件
# )

# set(HEADERS
#   include/local_se2t_bgkimap.hpp  # LocalTensorBuffer类的头文件
# )

# set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Export compile commands for clangd

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3")


# include_directories(${EIGEN3_INCLUDE_DIR})

# add_executable(localSe2tBGKImap ${SOURCES} ${HEADERS})  # 目标可执行文件名
# target_link_libraries(localSe2tBGKImap  ${TORCH_LIBRARIES})




cmake_minimum_required(VERSION 3.21)
project(LocalTensorBufferProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Export compile commands for clangd
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3")

find_package(Torch REQUIRED)

include_directories(include) 


add_executable(LocalTensorBufferProject src/main.cpp 
  src/local_se2t_bgkimap.cpp)  # 主函数文件和LocalTensorBuffer类的实现文件


target_link_libraries(LocalTensorBufferProject  ${TORCH_LIBRARIES})
