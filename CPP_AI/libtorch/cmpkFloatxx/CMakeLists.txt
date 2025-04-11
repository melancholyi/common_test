cmake_minimum_required(VERSION 3.21)
project(cmpkFloatxx)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Export compile commands for clangd

find_package(Eigen3 REQUIRED)
find_package(Torch REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(cmpkFloatxx cmpkFloatxx.cpp)


target_link_libraries(cmpkFloatxx  ${TORCH_LIBRARIES})
