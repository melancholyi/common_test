cmake_minimum_required(VERSION 3.21)
project(SGP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Export compile commands for clangd

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3")

find_package(Eigen3 REQUIRED)
find_package(Torch REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(SGP_libtorch SGP_libtorch.cpp)


target_link_libraries(SGP_libtorch  ${TORCH_LIBRARIES})
