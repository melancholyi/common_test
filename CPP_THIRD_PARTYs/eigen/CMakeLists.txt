cmake_minimum_required(VERSION 3.10)
project(commonUsage)

set(CMAKE_CXX_STANDARD 11)

# Find Eigen
find_package(Eigen3 REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3")

# Add executable
add_executable(commonUsage commonUsage.cpp)

# Link Eigen
target_include_directories(commonUsage PRIVATE ${EIGEN3_INCLUDE_DIRS})