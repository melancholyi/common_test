# Set the minimum version of CMake required
cmake_minimum_required(VERSION 3.10)

# Set the project name
project(FibonacciHeapProject)

# Specify the C++ standard to use
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include the directory where the header file is located
include_directories(${CMAKE_SOURCE_DIR}/FibonacciHeap)

# Add the executable target
add_executable(${PROJECT_NAME} main.cpp)

# If you have any additional source files, add them here
# add_executable(${PROJECT_NAME} main.cpp additional_source.cpp)

# If you need to link any libraries, specify them here
# target_link_libraries(${PROJECT_NAME} library_name)