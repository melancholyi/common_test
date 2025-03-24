cmake_minimum_required(VERSION 3.21)
project(tensorHelloWorld)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Export compile commands for clangd

# Add the path to LibTorch headers
include_directories(/usr/local/include/libtorch/include)

# Add the path to LibTorch libraries
link_directories(/usr/local/include/libtorch/lib)

find_package(Torch REQUIRED)
add_executable(tensorHelloWorld tensorHelloWorld.cpp)


target_link_libraries(tensorHelloWorld  ${TORCH_LIBRARIES})