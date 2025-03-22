cmake_minimum_required(VERSION 3.21)
project(tensorHelloWorld)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Export compile commands for clangd

find_package(Torch REQUIRED)
add_executable(tensorHelloWorld tensorHelloWorld.cpp)


target_link_libraries(tensorHelloWorld  ${TORCH_LIBRARIES})