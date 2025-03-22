cmake_minimum_required(VERSION 3.21)
project(tensorHelloWorld)

find_package(Torch REQUIRED)
add_executable(tensorHelloWorld tensorHelloWorld.cpp)
target_link_libraries(tensorHelloWorld ${TORCH_LIBRARIES})