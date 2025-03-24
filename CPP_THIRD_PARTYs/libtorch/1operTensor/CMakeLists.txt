cmake_minimum_required(VERSION 3.21)
project(tensorBasicOper)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Export compile commands for clangd

find_package(Torch REQUIRED)
add_executable(tensorBasicOper tensorBasicOper.cpp)


target_link_libraries(tensorBasicOper  ${TORCH_LIBRARIES})