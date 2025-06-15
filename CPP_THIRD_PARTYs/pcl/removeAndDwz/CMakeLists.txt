cmake_minimum_required(VERSION 3.10)
project(removeAndDwz)

find_package(PCL REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(removeAndDwz removeAndDwz.cpp)
target_link_libraries(removeAndDwz ${PCL_LIBRARIES})