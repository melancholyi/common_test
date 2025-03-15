cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(kdtreeTest)

find_package(PCL REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(kdtreeTest kdtreeTest.cpp)
target_link_libraries(kdtreeTest ${PCL_LIBRARIES})