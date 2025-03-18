cmake_minimum_required(VERSION 3.0.0)
project(plotCpp)   

set(CMAKE_CXX_STANDARD 17)

# find_package
find_package(PythonLibs REQUIRED) 

# set variables
set(PYTHON_INCLUDE_DIRS /usr/include/python3.10)
set(PYTHON_LIBRARIES /usr/lib/x86_64-linux-gnu/libpython3.10.so)

# include_directories
include_directories(${PYTHON_INCLUDE_DIRS})

# executable
add_executable(plotCpp main.cpp)

# target_link  
target_link_libraries(plotCpp ${PYTHON_LIBRARIES})