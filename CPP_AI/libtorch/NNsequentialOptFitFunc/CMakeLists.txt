cmake_minimum_required(VERSION 3.21)
project(NNsequentialOptFitFunc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Export compile commands for clangd

# find_package
find_package(Eigen3 REQUIRED)
find_package(Torch REQUIRED)
find_package(PythonLibs REQUIRED) 

# set variables
set(PYTHON_INCLUDE_DIRS /usr/include/python3.10)
set(PYTHON_LIBRARIES /usr/lib/x86_64-linux-gnu/libpython3.10.so)

# include_directories
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${PYTHON_INCLUDE_DIRS})

# exec and target-link
add_executable(NNsequentialOptFitFunc NNsequentialOptFitFunc.cpp)
add_executable(sinFittingTest sinFittingTest.cpp)
add_executable(sinFittingTest2 sinFittingTest2.cpp)

target_link_libraries(NNsequentialOptFitFunc  ${TORCH_LIBRARIES})
target_link_libraries(NNsequentialOptFitFunc  ${PYTHON_LIBRARIES})
target_link_libraries(sinFittingTest  ${TORCH_LIBRARIES})
target_link_libraries(sinFittingTest   ${PYTHON_LIBRARIES})
target_link_libraries(sinFittingTest2  ${TORCH_LIBRARIES})
target_link_libraries(sinFittingTest2   ${PYTHON_LIBRARIES})
