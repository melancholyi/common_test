cmake_minimum_required(VERSION 3.18)
project(MatrixMulCompare LANGUAGES CXX CUDA)

# set 
set(CMAKE_CXX_STANDARD 17)

# Find LibTorch
find_package(Torch REQUIRED)
find_package(PythonLibs REQUIRED) 

# set variables
set(PYTHON_INCLUDE_DIRS /usr/include/python3.10)
set(PYTHON_LIBRARIES /usr/lib/x86_64-linux-gnu/libpython3.10.so)

# include_directories
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${PYTHON_INCLUDE_DIRS})

# Include directories
include_directories(${TORCH_INCLUDE_DIRS})

# Add executable
add_executable(matrix_mul main.cu)

# Link libraries
target_link_libraries(matrix_mul PRIVATE ${TORCH_LIBRARIES} cudart ${PYTHON_LIBRARIES})

# Set C++ standard
set_property(TARGET matrix_mul PROPERTY CXX_STANDARD 14)