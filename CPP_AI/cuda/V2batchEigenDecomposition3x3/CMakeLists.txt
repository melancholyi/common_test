cmake_minimum_required(VERSION 3.10)
project(cuda_eigen_decomposition)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)


cuda_add_executable(eigen_decomp
    eigen_decomp.cu
)

# target_link_libraries(eigen_decomp PRIVATE CUDA::cudart)  # Adjust based on your CUDA libraries

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(eigen_decomp PRIVATE -G -O0)
else()
    target_compile_options(eigen_decomp PRIVATE -O3)
endif()