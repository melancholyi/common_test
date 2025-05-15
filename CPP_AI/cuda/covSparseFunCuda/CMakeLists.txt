cmake_minimum_required(VERSION 3.10)
project(covSparseFunCuda)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3")

find_package(CUDA REQUIRED)
find_package(Torch REQUIRED)


cuda_add_executable(covSparseFunCuda
    covSparseFunCuda.cu
)

target_link_libraries(covSparseFunCuda  ${TORCH_LIBRARIES})
# target_link_libraries(eigen_decomp PRIVATE Eigen3::Eigen)
# target_link_libraries(eigen_decomp PRIVATE CUDA::cudart)  # Adjust based on your CUDA libraries

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(covSparseFunCuda PRIVATE -G -O0)
else()
    target_compile_options(covSparseFunCuda PRIVATE -O3)
endif()