cmake_minimum_required(VERSION 3.21)
project(LocalTensorBufferProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Export compile commands for clangd
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3")

find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

include_directories(include) 

cuda_add_library(cuda_kernel_lib STATIC
    src/cuda_kernel.cu
)

add_executable(LocalTensorBufferProject src/main.cpp 
  src/local_se2t_bgkimap.cpp)  # 主函数文件和LocalTensorBuffer类的实现文件


target_link_libraries(LocalTensorBufferProject  ${TORCH_LIBRARIES} cuda_kernel_lib)


if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(LocalTensorBufferProject PRIVATE -G -O0)
else()
    target_compile_options(LocalTensorBufferProject PRIVATE -O3)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# 架构指定（根据实际GPU调整）
set(CMAKE_CUDA_ARCHITECTURES "89")  # RTX 20系列为75，A100为80

# NVCC优化选项
target_compile_options(LocalTensorBufferProject PRIVATE
$<$<COMPILE_LANGUAGE:CUDA>:
  -O3
  --use_fast_math
  --ftz=true
  --prec-div=false
  --prec-sqrt=false
  --generate-line-info  # 保留调试信息（可选）
>)

