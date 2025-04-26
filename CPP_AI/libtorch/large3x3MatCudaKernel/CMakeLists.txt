cmake_minimum_required(VERSION 3.18)
project(EigenDecomposition)

# 寻找 LibTorch
find_package(Torch REQUIRED)

# 寻找 CUDA
find_package(CUDA REQUIRED)

include_directories(${TORCH_INCLUDE_DIRS})

# 添加可执行文件
add_executable(EigenDecomposition main.cpp matrix3x3eigendecomp.cu)

# 链接 LibTorch 和 CUDA 库
target_link_libraries(EigenDecomposition ${TORCH_LIBRARIES} ${CUDA_LIBRARIES})

# 设置 CUDA 编译选项
set_target_properties(EigenDecomposition PROPERTIES
                      CUDA_ARCHITECTURES "native"
                      CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")