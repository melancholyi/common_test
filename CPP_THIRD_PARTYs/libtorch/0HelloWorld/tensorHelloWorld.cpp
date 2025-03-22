/*
 * @Author: chasey && melancholycy@gmail.com
 * @Date: 2025-03-22 06:41:27
 * @LastEditTime: 2025-03-22 08:43:24
 * @FilePath: /test/CPP_THIRD_PARTYs/libtorch/0HelloWorld/tensorHelloWorld.cpp
 * @Description: 
 * @Reference: 
 * Copyright (c) 2025 by chasey && melancholycy@gmail.com, All Rights Reserved. 
 */
#include <iostream>
#include <torch/torch.h>

void compareCPUAndGPU(){
  // 设置矩阵大小
  int64_t matrix_size = 10000;// NOTE: the matrix's dim must larger enough, otherwise the GPU is slower than CPU bacause of the data transfer time from CPU to GPU.

  // 创建两个随机矩阵
  torch::Tensor cpu_a = torch::rand({matrix_size, matrix_size}, torch::kCPU);
  torch::Tensor cpu_b = torch::rand({matrix_size, matrix_size}, torch::kCPU);

  // 将矩阵复制到 GPU
  torch::Tensor gpu_a = cpu_a.to(torch::kCUDA);
  torch::Tensor gpu_b = cpu_b.to(torch::kCUDA);

  // CPU 矩阵乘法
  auto start_cpu = std::chrono::high_resolution_clock::now();
  torch::Tensor cpu_result = torch::matmul(cpu_a, cpu_b);
  auto end_cpu = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> cpu_duration = end_cpu - start_cpu;
  std::cout << "CPU matmul time: " << cpu_duration.count() << " seconds" << std::endl;

  // GPU 矩阵乘法
  auto start_gpu = std::chrono::high_resolution_clock::now();
  torch::Tensor gpu_result = torch::matmul(gpu_a, gpu_b);
  auto end_gpu = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> gpu_duration = end_gpu - start_gpu;
  std::cout << "GPU matmul time: " << gpu_duration.count() << " seconds" << std::endl;
}



int main() {

  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    std::cout << "=====CUDA is available! Training on GPU." << std::endl;
    device = torch::kCUDA;
  } else {
    std::cout << "=====Training on CPU." << std::endl;
  }


  // 创建一个(2,3)张量
  torch::Tensor tensor = torch::randn({2, 3}).to(device);
  std::cout << "=====tensor: origin tensor\n" << tensor << std::endl;
  tensor += tensor;
  std::cout << "=====tensor: tensor += tensor\n" << tensor << std::endl;
  std::cout << tensor << std::endl;
  std::cout << "\nWelcome to LibTorch!" << std::endl;


  std::cout << "\n==========compareCPUAndGPU==========\n";
  compareCPUAndGPU();

  

  return 0;
}