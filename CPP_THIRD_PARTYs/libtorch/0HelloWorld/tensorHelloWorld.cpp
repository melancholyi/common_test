/*
 * @Author: chasey && melancholycy@gmail.com
 * @Date: 2025-03-22 06:41:27
 * @LastEditTime: 2025-03-24 11:59:25
 * @FilePath: /test/CPP_THIRD_PARTYs/libtorch/0HelloWorld/tensorHelloWorld.cpp
 * @Description: 
 * @Reference: 
 * Copyright (c) 2025 by chasey && melancholycy@gmail.com, All Rights Reserved. 
 */
#include <iostream>
#include <chrono>
#include <torch/torch.h>
#include <Eigen/Core>
#include <cmath>

void compareCPUAndGPUTest(){
  // 设置矩阵大小
  int64_t matrix_size = 10000; // NOTE: the matrix's dim must larger enough, otherwise the GPU is slower than CPU bacause of the data transfer time from CPU to GPU.

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

void vec2tensorGPU(){
  std::vector<double> vec = {1, 2, 3, 4, 5, 6};
  torch::Tensor tensor = torch::from_blob(vec.data(), {3, 2}, torch::kFloat64).to(torch::kCUDA);
  std::cout << "====vec2tensor\n" << tensor << std::endl;
  
  
  // 将CUDA张量转移到CPU上 将CPU张量转换为Eigen::MatrixXd
  torch::Tensor cpu_tensor_from_cuda = tensor.to(torch::kCPU).contiguous();
  double* data_ptr = cpu_tensor_from_cuda.data_ptr<double>();  
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mat_tensor(tensor.size(0), tensor.size(1));
  std::memcpy(mat_tensor.data(), data_ptr, tensor.size(0) * tensor.size(1) * sizeof(double));
  std::cout << "====vec2tensor: mat_tensor\n" << mat_tensor << std::endl;
  
}


//////////////////////////////////////PART: computeDistMat /////////////////////////////////////////
using ComputeDistMatFunType = std::function<void(const torch::Tensor&, const torch::Tensor&, torch::Tensor&)>;
const double M2PI = 2.0 * M_PI;
const double kernelScaler_ = 1.0;

void computeDistMat(const torch::Tensor& predX, const torch::Tensor& trainX, torch::Tensor& distMat){
  int num_pred = predX.size(0);
  int num_train = trainX.size(0);
  int num_dim = predX.size(1);

  auto predX_broadcast = predX.unsqueeze(1).expand({num_pred, num_train, num_dim});
  auto trainX_broadcast = trainX.unsqueeze(0).expand({num_pred, num_train, num_dim});
  auto diff = predX_broadcast - trainX_broadcast;

  distMat = (diff * diff).sum({2}).sqrt();
}

void covSparse(const torch::Tensor& distMat, torch::Tensor& kernel) {
  kernel = ((2.0 + (distMat * M2PI).cos()) * (1.0 - distMat) / 3.0 +
                  (distMat * M2PI).sin() / M2PI) * kernelScaler_;

  // // kernel's elem is masked with 0.0 if dist > kernelLen_
  kernel = kernel * (kernel > 0.0).to(torch::kFloat64);
}


void computeDistAndcovSparse(){
  std::vector<double> predXvec{1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12};

  std::vector<double> trainXvec{1, 2, 3, 4, 
      5, 6, 7, 8, 
      9, 10,11, 12, 
      13, 14, 15, 16, 
      17, 18, 19, 20};

  int INPUT_DIM = 4;
  torch::Tensor predX = torch::from_blob(predXvec.data(), {(int64_t)(predXvec.size()/INPUT_DIM), INPUT_DIM}, torch::kFloat64).to(torch::kCUDA);
  torch::Tensor trainX = torch::from_blob(trainXvec.data(), {(int64_t)(trainXvec.size()/INPUT_DIM), INPUT_DIM}, torch::kFloat64).to(torch::kCUDA);
  torch::Tensor distMat;
  auto computeDistMatFunc_ = std::bind(&computeDistMat, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
  computeDistMatFunc_(predX, trainX, distMat);
  std::cout << "=====OUT distMat: \n" << distMat << std::endl;

  torch::Tensor kernel;
  covSparse(distMat, kernel);
  std::cout << "=====OUT kernel: \n" << kernel << std::endl;
  
  auto kernel_mult = kernel.matmul(trainX);
  std::cout << "=====OUT kernel_mult: \n" << kernel_mult << std::endl;
  
}


int main() {
  //! PART: 0 test
  std::cout << "Libtorch version: " << TORCH_VERSION << std::endl;

  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    std::cout << "=====CUDA is available! Training on GPU." << std::endl;
    device = torch::kCUDA;
  } else {
    std::cout << "=====Training on CPU." << std::endl;
  }

  //! PART: 1 basic usage
  std::cout << "\n==========basic usage==========\n";
  // 创建一个(2,3)张量
  torch::Tensor tensor = torch::randn({2, 3}).to(device);
  std::cout << "=====tensor: origin tensor\n" << tensor << std::endl;
  tensor += tensor;
  std::cout << "=====tensor: tensor += tensor\n" << tensor << std::endl;
  std::cout << tensor.sin() << std::endl;
  std::cout << "\nWelcome to LibTorch!" << std::endl;


  //! PART: 2 compareCPUAndGPU
  std::cout << "\n==========compareCPUAndGPU==========\n";
  compareCPUAndGPUTest();

  //! PART: 3 Eigen Decomposition
  std::cout << "\n==========Eigen Decomposition==========\n";
  // Eigen decomposition
  torch::Tensor tensor1 = torch::arange(1.0, 10.0).reshape({3, 3});
  auto eig = torch::linalg::eig(tensor1);
  auto [eigval, eigvec] = eig;
  std::cout << "Eigenvalues: " << eigval << std::endl;
  std::cout << "Eigenvectors: " << eigvec << std::endl;

  auto tensor1_sin = tensor1.sqrt();
  std::cout << "tensor1_sin: " << tensor1_sin << std::endl;
  
  //! PART: 4 vec2tensor2EigMatrix
  std::cout << "\n==========vec2tensor2EigMatrix==========\n";
  vec2tensorGPU();

  //! PART: 5 computeDistMat and covSparse
  std::cout << "\n==========computeDist and covSparse==========\n";
  computeDistAndcovSparse();







  return 0;
}