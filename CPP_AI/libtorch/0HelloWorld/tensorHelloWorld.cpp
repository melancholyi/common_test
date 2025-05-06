/*
 * @Author: chasey && melancholycy@gmail.com
 * @Date: 2025-03-22 06:41:27
 * @LastEditTime: 2025-05-06 15:29:27
 * @FilePath: /test/CPP_AI/libtorch/0HelloWorld/tensorHelloWorld.cpp
 * @Description: 
 * @Reference: 
 * Copyright (c) 2025 by chasey && melancholycy@gmail.com, All Rights Reserved. 
 */
#include <iostream>
#include <chrono>
#include <torch/torch.h>
#include <Eigen/Core>
#include <cmath>
#include <cuda_runtime.h>
#include <cassert>
#include <cassert>

#include <vector>
#include <random>

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
  std::cout << "cuda is available: " << torch::cuda::is_available() << std::endl;
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

  std::cout << "=====torch.cdist() compute distance between vectors" << std::endl;
  auto cdist = torch::cdist(predX, trainX);//NOTE： eulidean distance matrix between two tensor. auto cdist_mat = torch::cdist(tensor1, tensor2)!!![3 x 2] [4 x 2] -> [3 x 4]
  std::cout << "=====cdist fun: \n" << cdist << std::endl;

  torch::Tensor kernel;
  covSparse(distMat, kernel);
  std::cout << "=====OUT kernel: \n" << kernel << std::endl;
  
  auto kernel_mult = kernel.matmul(trainX);
  std::cout << "=====OUT kernel_mult: \n" << kernel_mult << std::endl;
  
}


//////////////////////////////////////PART: create 10x10 matrix3d /////////////////////////////////////////
void create10X10Cov3d(){
  torch::Tensor diag_elements = torch::arange(0, 10 * 10 * 3).view({10, 10, 3});
  std::cout << "=====diag_elements: \n" << diag_elements << std::endl;
  torch::Tensor identity_matrix = torch::eye(3);
  torch::Tensor tensor = diag_elements.unsqueeze(-1) * identity_matrix;
  // std::cout << "=====tensor: \n" << tensor << std::endl;
  tensor = tensor.view({10, 10, 3, 3});
  // std::cout << "=====view over tensor: \n" << tensor << std::endl;

  std::cout << "Matrix at position (0,0):" << std::endl;
  std::cout << tensor.index({0, 0, 1, 1}) << std::endl;
}


/////////////////////////////////////PART: 7 more useful function test /////////////////////////////////////////
void moreUsefulFuncsTest(){
  ////////////////////////////////////// torch::outer /////////////////////////////////////////
  std::cout << "===== PART 7.1: torch::outer \n" << std::endl;
  // Create two 1D tensors
  at::Tensor a = torch::tensor({1, 2, 3}, torch::kFloat64);
  at::Tensor b = torch::tensor({4, 5, 6}, torch::kFloat64);

  at::Tensor outer_product = torch::outer(a, b);

  // Print the result
  std::cout << "Tensor a: " << a << std::endl << "Tensor b: " << b << std::endl;
  std::cout << "Outer product:" << std::endl << outer_product << std::endl;

  ////////////////////////////////////// torch::norm /////////////////////////////////////////
  std::cout << "===== PART 7.2: torch::norm \n" << std::endl;
  auto a_norm = a.norm(2, 0, true); // p = 2(norm formulation) dim = 0(operate on columns) keepdim = true(keep the dimension)
  /*
  a_norm:  3.7417,  sqrt(1^2 + 2^2 + 3^2) = 3.7417
  [ CPUDoubleType{1} ]
  */
  std::cout << "===== a.norm(2, 0, true): " << a_norm << std::endl;
  torch::Tensor a_normed = a/a_norm;
  std::cout << "===== a/a_norm: \n" << a_normed << std::endl;
  torch::Tensor a_norm_detach = a_normed.detach();

  torch::Tensor origin = torch::ones({3,1}, torch::kFloat64);
  std::cout << "===== origin: \n" << origin << std::endl;
  std::cout << "===== origin: \n" << origin.transpose(0,1) << std::endl;
  std::cout << "===== origin.flatten().diag(): \n" << origin.flatten().diag() << std::endl;


  //////////////////////////////////// mask select /////////////////////////////////////////
  std::cout << "===== PART 7.3: tensor.masked_select() \n" << std::endl;
  torch::Tensor tensor_origin = torch::arange(1, 17).reshape({4, 4}).to(torch::kFloat32);
  std::cout << "tensor_origin: \n" << tensor_origin << std::endl;
  torch::Tensor mask = tensor_origin > 8;
  std::cout << "mask: \n" << mask << std::endl;
  torch::Tensor selected = tensor_origin.masked_select(mask);//type: [ CPUFloatType{8} ]
  std::cout << "selected: \n" << selected << std::endl;
  torch::Tensor tensor_mult_mask = tensor_origin * mask;
  std::cout << "tensor_mult_mask: \n" << tensor_mult_mask << std::endl;

  //////////////////////////////////////// tensor.unfold ////////////////////////////////////////////

  //////////////////////////////////////// tensor.einsum //////////////////////////////////////////
  

}

/////////////////////////////////////PART: torch::detach /////////////////////////////////////////
void torchDetachTest(){
  // ===== didn't use detach
  torch::Tensor a1 = torch::tensor({{1.0, 2.0}, {3.0, 4.0}}, torch::requires_grad());
  torch::Tensor b1 = a1 * 2;
  torch::Tensor c1 = b1 + b1;
  c1.sum().backward();// c1 = 4 * a1  d(c1)/d(a1) = 4
  /*
  在 PyTorch 和 LibTorch 中，backward() 函数用于执行反向传播，计算梯度。但是，backward() 函数需要一个标量（scalar）作为输入，因为它计算的是标量关于张量的梯度。
  如果直接对一个多元素的张量调用 backward()，会报错，因为无法确定如何对一个非标量张量进行梯度计算。
  为什么需要 sum()？
  当你有一个多元素的张量（例如一个矩阵或向量）时，你需要将其转换为一个标量，以便能够调用 backward()。sum() 函数的作用是将张量的所有元素相加，从而生成一个标量。这样，你就可以对这个标量调用 backward() 来计算梯度。
  为什么需要标量？
  在反向传播中，梯度是通过链式法则计算的。链式法则要求从一个标量输出开始，逐步计算每个张量的梯度。如果输入不是一个标量，链式法则无法正常工作，因为无法确定如何从一个非标量张量开始计算梯度。
  */
  std::cout << "Gradients of a (without detach):\n" << a1.grad() << std::endl;


  // ===== use detach
  torch::Tensor a2 = torch::tensor({{1.0, 2.0}, {3.0, 4.0}}, torch::requires_grad());
  torch::Tensor b2 = a2 * 2;
  auto b2_detached = b2.detach();
  torch::Tensor c2 = b2_detached + b2_detached;
  // c2.sum().backward();// error, because auto b2_detached = b2.detach(); the autograd was detach at b2.
  b2.sum().backward();// The autograd computation graph is end at b2, so the gradients of a2 is 2
  std::cout << "Gradients of a (with detach at b2_detached):\n" << a2.grad() << std::endl;
}
/////////////////////////////////////PART: [x, y, theta, t] custom distance /////////////////////////////////////////
void testCustomDistXYThetaT(){
  //NOTE: parameters  
  // kernel len, distance angleLen timeLen 
  std::vector<double> kernelLen{1.0, 0.5, 1.0};

  // 初始化predX和trainX
  torch::Tensor predX = torch::arange(1, 13).reshape({3, 4}).to(torch::kFloat32); // 形状为(3,4)
  torch::Tensor trainX = torch::arange(1, 9).reshape({2, 4}).to(torch::kFloat32); // 形状为(2,4)
  std::cout << "===== Origin Dataset =====" << std::endl;
  std::cout << "=predX: \n" << predX << std::endl;
  std::cout << "=trainX: \n" << trainX << std::endl;

  // 计算欧几里得距离（前两维）
  torch::Tensor predX_xy = predX.index({torch::indexing::Slice(), torch::indexing::Slice(0, 2)});
  torch::Tensor trainX_xy = trainX.index({torch::indexing::Slice(), torch::indexing::Slice(0, 2)});
  torch::Tensor euclidean_dist = torch::sqrt(torch::sum(torch::pow(predX_xy.unsqueeze(1) - trainX_xy.unsqueeze(0), 2), 2)).unsqueeze(2);
  torch::Tensor edist_divlen = euclidean_dist / kernelLen[0];
  std::cout << "===== the first two dimensions' euclidean distance =====" << std::endl;
  std::cout << "predX_xy: \n" << predX_xy << std::endl;
  std::cout << "trainX_xy: \n" << trainX_xy << std::endl;
  std::cout << "euclidean_dist: \n" << euclidean_dist << std::endl;
  std::cout << "edist_divlen: \n" << edist_divlen << std::endl;

  // 计算角度差的绝对值（第三维）
  torch::Tensor predX_angle = predX.index({torch::indexing::Slice(), 2});
  torch::Tensor trainX_angle = trainX.index({torch::indexing::Slice(), 2});
  torch::Tensor angle_diff = torch::abs(predX_angle.unsqueeze(1) - trainX_angle.unsqueeze(0)).unsqueeze(2);
  torch::Tensor angle_diff_divlen = angle_diff / kernelLen[1];
  std::cout << "===== the third dimension's angle difference =====" << std::endl;
  std::cout << "predX_angle: \n" << predX_angle << std::endl;
  std::cout << "trainX_angle: \n" << trainX_angle << std::endl;
  std::cout << "angle_diff: \n" << angle_diff << std::endl;
  std::cout << "angle_diff_divlen: \n" << angle_diff_divlen << std::endl;

  // 计算指数距离（第四维）
  torch::Tensor predX_theta = predX.index({torch::indexing::Slice(), 3});
  torch::Tensor trainX_theta = trainX.index({torch::indexing::Slice(), 3});
  torch::Tensor exp_dist = torch::exp(-torch::abs(predX_theta.unsqueeze(1) - trainX_theta.unsqueeze(0))).unsqueeze(2);
  torch::Tensor exp_dist_divlen = exp_dist / kernelLen[2];
  std::cout << "===== the fourth dimension's exp distance =====" << std::endl;
  std::cout << "predX_theta: \n" << predX_theta << std::endl;
  std::cout << "trainX_theta: \n" << trainX_theta << std::endl;
  std::cout << "exp_dist: \n" << exp_dist << std::endl;
  std::cout << "exp_dist_divlen: \n" << exp_dist_divlen << std::endl;

  // 将三个距离合并成一个张量
  torch::Tensor final_dist = torch::cat({euclidean_dist, angle_diff, exp_dist}, 2);
  std::cout << "===== Final distance =====" << std::endl;
  std::cout << "Final distance shape: " << final_dist.sizes() << std::endl;
  std::cout << "Final distance: " << final_dist << std::endl;


  // compute kernel for everyone elem  
  const double M2PI = 2.0 * M_PI;
  double kernelScaler_ = 1.0;
  torch::Tensor kernel = ((2.0 + (final_dist * M2PI).cos()) * (1.0 - final_dist) / 3.0 +
                  (final_dist * M2PI).sin() / M2PI) * kernelScaler_;
  std::cout << "===== kernel(origin): \n" << kernel << std::endl;
  // kernel's elem is masked with 0.0 if dist > kernelLen_
  kernel = kernel * (kernel > 0.0).to(torch::kFloat64);
  std::cout << "===== kernel(in len): \n" << kernel << std::endl;

  // 计算加权和
  torch::Tensor kernel_sum = kernel.sum(2);
  std::cout << "===== kernel_sum: \n" << kernel_sum << std::endl;
}
/////////////////////////////////////PART: testAutoGradGPUTensor /////////////////////////////////////////
void testAutoGradGPUTensor(){
  // 创建一个需要求导的tensor
  torch::Tensor x = torch::randn({2, 3}, torch::kCUDA).requires_grad_();
  torch::Tensor weight = torch::randn({4, 3}, torch::kCUDA).requires_grad_();
  
  // 定义一个简单的线性函数
  auto y = x.mm(weight.t());
  
  // 计算梯度
  y.sum().backward();
  
  // 输出梯度
  std::cout << "Gradient of x:" << std::endl;
  std::cout << x.grad() << std::endl;
  
  std::cout << "Gradient of weight:" << std::endl;
  std::cout << weight.grad() << std::endl;
}


///////////////////////////////////PART: MIN-DELTA-MAX generate matrix
// 函数定义
torch::Tensor generate_arithmetic_sequences(
  const std::vector<float>& min_list,
  const std::vector<float>& max_list,
  int count
) {
  // 检查min_list和max_list的大小是否一致
  assert(min_list.size() == max_list.size());

  // 将min_list和max_list转换为torch::Tensor
  torch::Tensor min_tensor = torch::tensor(min_list, torch::kFloat32);
  torch::Tensor max_tensor = torch::tensor(max_list, torch::kFloat32);

  // 生成一个从0到count-1的序列
  torch::Tensor indices = torch::arange(0, count, torch::kFloat32);

  // 使用广播机制生成等差数列
  torch::Tensor result = min_tensor.unsqueeze(1) + (max_tensor - min_tensor).unsqueeze(1) * indices / (count - 1);

  return result;
}

/////////////////////////////////////////PART:13 meanVec2TensorAndChangeDim /////////////////////////////////////////
void meanVec2TensorAndChangeDim(){
  // 创建一个包含 49 个 Eigen::Vector3d 的 std::vector
  std::vector<Eigen::Vector3d> vec;
  for (int i = 0; i < 49; ++i) {
      vec.emplace_back(Eigen::Vector3d(i * 3, i * 3 + 1, i * 3 + 2));
      std::cout << "vec[" << i << "] = " << vec[i].transpose() << std::endl;
  }

  // 将 std::vector<Eigen::Vector3d> 转换为形状为 [3, 7, 7] 的 Tensor
  torch::Tensor tensor = torch::from_blob(vec.data(), {49, 3}, torch::kFloat64).view({3, 7, 7});
  std::cout << "Original tensor shape: " << tensor.sizes() << std::endl;
  std::cout << "Original tensor: " << tensor << std::endl;

  // 使用 permute 方法改变维度顺序为 [7, 7, 3]
  torch::Tensor permuted_tensor = tensor.permute({2, 1, 0});
  std::cout << "Permuted tensor shape: " << permuted_tensor.sizes() << std::endl;

  for(int i = 0 ; i < 7; i++){
    for(int j = 0; j < 7; j++){
        std::cout << "permuted_tensor[" << i << "][" << j << "] = " << (permuted_tensor[i][j]).unsqueeze(0) << std::endl;
    }
  }
}

///////////////////////////////////PART:14 unsqueeze and view /////////////////////////////////////////
void unsqueezeTest(){
  // Create a tensor of shape [7x7]
  torch::Tensor tensor_7x7 = torch::arange(1, 50).reshape({7, 7}).to(torch::kFloat32);
  std::cout << "Original tensor_7x7:\n" << tensor_7x7 << std::endl;

  // Add a dimension at position 2, shape becomes [7x7x3]
  torch::Tensor tensor_7x7x3 = tensor_7x7.unsqueeze(2).expand({7, 7, 3});
  std::cout << "Broadcasted tensor_7x7x3:\n" << tensor_7x7x3 << std::endl;

  // Add a dimension at position 2, shape becomes [4x7x7x3]
  torch::Tensor tensor_4x7x7x3 = tensor_7x7x3.unsqueeze(0).expand({4, 7, 7, 3});
  std::cout << "Broadcasted tensor_4x7x7x3[0]:\n" << tensor_4x7x7x3[0] << std::endl; //NOTE: = tensor_7x7x3
  std::cout << "Broadcasted tensor_4x7x7x3[1]:\n" << tensor_4x7x7x3[1] << std::endl; //NOTE: = tensor_7x7x3
  std::cout << "Broadcasted tensor_4x7x7x3[2]:\n" << tensor_4x7x7x3[2] << std::endl; //NOTE: = tensor_7x7x3
  std::cout << "Broadcasted tensor_4x7x7x3[3]:\n" << tensor_4x7x7x3[3] << std::endl; //NOTE: = tensor_7x7x3

  // Add a dimension at position 2, shape becomes [2x4x7x7x3]
  torch::Tensor tensor_2x4x7x7x3 = tensor_4x7x7x3.unsqueeze(0).expand({2, 4, 7, 7, 3}); //
  // std::cout << "Broadcasted tensor_2x4x7x7x3[0]:\n" << tensor_2x4x7x7x3[0] << std::endl; //NOTE: = tensor_4x7x7x3
  // std::cout << "Broadcasted tensor_2x4x7x7x3[1]:\n" << tensor_2x4x7x7x3[1] << std::endl; //NOTE: = tensor_4x7x7x3


  
}

//////////////////////////////////PART:15 stack tensor /////////////////////////////////////////
void stackTensors(){
  // Create a vector of tensors
  std::vector<torch::Tensor> tensors;
  for (int i = 0; i < 5; ++i) {
    // torch::Tensor tensor = torch::ones({7, 7})*i;
    torch::Tensor tensor = torch::rand({7, 7});
    tensors.push_back(tensor);
  }

  //NOTE: First stack data, Second unsqueeze&expand
  // Stack the tensors along a new dimension
  torch::Tensor stacked_tensor = torch::stack(tensors, /*dim=*/0);// [5, 7, 7]
  std::cout << "Stacked tensor shape: " << stacked_tensor.sizes() << std::endl;

  // torch::Tensor tensor_5x42x7x7 = stacked_tensor.unsqueeze(1).expand({5, 42, 7, 7});
  // std::cout << "tensor_5x42x7x7 shape: " << tensor_5x42x7x7.sizes() << std::endl;

  // torch::Tensor tensor_5x43x42x7x7 = tensor_5x42x7x7.unsqueeze(1).expand({5, 43, 42, 7, 7});
  // std::cout << "tensor_5x43x42x7x7 shape: " << tensor_5x43x42x7x7.sizes() << std::endl;
  torch::Tensor tensor_5x43x42x7x7 = stacked_tensor.unsqueeze(1).unsqueeze(1).expand({-1, 43, 42, -1, -1});
  std::cout << "tensor_5x43x42x7x7 shape: " << tensor_5x43x42x7x7.sizes() << std::endl;

  torch::Tensor tensor_5x43x42x7x7x3 = tensor_5x43x42x7x7.unsqueeze(-1).expand({-1, -1, -1, -1, -1, 3}); 
  std::cout << "tensor_5x43x42x7x7x3 shape: " << tensor_5x43x42x7x7x3.sizes() << std::endl;

  //NOTE: First unsqueeze&expand, Second stack data
  std::vector<torch::Tensor> tensors_unsqueezed;
  bool is_first = true;
  for(auto origin : tensors){
    torch::Tensor tensor_42x7x7 = origin.unsqueeze(0).expand({42, 7, 7});
    if(is_first){
      std::cout << "tensor_42x7x7 shape: " << tensor_42x7x7.sizes() << std::endl;
    }
    torch::Tensor tensor_43x42x7x7 = tensor_42x7x7.unsqueeze(0).expand({43, 42, 7, 7});
    if(is_first){
      std::cout << "tensor_43x42x7x7 shape: " << tensor_43x42x7x7.sizes() << std::endl;
    }
    torch::Tensor tensor_43x42x7x7x3 = tensor_43x42x7x7.unsqueeze(-1).expand({43, 42, 7, 7, 3});
    if(is_first){
      std::cout << "tensor_43x42x7x7x3 shape: " << tensor_43x42x7x7x3.sizes() << std::endl;
    }
    tensors_unsqueezed.push_back(tensor_43x42x7x7x3);
    is_first = false;
  }
  auto second_tensor_5x43x42x7x7x3 = torch::stack(tensors_unsqueezed, /*dim=*/0);// [5, 43, 42, 7, 7, 3]
  std::cout << "second_tensor_5x43x42x7x7x3 shape: " << second_tensor_5x43x42x7x7x3.sizes() << std::endl;

  auto diff = tensor_5x43x42x7x7x3 - second_tensor_5x43x42x7x7x3;
  auto diff_sum = diff.sum();

  std::cout << "two ways tensor diff_sum: " << diff_sum << std::endl;
}

//////////////////////////////////PART:16 batch eigen-Decomposition /////////////////////////////////////////
void batchMatrixEigenDecomposition(){
  
  // data_size 31, 37, 36, 3, 3
  auto device_ = torch::kCUDA;   // eig: 84 ~ 100ms   eigh: 38 ~ 42ms
  // auto device_ = torch::kCPU; // eig: 74 ~  80ms   eigh: 39 ~ 45ms
  // Create a tensor of shape [31, 37, 36, 3, 3] representing covariance matrices
  torch::Tensor covs = torch::randn({31,100, 100, 3, 3}).to(device_);
  covs = torch::matmul(covs.transpose(-1, -2), covs); // Ensure the matrices are symmetric

  // Flatten the first three dimensions to create a batch of matrices
  torch::Tensor covs_flat = covs.view({covs.size(0) * covs.size(1) * covs.size(2) , covs.size(3), covs.size(4)});

  // Create CUDA events for timing
  //----------------------------------------
  //----------------------------------------
  cudaEvent_t start, end;
  
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start);
  std::chrono::high_resolution_clock::time_point time_start_cpu;
  time_start_cpu = std::chrono::high_resolution_clock::now();


  // Perform eigenvalue decomposition on GPU
  std::cout << "covs_flat.device: " << covs_flat.device() << "  covs_flat.sizes(): " << covs_flat.sizes() << std::endl;
  // auto [eigenvalues_flat, eigenvectors_flat] = torch::linalg::eig(covs_flat/*", L"*/);
  auto [eigenvalues_flat, eigenvectors_flat] = torch::linalg::eigh(covs_flat, "L");
  std::cout << "!eigenvalues_flat.sizes: " << eigenvalues_flat.sizes() << std::endl;
  // std::cout << "eigenvalues_flat[0]:" << eigenvalues_flat[0] << std::endl;

  std::cout << "eigenvectors_flat.sizes: " << eigenvectors_flat.sizes() << std::endl;
  // std::cout << "eigenvectors_flat[0]:\n" << eigenvectors_flat[0] << std::endl;

  cudaEventRecord(end);
  cudaEventSynchronize(end);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, end);
  std::cout << "device: CUDA:0 memsurement | " << "Eigenvalue decomposition runtime: " << milliseconds << " ms" << std::endl;

  auto time_end_cpu = std::chrono::high_resolution_clock::now();
  auto duration_cpu = std::chrono::duration_cast<std::chrono::microseconds>(time_end_cpu - time_start_cpu);
  std::cout << "device: CPU    memsurement | " << "Eigenvalue decomposition runtime: " << duration_cpu.count()/1000.0 << " ms" << std::endl;


  //--------------------------------------- 
  //----------------------------------------
  
  // std::cout << "eqrly exit here" << std::endl;
  // exit(0);


  // // Reshape the results back to the original shape
  // eigenvalues_flat.to(torch::kCPU); 
  // eigenvectors_flat.to(torch::kCPU);
  // torch::Tensor eigenvalues = torch::real(eigenvalues_flat).view({31, 43, 42, 3});
  // torch::Tensor eigenvectors = torch::real(eigenvectors_flat).view({31, 43, 42, 3, 3});

  // // For comparison, perform the decomposition using nested loops
  // torch::Tensor eigenvalues_loop = torch::empty({31, 43, 42, 3});
  // torch::Tensor eigenvectors_loop = torch::empty({31, 43, 42, 3, 3});
  // for (int i = 0; i < 31; ++i) {
  //     for (int j = 0; j < 43; ++j) {
  //         for (int k = 0; k < 42; ++k) {
  //             auto [eigvals, eigvecs] = torch::linalg::eig(covs[i][j][k] /*", L"*/);
  //             eigenvalues_loop[i][j][k] = torch::real(eigvals);
  //             eigenvectors_loop[i][j][k] = torch::real(eigvecs);
  //         }
  //     }
  // }

  // // Compare the results from the two methods
  // auto eigenvalues_diff = (eigenvalues.to(torch::kCPU) - eigenvalues_loop).abs().max().item<float>();
  // auto eigenvectors_diff = (eigenvectors.to(torch::kCPU) - eigenvectors_loop).abs().max().item<float>();
  // std::cout << "Maximum difference in eigenvalues: " << eigenvalues_diff << std::endl;
  // std::cout << "Maximum difference in eigenvectors: " << eigenvectors_diff << std::endl;
  

  // // Print the data at the specified location
  // std::cout << "eigenvalues[12][16][13]:" << eigenvalues[12][16][13].unsqueeze(0) << std::endl;
  // std::cout << "eigenvectors[12][16][13]:\n" << eigenvectors[12][16][13] << std::endl;
  // std::cout << "eigenvalues_loop[12][16][13]:" << eigenvalues_loop[12][16][13].unsqueeze(0) << std::endl;
  // std::cout << "eigenvectors_loop[12][16][13]:\n" << eigenvectors_loop[12][16][13] << std::endl;

  // torch::Tensor tensor_3x3 = torch::arange(1, 10).reshape({3, 3}).to(torch::kFloat32);
  // auto [eigvals, eigvecs] = torch::linalg::eig(tensor_3x3);
  // std::cout << "tensor_3x3: \n" << tensor_3x3 << std::endl;
  // std::cout << "eigvals: \n" << eigvals << std::endl;
  // std::cout << "eigvecs: \n" << eigvecs << std::endl;
  // /*
  // tensor_3x3: 
  // 1  2  3
  // 4  5  6
  // 7  8  9
  // [ CPUFloatType{3,3} ]
  // eigvals: 
  // 1.6117e+01
  // -1.1168e+00
  // 2.9486e-07
  // [ CPUComplexFloatType{3} ]
  // eigvecs: 
  // -0.2320 -0.7858  0.4082
  // -0.5253 -0.0868 -0.8165
  // -0.8187  0.6123  0.4082
  // [ CPUComplexFloatType{3,3} ]
  // */
}

void batchMatrixEigenDecomposition2(){
}

/////////////////////////////////////////PART:17 batch select min eval and correspending-evec /////////////////////////////////////////

void selectMinEigenvalAndEigenVec(){
  // Example tensor shapes
  torch::Tensor tensor_se2_evals = torch::randn({31, 37, 36, 3}); // Eigenvalues
  torch::Tensor tensor_se2_evecs = torch::randn({31, 37, 36, 3, 3}); // Eigenvectors
  

  // extract min evals and indices
  auto [min_evals_value, min_evals_indices] = tensor_se2_evals.min(-1, true); 
  std::cout << "min_evals_value shape: " << min_evals_value.sizes() << std::endl;// [31, 37, 36, 1]
  std::cout << "min_evals_indices shape: " << min_evals_indices.sizes() << std::endl;//[31, 37, 36, 1]

  //! extract min evecs  
  auto expanded_indices = min_evals_indices.unsqueeze(3).expand({-1, -1, -1, 3, -1});
  std::cout << "expanded_indices shape: " << expanded_indices.sizes() << std::endl;// [31, 37, 36, 3, 1]
  torch::Tensor selected_evecs = tensor_se2_evecs.gather(4, expanded_indices);
  std::cout << "selected_evecs shape: " << selected_evecs.sizes() << std::endl;// [31, 37, 36, 3, 1]

  //! print certain to verify  
  std::cout << "tensor_se2_evals[0][0][0]:\n" << tensor_se2_evals[0][0][0] << std::endl;
  std::cout << "min_evals_value[0][0][0]:\n" << min_evals_value[0][0][0] << std::endl;
  std::cout << "min_evals_indices[0][0][0]:\n" << min_evals_indices[0][0][0] << std::endl;

  std::cout << "tensor_se2_evecs[0][0][0]:\n" << tensor_se2_evecs[0][0][0] << std::endl;
  std::cout << "selected_evecs[0][0][0]:\n" << selected_evecs[0][0][0] << std::endl;




  // // 提取最小特征值的索引
  // auto indices = torch::argmin(tensor_se2_evals, /*dim=*/-1);
  // std::cout << "indices shape: " << indices.sizes() << std::endl;

  // // 收集最小特征值，保持维度
  // auto min_vals = tensor_se2_evals.gather(/*dim=*/-1, indices.unsqueeze(-1));
  // std::cout << "min_vals shape: " << min_vals.sizes() << std::endl;

  // // 提取对应的特征向量
  // auto selected_evecs = tensor_se2_evecs.index({"...", indices, torch::indexing::Slice()});
  // std::cout << "selected_evecs shape: " << selected_evecs.sizes() << std::endl;
}

///////////////////////////////////////////PART: 18 mask tensor test //////////////////////////////////////
void maskTensorTest(){
  // 假设 selected_evecs 是一个形状为 [31, 36, 36, 3, 1] 的张量
  // 这里创建一个示例张量用于演示
  torch::Tensor selected_evecs = torch::randn({31, 36, 36, 3, 1});

  // 提取最后一个维度的第三个元素（索引为 2）
  auto third_elements = selected_evecs.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 2, 0});

  // 创建一个掩码，标记第三个元素为负的位置
  auto mask = third_elements < 0.0;

  // 将掩码扩展到与 selected_evecs 相同的形状
  auto mask_expanded = mask.unsqueeze(-1).unsqueeze(-1);

  // 使用掩码对向量进行取反操作
  auto corrected_evecs = torch::where(mask_expanded, -selected_evecs, selected_evecs);

  // 打印处理后的张量
  std::cout << "Processed tensor shape: " << corrected_evecs.sizes() << std::endl;

  // 打印一些随机位置的向量信息
  std::mt19937 gen(std::random_device{}());
  std::uniform_int_distribution<int> dis_i(0, 30); // 对应维度 31
  std::uniform_int_distribution<int> dis_j(0, 35); // 对应维度 36
  std::uniform_int_distribution<int> dis_k(0, 35); // 对应维度 36

  // 打印几个随机位置的向量
  for (int sample = 0; sample < 5; ++sample) {
      int i = dis_i(gen);
      int j = dis_j(gen);
      int k = dis_k(gen);

      // 获取原始向量和修正后的向量
      auto original_vector = selected_evecs[i][j][k];
      auto corrected_vector = corrected_evecs[i][j][k];
      bool is_negative = mask[i][j][k].item<bool>();

      std::cout << "\nSample " << sample + 1 << " - Position (" << i << ", " << j << ", " << k << "):" << std::endl;
      std::cout << "Original Vector: \n" << original_vector.squeeze(-1) << std::endl;
      std::cout << "Corrected Vector: \n" << corrected_vector.squeeze(-1) << std::endl;
      std::cout << "Mask (Is Negative): " << std::boolalpha << is_negative << std::endl;
  }
}

///////////////////////////////////////////PART: 19 tensor padding //////////////////////////////////////
void tensorPadding(){
  // 创建一个形状为 {5,5} 的 tensor
  torch::Tensor tensor = torch::rand({5, 5});
  std::cout << "原始 tensor:\n" << tensor << std::endl;

  // 定义填充参数：宽度左右各1和2，高度上下各1和2
  auto options = torch::nn::functional::PadFuncOptions({1, 2, 1, 2})
    .mode(torch::kConstant)  // 使用常数填充模式
    .value(0);               // 填充值为0
  torch::Tensor padded_tensor = torch::nn::functional::pad(tensor, options);

  std::cout << "填充后的 tensor:\n" << padded_tensor << std::endl;
}
//////////////////////////////////////////////PART: 20 se2 Tensor unfold //////////////////////////////////////
void se2TensorUnfold(){
  long windows_dimyaw = 2;
  long windows_dimxy = 3;

  std::vector<int> se2_dim = {5, 10, 10, 1};
  int product = std::accumulate(se2_dim.begin(), se2_dim.end(), 1, std::multiplies<int>());
  std::cout << "product: " << product << std::endl;

  auto tensor = torch::arange(0, product).reshape({se2_dim[0], se2_dim[1], se2_dim[2], se2_dim[3]}).to(torch::kFloat32);
  std::cout << "tensor.sizes(): " << tensor.sizes() << std::endl;

  auto yaw_unfold = tensor.unfold(0, windows_dimyaw, 1);
  std::cout << "yaw_unfold.sizes(): " << yaw_unfold.sizes() << std::endl;
  
  // std::cout << "tensor[0][0][0]: " << tensor[0][0][0]<< std::endl;
  // std::cout << "tensor[1][0][0]: " << tensor[1][0][0] << std::endl;
  // std::cout << "yaw_unfold[0][0][0]: " << yaw_unfold[0][0][0] << std::endl;
  
  // std::cout << "tensor[0]: " << tensor[0]<< std::endl;
  // std::cout << "tensor[1]: " << tensor[1] << std::endl;
  // std::cout << "yaw_unfold[0]: " << yaw_unfold[0] << std::endl;
  
  auto xy_unfold = yaw_unfold.unfold(1, windows_dimxy, 1).unfold(2, windows_dimxy, 1);
  std::cout << "xy_unfold.sizes(): " << xy_unfold.sizes() << std::endl;
}

/////////////////////////////////////////////PART: 21 tensor slice and assign //////////////////////////////////////
void tensorSliceAndAssign() {
    // 创建一个 7x6x10 的三维 tensor，默认为浮点类型，并在 GPU 上分配
    torch::Tensor big_tensor = torch::zeros({7, 6, 10}, torch::device(torch::kCUDA));

    // 创建7个单独的二维 tensor 4x6的张量，并填充特定的值
    std::vector<torch::Tensor> small_tensors;
    for (int i = 0; i < 7; ++i) {
        if (i == 0) {
            small_tensors.push_back(torch::arange(1, 25).view({4, 6})); // 生成24个元素
            std::cout << "small_tensors[0]: \n" << small_tensors[0] << std::endl;
        } else if (i == 1) {
            small_tensors.push_back(torch::arange(100, 124).view({4, 6})); // 生成24个元素
            std::cout << "small_tensors[1]: \n" << small_tensors[1] << std::endl;
        } else if (i == 2) {
            small_tensors.push_back(torch::arange(1000, 1024).view({4, 6})); // 生成24个元素
            std::cout << "small_tensors[" << i << "]: \n" << small_tensors[i] << std::endl;
        } else {
            // 生成其他张量数据，确保元素数量为24
            small_tensors.push_back(torch::arange(i * 1000 + 1, i * 1000 + 25).view({4, 6}));
            std::cout << "small_tensors[" << i << "]: \n" << small_tensors[i] << std::endl;
        }
    }

    // 为每个小 tensor 生成随机索引
    std::vector<std::tuple<int, int, int>> random_indices;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_dim1(0, 5); // 第二维的索引范围是 0-5
    std::uniform_int_distribution<> dis_dim2(0, 9); // 第三维的索引范围是 0-9

    for (int i = 0; i < 7; ++i) {
        int dim0 = i;
        int dim1 = dis_dim1(gen);
        int dim2 = dis_dim2(gen);
        random_indices.push_back(std::make_tuple(dim0, dim1, dim2));
        std::cout << "Random indices for small tensor " << i << ": (" << dim0 << ", " << dim1 << ", " << dim2 << ")" << std::endl;
    }

    // 将每个小 tensor 填充到三维 tensor 的相应位置
    for (int i = 0; i < 7; ++i) {
        int dim0 = std::get<0>(random_indices[i]);
        int dim1 = std::get<1>(random_indices[i]);
        int dim2 = std::get<2>(random_indices[i]);

        // 计算可以容纳的小 tensor 的尺寸
        int available_rows = std::min(4, static_cast<int>(big_tensor.size(1)) - dim1);
        int available_cols = std::min(6, static_cast<int>(big_tensor.size(2)) - dim2);

        // 裁剪小 tensor
        torch::Tensor cropped_small_tensor = small_tensors[i].slice(0, 0, available_rows).slice(1, 0, available_cols);

        // 填充到大 tensor
        big_tensor.index({dim0, torch::indexing::Slice(dim1, dim1 + available_rows), torch::indexing::Slice(dim2, dim2 + available_cols)})
            .copy_(cropped_small_tensor);
    }

    // 打印填充后的三维 tensor
    std::cout << "Big Tensor after filling:" << std::endl;
    std::cout << big_tensor << std::endl;

    // 定义窗口大小
    int windows_dim0 = 3;
    int windows_dim1 = 5;
    int windows_dim2 = 5;
    assert(windows_dim0 % 2 == 1 && windows_dim1 % 2 == 1 && windows_dim2 % 2 == 1);
      
    /*
    dim0:  origin0 - windows_dim0 + 1   
    dim1:  origin1 - windows_dim1 + 1   
    dim2:  origin2 - windows_dim2 + 1   
    */

    // 计算 padding 大小以保持 unfold 后的维度不变
    int pad_dim0 = (windows_dim0 - 1) / 2;
    int pad_dim1 = (windows_dim1 - 1) / 2;
    int pad_dim2 = (windows_dim2 - 1) / 2;
    std::cout << "pad_dim0: " << pad_dim0 << std::endl;//1
    std::cout << "pad_dim1: " << pad_dim1 << std::endl;//2
    std::cout << "pad_dim2: " << pad_dim2 << std::endl;//2


    // 定义填充参数：宽度左右各1和2，高度上下各1和2
    //NOTE: padding is the last dim -> first dim
    auto options = torch::nn::functional::PadFuncOptions({pad_dim2, pad_dim2, pad_dim1, pad_dim1, pad_dim0, pad_dim0})
      .mode(torch::kConstant)  // 使用常数填充模式
      .value(1e5);               // 填充值为0
    torch::Tensor big_tensor_padded = torch::nn::functional::pad(big_tensor, options);
    std::cout << "big_tensor_padded.sizes(): " << big_tensor_padded.sizes() << std::endl;

    // 执行 unfold 操作
    auto big_tensor_unfold = big_tensor_padded.unfold(0, windows_dim0, 1)
        .unfold(1, windows_dim1, 1).unfold(2, windows_dim2, 1);


    std::cout << "big_tensor_unfold.sizes(): " << big_tensor_unfold.sizes() << std::endl;
    std::cout << "big_tensor_unfold[6][4][9]: \n" << big_tensor_unfold[6][4][9].to(torch::kCPU) << std::endl;
}

///////////////////////////////////////////PART: 22 FixedTensorBuffer //////////////////////////////////////
class FixedTensorBuffer{
  public://membership function
    FixedTensorBuffer(const int& capacity): capacity_(capacity){
      device_ = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    }
    size_t size() const {
      return data_.size();
    }
    bool empty() const {
      return data_.empty();
    }
    void insert(const torch::Tensor& tensor) {
      if (data_.size() >= capacity_) {
        data_.pop_front();
      }
      auto temp = tensor;
      temp = temp.to(device_);
      data_.push_back(temp);
    }
    const torch::Tensor& getTensor(const size_t& index) const {
      if (index >= data_.size()) {
          throw std::out_of_range("Index out of range");
      }
      return data_[index];
    }
  private://membership function

  public://membership variable

  private://membership variable
    std::deque<torch::Tensor> data_;
    int capacity_;
    torch::DeviceType device_;
};



void testFixedTensorBuffer(){
  FixedTensorBuffer buffer(3);
  std::cout << "buffer.size(): " << buffer.size() << std::endl;
  std::cout << "buffer.empty(): " << buffer.empty() << std::endl;

  for (int i = 0; i < 5; ++i) {
    torch::Tensor tensor = torch::ones({i+1, i+1}) * i;
    tensor.to(torch::kCUDA);
    buffer.insert(tensor);
    std::cout << "buffer.size(): " << buffer.size() << std::endl;
    std::cout << "buffer.empty(): " << buffer.empty() << std::endl;
    std::cout << "buffer.getTensor("<< 0 <<"):\n" << buffer.getTensor(0) << std::endl;
  }
  std::cout << "buffer.getTensor("<< 2 <<"):\n" << buffer.getTensor(2) << std::endl;
}

//////////////////////////////////////////////PART: 23 LocalTensorBuffer //////////////////////////////////////
// class LocalTensorBuffer{
//   public://membership function
//     LocalTensorBuffer(const int& capacity): capacity_(capacity){
//       device_ = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
//     }
//     size_t size() const {
//       return data_.size();
//     }
//     bool empty() const {
//       return data_.empty();
//     }
//     /**
//      * @brief:: 
//      * @param {Tensor&} se2Info: 
//      * @param {Tensor&} gridPos: 
//      * @attention: 
//      * @return {*}
//      */    
//     void insert(const torch::Tensor& se2Info, const torch::Tensor& gridPos){
//       torch::Tensor tensor_fused, cov_fused;
//       auto temp1 = se2Info;
//       auto temp2 = gridPos;
//       temp1 = temp1.to(device_);
//       temp2 = temp2.to(device_);
//       if (data_.size() >= capacity_) {
//         //TODO: fuse the new data with the old data by using STBGKI regression
//         [tensor_fused, cov_fused]  = fuseThroughSpatioTemporalBGKI();
//         data_.pop_front();
//       }
//       data_.push_back({tensor_fused, temp2});
//     }
//     const std::pair<torch::Tensor, torch::Tensor>& getTensor(const size_t& index) const {
//       if (index >= data_.size()) {
//           throw std::out_of_range("Index out of range");
//       }
//       return data_[index];
//     }
//     void fuseThroughSpatioTemporalBGKI(){
//       if(data_.size() < capacity_){
//         std::cout << "buffer size not enough, now: " << data_.size() << std::endl;
//         return;
//       }
//     }
//   private://membership function
//     /**
//      * @brief:: 
//      * @attention: start1, start2: [x, y]  shape1, shape2: [h, w]. the must be 2D type
//      * @return {*}
//      */  
//     std::tuple<torch::indexing::Slice, torch::indexing::Slice, torch::indexing::Slice, torch::indexing::Slice> 
//     getOverlapRegion2D(const torch::Tensor& start1, const torch::IntArrayRef& shape1, 
//                       const torch::Tensor& start2, const torch::IntArrayRef& shape2, double resolution) {
//       // 计算两个张量在 x 和 y 方向上的重叠区域
//       double x1_start = start1[0].item<double>();
//       double x1_end = x1_start + (shape1[0] - 1) * resolution;
//       double y1_start = start1[1].item<double>();
//       double y1_end = y1_start + (shape1[1] - 1) * resolution;

//       double x2_start = start2[0].item<double>();
//       double x2_end = x2_start + (shape2[0] - 1) * resolution;
//       double y2_start = start2[1].item<double>();
//       double y2_end = y2_start + (shape2[1] - 1) * resolution;

//       // 计算重叠区域的边界
//       double overlap_x_start = std::max(x1_start, x2_start); // 0 
//       double overlap_x_end = std::min(x1_end, x2_end);       // 0.8
//       double overlap_y_start = std::max(y1_start, y2_start); // 0 
//       double overlap_y_end = std::min(y1_end, y2_end);       // 0.8

//       // 打印重叠区域
//       // std::cout << "Overlap Region: [" << overlap_x_start << ", " << overlap_x_end << "], ["
//       //           << overlap_y_start << ", " << overlap_y_end << "]" << std::endl;

//       // 计算重叠区域在 tensor1 中的索引范围
//       int tensor1_x_start = std::round((overlap_x_start - x1_start) / resolution);
//       int tensor1_x_end = std::round((overlap_x_end - x1_start) / resolution);
//       int tensor1_y_start = std::round((overlap_y_start - y1_start) / resolution);
//       int tensor1_y_end = std::round((overlap_y_end - y1_start) / resolution);
//       // std::cout << "tensor1 indices: [" << tensor1_x_start << ", " << tensor1_x_end << "], ["
//       //           << tensor1_y_start << ", " << tensor1_y_end << "]" << std::endl;

//       // 计算重叠区域在 tensor2 中的索引范围
//       int tensor2_x_start = std::round((overlap_x_start - x2_start) / resolution);// (0 - -1) / 0.2 = 5
//       int tensor2_x_end = std::round((overlap_x_end - x2_start) / resolution);
//       int tensor2_y_start = std::round((overlap_y_start - y2_start) / resolution);
//       int tensor2_y_end = std::round((overlap_y_end - y2_start) / resolution);
//       // std::cout << "tensor2 indices: [" << tensor2_x_start << ", " << tensor2_x_end << "], ["
//       //           << tensor2_y_start << ", " << tensor2_y_end << "]" << std::endl;

//       // 返回切片对象
//       return {torch::indexing::Slice(tensor1_x_start, tensor1_x_end + 1),
//               torch::indexing::Slice(tensor1_y_start, tensor1_y_end + 1),
//               torch::indexing::Slice(tensor2_x_start, tensor2_x_end + 1),
//               torch::indexing::Slice(tensor2_y_start, tensor2_y_end + 1)};
//     }

//   public://membership variable

//   private://membership variable
//     std::deque<std::pair<torch::Tensor, torch::Tensor>> data_;
//     int capacity_;
//     torch::DeviceType device_;
// };


// torch::Tensor generateGridTensor(int height, int width, float resolution, const std::pair<float, float>& start) {
//   // 创建一个形状为 height x width x 2 的张量，初始化为 0
//   auto options = torch::TensorOptions().dtype(torch::kF32);
//   torch::Tensor grid = torch::zeros({height, width, 2}, options);

//   // 填充张量
//   for (int i = 0; i < height; ++i) {
//       for (int j = 0; j < width; ++j) {
//           grid[i][j][0] = start.first + i * resolution;
//           grid[i][j][1] = start.second + j * resolution;
//       }
//   }
//   return grid;
// }

torch::Tensor generateGridTensor(int height, int width, float resolution, const std::pair<float, float>& start) {
  // 创建一个形状为 height x width x 2 的张量，初始化为 0
  auto options = torch::TensorOptions().dtype(torch::kF32);
  torch::Tensor grid = torch::zeros({height, width, 2}, options);

  // 填充张量
  for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
          grid[i][j][0] = std::round((start.first + i * resolution) * 1000.0) / 1000.0;
          grid[i][j][1] = std::round((start.second + j * resolution) * 1000.0) / 1000.0;
      }
  }
  return grid;
}



std::tuple<torch::indexing::Slice, torch::indexing::Slice, torch::indexing::Slice, torch::indexing::Slice> 
getOverlapRegion2D(const torch::Tensor& start1, const torch::IntArrayRef& shape1, 
                   const torch::Tensor& start2, const torch::IntArrayRef& shape2, float resolution) {
    // 计算两个张量在 x 和 y 方向上的重叠区域
    float x1_start = start1[0].item<float>();
    float x1_end = x1_start + (shape1[0] - 1) * resolution;
    float y1_start = start1[1].item<float>();
    float y1_end = y1_start + (shape1[1] - 1) * resolution;

    float x2_start = start2[0].item<float>();
    float x2_end = x2_start + (shape2[0] - 1) * resolution;
    float y2_start = start2[1].item<float>();
    float y2_end = y2_start + (shape2[1] - 1) * resolution;

    // 计算重叠区域的边界
    float overlap_x_start = std::max(x1_start, x2_start); // 0 
    float overlap_x_end = std::min(x1_end, x2_end);       // 0.8
    float overlap_y_start = std::max(y1_start, y2_start); // 0 
    float overlap_y_end = std::min(y1_end, y2_end);       // 0.8

    // 打印重叠区域
    // std::cout << "Overlap Region: [" << overlap_x_start << ", " << overlap_x_end << "], ["
    //           << overlap_y_start << ", " << overlap_y_end << "]" << std::endl;

    // 计算重叠区域在 tensor1 中的索引范围
    int tensor1_x_start = std::round((overlap_x_start - x1_start) / resolution);
    int tensor1_x_end = std::round((overlap_x_end - x1_start) / resolution);
    int tensor1_y_start = std::round((overlap_y_start - y1_start) / resolution);
    int tensor1_y_end = std::round((overlap_y_end - y1_start) / resolution);
    // std::cout << "tensor1 indices: [" << tensor1_x_start << ", " << tensor1_x_end << "], ["
    //           << tensor1_y_start << ", " << tensor1_y_end << "]" << std::endl;

    // 计算重叠区域在 tensor2 中的索引范围
    int tensor2_x_start = std::round((overlap_x_start - x2_start) / resolution);// (0 - -1) / 0.2 = 5
    int tensor2_x_end = std::round((overlap_x_end - x2_start) / resolution);
    int tensor2_y_start = std::round((overlap_y_start - y2_start) / resolution);
    int tensor2_y_end = std::round((overlap_y_end - y2_start) / resolution);
    // std::cout << "tensor2 indices: [" << tensor2_x_start << ", " << tensor2_x_end << "], ["
    //           << tensor2_y_start << ", " << tensor2_y_end << "]" << std::endl;

    // 返回切片对象
    return {torch::indexing::Slice(tensor1_x_start, tensor1_x_end + 1),
            torch::indexing::Slice(tensor1_y_start, tensor1_y_end + 1),
            torch::indexing::Slice(tensor2_x_start, tensor2_x_end + 1),
            torch::indexing::Slice(tensor2_y_start, tensor2_y_end + 1)};
}


void testLocalTensorBuffer(){
  float resolution = 1.0;
  auto target_tensor = generateGridTensor(10, 10, resolution, {0.0, 0.0});
  auto target_value = torch::ones_like(target_tensor) * 1.0;
  // std::cout << "target_tensor: \n" << target_tensor << std::endl;
  // std::cout << "target_value: \n" << target_value << std::endl;

  std::vector<std::pair<double, double>> starts = {{-5, -5}, {-5, 5}, {5, -5}, {5, 5}};
  std::vector<torch::Tensor> tensor_pos_vec;
  for (const auto& start : starts) {
    auto tensor_pos = generateGridTensor(10, 10, resolution, start);
    tensor_pos_vec.push_back(tensor_pos);
    // std::cout << "tensor_pos: \n" << tensor_pos << std::endl;
  }

  auto sizes = target_tensor.sizes();
  torch::Tensor tensor_all_cat = zeros_like(target_tensor).unsqueeze(0).expand({(long)tensor_pos_vec.size(), -1, -1, -1});
  std::cout << "tensor_all_cat.sizes(): " << tensor_all_cat.sizes() << std::endl;
  
  std::vector<torch::Tensor> tensor_cats;
  
  for(const auto& tensor_pos : tensor_pos_vec){
    static int count = -1;
    count++;
    std::cout << "==========tensor_pos new frame" << std::endl;
    //! overlap region extraction
    auto [tensor1_slicex, tensor1_slicey, tensor2_slicex, tensor2_slicey] = getOverlapRegion2D(target_tensor[0][0], target_tensor.sizes(), tensor_pos[0][0], tensor_pos.sizes(), resolution);
    auto tensor1_overlap = target_tensor.index({tensor1_slicex, tensor1_slicey});
    auto tensor2_overlap = tensor_pos.index({tensor2_slicex, tensor2_slicey});      
    auto temp_tensor = torch::zeros_like(target_tensor);
    temp_tensor.index_put_({tensor1_slicex, tensor1_slicey, torch::indexing::Slice()}, tensor2_overlap);
    tensor_cats.push_back(temp_tensor.unsqueeze(0));
  }
  std::cout << "tensor_cats.size(): " << tensor_cats.size() << std::endl;
  auto fally_tensor = torch::cat(tensor_cats, 0);
  std::cout << "fally_tensor.sizes(): " << fally_tensor.sizes() << std::endl;
  std::cout << (fally_tensor[0] - tensor_cats[0]).abs().sum() << std::endl;
  std::cout << (fally_tensor[1] - tensor_cats[1]).abs().sum() << std::endl;
  std::cout << (fally_tensor[2] - tensor_cats[2]).abs().sum() << std::endl;
  std::cout << (fally_tensor[3] - tensor_cats[3]).abs().sum() << std::endl;
}


//////////////////////////////////////////////PART: 24 LocalTensorBuffer //////////////////////////////////////
float res_yaw = 0.2;
float res_xy = 0.2;

class LocalTensorBuffer{
  private:
    struct MapInfo{
      torch::Tensor se2Info;
      torch::Tensor gridPos;
      float timestamp;
    };
  public://membership function
    LocalTensorBuffer(const int& capacity, const torch::Tensor& yawTensor): capacity_(capacity), yawTensor_(yawTensor){
      device_ = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
      dtype_ = torch::kFloat32;
    }
    size_t size() const {
      return data_.size();
    }
    bool empty() const {
      return data_.empty();
    }
    /**
     * @brief:: 
     * @param {Tensor&} se2Info: 
     * @param {Tensor&} gridPos: 
     * @attention: 
     * @return {*}
     */    
    void insert(const torch::Tensor& se2Info, const torch::Tensor& gridPos, const float& timestamp){
      torch::Tensor tensor_fused, cov_fused;
      auto temp1 = se2Info;
      auto temp2 = gridPos;
      temp1 = temp1.to(device_);
      temp2 = temp2.to(device_);
      if (data_.size() >= capacity_) {
        //TODO: fuse the new data with the old data by using STBGKI regression
        std::cout << "🐶 fuse the new data with the old data by using STBGKI regression" << std::endl;
        auto [tensor_fused, cov_fused]  = fuseThroughSpatioTemporalBGKI(temp1, temp2, timestamp);
        data_.pop_front();
      }else{
        std::cout << "🧋 buffer size not enough, now: " << data_.size() << std::endl;
        // std::cout << "INPUT: se2info.sizes(): " << se2Info.sizes() << " gridPos.sizes():" << gridPos.sizes() << std::endl;
        // std::cout << "INPUT: se2info.abs().sum(): \n" << se2Info.abs().sum() << std::endl;
        // std::cout << "INPUT: gridPos[0][0]: \n" << gridPos[0][0] << std::endl;
        // std::cout << "INPUT: gridPos[36][35]: \n" << gridPos[36][35] << std::endl;
        // std::cout << "timestamp: " << timestamp << std::endl;
        tensor_fused = temp1;
      }
      data_.push_back({tensor_fused, temp2, timestamp});
    }

    std::tuple<torch::Tensor, torch::Tensor> fuseThroughSpatioTemporalBGKI(const torch::Tensor& new_se2Info, const torch::Tensor& new_gridPos, const float& new_timestamp){
      if(data_.size() < capacity_){
        std::cout << "buffer size not enough, now: " << data_.size() << std::endl;
        return {torch::Tensor(), torch::Tensor()};
      }

      //! parameters
      int windows_dimyaw = 5;
      int windows_dimxy = 9;
      int64_t pad_dimxy = (windows_dimxy - 1)/2;
      int64_t pad_dimyaw = (windows_dimyaw - 1)/2;
      torch::Tensor offset = torch::ones({2}).to(dtype_).to(device_) * res_xy * pad_dimxy;
      // std::cout << "!!!!! offset: " << offset << std::endl;
      
      auto options_pad_se2Info = torch::nn::functional::PadFuncOptions({0, 0, pad_dimxy, pad_dimxy, pad_dimxy, pad_dimxy, pad_dimyaw, pad_dimyaw})
        .mode(torch::kConstant)  // 使用常数填充模式
        .value(0);               // 填充值为0
      // std::cout << "!!!!!!!!!!" << std::endl;
      //! extract the overlap region se2TimeY([]) and se2TimeX([])
      std::vector<torch::Tensor> se2TimeY, se2TimeX;
      auto new_se2Info_padded = torch::nn::functional::pad(new_se2Info, options_pad_se2Info);
      // Extract the right pad_dimyaw region from the original tensor
      auto padded_size_dim0 = new_se2Info_padded.size(0);
      torch::Tensor yaw_right_region = new_se2Info_padded.slice(0, padded_size_dim0 - 2*pad_dimyaw, padded_size_dim0 - pad_dimyaw);//0:31:33
      torch::Tensor yaw_left_region = new_se2Info_padded.slice(0, pad_dimyaw, 2*pad_dimyaw);//0:2:3
      new_se2Info_padded.slice(0, 0, pad_dimyaw).copy_(yaw_right_region);
      new_se2Info_padded.slice(0, padded_size_dim0 - pad_dimyaw, padded_size_dim0).copy_(yaw_left_region);
      se2TimeY.push_back(new_se2Info_padded.unsqueeze(0));

      //NOTE: for-each history data
      std::cout << "\n----------------------------------------" << "for-each history data" << "----------------------------------------" << std::endl;
      for(auto data : data_){
        static int count = -1;
        count++;
        std::cout << "==========count index: " << count << std::endl;
        //! overlap region extraction
        std::pair<int, int> shape1 = {new_gridPos.size(0) + 2 * pad_dimxy, new_gridPos.size(1) + 2 * pad_dimxy};
        // std::pair<int, int> shape2 = {data.gridPos.size(0) + 2 * pad_dimxy, data.gridPos.size(1) + 2 * pad_dimxy};//DEBUG:
        std::pair<int, int> shape2 = {data.gridPos.size(0), data.gridPos.size(1)};//
        // std::cout << "shape1: " << shape1 << std::endl;
        // std::cout << "shape2: " << shape2 << std::endl;
        // std::cout << "new_gridPos[0][0]: \n" << new_gridPos[0][0] << std::endl;
        // std::cout << "new_gridPos[0][0] - offset: \n" << new_gridPos[0][0] - offset << std::endl;
        // std::cout << "data.gridPos[0][0]: \n" << data.gridPos[0][0] << std::endl;
        // std::cout << "data.gridPos[0][0] - offset: \n" << data.gridPos[0][0] - offset << std::endl;
        auto [new_sx, new_sy, old_sx, old_sy] = 
          // getOverlapRegion2D(new_gridPos[0][0] - offset, shape1, data.gridPos[0][0] - offset, shape2, res_xy);//DEBUG:1
          //这里只是将最新的数据进行padding扩大范围，然后历史数据就不padding了，这样对标最新数据即可
          getOverlapRegion2D(new_gridPos[0][0] - offset, shape1, data.gridPos[0][0], shape2, res_xy);
          // getOverlapRegion2D(new_gridPos[0][0], new_gridPos.sizes(), data.gridPos[0][0], data.gridPos.sizes(), res_xy);
        // std::cout << "!!!!!overlap over!!!!!!!!!" << std::endl;
        // std::cout << "new_sx: " << new_sx << std::endl;
        // std::cout << "new_sy: " << new_sy << std::endl;
        // std::cout << "old_sx: " << old_sx << std::endl;
        // std::cout << "old_sy: " << old_sy << std::endl;
        // std::cout << "data.gridPos[15][14]: " << data.gridPos[15][14] << std::endl;
        // std::cout << "data.gridPos[36][35]: " << data.gridPos[36][35] << std::endl;
        auto overlap = data.gridPos.index({old_sx,old_sy, torch::indexing::Slice()});
        std::cout << "overlap[0][0]: \n" << overlap[0][0] << std::endl;
        std::cout << "overlap[overlap.size(0)-1][overlap.size(1)-1]: \n" << overlap[overlap.size(0)-1][overlap.size(1)-1] << std::endl;
      

        //! se2TimeY
        auto se2TimeY_temp = torch::zeros_like(new_se2Info_padded);//zeros!! [35, 45, 44, 4]
        std::cout << "se2TimeY_temp.sizes(): " << se2TimeY_temp.sizes() << std::endl;
        auto overlap_extracted = data.se2Info.index({torch::indexing::Slice(), old_sx,old_sy, torch::indexing::Slice()});//[31, 23, 22, 4]
        std::cout << "overlap_extracted.sizes(): " << overlap_extracted.sizes() << std::endl;
        auto options_pad_yaw = torch::nn::functional::PadFuncOptions({0, 0, 0, 0, 0, 0, pad_dimyaw, pad_dimyaw})
          .mode(torch::kConstant)  // 使用常数填充模式
          .value(0);               // 填充值为0
        auto overlap_extracted_padded = torch::nn::functional::pad(overlap_extracted, options_pad_yaw);
        std::cout << "overlap_extracted_padded.sizes(): " << overlap_extracted_padded.sizes() << std::endl;
        
        auto padded_size_dim0 = overlap_extracted_padded.size(0);
        torch::Tensor yaw_right_region = overlap_extracted_padded.slice(0, padded_size_dim0 - 2*pad_dimyaw, padded_size_dim0 - pad_dimyaw);//0-31:33
        torch::Tensor yaw_left_region = overlap_extracted_padded.slice(0, pad_dimyaw, 2*pad_dimyaw);//0-2:3
        overlap_extracted_padded.slice(0, 0, pad_dimyaw).copy_(yaw_right_region);
        overlap_extracted_padded.slice(0, padded_size_dim0 - pad_dimyaw, padded_size_dim0).copy_(yaw_left_region);
        auto temp_print = overlap_extracted_padded.slice(0, 0, 2) - overlap_extracted.slice(0, 29, 31);
        std::cout << "temp_print.abs().sum(): " << temp_print.abs().sum() << std::endl;
        se2TimeY_temp.index_put_({torch::indexing::Slice(), new_sx,new_sy, torch::indexing::Slice()}, overlap_extracted_padded);

        std::cout << "se2TimeY_temp.abs().sum(): " << se2TimeY_temp.abs().sum() << std::endl;
        std::cout << "overlap_extracted_padded.abs().sum(): " << overlap_extracted_padded.abs().sum() << std::endl;
        se2TimeY.push_back(se2TimeY_temp.unsqueeze(0));
        
        // //! se2TimeX  
        // auto yaw_expand = yawTensor_.unsqueeze(1).unsqueeze(1).expand({-1, new_gridPos.size(0), new_gridPos.size(1), -1}).to(dtype_).to(device_);//31x37x36x1
        // auto gridPos_expand = new_gridPos.unsqueeze(0).expand({yawTensor_.size(0), -1, -1, -1}).to(dtype_).to(device_);//31x37x36x2
        // auto timestamp_expand = torch::ones({yawTensor_.size(0), new_gridPos.size(0), new_gridPos.size(1), 1}).to(dtype_).to(device_) * new_timestamp;//31x37x36x1
        // auto se2TimeX_temp = torch::cat({yaw_expand, gridPos_expand, timestamp_expand}, 3).to(dtype_).to(device_);//31x37x36x4
        // std::cout << "se2TimeX_temp.sizes(): " << se2TimeX_temp.sizes() << std::endl;
        // se2TimeX.push_back(se2TimeX_temp.unsqueeze(0));
      }
      std::cout << "\n----------------------------------------" << "Begin to cat data" << "----------------------------------------" << std::endl;
      //! cat data  
      auto se2TimeY_cated = torch::cat(se2TimeY, 0);//3x31x37x36x4, where 3 is the timestamp frame
      // auto se2TimeX_cated = torch::cat(se2TimeX, 0);//3x31x37x36x4
      // std::cout << "se2TimeY_cated.sizes(): " << se2TimeY_cated.sizes() << std::endl;
      // std::cout << "se2TimeX_cated.sizes(): " << se2TimeX_cated.sizes() << std::endl;

      //！ unfold
      // auto se2TimeX_unfolded = se2TimeX_cated.unfold(1, windows_dimyaw, 1).unfold(2, windows_dimxy, 1).unfold(3, windows_dimxy, 1);//[3, 27, 29, 28, 4, 5, 9, 9]
      // std::cout << "se2TimeX_unfolded.sizes(): " << se2TimeX_unfolded.sizes() << std::endl;
      auto se2TimeY_unfolded = se2TimeY_cated.unfold(1, windows_dimyaw, 1).unfold(2, windows_dimxy, 1).unfold(3, windows_dimxy, 1);//[3, 27, 29, 28, 4, 5, 9, 9]
      // std::cout << "se2TimeY_unfolded.sizes(): " << se2TimeY_unfolded.sizes() << std::endl;

      //! permute 
      // auto se2TimeX_unfolded_permute = se2TimeX_unfolded.permute({1, 2, 3, 0, 5, 6, 7, 4});
      // std::cout << "se2TimeX_unfolded_permute.sizes(): " << se2TimeX_unfolded_permute.sizes() << std::endl;//[27, 29, 28, 3, 5, 9, 9, 4]
      auto se2TimeY_unfolded_permute = se2TimeY_unfolded.permute({1, 2, 3, 0, 5, 6, 7, 4});
      // std::cout << "se2TimeY_unfolded_permute.sizes(): " << se2TimeY_unfolded_permute.sizes() << std::endl;//[27, 29, 28, 3, 5, 9, 9, 4]

      //! reshape 
      // auto shapeX = se2TimeX_unfolded_permute.sizes();
      // auto se2TimeX_unfolded_permute_reshaped = se2TimeX_unfolded_permute.reshape({shapeX[0], shapeX[1], shapeX[2], -1, shapeX[7]});//[27, 29, 28, 1215, 4]
      // std::cout << "se2TimeX_unfolded_permute_reshaped.sizes(): " << se2TimeX_unfolded_permute_reshaped.sizes() << std::endl;
      auto shapeY = se2TimeY_unfolded_permute.sizes();
      auto se2TimeY_unfolded_permute_reshaped = se2TimeY_unfolded_permute.reshape({shapeY[0], shapeY[1], shapeY[2], -1, shapeY[7]});//[27, 29, 28, 1215, 4]
      std::cout << "se2TimeY_unfolded_permute_reshaped.sizes(): " << se2TimeY_unfolded_permute_reshaped.sizes() << std::endl;
      
      
      return {torch::Tensor(), torch::Tensor()};
    }
  private://membership function
    /**
     * @brief:: 
     * @attention: start1, start2: [x, y]  shape1, shape2: [h, w]. the must be 2D type
     * @return {*}
     */  
    // std::tuple<torch::indexing::Slice, torch::indexing::Slice, torch::indexing::Slice, torch::indexing::Slice> 
    // getOverlapRegion2D(const torch::Tensor& start1, const torch::IntArrayRef& shape1, 
    //                   const torch::Tensor& start2, const torch::IntArrayRef& shape2, double resolution) {
    //   // 计算两个张量在 x 和 y 方向上的重叠区域
    //   double x1_start = start1[0].item<double>();
    //   double x1_end = x1_start + (shape1[0] - 1) * resolution;
    //   double y1_start = start1[1].item<double>();
    //   double y1_end = y1_start + (shape1[1] - 1) * resolution;

  
    std::tuple<torch::indexing::Slice, torch::indexing::Slice, torch::indexing::Slice, torch::indexing::Slice> 
    getOverlapRegion2D(const torch::Tensor& start1, const std::pair<int, int>& shape1, 
                      const torch::Tensor& start2, const std::pair<int, int>& shape2, double resolution) {
      // 计算两个张量在 x 和 y 方向上的重叠区域
      double x1_start = start1[0].item<double>();
      double x1_end = x1_start + (shape1.first - 1) * resolution;
      double y1_start = start1[1].item<double>();
      double y1_end = y1_start + (shape1.second - 1) * resolution;
      double x2_start = start2[0].item<double>();
      double x2_end = x2_start + (shape2.first - 1) * resolution;
      double y2_start = start2[1].item<double>();
      double y2_end = y2_start + (shape2.second - 1) * resolution;

      // 计算重叠区域的边界
      double overlap_x_start = std::max(x1_start, x2_start); // 0 
      double overlap_x_end = std::min(x1_end, x2_end);       // 0.8
      double overlap_y_start = std::max(y1_start, y2_start); // 0 
      double overlap_y_end = std::min(y1_end, y2_end);       // 0.8

      // 打印重叠区域
      std::cout << "Overlap Region[x_start, x_end], [y_start, y_end]: [" << overlap_x_start << ", " << overlap_x_end << "], ["
                << overlap_y_start << ", " << overlap_y_end << "]" << std::endl;

      // 计算重叠区域在 tensor1 中的索引范围
      int tensor1_x_start = std::round((overlap_x_start - x1_start) / resolution);
      int tensor1_x_end = std::round((overlap_x_end - x1_start) / resolution);
      int tensor1_y_start = std::round((overlap_y_start - y1_start) / resolution);
      int tensor1_y_end = std::round((overlap_y_end - y1_start) / resolution);
      // std::cout << "tensor1 indices: [" << tensor1_x_start << ", " << tensor1_x_end << "], ["
      //           << tensor1_y_start << ", " << tensor1_y_end << "]" << std::endl;

      // 计算重叠区域在 tensor2 中的索引范围
      int tensor2_x_start = std::round((overlap_x_start - x2_start) / resolution);// (0 - -1) / 0.2 = 5
      int tensor2_x_end = std::round((overlap_x_end - x2_start) / resolution);
      int tensor2_y_start = std::round((overlap_y_start - y2_start) / resolution);
      int tensor2_y_end = std::round((overlap_y_end - y2_start) / resolution);
      // std::cout << "tensor2 indices: [" << tensor2_x_start << ", " << tensor2_x_end << "], ["
      //           << tensor2_y_start << ", " << tensor2_y_end << "]" << std::endl;

      // 返回切片对象
      return {torch::indexing::Slice(tensor1_x_start, tensor1_x_end + 1),
              torch::indexing::Slice(tensor1_y_start, tensor1_y_end + 1),
              torch::indexing::Slice(tensor2_x_start, tensor2_x_end + 1),
              torch::indexing::Slice(tensor2_y_start, tensor2_y_end + 1)};
    }

  public://membership variable

  private://membership variable
    int capacity_;
    torch::Tensor yawTensor_;  
    torch::DeviceType device_;
    torch::Dtype dtype_;
    std::deque<MapInfo> data_;
};

void testLocalTensorBufferFuseThroughSTBGKI(){
  auto device_ = torch::kCUDA;
  auto dtype_ = torch::kFloat32;

  auto yaw_31x1 = (torch::arange(-15, 16).to(dtype_) * res_yaw).unsqueeze(1);
  std::cout << "yaw_31x1.sizes(): " << yaw_31x1.sizes() << std::endl;
  // std::cout << "yaw_31x1: \n" << yaw_31x1 << std::endl; // [-3.0 ~ +3.0]

  std::vector<std::pair<double, double>> starts = {{-3.7, -3.6}, {-3.7, 3.6}, {3.7, -3.6}, {0.1, 0.0}};
  std::vector<float> timestamps = {0.0, 1.0, 2.0, 3.0};

  LocalTensorBuffer buffer(3, yaw_31x1);


  for(auto & start : starts){
    static int count = -1;
    count++;
    auto gridPos = generateGridTensor(37, 36, res_xy, start); // 31 x 37 x 36 x 2
    std::cout << "gridPos[0][0]: \n" << gridPos[0][0] << std::endl;
    std::cout << "gridPos[36][35]: \n" << gridPos[36][35] << std::endl;
    
    auto se2Info = count * torch::ones({31, 37, 36, 4}).to(device_).to(dtype_);// 31 x 37 x 36 x 4, nx ny nz trav  
    buffer.insert(se2Info, gridPos, timestamps[count]);

    // std::cout << "==========se2Info new frame" << std::endl;
    // std::cout << "INPUT: se2info.sizes(): " << se2Info.sizes() << " gridPos.sizes():" << gridPos.sizes() << std::endl; 
    // std::cout << "INPUT: se2info.abs().sum(): " << se2Info.abs().sum() << std::endl;
    // std::cout << "INPUT: gridPos.abs().sum(): " << gridPos.abs().sum() << std::endl;
  }



  /********* test */
  std::cout << "\n=========== TEST REGION ============" << std::endl;
  
  //! origin tensor
  auto small_tensor = torch::rand({31, 37, 36, 4}).to(device_).to(dtype_);

  //! padding options
  int windows_dimyaw = 5;
  int windows_dimxy = 9;
  int64_t pad_dimxy = (windows_dimxy - 1)/2;
  int64_t pad_dimyaw = (windows_dimyaw - 1)/2;
  auto options = torch::nn::functional::PadFuncOptions({0, 0, pad_dimxy, pad_dimxy, pad_dimxy, pad_dimxy, pad_dimyaw, pad_dimyaw})
    .mode(torch::kConstant)  // 使用常数填充模式
    .value(0);               // 填充值为0

  //! padding
  torch::Tensor small_tensor_padded = torch::nn::functional::pad(small_tensor, options);
  std::cout << "small_tensor_padded.sizes(): " << small_tensor_padded.sizes() << std::endl;
  /*
  dim0 31 -> orig0 + 2 * dim0 | 31 + 2 * 2 = 35 
  dim1 37 -> orig1 + 2 * dim1 | 37 + 2 * 4 = 45
  dim2 36 -> orig2 + 2 * dim2 | 36 + 2 * 4 = 44
  dim3 04 -> orig3 + 2 * dim3 | 04 + 2 * 0 = 04
  */

  //! yaw loop setting  
  // Extract the right pad_dimyaw region from the original tensor and reverse it
  auto padded_size_dim0 = small_tensor_padded.size(0);
  torch::Tensor yaw_right_region = small_tensor_padded.slice(0, padded_size_dim0 - 2*pad_dimyaw, padded_size_dim0 - pad_dimyaw);//0:31:33
  torch::Tensor yaw_left_region = small_tensor_padded.slice(0, pad_dimyaw, 2*pad_dimyaw);//0:2:3
  std::cout << "right_region.sizes(): " << yaw_right_region.sizes() << std::endl;
  std::cout << "left_region.sizes(): " << yaw_left_region.sizes() << std::endl;
  small_tensor_padded.slice(0, 0, pad_dimyaw).copy_(yaw_right_region);
  small_tensor_padded.slice(0, padded_size_dim0 - pad_dimyaw, padded_size_dim0).copy_(yaw_left_region);



  //! debug OK OK
  auto dim_test = 1;
  auto test1 = small_tensor_padded[dim_test].slice(0, pad_dimxy, small_tensor_padded[dim_test].size(0) - pad_dimxy).slice(1, pad_dimxy, small_tensor_padded[dim_test].size(1) - pad_dimxy);
  std::cout << "test1.sizes(): " << test1.sizes() << std::endl;
  auto test2 = small_tensor[small_tensor.size(0)-pad_dimyaw+dim_test];;
  std::cout << "test2.sizes(): " << test2.sizes() << std::endl;
  std::cout << "test1 - test2: " << (test1 - test2).abs().sum() << std::endl;


  

  




  

  
}





int main() {
  //! PART: 0 test
  std::cout << "Libtorch version:  " << TORCH_VERSION << std::endl;

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
  // compareCPUAndGPUTest();

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


  //! PART: 6 create 10x10 matrix3d  
  std::cout << "\n==========create 10x10 matrix3d==========\n";
  create10X10Cov3d();

  //! PART: 7 more useful functions  
  std::cout << "\n==========PART: 7 more useful functions==========\n";
  moreUsefulFuncsTest();

  //! PART: 8 torch::detach
  std::cout << "\n==========torch::detach==========\n";
  torchDetachTest();

  
  //! PART: 9 [x, y, theta, exp(-t)] 's custom distance 
  std::cout << "\n==========[x, y, theta, exp(-t)]'s custom distance==========\n";
  testCustomDistXYThetaT();

  //! PART: 10 auto grad tensor gpu version
  testAutoGradGPUTensor();

  //! PART: 11 create kernelLen tensor
  auto klen_tensor = generate_arithmetic_sequences({0.1, 0.2, 0.3}, {1, 1, 1}, 10);
  std::cout << "klen_tensor.sizes()" << klen_tensor.sizes() << std::endl;
  std::cout << "klen_tensor: \n" << klen_tensor << std::endl;

  //! PART: 12 test unorder_map<int, torch::Tensor>
  std::unordered_map<int, torch::Tensor> tensor_map;
  for (int i = 0; i < 3; ++i) {
    tensor_map[i] = torch::rand({3, 3}).to(device);
  }
  std::cout << "tensor_map[0].sizes()" << tensor_map[0].sizes() << std::endl;
  std::cout << "tensor_map[0]: \n" << tensor_map[0] << std::endl;
  std::cout << "tensor_map[1].sizes()" << tensor_map[1].sizes() << std::endl;
  std::cout << "tensor_map[1]: \n" << tensor_map[1] << std::endl;

  //! PART: 13 change dim  
  std::cout << "\n==========PART13: change dim==========\n";
  meanVec2TensorAndChangeDim();


  //! PART: 14 unsqueeze and view
  std::cout << "\n==========PART14: unsqueeze and view==========\n";
  unsqueezeTest();

  //! PART: 15 stack tensor
  std::cout << "\n==========PART15: stack tensor==========\n";
  stackTensors();

  //! PART: 16 batch eigen-Decomposition
  std::cout << "\n==========PART16: batch eigen-Decomposition==========\n";
  batchMatrixEigenDecomposition();
  // batchMatrixEigenDecomposition2();

  //！ PART: 17 batch select min eval and correspending-evec
  std::cout << "\n==========PART17: batch select min eval and correspending-evec==========\n";
  selectMinEigenvalAndEigenVec();


  //! PART: 18 mask tensor
  std::cout << "\n==========PART18: mask tensor==========\n";
  maskTensorTest();

  
  // torch::Tensor tensor1_part18 = torch::rand({41292, 4}).to(device);
  // torch::Tensor tensor2_part18 = torch::rand({41292, 4}).to(device);
  // torch::Tensor tensor12_cdist = torch::cdist(tensor1_part18, tensor2_part18, 2);
  // std::cout << "tensor12_cdist.size(): " << tensor12_cdist.sizes() << std::endl;

  //! PART: 19 tensor padding
  std::cout << "\n==========PART19: tensor padding==========\n";
  tensorPadding();

  //! PART: 20 se2 Tensor unfold
  std::cout << "\n==========PART20: se2 Tensor unfold==========\n";
  se2TensorUnfold();

  
  // {
  //   // 创建 tensor1 和 tensor2
  //   torch::Tensor tensor1 = torch::arange(0, 24).reshape({2, 3, 4});
  //   std::cout << "tensor1: \n" << tensor1 << std::endl;

  //   torch::Tensor tensor2 = torch::arange(1000, 1004).reshape({1, 4});
  //   std::cout << "tensor2: \n" << tensor2 << std::endl;

  //   // 创建一个全零张量，形状为 [2, 1, 4]（假设我们要填充到第二维度的第4个位置）
  //   torch::Tensor zero_tensor = torch::zeros({2, 1, 4}, tensor1.options());
  //   // 将 tensor2 的数据填充到全零张量的指定位置（例如第二维度的第4个位置）
  //   zero_tensor.index_put_({torch::indexing::Slice(), 3, torch::indexing::Slice()}, tensor2);

  //   // 将填充好的全零张量与 tensor1 拼接
  //   torch::Tensor result = torch::cat({tensor1, zero_tensor.unsqueeze(1)}, 1);
  //   std::cout << "Resulting tensor: \n" << result << std::endl;

  // }
  
  //! PART: 21 tensor slice and assign
  std::cout << "\n==========PART21: tensor slice and assign==========\n";
  tensorSliceAndAssign();

  //! PART: 22 FixedTensorBuffer
  std::cout << "\n==========PART22: FixedTensorBuffer==========\n";
  testFixedTensorBuffer();

  //! PART: 23 LocalTensorBuffer
  std::cout << "\n==========PART23: LocalTensorBuffer==========\n";
  testLocalTensorBuffer();

  //! PART: 24 LocalTensorBuffer fuse through STBGKI
  std::cout << "\n==========PART24: LocalTensorBuffer fuse through STBGKI==========1\n";
  testLocalTensorBufferFuseThroughSTBGKI();

  
  return 0;
}