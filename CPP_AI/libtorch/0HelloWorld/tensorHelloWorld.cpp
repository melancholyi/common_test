/*
 * @Author: chasey && melancholycy@gmail.com
 * @Date: 2025-03-22 06:41:27
 * @LastEditTime: 2025-04-19 02:57:55
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
  // Create a tensor of shape [5, 43, 42, 3, 3] representing covariance matrices
  torch::Tensor covs = torch::randn({5, 43, 42, 3, 3});
  covs = torch::matmul(covs.transpose(-1, -2), covs); // Ensure the matrices are symmetric

  // Flatten the first three dimensions to create a batch of matrices
  torch::Tensor covs_flat = covs.view({5 * 43 * 42, 3, 3});

  // Perform batch eigenvalue decomposition
  auto [eigenvalues_flat, eigenvectors_flat] = torch::linalg::eig(covs_flat);

  // Reshape the results back to the original shape
  torch::Tensor eigenvalues = torch::real(eigenvalues_flat).view({5, 43, 42, 3});
  torch::Tensor eigenvectors = torch::real(eigenvectors_flat).view({5, 43, 42, 3, 3});

  // For comparison, perform the decomposition using nested loops
  torch::Tensor eigenvalues_loop = torch::empty({5, 43, 42, 3});
  torch::Tensor eigenvectors_loop = torch::empty({5, 43, 42, 3, 3});
  for (int i = 0; i < 5; ++i) {
      for (int j = 0; j < 43; ++j) {
          for (int k = 0; k < 42; ++k) {
              auto [eigvals, eigvecs] = torch::linalg::eig(covs[i][j][k]);
              eigenvalues_loop[i][j][k] = torch::real(eigvals);
              eigenvectors_loop[i][j][k] = torch::real(eigvecs);
          }
      }
  }

  // Compare the results from the two methods
  auto eigenvalues_diff = (eigenvalues - eigenvalues_loop).abs().max().item<float>();
  auto eigenvectors_diff = (eigenvectors - eigenvectors_loop).abs().max().item<float>();
  std::cout << "Maximum difference in eigenvalues: " << eigenvalues_diff << std::endl;
  std::cout << "Maximum difference in eigenvectors: " << eigenvectors_diff << std::endl;
  

  // Print the data at the specified location
  std::cout << "eigenvalues[2][6][15]:" << eigenvalues[2][6][15].unsqueeze(0) << std::endl;
  std::cout << "eigenvectors[2][6][15]:\n" << eigenvectors[2][6][15] << std::endl;
  std::cout << "eigenvalues_loop[2][6][15]:" << eigenvalues_loop[2][6][15].unsqueeze(0) << std::endl;
  std::cout << "eigenvectors_loop[2][6][15]:\n" << eigenvectors_loop[2][6][15] << std::endl;
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



  return 0;
}