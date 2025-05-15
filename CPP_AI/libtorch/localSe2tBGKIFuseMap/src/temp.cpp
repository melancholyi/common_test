/*
 * @Author: chasey && melancholycy@gmail.com
 * @Date: 2025-05-14 12:30:39
 * @LastEditTime: 2025-05-14 12:44:12
 * @FilePath: /test/CPP_AI/libtorch/localSe2tBGKIFuseMap/src/temp.cpp
 * @Description: 
 * @Reference: 
 * Copyright (c) 2025 by chasey && melancholycy@gmail.com, All Rights Reserved. 
 */
void negLogMLLOptimHyperParams(){
  auto optimizer_adam = torch::optim::Adam(std::vector<torch::Tensor>({kernelLenTensor_}), torch::optim::AdamOptions(lrHyper_));

  bool is_traincov = isTrainCov_;
  if(!isTrainCov_){
    isTrainCov_ = true;
    trainCov_ = torch::ones({trainY_.size(0), trainY_.size(1)}, dtype_).to(device_)*1e10;
  }

  auto closure = [&]() -> torch::Tensor{
    //loss zero grad  
    optimizer_adam.zero_grad();
    torch::Tensor dist_mat, kernel;
    torch::Tensor yTensor, kTensor;
    torch::Tensor y_ones = torch::ones({trainY_.size(0), trainY_.size(1)}, dtype_).to(device_); // 优化：只创建一列

    // 计算距离矩阵和核函数
    computeDistmatAndKernelFunc_(trainX_, trainX_, kernelLenTensor_, dist_mat, kernel);

    // 提前计算对角线元素
    auto kernel_diag = kernel.diag();

    // 合并计算 yTensor 和 kTensor
    auto numerator = kernel.matmul(trainY_) - kernel_diag * trainY_;
    auto denominator = kernel.matmul(y_ones) - kernel_diag;
    kTensor = 1.0 / (denominator + delta_);
    yTensor = numerator * kTensor; 

    auto loss = 0.5 * ((kTensor + trainCov_).log() + (yTensor - trainY_).pow(2)/(kTensor + trainCov_)).sum(); // [1]

    loss.backward();

    return loss;                                                                                                        
  };

  for(int i = 0; i < optimEpochs_; ++i){
    auto loss = closure();
    optimizer_adam.step();
  }
  isTrainCov_ = is_traincov;
}