/*
 * @Author: chasey && melancholycy@gmail.com
 * @Date: 2025-03-22 06:41:27
 * @LastEditTime: 2025-03-22 06:49:27
 * @FilePath: /test/CPP_THIRD_PARTYs/libtorch/0HelloWorld/main.cpp
 * @Description: 
 * @Reference: 
 * Copyright (c) 2025 by chasey && melancholycy@gmail.com, All Rights Reserved. 
 */
#include <iostream>
#include <torch/torch.h>

int main() {
  // 创建一个(2,3)张量
  torch::Tensor tensor = torch::zeros({2, 3});
  std::cout << tensor << std::endl;
  std::cout << "\nWelcome to LibTorch!" << std::endl;

  return 0;
}