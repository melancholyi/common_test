/*
 * @Author: chasey && melancholycy@gmail.com
 * @Date: 2025-05-11 10:35:08
 * @LastEditTime: 2025-05-11 13:24:34
 * @FilePath: /test/CPP_AI/libtorch/localSe2tBGKIFuseMap/src/main.cpp
 * @Description: 
 * @Reference: 
 * Copyright (c) 2025 by chasey && melancholycy@gmail.com, All Rights Reserved. 
 */

#include "local_se2t_bgkimap.hpp"
// float res_yaw = 0.2;
// float res_xy = 0.2;

void testLocalTensorBufferFuseThroughSTBGKI(){
  auto device_ = torch::kCUDA;
  auto dtype_ = torch::kFloat32;

  auto yaw_31x1 = (torch::arange(-15, 16).to(dtype_) * res_yaw).unsqueeze(1);
  std::cout << "yaw_31x1.sizes(): " << yaw_31x1.sizes() << std::endl;
  // std::cout << "yaw_31x1: \n" << yaw_31x1 << std::endl; // [-3.0 ~ +3.0]

  std::vector<std::pair<double, double>> starts = {{-3.7, -3.6}, {-3.7, 3.6}, {3.7, -3.6}, {0.1, 0.0}};
  std::vector<float> timestamps = {0.0, 1.0, 2.0, 3.0};

  LocalTensorBuffer buffer(3, yaw_31x1, 0.2, 0.2, {1.0, 0.4, 0.8});


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
}


int main(){
    testLocalTensorBufferFuseThroughSTBGKI();



    return 0;
}