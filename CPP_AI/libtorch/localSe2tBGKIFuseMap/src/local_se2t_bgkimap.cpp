#include "local_se2t_bgkimap.hpp"

void LocalTensorBuffer::printCudaMemoryInfo(const char* step) {
  size_t freeBytes, totalBytes;
  cudaError_t err = cudaMemGetInfo(&freeBytes, &totalBytes);
  if (err != cudaSuccess) {
      std::cerr << "cudaMemGetInfo failed: " << cudaGetErrorString(err) << std::endl;
      return;
  }
  float freeGB = freeBytes / 1073741824.0;
  float totalGB = totalBytes / 1073741824.0;
  float usedGB = totalGB - freeGB;
  std::cout << std::endl << step << " - CUDA Memory Usage: ![" << usedGB << "]! GB used / ![" << freeGB << "]! GB free / ![" << totalGB << "]! GB total" << std::endl;
}

void LocalTensorBuffer::updatePadDimAndUnfoldWindows(){
    padDimYaw_ = std::round(kLenTimeYawGrid_[1].item<float>() / resYaw_); 
    padDimXY_ = std::round(kLenTimeYawGrid_[2].item<float>() / resGrid_);
    windowsDimYaw_ = padDimYaw_ * 2 + 1;
    windowsDimXY_ = padDimXY_ * 2 + 1;
}

void LocalTensorBuffer::extractOverlapRegion(std::vector<torch::Tensor>& se2TimeX, std::vector<torch::Tensor>& se2TimeY, const torch::Tensor& new_se2Info, const torch::Tensor& new_gridPos, const float& new_timestamp){
  torch::Tensor offset = torch::ones({2}).to(dtype_).to(device_) * resGrid_ * padDimXY_;// ignore

  auto options_pad_se2Info = torch::nn::functional::PadFuncOptions({0, 0, padDimXY_, padDimXY_, padDimXY_, padDimXY_, padDimYaw_, padDimYaw_})
    .mode(torch::kConstant)  // 使用常数填充模式
    .value(0);               // 填充值为0
  auto options_pad_yaw = torch::nn::functional::PadFuncOptions({0, 0, 0, 0, 0, 0, padDimYaw_, padDimYaw_})
    .mode(torch::kConstant)  // 使用常数填充模式
    .value(0);               // 填充值为0
  auto options_pad_yaw1Dim = torch::nn::functional::PadFuncOptions({0, 0, padDimYaw_, padDimYaw_})
    .mode(torch::kConstant)  // 使用常数填充模式
    .value(0);               // 填充值为0

  auto new_se2Info_padded = torch::nn::functional::pad(new_se2Info, options_pad_se2Info);//[35, 45, 44, 4] // TODO: 1.0574 MB

  // yaw single, yaw_tensor_padded shape:[35, 1]
  auto yaw_tensor_padded = torch::nn::functional::pad(yawTensor_, options_pad_yaw1Dim);
  auto padded_size_dim02 = yaw_tensor_padded.size(0);
  torch::Tensor yaw_right_region_2 = yaw_tensor_padded.slice(0, padded_size_dim02 - 2*padDimYaw_, padded_size_dim02 - padDimYaw_);//0-31:33 [2, 45, 44, 4]
  torch::Tensor yaw_left_region_2 = yaw_tensor_padded.slice(0, padDimYaw_, 2*padDimYaw_);//0-2:3
  yaw_tensor_padded.slice(0, 0, padDimYaw_).copy_(yaw_right_region_2);
  yaw_tensor_padded.slice(0, padded_size_dim02 - padDimYaw_, padded_size_dim02).copy_(yaw_left_region_2);

  //! extract the overlap region se2TimeY([]) and se2TimeX([])
  std::cout << "\n----------------------------------------" << "for-each history data" << "----------------------------------------" << std::endl;
  for(auto data : data_){
    //! overlap region extraction
    std::pair<int, int> shape1 = {new_gridPos.size(0) + 2 * padDimXY_, new_gridPos.size(1) + 2 * padDimXY_};
    std::pair<int, int> shape2 = {data.gridPos.size(0), data.gridPos.size(1)};//
    //这里只是将最新的数据进行padding扩大范围，然后历史数据就不padding了，这样对标最新数据即可  
    auto [new_sx, new_sy, old_sx, old_sy] = 
      getOverlapRegion2D(new_gridPos[0][0] - offset, shape1, data.gridPos[0][0], shape2, resGrid_);


    //! PART: se2TimeY
    auto se2TimeY_temp = torch::zeros_like(new_se2Info_padded);//zeros!! [35, 45, 44, 4]
    // 提取的历史数据的重叠区域
    auto overlap_se2Info = data.se2Info.index({torch::indexing::Slice(), old_sx,old_sy, torch::indexing::Slice()});//[31, 23, 22, 4]
    
    // 填充提取出来的数据的yaw部分
    auto overlap_se2Info_padded = torch::nn::functional::pad(overlap_se2Info, options_pad_yaw);
    // 循环填充，左右新空白部分填充为原始数据的右边和左边的值
    auto padded_size_dim0 = overlap_se2Info_padded.size(0);
    torch::Tensor yaw_right_region = overlap_se2Info_padded.slice(0, padded_size_dim0 - 2*padDimYaw_, padded_size_dim0 - padDimYaw_);//0-31:33
    torch::Tensor yaw_left_region = overlap_se2Info_padded.slice(0, padDimYaw_, 2*padDimYaw_);//0-2:3
    overlap_se2Info_padded.slice(0, 0, padDimYaw_).copy_(yaw_right_region);
    overlap_se2Info_padded.slice(0, padded_size_dim0 - padDimYaw_, padded_size_dim0).copy_(yaw_left_region);
    auto temp_print = overlap_se2Info_padded.slice(0, 0, 2) - overlap_se2Info.slice(0, 29, 31);

    // 赋值历史数据至新的数据（形状相同但均为0数据）
    se2TimeY_temp.index_put_({torch::indexing::Slice(), new_sx,new_sy, torch::indexing::Slice()}, overlap_se2Info_padded);
    se2TimeY.push_back(se2TimeY_temp.unsqueeze(0));

    //! PART: se2TimeX
    auto se2TimeX_temp = torch::ones_like(new_se2Info_padded)*1e5;//1e5!! [35, 45, 44, 4]

    // grid-[35, 23, 22, 2]
    auto overlap_grid = data.gridPos.index({old_sx,old_sy, torch::indexing::Slice()});//size [old_sx, old_sy, 2]
    auto overlap_grid_expand = overlap_grid.unsqueeze(0).expand({yaw_tensor_padded.size(0), -1, -1, -1}).to(dtype_).to(device_);//[35, old_sx, old_sy, 2]
    // yaw-[35, 23, 22, 1]
    auto yaw_tensor_expand = yaw_tensor_padded.unsqueeze(1).unsqueeze(1).expand({-1, overlap_grid.size(0), overlap_grid.size(1), -1}).to(dtype_).to(device_);
    // time-[35, 23, 22, 1]
    auto timestamp_expand = torch::ones({yaw_tensor_padded.size(0), overlap_grid.size(0), overlap_grid.size(1), 1}).to(dtype_).to(device_) * data.timestamp;
    //ATTENTION: time yaw grid
    auto overlap_se2Time_catted = torch::cat({timestamp_expand, yaw_tensor_expand, overlap_grid_expand}, 3).to(dtype_).to(device_);
    
    se2TimeX_temp.index_put_({torch::indexing::Slice(), new_sx,new_sy, torch::indexing::Slice()}, overlap_se2Time_catted);
    se2TimeX.push_back(se2TimeX_temp.unsqueeze(0));
  }

  printCudaMemoryInfo("2. after for-each history data");
  //2. after for-each history data - CUDA Memory Usage: ![1.06262]! GB used / ![6.66046]! GB free / ![7.72308]! GB total

  //! new data new_se2Info_padded
  std::cout << "\n----------------------------------------" << "new data" << "----------------------------------------" << std::endl;
  //! PART: 
  // Extract the right padDimYaw_ region from the original tensor
  auto padded_size_dim0 = new_se2Info_padded.size(0);
  torch::Tensor yaw_right_region = new_se2Info_padded.slice(0, padded_size_dim0 - 2*padDimYaw_, padded_size_dim0 - padDimYaw_);//0:31:33
  torch::Tensor yaw_left_region = new_se2Info_padded.slice(0, padDimYaw_, 2*padDimYaw_);//0:2:3
  new_se2Info_padded.slice(0, 0, padDimYaw_).copy_(yaw_right_region);
  new_se2Info_padded.slice(0, padded_size_dim0 - padDimYaw_, padded_size_dim0).copy_(yaw_left_region);
  se2TimeY.push_back(new_se2Info_padded.unsqueeze(0));

  //! PART:
  //! new data new_se2Xvalue_padded
  auto se2TimeX_temp = torch::ones_like(new_se2Info_padded)*1e5;//1e5!! [35, 45, 44, 4] NOTE: 1.0574 MB
  //yaw_expand: [35, 37, 36, 1]  NOTE: 0.177841187 MB
  auto yaw_tensor_expand = yaw_tensor_padded.unsqueeze(1).unsqueeze(1).expand({-1, new_se2Info.size(1), new_se2Info.size(2), -1}).to(dtype_).to(device_);
  //timestamp_expand: [35, 37, 36, 1] NOTE: 0.177841187 MB
  auto timestamp_expand = torch::ones({yaw_tensor_expand.size(0), new_se2Info.size(1), new_se2Info.size(2), 1}).to(dtype_).to(device_) * new_timestamp;
  //gridPos_expand: [35, 37, 36, 2] NOTE: 0.177841187 MB
  auto new_gridPos_expand = new_gridPos.unsqueeze(0).expand({yaw_tensor_padded.size(0), -1, -1, -1}).to(dtype_).to(device_);
  auto overlap_se2Time_catted = torch::cat({timestamp_expand, yaw_tensor_expand, new_gridPos_expand}, 3).to(dtype_).to(device_);
  auto new_sx2 = torch::indexing::Slice(padDimXY_, new_se2Info_padded.size(1)-padDimXY_);
  auto new_sy2 = torch::indexing::Slice(padDimXY_, new_se2Info_padded.size(2)-padDimXY_);
  se2TimeX_temp.index_put_({torch::indexing::Slice(), new_sx2, new_sy2, torch::indexing::Slice()}, overlap_se2Time_catted);
  se2TimeX.push_back(se2TimeX_temp.unsqueeze(0));
}





std::tuple<torch::Tensor, torch::Tensor> LocalTensorBuffer::assembleSe2tTrainData(const std::vector<torch::Tensor>& se2TimeX, const std::vector<torch::Tensor>& se2TimeY){
  std::cout << "\n----------------------------------------" << "Begin to cat data" << "----------------------------------------" << std::endl;
  //========== se2TimeY train handle
  // cat data  
  auto se2TimeY_cated = torch::cat(se2TimeY, 0);//[4, 35, 45, 44, 4] 
  // NOTE: 1.890197754 MB data_ptr<float>(): 0x77fcf5384c00 NOTE: as the same as before

  // unfold
  auto se2TimeY_unfolded = se2TimeY_cated.unfold(1, windowsDimYaw_, 1).unfold(2, windowsDimXY_, 1).unfold(3, windowsDimXY_, 1);//[4, 31, 37, 36, 4, 5, 9, 9] 
  // NOTE: 406.458 MB data_ptr<float>(): 0x77fcf5384c00 NOTE: as the same as before

  // permute 
  auto se2TimeY_unfolded_permute = se2TimeY_unfolded.permute({1, 2, 3, 0, 5, 6, 7, 4}); //[31, 37, 36, 4, 5, 9, 9, 4] 
  // NOTE: 406.458 MB data_ptr<float>(): 0x77fcf5384c00 NOTE: as the same as before

  // reshape 
  auto shapeY = se2TimeY_unfolded_permute.sizes();
  auto se2TimeY_unfolded_permute_reshaped = se2TimeY_unfolded_permute.reshape({shapeY[0], shapeY[1], shapeY[2], -1, shapeY[7]});//[31, 37, 36, 1620, 4]
  // NOTE: 541.944 MB data_ptr<float>(): 0x77fcac000000
  // 406.458 MB + 541.944 MB = 948.402 MB

  //========== se2TimeX train handle
  // cat
  auto se2TimeX_cated = torch::cat(se2TimeX, 0);//[4, 35, 45, 44, 4]  
  // NOTE: 1.890197754 MB data_ptr<float>(): 0x77fcf57bfa00 NOTE: as the same as before

  // unfold
  auto se2TimeX_unfolded = se2TimeX_cated.unfold(1, windowsDimYaw_, 1).unfold(2, windowsDimXY_, 1).unfold(3, windowsDimXY_, 1);//[4, 31, 37, 36, 4, 5, 9, 9]
  // NOTE: 406.458 MB data_ptr<float>(): 0x77fcf57bfa00 NOTE: as the same as before

  // permute
  auto se2TimeX_unfolded_permute = se2TimeX_unfolded.permute({1, 2, 3, 0, 5, 6, 7, 4});//[31, 37, 36, 4, 5, 9, 9, 4]
  // NOTE: 406.458 MB data_ptr<float>(): 0x77fcf57bfa00 NOTE: as the same as before

  // reshaped
  auto shapeX = se2TimeX_unfolded_permute.sizes();
  auto se2TimeX_unfolded_permute_reshaped = se2TimeX_unfolded_permute.reshape({shapeX[0], shapeX[1], shapeX[2], -1, shapeX[7]});//[31, 37, 36, 1620, 4]
  // NOTE: 541.944 MB data_ptr<float>(): 0x77fc6a000000

  return {se2TimeX_unfolded_permute_reshaped, se2TimeY_unfolded_permute_reshaped};
}

torch::Tensor LocalTensorBuffer::computeSe2tDistMat(const torch::Tensor& se2tTrainX, const torch::Tensor& se2tPredX){
  auto slice = torch::indexing::Slice();
  // 定义一个lambda函数来简化索引表达式
  auto makeSlice = [slice](int dim) {
      return std::vector<torch::indexing::TensorIndex>{slice, slice, slice, slice, dim};
  };

  // 计算时间戳的绝对差值
  torch::Tensor timestamp_diff = torch::abs(se2tTrainX.index(makeSlice(0)) - se2tPredX.index(makeSlice(0)));
  std::cout << "timestamp_diff.sizes(): " << timestamp_diff.sizes() << std::endl;           

  // 计算yaw的绝对差值，并规范化到[-pi, pi]
  torch::Tensor yaw_diff = torch::abs(se2tTrainX.index(makeSlice(1)) - se2tPredX.index(makeSlice(1)));
  // 使用公式: (diff + π) % (2π) - π 来确保差值在[-π, π]范围内
  yaw_diff = (yaw_diff + torch::acos(torch::tensor(-1.0))) % (2 * torch::acos(torch::tensor(-1.0))) - torch::acos(torch::tensor(-1.0));
  std::cout << "yaw_diff.sizes(): " << yaw_diff.sizes() << std::endl;

  // 计算grid位置的欧几里得距离
  // 获取gridX和gridY的差值
  torch::Tensor gridX_diff = se2tTrainX.index(makeSlice(2)) - se2tPredX.index(makeSlice(2));
  torch::Tensor gridY_diff = se2tTrainX.index(makeSlice(3)) - se2tPredX.index(makeSlice(3));

  // 计算欧几里得距离
  torch::Tensor grid_dist = torch::sqrt(gridX_diff.square() + gridY_diff.square());

  // 将三个距离维度组合成最终张量 [31,37,36,1620,3]
  auto se2t_dist = torch::cat({timestamp_diff.unsqueeze(-1), yaw_diff.unsqueeze(-1), grid_dist.unsqueeze(-1)}, /*dim=*/-1);
  std::cout << "se2_dist.sizes(): " << se2t_dist.sizes() << std::endl;
  return se2t_dist;
}

torch::Tensor LocalTensorBuffer::computeSe2tKernel(const torch::Tensor& se2tDistMat){
  assert(se2tDistMat.size(-1) == kLenTimeYawGrid_.size(0));
  const auto M2PI = 2.0 * M_PI;

  auto klen = kLenTimeYawGrid_.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0);

  torch::Tensor se2_kernel; // [31, 37, 36, 1620, 3] ATTENTION: 0.74758GB
  //local region, save cuda memory  
  auto term11 = se2tDistMat.clone();
  term11.div_(klen).mul_(M2PI).cos_().add_(2.0);
  
  auto term12 = se2tDistMat.clone();
  term12.div_(klen).sub_(1.0).mul_(-0.333333);

  auto term2 = se2tDistMat.clone();
  term2.div_(klen).mul_(M2PI).sin_().div_(M2PI);

  se2_kernel = term11.mul_(term12).add_(term2);
  se2_kernel.mul_((se2_kernel > 0.0).to(torch::kFloat32));
  se2_kernel = se2_kernel.sum(-1, true);
  return se2_kernel;
}

std::tuple<torch::Tensor, torch::Tensor> LocalTensorBuffer::fuseThroughSpatioTemporalBGKI(const torch::Tensor& new_se2Info, const torch::Tensor& new_gridPos, const float& new_timestamp){
  if(data_.size() < capacity_){
    std::cout << "buffer size not enough, now: " << data_.size() << std::endl;
    return {torch::Tensor(), torch::Tensor()};
  }
  printCudaMemoryInfo("0. before fuse, init state");

  //PART: 1 extract overlap region of history data and new data
  std::cout << "\n----------------------------------------" << "STBGKI-Prepare data, extract overlap region" << "----------------------------------------" << std::endl;
  std::vector<torch::Tensor> se2TimeX, se2TimeY;
  extractOverlapRegion(se2TimeX, se2TimeY, new_se2Info, new_gridPos, new_timestamp);
  printCudaMemoryInfo("3. After new-added date ");

  //PART: 2 construct se2trainX and se2trainY
  std::cout << "\n----------------------------------------" << "STBGKI-Construct Se2TrainData" << "----------------------------------------" << std::endl;
  auto [se2t_trainX, se2t_trainY] = assembleSe2tTrainData(se2TimeX, se2TimeY); // [31, 37, 36, 1620, 4]  ATTENTION: 0.99678GB
  torch::Tensor se2t_predX = new_se2Info.unsqueeze(-2); // [31, 37, 36, 4] 
  printCudaMemoryInfo("4. After cat, unfold, permute, reshape TENSOR");

  //PART: 3 compute se2_dist
  std::cout << "\n----------------------------------------" << "STBGKI-Compute Se2Dist" << "----------------------------------------" << std::endl;
  torch::Tensor se2t_dist = computeSe2tDistMat(se2t_trainX, se2t_predX); // [31, 37, 36, 1620, 3] ATTENTION: 0.74758GB 
  printCudaMemoryInfo("5. After compute se2_dist matrix");

  //ATTENTION: free tensor's memory named !!![se2t_trainX & se2t_predX]!!!
  se2t_trainX = torch::Tensor();
  se2t_predX = torch::Tensor();

  //PART: 4 compute se2_kernel, last-dim:[time, yaw, grid]  
  std::cout << "\n----------------------------------------" << "STBGKI-Compute covSparse" << "----------------------------------------" << std::endl;
  const auto M2PI = 2.0 * M_PI;
  torch::Tensor se2t_kernel = computeSe2tKernel(se2t_dist); // [31, 37, 36, 1620, 3] ATTENTION: 0.74758GB
  std::cout << "se2t_kernel.sizes(): " << se2t_kernel.sizes() << std::endl;
  printCudaMemoryInfo("se2_kernel compute over");

  //ATTENTION: free tensor's memory named !!![se2t_dist]!!!
  se2t_dist= torch::Tensor();

  c10::cuda::CUDACachingAllocator::emptyCache();
  printCudaMemoryInfo("After c10::cuda::CUDACachingAllocator::emptyCache()");
  
  std::cout << "=====Tensors freed memory" << std::endl;
  std::cout << "se2t_trainX.sizes(): " << se2t_trainX.sizes() << " se2t_trainX.use_count(): " << se2t_trainX.use_count() << std::endl;// size:[0] use_count:0
  std::cout << "se2t_predX.sizes(): " << se2t_predX.sizes() << " se2t_predX.use_count(): " << se2t_predX.use_count() << std::endl;// size:[0] use_count:0
  std::cout << "se2t_dist.sizes(): " << se2t_dist.sizes() << " se2t_dist.use_count(): " << se2t_dist.use_count() << std::endl;// size:[0] use_count:0
  std::cout << std::endl;
  std::cout << "=====Tensors unfreed memory" << std::endl;
  std::cout << "se2t_trainY.sizes(): " << se2t_trainY.sizes() << " se2t_trainY.use_count(): " << se2t_trainY.use_count() << std::endl;// size:[31, 37, 36, 1620, 4] use_count:1
  std::cout << "se2t_kernel.sizes(): " << se2t_kernel.sizes() << " se2t_kernel.use_count(): " << se2t_kernel.use_count() << std::endl;// size:[31, 37, 36, 1620, 1] use_count:1



  //PART: 5 BGKI fuse  
  // se2t_kernel:[31, 37, 36, 1620, 1]
  // se2t_trainY:[31, 37, 36, 1620, 4]
  // se2t_trainSigma2:[31, 37, 36, 1620, 1] //TODO: construct this   

  auto se2t_trainSigma2 = torch::ones({31, 37, 36, 1620, 1}).to(device_).to(dtype_);
  se2t_kernel.div_(se2t_trainSigma2);
  
  auto mu0 = new_se2Info; // size:[31, 37, 36, 4]
  auto sigma20 = torch::ones({31, 37, 36, 1}).to(device_).to(dtype_);//size:[31, 37, 36, 1] //TODO: optim this  
  sigma20.mul_(1e-4);

  auto kbar = se2t_kernel.sum({-2, -1}, false).unsqueeze(-1) + 1/sigma20; //size:[31, 37, 36]
  auto ybar = (se2t_trainY.mul_(se2t_kernel)).sum({-2}, false) + mu0.div(sigma20); // size:[31, 37, 36, 4] NOTE: 4 is the se2Info dim  

  ybar.div_(kbar.add_(1e-6));
  kbar.reciprocal_();

  std::cout << "kbar.sizes(): " << kbar.sizes() << std::endl;
  std::cout << "ybar.sizes(): " << ybar.sizes() << std::endl;

  std::cout << "ybar-mu0: " << (ybar - mu0).abs().sum() << std::endl;
  std::cout << "mu0: " << mu0.abs().sum() << "=? " << 31 * 37 * 36 * 4 << std::endl;
  std::cout << "ybar: " << ybar.abs().sum() << std::endl;

  // std::cout << "mu0: " << mu0 << std::endl;
  // std::cout << "ybar: " << ybar << std::endl;
  


  return {ybar, kbar};
}

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
LocalTensorBuffer::getOverlapRegion2D(const torch::Tensor& start1, const std::pair<int, int>& shape1, 
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


