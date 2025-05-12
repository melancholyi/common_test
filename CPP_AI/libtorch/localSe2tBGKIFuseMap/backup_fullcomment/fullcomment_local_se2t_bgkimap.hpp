#ifndef LOCAL_SE2T_BGKIMAP_HPP
#define LOCAL_SE2T_BGKIMAP_HPP

#include <iostream>
#include <chrono>
#include <torch/torch.h>
#include <cmath>
#include <cuda_runtime.h>
#include <cassert>
#include <vector>
#include <random>
#include <deque>
void print_cuda_memory_info(const char* step);
torch::Tensor generateGridTensor(int height, int width, float resolution, const std::pair<float, float>& start);


class LocalTensorBuffer{
  private:
    struct MapInfo{
      torch::Tensor se2Info;
      torch::Tensor gridPos;
      float timestamp;
    };
  public://membership function
    LocalTensorBuffer(const int& capacity, const torch::Tensor& yawTensor, const float& resYaw, const float& resGrid, const std::vector<float>& kLenTimeYawGrid): 
    capacity_(capacity), yawTensor_(yawTensor), resYaw_(resYaw), resGrid_(resGrid){
      device_ = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
      dtype_ = torch::kFloat32;

      kLenTimeYawGrid_ = torch::tensor(kLenTimeYawGrid, dtype_).to(device_);

      updatePadDimAndUnfoldWindows();
      
      std::cout << "padDimYaw_: " << padDimYaw_ << std::endl;
      std::cout << "padDimXY_: " << padDimXY_ << std::endl;
      std::cout << "windowsDimYaw_: " << windowsDimYaw_ << std::endl;
      std::cout << "windowsDimXY_: " << windowsDimXY_ << std::endl;
    
    }

  
    size_t size() const {
      return data_.size();
    }
    bool empty() const {
      return data_.empty();
    }
 
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

    void extractOverlapRegion(std::vector<torch::Tensor>& se2TimeX, std::vector<torch::Tensor>& se2TimeY, const torch::Tensor& new_se2Info, const torch::Tensor& new_gridPos, const float& new_timestamp);
    std::tuple<torch::Tensor, torch::Tensor> assembleSe2tTrainData(const std::vector<torch::Tensor>& se2TimeX, const std::vector<torch::Tensor>& se2TimeY);
    torch::Tensor computeSe2tDistMat(const torch::Tensor& se2tTrainX, const torch::Tensor& se2tPredX);
    torch::Tensor computeSe2tKernel(const torch::Tensor& se2tDistMat);
    std::tuple<torch::Tensor, torch::Tensor> fuseThroughSpatioTemporalBGKI(const torch::Tensor& new_se2Info, const torch::Tensor& new_gridPos, const float& new_timestamp);

  private://membership function
    void updatePadDimAndUnfoldWindows();


    /**
     * @brief:: 
     * @attention: start1, start2: [x, y]  shape1, shape2: [h, w]. the must be 2D type
     * @return {*}
     */  
    std::tuple<torch::indexing::Slice, torch::indexing::Slice, torch::indexing::Slice, torch::indexing::Slice> 
    getOverlapRegion2D(const torch::Tensor& start1, const std::pair<int, int>& shape1, 
                      const torch::Tensor& start2, const std::pair<int, int>& shape2, double resolution);

  public://membership variable

  private://membership variable
    int capacity_;
    torch::Tensor yawTensor_;  
    float resYaw_;
    float resGrid_;
    torch::Tensor kLenTimeYawGrid_;
    int padDimYaw_; // std::round(Klen_yaw / resYaw_);
    int padDimXY_;  // std::round(Klen_xy / resGrid_);
    int windowsDimYaw_; // 2 * padDimYaw_ + 1;
    int windowsDimXY_;  // 2 * padDimXY_ + 1;

    torch::DeviceType device_;
    torch::Dtype dtype_;
    std::deque<MapInfo> data_;
};


#endif