#ifndef LOCAL_SE2T_BGKIMAP_HPP
#define LOCAL_SE2T_BGKIMAP_HPP

#include <iostream>
#include <chrono>
#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <cmath>
#include <cuda_runtime.h>
#include <cassert>
#include <vector>
#include <random>
#include <deque>

#include "cuda_kernel.h"



torch::Tensor generateGridTensor(int height, int width, float resolution, const std::pair<float, float>& start);


class MeasureCUDARuntime{
  public:
    MeasureCUDARuntime(){

    }
    void start(){
      cudaEventCreate(&start_);
      cudaEventCreate(&end_);
      cudaEventRecord(start_);
    }  

    float end(){
      cudaEventRecord(end_);
      cudaEventSynchronize(end_);
      float milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, start_, end_);
      return milliseconds;
    }

  private:
    cudaEvent_t start_, end_;

};



class LocalTensorBuffer{
  private:
    struct MapInfo{
      torch::Tensor se2Info; //[yawDim, xDim, yDim, valueDim](valueDim = 3: nx, ny, trav)
      torch::Tensor gridPos; //[        xDIm, yDim, gridDim](gridDim = 2: x, y)
      torch::Tensor variance;//[yawDim, xDim, yDim, 1]
      float timestamp;
    };
  public://membership function
    LocalTensorBuffer(const int& capacity, const torch::Tensor& yawTensor, const float& resYaw, const float& resGrid, const std::vector<float>& kLenTimeYawGrid): 
    capacity_(capacity), yawTensor_(yawTensor), resYaw_(resYaw), resGrid_(resGrid){
      device_ = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
      // device_ = torch::kCPU;
      std::cout << "device_: " << device_ << std::endl;
      dtype_ = torch::kFloat32;

      kLenTimeYawGrid_ = torch::tensor(kLenTimeYawGrid, dtype_).to(device_);
      variancePrior_ = torch::ones({1}).to(device_).to(dtype_) * varianceInit_;

      updatePadDimAndUnfoldWindows();
      
      // std::cout << "padDimYaw_: " << padDimYaw_ << std::endl;
      // std::cout << "padDimXY_: " << padDimXY_ << std::endl;
      // std::cout << "windowsDimYaw_: " << windowsDimYaw_ << std::endl;
      // std::cout << "windowsDimXY_: " << windowsDimXY_ << std::endl;
    }

    size_t size() const {
      return data_.size();
    }

    bool empty() const {
      return data_.empty();
    }
 
    void insert(const torch::Tensor& se2Info, const torch::Tensor& gridPos, const float& timestamp){
      static MeasureCUDARuntime timer;
      torch::Tensor tensor_fused, cov_fused;
      auto temp1 = se2Info;
      auto temp2 = gridPos;
      temp1 = temp1.to(device_);
      // std::cout << "insert se2Info.abs().sum(): " << se2Info.abs().sum() << std::endl;
      temp2 = temp2.to(device_);
      if (data_.size() >= capacity_) {
        timer.start();
        std::chrono::high_resolution_clock::time_point time_start_cpu;
        time_start_cpu = std::chrono::high_resolution_clock::now();
        
        auto [ybar, kbar]  = fuseThroughSpatioTemporalBGKI(temp1, temp2, timestamp);
        
        float runtime_ms = timer.end();
        std::cout << "runtime_ms: " << runtime_ms << " ms" << std::endl;
        auto time_end_cpu = std::chrono::high_resolution_clock::now();
        auto duration_cpu = std::chrono::duration_cast<std::chrono::microseconds>(time_end_cpu - time_start_cpu);
        std::cout << "device: CPU    memsurement | " << "Eigenvalue decomposition runtime: " << duration_cpu.count()/1000.0 << " ms" << std::endl;
        
        
        tensor_fused = ybar;
        cov_fused = kbar;
        data_.pop_front();
      }else{
        tensor_fused = temp1;
        cov_fused = torch::ones({se2Info.size(0), se2Info.size(1), se2Info.size(2), 1}).to(dtype_).to(device_);
        cov_fused.mul_(varianceInit_);
        // std::cout << "!!!!!cov_fused.abs().sum():" << cov_fused.abs().sum() << std::endl;
      }
      data_.push_back({tensor_fused, temp2, cov_fused, timestamp});
      // std::cout << "DEBUG INSERT OVER: data_.back().se2Info.sum():" << data_.back().se2Info.sum() << std::endl;
      // std::cout << "DEBUG INSERT OVER: data_.back().gridPos.sum():" << data_.back().gridPos.sum() << std::endl;
    }

  private://membership inline function
    inline float estimateTensorMemoryGB(const torch::Tensor& tensor) {
      int64_t num_elements = 1;
      for (int64_t dim_size : tensor.sizes()) {
          num_elements *= dim_size;
      }
      int64_t bytes_per_element = tensor.dtype().itemsize();
      int64_t total_bytes = num_elements * bytes_per_element;
      float total_gb = static_cast<float>(total_bytes) / (1024.0f * 1024.0f * 1024.0f);
      return total_gb;
    }
  

  private://membership function
    void printCudaMemoryInfo(const char* step);
    float getFreeCUDAMemoryGB();
    void updatePadDimAndUnfoldWindows();
    /**
     * @brief:: 
     * @attention: start1, start2: [x, y]  shape1, shape2: [h, w]. the must be 2D type
     * @return {*}
     */  
    std::tuple<torch::indexing::Slice, torch::indexing::Slice, torch::indexing::Slice, torch::indexing::Slice> 
    getOverlapRegion2D(const torch::Tensor& start1, const std::pair<int, int>& shape1, 
                       const torch::Tensor& start2, const std::pair<int, int>& shape2, double resolution);
    void extractOverlapRegion(std::vector<torch::Tensor>& se2TimeX, std::vector<torch::Tensor>& se2TimeY, const std::deque<MapInfo>& otherDatas, const torch::Tensor& new_se2Info, const torch::Tensor& new_gridPos, const float& new_timestamp);
    std::tuple<torch::Tensor, torch::Tensor> assembleSe2tTrainData(const std::vector<torch::Tensor>& se2TimeX, const std::vector<torch::Tensor>& se2TimeY);
    torch::Tensor computeSe2tDistMat(const torch::Tensor& se2tTrainX, const torch::Tensor& se2tPredX);
    torch::Tensor computeSe2tKernel(const torch::Tensor& se2tDistMat);
    torch::Tensor computeSe2tDistMatCUDAKernel(const torch::Tensor& se2tTrainX, const torch::Tensor& se2tPredX);
    torch::Tensor computeSe2tKernelCUDAKernel(const torch::Tensor& se2tDistMat);

    torch::Tensor batchHandleTensorOperator(std::function<torch::Tensor(const torch::Tensor&)> _operFun, float _funUseGB, const torch::Tensor& _inputTensor);
    std::tuple<torch::Tensor, torch::Tensor> fuseThroughSpatioTemporalBGKI(const torch::Tensor& new_se2Info, const torch::Tensor& new_gridPos, const float& new_timestamp);

    void negLogMLLOptimHyperParameters(const torch::Tensor& new_se2Info, const torch::Tensor& new_gridPos, const float& new_timestamp);

  public://membership variable

  private://membership variable
    int capacity_;
    torch::Tensor yawTensor_;  
    float resYaw_;
    float resGrid_;
    
    torch::DeviceType device_;
    torch::Dtype dtype_;

    torch::Tensor kLenTimeYawGrid_;
    torch::Tensor variancePrior_;
    int padDimYaw_; // std::round(Klen_yaw / resYaw_);
    int padDimXY_;  // std::round(Klen_xy / resGrid_);
    int windowsDimYaw_; // 2 * padDimYaw_ + 1;
    int windowsDimXY_;  // 2 * padDimXY_ + 1;

    const float varianceInit_ = 1e-4;
    const float delta_ = 1e-6;


    std::deque<MapInfo> data_;
};


#endif