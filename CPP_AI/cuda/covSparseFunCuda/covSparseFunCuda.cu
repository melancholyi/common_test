#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include "torch/torch.h"
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
int64_t computeTensorElemCount(const torch::Tensor& tensor) {
    int64_t num_elements = 1;
    for (int64_t dim_size : tensor.sizes()) {
        num_elements *= dim_size;
    }
    return num_elements;
}


torch::Tensor computeSe2tKernel(const torch::Tensor& se2tDistMat, 
    const torch::Tensor& kLenTimeYawGrid) {
  // std::cout << "call once" << std::endl;
  assert(se2tDistMat.size(-1) == kLenTimeYawGrid.size(0));
  const auto M2PI = 2.0 * M_PI;

  auto klen = kLenTimeYawGrid.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0);

  torch::Tensor se2_kernel; // [31, 37, 36, 1620, 3] ATTENTION: 0.74758GB
  //local region, save cuda memory  

  MeasureCUDARuntime timer;
  timer.start();
  auto term11 = se2tDistMat.clone(); //NOTE: 0.74GB
  term11.div_(klen).mul_(M2PI).cos_().add_(2.0);
  auto term11_ms = timer.end();
  //std::cout << "term11_ms: " << term11_ms << std::endl; //term11_ms: 34.7003
  
  timer.start();
  auto term12 = se2tDistMat.clone(); //NOTE: 0.74GB
  term12.div_(klen).sub_(1.0).mul_(-0.333333);
  auto term12_ms = timer.end();
  //std::cout << "term12_ms: " << term12_ms << std::endl; // term12_ms: 23.9626

  timer.start();
  auto term2 = se2tDistMat.clone(); //NOTE: 0.74GB
  term2.div_(klen).mul_(M2PI).sin_().div_(M2PI);
  auto term2_ms = timer.end();
  //std::cout << "term2_ms: " << term2_ms << std::endl; //term2_ms: 27.8292

  // auto term3 = se2tDistMat.clone(); //NOTE: 0.74GB
  // auto term4 = se2tDistMat.clone(); //NOTE: 0.74GB
  // auto term5 = se2tDistMat.clone(); //NOTE: 0.74GB
  // auto term6 = se2tDistMat.clone(); //NOTE: 0.74GB
  // auto term7 = se2tDistMat.clone(); //NOTE: 0.74GB

  timer.start();
  se2_kernel = term11.mul_(term12).add_(term2).clamp_min_(0.0);
  se2_kernel = se2_kernel.sum(-1, true);
  auto se2_kernel_ms = timer.end();
  //std::cout << "se2_kernel_ms: " << se2_kernel_ms << std::endl;//se2_kernel_ms: 25.7792

  // exit(0);
  return se2_kernel;
}

__global__ void se2_kernel_cuda(
    const float* se2tDistMat,
    const float* kLen,
    float* output,
    int num_elements,
    int num_channels
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    const float M2PI = 2.0f * M_PI;
    float sum = 0.0f;

    for (int c = 0; c < num_channels; ++c) {
        const int input_idx = idx * num_channels + c;
        const float input_val = se2tDistMat[input_idx];
        const float klen_val = kLen[c];
        
        const float d_scaled = input_val / klen_val;
        const float angle = d_scaled * M2PI;
        
        float cos_val, sin_val;
        sincosf(angle, &sin_val, &cos_val); // 同时计算sin和cos
        
        const float term11 = cos_val + 2.0f;
        const float term12 = (d_scaled - 1.0f) * (-1.0f/3.0f);
        const float term2 = sin_val / M2PI;
        
        float result = term11 * term12 + term2;
        result = fmaxf(result, 0.0f);
        sum += result;
    }
    
    output[idx] = sum;
}
torch::Tensor computeSe2tKernelOptimized(
    const torch::Tensor& se2tDistMat,
    const torch::Tensor& kLenTimeYawGrid
) {
    // 确保输入为CUDA float32张量
    TORCH_CHECK(se2tDistMat.is_cuda() && se2tDistMat.dtype() == torch::kFloat32);
    TORCH_CHECK(kLenTimeYawGrid.is_cuda() && kLenTimeYawGrid.dtype() == torch::kFloat32);
    TORCH_CHECK(se2tDistMat.size(-1) == 3, "Last dim must be 3");
    TORCH_CHECK(kLenTimeYawGrid.size(0) == 3, "kLen dim must be 3");

    // 计算输出形状 [..., 1]
    auto sizes = se2tDistMat.sizes().vec();
    sizes.back() = 1; // 将最后一个维度设为1

    // 分配输出张量
    torch::Tensor output = torch::empty(sizes, se2tDistMat.options());
    const int num_elements = output.numel();
    const int num_channels = 3;

    // 启动参数配置
    const int threads_per_block = 256; // 4.33971 ms
    const int blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    // 获取数据指针
    const float* se2t_data = se2tDistMat.data_ptr<float>();
    const float* kLen_data = kLenTimeYawGrid.data_ptr<float>();
    float* output_data = output.data_ptr<float>();

    // 启动CUDA Kernel
    se2_kernel_cuda<<<blocks, threads_per_block>>>(
        se2t_data,
        kLen_data,
        output_data,
        num_elements,
        num_channels
    );

    return output;
}

//NOTE: optim verion, but didn't improve a lot
// __global__ void se2_kernel_cuda_optimized(
//     const float* __restrict__ se2tDistMat,
//     const float* __restrict__ kLen,
//     float* __restrict__ output,
//     int num_elements
// ) {
//     extern __shared__ float s_kLen[]; // 动态共享内存缓存kLen参数

//     // 加载kLen到共享内存（每个线程块只加载一次）
//     if (threadIdx.x < 3) {
//         s_kLen[threadIdx.x] = kLen[threadIdx.x];
//     }
//     __syncthreads();

//     const int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= num_elements) return;

//     const float M2PI = 2.0f * M_PI;
//     float sum = 0.0f;

//     // 向量化加载三个通道的数据（float3）
//     const float3 input_vals = *reinterpret_cast<const float3*>(&se2tDistMat[idx*3]);

//     // 手动展开循环处理三个通道
//     // 通道0
//     {
//         const float d_scaled = input_vals.x / s_kLen[0];
//         const float angle = d_scaled * M2PI;
//         float sin_val, cos_val;
//         sincosf(angle, &sin_val, &cos_val);
//         sum += fmaxf((cos_val + 2.0f) * (1.0f - d_scaled) * 0.333333333f + sin_val / M2PI, 0.0f);
//     }

//     // 通道1
//     {
//         const float d_scaled = input_vals.y / s_kLen[1];
//         const float angle = d_scaled * M2PI;
//         float sin_val, cos_val;
//         sincosf(angle, &sin_val, &cos_val);
//         sum += fmaxf((cos_val + 2.0f) * (1.0f - d_scaled) * 0.333333333f + sin_val / M2PI, 0.0f);
//     }

//     // 通道2
//     {
//         const float d_scaled = input_vals.z / s_kLen[2];
//         const float angle = d_scaled * M2PI;
//         float sin_val, cos_val;
//         sincosf(angle, &sin_val, &cos_val);
//         sum += fmaxf((cos_val + 2.0f) * (1.0f - d_scaled) * 0.333333333f + sin_val / M2PI, 0.0f);
//     }

//     output[idx] = sum;
// }

// torch::Tensor computeSe2tKernelOptimized( // {31, 37, 36, 1260, 3} 4.2ms
//     const torch::Tensor& se2tDistMat,
//     const torch::Tensor& kLenTimeYawGrid
// ) {
//     // 确保输入为CUDA float32张量
//     TORCH_CHECK(se2tDistMat.is_cuda() && se2tDistMat.dtype() == torch::kFloat32);
//     TORCH_CHECK(kLenTimeYawGrid.is_cuda() && kLenTimeYawGrid.dtype() == torch::kFloat32);
//     TORCH_CHECK(se2tDistMat.size(-1) == 3, "Last dim must be 3");
//     TORCH_CHECK(kLenTimeYawGrid.size(0) == 3, "kLen dim must be 3");

//     // 计算输出形状 [..., 1]
//     auto sizes = se2tDistMat.sizes().vec();
//     sizes.back() = 1; // 将最后一个维度设为1

//     // 分配输出张量
//     torch::Tensor output = torch::empty(sizes, se2tDistMat.options());
//     const int num_elements = output.numel();
//     // const int num_channels = 3;

//     // 启动参数调整
//     const int threads_per_block = 128; // 根据测试调整最佳线程数
//     const int blocks = (num_elements + threads_per_block - 1) / threads_per_block;
//     const size_t shared_mem_size = 3 * sizeof(float);

//     // 获取数据指针
//     const float* se2t_data = se2tDistMat.data_ptr<float>();
//     const float* kLen_data = kLenTimeYawGrid.data_ptr<float>();
//     float* output_data = output.data_ptr<float>();


//     // 启动优化后的kernel
//     se2_kernel_cuda_optimized<<<blocks, threads_per_block, shared_mem_size>>>(
//         se2t_data,
//         kLen_data,
//         output_data,
//         num_elements
//     );

//     return output;
// }

__global__ void computeSe2tDistMatKernel(
    const float* train_data,
    int train_stride0, int train_stride1, int train_stride2, int train_stride3, int train_stride4,
    const float* pred_data,
    int pred_stride0, int pred_stride1, int pred_stride2, int pred_stride3, int pred_stride4,
    float* output_data,
    int output_stride0, int output_stride1, int output_stride2, int output_stride3, int output_stride4,
    int s0, int s1, int s2, int s3) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= s0 * s1 * s2 * s3) return;

    // 分解线性索引到4D坐标 [i, j, k, l]
    int l = idx % s3;
    int remaining = idx / s3;
    int k = remaining % s2;
    remaining /= s2;
    int j = remaining % s1;
    int i = remaining / s1;

    // 计算各分量在输入张量中的偏移量
    const int train_time_offset = i * train_stride0 + j * train_stride1 + k * train_stride2 + l * train_stride3 + 0 * train_stride4;
    const int train_yaw_offset = i * train_stride0 + j * train_stride1 + k * train_stride2 + l * train_stride3 + 1 * train_stride4;
    const int train_gridX_offset = i * train_stride0 + j * train_stride1 + k * train_stride2 + l * train_stride3 + 2 * train_stride4;
    const int train_gridY_offset = i * train_stride0 + j * train_stride1 + k * train_stride2 + l * train_stride3 + 3 * train_stride4;

    const int pred_time_offset = i * pred_stride0 + j * pred_stride1 + k * pred_stride2 + l * pred_stride3 + 0 * pred_stride4;
    const int pred_yaw_offset = i * pred_stride0 + j * pred_stride1 + k * pred_stride2 + l * pred_stride3 + 1 * pred_stride4;
    const int pred_gridX_offset = i * pred_stride0 + j * pred_stride1 + k * pred_stride2 + l * pred_stride3 + 2 * pred_stride4;
    const int pred_gridY_offset = i * pred_stride0 + j * pred_stride1 + k * pred_stride2 + l * pred_stride3 + 3 * pred_stride4;

    // 计算时间差
    const float timestamp_diff = fabsf(train_data[train_time_offset] - pred_data[pred_time_offset]);

    // 计算yaw差并规范化
    float yaw_diff = fabsf(train_data[train_yaw_offset] - pred_data[pred_yaw_offset]);
    if (yaw_diff < 100.0f) {
        yaw_diff = fmodf(yaw_diff + M_PI, 2 * M_PI) - M_PI;
        yaw_diff = fabsf(yaw_diff);
    }

    // 计算欧氏距离
    const float gridX_diff = train_data[train_gridX_offset] - pred_data[pred_gridX_offset];
    const float gridY_diff = train_data[train_gridY_offset] - pred_data[pred_gridY_offset];
    const float grid_dist = sqrtf(gridX_diff * gridX_diff + gridY_diff * gridY_diff);

    // 计算输出偏移量
    const int output_offset0 = i * output_stride0 + j * output_stride1 + k * output_stride2 + l * output_stride3 + 0 * output_stride4;
    const int output_offset1 = i * output_stride0 + j * output_stride1 + k * output_stride2 + l * output_stride3 + 1 * output_stride4;
    const int output_offset2 = i * output_stride0 + j * output_stride1 + k * output_stride2 + l * output_stride3 + 2 * output_stride4;

    // 写入结果
    output_data[output_offset0] = timestamp_diff;
    output_data[output_offset1] = yaw_diff;
    output_data[output_offset2] = grid_dist;
}

torch::Tensor computeSe2tDistMatCUDAKernelVersion(
    const torch::Tensor& se2tTrainX, const torch::Tensor& se2tPredX) {

    // 输入检查
    TORCH_CHECK(se2tTrainX.dim() == 5, "se2tTrainX must be 5D tensor");
    TORCH_CHECK(se2tPredX.dim() == 5, "se2tPredX must be 5D tensor");
    TORCH_CHECK(se2tTrainX.size(4) == 4, "Last dim of se2tTrainX must be 4");
    TORCH_CHECK(se2tPredX.size(4) == 4, "Last dim of se2tPredX must be 4");
    TORCH_CHECK(se2tTrainX.is_cuda() && se2tPredX.is_cuda(), "Inputs must be CUDA tensors");

    // 广播张量
    auto broadcasted = torch::broadcast_tensors({se2tTrainX, se2tPredX});
    auto expanded_train = broadcasted[0];
    auto expanded_pred = broadcasted[1];

    // 准备输出张量
    std::vector<int64_t> output_shape;
    for (int64_t dim : expanded_train.sizes().vec()) output_shape.push_back(dim);
    output_shape[4] = 3; // 最后维度改为3
    auto options = torch::TensorOptions().dtype(expanded_train.dtype()).device(expanded_train.device());
    torch::Tensor se2t_dist = torch::empty(output_shape, options);

    // 计算总元素数（前四维的乘积）
    const int64_t total_elements = expanded_train.size(0) * expanded_train.size(1) * 
                                 expanded_train.size(2) * expanded_train.size(3);

    // 配置CUDA kernel参数
    const int threads_per_block = 256;
    const int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    // 获取张量信息
    const float* train_ptr = expanded_train.data_ptr<float>();
    const float* pred_ptr = expanded_pred.data_ptr<float>();
    float* output_ptr = se2t_dist.data_ptr<float>();

    auto train_strides = expanded_train.strides();
    auto pred_strides = expanded_pred.strides();
    auto output_strides = se2t_dist.strides();

    // 启动kernel
    computeSe2tDistMatKernel<<<num_blocks, threads_per_block>>>(
        train_ptr,
        train_strides[0], train_strides[1], train_strides[2], train_strides[3], train_strides[4],
        pred_ptr,
        pred_strides[0], pred_strides[1], pred_strides[2], pred_strides[3], pred_strides[4],
        output_ptr,
        output_strides[0], output_strides[1], output_strides[2], output_strides[3], output_strides[4],
        expanded_train.size(0), expanded_train.size(1), expanded_train.size(2), expanded_train.size(3)
    );

    return se2t_dist;
}

torch::Tensor computeSe2tDistMat(const torch::Tensor& se2tTrainX, const torch::Tensor& se2tPredX){
  auto slice = torch::indexing::Slice();
  // 定义一个lambda函数来简化索引表达式
  auto makeSlice = [slice](int dim) {
      return std::vector<torch::indexing::TensorIndex>{slice, slice, slice, slice, dim};
  };

  // 计算时间戳的绝对差值
  torch::Tensor timestamp_diff = torch::abs(se2tTrainX.index(makeSlice(0)) - se2tPredX.index(makeSlice(0)));//NOTE: 0.25GB
  // std::cout << "timestamp_diff.data_ptr<float>(): " << timestamp_diff.data_ptr<float>() << std::endl;
  // std::cout << "timestamp_diff.sizes(): " << timestamp_diff.sizes() << std::endl;           

  // 计算yaw的绝对差值，并规范化到[-pi, pi]
  torch::Tensor yaw_diff = torch::abs(se2tTrainX.index(makeSlice(1)) - se2tPredX.index(makeSlice(1)));//NOTE: 0.25GB
  static auto pi = torch::acos(torch::tensor(-1.0));
  yaw_diff = torch::where(torch::abs(yaw_diff) < 100.0, (yaw_diff + pi) % (2 * pi) - pi, yaw_diff);
  // std::cout << "yaw_diff.data_ptr<float>(): " << yaw_diff.data_ptr<float>() << std::endl;
  // std::cout << "yaw_diff.sizes(): " << yaw_diff.sizes() << std::endl;

  // 计算grid位置的欧几里得距离
  // 获取gridX和gridY的差值
  torch::Tensor gridX_diff = se2tTrainX.index(makeSlice(2)) - se2tPredX.index(makeSlice(2));//NOTE: 0.25GB
  torch::Tensor gridY_diff = se2tTrainX.index(makeSlice(3)) - se2tPredX.index(makeSlice(3));//NOTE: 0.25GB
  // std::cout << "gridX_diff.data_ptr<float>(): " << gridX_diff.data_ptr<float>() << std::endl;
  // std::cout << "gridX_diff.sizes(): " << gridX_diff.sizes() << std::endl;
  // std::cout << "gridY_diff.data_ptr<float>(): " << gridY_diff.data_ptr<float>() << std::endl;
  // std::cout << "gridY_diff.sizes(): " << gridY_diff.sizes() << std::endl;

  // 计算欧几里得距离
  torch::Tensor grid_dist = torch::sqrt(gridX_diff.square() + gridY_diff.square());//NOTE: 0.25GB
  // std::cout << "grid_dist.data_ptr<float>(): " << grid_dist.data_ptr<float>() << std::endl;


  // 将三个距离维度组合成最终张量 [31,37,36,1620,3]
  auto se2t_dist = torch::cat({timestamp_diff.unsqueeze(-1), yaw_diff.unsqueeze(-1), grid_dist.unsqueeze(-1)}, /*dim=*/-1);//NOTE: 0.75GB
  // std::cout << "se2t_dist.data_ptr<float>(): " << se2t_dist.data_ptr<float>() << std::endl;

  // std::cout << "se2_dist.sizes(): " << se2t_dist.sizes() << std::endl;
  return se2t_dist;
}

int main() {
    MeasureCUDARuntime timer;

    /////////////////////////////////////////////////////// se2 kernel ////////////////////////////////////////////////////////////
    std::cout << "/////////////////////////////////////////////////////// se2t kernel ////////////////////////////////////////////////////////////" << std::endl;
    // 测试代码
    torch::Tensor se2tDistMat = torch::ones({31, 37, 36, 1260, 3}, torch::kCUDA);
    torch::Tensor kLenTimeYawGrid = torch::ones({3}, torch::kCUDA);

    //PART: CUDA Kernel  
    timer.start();
    torch::Tensor result_cudakernel = computeSe2tKernelOptimized(se2tDistMat, kLenTimeYawGrid);
    std::cout << "==================== CUDA Kernel =====================" << std::endl;
    std::cout << "CUDA kernel execution time: " << timer.end() << " ms" << std::endl;
    std::cout << "result_cudakernel.size(): " << result_cudakernel.sizes() << std::endl;
    std::cout << "computeTensorElemCount(se2tDistMat): " << computeTensorElemCount(se2tDistMat) << std::endl;
    std::cout << "result_cudakernel.abs().sum(): " << result_cudakernel.abs().sum() << std::endl;

    //PART: libtorch version
    timer.start();
    torch::Tensor result_libtorch = computeSe2tKernel(se2tDistMat, kLenTimeYawGrid);
    std::cout << "\n==================== libtorch =====================" << std::endl;
    std::cout << "libtorch execution time: " << timer.end() << " ms" << std::endl;
    std::cout << "result_libtorch.size(): " << result_libtorch.sizes() << std::endl;
    std::cout << "computeTensorElemCount(se2tDistMat): " << computeTensorElemCount(se2tDistMat) << std::endl;
    std::cout << "result_libtorch.abs().sum(): " << result_libtorch.abs().sum() << std::endl;


    std::cout << "/////////////////////////////////////////////////////// se2t distMat ////////////////////////////////////////////////////////////" << std::endl;
    torch::Tensor se2tTrainX = torch::ones({31, 37, 36, 1260, 4}, torch::kCUDA);
    torch::Tensor se2tPredX  = torch::ones({31, 37, 36, 1, 4}, torch::kCUDA) * 2;

    //PART: compute se2t_dist matrix 
    timer.start();
    auto se2t_dist_cuda = computeSe2tDistMatCUDAKernelVersion(se2tTrainX, se2tPredX);
    std::cout << "==================== CUDA Kernel =====================" << std::endl;
    std::cout << "CUDA kernel execution time: " << timer.end() << " ms" << std::endl;
    std::cout << "se2t_dist_cuda.size(): " << se2t_dist_cuda.sizes() << std::endl;


    auto se2t_dist_libtorch = computeSe2tDistMat(se2tTrainX, se2tPredX);
    std::cout << "\n==================== libtorch =====================" << std::endl;
    std::cout << "libtorch execution time: " << timer.end() << " ms" << std::endl;
    std::cout << "se2t_dist_libtorch.size(): " << se2t_dist_libtorch.sizes() << std::endl;

    std::cout << "\n==================== test is equal ====================" << std::endl;
    std::cout << "se2t_dist_cuda.abs().sum(): " << se2t_dist_cuda.abs().sum() << std::endl;
    std::cout << "se2t_dist_libtorch.abs().sum(): " << se2t_dist_libtorch.abs().sum() << std::endl;
    std::cout << "(se2t_dist_libtorch-se2t_dist_cuda).abs().sum(): " << (se2t_dist_libtorch-se2t_dist_cuda).abs().sum() << std::endl;
    
    //NOTE: print some data to check   
    // auto slice03 = torch::indexing::Slice(0, 3);
    // auto slice = torch::indexing::Slice();
    // auto se2t_dist_cuda_slice03 = se2t_dist_cuda.index({slice03, slice03, slice03, slice03, slice});
    // std::cout << "se2t_dist_cuda_slice03.sizes(): " << se2t_dist_cuda_slice03.sizes() << std::endl;
    // std::cout << "se2t_dist_cuda_slice03: \n" << se2t_dist_cuda_slice03 << std::endl;

    // std::cout << "\n---------------------------------------- cutting line ----------------------------------------\n" << std::endl;
    // auto se2t_dist_libtorch_slice03 = se2t_dist_libtorch.index({slice03, slice03, slice03, slice03, slice});
    // std::cout << "se2t_dist_libtorch_slice03.sizes(): " << se2t_dist_libtorch_slice03.sizes() << std::endl;
    // std::cout << "se2t_dist_libtorch_slice03: \n" << se2t_dist_libtorch_slice03 << std::endl;


    
    

    return 0;
}