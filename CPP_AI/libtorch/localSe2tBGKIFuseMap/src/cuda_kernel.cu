#include "cuda_kernel.h"

//PART: ========== CUDA Kernel Function ==========
__global__ void computeSe2tCovSparseKernel(
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

//========== Interface function ==========
void computeSe2tCovSparseKernelInterface(
    const float* se2tDistMat,
    const float* kLen,
    float* output,
    int num_elements,
    int num_channels,
    int blocks,
    int threads_per_block
){
    //begin cuda kernel
    computeSe2tCovSparseKernel<<<blocks, threads_per_block>>>(
        se2tDistMat,
        kLen,
        output,
        num_elements,
        num_channels
    );
}

void computeSe2tDistMatKernelInterface(
    const float* train_data,
    int train_stride0, int train_stride1, int train_stride2, int train_stride3, int train_stride4,
    const float* pred_data,
    int pred_stride0, int pred_stride1, int pred_stride2, int pred_stride3, int pred_stride4,
    float* output_data,
    int output_stride0, int output_stride1, int output_stride2, int output_stride3, int output_stride4,
    int s0, int s1, int s2, int s3,
    int blocks, int threads_per_block
){
    //begin cuda kernel
    computeSe2tDistMatKernel<<<blocks, threads_per_block>>>(
        train_data,
        train_stride0, train_stride1, train_stride2, train_stride3, train_stride4,
        pred_data,
        pred_stride0, pred_stride1, pred_stride2, pred_stride3, pred_stride4,
        output_data,
        output_stride0, output_stride1, output_stride2, output_stride3, output_stride4,
        s0,s1,s2,s3
    );
}

