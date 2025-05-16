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

__global__ void fused_bgki_kernel(
    float* se2tKernel,
    const float* se2tTrainSigma2,
    const float* se2tTrainY,
    const float* se2tInfo,
    float* kbar,
    float* ybar,
    float varianceInit,
    float delta,
    int dim_i, int dim_j, int dim_k, int dim_l, int dim_c) 
{
    // 三维索引计算
    const int i = blockIdx.z;
    const int j = blockIdx.y;
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= dim_i || j >= dim_j || k >= dim_k) return;

    // 第一步：处理除法操作 (in-place)
    for(int l=0; l<dim_l; ++l){
        const int kernel_idx = ((i * dim_j + j) * dim_k + k) * dim_l + l;
        se2tKernel[kernel_idx] /= se2tTrainSigma2[kernel_idx];
    }

    // 第二步：计算kbar
    float ksum = 0.0f;
    for(int l=0; l<dim_l; ++l){
        const int kernel_idx = ((i * dim_j + j) * dim_k + k) * dim_l + l;
        ksum += se2tKernel[kernel_idx];
    }
    const float kbar_val = ksum + 1.0f / varianceInit;
    kbar[(i * dim_j + j) * dim_k + k] = kbar_val;

    // 第三步：计算ybar
    for(int c=0; c<dim_c; ++c){
        float ysum = 0.0f;
        for(int l=0; l<dim_l; ++l){
            const int kernel_idx = ((i * dim_j + j) * dim_k + k) * dim_l + l;
            const int y_idx = (((i * dim_j + j) * dim_k + k) * dim_l + l) * dim_c + c;
            ysum += se2tTrainY[y_idx] * se2tKernel[kernel_idx];
        }
        
        const int mu_idx = ((i * dim_j + j) * dim_k + k) * 4 + c; // 假设se2tInfo最后一维>=dim_c
        ysum += se2tInfo[mu_idx] / varianceInit;
        
        const int ybar_idx = ((i * dim_j + j) * dim_k + k) * dim_c + c;
        ybar[ybar_idx] = ysum / (kbar_val + delta);
    }

    // 第四步：kbar取倒数 (in-place)
    kbar[(i * dim_j + j) * dim_k + k] = 1.0f / kbar_val;
}

template <unsigned BLOCK_SIZE>
__global__ void computeYbarKbarFusedKernel(
    float* se2tKernel,
    const float* se2tTrainSigma2,
    const float* se2tTrainY,
    const float* newSe2Info,
    float* kbar,
    float* ybar,
    float varianceInit,
    float delta,
    int dim_i, int dim_j, int dim_k, int dim_l, int dim_c) 
{
    // 共享内存存储中间结果
    __shared__ float smem_kbar[BLOCK_SIZE];
    __shared__ float smem_y[BLOCK_SIZE * 3]; // 假设dim_c=3

    // 三维索引计算
    const int global_idx = blockIdx.x;
    const int i = global_idx / (dim_j * dim_k);
    const int remainder = global_idx % (dim_j * dim_k);
    const int j = remainder / dim_k;
    const int k = remainder % dim_k;

    // 线程局部存储
    float local_kbar = 0.0f;
    float local_y[3] = {0.0f};

    // 遍历所有l维度
    for (int l_base = 0; l_base < dim_l; l_base += BLOCK_SIZE) {
        const int l = l_base + threadIdx.x;
        if (l < dim_l) {
            // 计算全局索引
            const int kernel_idx = ((i * dim_j + j) * dim_k + k) * dim_l + l;
            
            // 执行除法并更新全局内存
            const float sigma2 = se2tTrainSigma2[kernel_idx];
            const float kernel_val = se2tKernel[kernel_idx] / sigma2;
            se2tKernel[kernel_idx] = kernel_val;  // 原位更新

            // 累加kbar
            local_kbar += kernel_val;

            // 累加ybar
            for (int c = 0; c < dim_c; ++c) {
                const int y_idx = (((i * dim_j + j) * dim_k + k) * dim_l + l) * dim_c + c;
                local_y[c] += se2tTrainY[y_idx] * kernel_val;
            }
        }
    }

    // 块内归约
    smem_kbar[threadIdx.x] = local_kbar;
    for (int c = 0; c < dim_c; ++c) {
        smem_y[threadIdx.x * dim_c + c] = local_y[c];
    }
    __syncthreads();

    // 并行归约 (假设BLOCK_SIZE为2的幂)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem_kbar[threadIdx.x] += smem_kbar[threadIdx.x + s];
            for (int c = 0; c < dim_c; ++c) {
                smem_y[threadIdx.x * dim_c + c] += smem_y[(threadIdx.x + s) * dim_c + c];
            }
        }
        __syncthreads();
    }

    // 最终结果计算
    if (threadIdx.x == 0) {
        // 计算kbar
        const float kbar_val = 1.0f / (smem_kbar[0] + 1.0f/varianceInit + delta);
        kbar[global_idx] = kbar_val;

        // 计算ybar
        for (int c = 0; c < dim_c; ++c) {
            const int info_idx = ((i * dim_j + j) * dim_k + k) * dim_c + c;
            ybar[info_idx] = (smem_y[c] + newSe2Info[info_idx]/varianceInit) * kbar_val;
        }
    }
}


//========== Interface function ==========
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


// CUDA Kernel 3: Fused kernel for kbar and ybar computation
void computeYbarKbarInterface(
    float* se2tKernel,
    const float* se2tTrainSigma2,
    const float* se2tTrainY,
    const float* newSe2Info,
    float* kbar,
    float* ybar,
    float varianceInit,
    float delta,
    int dim_i, int dim_j, int dim_k, int dim_l, int dim_c)
{
    const int num_blocks = dim_i * dim_j * dim_k;
    const int block_size = 256;
    
    computeYbarKbarFusedKernel<256><<<num_blocks, block_size>>>(
        se2tKernel, se2tTrainSigma2, se2tTrainY, newSe2Info,
        kbar, ybar, varianceInit, delta,
        dim_i, dim_j, dim_k, dim_l, dim_c);
}