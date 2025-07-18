/*
 * @Author: chasey && melancholycy@gmail.com
 * @Date: 2025-05-15 12:24:49
 * @LastEditTime: 2025-05-17 03:38:48
 * @FilePath: /test/CPP_AI/libtorch/localSe2tBGKIFuseMap/include/cuda_kernel.h
 * @Description: 
 * @Reference: 
 * Copyright (c) 2025 by chasey && melancholycy@gmail.com, All Rights Reserved. 
 */
#pragma once
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

void computeSe2tCovSparseKernelLancher(
    const float* se2tDistMat,
    const float* kLen,
    float* output,
    int num_elements,
    int num_channels,
    int blocks,
    int threads_per_block
);

void computeSe2tDistMatKernelLancher(
    const float* train_data,
    int train_stride0, int train_stride1, int train_stride2, int train_stride3, int train_stride4,
    const float* pred_data,
    int pred_stride0, int pred_stride1, int pred_stride2, int pred_stride3, int pred_stride4,
    float* output_data,
    int output_stride0, int output_stride1, int output_stride2, int output_stride3, int output_stride4,
    int s0, int s1, int s2, int s3,
    int blocks, int threads_per_block);

void computeYbarKbarLancher(
    float* se2tKernel,
    const float* se2tTrainSigma2,
    const float* se2tTrainY,
    const float* newSe2Info,
    float* kbar,
    float* ybar,
    float varianceInit,
    float delta,
    int dim_i, int dim_j, int dim_k, int dim_l, int dim_c);

#ifdef __cplusplus
}
#endif