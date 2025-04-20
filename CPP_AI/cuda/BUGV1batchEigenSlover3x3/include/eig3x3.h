/*
 * @Author: chasey && melancholycy@gmail.com
 * @Date: 2025-04-20 05:01:38
 * @LastEditTime: 2025-04-20 05:01:45
 * @FilePath: /test/CPP_AI/cuda/batchEigenSlover3x3/include/eig3x3.h
 * @Description: 
 * @Reference: 
 * Copyright (c) 2025 by chasey && melancholycy@gmail.com, All Rights Reserved. 
 */
#pragma once
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

void batchedEigen3x3(
    const float* d_input, 
    float* d_eigenvalues,
    float* d_eigenvectors,
    int num_matrices,
    cudaStream_t stream = 0
);

#ifdef __cplusplus
}
#endif