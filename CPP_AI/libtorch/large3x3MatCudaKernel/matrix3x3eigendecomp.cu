#include "matrix3x3eigendecomp.h"


__device__ void swap(float* a, float* b) {
    float temp = *a;
    *a = *b;
    *b = temp;
}

__device__ void jacobi3x3(const float* A, float* eigVals, float* eigVecs) {
    float V[9] = {1.0f, 0.0f, 0.0f,
                  0.0f, 1.0f, 0.0f,
                  0.0f, 0.0f, 1.0f};
    float B[9];
    for (int i = 0; i < 9; ++i) B[i] = A[i];

    const int maxIter = 50;
    const float epsilon = 1e-6f;

    for (int iter = 0; iter < maxIter; ++iter) {
        int p = 0, q = 1;
        float maxVal = fabsf(B[1]);
        if (fabsf(B[2]) > maxVal) { p = 0; q = 2; maxVal = fabsf(B[2]); }
        if (fabsf(B[5]) > maxVal) { p = 1; q = 2; maxVal = fabsf(B[5]); }

        if (maxVal < epsilon) break;

        float theta = 0.5f * atan2f(2 * B[p*3 + q], B[q*3 + q] - B[p*3 + p]);
        float c = cosf(theta);
        float s = sinf(theta);

        float Bpp = B[p*3 + p];
        float Bqq = B[q*3 + q];
        float Bpq = B[p*3 + q];

        B[p*3 + p] = c*c*Bpp + s*s*Bqq - 2*c*s*Bpq;
        B[q*3 + q] = s*s*Bpp + c*c*Bqq + 2*c*s*Bpq;
        B[p*3 + q] = 0.0f;
        B[q*3 + p] = 0.0f;

        for (int r = 0; r < 3; ++r) {
            if (r != p && r != q) {
                float Brp = B[r*3 + p];
                float Brq = B[r*3 + q];
                B[r*3 + p] = c*Brp - s*Brq;
                B[p*3 + r] = B[r*3 + p];
                B[r*3 + q] = s*Brp + c*Brq;
                B[q*3 + r] = B[r*3 + q];
            }
        }

        for (int r = 0; r < 3; ++r) {
            float Vrp = V[r*3 + p];
            float Vrq = V[r*3 + q];
            V[r*3 + p] = c*Vrp - s*Vrq;
            V[r*3 + q] = s*Vrp + c*Vrq;
        }
    }

    eigVals[0] = B[0];
    eigVals[1] = B[4];
    eigVals[2] = B[8];

    if (eigVals[0] < eigVals[1]) {
        swap(&eigVals[0], &eigVals[1]);
        for (int r = 0; r < 3; ++r) swap(&V[r*3 + 0], &V[r*3 + 1]);
    }
    if (eigVals[0] < eigVals[2]) {
        swap(&eigVals[0], &eigVals[2]);
        for (int r = 0; r < 3; ++r) swap(&V[r*3 + 0], &V[r*3 + 2]);
    }
    if (eigVals[1] < eigVals[2]) {
        swap(&eigVals[1], &eigVals[2]);
        for (int r = 0; r < 3; ++r) swap(&V[r*3 + 1], &V[r*3 + 2]);
    }

    for (int i = 0; i < 9; ++i) eigVecs[i] = V[i];
}

__global__ void eigenDecompositionKernel(const float* matrices, float* eigenvalues, float* eigenvectors, int numMatrices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numMatrices) return;

    float matrix[9];
    for (int i = 0; i < 9; ++i) matrix[i] = matrices[idx*9 + i];

    float eigVals[3];
    float eigVecs[9];
    jacobi3x3(matrix, eigVals, eigVecs);

    for (int i = 0; i < 3; ++i) eigenvalues[idx*3 + i] = eigVals[i];
    for (int i = 0; i < 9; ++i) eigenvectors[idx*9 + i] = eigVecs[i];
}



void eigenDecompositionLauncher(const float* matrices, float* eigenvalues, float* eigenvectors, int numMatrices) {
    const int blockSize = 256;
    const int gridSize = (numMatrices + blockSize - 1) / blockSize;
    eigenDecompositionKernel<<<gridSize, blockSize>>>(matrices, eigenvalues, eigenvectors, numMatrices);
}

/**
 * @brief Launches the eigen decomposition kernel for a batch of 3x3 matrices.
 * @attention: 1. torch::Tensor should be contiguous. 
 * 2. The matrices should be torch::Float32 Dtype
 * 3. The tensors must be at torch::kCUDA device.
 * 4. The matrix's dim only support 3x3 shape
 * 5. The result eigenvalues large -> small
 */
void eigenDecompositionLauncher(const torch::Tensor matrices, torch::Tensor eigenvalues, torch::Tensor eigenvectors, const int blockSize) {
    int numMatrices = matrices.size(0);
    const int gridSize = (numMatrices + blockSize - 1) / blockSize;
    const float* matrices_ptr = matrices.data_ptr<float>();
    float* eigenvalues_ptr = eigenvalues.data_ptr<float>();
    float* eigenvectors_ptr = eigenvectors.data_ptr<float>();
    eigenDecompositionKernel<<<gridSize, blockSize>>>(matrices_ptr, eigenvalues_ptr, eigenvectors_ptr, numMatrices);
}