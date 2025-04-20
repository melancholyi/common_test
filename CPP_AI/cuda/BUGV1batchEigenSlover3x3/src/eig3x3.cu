#include "eig3x3.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <device_launch_parameters.h>

__device__ void computeEigenvectors(const float* m, float* vecs, const float* vals) {
    // 简化的特征向量计算（完整实现需要更多数学处理）
    for(int i = 0; i < 3; ++i) {
        const float a = m[0] - vals[i];
        const float b = m[1];
        const float c = m[2];
        const float d = m[4] - vals[i];
        const float e = m[5];
        
        // 计算叉乘作为特征向量
        vecs[i*3] = b*e - c*d;
        vecs[i*3+1] = c*a - e*0;
        vecs[i*3+2] = a*d - b*b;
        
        // 归一化
        float norm = sqrtf(vecs[i*3]*vecs[i*3] + 
                    vecs[i*3+1]*vecs[i*3+1] + 
                    vecs[i*3+2]*vecs[i*3+2]);
        if(norm > 1e-6f) {
            vecs[i*3] /= norm;
            vecs[i*3+1] /= norm;
            vecs[i*3+2] /= norm;
        }
    }
}


__global__ void eig3x3Kernel(
    const float* input,
    float* eigenvalues,
    float* eigenvectors,
    int num_matrices)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= num_matrices) return;

    const float* m = &input[idx * 6]; // 只存储上三角部分
    float* vals = &eigenvalues[idx * 3];
    float* vecs = &eigenvectors[idx * 9];

    // 展开对称矩阵元素
    const float a = m[0], b = m[1], c = m[2];
    const float d = m[3], e = m[4];
    const float f = m[5];

    // 计算特征值
    const float tr = a + d + f;
    const float tr2 = a*a + d*d + f*f + 2*(b*b + c*c + e*e);
    const float det = a*(d*f - e*e) - b*(b*f - c*e) + c*(b*e - c*d);
    
    const float p = (tr2 - tr*tr)/6.0f;
    const float q = (-tr*tr*tr + 4.5f*tr*(tr2 - tr*tr) - 13.5f*det)/27.0f;
    const float sqrt_p = sqrtf(fabsf(p));
    const float theta = acosf(q / (p*sqrt_p)) / 3.0f;
    
    vals[0] = tr/3.0f + 2*sqrt_p*cosf(theta);
    vals[1] = tr/3.0f - sqrt_p*(cosf(theta) + sqrtf(3)*sinf(theta));
    vals[2] = tr/3.0f - sqrt_p*(cosf(theta) - sqrtf(3)*sinf(theta));

    // 计算特征向量
    computeEigenvectors(m, vecs, vals);
}


// __device__ void computeEigenvectors(const float* m, float* vecs, const float* vals) {
//     for(int i = 0; i < 3; ++i) {
//         const float lambda = vals[i];
        
//         // 构造矩阵 (A - λI)
//         const float a = m[0] - lambda;  // [0] a11
//         const float b = m[1];           // [1] a12
//         const float c = m[2];           // [2] a13
//         const float d = m[3];           // [3] a21
//         const float e = m[4] - lambda;  // [4] a22
//         const float f = m[5];           // [5] a23
//         const float g = m[6];           // [6] a31
//         const float h = m[7];           // [7] a32
//         const float k = m[8] - lambda;  // [8] a33

//         // 计算叉乘作为特征向量
//         vecs[i*3]   = b*f - c*e;  // 第一行余子式
//         vecs[i*3+1] = c*d - a*f;  // 第二行余子式
//         vecs[i*3+2] = a*e - b*d;  // 第三行余子式

//         // 归一化
//         float norm = sqrtf(vecs[i*3]*vecs[i*3] + 
//                           vecs[i*3+1]*vecs[i*3+1] + 
//                           vecs[i*3+2]*vecs[i*3+2]);
//         if(norm > 1e-6f) {
//             vecs[i*3]   /= norm;
//             vecs[i*3+1] /= norm;
//             vecs[i*3+2] /= norm;
//         }
//     }
// }



// __global__ void eig3x3Kernel(
//     const float* input,
//     float* eigenvalues,
//     float* eigenvectors,
//     int num_matrices)
// {
//     const int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if(idx >= num_matrices) return;

//     const float* m = &input[idx * 9];  // 读取完整9元素
//     float* vals = &eigenvalues[idx * 3];
//     float* vecs = &eigenvectors[idx * 9];

//     // 计算对称矩阵的特征值（A = (M + M^T)/2）
//     const float a = (m[0] + m[0])/2;  // a11
//     const float b = (m[1] + m[3])/2;  // a12
//     const float c = (m[2] + m[6])/2;  // a13
//     const float d = (m[4] + m[4])/2;  // a22
//     const float e = (m[5] + m[7])/2;  // a23
//     const float f = (m[8] + m[8])/2;  // a33

//     // 特征方程参数
//     const float tr = a + d + f;
//     const float tr2 = a*a + d*d + f*f + 2*(b*b + c*c + e*e);
//     const float det = a*(d*f - e*e) - b*(b*f - c*e) + c*(b*e - c*d);

//     // 求解三次方程
//     const float p = (tr2 - tr*tr)/6.0f;
//     const float q = (-tr*tr*tr + 4.5f*tr*(tr2 - tr*tr) - 13.5f*det)/27.0f;
//     const float sqrt_p = sqrtf(fabsf(p));
//     const float theta = acosf(q / (p*sqrt_p)) / 3.0f;
    
//     vals[0] = tr/3.0f + 2*sqrt_p*cosf(theta);
//     vals[1] = tr/3.0f - sqrt_p*(cosf(theta) + sqrtf(3)*sinf(theta));
//     vals[2] = tr/3.0f - sqrt_p*(cosf(theta) - sqrtf(3)*sinf(theta));

//     computeEigenvectors(m, vecs, vals);
// }



void batchedEigen3x3(
    const float* d_input,
    float* d_eigenvalues,
    float* d_eigenvectors,
    int num_matrices,
    cudaStream_t stream)
{
    const int blockSize = 256;
    const int gridSize = (num_matrices + blockSize - 1) / blockSize;
    
    eig3x3Kernel<<<gridSize, blockSize, 0, stream>>>(
        d_input, d_eigenvalues, d_eigenvectors, num_matrices
    );
}