#ifndef MATRIX3X3EIGENDECOMP_H
#define MATRIX3X3EIGENDECOMP_H

#include <math.h>
#include <cuda_runtime.h>
#include "torch/torch.h"



void eigenDecompositionLauncher(const float* matrices, float* eigenvalues, float* eigenvectors, int numMatrices);
void eigenDecompositionLauncher(const torch::Tensor matrices, torch::Tensor eigenvalues, torch::Tensor eigenvectors, const int blockSize = 256);

#endif // MATRIX3X3EIGENDECOMP_H