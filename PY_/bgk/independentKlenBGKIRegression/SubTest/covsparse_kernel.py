# '''
# Author: chasey && melancholycy@gmail.com
# Date: 2025-05-20 09:00:49
# LastEditTime: 2025-05-21 07:15:05
# FilePath: /POAM/src/kernels/covsparse_kernel.py
# Description: ATTENTION: only support 2D Grid Dataset
# Reference: 
# Copyright (c) 2025 by chasey && melancholycy@gmail.com, All Rights Reserved. 
# '''

from typing import Union

import gpytorch
import torch
import torch.nn as nn
from linear_operator.operators import LinearOperator
from torch import Tensor
import numpy as np


class covSparseKernel2D(gpytorch.kernels.Kernel):
    M2PI = 2 * torch.pi
    def __init__(
        self,
        kLenMat: Tensor,
        kScaleMat: Tensor,
        minX: float,
        minY: float,
        resolution: float,  
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        # self.kLenMat_ = kLenMat
        # self.kScaleMat_ = kScaleMat
        self.register_parameter(name="kLenMat_", parameter=nn.Parameter(kLenMat))
        self.register_parameter(name="kScaleMat_", parameter=nn.Parameter(kScaleMat))
        self.minX_ = minX
        self.minY_ = minY
        self.resolution_ = resolution
        
    def forward(
        self,
        x1: Tensor, 
        x2: Tensor, # [M, D], here D = 2
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params
    ) -> Union[Tensor, LinearOperator]: 
        """
        param:  x1 Tensor, shape:[N, D], here D = 2
        param:  x2 Tensor, shape:[M, D], here D = 2
        return: covKernel, shape:[N, M]
        """
        # Dimension check for x1
        if len(x1.shape) != 2 or x1.shape[1] != 2:
            raise ValueError(f"x1 must be a 2D tensor with shape [N, 2]. Got {x1.shape}")
        
        # Dimension check for x2
        if len(x2.shape) != 2 or x2.shape[1] != 2:
            raise ValueError(f"x2 must be a 2D tensor with shape [M, 2]. Got {x2.shape}")
        
        cdist = self.covar_dist(x1, x2, **params) # [N, M]
        
        idxs_x = ((x1[:, 0] - self.minX_) / self.resolution_).long()
        idxs_y = ((x1[:, 1] - self.minY_) / self.resolution_).long()
        idxs_x = np.clip(idxs_x, 0, self.kLenMat_.size(0) - 1) # limit idxs_x to valid range[0, len(x)-1]
        idxs_y = np.clip(idxs_y, 0, self.kLenMat_.size(1) - 1) # limit idxs_y to valid range[0, len(y)-1]
        klens_extracted = self.kLenMat_[idxs_x, idxs_y] # shape:torch.Size([len(x)])
        kscale_extracted = self.kScaleMat_[idxs_x, idxs_y] # shape:torch.Size([len(x)])
        
        cdist/=klens_extracted.unsqueeze(1) # [N, M]
        kernel = ((2 + (cdist * self.M2PI).cos()) * (1 - cdist) / 3.0 + (cdist * self.M2PI).sin() / self.M2PI) * kscale_extracted.unsqueeze(1) # [N, M]
        kernel.clamp_min_(0.0)
        return kernel



if __name__ == "__main__":
    # --- Parameters ---
    # torch::randn 标准正态分布（均值为 0，标准差为 1）
    kLenMat = torch.randn(5, 5).abs().mul(5).requires_grad_(True)    # 2D length scale matrix
    kScaleMat = torch.randn(5, 5).abs().mul(5).requires_grad_(True)  
    minX, minY = 0.0, 0.0  # Minimum coordinates
    resolution = 1.0  # Grid resolution (unused in this test)
    

    kLenMat_origin = kLenMat.clone()
    kScaleMat_origin = kScaleMat.clone()

    # --- Create Kernel Instance ---
    kernel = covSparseKernel2D(
        kLenMat=kLenMat,
        kScaleMat=kScaleMat,
        minX=minX,
        minY=minY,
        resolution=resolution
    )
    print(f'kernel.parameters():\n{kernel.parameters()}')
    for name, param in kernel.named_parameters():
        print(f"Parameter name: {name}")
        print(f"Shape: {param.shape}")
        print(f"Values :\n{param.data}\n")  # 打印前3x3的数值示例
    """
    Shape: torch.Size([5, 5])
    Values :
    tensor([[6.6539, 1.1806, 0.5332, 3.2024, 4.9376],
            [0.8231, 3.5646, 1.2104, 3.8079, 1.7146],
            [6.3919, 8.9196, 2.2859, 3.1165, 1.1214],
            [1.3693, 4.8413, 3.5816, 2.0752, 3.6394],
            [2.4428, 6.9549, 8.9536, 4.7128, 1.1083]])

    Parameter name: kScaleMat_
    Shape: torch.Size([5, 5])
    Values :
    tensor([[2.2723, 0.3439, 3.7777, 7.8296, 4.3742],
            [1.0249, 0.9795, 5.8153, 6.1164, 3.3170],
            [8.9017, 1.6697, 7.0586, 2.7283, 2.4702],
            [4.5605, 2.5372, 2.6443, 0.5061, 2.4531],
            [4.6759, 0.0318, 2.6207, 8.1844, 4.5276]])
    """


    optimizer = torch.optim.SGD(kernel.parameters(), lr=0.01)

    # --- Sample Input Points ---
    x1 = torch.tensor([[0.0, 0.0], [1.5, 1.5], [2.5, 2.5], [3.5, 3.5], [4.5, 4.5]]) 
    x2 = torch.tensor([[1.0, 1.0], [2.0, 2.0]])

    # --- Compute Distance Matrix ---
    kernel_matrix = kernel(x1, x1)

    # --- Print Result ---
    print("Distance Matrix (Squared Euclidean Distances):")
    dense_kernel_matrix = kernel_matrix.evaluate()  # Convert to dense tensor
    print(dense_kernel_matrix)

    # 计算损失函数，例如tensor2的均值
    loss = dense_kernel_matrix.sum()*100**2  # Sum all elements of the dense tensor

    # 反向传播计算梯度
    loss.backward()

    # 执行优化步骤来更新tensor1的值
    optimizer.step()
    print(f'\n========== Bactward Over ==========')
    print(f'kLenMat change value:\n{kLenMat - kLenMat_origin}')
    print(f'kScaleMat change value:\n{kScaleMat - kScaleMat_origin}')


    print(f'kernel.parameters():\n{kernel.parameters()}')
    for name, param in kernel.named_parameters():
        print(f"Parameter name: {name}")
        print(f"Shape: {param.shape}")
        print(f"Values :\n{param.data}\n")  # 打印前3x3的数值示例
    

