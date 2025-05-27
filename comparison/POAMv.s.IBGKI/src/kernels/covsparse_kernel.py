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
import matplotlib.pyplot as plt


class CovSparseKernel2D(gpytorch.kernels.Kernel):
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

    # def __process_cdist1(self, input_tensor):
    #     # 将 input_tensor 复制一份，避免修改原始张量
    #     processed_tensor = input_tensor.clone()
    #     # 使用 while 循环，检查每行是否满足条件
    #     # 为了演示效果，这里只执行一次，实际使用中可以根据需要调整循环条件
    #     times = torch.ones(input_tensor.shape[0], dtype=torch.int, device=input_tensor.device)  # 初始化一个布尔张量
    #     while True:
    #         all_greater_than_1 = torch.all(processed_tensor > 1.0, dim=1)
    #         times[all_greater_than_1] += 1
    #         # 检查是否存在满足条件的行
    #         if not torch.any(all_greater_than_1):
    #             break
    #         # 将满足条件的行元素除以 2
    #         processed_tensor[all_greater_than_1] /= 2
    #         # print(f'whiling processed_tensor:\n{processed_tensor}')
    #     return processed_tensor

    def __process_cdist1(self, input_tensor):
        # 将 input_tensor 复制一份，避免修改原始张量
        processed_tensor = input_tensor.clone()
        # 使用 while 循环，检查每行是否满足条件
        # 为了演示效果，这里只执行一次，实际使用中可以根据需要调整循环条件
        times = torch.ones(input_tensor.shape[0], dtype=torch.int, device=input_tensor.device)  # 初始化一个张量
        while True:
            all_greater_than_1 = torch.all(processed_tensor > 1.0, dim=1)
            times[all_greater_than_1] += 1
            # 检查是否存在满足条件的行
            if not torch.any(all_greater_than_1):
                break
            # 将满足条件的行元素除以 2
            processed_tensor[all_greater_than_1] /= 2
            # print(f'whiling processed_tensor:\n{processed_tensor}')
        # 将times转换为50x50数据
        if times.numel() == 2500:
            times_50x50 = times[:50*50].view(50, 50).cpu().numpy()  # 取前2500个元素并reshape为50x50
            # 存储为txt文件
            np.savetxt("times_data.txt", times_50x50, fmt="%d")
            # 可视化
            plt.figure(figsize=(10, 8))
            plt.imshow(times_50x50, cmap='viridis')
            plt.colorbar(label='Times')
            plt.title('Times Visualization')
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            # 保存可视化结果到文件
            plt.savefig("times_visualization.png")
            plt.close()

        # offset 
        threshold = 0.8
        all_greater_than_0dot9 = torch.all(processed_tensor > threshold, dim=1)
        processed_tensor[all_greater_than_0dot9] *= threshold  # 将满足条件的行元素减去 0.2
        return processed_tensor

    def __process_cdist2(self, cdist: Tensor, threshold = 1.0, eps = 1e-6) -> Tensor:
        """
        while: cdist_line[where cdist > 1.0] /= 2
        """
        # 计算每行的最小值
        min_vals = cdist.min(dim=1).values
        N, M = cdist.shape
        print(f'M: {M}')
        required_count = np.ceil(threshold * M)  # 计算所需阈值
        # 统计每行大于1的元素数量
        count_over_1 = (cdist > 1).sum(dim=1)
        mask = count_over_1 >= required_count  # 标记需处理的行
        if not mask.any():
            return cdist  # 若无行需处理，直接返回
        print(f'mask.shape: {mask.shape}')  # Debugging line
        # # 确定需要处理的行（所有元素>1）
        # mask = min_vals > 1

        # 计算log2并取整得到最大次数
        log_vals = torch.log2(torch.clamp(min_vals - eps, min=eps))  # 避免对数非正数
        k_max = torch.floor(log_vals)
        # 确定最终除数次数（k_max + 1）
        k_total = torch.where(mask, k_max + 1, torch.zeros_like(k_max))
        # 计算除数并扩展维度以便广播
        divisors = 2 ** k_total
        divisors_expanded = divisors.view(-1, 1)
        # 执行除法
        result = cdist / divisors_expanded
        return result
    
    def __process_cdist3(self, cdist: Tensor, eps = 1e-6) -> Tensor:
        """
        while: cdist_line[where cdist > 1.0] /= 2
        """
        # 计算每行的最小值
        min_vals = cdist.min(dim=1).values

        ################################ 
        # 这里是对所有满足条件(行所有元素均>1)的行进行处理，就是一直除以2，直到至少有一个元素<=1
        # 目的是为了处理local核特点，核以外的所有点权重都是0,但是探索初期仅有少量点，可能会出现所有点的距离都大于1.0的情况，所以
        # 当预测点的核附近没有点时，所有点的距离都大于1.0，这样会导致核值为0，无法进行预测。
        # 这种情况下，就是将此处的核长度扩倍，如果依旧没有点在核内，则继续扩倍，直到至少有一个点在核内。
        # 虽然这样预测值的均值非常不稳定，但是同时也会有一个方差估计，可以用于评估多么不稳定以供参考，可用于information gain的计算。
        ################################
        # 确定需要处理的行（所有元素>1）
        mask = min_vals > 1
        # 计算log2并取整得到最大次数
        log_vals = torch.log2(torch.clamp(min_vals - eps, min=eps))  # 避免对数非正数
        k_max = torch.floor(log_vals)
        # 确定最终除数次数（k_max + 1）
        k_total = torch.where(mask, k_max + 1, torch.zeros_like(k_max))
        # 计算除数并扩展维度以便广播
        divisors = 2 ** k_total
        divisors_expanded = divisors.view(-1, 1)
        # 执行除法
        result = cdist / divisors_expanded

        ################################ 
        # 这里是为了进一步处理，初期时候数据实在太少，就算保证至少有一个点在核内部，但是依旧不够
        # 因此这里进一步进行更加严格的处理，在扩大一倍核长度
        min_vals2 = result.min(dim=1).values
        mask2 = min_vals2 > 0.5

        result[mask2] -= 0.2  # 将满足条件的行元素除以 2

        return result
    
    def __process_cdist4(self, cdist: Tensor, gamma=0.9) -> Tensor:
        device = cdist.device
        # 确定需要处理的行（所有元素>1）
        row_mask = (cdist > 1).all(dim=1)
        if not row_mask.any():
            return cdist  # 无行需要处理
        
        selected_rows = cdist[row_mask]  # 提取符合条件的行
        
        # 计算每个元素的衰减次数k_ij
        log_gamma = torch.log(torch.tensor(gamma, device=device))
        k_ij = torch.ceil(torch.log(1.0 / selected_rows) / log_gamma)
        
        # 取每行的最小衰减次数
        k_i = k_ij.min(dim=1, keepdim=True).values
        
        # 计算衰减因子并应用
        scale_factors = gamma ** k_i.float()
        scaled_rows = selected_rows * scale_factors
        
        # 更新原始张量
        cdist[row_mask] = scaled_rows
        return cdist

    def __process_cdist5(self, cdist: Tensor, klen: Tensor) -> Tensor:

        # 假设cdist是一个[N, M]的张量，klen是一个[N]的张量
        original_cdist = cdist.clone()  # 保存原始cdist
        klen_init = klen.clone()  # 初始klen
        index = torch.zeros_like(klen, dtype=torch.int)  # 记录每行的处理次数

        while True:
            # 计算当前cdist
            current_cdist = original_cdist / klen.unsqueeze(1)
            # 检查哪些行所有元素都大于1
            mask = (current_cdist > 1).all(dim=1)
            
            # 如果没有满足条件的行，退出循环
            if not mask.any():
                break
            
            # 获取满足条件的行索引
            rows = mask.nonzero().squeeze(dim=1)
            
            # 计算增量：klen[rows] / (2^(index[rows] + 1))
            increments = klen_init[rows] / (2 ** (index[rows] + 1))
            
            # 更新klen和index
            klen[rows] += increments
            index[rows] += 1
        return original_cdist / klen.unsqueeze(1)

        
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

        # TODO: while cdist[where cdist > 1.0] /=2
        # cdist = self.__process_cdist(cdist)  # Process cdist to ensure every line has at least one element <= 1.0
        # cdist = self.__process_cdist5(cdist, klen=klens_extracted)  # Process cdist to ensure every line has at least one element <= 1.0
        cdist = self.__process_cdist1(cdist)

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
    kernel = CovSparseKernel2D(
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
    

