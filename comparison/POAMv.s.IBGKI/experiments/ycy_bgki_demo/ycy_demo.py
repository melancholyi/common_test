'''
Author: chasey && melancholycy@gmail.com
Date: 2025-05-22 04:24:31
LastEditTime: 2025-05-23 09:03:01
FilePath: /POAM/experiments/ycy_bgki_demo/ycy_demo.py
Description: 
Reference: 
Copyright (c) 2025 by chasey && melancholycy@gmail.com, All Rights Reserved. 
'''

import src
import numpy as np
import torch



if __name__ == "__main__":
    # Example usage
    x_train = np.random.rand(100, 2) * 5  # 100 samples, 2 features
    y_train = np.random.rand(100, 1) * 5 # 100 samples, 1 target

    # # 提取每列的统计信息
    # # 第一列
    # col1 = x_train[:, 0]
    # min_col1 = np.min(col1)
    # max_col1 = np.max(col1)
    # mean_col1 = np.mean(col1)
    # var_col1 = np.var(col1)

    # # 第二列
    # col2 = x_train[:, 1]
    # min_col2 = np.min(col2)
    # max_col2 = np.max(col2)
    # mean_col2 = np.mean(col2)
    # var_col2 = np.var(col2)

    # # 打印结果
    # print("第一列统计信息：")
    # print(f"  最小值：{min_col1:.4f}")
    # print(f"  最大值：{max_col2:.4f}")
    # print(f"  均值：{mean_col1:.4f}")
    # print(f"  方差：{var_col1:.4f}\n")

    # print("第二列统计信息：")
    # print(f"  最小值：{min_col2:.4f}")
    # print(f"  最大值：{max_col2:.4f}")
    # print(f"  均值：{mean_col2:.4f}")
    # print(f"  方差：{var_col2:.4f}")

    # print(f'x_train:\n', x_train)
    # print(f'y_train:\n', y_train)


    kLenMat = torch.randn(5, 5).abs().mul(5).requires_grad_(True)    # 2D length scale matrix
    kScaleMat = torch.randn(5, 5).abs().mul(5).requires_grad_(True)  
    minX, minY = 0.0, 0.0  # Minimum coordinates
    resolution = 1.0  # Grid resolution (unused in this test)

    kernel = src.kernels.CovSparseKernel2D(kLenMat=kLenMat, kScaleMat=kScaleMat, minX=minX, minY=minY, resolution=resolution)  # Example kernel
    model = src.models.IndependentBGKIModel(x_train, y_train, kernel)
     
    model.add_data(x_train, y_train)
    model.add_data(x_train, y_train)

    x_pred = np.random.rand(10, 2)  # 10 samples, 2 features
    mu, Sigma2 = model.predict(x_pred)
    print(f'mu.shape:', mu.shape)

    model.optimize(num_steps=10)