'''
Author: chasey && melancholycy@gmail.com
Date: 2025-04-09 02:30:07
LastEditTime: 2025-04-09 02:32:48
FilePath: /test/PY_AI/ML/EM/LinearModelTest/em.py
Description: 
Reference: 
Copyright (c) 2025 by chasey && melancholycy@gmail.com, All Rights Reserved. 
'''

import numpy as np
import matplotlib.pyplot as plt

# 生成模拟数据
np.random.seed(42)
N = 1000
K = 2

# 模拟两个线性模型
X = np.random.rand(N, 1) * 10
Y1 = 2 * X + 1 + np.random.normal(0, 1, (N, 1))
Y2 = -1 * X + 3 + np.random.normal(0, 1, (N, 1))

# 混合数据
Z = np.random.binomial(1, 0.5, (N, 1))
Y = Z * Y1 + (1 - Z) * Y2

# 添加偏置项
X = np.hstack((np.ones((N, 1)), X))

# EM算法参数初始化
pi = np.ones(K) / K
w = np.random.randn(K, 2)
sigma2 = np.ones(K)

# EM算法迭代
n_iter = 100
for _ in range(n_iter):
    # E步：计算责任
    gamma = np.zeros((N, K))
    for k in range(K):
        # 确保 Y 和 X @ w[k].reshape(-1, 1) 的形状一致
        Y_flat = Y.flatten()  # 将 Y 转换为一维数组
        Xw = X @ w[k].reshape(-1, 1)  # X @ w[k] 的结果是 (N, 1)
        Xw_flat = Xw.flatten()  # 将 Xw 转换为一维数组
        
        # 计算高斯概率密度
        prob = pi[k] * np.exp(-0.5 * ((Y_flat - Xw_flat) ** 2) / sigma2[k])
        gamma[:, k] = prob  # 确保 gamma[:, k] 是一维数组
    
    # 归一化责任
    gamma = gamma / gamma.sum(axis=1, keepdims=True)
    
    # M步：更新参数
    for k in range(K):
        Nk = gamma[:, k].sum()
        # 更新混合权重
        pi[k] = Nk / N
        
        # 更新线性回归权重
        X_weighted = X * gamma[:, k].reshape(-1, 1)
        w[k] = np.linalg.inv(X.T @ X_weighted) @ (X_weighted.T @ Y).flatten()
        
        # 更新方差
        residuals = Y - X @ w[k].reshape(-1, 1)
        sigma2[k] = (gamma[:, k] * (residuals ** 2)).sum() / Nk

# 输出结果
print("混合权重:", pi)
print("回归权重:", w)
print("方差:", sigma2)

# 绘图
plt.scatter(X[:, 1], Y, alpha=0.5, label='Data')
for k in range(K):
    x_line = np.linspace(0, 10, 100)
    y_line = w[k, 0] + w[k, 1] * x_line
    plt.plot(x_line, y_line, label=f'Model {k+1}')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Mixture of Linear Regressions with EM Algorithm')
plt.show()