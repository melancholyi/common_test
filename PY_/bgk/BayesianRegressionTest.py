'''
Author: chasey && melancholycy@gmail.com
Date: 2025-03-29 03:50:19
LastEditTime: 2025-03-29 03:50:22
FilePath: /test/PY_/bgk/BayesianRegressionTest.py
Description: 
Reference: 
Copyright (c) 2025 by chasey && melancholycy@gmail.com, All Rights Reserved. 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class BayesianKernelRegression(nn.Module):
    def __init__(self, input_dim, kernel_dim):
        super(BayesianKernelRegression, self).__init__()
        self.kernel = nn.Parameter(torch.randn(input_dim, kernel_dim))
        self.log_scale = nn.Parameter(torch.tensor(0.0))  # 用于建模噪声的对数标准差

    def forward(self, x):
        # 计算核函数
        kernel_output = torch.matmul(x, self.kernel)
        # 假设输出是一个高斯分布
        mean = kernel_output.sum(dim=1)
        scale = torch.exp(self.log_scale)
        return mean, scale
    

def nll_loss(mean, scale, y):
    # 假设预测分布是高斯分布
    dist = torch.distributions.Normal(mean, scale)
    # 计算负对数似然
    loss = -dist.log_prob(y).mean()
    return loss

# 创建模拟数据
x_train = torch.randn(100, 10)  # 100 个样本，每个样本有 10 个特征
y_train = torch.randn(100)  # 100 个目标值

# 初始化模型和优化器
model = BayesianKernelRegression(input_dim=10, kernel_dim=5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练循环
for epoch in range(100):
    optimizer.zero_grad()
    mean, scale = model(x_train)
    loss = nll_loss(mean, scale, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")


# 预测
with torch.no_grad():
    mean, scale = model(x_train)

# 可视化结果
plt.figure(figsize=(10, 6))
plt.errorbar(y_train.numpy(), mean.numpy(), yerr=scale.item(), fmt='o', label='Predictions')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', label='True Values')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Bayesian Kernel Regression Predictions')
plt.legend()
plt.show()