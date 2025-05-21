'''
Author: chasey && melancholycy@gmail.com
Date: 2025-05-21 04:46:00
LastEditTime: 2025-05-21 06:29:40
FilePath: /test/PY_/bgk/independentKlenBGKIRegression/SubTest/extractIndexBackward.py
Description: 
Reference: 
Copyright (c) 2025 by chasey && melancholycy@gmail.com, All Rights Reserved. 
'''
import torch

# 创建一个5x5的浮点类型张量tensor1，并启用梯度计算
tensor1 = torch.randn(5, 5, requires_grad=True)
print("原始tensor1:")
print(tensor1)

# 给定的最小坐标和分辨率
minx, miny = 0.0, 0.0
resolution = 1.0  # 假设分辨率为1.0，根据实际情况调整

# 给定的另外5个浮点类型的位置
points = torch.tensor([
    [0.5, 0.5],
    [1.2, 1.2],
    [2.3, 2.3],
    [3.4, 3.4],
    [4.5, 4.5]
], dtype=torch.float32)

# 计算每个点的索引
rows = ((points[:, 0] - minx) / resolution).long()
cols = ((points[:, 1] - miny) / resolution).long()

# 确保索引在tensor1的范围内
rows = torch.clamp(rows, 0, tensor1.size(0) - 1)
cols = torch.clamp(cols, 0, tensor1.size(1) - 1)

# 通过计算得到的索引提取元素，形成一维的tensor2
tensor2 = tensor1[rows, cols]
print(f"\n提取元素构成的tensor2: tensor2.shape:{tensor2.shape}")

print(tensor2)

# 定义一个优化器
optimizer = torch.optim.SGD([tensor1], lr=0.01)

# 计算过程：tensor2与权重矩阵相乘，加上偏置，再应用ReLU激活函数
tensor2 = tensor2 * 2
print("\n经过复杂计算后的tensor2:")
print(tensor2)

# 计算损失函数，例如tensor2的均值
loss = tensor2.sum()

# 反向传播计算梯度
loss.backward()

# 执行优化步骤来更新tensor1的值
optimizer.step()

# 输出更新后的tensor1
print("\n更新后的tensor1:")
print(tensor1)

# 输出原始tensor1的梯度
print("\n原始tensor1的梯度:")
print(tensor1.grad)