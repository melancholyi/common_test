'''
Author: chasey && melancholycy@gmail.com
Date: 2025-03-30 07:14:17
LastEditTime: 2025-03-30 07:14:20
FilePath: /test/PY_/plot/plotSoftplus.py
Description: 
Reference: 
Copyright (c) 2025 by chasey && melancholycy@gmail.com, All Rights Reserved. 
'''
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F  
"""
\[
\text{Softplus}(x) = \frac{1}{\beta} \cdot \log(1 + e^{\beta x})
\]
"""
def softplus(x, beta=1.0, threshold=20.0):
    """Transform the input to positive output using Softplus."""
    return F.softplus(x, beta=beta, threshold=threshold) + 1e-6

# 生成 x 的值
x = np.linspace(-5, 5, 400)  # 从 -5 到 5 生成 400 个点

# 将 NumPy 数组转换为 PyTorch 张量
x_tensor = torch.tensor(x, dtype=torch.float32)

# 计算 Softplus 函数的值
y_tensor = softplus(x_tensor, beta=1.0, threshold=20.0)

# 将 PyTorch 张量转换回 NumPy 数组
y = y_tensor.numpy()

# 绘制 Softplus 函数
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Softplus Function')
plt.title('Softplus Function')
plt.xlabel('x')
plt.ylabel('Softplus(x)')
plt.grid(True)
plt.legend()
plt.show()