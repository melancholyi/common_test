'''
Author: chasey && melancholycy@gmail.com
Date: 2025-03-25 05:42:10
LastEditTime: 2025-03-25 05:42:13
FilePath: /test/PY_/plotRBFFunc.py
Description: 
Reference: 
Copyright (c) 2025 by chasey && melancholycy@gmail.com, All Rights Reserved. 
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义二维高斯核函数
def gaussian_rbf_2d(x, y, xc, yc, sigma):
    return np.exp(-((x - xc)**2 + (y - yc)**2) / (2 * sigma**2))

# 设置参数
xc, yc = 0.0, 0.0  # 中心点
sigma = 1.0        # 宽度参数

# 创建网格点
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = gaussian_rbf_2d(X, Y, xc, yc, sigma)

# 绘制三维图像
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_title('2D Gaussian RBF')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('k(||(x,y) - (xc,yc)||)')
plt.show()