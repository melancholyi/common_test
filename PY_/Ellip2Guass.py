'''
Author: chasey melancholycy@gmail.com
Date: 2025-01-30 06:20:14
FilePath: /mesh_planner/test/python/Ellip2Guass.py
Description: 

Copyright (c) 2025 by chasey (melancholycy@gmail.com), All Rights Reserved. 
'''
import numpy as np
import matplotlib.pyplot as plt

def sampleEllipByGaussian(center, a, b, theta, samplepts):
    # 将角度从度转换为弧度
    # theta = np.deg2rad(theta)
    
    # 定义旋转矩阵 R
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    # print(R)
    
    # 定义缩放矩阵 S
    S = np.array([
        [(a/3)**2, 0],
        [0, (b/3)**2]
    ])
    # print(S)
    
    # 计算协方差矩阵 cov = R @ S @ R^T
    cov = R @ S @ R.T
    print(cov)
    
    # 生成高斯分布的样本点
    samples = np.random.multivariate_normal(center, cov, samplepts)
    return samples

def plotEllipsoid(ax, center, a, b, theta):
    # 将角度从度转换为弧度
    # theta = np.deg2rad(theta)
    
    # 生成椭圆的参数方程
    theta_vals = np.linspace(0, 2 * np.pi, 100)
    x_ellipse = a * np.cos(theta_vals)
    y_ellipse = b * np.sin(theta_vals)
    
    # 旋转椭圆
    rotated_x = x_ellipse * np.cos(theta) - y_ellipse * np.sin(theta)
    rotated_y = x_ellipse * np.sin(theta) + y_ellipse * np.cos(theta)
    
    # 绘制椭圆
    ax.plot(rotated_x + center[0], rotated_y + center[1], 'b-')

# 定义椭圆的中心、半长轴、半短轴和旋转角度
center = np.array([1, 1])
a = 0.5 
b = 0.2
theta = np.deg2rad(0)  # 角度

# 生成高斯分布的样本点
samples = sampleEllipByGaussian(center, a, b, theta, 50)

# 创建绘图
fig, ax = plt.subplots()
ax.set_aspect('equal')

# 绘制高斯分布的样本点
ax.scatter(samples[:, 0], samples[:, 1], color='red', alpha=0.5, label='Gaussian Samples')

# 绘制椭圆
plotEllipsoid(ax, center, a, b, theta)

# 设置图例和标题
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Ellipse and Gaussian Samples')

# 显示图形
plt.show()