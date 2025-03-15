'''
Author: chasey melancholycy@gmail.com
Date: 2025-01-30 02:02:41
FilePath: /mesh_planner/test/python/sampleInEllipsoid.py
Description: 

Copyright (c) 2025 by chasey (melancholycy@gmail.com), All Rights Reserved. 
'''
import numpy as np
import matplotlib.pyplot as plt

def sample_ellipse(a, b, theta, res):
    # 将角度转换为弧度
    theta_rad = np.deg2rad(theta)
    
    # 计算采样点
    samples = []
    r_max = 1.0  # 椭圆的半径范围从0到1
    phi_max = 2 * np.pi  # 参数phi的范围从0到2π

    # 计算步长
    dr = res / a  # r的步长
    dphi = res / b  # phi的步长

    for r in np.arange(0, r_max, dr):
        for phi in np.arange(0, phi_max, dphi):
            # 计算旋转后的坐标
            x = r * np.cos(phi) * np.cos(theta_rad) - r * np.sin(phi) * np.sin(theta_rad)
            y = r * np.cos(phi) * np.sin(theta_rad) + r * np.sin(phi) * np.cos(theta_rad)
            
            # 检查点是否在椭圆内部
            if (x**2 / a**2 + y**2 / b**2) <= 1:
                samples.append((x, y))

    return np.array(samples)

def plot_ellipse(a, b, theta, samples, pts):
    # 将角度转换为弧度
    theta_rad = np.deg2rad(theta)
    
    # 绘制椭圆
    phi = np.linspace(0, 2 * np.pi, pts)
    x_ellipse = a * np.cos(phi) * np.cos(theta_rad) - b * np.sin(phi) * np.sin(theta_rad)
    y_ellipse = a * np.cos(phi) * np.sin(theta_rad) + b * np.sin(phi) * np.cos(theta_rad)
    
    plt.figure(figsize=(6, 6))
    plt.plot(x_ellipse, y_ellipse, 'b-', label='Ellipse')
    plt.scatter(samples[:, 0], samples[:, 1], s=10, c='r', label='Samples')
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    plt.title('Ellipse Sampling')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.axis('equal')  # 保持比例
    plt.show()



# 参数
a = 0.5  # 半长轴
b = 0.2  # 半短轴
theta = 0  # 旋转角度
res = 0.05  # 分辨率

# 采样
samples = sample_ellipse(a, b, theta, res)

# 可视化
plot_ellipse(a, b, theta, samples, 1000)