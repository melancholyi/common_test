'''
Author: chasey melancholycy@gmail.com
Date: 2025-01-25 12:35:02
FilePath: /mesh_planner/test/python/test.py
Description: 

Copyright (c) 2025 by chasey (melancholycy@gmail.com), All Rights Reserved. 
'''
import numpy as np
import matplotlib.pyplot as plt

def sample_point_in_ellipse(a, b):
    """
    在椭圆内部随机采样一个点
    :param a: 椭圆的半长轴
    :param b: 椭圆的半短轴
    :return: 采样点的坐标 (x, y)
    """
    # 生成随机半径 r 和角度 theta
    r = np.sqrt(np.random.rand())  # 半径 r 的平方在 [0, 1] 内均匀分布
    theta = np.random.uniform(0, 2 * np.pi)  # 角度 theta 在 [0, 2π] 内均匀分布

    # 计算单位圆内的点 (u, v)
    u = r * np.cos(theta)
    v = r * np.sin(theta)

    # 转换到椭圆坐标系
    x = a * u
    y = b * v

    return x, y
    
# 示例：在椭圆内部采样多个点
a = 5  # 半长轴
b = 3  # 半短轴
num_samples = 1000  # 采样点的数量

# 采样点
samples = [sample_point_in_ellipse(a, b) for _ in range(num_samples)]

# 提取采样点的 x 和 y 坐标
x_samples, y_samples = zip(*samples)

# 绘制椭圆和采样点
theta = np.linspace(0, 2 * np.pi, 1000)
ellipse_x = a * np.cos(theta)
ellipse_y = b * np.sin(theta)

plt.figure(figsize=(8, 6))
plt.plot(ellipse_x, ellipse_y, label="Ellipse", color="blue", linewidth=2)
plt.scatter(x_samples, y_samples, color="red", s=10, alpha=0.5, label="Sample Points")
plt.title("Random Sampling Inside an Ellipse")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.axis("equal")  # 保持椭圆的形状
plt.show()