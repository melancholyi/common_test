'''
Author: chasey melancholycy@gmail.com
Date: 2025-01-24 13:08:24
FilePath: /mesh_planner/test/python/plotEllip2D.py
Description: 

Copyright (c) 2025 by chasey (melancholycy@gmail.com), All Rights Reserved. 
'''
import numpy as np
import matplotlib.pyplot as plt

# 计算旋转矩阵
def rotation_matrix(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

def computeEllipsoidP(R, S):
    return R @ S @ np.transpose(R) 

# 绘制椭球
def plot_ellipsoid(ax, center, a, b, rot_yaw):
    theta = np.linspace(0, 2 * np.pi, 100)
    x_ellipse = a * np.cos(theta)
    y_ellipse = b * np.sin(theta)
    
    # 应用旋转
    R = rotation_matrix(rot_yaw)
    rotated_x = x_ellipse * R[0, 0] + y_ellipse * R[0, 1]
    rotated_y = x_ellipse * R[1, 0] + y_ellipse * R[1, 1]


    # P
    S = np.array([
        [1/(a*a), 0],
        [0, 1/(b*b)]
    ])
    # print(R)
    # print(S)
    P = computeEllipsoidP(R, S)
    print(P)

    
    # 绘制椭球
    ax.plot(rotated_x + center[0], rotated_y + center[1], 'o-')

# 创建绘图
fig, ax = plt.subplots()

# 绘制第一个椭球：a=1, b=2, 旋转角度为0度，中心在(1, 2)
plot_ellipsoid(ax, (1, 2), 2, 1, np.radians(45))

# 绘制第二个椭球：a=3, b=1, 旋转角度为30度，中心在(3, 5)
plot_ellipsoid(ax, (6, 2), 3, 1, np.radians(45))  # 将角度转换为弧度

# 设置坐标轴比例相等
ax.set_aspect('equal')

# 设置坐标轴范围
plt.xlim(-10, 15)
plt.ylim(-10, 15)

# 添加标题和标签
ax.set_title('2D Ellipsoids with Different Rotations')
ax.set_xlabel('X')
ax.set_ylabel('Y')

# 显示图形
plt.grid(True)
plt.show()