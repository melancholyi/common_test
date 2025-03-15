'''
Author: chasey melancholycy@gmail.com
Date: 2025-01-31 05:45:39
FilePath: /mesh_planner/test/python/elevGaussian.py
Description: 

Copyright (c) 2025 by chasey (melancholycy@gmail.com), All Rights Reserved. 
'''
import numpy as np

# 设置随机种子以确保结果可复现
np.random.seed(42)

# 生成随机点
num_points = 1000
x = np.random.uniform(0, 1, num_points)
y = np.random.uniform(0, 1, num_points)
z = np.random.uniform(0, 10, num_points)

# 组合成三维坐标点
points = np.column_stack((x, y, z))
print("points:")
print(points)

# 计算均值
mean = np.mean(points, axis=0)
print("均值：")
print(mean)

# 计算协方差矩阵
cov_matrix = np.cov(points, rowvar=False)
print("\n协方差矩阵：")
print(cov_matrix)

# 进行特征值分解
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print("\n特征值：")
print(eigenvalues)
print("\n特征向量：")
print(eigenvectors)

# 提取与z轴相关性最强的特征向量和特征值
z_axis_vector = np.array([0, 0, 1])  # z轴方向向量
max_z_component = -1
max_z_index = -1

for i in range(len(eigenvalues)):
    # 计算特征向量在z轴方向上的分量绝对值
    z_component = abs(np.dot(eigenvectors[:, i], z_axis_vector))
    if z_component > max_z_component:
        max_z_component = z_component
        max_z_index = i

# 提取与z轴相关性最强的特征向量和特征值
z_related_eigenvalue = eigenvalues[max_z_index]
z_related_eigenvector = eigenvectors[:, max_z_index]

print("\n与z轴相关性最强的特征值：")
print(z_related_eigenvalue)
print("\n与z轴相关性最强的特征向量：")
print(z_related_eigenvector)