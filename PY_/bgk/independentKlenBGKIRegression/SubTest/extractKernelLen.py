'''
Author: chasey && melancholycy@gmail.com
Date: 2025-05-20 08:49:02
LastEditTime: 2025-05-20 08:51:18
FilePath: /test/PY_/bgk/independentKlenBGKIRegression/SubTest/extractKernelLen.py
Description: 
Reference: 
Copyright (c) 2025 by chasey && melancholycy@gmail.com, All Rights Reserved. 
'''
import numpy as np

# 定义网格参数
minx, maxx = 0.0, 10.0
miny, maxy = 0.0, 10.0
resolution = 0.2

# 创建网格
x = np.arange(minx, maxx + resolution, resolution)
y = np.arange(miny, maxy + resolution, resolution)
X, Y = np.meshgrid(x, y)
print(f"X.shape: {X.shape}, Y.shape: {Y.shape}")

# 假设有一些数据存储在网格中，例如每个网格点的值
grid_data = np.random.rand(len(y), len(x))  # 示例数据

# 函数：将点映射到网格并提取数据
def get_grid_data(points):
    # 确定网格索引
    idx_x = ((points[:, 0] - minx) / resolution).astype(int)
    idx_y = ((points[:, 1] - miny) / resolution).astype(int)
    
    # 确保索引在有效范围内
    idx_x = np.clip(idx_x, 0, len(x) - 1) # limit idx_x to valid range[0, len(x)-1]
    idx_y = np.clip(idx_y, 0, len(y) - 1) # limit idx_y to valid range[0, len(y)-1]
    
    # 提取数据
    return grid_data[idx_y, idx_x]

# 示例用法
points = np.random.rand(10, 2) * 10.0  # 随机生成10个点
point_data = get_grid_data(points)

print("Point data:", point_data)