'''
Author: chasey melancholycy@gmail.com
Date: 2025-02-25 11:16:40
FilePath: /mesh_planner/test/pathPlanner/FMM/scikit-fmm/simple.py
Description: 

Copyright (c) 2025 by chasey (melancholycy@gmail.com), All Rights Reserved. 
'''
import numpy as np
import matplotlib.pyplot as plt
import skfmm

# 创建 5x5 网格
grid_size = 5000
phi = np.ones((grid_size, grid_size))

# 将中心单元设置为 -1
phi[grid_size // 2, grid_size // 2] = 0

# 计算距离
distances = skfmm.distance(phi, dx = 10/grid_size)
print(distances)
print(np.sqrt(2)/2)


# 可视化结果
plt.imshow(distances, cmap='viridis', origin='lower')
plt.colorbar(label='Distance')
plt.title('Distance from Center Cell')
plt.show()