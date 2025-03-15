'''
Author: chasey melancholycy@gmail.com
Date: 2025-02-25 11:09:37
FilePath: /mesh_planner/test/pathPlanner/FMM/scikit-fmm/simpleEnv.py
Description: 

Copyright (c) 2025 by chasey (melancholycy@gmail.com), All Rights Reserved. 
'''
import numpy as np
import matplotlib.pyplot as plt
from skfmm import distance

# 定义一个 2D 网格地图
# 1 表示障碍物，-1 表示目标点，0 表示自由单元格
grid = np.ones((10, 10))  # 10x10 的网格
grid[5, 5] = -1  # 目标点在 (5, 5)
# grid[2, 2] = 1   # 障碍物在 (2, 2)
# grid[7, 3] = 1   # 障碍物在 (7, 3)

# 计算从目标点到每个自由单元格的距离
distances = distance(grid)

# 可视化网格地图和距离场
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# 绘制网格地图
ax[0].imshow(grid, cmap='viridis', origin='lower')
ax[0].set_title("Grid Map")
ax[0].set_xlabel("X")
ax[0].set_ylabel("Y")
ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)

# 绘制距离场
ax[1].imshow(distances, cmap='viridis', origin='lower')
ax[1].set_title("Distance Field")
ax[1].set_xlabel("X")
ax[1].set_ylabel("Y")
ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)

# 添加颜色条
cbar = fig.colorbar(ax[1].imshow(distances, cmap='viridis', origin='lower'), ax=ax[1])
cbar.set_label("Distance")

plt.tight_layout()
plt.show()