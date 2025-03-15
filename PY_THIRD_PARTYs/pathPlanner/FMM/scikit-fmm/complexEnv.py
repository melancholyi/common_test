import numpy as np
import matplotlib.pyplot as plt
from skfmm import distance

# 定义一个 2D 网格地图
# 1 表示障碍物，-1 表示目标点，0 表示自由单元格
def create_grid(size, obstacles, goal):
    grid = np.zeros(size)  # 初始化网格为自由单元格
    grid[goal] = -1  # 设置目标点

    # 添加障碍物
    for obs in obstacles:
        if obs['type'] == 'rectangle':
            x_start, y_start, width, height = obs['params']
            grid[x_start:x_start+width, y_start:y_start+height] = 1
        elif obs['type'] == 'line':
            x_start, y_start, length, direction = obs['params']
            if direction == 'horizontal':
                grid[x_start:x_start+length, y_start] = 1
            elif direction == 'vertical':
                grid[x_start, y_start:y_start+length] = 1
    return grid

# 创建网格
size = (20, 20)  # 网格大小
goal = (10, 10)  # 目标点
obstacles = [
    {'type': 'rectangle', 'params': (2, 2, 5, 3)},  # 矩形障碍物
    {'type': 'rectangle', 'params': (12, 5, 3, 6)},  # 矩形障碍物
    {'type': 'line', 'params': (8, 15, 4, 'vertical')},  # 垂直线障碍物
    {'type': 'line', 'params': (15, 8, 3, 'horizontal')},  # 水平线障碍物
]

grid = create_grid(size, obstacles, goal)

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