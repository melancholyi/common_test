import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

# 定义判断点是否在椭圆内的函数
def isPtInEllipsoid(pt, center, aSquaInv, bSquaInv, theta):
    """
    判断点是否在椭圆内。
    
    参数:
    pt: 点的坐标 (x, y)
    center: 椭圆中心的坐标 (cx, cy)
    aSquaInv: 半长轴的平方的倒数 1/a^2
    bSquaInv: 半短轴的平方的倒数 1/b^2
    theta: 椭圆的旋转角度（单位：度）
    
    返回:
    True 如果点在椭圆内，否则返回 False
    """
    x, y = pt
    cx, cy = center
    x_centered = x - cx
    y_centered = y - cy
    
    theta_rad = np.deg2rad(theta)
    x_rotated = x_centered * np.cos(theta_rad) + y_centered * np.sin(theta_rad)
    y_rotated = -x_centered * np.sin(theta_rad) + y_centered * np.cos(theta_rad)
    
    return (x_rotated**2 * aSquaInv) + (y_rotated**2 * bSquaInv) <= 1

# 定义栅格参数
grid_min_x, grid_min_y = 0, 0
grid_max_x, grid_max_y = 2, 2
resolution = 0.2

# 定义椭圆参数
ellipse_center_x, ellipse_center_y = 1.1, 1.1
semi_major_axis = resolution * 3 + resolution/2  # 半长轴 
semi_minor_axis = resolution * 2 + resolution/2  # 半短轴
rotation_angle = 180  # 旋转角度（单位：度）

# 计算 a^2 和 b^2 的倒数
aSquaInv = 1 / (semi_major_axis ** 2)
bSquaInv = 1 / (semi_minor_axis ** 2)

# 创建栅格
x = np.arange(grid_min_x, grid_max_x, resolution)
y = np.arange(grid_min_y, grid_max_y, resolution)
X, Y = np.meshgrid(x, y)

# 绘制结果
plt.figure(figsize=(6, 6))
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        # 栅格的左下角坐标
        grid_x = X[i, j]
        grid_y = Y[i, j]
        # 栅格中心点坐标
        grid_center_x = grid_x + resolution / 2
        grid_center_y = grid_y + resolution / 2
        # 判断栅格中心点是否在椭圆内
        if isPtInEllipsoid((grid_center_x, grid_center_y), 
                           (ellipse_center_x, ellipse_center_y), 
                           aSquaInv, bSquaInv, rotation_angle):
            color = 'red'
        else:
            color = 'white'
        # 绘制栅格
        rectangle = Rectangle((grid_x, grid_y), resolution, resolution,
                              facecolor=color, edgecolor='black', linewidth=0.5)
        plt.gca().add_patch(rectangle)

# 绘制椭圆
ellipse = Ellipse(xy=(ellipse_center_x, ellipse_center_y),
                  width=2 * semi_major_axis, height=2 * semi_minor_axis,
                  angle=rotation_angle, edgecolor='blue', facecolor='none', linewidth=2)
plt.gca().add_patch(ellipse)

# 设置绘图范围和比例
plt.xlim(grid_min_x - resolution, grid_max_x)
plt.ylim(grid_min_y - resolution, grid_max_y)
plt.gca().set_aspect('equal', adjustable='box')
plt.title("Elliptical Grid with Ellipse")
plt.show()




"""
old version
"""
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Ellipse, Rectangle

# # 定义栅格参数
# grid_min_x, grid_min_y = 0, 0
# grid_max_x, grid_max_y = 2, 2
# resolution = 0.25

# # 定义椭圆参数
# ellipse_center_x, ellipse_center_y = 1, 1
# semi_major_axis = 1.0  # 半长轴
# semi_minor_axis = 0.5  # 半短轴
# rotation_angle = 30  # 旋转角度（单位：度）

# # 创建栅格
# x = np.arange(grid_min_x, grid_max_x, resolution)
# y = np.arange(grid_min_y, grid_max_y, resolution)
# X, Y = np.meshgrid(x, y)

# # 将栅格左下角坐标转换为相对于椭圆中心的坐标
# X_centered = X + resolution / 2 - ellipse_center_x
# Y_centered = Y + resolution / 2 - ellipse_center_y

# # 旋转坐标（如果需要）
# rotation_angle_rad = np.deg2rad(rotation_angle)
# X_rotated = X_centered * np.cos(rotation_angle_rad) + Y_centered * np.sin(rotation_angle_rad)
# Y_rotated = -X_centered * np.sin(rotation_angle_rad) + Y_centered * np.cos(rotation_angle_rad)

# # 判断每个点是否在椭圆内
# ellipse_mask = (X_rotated**2 / semi_major_axis**2) + (Y_rotated**2 / semi_minor_axis**2) <= 1

# # 绘制结果
# plt.figure(figsize=(6, 6))
# for i in range(X.shape[0]):
#     for j in range(X.shape[1]):
#         # 栅格的左下角坐标
#         grid_x = X[i, j]
#         grid_y = Y[i, j]
#         # 栅格的颜色
#         color = 'red' if ellipse_mask[i, j] else 'white'
#         # 绘制栅格
#         rectangle = Rectangle((grid_x, grid_y), resolution, resolution,
#                               facecolor=color, edgecolor='black', linewidth=0.5)
#         plt.gca().add_patch(rectangle)

# # 绘制椭圆
# ellipse = Ellipse(xy=(ellipse_center_x, ellipse_center_y),
#                   width=2 * semi_major_axis, height=2 * semi_minor_axis,
#                   angle=rotation_angle, edgecolor='blue', facecolor='none', linewidth=2)
# plt.gca().add_patch(ellipse)

# # 设置绘图范围和比例
# plt.xlim(grid_min_x - resolution, grid_max_x)
# plt.ylim(grid_min_y - resolution, grid_max_y)
# plt.gca().set_aspect('equal', adjustable='box')
# plt.title("Elliptical Grid with Ellipse")
# plt.show()