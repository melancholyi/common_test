# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Ellipse

# # 生成 Z 字形路径
# def generate_z_path(start, length):
#     path = [start]
#     current = np.array(start)
#     directions = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    
#     # Z 字形路径的生成逻辑
#     for i in range(length - 1):
#         if i % 3 == 0:
#             direction = directions[1]  # 斜向上
#         elif i % 3 == 1:
#             direction = directions[3]  # 斜向下
#         else:
#             direction = directions[0]  # 向右
#         current += direction
#         path.append(tuple(current))
    
#     return path

# # 计算点到直线的垂直距离
# def perpendicular_distance(point, line_start, line_end):
#     line_vector = np.array(line_end) - np.array(line_start)
#     point_vector = np.array(point) - np.array(line_start)
#     distance = np.linalg.norm(np.cross(line_vector, point_vector)) / np.linalg.norm(line_vector)
#     return distance

# # 计算椭圆参数
# def calculate_ellipse_params(start, end, path):
#     F1 = np.array(start)
#     F2 = np.array(end)
#     c = np.linalg.norm(F2 - F1) / 2  # 焦点距离的一半
#     a = 1.5 * c  # 长轴长度的一半
    
#     # 计算每个路径点到起点和终点连线的垂直距离
#     distances = [perpendicular_distance(p, start, end) for p in path]
#     max_distance = max(distances)
    
#     b = 2.0 * max_distance  # 短轴长度的一半
#     return a, b, c

# # 计算旋转角度（yaw）
# def calculate_yaw(start, end):
#     dx = end[0] - start[0]
#     dy = end[1] - start[1]
#     yaw = np.degrees(np.arctan2(dy, dx))  # 转换为度数
#     return yaw

# # 绘制路径和椭圆
# def plot_path_and_ellipse(path, a, b, center, yaw):
#     fig, ax = plt.subplots()
    
#     # 绘制路径
#     path_x, path_y = zip(*path)
#     ax.plot(path_x, path_y, marker='o', linestyle='-', color='blue', label='Path')
    
#     # 绘制椭圆
#     ellipse = Ellipse(center, 2*a, 2*b, angle=yaw, edgecolor='red', facecolor='none', linewidth=2, label='Ellipse')
#     ax.add_patch(ellipse)
    
#     # 设置图形属性
#     ax.set_aspect('equal')
#     ax.legend()
#     ax.set_title("Z-shaped Path and Ellipse")
#     plt.grid(True)
#     plt.show()

# # 主程序
# if __name__ == "__main__":
#     start = (0, 0)  # 起点
#     length = 10  # 路径长度
#     path = generate_z_path(start, length)
#     end = path[-1]  # 终点
#     center = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)  # 椭圆中心
#     a, b, c = calculate_ellipse_params(start, end, path)
#     yaw = calculate_yaw(start, end)  # 计算旋转角度
    
#     plot_path_and_ellipse(path, a, b, center, yaw)




import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# 生成 Z 字形路径
def generate_z_path(start, length):
    path = [start]
    current = np.array(start)
    directions = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    
    # Z 字形路径的生成逻辑
    for i in range(length - 1):
        if i % 3 == 0:
            direction = directions[1]  # 斜向上
        elif i % 3 == 1:
            direction = directions[3]  # 斜向下
        else:
            direction = directions[0]  # 向右
        current += direction
        path.append(tuple(current))
    
    return path

# 计算点到直线的垂直距离
def perpendicular_distance(point, line_start, line_end):
    line_vector = np.array(line_end) - np.array(line_start)
    point_vector = np.array(point) - np.array(line_start)
    distance = np.linalg.norm(np.cross(line_vector, point_vector)) / np.linalg.norm(line_vector)
    return distance

# 计算椭圆参数
def calculate_ellipse_params(start, end, path):
    F1 = np.array(start)
    F2 = np.array(end)
    c = np.linalg.norm(F2 - F1) / 2  # 焦点距离的一半
    a = 1.5 * c  # 长轴长度的一半
    
    # 计算每个路径点到起点和终点连线的垂直距离
    distances = [perpendicular_distance(p, start, end) for p in path]
    max_distance = max(distances)
    
    b = 2.0 * max_distance  # 短轴长度的一半
    return a, b, c

# 计算旋转角度（yaw）
def calculate_yaw(start, end):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    yaw = np.degrees(np.arctan2(dy, dx))  # 转换为度数
    return yaw

# 判断点是否在椭圆内
def is_point_in_ellipse(point, center, a, b, yaw):
    """
    判断点是否在旋转椭圆内
    :param point: (x, y) 点的坐标
    :param center: (cx, cy) 椭圆中心的坐标
    :param a: 椭圆的长轴长度的一半
    :param b: 椭圆的短轴长度的一半
    :param yaw: 椭圆的旋转角度（相对于 x 轴，单位为度）
    :return: True 如果点在椭圆内，否则 False
    """
    # 将点平移到椭圆中心为原点的坐标系
    translated_point = np.array(point) - np.array(center)
    
    # 将旋转角度转换为弧度
    angle_rad = np.radians(yaw)
    
    # 构造旋转矩阵
    rotation_matrix = np.array([
        [np.cos(angle_rad), np.sin(angle_rad)],
        [-np.sin(angle_rad), np.cos(angle_rad)]
    ])
    
    # 将点逆向旋转回未旋转的坐标系
    rotated_point = np.dot(rotation_matrix, translated_point)
    
    # 检查点是否在椭圆内
    return (rotated_point[0]**2 / a**2 + rotated_point[1]**2 / b**2) <= 1

# 随机生成点并判断是否在椭圆内
def generate_random_points_and_plot(path, a, b, center, yaw, num_points=100):
    # 随机生成点
    points = np.random.uniform(low=-10, high=10, size=(num_points, 2))
    
    # 判断每个点是否在椭圆内
    colors = ['red' if is_point_in_ellipse(point, center, a, b, yaw) else 'green' for point in points]
    
    # 绘制点
    fig, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1], c=colors, s=20)
    
    # 绘制椭圆
    ellipse = Ellipse(center, 2*a, 2*b, angle=yaw, edgecolor='red', facecolor='none', linewidth=2, label='Ellipse')
    ax.add_patch(ellipse)
    
    # 绘制路径
    path_x, path_y = zip(*path)
    ax.plot(path_x, path_y, marker='o', linestyle='-', color='blue', label='Path')
    
    # 设置图形属性
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title("Z-shaped Path, Ellipse, and Random Points")
    plt.grid(True)
    plt.show()

# 主程序
if __name__ == "__main__":
    start = (0, 0)  # 起点
    length = 10  # 路径长度
    path = generate_z_path(start, length)
    end = path[-1]  # 终点
    center = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)  # 椭圆中心
    a, b, c = calculate_ellipse_params(start, end, path)
    yaw = calculate_yaw(start, end)  # 计算旋转角度
    
    generate_random_points_and_plot(path, a, b, center, yaw, num_points=200)