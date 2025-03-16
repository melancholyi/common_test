import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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

# 计算矩形参数
def calculate_rectangle_params(start, end, path):
    # 计算起点到终点的距离（矩形长度）
    rect_length = 2.0 * np.linalg.norm(np.array(end) - np.array(start))
    
    # 计算路径点到起点和终点连线的垂直距离（矩形宽度）
    distances = [perpendicular_distance(p, start, end) for p in path]
    rect_width = 3.0 * max(distances)
    
    # 计算矩形的旋转角度（与 x 轴的夹角）
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    yaw = np.degrees(np.arctan2(dy, dx))  # 转换为度数
    
    # 计算矩形中心（短轴中垂线的中点）
    center_x = (start[0] + end[0]) / 2
    center_y = (start[1] + end[1]) / 2
    center = (center_x, center_y)
    
    return center, rect_length, rect_width, yaw

# 判断点是否在矩形内
def is_point_in_rectangle(point, center, length, width, yaw):
    """
    判断点是否在旋转矩形内
    :param point: (x, y) 点的坐标
    :param center: (cx, cy) 矩形中心的坐标
    :param length: 矩形的长度
    :param width: 矩形的宽度
    :param yaw: 矩形的旋转角度（相对于 x 轴，单位为度）
    :return: True 如果点在矩形内，否则 False
    """
    # 将点平移到矩形中心为原点的坐标系
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
    
    # 检查点是否在矩形内
    half_length = length / 2
    half_width = width / 2
    return (-half_length <= rotated_point[0] <= half_length) and (-half_width <= rotated_point[1] <= half_width)

# 逆向旋转点
def rotate_point_around_center(point, center, angle):
    """
    逆向旋转点绕中心点
    :param point: (x, y) 点的坐标
    :param center: (cx, cy) 中心点的坐标
    :param angle: 旋转角度（弧度）
    :return: 旋转后的点坐标
    """
    px, py = point
    cx, cy = center
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    
    # 平移点到原点
    px -= cx
    py -= cy
    
    # 逆向旋转
    new_px = cos_theta * px + sin_theta * py
    new_py = -sin_theta * px + cos_theta * py
    
    # 平移回原位置
    new_px += cx
    new_py += cy
    
    return new_px, new_py


# 随机生成点并判断是否在矩形内
def generate_random_points_and_plot(path, center, length, width, yaw, num_points=100):
    # 随机生成点
    points = np.random.uniform(low=-10, high=10, size=(num_points, 2))
    
    # 判断每个点是否在矩形内
    colors = ['red' if is_point_in_rectangle(point, center, length, width, yaw) else 'green' for point in points]
    
    # 绘制点
    fig, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1], c=colors, s=20)

    # 计算未旋转的左下角坐标
    unrotated_bottom_left_x = center[0] - length / 2
    unrotated_bottom_left_y = center[1] - width / 2

    # 逆向旋转左下角坐标
    angle_rad = np.radians(yaw)
    bottom_left_x, bottom_left_y = rotate_point_around_center(
        (unrotated_bottom_left_x, unrotated_bottom_left_y), center, -angle_rad
    )
    
    # 绘制矩形
    rectangle = Rectangle(
        (bottom_left_x, bottom_left_y),
        length,
        width,
        angle=yaw,
        edgecolor='red',
        facecolor='none',
        linewidth=2,
        label='Rectangle'
    )
    ax.add_patch(rectangle)
    
    # 绘制路径
    path_x, path_y = zip(*path)
    ax.plot(path_x, path_y, marker='o', linestyle='-', color='blue', label='Path')
    
    # 设置图形属性
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title("Z-shaped Path, Rectangle, and Random Points")
    plt.grid(True)
    plt.show()

# 主程序
if __name__ == "__main__":
    start = (0, 0)  # 起点
    length = 10  # 路径长度
    path = generate_z_path(start, length)
    end = path[-1]  # 终点
    center, rect_length, rect_width, yaw = calculate_rectangle_params(start, end, path)
    
    generate_random_points_and_plot(path, center, rect_length, rect_width, yaw, num_points=200)