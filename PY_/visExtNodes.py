'''
Author: chasey melancholycy@gmail.com
Date: 2025-02-20 05:09:06
FilePath: /utils_ws/test/pythonTest/visExtNodes.py
Description: 

Copyright (c) 2025 by chasey (melancholycy@gmail.com), All Rights Reserved. 
'''
import numpy as np
import matplotlib.pyplot as plt

def read_nodes_from_file(filename):
    """
    从文件中读取节点数据 (x, y, yaw)
    :param filename: 文件路径
    :return: 包含所有节点的列表 [(x, y, yaw)]
    """
    nodes = []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 3:  # 确保每行有三个值
                x, y, yaw = map(float, parts)
                nodes.append((x, y, yaw))
    return nodes

def visualize_nodes(nodes):
    """
    可视化节点数据
    :param nodes: 包含 (x, y, yaw) 的列表
    """
    x = [node[0] for node in nodes]
    y = [node[1] for node in nodes]
    yaw = [node[2] for node in nodes]

    plt.figure(figsize=(10, 8))
    plt.plot(x, y, 'o-', label="Trajectory", markersize=4)

    # 添加方向箭头
    arrow_length = 0.1  # 箭头长度
    for xi, yi, yawi in zip(x, y, yaw):
        dx = np.cos(yawi) * arrow_length
        dy = np.sin(yawi) * arrow_length
        plt.arrow(xi, yi, dx, dy, head_width=0.1, head_length=0.2, fc='r', ec='r')

    # 添加起点和终点标记
    plt.scatter(x[0], y[0], color='green', s=100, label='Start Point', zorder=5)
    plt.scatter(x[-1], y[-1], color='blue', s=100, label='End Point', zorder=5)

    plt.title("Extended Nodes Visualization")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# 主函数
if __name__ == "__main__":
    filename = "extended_nodes.txt"  # 文件路径
    nodes = read_nodes_from_file(filename)
    visualize_nodes(nodes)