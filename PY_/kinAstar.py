import heapq
import math
import matplotlib.pyplot as plt

# 定义节点类
class Node:
    def __init__(self, x, y, theta, g, h, parent=None):
        self.x = x
        self.y = y
        self.theta = theta
        self.g = g
        self.h = h
        self.parent = parent

    def __lt__(self, other):
        return self.g + self.h < other.g + other.h

# 计算启发式距离（欧几里得距离）
def heuristic(current, goal):
    return math.sqrt((goal.x - current.x) ** 2 + (goal.y - current.y) ** 2)

# 检查是否在障碍物内
def is_obstacle(x, y, obstacles):
    for (ox, oy) in obstacles:
        if x == ox and y == oy:
            return True
    return False

# 生成邻居节点（考虑动力学特性）
def generate_neighbors(current, goal, max_speed, max_steering, dt):
    neighbors = []
    speed = max_speed  # 假设恒定速度
    d_theta = max_steering * dt  # 最大转向角变化

    for steering in [d_theta, 0, -d_theta]:
        theta = current.theta + steering
        x = current.x + speed * math.cos(theta) * dt
        y = current.y + speed * math.sin(theta) * dt

        # 检查是否超出边界或在障碍物内
        if 0 <= x <= 100 and 0 <= y <= 100 and not is_obstacle(x, y, obstacles):
            g = current.g + speed * dt
            h = heuristic(Node(x, y, theta, 0, 0), goal)
            neighbors.append(Node(x, y, theta, g, h, current))

    return neighbors

# 重建路径
def reconstruct_path(node):
    path = []
    while node:
        path.append((node.x, node.y))
        node = node.parent
    return path[::-1]

# A*算法
def a_star(start, goal, obstacles, max_speed, max_steering, dt):
    ## begin plan
    open_list = []
    heapq.heappush(open_list, start)
    closed_set = set()

    count = 0
    while open_list:
        count += 1
        if count == 1000:
            count = 0
            print("current: (", current.x, ' ', current.y, ')')
            print("Still searching...")
        current = heapq.heappop(open_list)

        if abs(current.x - goal.x) < 0.1 and abs(current.y - goal.y) < 0.1:
            return reconstruct_path(current)

        closed_set.add((current.x, current.y))

        neighbors = generate_neighbors(current, goal, max_speed, max_steering, dt)
        for neighbor in neighbors:
            if (neighbor.x, neighbor.y) in closed_set:
                continue

            if not any((n.x, n.y) == (neighbor.x, neighbor.y) for n in open_list):
                heapq.heappush(open_list, neighbor)

    return []

# 可视化路径
def visualize_path(path, obstacles):
    plt.figure(figsize=(8, 8))
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.grid(True)

    # 绘制障碍物
    for (ox, oy) in obstacles:
        plt.plot(ox, oy, "sk", markersize=10)

    # 绘制路径
    if path:
        path_x, path_y = zip(*path)
        plt.plot(path_x, path_y, "-r", linewidth=2)
        plt.plot(start.x, start.y, "go", markersize=10)  # 起点
        plt.plot(goal.x, goal.y, "bo", markersize=10)    # 终点

    plt.title("Path Planning with A* and Differential Drive Dynamics")
    plt.show()

# 主程序
if __name__ == "__main__":
    # 参数设置
    start = Node(10, 10, 0, 0, 0)  # 起点
    goal = Node(100, 100, 0, 0, 0)   # 终点
    obstacles = [(40, 40), (50, 50), (70, 70), (80, 80)]  # 障碍物
    max_speed = 1.0  # 最大速度
    max_steering = math.pi / 4  # 最大转向角
    dt = 1.0  # 时间步长

    # 运行A*算法
    path = a_star(start, goal, obstacles, max_speed, max_steering, dt)

    # 可视化路径
    visualize_path(path, obstacles)