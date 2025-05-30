import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle, Circle

# 参数设置
r = 1.0  # 绕外部圆心的半径（米）
omega_orbit = np.pi / 5  # 绕外部圆心的角速度（弧度/秒）
L = 0.5  # 轮距（米）
omega = np.pi * 5  # 自转角速度（弧度/秒）
robot_length = 0.8  # 机器人机体长度（米）
robot_width = 0.3  # 机器人机体宽度（米）
wheel_radius = 0.1  # 轮子半径（米）

# 计算线速度
v = omega_orbit * r

# 计算左右轮速度
v_left = (v / 2) - (omega * L / 2)
v_right = (v / 2) + (omega * L / 2)

# 时间参数
dt = 0.1  # 时间步长（秒）
total_time = 20.0  # 总时间（秒）
num_steps = int(total_time / dt)

# 初始化机器人的位置和角度
# 机器人初始位置在外部圆心（0,0）的右侧，距离为 r
x = r
y = 0.0
theta = 0.0  # 初始角度，相对于机器人自身坐标系

# 存储机器人的位置和角度历史
x_history = [x]
y_history = [y]
theta_history = [theta]
orbit_history = [0.0]  # 绕外部圆心的旋转角度历史

# 更新机器人位置和角度
for i in range(num_steps):
    # 更新绕外部圆心的旋转角度
    orbit_angle = omega_orbit * (i + 1) * dt
    orbit_history.append(orbit_angle)
    
    # 更新机器人位置（相对于外部圆心）
    x = r * np.cos(orbit_angle)
    y = r * np.sin(orbit_angle)
    
    # 更新机器人的自转角度
    theta += omega * dt
    
    # 添加到历史记录
    x_history.append(x)
    y_history.append(y)
    theta_history.append(theta)

# 设置绘图
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-2 * r, 2 * r)
ax.set_ylim(-2 * r, 2 * r)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title('Differential Robot Motion')

# 绘制外部圆心和目标轨迹
external_center = plt.scatter(0, 0, color='blue', zorder=5)
target_circle = Circle((0, 0), r, fill=False, color='blue', linestyle='--', label='Target Orbit')
ax.add_patch(target_circle)
trajectory, = ax.plot([], [], 'g-', alpha=0.5, label='Robot Trajectory')

# 绘制机器人
robot_body, = ax.plot([], [], 'k-', lw=2)
left_wheel = Rectangle((0, 0), wheel_radius, wheel_radius/2, color='red', alpha=0.7)
right_wheel = Rectangle((0, 0), wheel_radius, wheel_radius/2, color='red', alpha=0.7)
ax.add_patch(left_wheel)
ax.add_patch(right_wheel)

# 初始化动画
def init():
    robot_body.set_data([], [])
    left_wheel.set_xy((0, 0))
    right_wheel.set_xy((0, 0))
    trajectory.set_data([], [])
    return robot_body, left_wheel, right_wheel, trajectory

# 更新动画
def update(frame):
    # 获取当前状态
    x = x_history[frame]
    y = y_history[frame]
    theta = theta_history[frame]
    orbit_angle = orbit_history[frame]
    
    # 更新轨迹
    trajectory.set_data(x_history[:frame+1], y_history[:frame+1])
    
    # 更新机器人位置
    # 计算机器人机体的两个端点（考虑自转角度 theta）
    robot_x1 = x - (robot_length / 2) * np.cos(theta)
    robot_y1 = y - (robot_length / 2) * np.sin(theta)
    robot_x2 = x + (robot_length / 2) * np.cos(theta)
    robot_y2 = y + (robot_length / 2) * np.sin(theta)
    robot_body.set_data([robot_x1, robot_x2], [robot_y1, robot_y2])
    
    # 更新轮子位置和角度
    # 左轮
    left_wheel_x = x - (robot_length / 2) * np.cos(theta) - (wheel_radius / 2) * np.sin(theta)
    left_wheel_y = y - (robot_length / 2) * np.sin(theta) + (wheel_radius / 2) * np.cos(theta)
    left_wheel.set_xy((left_wheel_x, left_wheel_y))
    left_wheel.angle = np.degrees(theta)
    
    # 右轮
    right_wheel_x = x + (robot_length / 2) * np.cos(theta) - (wheel_radius / 2) * np.sin(theta)
    right_wheel_y = y + (robot_length / 2) * np.sin(theta) + (wheel_radius / 2) * np.cos(theta)
    right_wheel.set_xy((right_wheel_x, right_wheel_y))
    right_wheel.angle = np.degrees(theta)
    
    return robot_body, left_wheel, right_wheel, trajectory

# 创建动画
ani = FuncAnimation(fig, update, frames=num_steps, init_func=init, blit=True, interval=dt*1000)

# 添加图例
ax.legend()

# 保存为 GIF
writer = PillowWriter(fps=1/dt)
ani.save("robot_animation.gif", writer=writer)

plt.show()