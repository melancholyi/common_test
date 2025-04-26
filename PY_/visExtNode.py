'''
Author: chasey melancholycy@gmail.com
Date: 2025-02-19 09:14:24
FilePath: /test/PY_/visExtNode.py
Description: 

Copyright (c) 2025 by chasey (melancholycy@gmail.com), All Rights Reserved. 
'''
import numpy as np
import matplotlib.pyplot as plt

# 定义状态转移函数
def state_trans(x, u, dt):
    """
    状态转移函数
    :param x: 当前状态 [x, y, theta]
    :param u: 控制输入 [v, w]
    :param dt: 时间步长
    :return: 新状态
    """
    x_new = np.zeros_like(x)
    x_new[0] = x[0] + np.cos(x[2]) * u[0] * dt
    x_new[1] = x[1] + np.sin(x[2]) * u[0] * dt
    x_new[2] = x[2] + u[1] * dt
    return x_new

# 参数定义
param = {
    'vMax': 2.0,
    'wMax': 1.0,
    'dt': 1.0,
    'collSampleTime': 0.06
}

# 初始状态
cur = np.array([0.0, 0.0, -0.51037])  # 初始位置 (x, y) 和方向 theta

# 用于存储轨迹的列表
trajectories = []

# 遍历速度和角速度
for v in np.arange(0, param['vMax']+1e-5, 1.0):
    for w in np.arange(-param['wMax']-1e-5, param['wMax']+1e-5, param['wMax']/2):
        # 初始状态
        x_temp = cur.copy()
        trajectory = [x_temp[:2]]  # 存储 (x, y) 轨迹
        for t in np.arange(0, param['dt'], param['collSampleTime']):
            # 更新状态
            x_temp = state_trans(x_temp, np.array([v, w]), param['collSampleTime'])
            trajectory.append(x_temp[:2])  # 保存 (x, y)
        trajectories.append((np.array(trajectory), v, w))  # 保存轨迹和对应的 v, w

# 可视化轨迹
plt.figure(figsize=(8, 8))
for trajectory, v, w in trajectories:
    plt.plot(trajectory[:, 0], trajectory[:, 1], label=f"v={v:.1f}, w={w:.1f}")
    plt.scatter(trajectory[:, 0], trajectory[:, 1], color='blue', label='Exted Point')


plt.scatter(cur[0], cur[1], color='red', label='Start Point')
plt.title("Trajectories of the Robot")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()