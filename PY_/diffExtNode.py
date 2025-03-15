'''
Author: chasey melancholycy@gmail.com
Date: 2025-02-22 12:41:08
FilePath: /mesh_planner/test/python/diffExtNode.py
Description:

Copyright (c) 2025 by chasey (melancholycy@gmail.com), All Rights Reserved. 
'''
import numpy as np
import matplotlib.pyplot as plt

param = {
    'vMax': 1.0,
    'wMax': 2.0,
    
    # 'vDelta': 0.5,
    # 'wDelta': 1.0,    

    'dt': 1.0,
    'collSampleTime': 0.2
}

# SE(2) state 
cur = np.array([0.0, 0.0, 0.0])

def stateTrans(x, u, dt):
    x_new = np.zeros_like(x)
    x_new[0] = x[0] + np.cos(x[2]) * u[0] * dt
    x_new[1] = x[1] + np.sin(x[2]) * u[0] * dt
    x_new[2] = x[2] + u[1] * dt
    return x_new

trajectories = []
for v in np.arange(-(param['vMax']+1e-5), param['vMax']+1e-5, param['vMax']/2):
    for w in np.arange(-param['wMax']-1e-5, param['wMax']+1e-5, param['wMax']/2):
        # 初始状态
        x_temp = cur.copy()
        trajectory = [x_temp[:2]]  # 存储 (x, y) 轨迹
        for t in np.arange(0, param['dt'], param['collSampleTime']):
            # 更新状态
            x_temp = stateTrans(x_temp, np.array([v, w]), param['collSampleTime'])
            trajectory.append(x_temp[:2])  # 保存 (x, y)
        trajectories.append((np.array(trajectory), v, w))  # 保存轨迹和对应的 v, w


# 可视化所有轨迹
plt.figure(figsize=(10, 10))
for trajectory, v, w in trajectories:
    # 绘制轨迹点
    plt.plot(trajectory[:, 0], trajectory[:, 1], label=f'v={v:.1f}, w={w:.1f}')
    # 绘制箭头
    for i in range(len(trajectory) - 1):
        pt1 = trajectory[i]
        pt2 = trajectory[i + 1]
        plt.arrow(pt1[0], pt1[1], pt2[0] - pt1[0], pt2[1] - pt1[1], 
                  head_width=0.01, head_length=0.05, fc='k', ec='k')

# 设置图表标题和图例
plt.title("Trajectories with Different Velocities and Angular Velocities")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.axis('equal')  # 保持比例
plt.show()