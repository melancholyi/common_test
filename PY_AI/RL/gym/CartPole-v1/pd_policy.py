'''
Author: chasey melancholycy@gmail.com
Date: 2025-03-07 03:36:27
FilePath: /mesh_planner/test/RL/gym/pd_policy.py
Description: 

Copyright (c) 2025 by chasey (melancholycy@gmail.com), All Rights Reserved. 
'''
import gym
import numpy as np

# 创建 CartPole-v1 环境
env = gym.make('CartPole-v1', render_mode="human")

# PD 控制器增益（根据需要调整这些值）
Kp = 0.5  # 比例增益
Kd = 0.3  # 微分增益

# 初始化环境
state, info = env.reset()

# 控制循环
for _ in range(1000):
    # 提取状态变量
    x, x_dot, theta, theta_dot = state

    # 使用 PD 控制器计算控制值
    error = theta  # 误差是摆的角度
    error_dot = theta_dot  # 误差的导数是摆的角速度
    control_value = Kp * error + Kd * error_dot

    # 将连续控制值映射到离散动作空间
    if control_value >= 0:
        action = 1  # 向右推车
    else:
        action = 0  # 向左推车

    # 在环境中执行动作
    state, reward, terminated, truncated, info = env.step(action)

    # 检查是否需要重置环境
    if terminated or truncated:
        print("Episode ended. Resetting environment.")
        state, info = env.reset()

# 关闭环境
env.close()