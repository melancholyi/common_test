'''
Author: chasey melancholycy@gmail.com
Date: 2025-03-07 03:17:11
FilePath: /mesh_planner/test/RL/pendulum-v1/pd_policy.py
Description: 

Copyright (c) 2025 by chasey (melancholycy@gmail.com), All Rights Reserved. 
'''
import gymnasium as gym
import numpy as np

# 创建倒立摆环境
env = gym.make("CartPole-v1", render_mode="human")

# 初始化环境
state, info = env.reset()
done = False
total_reward = 0.0

# 运行环境
for step in range(1000):  # 最大步数
    # 简单策略：根据角度和角速度选择动作
    angle = state[0]
    angular_velocity = state[1]
    action = -angle * 10.0 - angular_velocity * 3  # 简单的 PD 控制器

        # 将连续控制值映射到离散动作空间
    if action >= 0:
        action = 1  # 向右推车
    else:
        action = 0  # 向左推车
    
    # 执行动作并获取新的状态、奖励等信息
    next_state, reward, terminated, truncated, info = env.step(action)
    
    # 累计奖励
    total_reward += reward
    
    # 检查是否终止
    if terminated or truncated:
        print(f"Episode ended at step {step}. Total Reward: {total_reward}")
        break
    
    # 更新状态
    state = next_state

# 关闭环境
env.close()

print(f"Total Reward: {total_reward}")