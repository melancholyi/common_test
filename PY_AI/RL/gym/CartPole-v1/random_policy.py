'''
Author: chasey melancholycy@gmail.com
Date: 2025-03-07 03:30:06
FilePath: /mesh_planner/test/RL/gym/helloworld.py
Description: 

Copyright (c) 2025 by chasey (melancholycy@gmail.com), All Rights Reserved. 
'''
import gym

# 创建环境并指定渲染模式
env = gym.make('CartPole-v1', render_mode="human")

# 初始化环境
state, info = env.reset()

# 运行一定步数
for _ in range(1000):
    action = env.action_space.sample()  # 随机选择一个动作
    next_state, reward, terminated, truncated, info = env.step(action)  # 正确解包返回值
    
    if terminated or truncated:
        state, info = env.reset()

# 关闭环境
env.close()