import gymnasium as gym
import numpy as np

# 创建 Pendulum-v2 环境
env = gym.make('Pendulum-v1', render_mode='human')  # 设置为可视化模式

# 初始化环境
state, info = env.reset()

# 运行一定步数
for _ in range(1000):
    # 随机选择一个动作（这里只是示例，随机动作无法真正保持平衡）
    action = np.random.uniform(-2, 2, size=1)  # 动作范围是 [-2, 2]
    
    # 执行动作
    next_state, reward, terminated, truncated, info = env.step(action)
    
    # 如果终止或截断，则重新初始化环境
    if terminated or truncated:
        state, info = env.reset()
    else:
        state = next_state

# 关闭环境
env.close()