'''
Author: chasey && melancholycy@gmail.com
Date: 2025-04-23 08:53:14
LastEditTime: 2025-04-23 08:53:16
FilePath: /test/PY_/plot/plotAcos.py
Description: 
Reference: 
Copyright (c) 2025 by chasey && melancholycy@gmail.com, All Rights Reserved. 
'''
import numpy as np
import matplotlib.pyplot as plt

# 定义 x 值范围（反余弦函数的定义域是 [-1, 1]）
x = np.linspace(-1, 1, 400)  # 生成从 -1 到 1 的 400 个点

# 计算反余弦值
y = np.arccos(x)

# 创建图形
plt.figure(figsize=(8, 6))

# 绘制反余弦函数
plt.plot(x, y, label='acos(x)', color='b')

# 添加标题和标签
plt.title('acos(x) Function')
plt.xlabel('x')
plt.ylabel('acos(x)')

# 添加网格线
plt.grid(True)

# 添加图例
plt.legend()

# 显示图形
plt.show()