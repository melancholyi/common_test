'''
Author: chasey melancholycy@gmail.com
Date: 2025-02-25 12:02:38
FilePath: /mesh_planner/test/pathPlanner/FMM/scikit-fmm/test.py
Description: 

Copyright (c) 2025 by chasey (melancholycy@gmail.com), All Rights Reserved. 
'''
import numpy as np
import skfmm

# 定义一个简单的圆形界面
N = 100  # 网格分辨率
x, y = np.linspace(-5, 5, N), np.linspace(-5, 5, N)
X, Y = np.meshgrid(x, y)
phi = X**2 + Y**2 - 1  # 半径为1的圆

# 计算符号距离函数
distance = skfmm.distance(phi)

# 查询点 P(2,2) 的距离
P = (2, 2)
index = (np.abs(x - P[0])).argmin(), (np.abs(y - P[1])).argmin()
print(f"点 P(2,2) 的符号距离: {distance[index]}")