'''
Author: chasey melancholycy@gmail.com
Date: 2025-02-19 12:08:11
FilePath: /test/PY_/visKinPath.py
Description: 

Copyright (c) 2025 by chasey (melancholycy@gmail.com), All Rights Reserved. 
'''
import numpy as np
import matplotlib.pyplot as plt

# 定义点的数据
#NOTE: points1 ################################################
# points = [
#     (0.0, 0.0, 0.0),
#     (0.970294, 0.196686, 0.499995),
#     (1.62848, 0.556251, 0.49999),
#     (2.38571, 1.19404, 0.999985),
#     (2.79094, 1.82513, 0.99998),
#     (3.19618, 2.45623, 0.999975),
#     (3.73651, 3.29768, 0.99997),
#     (4.27684, 4.13914, 0.999965),
#     (4.81717, 4.98059, 0.99996),
#     (5.35751, 5.82204, 0.999955),
#     (5.89785, 6.66348, 0.99995),
#     (6.30311, 7.29456, 0.999945),
#     (6.84346, 8.136, 0.99994),
#     (7.53327, 8.84616, 0.499935),
#     (8.19148, 9.20569, 0.49993),
#     (9.06909, 9.68505, 0.499925),
#     (10.0149, 9.97755, -8e-05)
# ]

#NOTE: points2 ##############################################
# points = [
#     (0, 0, 0),
#     (0.5, -1e-06, -5e-06),
#     (1.5, -8e-06, -1e-05),
#     (2.5, -2e-05, -1.5e-05),
#     (3.5, -3.7e-05, -2e-05),
#     (3.5, -3.7e-05, 0.999975),
#     (4.04033, 0.841419, 0.99997),
#     (4.58065, 1.68287, 0.999965),
#     (5.12099, 2.52432, 0.99996),
#     (5.66133, 3.36577, 0.999955),
#     (6.20167, 4.20722, 0.99995),
#     (6.74201, 5.04866, 0.999945),
#     (7.28236, 5.8901, 0.99994),
#     (7.82272, 6.73154, 0.999935),
#     (8.36308, 7.57297, 0.99993),
#     (8.90344, 8.41441, 0.999925),
#     (9.44381, 9.25583, 0.99992),
#     (9.98418, 10.0973, 0.999915),
# ]

#NOTE: points3
points = [
    (0, 0, 0),
    (0.970294, 0.196686, 0.499995),
    (1.62848, 0.556251, 0.49999),
    (2.38571, 1.19404, 0.999985),
    (2.79094, 1.82513, 0.99998),
    (3.19618, 2.45623, 0.999975),
    (3.73651, 3.29768, 0.99997),
    (4.27684, 4.13914, 0.999965),
    (4.81717, 4.98059, 0.99996),
    (5.35751, 5.82204, 0.999955),
    (5.89785, 6.66348, 0.99995),
    (6.30311, 7.29456, 0.999945),
    (6.84346, 8.136, 0.99994),
    (7.53327, 8.84616, 0.499935),
    (8.19148, 9.20569, 0.49993),
    (9.06909, 9.68505, 0.499925),
    (10.0149, 9.97755, -8e-05)
]


points = [
        (0, 0, 0),
        (0, 0, 0.149997),
        (0, 0, 0.299994),
        (0, 0, 0.449991),
        (0, 0, 0.599988),
        (0, 0, 0.749985),
        (0.43902, 0.408976, 0.749982),
        (0.878041, 0.817951, 0.749979),
        (1.31706, 1.22692, 0.749976),
        (1.75609, 1.6359, 0.749973),
        (2.19511, 2.04487, 0.74997),
        (2.63414, 2.45384, 0.749967),
        (3.07317, 2.86281, 0.749964),
        (3.51219, 3.27177, 0.749961),
        (3.95122, 3.68074, 0.749958),
        (4.36457, 4.1149, 0.899955),
        (4.73756, 4.58488, 0.899952),
        (5.11055, 5.05486, 0.899949),
        (5.48354, 5.52483, 0.899946),
        (5.85653, 5.99481, 0.899943),
        (6.22952, 6.46478, 0.89994),
        (6.41602, 6.69977, 0.899937),
        (6.78902, 7.16974, 0.899934),
        (7.16201, 7.63971, 0.899931),
        (7.53501, 8.10968, 0.899928),
        (7.93516, 8.55604, 0.749925),
        (8.39752, 8.93758, 0.599922),
        (8.64514, 9.10696, 0.599919),
        (9.15932, 9.41512, 0.449916),
        (9.72492, 9.6088, 0.149913),
]


points = [
        (0, 0, 0),
        (0, 0, 0.149997),
        (0, 0, 0.299994),
        (0, 0, 0.449991),
        (0, 0, 0.599988),
        (0, 0, 0.749985),
        (0.43902, 0.408976, 0.749982),
        (0.878041, 0.817951, 0.749979),
        (1.31706, 1.22692, 0.749976),
        (1.75609, 1.6359, 0.749973),
        (2.19511, 2.04487, 0.74997),
        (2.63414, 2.45384, 0.749967),
        (3.07317, 2.86281, 0.749964),
        (3.51219, 3.27177, 0.749961),
        (3.95122, 3.68074, 0.749958),
        (4.36457, 4.1149, 0.899955),
        (4.73756, 4.58488, 0.899952),
        (5.11055, 5.05486, 0.899949),
        (5.48354, 5.52483, 0.899946),
        (5.85653, 5.99481, 0.899943),
        (6.22952, 6.46478, 0.89994),
        (6.41602, 6.69977, 0.899937),
        (6.78902, 7.16974, 0.899934),
        (7.16201, 7.63971, 0.899931),
        (7.53501, 8.10968, 0.899928),
        (7.93516, 8.55604, 0.749925),
        (8.39752, 8.93758, 0.599922),
        (8.64514, 9.10696, 0.599919),
        (9.15932, 9.41512, 0.449916),
        (9.72492, 9.6088, 0.149913),
]

points = [
        (0, 0, 0),
        (0, 0, 0.149997),
        (0, 0, 0.299994),
        (0, 0, 0.449991),
        (0, 0, 0.599988),
        (0, 0, 0.749985),
        (0.43902, 0.408976, 0.749982),
        (0.878041, 0.817951, 0.749979),
        (1.31706, 1.22692, 0.749976),
        (1.75609, 1.6359, 0.749973),
        (2.19511, 2.04487, 0.74997),
        (2.63414, 2.45384, 0.749967),
        (3.07317, 2.86281, 0.749964),
        (3.29268, 3.06729, 0.749961),
        (3.73171, 3.47625, 0.749958),
        (4.17074, 3.88522, 0.749955),
        (4.60977, 4.29418, 0.749952),
        (5.04881, 4.70314, 0.749949),
        (5.48784, 5.1121, 0.749946),
        (5.92688, 5.52106, 0.749943),
        (6.36591, 5.93002, 0.74994),
        (6.80495, 6.33898, 0.749937),
        (7.24399, 6.74793, 0.749934),
        (7.68303, 7.15689, 0.749931),
        (8.09639, 7.59104, 0.899928),
        (8.46939, 8.06101, 0.899925),
        (8.84239, 8.53097, 0.899922),
        (9.2154, 9.00094, 0.899919),
        (9.5884, 9.47091, 0.899916),
        (9.96141, 9.94087, 0.899913),
]


# 定义点的数据
#NOTE: points1 ################################################
points = [
    (0.0, 0.0, 0.0),
    (0.970294, 0.196686, 0.499995),
    (1.62848, 0.556251, 0.49999),
    (2.38571, 1.19404, 0.999985),
    (2.79094, 1.82513, 0.99998),
    (3.19618, 2.45623, 0.999975),
    (3.73651, 3.29768, 0.99997),
    (4.27684, 4.13914, 0.999965),
    (4.81717, 4.98059, 0.99996),
    (5.35751, 5.82204, 0.999955),
    (5.89785, 6.66348, 0.99995),
    (6.30311, 7.29456, 0.999945),
    (6.84346, 8.136, 0.99994),
    (7.53327, 8.84616, 0.499935),
    (8.19148, 9.20569, 0.49993),
    (9.06909, 9.68505, 0.499925),
    (10.0149, 9.97755, -8e-05)
]


# ok////////////////////////////////////////////////////////////////////////////////////////
points = [
    (0.0, 0.0, 0.0),
    (0.970294, 0.196686, 0.499995),
    (1.62848, 0.556251, 0.49999),
    (2.38571, 1.19404, 0.999985),
    (2.79094, 1.82513, 0.99998),
    (3.19618, 2.45623, 0.999975),
    (3.73651, 3.29768, 0.99997),
    (4.27684, 4.13914, 0.999965),
    (4.81717, 4.98059, 0.99996),
    (5.35751, 5.82204, 0.999955),
    (5.89785, 6.66348, 0.99995),
    (6.30311, 7.29456, 0.999945),
    (6.84346, 8.136, 0.99994),
    (7.53327, 8.84616, 0.499935),
    (8.19148, 9.20569, 0.49993),
    (9.06909, 9.68505, 0.499925),
    (10.0149, 9.97755, -8e-05)
]
# ////////////////////////////////////
points = [
        (0, 0, 0),
        (0.863757, 0.417237, 0.99999),
        (1.40407, 1.2587, 0.99998),
        (2.48472, 2.94162, 0.99997),
        (3.89892, 4.32661, 0.49996),
        (5.38058, 5.63918, 0.99995),
        (6.79481, 7.02415, 0.49994),
        (8.2765, 8.33669, 0.99993),
        (9.69076, 9.72163, 0.49992),
]

points = [
    (0, 0, 0),
    (0.5, -1e-06, -5e-06),
    (1.5, -8e-06, -1e-05),
    (2.5, -2e-05, -1.5e-05),
    (3.5, -3.7e-05, -2e-05),
    (3.5, -3.7e-05, 0.999975),
    (4.04033, 0.841419, 0.99997),
    (4.58065, 1.68287, 0.999965),
    (5.12099, 2.52432, 0.99996),
    (5.66133, 3.36577, 0.999955),
    (6.20167, 4.20722, 0.99995),
    (6.74201, 5.04866, 0.999945),
    (7.28236, 5.8901, 0.99994),
    (7.82272, 6.73154, 0.999935),
    (8.36308, 7.57297, 0.99993),
    (8.90344, 8.41441, 0.999925),
    (9.44381, 9.25583, 0.99992),
    (9.98418, 10.0973, 0.999915),
]

points = [
        (0, 0, 0),
        (0, 0, 0.999995),
        (0.135077, 0.210367, 0.99999),
        (0.270155, 0.420733, 0.999985),
        (0.405234, 0.631098, 0.99998),
        (0.540315, 0.841463, 0.999975),
        (0.675396, 1.05183, 0.99997),
        (0.810478, 1.26219, 0.999965),
        (0.945561, 1.47255, 0.99996),
        (1.08065, 1.68292, 0.999955),
        (1.21573, 1.89328, 0.99995),
        (1.35082, 2.10364, 0.999945),
        (1.48591, 2.314, 0.99994),
        (1.62099, 2.52436, 0.999935),
        (1.75608, 2.73472, 0.99993),
        (1.89117, 2.94507, 0.999925),
        (2.02627, 3.15543, 0.99992),
        (2.16136, 3.36579, 0.999915),
        (2.29645, 3.57614, 0.99991),
        (2.364, 3.68132, 0.999905),
        (2.364, 3.68132, 1.9999),
        (2.364, 3.68132, 2.9999),
        (2.364, 3.68132, -2.2833),
        (2.364, 3.68132, -1.2833),
        (2.364, 3.68132, -0.283305),
        (2.364, 3.68132, 0.71669),
        (2.5525, 3.84554, 0.716685),
        (2.74099, 4.00977, 0.71668),
        (2.92949, 4.17399, 0.716675),
        (3.11799, 4.33821, 0.71667),
        (3.30649, 4.50243, 0.716665),
        (3.49499, 4.66664, 0.71666),
        (3.68349, 4.83086, 0.716655),
        (3.87199, 4.99508, 0.71665),
        (4.0605, 5.15929, 0.716645),
        (4.249, 5.32351, 0.71664),
        (4.43751, 5.48772, 0.716635),
        (4.62601, 5.65193, 0.71663),
        (4.81452, 5.81614, 0.716625),
        (5.00302, 5.98035, 0.71662),
        (5.19153, 6.14456, 0.716615),
        (5.38004, 6.30877, 0.71661),
        (5.56855, 6.47298, 0.716605),
        (5.75706, 6.63719, 0.7166),
        (5.94557, 6.80139, 0.716595),
        (6.13408, 6.9656, 0.71659),
        (6.3226, 7.1298, 0.716585),
        (6.51111, 7.294, 0.71658),
        (6.69963, 7.45821, 0.716575),
        (6.88814, 7.62241, 0.71657),
        (7.07666, 7.78661, 0.716565),
        (7.26517, 7.95081, 0.71656),
        (7.45369, 8.115, 0.716555),
        (7.64221, 8.2792, 0.71655),
        (7.83073, 8.4434, 0.716545),
        (8.01925, 8.60759, 0.71654),
        (8.20777, 8.77179, 0.716535),
        (8.39629, 8.93598, 0.71653),
        (8.58482, 9.10017, 0.716525),
        (8.77334, 9.26436, 0.71652),
        (8.96186, 9.42856, 0.716515),
        (9.15039, 9.59275, 0.71651),
        (9.24465, 9.67484, 0.716505),
        (9.43318, 9.83903, 0.7165),
        (9.62171, 10.0032, 0.716495),
        (9.7358, 10.0406, -0.28351),
        (9.97582, 9.97065, -0.283515),
]

points = [
        (0, 0, 0),
        (0, 0, 0.499995),
        (0.438793, 0.23971, 0.49999),
        (0.877587, 0.479417, 0.499985),
        (1.2562, 0.798308, 0.99998),
        (1.52636, 1.21904, 0.999975),
        (1.79652, 1.63977, 0.99997),
        (2.06669, 2.06049, 0.999965),
        (2.33685, 2.48122, 0.99996),
        (2.60702, 2.90194, 0.999955),
        (2.87719, 3.32266, 0.99995),
        (3.14737, 3.74339, 0.999945),
        (3.41754, 4.16411, 0.99994),
        (3.68772, 4.58483, 0.999935),
        (3.82281, 4.79518, 0.99993),
        (4.16771, 5.15026, 0.499925),
        (4.60652, 5.38994, 0.49992),
        (5.04534, 5.62962, 0.499915),
        (5.48415, 5.86929, 0.49991),
        (5.92296, 6.10896, 0.499905),
        (6.36178, 6.34863, 0.4999),
        (6.58118, 6.46847, 0.499895),
        (6.95983, 6.78732, 0.99989),
        (7.23002, 7.20803, 0.999885),
        (7.50022, 7.62873, 0.99988),
        (7.77043, 8.04944, 0.999875),
        (8.11535, 8.40449, 0.49987),
        (8.55417, 8.64415, 0.499865),
        (8.77359, 8.76397, 0.49986),
        (9.15224, 9.08282, 0.999855),
        (9.42245, 9.50351, 0.99985),
        (9.76739, 9.85856, 0.499845),
        (9.9868, 9.97838, 0.49984),
]

# points = [
#     (0, 0, 0),
#     (0.5, -1e-06, -5e-06),
#     (1.5, -8e-06, -1e-05),
#     (2.5, -2e-05, -1.5e-05),
#     (3.5, -3.7e-05, -2e-05),
#     (3.5, -3.7e-05, 0.999975),
#     (4.04033, 0.841419, 0.99997),
#     (4.58065, 1.68287, 0.999965),
#     (5.12099, 2.52432, 0.99996),
#     (5.66133, 3.36577, 0.999955),
#     (6.20167, 4.20722, 0.99995),
#     (6.74201, 5.04866, 0.999945),
#     (7.28236, 5.8901, 0.99994),
#     (7.82272, 6.73154, 0.999935),
#     (8.36308, 7.57297, 0.99993),
#     (8.90344, 8.41441, 0.999925),
#     (9.44381, 9.25583, 0.99992),
#     (9.98418, 10.0973, 0.999915),
# ]

points = [
    (0.0, 0.0, 0.0),
    (0.970294, 0.196686, 0.499995),
    (1.62848, 0.556251, 0.49999),
    (2.38571, 1.19404, 0.999985),
    (2.79094, 1.82513, 0.99998),
    (3.19618, 2.45623, 0.999975),
    (3.73651, 3.29768, 0.99997),
    (4.27684, 4.13914, 0.999965),
    (4.81717, 4.98059, 0.99996),
    (5.35751, 5.82204, 0.999955),
    (5.89785, 6.66348, 0.99995),
    (6.30311, 7.29456, 0.999945),
    (6.84346, 8.136, 0.99994),
    (7.53327, 8.84616, 0.499935),
    (8.19148, 9.20569, 0.49993),
    (9.06909, 9.68505, 0.499925),
    (10.0149, 9.97755, -8e-05)
]

# points = [
#         (0, 0, 0),
#         (1, 1, 0),
#         (2, 2, 0),
#         (3, 3, 0),
#         (4, 4, 0),
#         (5, 5, 0),
#         (6, 6, 0),
#         (7, 7, 0),
#         (8, 8, 0),
#         (9, 9, 0),
#         (10, 10, 0),
# ]

# 提取 x, y 和 yaw
x = [p[0] for p in points]
y = [p[1] for p in points]
yaw = [p[2] for p in points]

# 可视化
plt.figure(figsize=(9, 8))
plt.plot(x, y, 'o-', label="Path", markersize=6)  # 绘制轨迹

# 添加方向箭头
for i in range(len(points)):
    dx = np.cos(yaw[i]) * 0.4  # 箭头长度
    dy = np.sin(yaw[i]) * 0.4  # 箭头长度
    plt.arrow(x[i], y[i], dx, dy, head_width=0.1, head_length=0.4, fc='r', ec='r')

plt.title("Points with Direction Arrows", fontsize=24)  # 增加标题字体大小
plt.xlabel("X Position", fontsize=24)  # 增加 x 轴标签字体大小
plt.ylabel("Y Position", fontsize=24)  # 增加 y 轴标签字体大小
plt.legend(fontsize=24)  # 增加图例字体大小
plt.grid(True)
plt.axis('equal')  # 保持比例
plt.show()