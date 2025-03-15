import numpy as np

# 定义两个高斯分布的参数
u1 = np.array([1.0, .20])  # 第一个分布的均值
cov1 = np.array([[1.0, 0.5], [0.5, 2.0]])  # 第一个分布的协方差矩阵
n1 = 1  # 第一个分布的点数量

u2 = np.array([3.0, 4.0])  # 第二个分布的均值
cov2 = np.array([[2.0, 1.0], [1.0, 3.0]])  # 第二个分布的协方差矩阵
n2 = 1  # 第二个分布的点数量

# 方法1：通过高斯分布参数合成的方法
# 计算合成后的均值
u = (n1 * u1 + n2 * u2) / (n1 + n2)

# # 计算合成后的协方差
# cov = (
#     n1 * (cov1 + np.outer(u1 - u, u1 - u)) +
#     n2 * (cov2 + np.outer(u2 - u, u2 - u))
# ) / (n1 + n2)

# 计算合成后的协方差
cov = (
    n1 * cov1 +
    n2 * cov2 + 
    np.outer(u1 - u2, u1 - u2) * n1 * n2 / (n1 + n2) 
) / (n1 + n2)


# 合成后的点数量
n3 = n1 + n2

print("通过高斯分布参数合成的方法：")
print("合成后的均值 u:", u)
print("合成后的协方差 cov:\n", cov)
print("合成后的点数量 n3:", n3)

# 方法2：直接用所有点计算的方法
# 生成分布两个的样本点
np.random.seed(0)  # 设置随机种子以保证结果可重复
points1 = np.random.multivariate_normal(u1, cov1, n1)
points2 = np.random.multivariate_normal(u2, cov2, n2)

# 合并所有点
all_points = np.vstack((points1, points2))

# 计算合并后的均值和协方差
u_direct = np.mean(all_points, axis=0)
cov_direct = np.cov(all_points, rowvar=False)

print("\n直接用所有点计算的方法：")
print("合成后的均值 u:", u_direct)
print("合成后的协方差 cov:\n", cov_direct)