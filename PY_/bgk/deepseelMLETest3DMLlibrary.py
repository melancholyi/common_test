'''
Author: chasey && melancholycy@gmail.com
Date: 2025-03-29 07:45:31
LastEditTime: 2025-03-29 07:52:05
FilePath: /test/PY_/bgk/deepseelMLETest3D.py
Description: 
Reference: 
Copyright (c) 2025 by chasey && melancholycy@gmail.com, All Rights Reserved. 
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics.pairwise import euclidean_distances
from mpl_toolkits.mplot3d import Axes3D

# 1. 生成带噪声的3D高程数据
np.random.seed(42)
N = 500
D = 2
X_train = np.random.rand(N, D) * 10  # 在10x10区域内随机生成(x,y)坐标
true_elevation = lambda x: 2 * np.sin(x[:,0]) + 3 * np.cos(x[:,1]) + x[:,0] + x[:,1]
y_train = true_elevation(X_train) + np.random.normal(0, 0.01, size=N)

# 2. 定义RBF核函数（支持多维输入）
def rbf_kernel(X, Z, l):
    pairwise_sq_dists = euclidean_distances(X, Z, squared=True)
    return np.exp(-pairwise_sq_dists / (2 * l**2))

# 3. 负对数边际似然函数
def negative_log_mll(params, X, y, mu0, sigma):
    lambda_, l = params
    K = rbf_kernel(X, X, l)
    N = X.shape[0]
    
    s = K.sum(axis=1) - np.diag(K)
    sum_kjyj = K.dot(y) - np.diag(K) * y
    
    lambda_plus_s = lambda_ + s
    mu = (lambda_ * mu0 + sum_kjyj) / lambda_plus_s
    residuals = y - mu
    
    term1 = -np.log(lambda_plus_s)
    term2 = (residuals**2 * lambda_plus_s) / (sigma**2)
    loss = 0.5 * np.sum(term1 + term2)
    return loss

# 4. 超参数优化
mu0 = y_train.mean()  # 使用训练数据的均值作为先验均值
sigma = 1.0  # 根据噪声水平设置
initial_params = [1.0, 2.0]
bounds = [(1e-6, None), (1e-6, None)]

result = minimize(negative_log_mll, initial_params, args=(X_train, y_train, mu0, sigma),
                 bounds=bounds, method='L-BFGS-B')
optimal_lambda, optimal_l = result.x
print(f"Optimal lambda: {optimal_lambda:.3f}, Optimal l: {optimal_l:.3f}")

# 5. 预测函数
def predict(X_train, y_train, X_test, lambda_, l, mu0, sigma):
    K_train_test = rbf_kernel(X_test, X_train, l)
    sum_k = K_train_test.sum(axis=1)
    sum_kjyj = K_train_test.dot(y_train)
    
    lambda_plus_sumk = lambda_ + sum_k
    mu = (lambda_ * mu0 + sum_kjyj) / lambda_plus_sumk
    var = sigma**2 / lambda_plus_sumk
    return mu, var

# 6. 生成网格测试点
x_grid = np.linspace(0, 10, 30)
y_grid = np.linspace(0, 10, 30)
X_test = np.array(np.meshgrid(x_grid, y_grid)).T.reshape(-1, 2)  # (900, 2)

# 预测均值和方差
mu_pred, var_pred = predict(X_train, y_train, X_test, optimal_lambda, optimal_l, mu0, sigma)

# 转换为网格矩阵
X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
Z_mesh = mu_pred.reshape(len(y_grid), len(x_grid))
Z_var = var_pred.reshape(len(y_grid), len(x_grid))

y_test_true = true_elevation(X_test)  # 真实高程值
mse = np.mean((mu_pred - y_test_true)**2)
print(f"\n预测均方误差 (MSE): {mse:.4f}")



# 7. 三维可视化
fig = plt.figure(figsize=(18, 6))

# 真实高程曲面
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X_mesh, Y_mesh, true_elevation(X_test).reshape(30,30), 
                cmap='viridis', alpha=0.8)
ax1.scatter(X_train[:,0], X_train[:,1], y_train, c='r', s=20, label='Noisy Data')
ax1.set_title('True Elevation Map')

# 预测曲面
ax2 = fig.add_subplot(132, projection='3d')
surf = ax2.plot_surface(X_mesh, Y_mesh, Z_mesh, cmap='plasma', alpha=0.8)
ax2.scatter(X_train[:,0], X_train[:,1], y_train, c='r', s=20)
ax2.set_title(f'Predicted Elevation\nMSE={mse:.2f}, λ={optimal_lambda:.2f}, l={optimal_l:.2f}')
fig.colorbar(surf, ax=ax2)

# 不确定度曲面
ax3 = fig.add_subplot(133, projection='3d')
var_surf = ax3.plot_surface(X_mesh, Y_mesh, Z_var, cmap='coolwarm', alpha=0.8)
ax3.set_title('Prediction Variance')
fig.colorbar(var_surf, ax=ax3)

plt.tight_layout()
plt.show()