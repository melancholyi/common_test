import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics.pairwise import euclidean_distances

# 1. 生成带噪声的 sin 曲线数据
np.random.seed(42)
N = 100
X = np.linspace(0, 2*np.pi, N).reshape(-1, 1)  # 输入 x 的位置 (100, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.5, N)  # 目标值 (100,)

# 2. 修正的 RBF 核函数（支持两个不同输入）
def rbf_kernel(X, Z, l):  # <--- 关键修改：接受两个输入
    pairwise_sq_dists = euclidean_distances(X, Z, squared=True)
    return np.exp(-pairwise_sq_dists / (2 * l**2))

# 3. 负对数边际似然函数（保持原逻辑）
def negative_log_mll(params, X, y, mu0, sigma):
    lambda_, l = params
    K = rbf_kernel(X, X, l)  # <--- 改为两个相同输入
    N = X.shape[0]
    
    s = K.sum(axis=1) - np.diag(K)  # 排除对角元素
    sum_kjyj = K.dot(y) - np.diag(K) * y
    
    lambda_plus_s = lambda_ + s
    mu = (lambda_ * mu0 + sum_kjyj) / lambda_plus_s
    residuals = y - mu
    
    term1 = -np.log(lambda_plus_s)
    term2 = (residuals**2 * lambda_plus_s) / (sigma**2)
    loss = 0.5 * np.sum(term1 + term2)
    return loss

# 4. 超参数优化
mu0 = 0.0
sigma = 0.2
initial_params = [1.0, 0.5]
bounds = [(1e-6, None), (1e-6, None)]

result = minimize(negative_log_mll, initial_params, args=(X, y, mu0, sigma),
                 bounds=bounds, method='L-BFGS-B')
optimal_lambda, optimal_l = result.x
print(f"Optimal lambda: {optimal_lambda:.3f}, Optimal l: {optimal_l:.3f}")

# 5. 修正的预测函数
def predict(X_train, y_train, X_test, lambda_, l, mu0, sigma):
    # 计算测试点与训练点之间的核矩阵 (200, 100)
    K_train_test = rbf_kernel(X_test, X_train, l)  # <--- 关键修改
    
    # 计算每个测试点的核权重和 (200,)
    sum_k = K_train_test.sum(axis=1)
    # 计算加权目标值和 (200,)
    sum_kjyj = K_train_test.dot(y_train)
    
    lambda_plus_sumk = lambda_ + sum_k
    mu = (lambda_ * mu0 + sum_kjyj) / lambda_plus_sumk
    var = sigma**2 / lambda_plus_sumk
    return mu, var

# 6. 生成测试点并进行预测
X_test = np.linspace(0, 2*np.pi, 200).reshape(-1, 1)
mu_pred, var_pred = predict(X, y, X_test, optimal_lambda, optimal_l, mu0, sigma)

# 7. 可视化结果
plt.figure(figsize=(10, 6))
plt.scatter(X, y, c='r', s=20, label='Noisy Data')
plt.plot(X_test, np.sin(X_test), 'b-', label='True sin(x)')
plt.plot(X_test, mu_pred, 'k--', lw=2, label='Predicted Mean')
plt.fill_between(X_test.ravel(),
                 mu_pred - 2*np.sqrt(var_pred),
                 mu_pred + 2*np.sqrt(var_pred),
                 color='gray', alpha=0.3, label='95% Confidence')
plt.title(f"Bayesian Regression (λ={optimal_lambda:.2f}, l={optimal_l:.2f})")
plt.legend()
plt.show()