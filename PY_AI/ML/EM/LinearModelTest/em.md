EM算法（Expectation-Maximization Algorithm）通常用于处理含有隐变量的概率模型，例如高斯混合模型（GMM）或隐马尔可夫模型（HMM）。虽然EM算法本身并不是直接用于回归任务的，但可以将其应用于某些特定的回归模型，例如混合线性回归模型（Mixture of Linear Regressions）。

以下是一个使用EM算法实现混合线性回归模型的公式和示例代码。

### 混合线性回归模型与EM算法

#### 模型定义
假设我们有一个数据集 \((X, Y)\)，其中 \(X\) 是输入特征，\(Y\) 是目标变量。混合线性回归模型假设数据来自 \(K\) 个不同的线性回归模型，每个模型对应一个高斯分布。每个数据点 \((x_i, y_i)\) 由以下过程生成：
1. 从 \(K\) 个模型中选择一个，选择的概率为 \(\pi_k\)（混合权重）。
2. 使用选中的模型 \(k\) 生成 \(y_i\)，即 \(y_i = w_k^T x_i + \epsilon_i\)，其中 \(\epsilon_i \sim \mathcal{N}(0, \sigma_k^2)\)。

#### 目标
使用EM算法估计模型参数：
- 混合权重 \(\pi_k\)
- 线性回归权重 \(w_k\)
- 方差 \(\sigma_k^2\)

### EM算法步骤

#### E步（期望步）
计算每个数据点 \((x_i, y_i)\) 属于每个模型 \(k\) 的后验概率（责任）：
\[
\gamma_{ik} = \frac{\pi_k \mathcal{N}(y_i | w_k^T x_i, \sigma_k^2)}{\sum_{j=1}^K \pi_j \mathcal{N}(y_i | w_j^T x_i, \sigma_j^2)}
\]

#### M步（最大化步）
更新模型参数：
1. 更新混合权重：
\[
\pi_k = \frac{1}{N} \sum_{i=1}^N \gamma_{ik}
\]
2. 更新线性回归权重：
\[
w_k = \left( \sum_{i=1}^N \gamma_{ik} x_i x_i^T \right)^{-1} \left( \sum_{i=1}^N \gamma_{ik} x_i y_i \right)
\]
3. 更新方差：
\[
\sigma_k^2 = \frac{1}{N} \sum_{i=1}^N \gamma_{ik} (y_i - w_k^T x_i)^2
\]

### 示例代码

以下是一个使用Python和NumPy实现混合线性回归模型的示例代码：

```python
import numpy as np
import matplotlib.pyplot as plt

# 生成模拟数据
np.random.seed(42)
N = 1000
K = 2

# 模拟两个线性模型
X = np.random.rand(N, 1) * 10
Y1 = 2 * X + 1 + np.random.normal(0, 1, (N, 1))
Y2 = -1 * X + 3 + np.random.normal(0, 1, (N, 1))

# 混合数据
Z = np.random.binomial(1, 0.5, (N, 1))
Y = Z * Y1 + (1 - Z) * Y2

# 添加偏置项
X = np.hstack((np.ones((N, 1)), X))

# EM算法参数初始化
pi = np.ones(K) / K
w = np.random.randn(K, 2)
sigma2 = np.ones(K)

# EM算法迭代
n_iter = 100
for _ in range(n_iter):
    # E步：计算责任
    gamma = np.zeros((N, K))
    for k in range(K):
        # 确保 Y 和 X @ w[k].reshape(-1, 1) 的形状一致
        Y_flat = Y.flatten()  # 将 Y 转换为一维数组
        Xw = X @ w[k].reshape(-1, 1)  # X @ w[k] 的结果是 (N, 1)
        Xw_flat = Xw.flatten()  # 将 Xw 转换为一维数组
        
        # 计算高斯概率密度
        prob = pi[k] * np.exp(-0.5 * ((Y_flat - Xw_flat) ** 2) / sigma2[k])
        gamma[:, k] = prob  # 确保 gamma[:, k] 是一维数组
    
    # 归一化责任
    gamma = gamma / gamma.sum(axis=1, keepdims=True)
    
    # M步：更新参数
    for k in range(K):
        Nk = gamma[:, k].sum()
        # 更新混合权重
        pi[k] = Nk / N
        
        # 更新线性回归权重
        X_weighted = X * gamma[:, k].reshape(-1, 1)
        w[k] = np.linalg.inv(X.T @ X_weighted) @ (X_weighted.T @ Y).flatten()
        
        # 更新方差
        residuals = Y - X @ w[k].reshape(-1, 1)
        sigma2[k] = (gamma[:, k] * (residuals ** 2)).sum() / Nk

# 输出结果
print("混合权重:", pi)
print("回归权重:", w)
print("方差:", sigma2)

# 绘图
plt.scatter(X[:, 1], Y, alpha=0.5, label='Data')
for k in range(K):
    x_line = np.linspace(0, 10, 100)
    y_line = w[k, 0] + w[k, 1] * x_line
    plt.plot(x_line, y_line, label=f'Model {k+1}')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Mixture of Linear Regressions with EM Algorithm')
plt.show()
```

### 代码说明
1. **数据生成**：生成了两个线性模型的数据，并将它们混合。
2. **EM算法**：
   - **E步**：计算每个数据点属于每个模型的责任（后验概率）。
   - **M步**：更新混合权重、线性回归权重和方差。
3. **结果**：输出混合权重、回归权重和方差，并绘制结果图。

运行上述代码后，你将看到两个线性回归模型的拟合结果，以及它们的混合权重和方差。