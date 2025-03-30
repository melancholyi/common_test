<!--
 * @Author: chasey && melancholycy@gmail.com
 * @Date: 2025-03-29 14:35:36
 * @LastEditTime: 2025-03-29 14:36:46
 * @FilePath: /test/PY_/bgk/negMLLFunc.md
 * @Description: 
 * @Reference: 
 * Copyright (c) 2025 by chasey && melancholycy@gmail.com, All Rights Reserved. 
-->
以下是负对数边际似然损失函数的详细推导过程：

---
### 1. 模型设定
给定贝叶斯回归模型：
- 观测似然：$y_i \sim \mathcal{N}(\mu, \sigma^2)$
- 先验分布：$\mu \sim \mathcal{N}(\mu_0, \sigma^2/\lambda)$
- 核加权似然：$p(\mu^*|x^*, \mathcal{D}) \propto \prod_{i=1}^N \mathcal{N}(y_i|\mu, \sigma^2)^{k(x_i,x^*)} \cdot \mathcal{N}(\mu|\mu_0, \sigma^2/\lambda)$

### 2. 后验分布的均值和方差
通过共轭先验的性质，后验分布仍为高斯分布：
$$
\mu^* | \mathcal{D}, x^* \sim \mathcal{N}\left( \frac{\lambda\mu_0 + \sum_{i=1}^N k(x_i,x^*) y_i}{\lambda + \sum_{i=1}^N k(x_i,x^*)}, \ \frac{\sigma^2}{\lambda + \sum_{i=1}^N k(x_i,x^*)} \right)
$$

### 3. 留一预测分布
为计算边际似然，采用留一法（Leave-One-Out, LOO），即对每个样本 $i$，用其余样本预测 $y_i$。此时：
- 核权重和：$s_i = \sum_{j \neq i} k(x_i, x_j)$
- 预测均值：$\mu_i = \frac{\lambda\mu_0 + \sum_{j \neq i} k(x_i,x_j) y_j}{\lambda + s_i}$
- 预测方差：$\sigma_i^2 = \frac{\sigma^2}{\lambda + s_i}$

### 4. 负对数边际似然推导
每个观测的似然为高斯分布：
$$
p(y_i | \mathcal{D}_{-i}, x_i) = \mathcal{N}\left( y_i \big| \mu_i, \sigma_i^2 \right)
$$
其负对数似然为：
$$
-\log p(y_i | \mathcal{D}_{-i}, x_i) = \frac{1}{2} \log(2\pi\sigma_i^2) + \frac{(y_i - \mu_i)^2}{2\sigma_i^2}
$$
代入 $\sigma_i^2 = \frac{\sigma^2}{\lambda + s_i}$：
$$
= \frac{1}{2} \log\left( \frac{2\pi\sigma^2}{\lambda + s_i} \right) + \frac{(y_i - \mu_i)^2 (\lambda + s_i)}{2\sigma^2}
$$
忽略常数项 $\log(2\pi\sigma^2)$，简化得：
$$
= \frac{1}{2} \left[ -\log(\lambda + s_i) + \frac{(y_i - \mu_i)^2 (\lambda + s_i)}{\sigma^2} \right]
$$
总损失为所有样本的负对数似然之和：
$$
\text{loss} = \frac{1}{2} \sum_{i=1}^N \left[ -\log(\lambda + s_i) + \frac{(y_i - \mu_i)^2 (\lambda + s_i)}{\sigma^2} \right]
$$

### 5. 关键公式更正
用户原式中 **log项符号错误**，正确公式应包含负号：
$$
\boxed{
\text{loss} = \frac{1}{2} \sum_{i=1}^N \left[ -\log(\lambda + s_i) + \frac{(y_i - \mu_i)^2 (\lambda + s_i)}{\sigma^2} \right]
}
$$

### 6. 代码实现修正
```python
def negative_log_mll(params, X, y, mu0, sigma):
    lambda_, l = params
    K = rbf_kernel(X, l)
    N = X.shape[0]
    
    s = K.sum(axis=1) - np.diag(K)  # 排除自身核权重
    sum_kjyj = K.dot(y) - K.diagonal() * y  # 排除j=i的项
    
    lambda_plus_s = lambda_ + s
    mu = (lambda_ * mu0 + sum_kjyj) / lambda_plus_s
    residuals = y - mu
    
    # 正确符号：-log(lambda + s_i)
    term1 = -np.log(lambda_plus_s) 
    term2 = (residuals**2 * lambda_plus_s) / (sigma**2)
    loss = 0.5 * np.sum(term1 + term2)
    
    return loss
```

### 7. 推导总结
1. **后验分布**：通过贝叶斯公式结合高斯先验与似然，得到预测均值和方差。
2. **留一法**：计算每个点的预测分布时排除自身，避免过拟合。
3. **符号更正**：负对数似然中的对数项应为负号，确保损失函数正确性。

此推导确保了超参数优化的数学严谨性，代码实现中需特别注意核权重的计算与符号处理。