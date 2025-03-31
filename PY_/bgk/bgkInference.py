'''
Author: chasey && melancholycy@gmail.com
Date: 2025-03-27 10:09:14
LastEditTime: 2025-03-31 13:42:23
FilePath: /test/PY_/bgk/bgkInference.py
Description: 
Reference: 
Copyright (c) 2025 by chasey && melancholycy@gmail.com, All Rights Reserved. 
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define the kernel function
def kernel(x, x_prime, l, s):
    d = np.abs(x - x_prime)
    if d <= l:
        term1 = (2 + np.cos(2 * np.pi * d / l)) / 3 * (1 - d / l)
        term2 = (1 / (2 * np.pi)) * np.sin(2 * np.pi * d / l)
        return s * (term1 + term2)
    else:
        return 0.0

# Vectorize the kernel for arrays
def kernel_matrix(X1, X2, l, s):
    return np.array([[kernel(x1, x2, l, s) for x2 in X2.flatten()] for x1 in X1.flatten()])

# Compute posterior mean and covariance
def compute_posterior(X_train, y_train, X_test, mu0, sigma2, l, s, lambda_):
    N = X_train.shape[0]

    # kernelMat shape: [predX.rows(), trainX.rows()]  (100, 50)
    K_star = kernel_matrix(X_test, X_train, l, s)
    # print(K_star.shape)

    sum_k = np.sum(K_star, axis=1)
    sum_ky = np.sum(K_star * y_train.T, axis=1)# axis = 1 sum for col(such as sum([100, dim])->[100, 1])  #[100, 50] @ [50, 1] = [100, 1]
    # print(sum_ky.shape)
    
    # Compute posterior mean
    mean = (lambda_ * mu0 + sum_ky) / (lambda_ + sum_k)
    
    # Compute posterior covariance
    cov = sigma2 / (lambda_ + sum_k)
    
    return mean, cov

# Negative log marginal likelihood function
def neg_log_marginal_likelihood(params, X_train, y_train, mu0, sigma2):
    l, s, lambda_ = params
    K = kernel_matrix(X_train, X_train, l, s)
    K += np.eye(len(X_train)) * sigma2  # Add noise
    try:
        K_inv = np.linalg.inv(K)
    except np.linalg.LinAlgError:
        return np.inf  # Return a large value if matrix is singular
    term1 = 0.5 * np.dot(y_train.T, np.dot(K_inv, y_train))
    term2 = 0.5 * np.linalg.slogdet(K)[1]
    term3 = 0.5 * len(y_train) * np.log(2 * np.pi)
    return term1 + term2 + term3

# Generate dataset
np.random.seed(42)
X_train = np.linspace(0, 2 * np.pi, 50)[:, None]
y_train = np.sin(X_train) + np.random.normal(0, 0.1, X_train.shape)
X_test = np.linspace(0, 2 * np.pi, 100)[:, None]

# Hyperparameter optimization
mu0 = 0.0
sigma2 = 0.1**2
initial_params = [0.5, 1.0, 0.01]  # Initial guess for l, s, lambda_


# # Optimize hyperparameters
# result = minimize(neg_log_marginal_likelihood, initial_params, args=(X_train, y_train, mu0, sigma2), 
#                   method='L-BFGS-B', bounds=[(0.1, None), (0.1, None), (0.1, None)])
# optimal_params = result.x
# l_opt, s_opt, lambda_opt = optimal_params

l_opt, s_opt, lambda_opt = initial_params


# Compute posterior with optimized hyperparameters
mean, cov = compute_posterior(X_train, y_train, X_test, mu0, sigma2, l_opt, s_opt, lambda_opt)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(X_train, y_train, 'ko', label='Training data')
plt.plot(X_test, mean, 'b-', label='Posterior mean')
plt.fill_between(X_test.flatten(), mean - np.sqrt(cov), mean + np.sqrt(cov), color='blue', alpha=0.2, label='Posterior covariance')
plt.plot(X_test, np.sin(X_test), 'r--', label='True function')
plt.legend()
plt.title('Gaussian Process Regression with Custom Kernel and Optimized Hyperparameters')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()

# Print optimized hyperparameters
print(f"Optimized hyperparameters:")
print(f"l = {l_opt:.4f}")
print(f"s = {s_opt:.4f}")
print(f"lambda = {lambda_opt:.4f}")