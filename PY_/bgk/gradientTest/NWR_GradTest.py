# import numpy as np
# import matplotlib.pyplot as plt

# # Generate training data
# np.random.seed(42)
# train_n = 100
# pred_n = 200
# data_range_min = -5
# data_range_max = 5
# h = 0.2  # Bandwidth
# noise_std = 0.1

# def generateData(x):
#     return 2 * np.sin(x) + 4 * np.cos(x) + x**2


# X_train = np.linspace(data_range_min, data_range_max, train_n)
# y_true = generateData(X_train)
# noise = np.random.normal(0, noise_std, train_n)
# y_train = y_true + noise

# # Generate test grid
# xStar = np.linspace(data_range_min+2*h, data_range_max-2*h, pred_n)


# # Nadaraya-Watson regression and gradient computation
# y_pred = np.zeros_like(xStar)
# grad_pred = np.zeros_like(xStar)

# for i, x in enumerate(xStar):
#     # Kernel weights
#     u = (x - X_train) / h
#     kernel_vals = np.exp(-0.5 * u**2)
    
#     # Numerator and denominator for NW
#     numerator = np.sum(kernel_vals * y_train)
#     denominator = np.sum(kernel_vals)
#     y_pred[i] = numerator / denominator
    
#     # Derivatives for gradient
#     dK = (- (x - X_train) / h**2) * kernel_vals
#     N_prime = np.sum(dK * y_train)
#     D_prime = np.sum(dK)
#     if denominator != 0:
#         grad_pred[i] = (N_prime * denominator - numerator * D_prime) / (denominator**2)
#     else:
#         grad_pred[i] = 0

# # True gradient
# true_grad = 2 * np.cos(xStar) - 4 * np.sin(xStar) + 2 * xStar

# # Plot results
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.scatter(X_train, y_train, s=15, label='Training data', alpha=0.6)
# plt.plot(xStar, y_pred, 'r', label='NWR Estimate')
# plt.plot(xStar, 2*np.sin(xStar) + 4*np.cos(xStar) + xStar**2, 'g-', label='True Function')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend(fontsize=24)  # 设置字体大小为12

# plt.subplot(1, 2, 2)
# plt.plot(xStar, grad_pred, 'b', label='NWR Gradient')
# plt.plot(xStar, true_grad, 'm--', label='True Gradient')
# plt.xlabel('x')
# plt.ylabel('dy/dx')
# plt.legend(fontsize=24)  # 设置字体大小为12
# plt.tight_layout()
# plt.show()

# # Calculate MSE
# mse = np.mean((grad_pred - true_grad)**2)
# print(f"MSE between gradients: {mse:.4f}")



import numpy as np
import matplotlib.pyplot as plt

# Generate training data
np.random.seed(42)
train_n = 100
pred_n = 200
data_range_min = -5
data_range_max = 5
h = 0.2  # Bandwidth
noise_std = 0.01

def generateData(x):
    return 2 * np.sin(x) + 4 * np.cos(x) + x**2

X_train = np.linspace(data_range_min, data_range_max, train_n)
y_true = generateData(X_train)
noise = np.random.normal(0, noise_std, train_n)
y_train = y_true + noise

# Generate test grid
xStar = np.linspace(data_range_min+2*h, data_range_max-2*h, pred_n)

# Nadaraya-Watson regression and gradient computation
y_pred = np.zeros_like(xStar)
grad_pred = np.zeros_like(xStar)

for i, x in enumerate(xStar):
    u = (x - X_train) / h
    kernel_vals = np.exp(-0.5 * u**2)
    numerator = np.sum(kernel_vals * y_train)
    denominator = np.sum(kernel_vals)
    y_pred[i] = numerator / denominator
    dK = (- (x - X_train) / h**2) * kernel_vals
    N_prime = np.sum(dK * y_train)
    D_prime = np.sum(dK)
    grad_pred[i] = (N_prime * denominator - numerator * D_prime) / (denominator**2) if denominator !=0 else 0

# True gradient
true_grad = 2 * np.cos(xStar) - 4 * np.sin(xStar) + 2 * xStar

# 线性插值预测和梯度计算
y_linear = np.interp(xStar, X_train, y_train)
delta_x = X_train[1] - X_train[0]
dydx_train = np.diff(y_train) / delta_x
indices = ((xStar - X_train[0]) / delta_x).astype(int)
indices = np.clip(indices, 0, len(dydx_train)-1)
grad_linear = dydx_train[indices]

# Plot original results (NWR)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, s=15, label='Training data', alpha=0.9)
plt.plot(xStar, y_pred, 'r', label='NWR Estimate')
plt.plot(xStar, generateData(xStar), 'g-', label='True Function')
# Plot gradients
plt.quiver(xStar, y_pred, np.ones_like(xStar), grad_pred, 
           color='orange', angles='xy', scale_units='xy', scale=10, width=0.003,
           label='Gradient (Slope)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(fontsize=24)

plt.subplot(1, 2, 2)
plt.plot(xStar, grad_pred, 'b', label='NWR Gradient')
plt.plot(xStar, true_grad, 'm--', label='True Gradient')
plt.xlabel('x')
plt.ylabel('dy/dx')
plt.legend(fontsize=24)
plt.tight_layout()
plt.show()

# Plot linear interpolation results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, s=15, label='Training data', alpha=0.6)
plt.plot(xStar, y_linear, 'c', label='Linear Interpolation')
plt.plot(xStar, generateData(xStar), 'g-', label='True Function')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(fontsize=24)

plt.subplot(1, 2, 2)
plt.plot(xStar, grad_linear, 'b', label='Linear Gradient')
plt.plot(xStar, true_grad, 'm--', label='True Gradient')
plt.xlabel('x')
plt.ylabel('dy/dx')
plt.legend(fontsize=24)
plt.tight_layout()

# Calculate MSEs
mse_nwr = np.mean((grad_pred - true_grad)**2)
mse_linear = np.mean((grad_linear - true_grad)**2)
print(f"NWR Gradient MSE: {mse_nwr:.4f}")
print(f"Linear Interpolation Gradient MSE: {mse_linear:.4f}")


plt.show()

