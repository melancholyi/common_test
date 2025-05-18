import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 0. parameters  
torch.manual_seed(42)
N = 500
D = 2
noise_data = 0.01
optimizer_type = 'Adam' # 'Adam' or 'LBFGS'
optim_lr = 0.005
epochs = 1000
optim_maxiter = 1000

# 1. 生成带噪声的3D高程数据 (PyTorch Tensor)
min_val = 0
max_val = 10
res = 0.2
Ngrid = int((max_val - min_val) / res) + 1
points = [torch.linspace(min_val, max_val, Ngrid) for _ in range(D)] # 0, 0.2, 0.4, ..., 10 | all point count: 10/0.2+1=51
grid = torch.meshgrid(*points, indexing='ij')  # 使用 ij 索引方式
X_train_grid = torch.stack([grid[i].flatten() for i in range(D)], dim=-1)
X_train = X_train_grid #NOTE:torch.Size([2601, 2])  [51x51, 2]
print("X_train.shape: ", X_train.shape)
true_elevation = lambda x: 10 * torch.sin(5*x[:, 0]) + 10 * torch.cos(5 * x[:, 1]) + 10 * x[:, 0] + x[:, 1]
# true_elevation = lambda x: 10 * torch.sin(1*x[:, 0]) + 10 * torch.cos(1 * x[:, 1]) + 10 * x[:, 0] + x[:, 1]
y_train = true_elevation(X_train) + torch.randn(X_train.size(0)) * noise_data  # NOTE:torch.Size([2601])
print("y_train.shape: ", y_train.shape)

# 2. 定义PyTorch版本的RBF核函数
def sparseKernel(X, Z, kLen, kScalar):
    M2PI = 2 * torch.pi
    cdist = torch.cdist(X, Z)  # 欧氏距离平方 [m, n] m:predX.shape, n:trainX.shape
    print("cdist.shape: ", cdist.shape) 
    cdist /= kLen.unsqueeze(1) # kLen : length scale should  shape:[m]
    kernel = ((2 + (cdist * M2PI).cos()) * (1 - cdist) / 3.0 + (cdist * M2PI).sin() / M2PI)
    kernel = kernel * (kernel > 0.0)
    kernel *= kScalar.unsqueeze(1)  # kScalar : kernel scale should  shape:[m]
    # print(f'sparse kernel_shape: {kernel.shape}')
    return kernel

# 3. 负对数边际似然函数 (PyTorch可微分版本)
def negative_log_mll(params, X, y, mu0, sigma):
    """
    params: (kLen, kScalar) hyperparameters waiting to be optimized
    X: training data (N, D)
    y: training labels (N,)
    mu0: prior mean (scalar)
    sigma: prior variance (scalar)
    """
    kLen, kScalar = params
    lambda_ = 0.01
    K = sparseKernel(X, X, kLen, kScalar)

    # basic variables
    s = K.sum(dim=1) - torch.diag(K)  # 排除对角元素
    sum_kjyj = K @ y - torch.diag(K) * y  # 排除j=i项
    
    # compute posterior mean
    mu = (lambda_ * mu0 + sum_kjyj) / (lambda_ + s)

    # compute negative log marginal likelihood nll-loss
    term1 = -torch.log(lambda_ + s)
    term2 = ((y - mu)**2 * (lambda_ + s)) / (sigma**2)
    loss = 0.5 * torch.sum(term1 + term2)
    return loss

# 4. 超参数优化 (PyTorch LBFGS)
# 初始化可训练参数
lambda_ = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
# l = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
kLen = torch.ones(1, dtype=torch.float32, requires_grad=True)
kScalar = torch.ones(1, dtype=torch.float32, requires_grad=False)
mu0 = y_train.detach()
sigma = torch.tensor(1.0, dtype=torch.float32)

if optimizer_type == 'LBFGS':
    # 使用LBFGS优化器
    optimizer = torch.optim.LBFGS([kLen, kScalar], lr=optim_lr, max_iter=optim_maxiter)
elif optimizer_type == 'Adam':
    # 使用Adam优化器
    optimizer = torch.optim.Adam([kLen, kScalar], lr=optim_lr)
    print("use Adam")

def closure():
    optimizer.zero_grad()
    loss = negative_log_mll((kLen, kScalar), X_train, y_train, mu0, sigma)
    loss.backward()
    print(f'Loss: {loss.item():.4f}, λ: {lambda_.item():.4f}')
    return loss

# 运行优化
last_loss = 0
for i in range(epochs):  # LBFGS可能需要多次调用closure
    loss_now = optimizer.step(closure)
    if np.abs((loss_now - last_loss).detach().numpy()) < 0.01:
        print(f"break at index: {i}")
        break

    last_loss = loss_now

optimal_lambda = lambda_.item() if lambda_.item() > 0.0 else 0.0


# #============================================ cutting line ===================================================
# # ATTENTION: vis Train data(2.5D plot) and vis kLen by using color-map
# # Reshape y_train to 51x51 for visualization
# y_train_reshaped = y_train.reshape(51, 51)

# # Create the visualization
# fig, axs = plt.subplots(1, 3, figsize=(15, 6))  # 1 row, 2 columns

# # Plot y_train
# cax1 = axs[0].imshow(y_train_reshaped, cmap='viridis', extent=[min_val, max_val, min_val, max_val])
# axs[0].set_title("y_train Visualization")
# axs[0].set_xlabel("X-axis")
# axs[0].set_ylabel("Y-axis")
# fig.colorbar(cax1, ax=axs[0])

# # Plot l (all ones)
# # Reshape to [51, 51]
# kLen_reshaped = kLen.reshape(51, 51)
# # Convert to numpy array for visualization (required by matplotlib)
# kLen_numpy = kLen_reshaped.detach().cpu().numpy()  # detach from computation graph and move to CPU
# cax2 = axs[1].imshow(kLen_numpy, cmap='viridis', extent=[min_val, max_val, min_val, max_val])
# axs[1].set_title("l Visualization")
# axs[1].set_xlabel("X-axis")
# axs[1].set_ylabel("Y-axis")
# fig.colorbar(cax2, ax=axs[1])
# # Plot kScalar (all ones)
# # Reshape to [51, 51]
# kScalar_reshaped = kScalar.reshape(51, 51)
# # Convert to numpy array for visualization (required by matplotlib)
# kScalar_numpy = kScalar_reshaped.detach().cpu().numpy()  # detach from computation graph and move to CPU
# cax3 = axs[2].imshow(kScalar_numpy, cmap='viridis', extent=[min_val, max_val, min_val, max_val])
# axs[2].set_title("kScalar Visualization")
# axs[2].set_xlabel("X-axis")
# axs[2].set_ylabel("Y-axis")
# # Add colorbars
# fig.colorbar(cax3, ax=axs[2])
# # Adjust layout
# plt.tight_layout()
# # Show the plot
# plt.show()

######################################################## cutting line #####################################################

# 5. 预测函数 (修正参数顺序)
def predict(X_train, y_train, X_test, lambda_, kLen, kScalar, mu0, sigma):  # <-- 修正参数列表
    K_train_test = sparseKernel(X_test, X_train, kLen, kScalar)
    sum_k = K_train_test.sum(dim=1)  # (n_test,)
    sum_kjyj = K_train_test @ y_train  # (n_test,)
    
    lambda_plus_sumk = lambda_ + sum_k
    mu = (lambda_ * mu0 + sum_kjyj) / lambda_plus_sumk
    var = sigma**2 / lambda_plus_sumk
    return mu.detach(), var.detach()

# 6. 生成网格测试点 (转换为Tensor)
slice_count = 51
x_grid = torch.linspace(0, 10, slice_count)
y_grid = torch.linspace(0, 10, slice_count)
X_test = torch.stack(torch.meshgrid(x_grid, y_grid, indexing='xy'), dim=-1).reshape(-1, 2)  # (1601, 2)

# 进行预测 (修正调用参数)
mu_pred, var_pred = predict(X_train, y_train, X_test, optimal_lambda, kLen, kScalar, mu0, sigma)  # <-- 添加y_train

# 计算MSE
y_test = true_elevation(X_test)
mse = torch.mean((mu_pred - y_test)**2).item()
print(f"Test MSE: {mse:.4f}")

# 7. 可视化 (转换为NumPy)
# 7. 三维可视化（完整三视图）
X_test_np = X_test.numpy()  # 转换为NumPy数组用于可视化
X_mesh = X_test_np[:, 0].reshape(slice_count, slice_count)
Y_mesh = X_test_np[:, 1].reshape(slice_count, slice_count)
Z_mesh_true = true_elevation(X_test).detach().numpy().reshape(slice_count, slice_count)  # 真实高程
Z_mesh_pred = mu_pred.numpy().reshape(slice_count, slice_count)  # 预测高程
Z_var = var_pred.numpy().reshape(slice_count, slice_count)  # 预测方差

fig = plt.figure(figsize=(18, 6))

# 真实高程曲面
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X_mesh, Y_mesh, Z_mesh_true, 
                cmap='viridis', alpha=0.8)
ax1.scatter(X_train[:,0].numpy(), X_train[:,1].numpy(), y_train.numpy(), 
           c='r', s=1, label='Noisy Data')
ax1.set_title('True Elevation Map')

# 预测曲面
ax2 = fig.add_subplot(132, projection='3d')
surf = ax2.plot_surface(X_mesh, Y_mesh, Z_mesh_pred, cmap='plasma', alpha=0.8)
ax2.scatter(X_train[:,0].numpy(), X_train[:,1].numpy(), y_train.numpy(), 
           c='r', s=1)
ax2.set_title(f'Predicted Elevation\nMSE={mse:.2f}')
fig.colorbar(surf, ax=ax2)

# 不确定度曲面
ax3 = fig.add_subplot(133, projection='3d')
var_surf = ax3.plot_surface(X_mesh, Y_mesh, Z_var, 
                          cmap='coolwarm', alpha=0.8)
ax3.set_title('Prediction Variance (σ²)')
fig.colorbar(var_surf, ax=ax3)

plt.tight_layout()
plt.show()