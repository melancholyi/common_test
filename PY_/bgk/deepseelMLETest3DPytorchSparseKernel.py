import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 0. parameters  
torch.manual_seed(42)
N = 500
D = 2
noise_data = 0.01
optimizer_type = 'Adam'  # 'Adam' or 'LBFGS'
optim_lr = 0.005
epochs = 1000
optim_maxiter = 100


# 1. 生成带噪声的3D高程数据 (PyTorch Tensor)
####################################################
X_train_rand = torch.rand(N, D) * 10  # NOTE:(100, 2) # random points in [0, 10]x[0, 10]
X_train = X_train_rand.clone()  # NOTE:(100, 2)
true_elevation = lambda x: 10 * torch.sin(5*x[:, 0]) + 10 * torch.cos(5 * x[:, 1]) + 10 * x[:, 0] + x[:, 1]
y_train = true_elevation(X_train) + torch.randn(N) * noise_data  # NOTE:(100,)

############################################################
min_val = 0
max_val = 10
res = 0.2
Ngrid = int((max_val - min_val) / res) + 1
points = [torch.linspace(min_val, max_val, Ngrid) for _ in range(D)]
grid = torch.meshgrid(*points, indexing='ij')  # 使用 ij 索引方式
X_train_grid = torch.stack([grid[i].flatten() for i in range(D)], dim=-1)

X_train = X_train_grid
print("X_train.shape: ", X_train.shape)
y_train = true_elevation(X_train) + torch.randn(X_train.size(0)) * noise_data  # NOTE:(100,)
print("y_train.shape: ", y_train.shape)

# 2. 定义PyTorch版本的RBF核函数
def rbf_kernel(X, Z, l):
    """
    X: NOTE:shape (n, d) 
    Z: NOTE:shape (m, d) 
    return: NOTE:shape (n, m)
    """
    pairwise_sq_dists = torch.cdist(X, Z, p=2).square()  # 欧氏距离平方
    return torch.exp(-pairwise_sq_dists / (2 * l**2))

def sparseKernel(X, Z, l):
    M2PI = 2 * torch.pi
    cdist = torch.cdist(X, Z)  # 欧氏距离平方
    cdist /= l
    
    kernel = ((2 + (cdist * M2PI).cos()) * (1 - cdist) / 3.0 + (cdist * M2PI).sin() / M2PI)
    kernel = kernel * (kernel > 0.0)
    # print(f'sparse kernel_shape: {kernel.shape}')
    return kernel
    
class ThreeLayerTanhNN(torch.nn.Sequential):
    def __init__(self, dim_input: int, dim_hidden: int, dim_output: int, softmax: bool = True,) -> None:
        super().__init__()
        self.add_module("linear1", torch.nn.Linear(dim_input, dim_hidden))
        self.add_module("activation1", torch.nn.Tanh())
        self.add_module("linear2", torch.nn.Linear(dim_hidden, dim_hidden))
        self.add_module("activation2", torch.nn.Tanh())
        self.add_module("linear3", torch.nn.Linear(dim_hidden, dim_output))
        if softmax:
            self.add_module("activation3", torch.nn.Softmax(dim=1))


def attentiveKernel(X, Z, klen_array):
    """
    X: NOTE:shape (n, d) 
    Z: NOTE:shape (m, d) 
    klen_array: NOTE:shape (klen_array,)
    return: NOTE:shape (n, m)
    """


# 3. 负对数边际似然函数 (PyTorch可微分版本)
def negative_log_mll(params, X, y, mu0, sigma):
    """
    params: (lambda, l) hyperparameters waiting to be optimized
    X: training data (N, D)
    y: training labels (N,)
    mu0: prior mean (scalar)
    sigma: prior variance (scalar)
    """
    lambda_, l = params
    # K = rbf_kernel(X, X, l)  # NOTE:(N, N)
    K = sparseKernel(X, X, l)
    N = X.shape[0]
    
    """
    torch.diag(K)
    K: [[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]]
    torch.diag(K): [1, 5, 9]
    """
    # basic variables
    s = K.sum(dim=1) - torch.diag(K)  # 排除对角元素
    sum_kjyj = K @ y - torch.diag(K) * y  # 排除j=i项
    
    # compute posterior mean
    mu = (lambda_ * mu0 + sum_kjyj) / (lambda_ + s) # torch.Size([2601])
    # print('sum_kjyj.shape: ', sum_kjyj.shape)

    # compute negative log marginal likelihood nll-loss
    term1 = -torch.log(lambda_ + s)
    term2 = ((y - mu)**2 * (lambda_ + s)) / (sigma**2)
    loss = 0.5 * torch.sum(term1 + term2)
    return loss

# 4. 超参数优化 (PyTorch LBFGS)
# 初始化可训练参数
lambda_ = torch.tensor([0.0001], requires_grad=False, dtype=torch.float32)
l = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
# mu0 = torch.mean(y_train).detach()
mu0 = y_train.detach()
print(f'mu0: {mu0}')
# print('y_train shape:', y_train.shape)

sigma = torch.tensor(1.0, dtype=torch.float32)

if optimizer_type == 'LBFGS':
    # 使用LBFGS优化器
    optimizer = torch.optim.LBFGS([lambda_, l], lr=optim_lr, max_iter=optim_maxiter)
elif optimizer_type == 'Adam':
    # 使用Adam优化器
    optimizer = torch.optim.Adam([lambda_, l], lr=optim_lr)

def closure():
    optimizer.zero_grad()
    loss = negative_log_mll((lambda_, l), X_train, y_train, mu0, sigma)
    loss.backward()
    print(f'Loss: {loss.item():.4f}, λ: {lambda_.item():.4f}, l: {l.item():.4f}')
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
optimal_l = l.item()
print(f"Optimal lambda: {optimal_lambda:.3f}, Optimal l: {optimal_l:.3f}")

# 5. 预测函数 (修正参数顺序)
def predict(X_train, y_train, X_test, lambda_, l, mu0, sigma):  # <-- 修正参数列表
    # K_train_test = rbf_kernel(X_test, X_train, l)  # (n_test, n_train)
    K_train_test = sparseKernel(X_test, X_train, l)
    sum_k = K_train_test.sum(dim=1)  # (n_test,)
    sum_kjyj = K_train_test @ y_train  # (n_test,)
    
    lambda_plus_sumk = lambda_ + sum_k
    mu = (lambda_ * mu0 + sum_kjyj) / lambda_plus_sumk
    var = sigma**2 / lambda_plus_sumk
    return mu.detach(), var.detach()

# 6. 生成网格测试点 (转换为Tensor)
slice_count = 51
x_grid = torch.linspace(0, 10, 51)
y_grid = torch.linspace(0, 10, 51)
X_test = torch.stack(torch.meshgrid(x_grid, y_grid, indexing='xy'), dim=-1).reshape(-1, 2)  # (900, 2)

# 进行预测 (修正调用参数)
mu_pred, var_pred = predict(X_train, y_train, X_test, optimal_lambda, optimal_l, mu0, sigma)  # <-- 添加y_train

# 计算MSE
y_test = true_elevation(X_test)
mse = torch.mean((mu_pred - y_test)**2).item()
print(f"Test MSE: {mse:.4f}")

# 7. 可视化 (转换为NumPy)
# 7. 三维可视化（完整三视图）
X_test_np = X_test.numpy()  # 转换为NumPy数组用于可视化
X_mesh = X_test_np[:, 0].reshape(51, 51)
Y_mesh = X_test_np[:, 1].reshape(51, 51)
Z_mesh_true = true_elevation(X_test).detach().numpy().reshape(51, 51)  # 真实高程
Z_mesh_pred = mu_pred.numpy().reshape(51, 51)  # 预测高程
Z_var = var_pred.numpy().reshape(51, 51)  # 预测方差

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
ax2.set_title(f'Predicted Elevation\nMSE={mse:.2f}, λ={optimal_lambda:.2f}, l={optimal_l:.2f}')
fig.colorbar(surf, ax=ax2)

# 不确定度曲面
ax3 = fig.add_subplot(133, projection='3d')
var_surf = ax3.plot_surface(X_mesh, Y_mesh, Z_var, 
                          cmap='coolwarm', alpha=0.8)
ax3.set_title('Prediction Variance (σ²)')
fig.colorbar(var_surf, ax=ax3)

plt.tight_layout()
plt.show()