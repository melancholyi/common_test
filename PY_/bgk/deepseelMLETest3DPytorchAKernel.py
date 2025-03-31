import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. 生成带噪声的3D高程数据 (PyTorch Tensor)
torch.manual_seed(42)
delta = 1e-5
N = 200
D = 2
noise_data = 0.0001
optimizer_type = 'Adam'  # or 'LBFGS'
optim_lr = 0.001
optim_maxiter = 100
optim_epochs = 10 if optimizer_type == 'LBFGS' else 100
base_kernel = 'sparse'  # or 'rbf'
klen_array = torch.arange(0.1, 1.0 + delta, 0.2) if base_kernel == 'rbf' else torch.arange(0.1, 2.0 + delta, 0.2)
input_dim = 2 
hidden_dim = 8 
output_dim = len(klen_array)
print(f'klen_array: {klen_array}')

X_train = torch.rand(N, D) * 10  # NOTE:(100, 2)
true_elevation = lambda x: 2 * torch.sin(x[:, 0]) + 3 * torch.cos(x[:, 1]) + x[:, 0] + x[:, 1]
y_train = true_elevation(X_train) + torch.randn(N) * noise_data  # NOTE:(100,)

# 2. 定义PyTorch版本的RBF核函数
def sparseKernel(X, Z, l):
    M2PI = 2 * torch.pi
    cdist = torch.cdist(X, Z)  # 欧氏距离平方
    cdist /= l
    
    kernel = ((2 + (cdist * M2PI).cos()) * (1 - cdist) / 3.0 + (cdist * M2PI).sin() / M2PI)
    kernel = kernel * (kernel > 0.0)
    # print(f'kernel_shape: {kernel.shape}')
    return kernel

def rbf_kernel(X, Z, l):
    """
    X: NOTE:shape (n, d) 
    Z: NOTE:shape (m, d) 
    return: NOTE:shape (n, m)
    """
    pairwise_sq_dists = torch.cdist(X, Z, p=2).square()  # 欧氏距离平方

    return torch.exp(-pairwise_sq_dists / (2 * l**2))
    
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

class AttentiveKernelNN(torch.nn.Module):
    def __init__(self, dim_input: int, dim_hidden: int, dim_output: int, softmax: bool = True) -> None:
        super().__init__()
        self.nn_weight = ThreeLayerTanhNN(dim_input, dim_hidden, dim_output, softmax)
        self.nn_instanse = ThreeLayerTanhNN(dim_input, dim_hidden, dim_output, softmax)

train_instance_selection = True 
weightNN = ThreeLayerTanhNN(input_dim, hidden_dim, output_dim)
print(f'weightNN model struct:{weightNN}')
instanceNN = ThreeLayerTanhNN(input_dim, hidden_dim, output_dim)
# weight_raw = weightNN(X_train)
# weight_norm = weight_raw / weight_raw.norm(dim=1, keepdim=True)  # 归一化权重
# for param in weightNN.named_parameters():
#     print(f'weightNN param: {param}')


def getNNWeight(X, detach_w=False):
    raw_w = weightNN(X)
    _w = raw_w / raw_w.norm(dim=1, keepdim=True)
    w = _w.detach() if detach_w else _w
    return w

def get_representations(X, detach_w=False, detach_z=False):
    raw_w = weightNN(X)
    _w = raw_w / raw_w.norm(dim=1, keepdim=True)
    w = _w.detach() if detach_w else _w

    raw_z = instanceNN(X)
    _z = raw_z / raw_z.norm(dim=1, keepdim=True)
    z = _z.detach() if detach_z else _z

    return w, z

def attentiveKernel(X, Z, klen_array):
    """
    X: NOTE:shape (n, d) 
    Z: NOTE:shape (m, d) 
    klen_array: NOTE:shape (klen_array,)
    return: NOTE:shape (n, m)
    """
    # wx = getNNWeight(X, detach_w=True)
    # wz = getNNWeight(Z, detach_w=True)
    if train_instance_selection:
        # print(f'=====Instanse Selection=====')
        w1, z1 = get_representations(X, detach_w=True)
        w2, z2 = get_representations(Z, detach_w=True)
    else :
        # print(f'=====KernelLen Selection=====')
        w1, z1 = get_representations(X, detach_z=True)
        w2, z2 = get_representations(Z, detach_z=True)
    mask = z1 @ z2.t()
    cov_mat = 0.0
    sim_list = []
    for i in range(len(klen_array)):
        similarity = torch.outer(w1[:, i], w2[:, i])  # (n, m)
        sim_list.append(similarity)
        klen = klen_array[i]
        if base_kernel == 'rbf':
            cov_mat += rbf_kernel(X, Z, klen) * similarity
        else:
            cov_mat += sparseKernel(X, Z, klen) * similarity
    # print(f'sim_list: {sim_list}')
    cov_mat *= mask
    return cov_mat


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
    K_AK = attentiveKernel(X, X, klen_array)  # NOTE:(N, N)
    
    K = K_AK

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
    mu = (lambda_ * mu0 + sum_kjyj) / (lambda_ + s)

    # compute negative log marginal likelihood nll-loss
    term1 = -torch.log(lambda_ + s)
    term2 = ((y - mu)**2 * (lambda_ + s)) / (sigma**2)
    loss = 0.5 * torch.sum(term1 + term2)
    return loss

# 4. 超参数优化 (PyTorch LBFGS)
# 初始化可训练参数
lambda_ = torch.tensor([0.0], requires_grad=True, dtype=torch.float32)
l = torch.tensor([10.0], requires_grad=True, dtype=torch.float32)
mu0 = torch.mean(y_train).detach()
sigma = torch.tensor(1.0, dtype=torch.float32)

params_list = list(weightNN.parameters()) + list(instanceNN.parameters()) # + [lambda_, l]
if optimizer_type == 'LBFGS':
    optimizer = torch.optim.LBFGS(params_list, lr=optim_lr, max_iter=optim_maxiter)
elif optimizer_type == 'Adam':
    optimizer = torch.optim.Adam(params_list, lr=optim_lr)

print(f'optimizer: {optimizer}')

def closure():
    optimizer.zero_grad()
    loss = negative_log_mll((lambda_, l), X_train, y_train, mu0, sigma)
    loss.backward()
    return loss

# 运行优化
for _ in range(optim_epochs):  # LBFGS可能需要多次调用closure
    optimizer.step(closure)

train_instance_selection = False
for _ in range(optim_epochs):  # LBFGS可能需要多次调用closure
    optimizer.step(closure)




optimal_lambda = lambda_.item()
optimal_l = l.item()
print(f"Optimal lambda: {optimal_lambda:.3f}, Optimal l: {optimal_l:.3f}")

# 5. 预测函数 (修正参数顺序)
def predict(X_train, y_train, X_test, lambda_, l, mu0, sigma):  # <-- 修正参数列表
    K_train_test = attentiveKernel(X_test, X_train, klen_array)  # (n_test, n_train)
    sum_k = K_train_test.sum(dim=1)  # (n_test,)
    sum_kjyj = K_train_test @ y_train  # (n_test,)
    
    lambda_plus_sumk = lambda_ + sum_k
    mu = (lambda_ * mu0 + sum_kjyj) / lambda_plus_sumk
    var = sigma**2 / lambda_plus_sumk
    return mu.detach(), var.detach()

# 6. 生成网格测试点 (转换为Tensor)
x_grid = torch.linspace(0, 10, 30)
y_grid = torch.linspace(0, 10, 30)
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
X_mesh = X_test_np[:, 0].reshape(30, 30)
Y_mesh = X_test_np[:, 1].reshape(30, 30)
Z_mesh_true = true_elevation(X_test).detach().numpy().reshape(30, 30)  # 真实高程
Z_mesh_pred = mu_pred.numpy().reshape(30, 30)  # 预测高程
Z_var = var_pred.numpy().reshape(30, 30)  # 预测方差

fig = plt.figure(figsize=(18, 6))

# 真实高程曲面
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X_mesh, Y_mesh, Z_mesh_true, 
                cmap='viridis', alpha=0.8)
ax1.scatter(X_train[:,0].numpy(), X_train[:,1].numpy(), y_train.numpy(), 
           c='r', s=20, label='Noisy Data')
ax1.set_title('True Elevation Map')

# 预测曲面
ax2 = fig.add_subplot(132, projection='3d')
surf = ax2.plot_surface(X_mesh, Y_mesh, Z_mesh_pred, cmap='plasma', alpha=0.8)
ax2.scatter(X_train[:,0].numpy(), X_train[:,1].numpy(), y_train.numpy(), 
           c='r', s=20)
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