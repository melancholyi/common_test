import torch
import gpytorch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 定义真实高程函数
def true_elevation(X):
    x = X[:, 0]
    y = X[:, 1]
    return 10*torch.sin(5*x) + 10*torch.cos(5*y) + 10*x+y

# 设置噪声数据
noise_data = 0.01

# 设置参数
min_val = 0
max_val = 10
res = 0.2
D = 2  # 2D网格

Ngrid = int((max_val - min_val) / res) + 1
points = [torch.linspace(min_val, max_val, Ngrid) for _ in range(D)]
grid = torch.meshgrid(*points, indexing='ij')  # 使用 ij 索引方式
X_train_grid = torch.stack([grid[i].flatten() for i in range(D)], dim=-1)

X_train = X_train_grid
print("X_train.shape: ", X_train.shape)
y_train = true_elevation(X_train) + torch.randn(X_train.size(0)) * noise_data  

# 选择诱导点
inducing_points = X_train[::100, :]  # 每隔100个点选一个作为诱导点

# 初始化模型和似然函数
class GPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

model = GPModel(inducing_points=inducing_points)
likelihood = gpytorch.likelihoods.GaussianLikelihood()

# 设置训练模式
model.train()
likelihood.train()

# 训练模型
optimizer = torch.optim.Adam([
    {'params': model.parameters()},  # 包含模型的参数
], lr=0.1)

# "Loss" 是模型的负边缘对数似然
mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=y_train.size(0))

training_iter = 500
for i in range(training_iter):
    optimizer.zero_grad()
    output = model(X_train)
    loss = -mll(output, y_train)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
    optimizer.step()

# 使用模型进行预测
model.eval()
likelihood.eval()

# 创建预测网格
X_pred = X_train_grid

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(model(X_pred))
    y_pred = observed_pred.mean
    var_pred = observed_pred.variance

# 计算MSE
true_values = true_elevation(X_pred)
mse = ((y_pred - true_values) ** 2).mean()
print(f"Mean Squared Error (MSE): {mse.item()}")


# 绘图
fig = plt.figure(figsize=(18, 5))

# 函数真值
ax = fig.add_subplot(131, projection='3d')
ax.plot_surface(grid[0].numpy(), grid[1].numpy(), true_elevation(X_pred).reshape(grid[0].shape).numpy(), cmap='viridis', alpha=0.7)
ax.scatter(X_train[:,0], X_train[:,1], y_train, c='red', marker='o', s=1)
ax.set_title('True Elevation')

# SGP预测结果
ax = fig.add_subplot(132, projection='3d')
ax.plot_surface(grid[0].numpy(), grid[1].numpy(), y_pred.numpy().reshape(grid[0].shape), cmap='viridis', alpha=0.7)
ax.set_title(f'SGP Predicted Elevation\nMSE={mse:.2f}')

# SGP预测方差
ax = fig.add_subplot(133, projection='3d')
ax.plot_surface(grid[0].numpy(), grid[1].numpy(), var_pred.numpy().reshape(grid[0].shape), cmap='viridis', alpha=0.7)
ax.set_title('SGP Prediction Variance')

plt.tight_layout()
plt.show()