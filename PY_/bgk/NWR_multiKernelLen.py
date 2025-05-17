import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成数据
def true_elevation(X):
    x = X[:, 0]
    y = X[:, 1]
    return torch.sin(x) + torch.cos(y) + 0.05 * (x**2 + y**2)

# 训练数据 (0.2分辨率)
x_train = torch.arange(0, 10.01, 0.2)
y_train = torch.arange(0, 10.01, 0.2)
X_train = torch.stack(torch.meshgrid(x_train, y_train, indexing='xy'), dim=-1).reshape(-1, 2)
y_train = true_elevation(X_train) + torch.randn(X_train.shape[0]) * 0.1

# 测试数据 (0.1分辨率)
x_test = torch.arange(0, 10.01, 0.1)
y_test = torch.arange(0, 10.01, 0.1)
X_test = torch.stack(torch.meshgrid(x_test, y_test, indexing='xy'), dim=-1).reshape(-1, 2)

# 2. 定义核函数
def sparseKernel(X, Z, l):
    M2PI = 2 * torch.pi
    cdist = torch.cdist(X, Z)
    l = l.unsqueeze(1)  # 添加维度用于广播
    cdist = cdist / l
    kernel = ((2 + (cdist * M2PI).cos()) * (1 - cdist)/3 + 
             (cdist * M2PI).sin()/(M2PI))
    return kernel * (kernel > 0)

# 3. 定义可学习的长度尺度模型
class LengthModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(2, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
            torch.nn.Softplus()
        )
        self.sigma = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, X):
        return self.fc(X).squeeze()

model = LengthModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 4. 训练过程
n_epochs = 100
for epoch in range(n_epochs):
    optimizer.zero_grad()
    
    # 计算训练数据的长度尺度
    l_train = model(X_train)
    
    # 计算核矩阵
    K = sparseKernel(X_train, X_train, l_train)
    
    # Nadaraya-Watson回归预测
    weights = K / K.sum(dim=1, keepdim=True)
    mu_pred = weights @ y_train.float()
    
    # 计算边缘对数似然
    sigma = model.sigma
    loss = 0.5 * (mu_pred - y_train).pow(2).sum() / sigma**2 + \
           y_train.size(0) * torch.log(sigma) + \
           0.5 * y_train.size(0) * np.log(2*np.pi)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}, Sigma: {sigma.item():.4f}')

# 5. 预测
with torch.no_grad():
    l_test = model(X_test)
    K_test = sparseKernel(X_test, X_train, l_test)
    weights_test = K_test / K_test.sum(dim=1, keepdim=True)
    mu_pred = weights_test @ y_train.float()
    var_pred = (model.sigma**2) / K_test.sum(dim=1)

# 6. 可视化
X_plot = X_test.numpy().reshape(101, 101, 2)
Z_true = true_elevation(X_test).reshape(101, 101).numpy()
Z_pred = mu_pred.reshape(101, 101).numpy()
Z_var = var_pred.reshape(101, 101).numpy()

fig = plt.figure(figsize=(18, 6))

# 真实曲面
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X_plot[:,:,0], X_plot[:,:,1], Z_true, cmap='viridis')
ax1.set_title('True Elevation')

# 预测曲面
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X_plot[:,:,0], X_plot[:,:,1], Z_pred, cmap='plasma')
ax2.set_title(f'Predicted Elevation (MSE: {np.mean((Z_true-Z_pred)**2):.4f})')

# 方差曲面
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(X_plot[:,:,0], X_plot[:,:,1], Z_var, cmap='coolwarm')
ax3.set_title('Prediction Variance')

plt.tight_layout()
plt.show()