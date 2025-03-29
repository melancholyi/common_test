'''
Author: chasey && melancholycy@gmail.com
Date: 2025-03-28 02:03:03
LastEditTime: 2025-03-28 15:17:43
FilePath: /test/PY_/bgk/NWR_SparseKernel_MSE_Optim.py
Description: 
Reference: https://zh.d2l.ai/chapter_attention-mechanisms/nadaraya-waston.html
Copyright (c) 2025 by chasey && melancholycy@gmail.com, All Rights Reserved. 
'''

import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn



# Nadaraya-Watson Kernel Regression  (NWR) 
# Kernel: gaussian_rbf  
class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.w = nn.Parameter(torch.rand((1,), requires_grad=True))
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, queries, keys, values):
        # queries和attention_weights的形状为(查询个数，“键－值”对个数)
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        self.attention_weights = nn.functional.softmax(
            -((queries - keys) * self.w)**2 / 2, dim=1)
        # values的形状为(查询个数，“键－值”对个数)
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1)
    def getKernelLen(self):
        return 1.0/self.w

class NWSparseKernelRegression(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernelLen_ = nn.Parameter(torch.rand((1,), requires_grad=True))
        self.M2PI = 2.0 * torch.pi

    def getKernelLen(self):
        return 1.0/self.kernelLen_
    
    def forward(self, queries, keys, values):
        cdist = torch.cdist(queries, keys)
        cdist/=self.kernelLen_
        kernel = ((2.0 + (cdist * self.M2PI).cos()) * (1.0 - cdist) / 3.0 +
                  (cdist * self.M2PI).sin() / self.M2PI)

        # # kernel's elem is masked with 0.0 if dist > kernelLen_
        kernel = kernel * (kernel > 0.0)

        # print(f'kernel:\n{kernel}')

        ones = torch.ones_like(values)
        ybar = (kernel @ values) / (kernel @ ones) + 1e-6
        # print(f'queried_values:\n{ybar}')
        return ybar

####################### test code ##########################
sparsenet = NWSparseKernelRegression()
# 定义预测数据和训练数据
predXvec = [1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12]

trainXvec = [1, 2, 3, 4,
             5, 6, 7, 8,
             9, 10, 11, 12,
             13, 14, 15, 16,
             17, 18, 19, 20]

trainYvec = [10, 
             20,
             30, 
             40, 
             50]

# 输入维度
INPUT_DIM = 4
OUTPUT_DIM = 1

# 将数据转换为 PyTorch 张量
predX = torch.tensor(predXvec, dtype=torch.float64).view(len(predXvec) // INPUT_DIM, INPUT_DIM)
trainX = torch.tensor(trainXvec, dtype=torch.float64).view(len(trainXvec) // INPUT_DIM, INPUT_DIM)
trainY = torch.tensor(trainYvec, dtype=torch.float64).view(len(trainYvec) // OUTPUT_DIM, OUTPUT_DIM)

# 将张量移动到 GPU（如果可用）
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# predX = predX.to(device)
# trainX = trainX.to(device)
# trainY = trainY.to(device)

print(" predX shape:", predX.shape)
print("trainX shape:", trainX.shape)

sparsenet(predX, trainX, trainY)


######################### parameters #########################
n_train = 100  # 训练样本数
range = [0, 10]
noise = 0.01  
kernel_len = 0.2

epoch_num = 100
opt_lr = 0.01


######################### generate data #########################
def f(x):
    return 2 * torch.sin(x) + x**0.8

# train data
x_train, _ = torch.sort(torch.rand(n_train) * range[1])   # 排序后的训练样本
y_train = f(x_train) + torch.normal(0.0, noise, (n_train,))  # 训练样本的输出

# test data
x_test = torch.arange(range[0], range[1], 0.1)  # 测试样本
y_truth = f(x_test)  # 测试样本的真实输出
n_test = len(x_test)  # 测试样本数

######################### NWG basic regression #########################
net_basic = NWSparseKernelRegression()
x_train_2d = x_train.unsqueeze(1)
x_test_2d = x_test.unsqueeze(1)
y_train_2d = y_train.unsqueeze(1)
y_hat_unopt = net_basic(x_test_2d, x_train_2d, y_train_2d)

######################### NWG optimized regression ########################
net_optim = NWSparseKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net_optim.parameters(), lr=opt_lr)
# loss_list = []

epoch = 0
print(f'initial weight: {float(net_optim.getKernelLen()):.6f}')
while epoch < epoch_num:
    trainer.zero_grad()
    l = loss(net_optim(x_test_2d, x_train_2d, y_train_2d), y_train_2d)
    l.sum().backward()
    trainer.step()
    print(f'epoch: {epoch + 1}, loss: {float(l.sum()):.6f}, w: {float(net_optim.getKernelLen()):.6f}')
    epoch += 1

y_hat_optim = net_optim(x_test_2d, x_train_2d, y_train_2d).unsqueeze(1).detach()
y_hat_optim = y_hat_optim.squeeze()

######################### plot data #########################
# 绘制图像
plt.figure(figsize=(10, 6))

# 绘制真实函数
plt.plot(x_test.numpy(), y_truth.numpy(), label='True Function', color='blue', linewidth=5)
plt.plot(x_test.numpy(), y_hat_unopt.detach().numpy(), label='Pred Unopt Function', color='green', linewidth=1)
plt.plot(x_test.numpy(), y_hat_optim.numpy(), label='Pred Optim Function', color='purple', linewidth=5)

# 绘制训练数据
plt.scatter(x_train.numpy(), y_train.numpy(), label='Training Data', color='red', alpha=0.6)

# 添加图例和标签
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Training Data and True Function')
plt.grid(True)
plt.show()