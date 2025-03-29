'''
Author: chasey && melancholycy@gmail.com
Date: 2025-03-28 02:03:03
LastEditTime: 2025-03-28 14:01:10
FilePath: /test/PY_/bgk/NWR_MSE_Optim.py
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

# class NWSparseKernelRegression(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.kernelLen_ = nn.Parameter(torch.rand((1,), requires_grad=True))


######################### parameters #########################
n_train = 100  # 训练样本数
range = [0, 10]
noise = 0.1  
kernel_len = 0.2

epoch_num = 5
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
# X_repeat的形状:(n_test,n_train),
# 每一行都包含着相同的测试输入（例如：同样的查询）
X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
# x_train包含着键。attention_weights的形状：(n_test,n_train),
# 每一行都包含着要在给定的每个查询的值（y_train）之间分配的注意力权重
attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2 / kernel_len, dim=1)
# y_hat的每个元素都是值的加权平均值，其中的权重是注意力权重
y_hat_unopt = torch.matmul(attention_weights, y_train)

######################### NWG optimized regression #########################
# X_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输入
X_tile = x_train.repeat((n_train, 1)) # n_train: [100] X_tile: [100, 100] 
# Y_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输出
Y_tile = y_train.repeat((n_train, 1))

# keys的形状:('n_train'，'n_train'-1)
keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
# values的形状:('n_train'，'n_train'-1)
values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))

print(f'x_train.shape:{x_train.shape} y_train.shape:{y_train.shape}')
# print(f'x_train: {x_train}')
# print(f'X_tile[0]: {X_tile[0]}')

print(f'X_tile.shape:{X_tile.shape} Y_tile.shape:{Y_tile.shape}')
# print(f'y_train: {y_train}')
# print(f'Y_tile[0]: {Y_tile[0]}')

print(f'keys.shape:{keys.shape} values.shape:{values.shape}')



# print(f'x_train: {x_train}')
# print(f'X_tile: {X_tile}')
# print(f'keys: {keys}')
# print(f'values: {values}')


net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=opt_lr)
loss_list = []

epoch = 0
print(f'initial weight: {float(net.getKernelLen()):.6f}')
while epoch < epoch_num:
    trainer.zero_grad()
    l = loss(net(x_train, keys, values), y_train)
    l.sum().backward()
    trainer.step()
    print(f'epoch: {epoch + 1}, loss: {float(l.sum()):.6f}, w: {float(net.getKernelLen()):.6f}')
    loss_list.append([epoch+1, float(l.sum())])
    epoch += 1

# keys的形状:(n_test，n_train)，每一行包含着相同的训练输入（例如，相同的键）
keys = x_train.repeat((n_test, 1))
# value的形状:(n_test，n_train)
values = y_train.repeat((n_test, 1))
y_hat_optim = net(x_test, keys, values).unsqueeze(1).detach()

######################### plot data #########################
# 绘制图像
plt.figure(figsize=(10, 6))

# 绘制真实函数
plt.plot(x_test.numpy(), y_truth.numpy(), label='True Function', color='blue', linewidth=5)
plt.plot(x_test.numpy(), y_hat_unopt.numpy(), label='Pred Unopt Function', color='green', linewidth=1)
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