'''
Author: chasey && melancholycy@gmail.com
Date: 2025-05-27 06:52:42
LastEditTime: 2025-05-27 08:41:09
FilePath: /test/PY_/bgk/independentKlenBGKIRegression/SubTest/kernel_div2whilebeforebreak.py
Description: 
Reference: 
Copyright (c) 2025 by chasey && melancholycy@gmail.com, All Rights Reserved. 
'''
import torch
import numpy as np

# 创建一个随机的 tensor1，大小为 [N, M]
N = 100000
M = 3
tensor1 = torch.randn(N, M).abs().mul(4)

# 创建一个随机的 klen tensor，大小为 [N]
klen = torch.randint(1, 4, (N,))

# 利用 unsqueeze 将 klen 的维度扩展，使其可以与 tensor1 广播
tensor2 = tensor1 / klen.unsqueeze(-1)

# 定义一个函数来执行条件操作
def process_tensor(input_tensor, klen_tensor):
    # 将 input_tensor 复制一份，避免修改原始张量
    processed_tensor = input_tensor.clone()
    # 使用 while 循环，检查每行是否满足条件
    # 为了演示效果，这里只执行一次，实际使用中可以根据需要调整循环条件
    while True:
        all_greater_than_1 = torch.all(processed_tensor > 1, dim=1)
        # 检查是否存在满足条件的行
        if not torch.any(all_greater_than_1):
            break
        # 将满足条件的行元素除以 2
        processed_tensor[all_greater_than_1] /= 2
        # print(f'whiling processed_tensor:\n{processed_tensor}')
    return processed_tensor


def process_tensor2(x, eps=1e-6, threshold=0.6):
    N, M = x.shape
    required_count = np.ceil(threshold * M)  # 计算所需阈值
    # 统计每行大于1的元素数量
    count_over_1 = (x > 1).sum(dim=1)
    mask = count_over_1 >= required_count  # 标记需处理的行
    if not mask.any():
        return x  # 若无行需处理，直接返回
    
    # # 确定需要处理的行（所有元素>1）
    # mask = min_vals > 1
    
    

    # 计算每行的最小值
    min_vals = x.min(dim=1).values


    
    # 计算log2并取整得到最大次数
    log_vals = torch.log2(torch.clamp(min_vals - eps, min=eps))  # 避免对数非正数
    k_max = torch.floor(log_vals)
    # 确定最终除数次数（k_max + 1）
    k_total = torch.where(mask, k_max + 1, torch.zeros_like(k_max))
    # 计算除数并扩展维度以便广播
    divisors = 2 ** k_total
    divisors_expanded = divisors.view(-1, 1)
    # 执行除法
    result = x / divisors_expanded
    return result


# 对 tensor1 进行处理


# 打印结果
# print("原始 tensor1:")
# print(tensor1)
# print("\nklen tensor:")
# print(klen)

print(f'=========================== Processing tensor function one(while loop /2) ===========================')
processed_tensor1 = process_tensor(tensor1, klen)

print(f'=========================== Processing tensor function two (min /2) ===========================')
processed_tensor2 = process_tensor2(tensor1)



# print("\n-----处理后的 tensor1:")
# print(processed_tensor1)

# print("\n-----处理后的 tensor2:")
# print(processed_tensor2)

diff = torch.abs(processed_tensor1 - processed_tensor2).sum()
print("\n-----处理后的 tensor1 和 tensor2 的差异:")
print(diff) 