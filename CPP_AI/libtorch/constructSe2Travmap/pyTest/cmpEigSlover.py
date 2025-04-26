import torch
import time

merge_size = 1

# 生成 972000 个 3x3 的随机对称矩阵
num_matrices = 972000//merge_size
matrix_size = 3*merge_size
matrices = torch.randn(num_matrices, matrix_size, matrix_size)
# 保证矩阵是对称的
matrices = 0.5 * (matrices + matrices.transpose(1, 2))

# CPU 特征分解
start_time_cpu = time.time()
eigenvalues_cpu, eigenvectors_cpu = torch.linalg.eigh(matrices)
duration_cpu = time.time() - start_time_cpu
print(f"CPU 特征分解运行时间: {duration_cpu:.6f} 秒")

# GPU 特征分解
if torch.cuda.is_available():
    matrices_gpu = matrices.to('cuda')
    start_time_gpu = time.time()
    eigenvalues_gpu, eigenvectors_gpu = torch.linalg.eigh(matrices_gpu)
    duration_gpu = time.time() - start_time_gpu
    print(f"GPU 特征分解运行时间: {duration_gpu:.6f} 秒")
else:
    print("GPU 不可用，无法进行 GPU 特征分解")