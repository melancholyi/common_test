'''
Author: chasey && melancholycy@gmail.com
Date: 2025-04-20 04:47:23
LastEditTime: 2025-04-20 04:47:25
FilePath: /test/CPP_AI/libtorch/constructSe2Travmap/pyTest/cmpBlockEigSlover.py
Description: 
Reference: 
Copyright (c) 2025 by chasey && melancholycy@gmail.com, All Rights Reserved. 
'''
import torch
import time

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 参数设置（测试时减小规模避免OOM）
    num_matrices = 1000  # 测试用样本量（实际应用改为972000）
    matrix_size = 3
    
    # 生成对称测试数据（直接GPU生成避免传输开销）
    matrices = torch.randn(num_matrices, matrix_size, matrix_size, device=device)
    matrices = (matrices + matrices.mT) / 2  # 强制对称

    # ----------------- 方法1：直接批量计算 -----------------
    torch.cuda.synchronize()
    start = time.time()
    eigenvalues_batch, eigenvectors_batch = torch.linalg.eigh(matrices)
    torch.cuda.synchronize()
    time_batch = (time.time() - start) * 1000  # 毫秒
    
    # ----------------- 方法2：块对角矩阵拼接 -----------------
    # 注意：实际972000矩阵会导致显存爆炸，此处仅演示可行性
    torch.cuda.synchronize()
    start_construct = time.time()
    
    # 分块构建（避免一次性内存分配）
    chunk_size = 100  # 每个块包含100个小矩阵
    chunks = []
    for i in range(0, num_matrices, chunk_size):
        chunk = torch.block_diag(*matrices[i:i+chunk_size].cpu())  # CPU上拼接
        chunks.append(chunk.to(device))
    big_matrix = torch.block_diag(*chunks)  # 最终拼接
    
    torch.cuda.synchronize()
    time_construct = (time.time() - start_construct) * 1000
    
    # 特征分解
    start_eigh = time.time()
    eigenvalues_diag, eigenvectors_diag = torch.linalg.eigh(big_matrix)
    torch.cuda.synchronize()
    time_eigh = (time.time() - start_eigh) * 1000
    
    # ----------------- 结果分析 -----------------
    print("\n性能对比:")
    print(f"方法1（批量计算）耗时: {time_batch:.2f} ms")
    print(f"方法2（块对角）总耗时: {time_construct + time_eigh:.2f} ms")
    print(f"  ├─ 矩阵构造耗时: {time_construct:.2f} ms")
    print(f"  └─ 特征分解耗时: {time_eigh:.2f} ms")

    # 显存占用估算
    mem_batch = matrices.element_size() * matrices.nelement() / 1024**2  # MB
    mem_diag = big_matrix.element_size() * big_matrix.nelement() / 1024**3  # GB
    print(f"\n显存占用:")
    print(f"方法1: {mem_batch:.2f} MB")
    print(f"方法2: {mem_diag:.2f} GB")

    # 正确性验证（随机抽查一个矩阵）
    if num_matrices >= 1:
        idx = 0
        reconstructed = (eigenvectors_batch[idx] @ torch.diag(eigenvalues_batch[idx]) @ eigenvectors_batch[idx].T)
        error_batch = torch.norm(reconstructed - matrices[idx]).item()
        
        # 从块对角结果中提取
        start_idx = idx * matrix_size
        block = eigenvectors_diag[start_idx:start_idx+matrix_size, start_idx:start_idx+matrix_size]
        reconstructed_diag = block @ torch.diag(eigenvalues_diag[start_idx:start_idx+matrix_size]) @ block.T
        error_diag = torch.norm(reconstructed_diag - matrices[idx].cpu()).item()
        
        print(f"\n验证误差:")
        print(f"方法1误差: {error_batch:.2e}")
        print(f"方法2误差: {error_diag:.2e}")

if __name__ == "__main__":
    main()