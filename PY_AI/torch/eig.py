'''
Author: chasey && melancholycy@gmail.com
Date: 2025-03-24 01:54:25
LastEditTime: 2025-03-24 01:54:27
FilePath: /test/PY_AI/torch/eig.py
Description: 
Reference: 
Copyright (c) 2025 by chasey && melancholycy@gmail.com, All Rights Reserved. 
'''
import torch

A = torch.tensor([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]], dtype=torch.float32)

eigenvalues, eigenvectors = torch.linalg.eig(A)

print("Eigenvalues:\n", eigenvalues)
print("Eigenvectors:\n", eigenvectors)