'''
Author: chasey && melancholycy@gmail.com
Date: 2025-05-17 14:51:30
LastEditTime: 2025-05-17 14:51:32
FilePath: /test/PY_/bgk/independentKlenBGKIRegression/test.py
Description: 
Reference: 
Copyright (c) 2025 by chasey && melancholycy@gmail.com, All Rights Reserved. 
'''
import torch

# Example tensors (replace with actual data)
cdist = torch.randn(2601, 2601)  # Shape [2601, 2601]
klen = torch.randn(1)         # Shape [2601]

# Compute cdist / klen
result = cdist / klen.unsqueeze(1)  # Shape [2601, 2601]
print("Result shape:", result.shape)  # Should be [2601, 2601]