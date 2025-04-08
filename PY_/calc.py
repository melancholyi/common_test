'''
Author: chasey melancholycy@gmail.com
Date: 2025-01-28 15:22:29
FilePath: /test/PY_/calc.py
Description: 

Copyright (c) 2025 by chasey (melancholycy@gmail.com), All Rights Reserved. 
'''
import numpy as np 

# Given mean_z values
mean_z_values = np.array([3.41324, 3.75344, 4.25326])
mean_count = np.array([13593, 38814, 41926])

# Calculate the overall mean
overall_mean = sum(mean_z_values * mean_count)/sum(mean_count)

print(f"Overall Mean: {overall_mean:.5f}")

def getC(a, b):
    return np.sqrt(1- a**2 - b**2)

print(getC(0.18,0.15))
print(getC(0.20,0.15))
print(getC(0.21,0.16))
print(getC(0.19,0.14))


print(np.log(1e30))
