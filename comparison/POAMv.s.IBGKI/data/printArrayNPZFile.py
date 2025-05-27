'''
Author: chasey && melancholycy@gmail.com
Date: 2025-05-18 14:07:25
LastEditTime: 2025-05-18 14:09:31
FilePath: /POAM/data/printArrayNPZFile.py
Description: 
Reference: 
Copyright (c) 2025 by chasey && melancholycy@gmail.com, All Rights Reserved. 
'''
import numpy as np

def load_and_print_npz(file_name):
    # 加载 .npz 文件
    data = np.load(file_name)
    
    # 假设数组存储在文件的默认键下（通常是 'arr_0'）
    array = data['arr_0']
    
    # 打印数组的形状
    print("数组的形状:", array.shape)
    
    # 打印数组的前十个元素
    print(f"数组的前十个元素shape:{array[:10].shape}:\n{array[:10]}")

if __name__ == "__main__":
    file_name = "./arrays/n44w111.npz"  # 确保文件名正确
    load_and_print_npz(file_name)