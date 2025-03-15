'''
Author: chasey melancholycy@gmail.com
Date: 2024-12-27 09:34:32
FilePath: /mesh_planner/test/test.py
Description: 

Copyright (c) 2024 by chasey (melancholycy@gmail.com), All Rights Reserved. 
'''


feature_data = [0,1,2,3,4,5,6,7,8,9]

# 正确的列表推导式，使用range函数生成索引
train_x = [feature_data[2*i] for i in range(3)]

print(train_x)

xnum = 17
ynum = 1

numbers = [i for i in range(241)] 

