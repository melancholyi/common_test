'''
Author: chasey melancholycy@gmail.com
Date: 2025-03-13 14:26:57
FilePath: /mesh_planner/test/temp/quat2yaw.py
Description: 

Copyright (c) 2025 by chasey (melancholycy@gmail.com), All Rights Reserved. 
'''
import math

# Given quaternion values
qx = -0.0590493
qy = 0.0736535
qz = -0.0391971
qw = -0.994762

# Calculate yaw angle
yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))

# Convert yaw angle from radians to degrees (optional)
yaw_degrees = math.degrees(yaw)

print(f"Yaw angle (radians): {yaw}")
print(f"Yaw angle (degrees): {yaw_degrees}")