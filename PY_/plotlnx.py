'''
Author: chasey melancholycy@gmail.com
Date: 2025-03-01 10:27:34
FilePath: /mesh_planner/test/python/plotlnx.py
Description: 

Copyright (c) 2025 by chasey (melancholycy@gmail.com), All Rights Reserved. 
'''
import numpy as np
import matplotlib.pyplot as plt

# Define the range of x
x = np.linspace(0.01, 1000000, 1000)

# Calculate the natural logarithm of x
y = np.log(x)

# Plot the function
plt.plot(x, y)
plt.title('Plot of ln(x) from 0.01 to 1000000')
plt.xlabel('x')
plt.ylabel('ln(x)')
plt.grid(True)
plt.show()