'''
Author: chasey && melancholycy@gmail.com
Date: 2025-03-25 06:01:22
LastEditTime: 2025-03-25 06:01:40
FilePath: /test/PY_/bgk/plotRBFandCovSparse.py
Description: 
Reference: 
Copyright (c) 2025 by chasey && melancholycy@gmail.com, All Rights Reserved. 
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the kernel function from previous conversation
def kernel_function(d_i, l=1.0):
    k = np.zeros_like(d_i)
    valid_indices = d_i <= l
    
    cos_term = np.cos(2 * np.pi * d_i[valid_indices] / l)
    sin_term = np.sin(2 * np.pi * d_i[valid_indices] / l)
    
    k[valid_indices] = ((2 + cos_term) / 3) * (1 - d_i[valid_indices] / l) + sin_term / (2 * np.pi)
    
    return k

# Define the Gaussian RBF
def gaussian_rbf_2d(x, y, xc, yc, sigma):
    return np.exp(-((x - xc)**2 + (y - yc)**2) / (2 * sigma**2))

# Parameters
xc, yc = 0.0, 0.0  # Center point for both functions
sigma = 1.0        # Width parameter for Gaussian RBF
l = 1.0            # Length parameter for kernel function

# Create grid
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# Calculate distances for kernel function
D = np.sqrt(X**2 + Y**2)

# Compute values for both functions
Z_kernel = kernel_function(D, l)
Z_gaussian = gaussian_rbf_2d(X, Y, xc, yc, sigma)

# Create 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot both surfaces with different opacities and colors
ax.plot_surface(X, Y, Z_kernel, cmap='viridis', alpha=0.7, label='Kernel Function')
ax.plot_surface(X, Y, Z_gaussian, cmap='plasma', alpha=0.7, label='Gaussian RBF')

# Add labels and title
ax.set_title('Comparison of Kernel Function and Gaussian RBF')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Function Value')

# Add legend
ax.legend()

# Show plot
plt.show()