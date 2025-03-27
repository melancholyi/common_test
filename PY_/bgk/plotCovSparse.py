import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def kernel_function(d_i, l=1.0):
    """
    Compute the kernel function value for given distances d_i and length l.
    
    Parameters:
    d_i (numpy array): Array of distances.
    l (float): Length parameter of the kernel (default is 1.0).
    
    Returns:
    numpy array: Kernel values corresponding to each distance in d_i.
    """
    k = np.zeros_like(d_i)
    valid_indices = d_i <= l
    
    cos_term = np.cos(2 * np.pi * d_i[valid_indices] / l)
    sin_term = np.sin(2 * np.pi * d_i[valid_indices] / l)
    
    k[valid_indices] = ((2 + cos_term) / 3) * (1 - d_i[valid_indices] / l) + sin_term / (2 * np.pi)
    
    return k

# Create a grid of x and y values
x = np.linspace(-2, 2, 1000)
y = np.linspace(-2, 2, 1000)
X, Y = np.meshgrid(x, y)

# Calculate the distance from the center for each point
D = np.sqrt(X**2 + Y**2)

# Compute kernel values
Z = kernel_function(D)

# Create 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

# Add a color bar which maps values to colors
fig.colorbar(surf, shrink=0.5, aspect=5)

# Set labels
ax.set_xlabel('X Distance (m)')
ax.set_ylabel('Y Distance (m)')
ax.set_zlabel('Kernel Value')
ax.set_title('3D Kernel Function')

# Show the plot
plt.show()