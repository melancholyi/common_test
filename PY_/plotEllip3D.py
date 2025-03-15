'''
Author: chasey melancholycy@gmail.com
Date: 2025-01-24 13:06:16
FilePath: /mesh_planner/test/python/plotEllip.py
Description: 

Copyright (c) 2025 by chasey (melancholycy@gmail.com), All Rights Reserved. 
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

def plot_ellipsoid(ax, center, a, b, rot_yaw, color='b'):
    # Create a rotation object for the given yaw angle
    rot = R.from_euler('z', rot_yaw, degrees=True)
    # Create a rotation matrix
    R_matrix = rot.as_matrix()
    
    # Generate a meshgrid for plotting
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(-np.pi / 2, np.pi / 2, 100)
    x = center[0] + a * R_matrix[0, 0] * np.outer(np.cos(u), np.sin(v)) + a * R_matrix[0, 1] * np.outer(np.sin(u), np.sin(v))
    y = center[1] + b * R_matrix[1, 0] * np.outer(np.cos(u), np.sin(v)) + b * R_matrix[1, 1] * np.outer(np.sin(u), np.sin(v))
    
    # Plot the surface
    ax.plot_surface(x, y, 0 * x, color=color, alpha=0.3)

# Create a figure and a 3D axis
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the first ellipsoid
plot_ellipsoid(ax, np.array([1, 2]), 1, 2, 0, color='blue')

# Plot the second ellipsoid
plot_ellipsoid(ax, np.array([3, 5]), 3, 1, 30, color='red')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Ellipsoids with Different Orientations and Sizes')

# Set the aspect ratio to be equal for better visualization
ax.set_box_aspect([1, 1, 0.5])  # Adjust the z aspect ratio as needed

plt.show()