'''
Author: chasey melancholycy@gmail.com
Date: 2025-01-30 07:47:55
FilePath: /mesh_planner/test/python/sampleInCircle.py
Description: 

Copyright (c) 2025 by chasey (melancholycy@gmail.com), All Rights Reserved. 
'''
import numpy as np
import matplotlib.pyplot as plt

def sample_grid_points_in_circle(x0, y0, R, d=1):
    """
    Sample grid points within a circle.
    :param x0: x-coordinate of the circle center
    :param y0: y-coordinate of the circle center
    :param R: radius of the circle
    :param d: grid spacing (default is 1)
    :return: list of grid points (x, y) inside the circle
    """
    # Determine the bounding box of the circle
    x_min = int(np.floor(x0 - R))
    x_max = int(np.ceil(x0 + R))
    y_min = int(np.floor(y0 - R))
    y_max = int(np.ceil(y0 + R))

    grid_points = []

    # Iterate over the grid points within the bounding box
    for x in range(x_min, x_max + 1, d):
        for y in range(y_min, y_max + 1, d):
            # Check if the point is inside the circle
            if (x - x0)**2 + (y - y0)**2 <= R**2:
                grid_points.append((x, y))

    return grid_points

# Example usage
x0, y0 = 0, 0  # Circle center
R = 5  # Circle radius
d = 1  # Grid spacing

# Sample grid points
grid_points = sample_grid_points_in_circle(x0, y0, R, d)

# Extract x and y coordinates of the grid points
x_grid, y_grid = zip(*grid_points)

# Plot the circle and the grid points
theta = np.linspace(0, 2 * np.pi, 1000)
circle_x = x0 + R * np.cos(theta)
circle_y = y0 + R * np.sin(theta)

plt.figure(figsize=(8, 8))
plt.plot(circle_x, circle_y, label="Circle", color="blue", linewidth=2)
plt.scatter(x_grid, y_grid, color="red", s=50, label="Grid Points")
plt.title("Grid Points Inside a Circle")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.axis("equal")  # Keep the circle shape
plt.show()