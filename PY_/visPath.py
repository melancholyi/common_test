'''
Author: chasey melancholycy@gmail.com
Date: 2025-01-22 09:55:17
FilePath: /utils_ws/src/pythonTest/visPath.py
Description: 

Copyright (c) 2025 by chasey (melancholycy@gmail.com), All Rights Reserved. 
'''
import matplotlib.pyplot as plt

# List of points (x, y, z)
points = [
    (0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0), (2, 2, 0),
    (3, 2, 0), (3, 3, 0), (3, 4, 0), (4, 4, 0), (5, 4, 0),
    (5, 5, 0), (6, 5, 0), (6, 6, 0), (7, 6, 0), (7, 7, 0),
    (7, 8, 0), (8, 8, 0), (8, 9, 0), (9, 9, 0), (9, 10, 0),
    (10, 10, 0)
]

# Extract x and y coordinates for plotting
x_coords = [pt[0] for pt in points]
y_coords = [pt[1] for pt in points]

# Plot the path
plt.figure(figsize=(8, 8))
plt.plot(x_coords, y_coords, marker='o', linestyle='-', color='b', label='Path')

# Add grid and labels
plt.grid(True)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D Path Visualization')
plt.legend()

# Show the plot
plt.show()