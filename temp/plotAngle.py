'''
Author: chasey melancholycy@gmail.com
Date: 2025-03-13 13:42:51
FilePath: /mesh_planner/test/temp/plotAngle.py
Description: 

Copyright (c) 2025 by chasey (melancholycy@gmail.com), All Rights Reserved. 
'''
import matplotlib.pyplot as plt
import numpy as np

# Define the angles in radians
angle1 = 2.878
angle2 = -0.277

# Create a figure and axis
fig, ax = plt.subplots()

# Set the limits of the plot to show a unit circle
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)

# Draw a unit circle for reference
circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--')
ax.add_patch(circle)

# Draw the yaw angles as arrows
arrow_length = 1  # Length of the arrow
arrow_width = 0.05  # Width of the arrow

# Arrow for angle1
ax.arrow(0, 0, arrow_length * np.cos(angle1), arrow_length * np.sin(angle1),
         head_width=arrow_width, head_length=arrow_width*2, color='blue', label='Angle 1: {:.3f} rad'.format(angle1))

# Arrow for angle2
ax.arrow(0, 0, arrow_length * np.cos(angle2), arrow_length * np.sin(angle2),
         head_width=arrow_width, head_length=arrow_width*2, color='red', label='Angle 2: {:.3f} rad'.format(angle2))

# Add labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Visualization of Yaw Angles')
ax.legend()

# Set aspect ratio to be equal
ax.set_aspect('equal')

# Show the plot
plt.grid()
plt.show()