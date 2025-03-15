import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.transforms import Affine2D

# Define the mean and covariance matrix
mean = np.array([0, 0])
cov = np.array([[1, 0.5], [0.5, 1]])  # Example covariance matrix

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(cov)

# Sort eigenvalues and eigenvectors (eigenvalues are already sorted in ascending order)
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Calculate the semi-axes lengths (3-sigma rule)
semi_axes_lengths = 3 * np.sqrt(eigenvalues)

# Create a figure and axis
fig, ax = plt.subplots()

# Create an ellipse without rotation
ellipse = Ellipse(mean, semi_axes_lengths[0], semi_axes_lengths[1],
                 edgecolor='r', facecolor='none', lw=2)

# Apply rotation using Affine2D
rotation = Affine2D().rotate_deg(np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])))
ellipse.set_transform(rotation + ax.transData)

# Add the ellipse patch to the axis
ax.add_patch(ellipse)

# Set the limits of the plot
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)

# Set equal aspect ratio
ax.set_aspect('equal')

# Display the plot
plt.grid(True)
plt.show()