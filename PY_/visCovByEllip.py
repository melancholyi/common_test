import numpy as np
import matplotlib.pyplot as plt

# Define the mean and covariance matrix for the first Gaussian distribution
mean1 = [0, 0]
cov1 = [[1, 0.8], [0.8, 1]]

# Define the mean and covariance matrix for the second Gaussian distribution
mean2 = [-5, 5]
cov2 = [[2, 1], [1, 2]]

invcov1_eval, invcov1_evec = np.linalg.eigh(np.linalg.inv(cov1))
print("np.linalg.inv(cov1):", np.linalg.inv(cov1))
print("invcov1_eval:", invcov1_eval)
print("invcov1_evec:", invcov1_evec)
P1 = invcov1_evec @ (np.diag(invcov1_eval)*1/(3**2)) @ invcov1_evec.T
print("P1:", P1)
invcov2_eval, invcov2_evec = np.linalg.eigh(np.linalg.inv(cov2))
P2 = invcov2_evec @ (np.diag(invcov2_eval)*1/(3**2)) @ invcov2_evec.T
print("(np.diag(invcov2_eval)*1/(3**2)):\n", (np.diag(invcov2_eval)*1/(3**2)))
print("P2:", P2)



# Sample points from the first Gaussian distribution
num_samples = 1000
samples1 = np.random.multivariate_normal(mean1, cov1, num_samples)

# Sample points from the second Gaussian distribution
samples2 = np.random.multivariate_normal(mean2, cov2, num_samples)

# Compute eigenvalues and eigenvectors for the first Gaussian distribution
eigenvalues1, eigenvectors1 = np.linalg.eigh(cov1)
print("eigenvalues1:\n", eigenvalues1)
print("eigenvectors1:\n", eigenvectors1)


# Compute eigenvalues and eigenvectors for the second Gaussian distribution
eigenvalues2, eigenvectors2 = np.linalg.eigh(cov2)  
print("eigenvalues2:\n", eigenvalues2)  
print("eigenvectors2:\n", eigenvectors2)  

# Generate points on a unit circle
theta = np.linspace(0, 2 * np.pi, 1000)
x = np.cos(theta)
y = np.sin(theta)

# Scale and rotate points for the first Gaussian distribution
x_scaled1 = x * np.sqrt(eigenvalues1[0]) * 3  # half-axis length b
y_scaled1 = y * np.sqrt(eigenvalues1[1]) * 3  # half-axis length a
points1 = np.dot(np.column_stack((x_scaled1, y_scaled1)), eigenvectors1.T)
print("ellipsoid2 axis half-len:", np.sqrt(eigenvalues1[0]) * 3, " ", np.sqrt(eigenvalues1[1]) * 3)    

# Translate points to the mean of the first Gaussian distribution
points1 += mean1

# Scale and rotate points for the second Gaussian distribution
x_scaled2 = x * np.sqrt(eigenvalues2[0]) * 3  # half-axis length b
y_scaled2 = y * np.sqrt(eigenvalues2[1]) * 3  # half-axis length a
print("ellipsoid2 axis half-len:", np.sqrt(eigenvalues2[0]) * 3, " ", np.sqrt(eigenvalues2[1]) * 3)    

points2 = np.dot(np.column_stack((x_scaled2, y_scaled2)), eigenvectors2.T)

# Translate points to the mean of the second Gaussian distribution
points2 += mean2

# Plot sample points from the first Gaussian distribution
plt.scatter(samples1[:, 0], samples1[:, 1], label='Sample Points 1', alpha=0.5, color='blue')

# Plot the ellipse for the first Gaussian distribution
plt.plot(points1[:, 0], points1[:, 1], label='Ellipse 1', color='orange')

# Plot sample points from the second Gaussian distribution
plt.scatter(samples2[:, 0], samples2[:, 1], label='Sample Points 2', alpha=0.5, color='green')

# Plot the ellipse for the second Gaussian distribution
plt.plot(points2[:, 0], points2[:, 1], label='Ellipse 2', color='purple')

# Plot the mean points of both Gaussian distributions
plt.scatter(mean1[0], mean1[1], color='red', label='Mean 1')
plt.scatter(mean2[0], mean2[1], color='red', label='Mean 2')

# Set the chart title and axis labels
plt.title('Sample Points and Ellipses of Two Gaussian Distributions')
plt.xlabel('X')
plt.ylabel('Y')

# Set the axis scale
plt.axis('equal')

# Show the legend
plt.legend()

# Display the chart
plt.show()