import numpy as np
import matplotlib.pyplot as plt

# Generate training data
np.random.seed(42)
train_n = 100
pred_n = 200
data_range_min = -5
data_range_max = 5
h = 0.2  # Bandwidth
noise_std = 0.1

def generateData(x):
    return 2 * np.sin(x) + 4 * np.cos(x) + x**2

# Generate training data
X_train = np.linspace(data_range_min, data_range_max, train_n)
y_true = generateData(X_train)
noise = np.random.normal(0, noise_std, train_n)
y_train = y_true + noise

# Generate test grid
xStar = np.linspace(data_range_min + 2*h, data_range_max - 2*h, pred_n)

# Sort the training data for easier interpolation
sorted_indices = np.argsort(X_train)
X_train_sorted = X_train[sorted_indices]
y_train_sorted = y_train[sorted_indices]

# Linear interpolation function
def linear_interpolate(x, X_train, y_train):
    # Find the index where x would be inserted to keep the array sorted
    idx = np.searchsorted(X_train, x)
    
    # Handle edge cases
    if idx == 0:
        idx = 1
    elif idx == len(X_train):
        idx = len(X_train) - 1
    
    # Get the two nearest points
    x_left = X_train[idx - 1]
    x_right = X_train[idx]
    y_left = y_train[idx - 1]
    y_right = y_train[idx]
    
    # Calculate the slope
    slope = (y_right - y_left) / (x_right - x_left)
    
    # Calculate the prediction
    y_pred = y_left + slope * (x - x_left)
    
    # Calculate the gradient (slope)
    grad_pred = slope
    
    return y_pred, grad_pred

# Make predictions using linear interpolation
y_pred = np.zeros_like(xStar)
grad_pred = np.zeros_like(xStar)

for i, x in enumerate(xStar):
    y_pred[i], grad_pred[i] = linear_interpolate(x, X_train_sorted, y_train_sorted)

# Plot the results
plt.figure(figsize=(12, 6))

# Plot training data
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, s=15, label='Training Data', alpha=0.9)
# Plot true function
x_true = np.linspace(data_range_min, data_range_max, 1000)
y_true_func = generateData(x_true)
plt.plot(x_true, y_true_func, 'g-', label='True Function')

# Plot predictions
plt.plot(xStar, y_pred, 'r', label='Linear Interpolation Prediction')

# Plot gradients
plt.quiver(xStar, y_pred, np.ones_like(xStar), grad_pred, 
           color='orange', angles='xy', scale_units='xy', scale=10, width=0.003,
           label='Gradient (Slope)')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Interpolation Prediction')
plt.legend()
plt.grid(True)

# Plot gradient comparison
plt.subplot(1, 2, 2)

# Calculate true gradient
def true_gradient(x):
    return 2 * np.cos(x) - 4 * np.sin(x) + 2 * x

grad_true = true_gradient(xStar)

# Plot true gradient
plt.plot(xStar, grad_true, 'm--', label='True Gradient')

# Plot predicted gradient
plt.plot(xStar, grad_pred, 'b-', label='Predicted Gradient')

plt.xlabel('x')
plt.ylabel('Gradient')
plt.title('Gradient Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()