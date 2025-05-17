import numpy as np
import matplotlib.pyplot as plt

# Generate x values
x = np.linspace(-2, 5, 400)  # From -2 to 5 with 400 points

# Compute y = e^{-x}
y = np.exp(-x)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, label=r'$y = e^{-x}$')
plt.axhline(y=1, color='r', linestyle='--', label='y=1')  # Add horizontal line at y=1
plt.title(r'Plot of $e^{-x}$')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()