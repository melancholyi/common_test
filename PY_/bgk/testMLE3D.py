import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib import tri as mtri
from mpl_toolkits.mplot3d import Axes3D

def kernel(x1, x2, c):
    """Squared exponential kernel."""
    return np.exp(-c * np.linalg.norm(x1 - x2)**2)

def compute_posterior_params(X, y, X_star, mu0, sigma2, lambd, c):
    """
    Compute posterior mean and variance for mu* at X_star.
    
    Args:
        X: Training data locations (N x D)
        y: Training data observations (N x 1)
        X_star: New location(s) (M x D)
        mu0: Prior mean
        sigma2: Known variance
        lambd: Hyperparameter
        c: Kernel scale parameter
        
    Returns:
        mu_star: Posterior mean at X_star (M x 1)
        var_star: Posterior variance at X_star (M x 1)
    """
    N = X.shape[0]
    M = X_star.shape[0]
    
    mu_star = np.zeros(M)
    var_star = np.zeros(M)
    
    for m in range(M):
        k_sum = 0.0
        numerator = lambd * mu0
        for i in range(N):
            k_val = kernel(X[i], X_star[m], c)
            k_sum += k_val
            numerator += k_val * y[i]
        lambd_star = lambd + k_sum
        mu_star[m] = numerator / lambd_star
        var_star[m] = sigma2 / lambd_star
    
    return mu_star, var_star

def log_marginal_likelihood(params, X, y, mu0, sigma2, X_val, y_val):
    """
    Compute the log marginal likelihood for validation data.
    
    Args:
        params: Hyperparameters to optimize (lambd, c)
        X: Training data locations
        y: Training data observations
        mu0: Prior mean
        sigma2: Known variance
        X_val: Validation data locations
        y_val: Validation data observations
        
    Returns:
        Negative log marginal likelihood (for minimization)
    """
    lambd, c = params
    N = y_val.shape[0]
    
    # Compute posterior parameters for validation data
    mu_star, var_star = compute_posterior_params(X, y, X_val, mu0, sigma2, lambd, c)

    mll = 0.0
    for i in range(N):  
        mu = mu_star[i]
        var = var_star[i]
        mll += mu/var*mu + np.log(np.abs(var))
    mll /= (-2*N)

    return -mll
    # origin 
    # # Compute log likelihood for validation data
    # log_likelihood = 0.0
    # for i in range(len(y_val)):
    #     mu = mu_star[i]
    #     var = var_star[i] + sigma2  # Add observation noise
    #     log_likelihood += -0.5 * np.log(2 * np.pi * var) - 0.5 * (y_val[i] - mu)**2 / var
    
    # return -log_likelihood  # Minimize negative log likelihood



def loss_mse_opt(params, X, y, mu0, sigma2, X_val, y_val):
    """
    Compute the log marginal likelihood for validation data.
    
    Args:
        params: Hyperparameters to optimize (lambd, c)
        X: Training data locations
        y: Training data observations
        mu0: Prior mean
        sigma2: Known variance
        X_val: Validation data locations
        y_val: Validation data observations
        
    Returns:
        Negative log marginal likelihood (for minimization)
    """
    lambd, c = params
    N = X.shape[0]
    
    # Compute posterior parameters for validation data
    mu_star, var_star = compute_posterior_params(X, y, X_val, mu0, sigma2, lambd, c)

    return mse(y_val, mu_star)
    
    # # Compute log likelihood for validation data
    # log_likelihood = 0.0
    # for i in range(len(y_val)):
    #     mu = mu_star[i]
    #     var = var_star[i] + sigma2  # Add observation noise
    #     log_likelihood += -0.5 * np.log(2 * np.pi * var) - 0.5 * (y_val[i] - mu)**2 / var
    
    # return -log_likelihood  # Minimize negative log likelihood


def mle_bayesian_kernel_regression(X_train, y_train, X_val, y_val, mu0, sigma2, initial_params):
    """
    Perform Maximum Likelihood Estimation for Bayesian Kernel Regression.
    
    Args:
        X_train: Training data locations
        y_train: Training data observations
        X_val: Validation data locations
        y_val: Validation data observations
        mu0: Prior mean
        sigma2: Known variance
        initial_params: Initial guess for hyperparameters (lambd, c)
        
    Returns:
        Optimized hyperparameters (lambd, c)
    """
    result = minimize(
        log_marginal_likelihood,
        initial_params,
        args=(X_train, y_train, mu0, sigma2, X_val, y_val),
        method='L-BFGS-B',
        bounds=[(1e-6, None), (1e-6, None)]  # Ensure positive hyperparameters
    )
    
    return result.x

def plot_elevation_surface_with_true(X_train, y_train, X_grid, y_pred, y_true):
    """
    Plot 3D surface of predicted elevation and true elevation.
    
    Args:
        X_train: Training data locations
        y_train: Training data observations
        X_grid: Grid locations for prediction
        y_pred: Predicted means
        y_true: True values without noise
    """
    fig = plt.figure(figsize=(15, 8))
    
    # Create a 3D axis for the predicted elevation
    ax_pred = fig.add_subplot(121, projection='3d')
    
    # Plot training data on predicted elevation plot
    ax_pred.scatter(X_train[:, 0], X_train[:, 1], y_train, c='red', marker='o', label='Training Data with Noise', alpha=0.6)
    
    # Plot predicted elevation surface
    X1 = X_grid[:, 0].reshape(50, 50)
    X2 = X_grid[:, 1].reshape(50, 50)
    Y_pred = y_pred.reshape(50, 50)
    
    surf_pred = ax_pred.plot_surface(X1, X2, Y_pred, cmap='terrain', alpha=0.8)
    
    fig.colorbar(surf_pred, ax=ax_pred, label='Predicted Elevation')
    ax_pred.set_xlabel('X Coordinate')
    ax_pred.set_ylabel('Y Coordinate')
    ax_pred.set_zlabel('Elevation')
    ax_pred.set_title('Predicted Elevation Surface')
    ax_pred.legend()
    
    # Create a 3D axis for the true elevation
    ax_true = fig.add_subplot(122, projection='3d')
    
    # Plot true elevation surface
    Y_true = y_true.reshape(50, 50)
    
    surf_true = ax_true.plot_surface(X1, X2, Y_true, cmap='terrain', alpha=0.8)
    
    fig.colorbar(surf_true, ax=ax_true, label='True Elevation')
    ax_true.set_xlabel('X Coordinate')
    ax_true.set_ylabel('Y Coordinate')
    ax_true.set_zlabel('Elevation')
    ax_true.set_title('True Elevation Surface')
    
    plt.tight_layout()
    plt.show()


def calculate_mse(y_true, y_pred):
    """
    Calculate the Mean Squared Error (MSE) between true and predicted values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MSE value
    """
    return np.mean((y_true - y_pred) ** 2)


# Example usage for elevation prediction
if __name__ == "__main__":
    # Generate synthetic elevation data
    np.random.seed(42)
    N = 100
    D = 2
    X_train = np.random.rand(N, D) * 10  # Random locations in 10x10 area
    true_elevation = lambda x: 2 * np.sin(x[:, 0]) + 3 * np.cos(x[:, 1]) + x[:, 0] + x[:, 1]
    y_train = true_elevation(X_train) + np.random.normal(0, 1, size=N)
    
    # Create a grid for prediction
    x1 = np.linspace(0, 10, 50)
    x2 = np.linspace(0, 10, 50)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.vstack([X1.ravel(), X2.ravel()]).T
    
    # Generate true elevation values for the grid
    y_true_grid = true_elevation(X_grid)
    
    # Prior parameters
    mu0 = 0.0
    sigma2 = 1.0
    
    # Initial hyperparameters (lambd, c)
    initial_params = np.array([1.0, 0.1])
    
    # Perform MLE
    optimized_params = mle_bayesian_kernel_regression(X_train, y_train, X_train, y_train, mu0, sigma2, initial_params)
    lambd_opt, c_opt = optimized_params
    
    print(f"Optimized lambd: {lambd_opt}")
    print(f"Optimized c: {c_opt}")
    
    # Generate predictions
    # y_pred, y_var = compute_posterior_params(X_train, y_train, X_grid, mu0, sigma2, lambd_opt, c_opt)
    y_pred, y_var = compute_posterior_params(X_train, y_train, X_grid, mu0, sigma2, lambd_opt,c_opt)
    
    mse = calculate_mse(y_true_grid, y_pred)
    print(f'mse: {mse}')


    # Plot results
    plot_elevation_surface_with_true(X_train, y_train, X_grid, y_pred, y_true_grid)