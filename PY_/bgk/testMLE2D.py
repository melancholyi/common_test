import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

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
    N = X.shape[0]
    
    # Compute posterior parameters for validation data
    mu_star, var_star = compute_posterior_params(X, y, X_val, mu0, sigma2, lambd, c)
    
    # Compute log likelihood for validation data
    log_likelihood = 0.0
    for i in range(len(y_val)):
        mu = mu_star[i]
        var = var_star[i] + sigma2  # Add observation noise
        log_likelihood += -0.5 * np.log(2 * np.pi * var) - 0.5 * (y_val[i] - mu)**2 / var
    
    return -log_likelihood  # Minimize negative log likelihood

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

def plot_true_vs_pred(X_true, y_true, X_pred, y_pred, y_var=None):
    """
    Plot true curve vs predicted curve with optional confidence intervals.
    
    Args:
        X_true: True data locations
        y_true: True data values
        X_pred: Prediction locations
        y_pred: Predicted means
        y_var: Predicted variances (optional)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(X_true, y_true, label='True Curve', color='blue')
    plt.scatter(X_train, y_train, label='Training Data', color='red', marker='x')
    plt.plot(X_pred, y_pred, label='Predicted Curve', color='green', linestyle='--')
    
    if y_var is not None:
        y_std = np.sqrt(y_var)
        plt.fill_between(X_pred.flatten(), y_pred - y_std, y_pred + y_std, 
                         color='green', alpha=0.2, label='Prediction Interval')
    
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('True vs Predicted Curves')
    plt.show()

# Example usage
if __name__ == "__main__":
    # Generate synthetic data with a true curve
    np.random.seed(42)
    N = 30
    X_train = np.linspace(0, 10, N).reshape(-1, 1)
    true_curve = lambda x: 2 * np.sin(x) + 0.5 * x
    y_train = true_curve(X_train) + np.random.normal(0, 0.5, size=N).reshape(-1, 1)
    
    # Create validation data
    X_val = np.linspace(0, 10, 100).reshape(-1, 1)
    y_val_true = true_curve(X_val)
    
    # Prior parameters
    mu0 = 0.0
    sigma2 = 0.25
    
    # Initial hyperparameters (lambd, c)
    initial_params = np.array([1.0, 0.1])
    
    # Perform MLE
    optimized_params = mle_bayesian_kernel_regression(X_train, y_train, X_val, y_val_true, mu0, sigma2, initial_params)
    lambd_opt, c_opt = optimized_params
    
    print(f"Optimized lambd: {lambd_opt}")
    print(f"Optimized c: {c_opt}")
    
    # Generate predictions
    y_pred, y_var = compute_posterior_params(X_train, y_train, X_val, mu0, sigma2, lambd_opt, c_opt)
    
    # Plot results
    plot_true_vs_pred(X_val, y_val_true, X_val, y_pred, y_var)