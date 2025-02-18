import torch
import random
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

def set_seed(seed):
    torch.manual_seed(seed)                  # Sets the seed for CPU operations
    torch.cuda.manual_seed(seed)             # Sets the seed for GPU operations (if using CUDA)
    torch.cuda.manual_seed_all(seed)         # Sets the seed for all GPUs (if using multiple GPUs)
    random.seed(seed)                        # Set the seed for Python's built-in random module
    np.random.seed(seed)                     # Set the seed for NumPy's random number generator
    
    # Ensures deterministic behavior for some operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mmd(X, Y, gamma=1.0):
    """MMD using RBF kernel (k(x,y) = exp(-gamma * ||x-y||^2 / 2))"""
    XX = rbf_kernel(X, X, gamma)
    YY = rbf_kernel(Y, Y, gamma)
    XY = rbf_kernel(X, Y, gamma)
    #set_trace()
    return XX.mean() + YY.mean() - 2 * XY.mean()

def mse(theta, data): # theta: (k,) vector, data: (nxk) matrix.
    se = (theta.view(-1,k)-data)**2
    return se.mean(0)
