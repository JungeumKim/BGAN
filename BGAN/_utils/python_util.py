import torch
import random
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)                  # Sets the seed for CPU operations
    torch.cuda.manual_seed(seed)             # Sets the seed for GPU operations (if using CUDA)
    torch.cuda.manual_seed_all(seed)         # Sets the seed for all GPUs (if using multiple GPUs)
    random.seed(seed)                        # Set the seed for Python's built-in random module
    np.random.seed(seed)                     # Set the seed for NumPy's random number generator
    
    # Ensures deterministic behavior for some operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
