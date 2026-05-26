import random
import os
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """
    Sets all random seeds for PyTorch, NumPy, and Python standard library
    to ensure full deterministic reproducibility.
    
    Args:
        seed (int): The seed value to set. Defaults to 42.
    """
    # 1. Standard Python random generator
    random.seed(seed)
    
    # 2. NumPy random generator
    np.random.seed(seed)
    
    # 3. Environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 4. PyTorch CPU and GPU generators
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Multi-GPU support
        
        # 5. CUDA backend determinism
        # These settings ensure that PyTorch convolutional algorithms use deterministic methods
        # Tradeoff: May slightly reduce training speed, but is critical for scientific reproducibility.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    print(f"[SEED] Random seed set to {seed} for reproducibility (PyTorch CUDNN determinism: True)")
