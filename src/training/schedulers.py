import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

def get_warmup_cosine_scheduler(
    optimizer: Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr_ratio: float = 1e-3
) -> LambdaLR:
    """
    Creates a learning rate scheduler that combines a linear warmup phase
    with a cosine annealing decay phase.
    
    Mathematical formulation:
    - For epoch < warmup_epochs:
        lr = base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * (epoch / warmup_epochs))
    - For epoch >= warmup_epochs:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr = base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + cos(pi * progress)))
        
    Args:
        optimizer (Optimizer): The PyTorch optimizer.
        warmup_epochs (int): Number of epochs for linear warmup.
        total_epochs (int): Total expected training epochs (including warmup).
        min_lr_ratio (float): Lower bound for LR as a fraction of initial LR. Defaults to 1e-3.
        
    Returns:
        LambdaLR: A PyTorch learning rate scheduler.
    """
    assert total_epochs > warmup_epochs, "Total epochs must be strictly greater than warmup epochs."
    
    def lr_lambda(current_epoch: int) -> float:
        if current_epoch < warmup_epochs:
            # 1. Linear Warmup Phase
            # Interpolates from (min_lr_ratio * base_lr) to (1.0 * base_lr)
            if warmup_epochs == 0:
                return 1.0
            return min_lr_ratio + (1.0 - min_lr_ratio) * (current_epoch / warmup_epochs)
        else:
            # 2. Cosine Annealing Decay Phase
            # Calculates progress through the remaining epochs and decays via cosine wave
            progress = (current_epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
            
    return LambdaLR(optimizer, lr_lambda)
