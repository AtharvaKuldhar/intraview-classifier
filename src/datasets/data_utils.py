import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, Any, Tuple, Optional

from src.datasets.dental_dataset import DentalDataset

def get_transforms(input_size: int = 224, is_train: bool = False) -> A.Compose:
    """
    Creates albumentations transform pipelines.
    Enforces medical realism: strictly NO FLIPS (horizontal/vertical) or structural distortions.
    
    Args:
        input_size (int): Receptive field size for the CNN/Transformer. Defaults to 224.
        is_train (bool): If True, applies mild, safe augmentations (rotation, brightness/contrast)
                         to further prevent overfitting without breaking dental symmetry.
                         If False, only resizes and normalizes.
                         
    Returns:
        A.Compose: The Albumentations transform pipeline.
    """
    # Standard ImageNet normalization coefficients
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    
    if is_train:
        return A.Compose([
            A.Resize(input_size, input_size),
            # Safe minor geometric rotations (within ±10 degrees) to simulate camera alignment variations
            A.ShiftScaleRotate(
                shift_limit=0.05, 
                scale_limit=0.05, 
                rotate_limit=10, 
                p=0.5, 
                border_mode=0
            ),
            # Safe minor exposure/lighting adjustments simulating intraoral camera flash levels
            A.RandomBrightnessContrast(
                brightness_limit=0.1, 
                contrast_limit=0.1, 
                p=0.5
            ),
            # Light blur to simulate camera focus variation
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Normalize(mean=imagenet_mean, std=imagenet_std),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(input_size, input_size),
            A.Normalize(mean=imagenet_mean, std=imagenet_std),
            ToTensorV2()
        ])

def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    input_size: int = 224,
    num_workers: int = 2,
    split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    splits_json_path: Optional[str] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Factory function to prepare train, validation, and test PyTorch DataLoaders.
    Fulfills the Class Imbalance Strategy (AGENTS.md) by applying WeightedRandomSampler
    to guarantee balanced mini-batches in the training loop.
    
    Args:
        data_dir (str): Folder containing the augmented class folders.
        batch_size (int): Size of training batches. Defaults to 32.
        input_size (int): Width/height dimensions for input tensors. Defaults to 224.
        num_workers (int): Parallel loader workers. Defaults to 2.
        split_ratios (Tuple[float, float, float]): (train, val, test) proportions.
        seed (int): Fixed random seed for split reproducibility.
        splits_json_path (str, optional): Custom path to save/load split indices.
        
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader)
    """
    # 1. Instantiate the datasets with split-specific transforms
    train_dataset = DentalDataset(
        data_dir=data_dir,
        split='train',
        split_ratios=split_ratios,
        transform=get_transforms(input_size=input_size, is_train=True),
        seed=seed,
        splits_json_path=splits_json_path
    )
    
    val_dataset = DentalDataset(
        data_dir=data_dir,
        split='val',
        split_ratios=split_ratios,
        transform=get_transforms(input_size=input_size, is_train=False),
        seed=seed,
        splits_json_path=splits_json_path
    )
    
    test_dataset = DentalDataset(
        data_dir=data_dir,
        split='test',
        split_ratios=split_ratios,
        transform=get_transforms(input_size=input_size, is_train=False),
        seed=seed,
        splits_json_path=splits_json_path
    )
    
    # 2. Implement Class-Aware WeightedRandomSampler for Training
    # Extract training labels to compute frequencies
    train_labels = train_dataset.get_labels()
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    
    # Assign a weight to each individual sample based on its class
    sample_weights = np.array([class_weights[label] for label in train_labels])
    sample_weights_tensor = torch.from_numpy(sample_weights).double()
    
    # Instantiate WeightedRandomSampler
    # replacement=True means samples are drawn with replacement to balance classes per batch
    sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights_tensor),
        replacement=True
    )
    
    # 3. Create DataLoaders
    # pin_memory=True speeds up transfer to GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"[DATALOADERS] Initialized loaders successfully:")
    print(f"  └─ Train Split: {len(train_dataset)} images (Balanced via WeightedRandomSampler)")
    print(f"  └─ Val Split:   {len(val_dataset)} images")
    print(f"  └─ Test Split:  {len(test_dataset)} images")
    print(f"  └─ Batch Size:  {batch_size} | Input Dimensions: {input_size}x{input_size}")
    
    return train_loader, val_loader, test_loader
