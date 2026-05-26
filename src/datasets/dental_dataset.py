import os
import json
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Optional, Callable
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class DentalDataset(Dataset):
    """
    Custom PyTorch Dataset for Dental Image Classification.
    Features:
        - Scans the data directory for class folders and registers all images alphabetically.
        - Provides stratified train/val/test splits with deterministic seeding.
        - Persists splits in a JSON file to guarantee consistent cross-architecture comparisons.
        - Applies user-supplied albumentations/torchvision transforms.
    """
    CLASSES = [
        "lower_front",
        "lower_left",
        "lower_occlusal",
        "lower_right",
        "upper_front",
        "upper_left",
        "upper_occlusal",
        "upper_right"
    ]
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
        transform: Optional[Callable] = None,
        seed: int = 42,
        splits_json_path: Optional[str] = None
    ):
        """
        Args:
            data_dir (str): Path to the augmented/processed data folder containing subfolders for each class.
            split (str): One of 'train', 'val', or 'test'.
            split_ratios (Tuple[float, float, float]): (train_ratio, val_ratio, test_ratio). Must sum to 1.0.
            transform (Callable, optional): Image augmentation/normalization transform pipeline.
            seed (int): Random seed for stratified split. Defaults to 42.
            splits_json_path (str, optional): Custom path to save/load split indices.
        """
        assert split in ['train', 'val', 'test'], "Split must be 'train', 'val', or 'test'"
        assert abs(sum(split_ratios) - 1.0) < 1e-5, "Split ratios must sum to 1.0"
        
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.seed = seed
        
        # Determine path to save/load split indices
        if splits_json_path is None:
            # Save inside data_dir's parent directory if possible, or workspace
            parent_dir = os.path.dirname(data_dir)
            splits_dir = os.path.join(parent_dir, "splits")
            self.splits_json_path = os.path.join(splits_dir, "split_indices.json")
        else:
            self.splits_json_path = splits_json_path
            
        # 1. Discover all images and assign labels
        self.all_samples: List[Tuple[str, int]] = []
        self._scan_dataset()
        
        # 2. Partition dataset into stratified train/val/test splits
        self.samples = self._get_split_samples(split_ratios)
        
    def _scan_dataset(self):
        """Discovers all images under data_dir, ensuring class folders match defined classes."""
        for class_idx, class_name in enumerate(self.CLASSES):
            class_folder = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_folder):
                raise FileNotFoundError(
                    f"Expected class folder '{class_name}' at {class_folder}, but it was not found."
                )
                
            # Scan files with standard image extensions
            valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
            files = sorted([
                f for f in os.listdir(class_folder) 
                if f.lower().endswith(valid_extensions)
            ])
            
            for file in files:
                file_path = os.path.join(class_folder, file)
                self.all_samples.append((file_path, class_idx))
                
        if len(self.all_samples) == 0:
            raise RuntimeError(f"No valid images found in dataset directory: {self.data_dir}")
            
    def _get_split_samples(self, split_ratios: Tuple[float, float, float]) -> List[Tuple[str, int]]:
        """Handles deterministic, stratified split saving and loading."""
        # Check if splits JSON exists
        if os.path.exists(self.splits_json_path):
            try:
                with open(self.splits_json_path, 'r', encoding='utf-8') as f:
                    splits_dict = json.load(f)
                
                # Verify splits_dict contains correct split
                if self.split in splits_dict:
                    indices = splits_dict[self.split]
                    return [self.all_samples[idx] for idx in indices]
            except Exception as e:
                print(f"[DATASET] Warning: Failed to load split indices from {self.splits_json_path}: {e}. Recalculating...")
                
        # If splits JSON does not exist, compute it deterministically
        print(f"[DATASET] Splits JSON not found or invalid. Computing stratified split with seed {self.seed}...")
        
        paths = [s[0] for s in self.all_samples]
        labels = [s[1] for s in self.all_samples]
        indices = np.arange(len(self.all_samples))
        
        train_ratio, val_ratio, test_ratio = split_ratios
        
        # First split: Train vs Temp (Val + Test)
        temp_ratio = val_ratio + test_ratio
        train_idx, temp_idx, _, temp_labels = train_test_split(
            indices, labels, 
            test_size=temp_ratio, 
            random_state=self.seed, 
            stratify=labels
        )
        
        # Second split: Val vs Test within Temp
        val_relative_ratio = val_ratio / temp_ratio
        val_idx, test_idx = train_test_split(
            temp_idx, 
            test_size=(1 - val_relative_ratio), 
            random_state=self.seed, 
            stratify=temp_labels
        )
        
        # Save indices to JSON
        splits_dict = {
            "train": [int(idx) for idx in train_idx],
            "val": [int(idx) for idx in val_idx],
            "test": [int(idx) for idx in test_idx]
        }
        
        os.makedirs(os.path.dirname(self.splits_json_path), exist_ok=True)
        try:
            with open(self.splits_json_path, 'w', encoding='utf-8') as f:
                json.dump(splits_dict, f, indent=4)
            print(f"[DATASET] Stratified split indices saved successfully to: {self.splits_json_path}")
        except Exception as e:
            print(f"[DATASET] Error: Failed to save split indices to {self.splits_json_path}: {e}")
            
        # Return samples matching current split
        curr_indices = splits_dict[self.split]
        return [self.all_samples[idx] for idx in curr_indices]

    def get_labels(self) -> List[int]:
        """Returns the labels of all samples in the current split. Useful for WeightedRandomSampler."""
        return [s[1] for s in self.samples]
        
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path, label = self.samples[idx]
        
        # Read image using PIL (RGB conversion is important for dental clinical images)
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to load image at {image_path}: {e}")
            
        # Apply transforms (albumentations/torchvision)
        if self.transform:
            # Albumentations expects numpy arrays
            image_np = np.array(image)
            augmented = self.transform(image=image_np)
            image_tensor = augmented['image']
        else:
            # Fallback to standard conversion
            image_tensor = torch.from_numpy(np.array(image).transpose(2, 0, 1)).float() / 255.0
            
        return image_tensor, label
