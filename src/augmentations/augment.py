import os
import yaml
import cv2
import json
import shutil
import random
import numpy as np
import hashlib
from typing import Dict, Any, List, Set
from augmentation_transforms import get_transforms_for_tier

def set_seed(seed: int) -> None:
    """Sets random seeds for reproducibility across Python, NumPy, and OpenCV/Albumentations."""
    random.seed(seed)
    np.random.seed(seed)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load the YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_augmentation(config_path: str) -> None:
    """
    Main execution loop to augment images according to the class targets and tier configurations.
    """
    config = load_config(config_path)
    base_seed = config["seed"]
    input_dir = config["input_dir"]
    output_dir = config["output_dir"]
    class_configs = config["classes"]
    tier_configs = config["tiers"]
    
    print("=" * 60)
    print("STARTING CLASS-AWARE DATA AUGMENTATION")
    print("=" * 60)
    print(f"Base seed: {base_seed}")
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}\n")
    
    # Reset/clean output directory
    if os.path.exists(output_dir):
        print(f"Output directory {output_dir} already exists. Cleaning it...")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Mappings and reporting lists
    report_classes: Dict[str, Dict[str, int]] = {}
    
    # Process each class
    for class_idx, (class_name, class_cfg) in enumerate(class_configs.items()):
        target = class_cfg["target"]
        tier = class_cfg["tier"]
        
        class_input_path = os.path.join(input_dir, class_name)
        class_output_path = os.path.join(output_dir, class_name)
        os.makedirs(class_output_path, exist_ok=True)
        
        # Load clean unique images for this class
        if not os.path.isdir(class_input_path):
            print(f"Error: Processed class directory {class_input_path} not found. Skipping.")
            continue
            
        orig_files = [f for f in os.listdir(class_input_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        orig_files.sort()  # crucial for deterministic round-robin
        orig_count = len(orig_files)
        
        print(f"Class '{class_name}': {orig_count} unique, target {target} (Tier {tier})")
        
        # 1. Copy all original clean images to output directory (unchanged)
        # We also resize them to 224x224 so that both original and augmented files have the exact same shape.
        # Track hashes of these copied images to avoid saving identical augmented files.
        print(f"  Copying and standardizing {orig_count} original files...")
        copied_hashes: Set[str] = set()
        for orig_file in orig_files:
            src_file_path = os.path.join(class_input_path, orig_file)
            dest_file_path = os.path.join(class_output_path, orig_file)
            
            # Read, resize to 224x224, and write to output
            img = cv2.imread(src_file_path)
            img_resized = cv2.resize(img, (224, 224))
            cv2.imwrite(dest_file_path, img_resized)
            
            # Record hash of standardized image
            img_hash = hashlib.md5(img_resized.tobytes()).hexdigest()
            copied_hashes.add(img_hash)
            
        # 2. Perform augmentation if needed
        aug_needed = target - orig_count
        
        if aug_needed <= 0 or tier == 0:
            print(f"  No offline augmentation needed for '{class_name}'.")
            report_classes[class_name] = {
                "original": orig_count,
                "augmented_created": 0,
                "total_output": orig_count,
                "tier": tier
            }
            continue
            
        # Build transform pipeline for this tier
        tier_cfg = tier_configs[tier]
        transform_pipeline = get_transforms_for_tier(tier, tier_cfg)
        
        if transform_pipeline is None:
            print(f"  Warning: Transform pipeline for Tier {tier} is None. No augmentations generated.")
            report_classes[class_name] = {
                "original": orig_count,
                "augmented_created": 0,
                "total_output": orig_count,
                "tier": tier
            }
            continue
            
        print(f"  Generating {aug_needed} augmented images using Tier {tier} ({tier_cfg['name']})...")
        
        aug_created = 0
        retry_attempts = 0
        max_retries = 1000
        
        while aug_created < aug_needed and retry_attempts < max_retries:
            # Round-robin selection of original image to augment
            orig_file_idx = aug_created % orig_count
            orig_file_name = orig_files[orig_file_idx]
            orig_file_path = os.path.join(class_input_path, orig_file_name)
            
            # Read original image
            img = cv2.imread(orig_file_path)
            if img is None:
                print(f"  Warning: Failed to read {orig_file_path}. Skipping.")
                continue
                
            # Compute a unique seed for this specific augmented image instance
            # We add retry_attempts to the seed so if it's a duplicate, a different seed is used on retry.
            instance_seed = base_seed + (class_idx * 10000) + (orig_file_idx * 100) + aug_created + retry_attempts
            set_seed(instance_seed)
            
            # Apply transformation
            try:
                augmented = transform_pipeline(image=img)
                aug_img = augmented["image"]
                
                # Check for output validity
                if aug_img is not None and aug_img.size > 0:
                    # Calculate MD5 hash of augmented image in memory
                    aug_hash = hashlib.md5(aug_img.tobytes()).hexdigest()
                    
                    if aug_hash in copied_hashes:
                        # Image was unchanged by augmentation transforms (none of the prob triggered changes)
                        retry_attempts += 1
                        continue
                        
                    # Construct output filename
                    base_name, _ = os.path.splitext(orig_file_name)
                    new_file_name = f"{base_name}_aug{aug_created + 1:04d}.jpg"
                    new_file_path = os.path.join(class_output_path, new_file_name)
                    
                    # Save
                    cv2.imwrite(new_file_path, aug_img)
                    copied_hashes.add(aug_hash)
                    aug_created += 1
                else:
                    retry_attempts += 1
            except Exception as e:
                print(f"  Error applying transform to {orig_file_path}: {str(e)}")
                break
                
        if retry_attempts >= max_retries:
            print(f"  Warning: Reached maximum retries ({max_retries}) trying to generate unique augmentations.")
            
        print(f"  Successfully created {aug_created} augmented files.")
        report_classes[class_name] = {
            "original": orig_count,
            "augmented_created": aug_created,
            "total_output": orig_count + aug_created,
            "tier": tier
        }
        
    # Generate report
    report = {
        "base_seed": base_seed,
        "input_directory": input_dir,
        "output_directory": output_dir,
        "summary": report_classes,
        "total_images_in_augmented_dataset": sum(c["total_output"] for c in report_classes.values())
    }
    
    report_path = os.path.join(output_dir, "augmentation_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
        
    print("\n" + "=" * 60)
    print("AUGMENTATION PIPELINE COMPLETE")
    print("=" * 60)
    print(f"{'Class Name':<20} | {'Clean Unique':<12} | {'Aug Created':<12} | {'Total Balanced':<14} | {'Tier':<5}")
    print("-" * 70)
    for cls_name, info in report_classes.items():
        print(f"{cls_name:<20} | {info['original']:<12} | {info['augmented_created']:<12} | {info['total_output']:<14} | {info['tier']:<5}")
    print("-" * 70)
    total_orig = sum(info["original"] for info in report_classes.values())
    total_aug = sum(info["augmented_created"] for info in report_classes.values())
    total_all = sum(info["total_output"] for info in report_classes.values())
    print(f"{'TOTAL':<20} | {total_orig:<12} | {total_aug:<12} | {total_all:<14} | N/A")
    print("=" * 60)
    print(f"Report saved to: {report_path}")
    print("=" * 60)

if __name__ == "__main__":
    config_path = r"d:\Dental_Image_Classifier\src\configs\augmentation_config.yaml"
    run_augmentation(config_path)
