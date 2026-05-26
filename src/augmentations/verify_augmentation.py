import os
import cv2
import json
import hashlib
import matplotlib.pyplot as plt
from typing import Dict, List

def calculate_md5(file_path: str) -> str:
    """Calculate the MD5 checksum of a file."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

def run_verification(augmented_dir: str, output_plot_path: str) -> None:
    print("=" * 60)
    print("STARTING DATA AUGMENTATION VERIFICATION PIPELINE")
    print("=" * 60)
    
    if not os.path.exists(augmented_dir):
        print(f"Error: Augmented directory '{augmented_dir}' does not exist.")
        return
        
    classes = [d for d in os.listdir(augmented_dir) if os.path.isdir(os.path.join(augmented_dir, d))]
    classes.sort()
    
    total_files = 0
    corrupted_count = 0
    duplicate_hash_count = 0
    hashes_seen = set()
    
    class_file_counts = {}
    
    print("\nVerifying image integrity and checking for duplicates...")
    for cls in classes:
        cls_path = os.path.join(augmented_dir, cls)
        files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        class_file_counts[cls] = len(files)
        total_files += len(files)
        
        for f in files:
            fpath = os.path.join(cls_path, f)
            
            # 1. Check for integrity: can we open the image with OpenCV?
            try:
                img = cv2.imread(fpath)
                if img is None or img.size == 0:
                    corrupted_count += 1
                    print(f"  [CORRUPTED] Failed to load {fpath}")
                    continue
            except Exception as e:
                corrupted_count += 1
                print(f"  [CORRUPTED] Exception loading {fpath}: {str(e)}")
                continue
                
            # 2. Check for exact file duplicate hashes in output
            fhash = calculate_md5(fpath)
            if fhash in hashes_seen:
                duplicate_hash_count += 1
                print(f"  [DUPLICATE HASH] File {fpath} matches an existing file hash.")
            else:
                hashes_seen.add(fhash)
                
    print(f"\n--- INTEGRITY CHECKS SUMMARY ---")
    print(f"Total files checked: {total_files}")
    print(f"Corrupted images found: {corrupted_count}")
    print(f"Exact duplicates found in augmented output: {duplicate_hash_count}")
    
    # 3. Create a beautiful grid plot of original and augmented samples
    print("\nGenerating visual verification samples grid...")
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    
    # We will pick 4 representative classes that have augmentations and show a grid:
    # 1 original file and 4 of its augmented versions.
    sample_classes = [c for c in classes if c != "lower_left"][:4]
    
    fig, axes = plt.subplots(len(sample_classes), 5, figsize=(15, 3 * len(sample_classes)))
    plt.suptitle("Dental Image Classifier: Augmentation Samples Grid", fontsize=16, fontweight='bold')
    
    for row_idx, cls in enumerate(sample_classes):
        cls_path = os.path.join(augmented_dir, cls)
        files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Find original files (they don't contain "_aug")
        orig_files = [f for f in files if "_aug" not in f]
        orig_files.sort()
        
        if not orig_files:
            continue
            
        selected_orig = orig_files[0]
        # Find augmented files derived from this original
        base_name, _ = os.path.splitext(selected_orig)
        derived_augs = [f for f in files if f.startswith(base_name + "_aug")]
        derived_augs.sort()
        
        # Set up images to display (1 original + up to 4 augmented)
        display_files = [selected_orig] + derived_augs[:4]
        
        for col_idx in range(5):
            ax = axes[row_idx, col_idx]
            if col_idx < len(display_files):
                fpath = os.path.join(cls_path, display_files[col_idx])
                # Read with OpenCV, convert BGR to RGB for matplotlib
                img_bgr = cv2.imread(fpath)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                
                ax.imshow(img_rgb)
                if col_idx == 0:
                    ax.set_title(f"Original\n({cls})", fontsize=10, fontweight='bold')
                else:
                    ax.set_title(f"Augmented {col_idx}", fontsize=9)
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
                ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])
            
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visual sample grid successfully saved to: {output_plot_path}")
    
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)
    print(f"{'Class Name':<20} | {'Output Count':<12} | {'Target Status':<15}")
    print("-" * 55)
    for cls, count in class_file_counts.items():
        status = "PASSED" if count in (650, 684) else "FAILED"
        print(f"{cls:<20} | {count:<12} | {status:<15}")
    print("-" * 55)
    print(f"{'TOTAL':<20} | {total_files:<12} | {'PASSED' if corrupted_count == 0 and duplicate_hash_count == 0 else 'WARNING'}")
    print("=" * 60)

if __name__ == "__main__":
    augmented_dir = r"d:\Dental_Image_Classifier\data\augmented"
    output_plot_path = r"d:\Dental_Image_Classifier\outputs\plots\augmentation_samples.png"
    run_verification(augmented_dir, output_plot_path)
