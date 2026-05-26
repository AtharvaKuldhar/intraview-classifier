import os
import hashlib
import json
import shutil
from typing import Dict, List, Set, Tuple

# Class mapping: raw space-containing folder names to standard snake_case folder names
CLASS_MAPPING = {
    "Lower Front View": "lower_front",
    "Lower Left View": "lower_left",
    "Lower Occlusal View": "lower_occlusal",
    "Lower Right View": "lower_right",
    "Upper Front View": "upper_front",
    "Upper Left View": "upper_left",
    "Upper Occlusal View": "upper_occlusal",
    "Upper Right View": "upper_right"
}

def calculate_md5(file_path: str) -> str:
    """Calculate the MD5 checksum of a file in chunks to avoid memory issues."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

def run_deduplication(raw_dir: str, processed_dir: str) -> None:
    """
    Scans the raw data folder, identifies duplicates (within-class and cross-class),
    removes duplicates, and copies unique images to the processed folder.
    """
    print("=" * 60)
    print("STARTING DEDUPLICATION & CLEANING PIPELINE")
    print("=" * 60)
    
    # 1. First Pass: Compute hashes for all files
    print("\nScanning raw dataset and computing file MD5 hashes...")
    hash_to_files: Dict[str, List[Tuple[str, str]]] = {}  # hash -> list of (raw_class_name, file_path)
    class_raw_counts: Dict[str, int] = {raw_cls: 0 for raw_cls in CLASS_MAPPING.keys()}
    
    for raw_class in CLASS_MAPPING.keys():
        class_path = os.path.join(raw_dir, raw_class)
        if not os.path.isdir(class_path):
            print(f"Warning: Class directory {class_path} not found.")
            continue
            
        files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        class_raw_counts[raw_class] = len(files)
        
        for fname in files:
            fpath = os.path.join(class_path, fname)
            fhash = calculate_md5(fpath)
            
            if fhash not in hash_to_files:
                hash_to_files[fhash] = []
            hash_to_files[fhash].append((raw_class, fpath))
            
    print(f"Scanning complete. Processed {sum(class_raw_counts.values())} files.")
    print(f"Total unique file hashes found: {len(hash_to_files)}")

    # 2. Identify cross-class duplicates and within-class duplicates
    cross_class_dupes: List[Dict] = []
    within_class_dupe_count = 0
    unique_images_by_class: Dict[str, List[Tuple[str, str]]] = {cls_name: [] for cls_name in CLASS_MAPPING.values()}
    
    # Track which hashes are completely rejected because they are cross-class duplicates
    rejected_cross_class_hashes: Set[str] = set()

    for fhash, occurrences in hash_to_files.items():
        # Get unique raw class names for this hash
        classes_involved = list(set([occ[0] for occ in occurrences]))
        
        if len(classes_involved) > 1:
            # Cross-class duplicate! User requested to remove all copies of these completely.
            rejected_cross_class_hashes.add(fhash)
            cross_class_dupes.append({
                "hash": fhash,
                "classes_involved": [CLASS_MAPPING[rc] for rc in classes_involved],
                "files": [occ[1] for occ in occurrences]
            })
            continue
            
        # If it is only in one class
        raw_class = occurrences[0][0]
        processed_class_name = CLASS_MAPPING[raw_class]
        
        # Within-class duplicates handling: keep only the first occurrence
        unique_file_path = occurrences[0][1]
        unique_images_by_class[processed_class_name].append((fhash, unique_file_path))
        
        # How many within-class duplicates did we discard?
        if len(occurrences) > 1:
            within_class_dupe_count += (len(occurrences) - 1)

    print(f"\n--- DUPLICATE ANALYSIS SUMMARY ---")
    print(f"Exact within-class duplicate occurrences skipped: {within_class_dupe_count}")
    print(f"Cross-class duplicate groups identified (labels conflict): {len(cross_class_dupes)}")
    for i, dupe in enumerate(cross_class_dupes):
        classes_str = ", ".join(dupe["classes_involved"])
        print(f"  [{i+1}] Hash {dupe['hash'][:8]}... exists in: {classes_str}")
    print(f"--> ALL copies of the {len(cross_class_dupes)} cross-class duplicate images will be REMOVED.")

    # 3. Create processed folder structure and copy files
    print("\nCreating processed folder structure and copying clean files...")
    
    # Ensure processed directory exists
    if os.path.exists(processed_dir):
        print(f"Processed directory {processed_dir} already exists. Cleaning it...")
        shutil.rmtree(processed_dir)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Create snake_case folder for each class
    for clean_class in CLASS_MAPPING.values():
        os.makedirs(os.path.join(processed_dir, clean_class), exist_ok=True)
        
    class_processed_counts: Dict[str, int] = {}
    copied_mapping: Dict[str, str] = {}  # original path -> processed path
    
    for clean_class, images in unique_images_by_class.items():
        # Sort images by path to ensure deterministic naming
        images.sort(key=lambda x: x[1])
        class_processed_counts[clean_class] = len(images)
        
        for idx, (fhash, orig_path) in enumerate(images):
            # Consistently rename files as {class}_{index:04d}.jpg
            new_name = f"{clean_class}_{idx + 1:04d}.jpg"
            dest_path = os.path.join(processed_dir, clean_class, new_name)
            
            # Copy file
            shutil.copy2(orig_path, dest_path)
            copied_mapping[orig_path] = dest_path

    # 4. Generate JSON Report
    report = {
        "raw_counts": {CLASS_MAPPING[rc]: count for rc, count in class_raw_counts.items()},
        "processed_counts": class_processed_counts,
        "within_class_duplicates_removed": within_class_dupe_count,
        "cross_class_duplicates_removed_count": len(cross_class_dupes),
        "cross_class_duplicates_details": cross_class_dupes,
        "total_unique_dataset_size": sum(class_processed_counts.values())
    }
    
    report_path = os.path.join(processed_dir, "dedup_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
        
    print("\n" + "=" * 60)
    print("DEDUPLICATION SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Class Name':<20} | {'Raw Count':<10} | {'Unique Count':<12} | {'Duplicates Removed':<18}")
    print("-" * 70)
    for raw_cls, clean_cls in CLASS_MAPPING.items():
        raw_count = class_raw_counts[raw_cls]
        clean_count = class_processed_counts[clean_cls]
        dupes_removed = raw_count - clean_count
        print(f"{clean_cls:<20} | {raw_count:<10} | {clean_count:<12} | {dupes_removed:<18}")
    print("-" * 70)
    print(f"{'TOTAL':<20} | {sum(class_raw_counts.values()):<10} | {sum(class_processed_counts.values()):<12} | {sum(class_raw_counts.values()) - sum(class_processed_counts.values()):<18}")
    print("=" * 60)
    print(f"Report saved to: {report_path}")
    print("=" * 60)

if __name__ == "__main__":
    raw_dir = r"d:\Dental_Image_Classifier\data\raw"
    processed_dir = r"d:\Dental_Image_Classifier\data\processed"
    run_deduplication(raw_dir, processed_dir)
