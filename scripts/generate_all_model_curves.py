import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Orchestrator to generate Fig 9 and Fig 11 style curves for ALL trained models.")
    parser.add_argument(
        "--checkpoints_dir", 
        type=str, 
        default="checkpoints", 
        help="Root folder containing checkpoints."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="outputs/plots/model_curves", 
        help="Global outputs directory to save curves plots."
    )
    parser.add_argument(
        "--drive_prefix", 
        type=str, 
        default=None, 
        help="Optional drive path prefix for Google Colab run redirection."
    )
    return parser.parse_args()

def plot_fig9_training_curves(csv_path: str, model_display_name: str, save_paths: list):
    """Generates Training Accuracy and Loss shaded curves matching Fig 9 in requirements."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[CURVES] Error reading CSV {csv_path}: {e}")
        return
        
    epochs = df['epoch']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Training Accuracy Panel (Blue Shaded Curve)
    axes[0].plot(epochs, df['train_accuracy'], marker='o', markersize=4, color='#1f77b4', linewidth=2, label='Training Accuracy')
    axes[0].fill_between(epochs, df['train_accuracy'], color='#1f77b4', alpha=0.15)
    axes[0].set_title('Training Accuracy', fontsize=14, fontweight='bold', pad=10)
    axes[0].set_xlabel('Epochs', fontsize=11)
    axes[0].set_ylabel('Accuracy', fontsize=11)
    axes[0].set_ylim(0, 1.02)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(loc='lower right')
    
    # 2. Training Loss Panel (Green Shaded Curve)
    axes[1].plot(epochs, df['train_loss'], marker='o', markersize=4, color='#2ca02c', linewidth=2, label='Training Loss')
    axes[1].fill_between(epochs, df['train_loss'], color='#2ca02c', alpha=0.15)
    axes[1].set_title('Training Loss', fontsize=14, fontweight='bold', pad=10)
    axes[1].set_xlabel('Epochs', fontsize=11)
    axes[1].set_ylabel('Loss', fontsize=11)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend(loc='upper right')
    
    plt.suptitle(f"Training Progress Dashboard - {model_display_name}", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    for save_path in save_paths:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.close()

def plot_fig11_train_val_curves(csv_path: str, model_display_name: str, save_paths: list):
    """Generates Training vs Validation comparison curves matching Fig 11 in requirements."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[CURVES] Error reading CSV {csv_path}: {e}")
        return
        
    epochs = df['epoch']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Accuracy Panel (Train vs Val)
    axes[0].plot(epochs, df['train_accuracy'], color='#1f77b4', linewidth=2, label='Train Accuracy')
    if 'val_accuracy' in df.columns:
        axes[0].plot(epochs, df['val_accuracy'], color='#ff7f0e', linewidth=2, label='Validation Accuracy')
    axes[0].set_title('Training vs Validation Accuracy', fontsize=14, fontweight='bold', pad=10)
    axes[0].set_xlabel('Epochs', fontsize=11)
    axes[0].set_ylabel('Accuracy', fontsize=11)
    axes[0].set_ylim(0, 1.02)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(loc='lower right')
    
    # 2. Loss Panel (Train vs Val)
    axes[1].plot(epochs, df['train_loss'], color='#1f77b4', linewidth=2, label='Train Loss')
    if 'val_loss' in df.columns:
        axes[1].plot(epochs, df['val_loss'], color='#ff7f0e', linewidth=2, label='Validation Loss')
    axes[1].set_title('Training vs Validation Loss', fontsize=14, fontweight='bold', pad=10)
    axes[1].set_xlabel('Epochs', fontsize=11)
    axes[1].set_ylabel('Loss', fontsize=11)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend(loc='upper right')
    
    plt.suptitle(f"Training vs Validation Progress - {model_display_name}", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    for save_path in save_paths:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.close()

def main():
    args = parse_args()
    
    is_colab = os.path.exists("/content")
    drive_mounted = os.path.exists("/content/drive")
    
    # Drive prefix redirection
    checkpoints_dir = args.checkpoints_dir
    output_dir = args.output_dir
    
    if args.drive_prefix:
        checkpoints_dir = os.path.join(args.drive_prefix, "checkpoints")
        output_dir = os.path.join(args.drive_prefix, "outputs/plots/model_curves")
    elif is_colab and drive_mounted:
        drive_root = "/content/drive/MyDrive/dental_research"
        checkpoints_dir = os.path.join(drive_root, "checkpoints")
        output_dir = os.path.join(drive_root, "outputs/plots/model_curves")
        
    print("=" * 70)
    print("STARTING BULK TRAINING CURVES GENERATOR PIPELINE")
    print("=" * 70)
    print(f"Scanning root directory: '{checkpoints_dir}'")
    print(f"Global curves output directory: '{output_dir}'\n")
    
    if not os.path.exists(checkpoints_dir):
        print(f"[CURVES] Error: Checkpoints directory '{checkpoints_dir}' does not exist.")
        return
        
    # Walk directory to find all training_logs.csv files
    logs_found = []
    for root, dirs, files in os.walk(checkpoints_dir):
        if "training_logs.csv" in files:
            csv_path = os.path.join(root, "training_logs.csv")
            logs_found.append(csv_path)
            
    if not logs_found:
        print("[CURVES] Warning: No training_logs.csv files found. Make sure models have trained successfully first!")
        return
        
    print(f"[CURVES] Found {len(logs_found)} models with active training history logs.\n")
    
    # Map model names to clean display names for plotting
    model_name_map = {
        "resnet50": "ResNet-50",
        "densenet121": "DenseNet-121",
        "mobilenetv3_small": "MobileNetV3-Small",
        "efficientnet_b2": "EfficientNet-B2",
        "efficientnet_b3": "EfficientNet-B3",
        "convnext_tiny": "ConvNeXt-Tiny",
        "swin_tiny": "Swin-Tiny",
        "dinov2_small": "DINOv2-Small"
    }
    
    for idx, csv_path in enumerate(logs_found):
        # Infer model display name from folder structure
        parent_dir = os.path.basename(os.path.dirname(csv_path)) # e.g. resnet50_20260527_110146
        grandparent_dir = os.path.basename(os.path.dirname(os.path.dirname(csv_path))) # e.g. resnet50
        
        inferred_key = grandparent_dir if grandparent_dir in model_name_map else parent_dir.split("_")[0]
        display_name = model_name_map.get(inferred_key, inferred_key.upper())
        
        print(f"[{idx+1}/{len(logs_found)}] Processing history metrics for '{display_name}'...")
        
        # Prepare dual target paths for save (both in the specific run folder and global paper gallery)
        specific_run_dir = os.path.dirname(csv_path)
        
        fig9_local_path = os.path.join(specific_run_dir, "training_curves.png")
        fig9_global_path = os.path.join(output_dir, f"{inferred_key}_fig9_training_curves.png")
        
        fig11_local_path = os.path.join(specific_run_dir, "train_val_curves.png")
        fig11_global_path = os.path.join(output_dir, f"{inferred_key}_fig11_train_val_curves.png")
        
        # Generate shaded curves (Fig 9 style)
        plot_fig9_training_curves(
            csv_path=csv_path,
            model_display_name=display_name,
            save_paths=[fig9_local_path, fig9_global_path]
        )
        
        # Generate comparative curves (Fig 11 style)
        plot_fig11_train_val_curves(
            csv_path=csv_path,
            model_display_name=display_name,
            save_paths=[fig11_local_path, fig11_global_path]
        )
        
        print(f"  └─ Shaded curve saved to: {fig9_global_path}")
        print(f"  └─ Train vs Val curve saved to: {fig11_global_path}")
        
    print("\n" + "=" * 70)
    print("SUCCESS: Training curves compiled for all models successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
