import os
import sys
import json
import argparse
from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.datasets.data_utils import get_dataloaders
from src.models.model_factory import create_model
from src.utils.metrics import compute_metrics, compute_class_performance, get_confusion_matrix
from src.datasets.dental_dataset import DentalDataset

def parse_eval_args():
    parser = argparse.ArgumentParser(description="Standalone Evaluation Suite for Trained Dental Classifiers.")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to the model specific YAML configuration file."
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True, 
        help="Path to the saved PyTorch model checkpoint (.pth file)."
    )
    parser.add_argument(
        "--drive", 
        action="store_true", 
        default=True,
        help="If True, redirects output folders to Google Drive when running in Colab."
    )
    return parser.parse_args()

@torch.no_grad()
def run_evaluation(
    model: nn.Module, 
    test_loader: DataLoader, 
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Runs global evaluation on the test loader."""
    model.eval()
    all_y_true = []
    all_y_pred = []
    raw_predictions = []
    
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)
        
        all_y_true.append(labels.cpu().numpy())
        all_y_pred.append(preds.cpu().numpy())
        
        # Log individual item predictions for failure cases
        for idx in range(images.size(0)):
            raw_predictions.append({
                "true_label": int(labels[idx].cpu().item()),
                "predicted_label": int(preds[idx].cpu().item()),
                "probabilities": probs[idx].cpu().numpy().tolist()
            })
            
    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    
    return y_true, y_pred, raw_predictions

def plot_confusion_matrix(
    cm: np.ndarray, 
    class_names: List[str], 
    save_path: str,
    title: str = "Confusion Matrix"
):
    """Generates a highly-stylized, professional publication-grade confusion matrix heatmap."""
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=16, pad=20, fontweight='bold')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right', fontsize=10)
    plt.yticks(tick_marks, class_names, fontsize=10)
    
    # Draw counts and percentages inside each cell
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = cm_norm[i, j] * 100
            text = f"{val}\n({pct:.1f}%)"
            plt.text(
                j, i, text,
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if val > thresh else "black",
                fontweight='bold' if val > thresh or pct > 20 else 'normal',
                fontsize=9
            )
            
    plt.tight_layout()
    plt.ylabel('True Class View', fontsize=12, fontweight='semibold')
    plt.xlabel('Predicted Class View', fontsize=12, fontweight='semibold')
    
    # Save with high-dpi for paper insertion
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_curves(csv_path: str, save_path: str):
    """Plots and saves detailed training loss, accuracy, and learning rate progression."""
    if not os.path.exists(csv_path):
        print(f"[EVAL] Warning: CSV log not found at {csv_path}. Skipping training curve plot.")
        return
        
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[EVAL] Error: Failed to parse CSV logs at {csv_path}: {e}")
        return
        
    epochs = df['epoch']
    
    # Create multi-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Loss Panel
    axes[0].plot(epochs, df['train_loss'], label='Train Loss', color='#1f77b4', linewidth=2)
    if 'val_loss' in df.columns:
        axes[0].plot(epochs, df['val_loss'], label='Val Loss', color='#ff7f0e', linewidth=2, linestyle='--')
    axes[0].set_title('Cross-Entropy Loss Curve', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=10)
    axes[0].set_ylabel('Loss', fontsize=10)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(fontsize=10)
    
    # 2. Accuracy Panel
    if 'train_accuracy' in df.columns:
        axes[1].plot(epochs, df['train_accuracy'], label='Train Acc', color='#2ca02c', linewidth=2)
    if 'val_accuracy' in df.columns:
        axes[1].plot(epochs, df['val_accuracy'], label='Val Acc', color='#d62728', linewidth=2, linestyle='--')
    axes[1].set_title('Classification Accuracy Curve', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=10)
    axes[1].set_ylabel('Accuracy', fontsize=10)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend(fontsize=10)
    
    # 3. Learning Rate Panel
    axes[2].plot(epochs, df['learning_rate'], label='LR', color='#9467bd', linewidth=2)
    axes[2].set_title('Learning Rate Progression', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Epoch', fontsize=10)
    axes[2].set_ylabel('Learning Rate', fontsize=10)
    axes[2].set_yscale('log')
    axes[2].grid(True, linestyle=':', alpha=0.6)
    axes[2].legend(fontsize=10)
    
    plt.suptitle("Training Progress Dashboard", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[EVAL] Successfully saved training curves to: {save_path}")

def main():
    # Detect standalone command line invocation
    if len(sys.argv) > 1 and ("--config" in sys.argv or "--checkpoint" in sys.argv):
        args = parse_eval_args()
        config_path = args.config
        checkpoint_path = args.checkpoint
        use_drive = args.drive
    else:
        # Prevent failure if imported or missing arguments
        print("[EVAL] Usage: python evaluator.py --config <config_yaml> --checkpoint <checkpoint_pth>")
        return
        
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    model_cfg = config["model"]
    data_cfg = config["data"]
    
    # Setup paths redirection for Colab
    is_colab = os.path.exists("/content")
    drive_mounted = os.path.exists("/content/drive")
    output_root = ""
    if is_colab and use_drive and drive_mounted:
        output_root = "/content/drive/MyDrive/dental_research"
        
    # Resolve folders
    exp_dir = os.path.dirname(checkpoint_path)
    
    # Adjust dataset path if symlink was set up in Colab
    data_dir = data_cfg["data_dir"]
    if is_colab:
        data_dir = "data/augmented"
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EVAL] Evaluating checkpoint: {checkpoint_path}")
    print(f"[EVAL] Device: {device} | Data folder: {data_dir}")
    
    # 1. Instantiate the dataset loaders (specifically the test split)
    splits_json_path = os.path.join(output_root, "datasets", "splits", "split_indices.json") if output_root else None
    
    _, _, test_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=config["training"].get("batch_size", 32),
        input_size=data_cfg.get("input_size", 224),
        num_workers=data_cfg.get("num_workers", 2),
        split_ratios=tuple(data_cfg.get("split_ratios", [0.70, 0.15, 0.15])),
        seed=config["training"].get("seed", 42),
        splits_json_path=splits_json_path
    )
    
    # 2. Instantiate Model and Load Weights
    model = create_model(
        model_name=model_cfg["name"],
        pretrained=False,  # Weights loaded manually from checkpoint
        num_classes=model_cfg.get("num_classes", 8)
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    print(f"[EVAL] Checkpoint loaded successfully from epoch {checkpoint['epoch']} (Best Val Acc: {checkpoint.get('accuracy', 0.0):.4f})")
    
    # 3. Perform Test Inference
    y_true, y_pred, predictions_list = run_evaluation(model, test_loader, device)
    
    # 4. Compute Test Metrics
    test_metrics = compute_metrics(y_true, y_pred)
    class_perf = compute_class_performance(y_true, y_pred, DentalDataset.CLASSES)
    cm = get_confusion_matrix(y_true, y_pred)
    
    print("\n" + "=" * 50)
    print("GLOBAL TEST SET PERFORMANCE")
    print("=" * 50)
    for k, v in test_metrics.items():
        print(f"{k:<22}: {v:.4f}")
    print("=" * 50)
    
    # 5. Save Structured Evaluation Outputs
    # Save overall metrics
    metrics_save_path = os.path.join(exp_dir, "metrics.json")
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": model_cfg["name"],
            "checkpoint_epoch": checkpoint["epoch"],
            "global_metrics": test_metrics,
            "class_performance": class_perf,
            "confusion_matrix": cm.tolist()
        }, f, indent=4)
    print(f"[EVAL] Saved overall metrics to: {metrics_save_path}")
    
    # Save raw predictions (useful for failure case analysis)
    preds_save_path = os.path.join(exp_dir, "predictions.json")
    with open(preds_save_path, "w", encoding="utf-8") as f:
        json.dump(predictions_list, f, indent=4)
    print(f"[EVAL] Saved raw predictions table to: {preds_save_path}")
    
    # Save Confusion Matrix plot
    cm_save_path = os.path.join(exp_dir, "confusion_matrix.png")
    plot_confusion_matrix(
        cm=cm, 
        class_names=DentalDataset.CLASSES, 
        save_path=cm_save_path,
        title=f"Confusion Matrix - {model_cfg['name'].upper()}"
    )
    print(f"[EVAL] Saved confusion matrix heatmap to: {cm_save_path}")
    
    # Save training history plot if training_logs.csv is available
    csv_log_path = os.path.join(exp_dir, "training_logs.csv")
    if not os.path.exists(csv_log_path):
        # Fallback to check in sibling folders if paths were redirected
        sibling_csv = os.path.join(os.path.dirname(exp_dir), "training_logs.csv")
        if os.path.exists(sibling_csv):
            csv_log_path = sibling_csv
            
    curves_save_path = os.path.join(exp_dir, "training_curves.png")
    plot_training_curves(csv_path=csv_log_path, save_path=curves_save_path)
    
    # Export copy of plots to global outputs directory for convenience
    global_outputs_dir = os.path.join(output_root, "outputs", "plots") if output_root else "outputs/plots"
    os.makedirs(global_outputs_dir, exist_ok=True)
    
    # Create target copy paths
    shutil_copy = True
    try:
        import shutil
        shutil.copy(cm_save_path, os.path.join(global_outputs_dir, f"{model_cfg['name']}_confusion_matrix.png"))
        if os.path.exists(curves_save_path):
            shutil.copy(curves_save_path, os.path.join(global_outputs_dir, f"{model_cfg['name']}_training_curves.png"))
        print(f"[EVAL] Exported copies of plots to global output dir: {global_outputs_dir}")
    except Exception as e:
        print(f"[EVAL] Warning: Failed to export copies of plots: {e}")

if __name__ == "__main__":
    main()
