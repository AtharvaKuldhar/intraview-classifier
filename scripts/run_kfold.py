import os
import sys
import yaml
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold

# Add project root to python path for seamless imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.seed import set_seed
from src.utils.logger import ExperimentLogger
from src.datasets.data_utils import get_transforms
from src.datasets.dental_dataset import DentalDataset
from src.models.model_factory import create_model
from src.training.trainer import Trainer

def parse_kfold_args():
    parser = argparse.ArgumentParser(description="Stratified K-Fold Cross Validation Engine for Dental View Classification.")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to the model specific YAML configuration file (e.g., src/configs/resnet50.yaml)"
    )
    parser.add_argument(
        "--k", 
        type=int, 
        required=True,
        choices=[2, 3, 5, 7, 9],
        help="Number of folds (K) to execute."
    )
    parser.add_argument(
        "--drive", 
        action="store_true", 
        default=True,
        help="If True, redirects K-Fold checkpoints and outputs to Google Drive when on Colab."
    )
    return parser.parse_args()

def run_kfold_cross_validation(config_path: str, k: int, use_drive: bool = True):
    # 1. Parse configuration YAML
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    model_cfg = config["model"]
    train_cfg = config["training"]
    data_cfg = config["data"]
    paths_cfg = config["paths"]
    
    # Force strict seeding for reproducibility
    set_seed(train_cfg.get("seed", 42))
    
    # 2. Setup path prefixes for Google Colab
    is_colab = os.path.exists("/content")
    drive_mounted = os.path.exists("/content/drive")
    output_root = ""
    if is_colab and use_drive and drive_mounted:
        output_root = "/content/drive/MyDrive/dental_research"
        
    # Resolve directories
    kfold_outputs_dir = os.path.join(output_root, "outputs", "kfold")
    os.makedirs(kfold_outputs_dir, exist_ok=True)
    
    data_dir = data_cfg["data_dir"]
    if is_colab:
        data_dir = "data/augmented"
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 3. Instantiate the complete dataset (without splits)
    # We will use this master dataset to draw custom Stratified folds
    print(f"[KFOLD] Initializing master dataset from: '{data_dir}'...")
    master_dataset = DentalDataset(
        data_dir=data_dir,
        split='train',  # Dummy split parameter, ignored as we scan all images
        transform=None,  # Transforms will be applied dynamically inside dataloaders
        splits_json_path=os.path.join(output_root, "scratch_kfold_split.json") # dummy
    )
    
    all_samples = master_dataset.all_samples
    paths = [s[0] for s in all_samples]
    labels = np.array([s[1] for s in all_samples])
    
    print(f"[KFOLD] Successfully scanned {len(all_samples)} total images across {len(DentalDataset.CLASSES)} classes.")
    print(f"[KFOLD] Initiating Stratified {k}-Fold Cross-Validation split...")
    
    # 4. Prepare StratifiedKFold partitioner
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=train_cfg.get("seed", 42))
    
    # Arrays to accumulate scores across folds
    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_f1_scores = []
    
    # Transform pipelines
    input_size = data_cfg.get("input_size", 224)
    train_transform = get_transforms(input_size=input_size, is_train=True)
    val_transform = get_transforms(input_size=input_size, is_train=False)
    
    # 5. Loop over the K Folds
    for fold, (train_idx, test_idx) in enumerate(skf.split(paths, labels)):
        print("\n" + "=" * 60)
        print(f"[KFOLD] STARTING FOLD {fold + 1} OF {k}")
        print("=" * 60)
        
        # Define fold logger
        fold_name = f"kfold_{model_cfg['name']}_k{k}_fold{fold + 1}"
        log_dir = os.path.join(output_root, paths_cfg.get("log_dir", "logs"))
        logger = ExperimentLogger(log_dir=log_dir, experiment_name=fold_name, use_tensorboard=False)
        
        # Prepare datasets matching this fold indices
        # We wrap subsets to allow custom transforms per subset
        train_subset = DentalSubsetWrapper(master_dataset, train_idx, train_transform)
        test_subset = DentalSubsetWrapper(master_dataset, test_idx, val_transform)
        
        # Extract training labels for batch balancing
        train_subset_labels = [labels[idx] for idx in train_idx]
        class_counts = np.bincount(train_subset_labels)
        class_weights = 1.0 / class_counts
        sample_weights = np.array([class_weights[label] for label in train_subset_labels])
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=len(sample_weights),
            replacement=True
        )
        
        # Create dataloaders
        batch_size = train_cfg.get("batch_size", 32)
        num_workers = data_cfg.get("num_workers", 2)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        
        # Instantiate fresh model
        model = create_model(
            model_name=model_cfg["name"],
            pretrained=model_cfg.get("pretrained", True),
            num_classes=model_cfg.get("num_classes", 8),
            img_size=input_size
        )
        
        # Compute loss class weights
        loss_class_counts = torch.tensor(list(class_counts))
        loss_class_weights = 1.0 / loss_class_counts.float()
        loss_class_weights = loss_class_weights / loss_class_weights.sum() * 8.0
        loss_class_weights = loss_class_weights.to(device)
        loss_fn = nn.CrossEntropyLoss(weight=loss_class_weights)
        
        # Instantiate Trainer
        fold_checkpoint_dir = os.path.join(output_root, paths_cfg.get("checkpoint_dir", "checkpoints"), "kfold", model_cfg["name"], f"k{k}_fold{fold+1}")
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            logger=logger,
            device=device,
            checkpoint_dir=fold_checkpoint_dir,
            early_stopping_patience=train_cfg.get("early_stopping_patience", 7),
            grad_clip_norm=train_cfg.get("gradient_clip_max_norm", 1.0),
            amp=train_cfg.get("amp", True)
        )
        
        # Check if fold completed successfully in a previous run to support resuming on Colab!
        fold_log_file = os.path.join(log_dir, fold_name, "experiment.log")
        best_checkpoint = os.path.join(fold_checkpoint_dir, "best_model.pth")
        
        is_completed = False
        if os.path.exists(best_checkpoint) and os.path.exists(fold_log_file):
            try:
                with open(fold_log_file, "r", encoding="utf-8") as f:
                    log_content = f.read()
                if f"[FOLD {fold + 1}] Completed!" in log_content:
                    is_completed = True
            except Exception:
                pass
                
        if is_completed:
            print(f"[KFOLD] Fold {fold + 1} was already completed successfully in a previous run. Resuming...")
            trainer.load_checkpoint(best_checkpoint)
            _, eval_metrics = trainer.validate()
            
            fold_accuracies.append(eval_metrics["Val_Accuracy"])
            fold_precisions.append(eval_metrics["Val_Precision_Macro"])
            fold_recalls.append(eval_metrics["Val_Recall_Macro"])
            fold_f1_scores.append(eval_metrics["Val_F1_Macro"])
            
            logger.info(f"[FOLD {fold + 1}] Resumed and loaded! Test Acc: {eval_metrics['Val_Accuracy']:.4f} | Test Macro F1: {eval_metrics['Val_F1_Macro']:.4f}")
            logger.close()
            continue
            
        # Streamline epochs for K-Fold to prevent Colab GPU timeout!
        # Standard: 5 epochs Phase 1 (head calibration), 15 epochs Phase 2 (fine-tune)
        # This keeps total computation very clean while yielding highly representative metrics.
        peak_metrics = trainer.fit(
            phase1_epochs=5,
            phase2_epochs=15,
            phase1_lr=float(train_cfg.get("phase1_lr", 1e-3)),
            phase2_lr=float(train_cfg.get("phase2_lr", 1e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 0.01)),
            warmup_epochs=3
        )
        
        # Load best model for final evaluation metrics on this fold's test set
        best_checkpoint = os.path.join(fold_checkpoint_dir, "best_model.pth")
        trainer.load_checkpoint(best_checkpoint)
        _, eval_metrics = trainer.validate()
        
        # Accumulate metrics
        fold_accuracies.append(eval_metrics["Val_Accuracy"])
        fold_precisions.append(eval_metrics["Val_Precision_Macro"])
        fold_recalls.append(eval_metrics["Val_Recall_Macro"])
        fold_f1_scores.append(eval_metrics["Val_F1_Macro"])
        
        logger.info(f"[FOLD {fold + 1}] Completed! Test Acc: {eval_metrics['Val_Accuracy']:.4f} | Test Macro F1: {eval_metrics['Val_F1_Macro']:.4f}")
        logger.close()
        
    # 6. Compute comparative synthesis averages
    mean_acc, std_acc = np.mean(fold_accuracies), np.std(fold_accuracies)
    mean_prec, std_prec = np.mean(fold_precisions), np.std(fold_precisions)
    mean_rec, std_rec = np.mean(fold_recalls), np.std(fold_recalls)
    mean_f1, std_f1 = np.mean(fold_f1_scores), np.std(fold_f1_scores)
    
    print("\n" + "=" * 70)
    print(f"STRATIFIED K-FOLD COMPARATIVE SUMMARY (K={k}) FOR: {model_cfg['name'].upper()}")
    print("=" * 70)
    print(f"Accuracy  : {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")
    print(f"Precision : {mean_prec * 100:.2f}% ± {std_prec * 100:.2f}%")
    print(f"Recall    : {mean_rec * 100:.2f}% ± {std_rec * 100:.2f}%")
    print(f"F1-Score  : {mean_f1 * 100:.2f}% ± {std_f1 * 100:.2f}%")
    print("=" * 70)
    
    # Save outcomes to a persistent JSON inside Google Drive
    kfold_report_path = os.path.join(kfold_outputs_dir, f"{model_cfg['name']}_k{k}_report.json")
    with open(kfold_report_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": model_cfg["name"],
            "k": k,
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "mean_precision": mean_prec,
            "std_precision": std_prec,
            "mean_recall": mean_rec,
            "std_recall": std_rec,
            "mean_f1": mean_f1,
            "std_f1": std_f1,
            "fold_accuracies": fold_accuracies,
            "fold_f1s": fold_f1_scores
        }, f, indent=4)
    print(f"[KFOLD] Saved persistent summary report to: {kfold_report_path}")
    
    # Append to a markdown table replica in outputs/reports
    kfold_reports_root = os.path.join(output_root, "outputs", "reports")
    os.makedirs(kfold_reports_root, exist_ok=True)
    report_md_path = os.path.join(kfold_reports_root, "kfold_metrics_table.md")
    
    file_exists = os.path.exists(report_md_path)
    with open(report_md_path, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write("# K-Fold Cross Validation Performance Table\n\n")
            f.write("| Model | K-Folds | Accuracy % | Precision % | Recall % | F1-Score % |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        f.write(f"| {model_cfg['name'].upper()} | K={k} | {mean_acc*100:.2f}% ± {std_acc*100:.2f}% | {mean_prec*100:.2f}% ± {std_prec*100:.2f}% | {mean_rec*100:.2f}% ± {std_rec*100:.2f}% | {mean_f1*100:.2f}% ± {std_f1*100:.2f}% |\n")
        
    # Cleanup temporary json
    if os.path.exists(os.path.join(output_root, "scratch_kfold_split.json")):
        try:
            os.remove(os.path.join(output_root, "scratch_kfold_split.json"))
        except Exception:
            pass

class DentalSubsetWrapper(Dataset):
    """
    Subclass wrapper to enable distinct dynamic transforms 
    (training augmentations vs validation evaluations) on Subset indices.
    """
    def __init__(self, dataset: DentalDataset, indices: List[int], transform: Any):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.indices)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        original_idx = self.indices[idx]
        image_path, label = self.dataset.all_samples[original_idx]
        
        # PIL read
        from PIL import Image
        image = Image.open(image_path).convert('RGB')
        
        # Apply specific wrapper transform
        if self.transform:
            image_np = np.array(image)
            augmented = self.transform(image=image_np)
            image_tensor = augmented['image']
        else:
            image_tensor = torch.from_numpy(np.array(image).transpose(2, 0, 1)).float() / 255.0
            
        return image_tensor, label

if __name__ == "__main__":
    args = parse_kfold_args()
    run_kfold_cross_validation(args.config, args.k, args.drive)
