import os
import sys
import yaml
import argparse
import datetime
import torch
import torch.nn as nn

# Add project root to python path for seamless imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.seed import set_seed
from src.utils.logger import ExperimentLogger
from src.datasets.data_utils import get_dataloaders
from src.models.model_factory import create_model
from src.training.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Model Comparative Training Runner for Dental View Classification.")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to the model specific YAML configuration file (e.g., src/configs/resnet50.yaml)"
    )
    parser.add_argument(
        "--drive", 
        action="store_true", 
        default=True,
        help="If True, automatically redirects all outputs (checkpoints, logs, reports) to Google Drive when running in Colab."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Parse YAML Configuration
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")
        
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    # Extract config sections
    model_cfg = config["model"]
    train_cfg = config["training"]
    data_cfg = config["data"]
    paths_cfg = config["paths"]
    
    # 2. Set Random Seed for Absolute Reproducibility
    set_seed(train_cfg.get("seed", 42))
    
    # 3. Handle Google Drive paths redirection when running in Google Colab
    is_colab = os.path.exists("/content")
    drive_mounted = os.path.exists("/content/drive")
    
    # Root output path setup
    output_root = ""
    if is_colab and args.drive:
        if drive_mounted:
            output_root = "/content/drive/MyDrive/dental_research"
            print(f"[SYSTEM] Google Colab & Mounted Drive detected. Redirecting outputs to: {output_root}")
        else:
            print("[SYSTEM] Warning: Running in Colab, but Google Drive is NOT mounted. Saving locally in Colab runtime.")
            
    # Resolve directories
    checkpoint_dir = os.path.join(output_root, paths_cfg.get("checkpoint_dir", "checkpoints"), model_cfg["name"])
    log_dir = os.path.join(output_root, paths_cfg.get("log_dir", "logs"))
    exp_root_dir = os.path.join(output_root, paths_cfg.get("experiment_dir", "experiments"))
    
    # Adjust dataset path if symlink was set up in Colab
    data_dir = data_cfg["data_dir"]
    if is_colab:
        # Colab workspace symlink path
        data_dir = "data/augmented"
        print(f"[SYSTEM] Overriding dataset directory for Colab environment: '{data_dir}'")
        
    # Create experiment folder name with timestamp to prevent collisions
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{model_cfg['name']}_{timestamp}"
    experiment_dir = os.path.join(exp_root_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 4. Initialize Structured Experiment Logger
    logger = ExperimentLogger(
        log_dir=log_dir,
        experiment_name=experiment_name,
        use_tensorboard=True
    )
    
    # Log environment setup
    logger.info(f"Loaded Config: {args.config}")
    logger.info(f"PyTorch Version: {torch.__version__} | CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU Model: {torch.cuda.get_device_name(0)}")
        
    # Save a copy of the config YAML in the experiment folder for traceability (AGENTS.md rule)
    config_save_path = os.path.join(experiment_dir, "config.yaml")
    with open(config_save_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    logger.info(f"Saved configuration replica to: {config_save_path}")
    
    # 5. Determine Compute Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running execution on: {device}")
    
    # 6. Initialize Data Pipeline (Balanced loaders)
    # Map split path relative to current runtime context
    splits_json_path = os.path.join(output_root, "datasets", "splits", "split_indices.json") if output_root else None
    
    try:
        train_loader, val_loader, test_loader = get_dataloaders(
            data_dir=data_dir,
            batch_size=train_cfg.get("batch_size", 32),
            input_size=data_cfg.get("input_size", 224),
            num_workers=data_cfg.get("num_workers", 2),
            split_ratios=tuple(data_cfg.get("split_ratios", [0.70, 0.15, 0.15])),
            seed=train_cfg.get("seed", 42),
            splits_json_path=splits_json_path
        )
    except Exception as e:
        logger.error(f"Failed to initialize DataLoader pipeline: {e}")
        raise e
        
    # 7. Instantiate Model Backbone and Adapt Head
    model = create_model(
        model_name=model_cfg["name"],
        pretrained=model_cfg.get("pretrained", True),
        num_classes=model_cfg.get("num_classes", 8),
        img_size=data_cfg.get("input_size", 224)
    )
    
    # 8. Compute Class Weights for Cross-Entropy Loss to counter residual imbalances
    # Fulfills weighted loss strategy in AGENTS.md
    train_dataset = train_loader.dataset
    train_labels = train_dataset.get_labels()
    class_counts = torch.tensor(list(torch.bincount(torch.tensor(train_labels)).numpy()))
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum() * 8.0  # Normalized
    class_weights = class_weights.to(device)
    
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    logger.info(f"Computed normalized class-weights for loss function: {class_weights.tolist()}")
    
    # 9. Instantiate Trainer and Begin Fit Loops
    # Set model specific checkpoint folder to save best checkpoint
    model_checkpoint_dir = os.path.join(checkpoint_dir, experiment_name)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        logger=logger,
        device=device,
        checkpoint_dir=model_checkpoint_dir,
        early_stopping_patience=train_cfg.get("early_stopping_patience", 7),
        grad_clip_norm=train_cfg.get("gradient_clip_max_norm", 1.0),
        amp=train_cfg.get("amp", True)
    )
    
    # 10. Start Training (Phase 1 -> Phase 2)
    try:
        peak_metrics = trainer.fit(
            phase1_epochs=train_cfg.get("phase1_epochs", 10),
            phase2_epochs=train_cfg.get("phase2_epochs", 40),
            phase1_lr=float(train_cfg.get("phase1_lr", 1e-3)),
            phase2_lr=float(train_cfg.get("phase2_lr", 1e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 0.01)),
            warmup_epochs=train_cfg.get("warmup_epochs", 5)
        )
        
        # Log peak outcomes in TensorBoard and print
        logger.log_hyperparams(config=config, final_metrics=peak_metrics)
        logger.info("=" * 60)
        logger.info(f"TRAINING COMPLETE FOR: {model_cfg['name'].upper()}")
        logger.info(f"Peak Validation Metrics achieved: {peak_metrics}")
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.warning("[!] Training execution interrupted by user. Gracefully shutting down...")
    finally:
        # 11. Cleanup and close streams
        logger.close()

if __name__ == "__main__":
    main()
