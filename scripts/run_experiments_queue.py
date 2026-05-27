import os
import sys
import subprocess
import glob
import yaml
import time
from typing import List

def get_latest_checkpoint(checkpoint_root: str, model_name: str) -> str:
    """
    Searches the checkpoint folder for the most recently created timestamped directory
    and returns the path to its 'best_model.pth' checkpoint.
    """
    # Standard checkpoint directory resolved by train.py
    model_dir = os.path.join(checkpoint_root, "checkpoints", model_name)
    
    if not os.path.exists(model_dir):
        # Fallback to search locally if drive prefix wasn't applied
        model_dir = os.path.join("checkpoints", model_name)
        
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model checkpoint directory not found at: {model_dir}")
        
    # Get all timestamped subdirectories
    subdirs = [os.path.join(model_dir, d) for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    
    if not subdirs:
        raise FileNotFoundError(f"No timestamped experiment directories found inside: {model_dir}")
        
    # Sort directories by creation time (most recent last)
    subdirs.sort(key=os.path.getmtime)
    latest_dir = subdirs[-1]
    
    checkpoint_path = os.path.join(latest_dir, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Peak checkpoint 'best_model.pth' not found in: {latest_dir}")
        
    return checkpoint_path

def run_pipeline(configs: List[str], drive_prefix: str = ""):
    """
    Executes training and evaluation in sequence for a list of YAML configurations.
    
    Args:
        configs (List[str]): List of paths to configuration YAML files.
        drive_prefix (str): Optional path to Google Drive root folder.
    """
    total_configs = len(configs)
    print("=" * 70)
    print(f"LAUNCHING MULTI-MODEL EXPERIMENT PIPELINE QUEUE ({total_configs} MODELS)")
    print("=" * 70)
    
    for idx, config_path in enumerate(configs):
        start_time = time.time()
        print("\n" + "=" * 60)
        print(f"[{idx+1}/{total_configs}] PROCESSING bluePRINT: {config_path}")
        print("=" * 60)
        
        if not os.path.exists(config_path):
            print(f"[ERROR] Blueprint configuration not found: {config_path}. Skipping.")
            continue
            
        # Parse model name from YAML to identify checkpoints folder
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
            model_name = config_data["model"]["name"]
        except Exception as e:
            print(f"[ERROR] Failed to parse model name from {config_path}: {e}. Skipping.")
            continue
            
        # 1. RUN TRAINING RUNNER
        print(f"\n[RUNNER] Step 1/2: Initiating Training for '{model_name.upper()}'...")
        train_cmd = [sys.executable, "src/training/train.py", "--config", config_path]
        print(f"[RUNNER] Executing: {' '.join(train_cmd)}")
        
        train_res = subprocess.run(train_cmd)
        
        if train_res.returncode != 0:
            print(f"\n[ERROR] Training failed for '{model_name}' (exit code: {train_res.returncode}). Skipping evaluation.")
            continue
            
        print(f"[RUNNER] Training successfully completed for '{model_name}'!")
        
        # 2. RUN EVALUATOR ON GENERATED CHECKPOINT
        print(f"\n[RUNNER] Step 2/2: Initiating Standalone Evaluation for '{model_name.upper()}'...")
        try:
            checkpoint_path = get_latest_checkpoint(drive_prefix, model_name)
            print(f"[RUNNER] Located latest best checkpoint: {checkpoint_path}")
            
            eval_cmd = [sys.executable, "src/evaluation/evaluator.py", "--config", config_path, "--checkpoint", checkpoint_path]
            print(f"[RUNNER] Executing: {' '.join(eval_cmd)}")
            
            eval_res = subprocess.run(eval_cmd)
            
            if eval_res.returncode != 0:
                print(f"[ERROR] Evaluation execution failed for '{model_name}' (exit code: {eval_res.returncode}).")
            else:
                print(f"[RUNNER] Evaluation metrics and paper assets successfully compiled for '{model_name}'!")
                
        except Exception as e:
            print(f"[ERROR] Exception during evaluator location or run: {e}")
            
        duration = (time.time() - start_time) / 60
        print(f"\n[STATUS] Finished pipeline for '{model_name}' in {duration:.2f} minutes.")
        print("=" * 60)
        
    print("\n" + "=" * 70)
    print("ALL EXPERIMENT QUEUES COMPLETED SUCCESSFULY!")
    print("=" * 70)

if __name__ == "__main__":
    # Resolve Drive output directory root when running in Colab
    drive_mounted_path = "/content/drive/MyDrive/dental_research"
    drive_prefix = drive_mounted_path if os.path.exists(drive_mounted_path) else ""
    
    # Ordered list of configurations to process
    remaining_configs = [
        "src/configs/densenet121.yaml",
        "src/configs/mobilenetv3_small.yaml",
        "src/configs/efficientnet_b2.yaml",
        "src/configs/efficientnet_b3.yaml",
        "src/configs/convnext_tiny.yaml",
        "src/configs/dinov2_small.yaml"
    ]
    
    run_pipeline(remaining_configs, drive_prefix)
