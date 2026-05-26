import os
import sys
import logging
import csv
import time
from typing import Dict, Any, Optional
from torch.utils.tensorboard import SummaryWriter

class ExperimentLogger:
    """
    Unified logging framework for research-grade tracking.
    Supports:
        - Console logging with structured, timestamped formatting
        - CSV file logging for direct tabular analysis
        - TensorBoard SummaryWriter for dynamic visualization of training curves
    """
    def __init__(self, log_dir: str, experiment_name: str, use_tensorboard: bool = True):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.exp_path = os.path.join(log_dir, experiment_name)
        os.makedirs(self.exp_path, exist_ok=True)
        
        # 1. Setup Standard Logging (Console + File)
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers if active
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
            
        # Formatter
        formatter = logging.Formatter(
            fmt='[%(asctime)s] [%(levelname)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File Handler
        file_log_path = os.path.join(self.exp_path, "experiment.log")
        file_handler = logging.FileHandler(file_log_path, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        self.info(f"Initialized ExperimentLogger for '{experiment_name}'")
        self.info(f"Logs will be saved to: {self.exp_path}")
        
        # 2. Setup CSV Logger
        self.csv_path = os.path.join(self.exp_path, "training_logs.csv")
        self.csv_initialized = False
        
        # 3. Setup TensorBoard
        self.tb_writer = None
        if use_tensorboard:
            tb_log_dir = os.path.join(log_dir, "tensorboard", experiment_name)
            self.tb_writer = SummaryWriter(log_dir=tb_log_dir)
            self.info(f"TensorBoard event files will be written to: {tb_log_dir}")
            
    def info(self, msg: str):
        self.logger.info(msg)
        
    def warning(self, msg: str):
        self.logger.warning(msg)
        
    def error(self, msg: str):
        self.logger.error(msg)
        
    def log_metrics(self, epoch: int, metrics: Dict[str, float], lr: float, elapsed_time: float):
        """
        Logs a dictionary of metrics for a given epoch to Console, CSV, and TensorBoard.
        
        Args:
            epoch (int): The current epoch number.
            metrics (Dict[str, float]): Dictionary containing metric names and their values.
            lr (float): The current learning rate.
            elapsed_time (float): Time taken for the epoch (in seconds).
        """
        # Print to console
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.info(f"Epoch {epoch:03d} - LR: {lr:.6e} - {metric_str} - Duration: {elapsed_time:.1f}s")
        
        # Log to TensorBoard
        if self.tb_writer:
            self.tb_writer.add_scalar("Learning_Rate", lr, epoch)
            self.tb_writer.add_scalar("Epoch_Duration_Seconds", elapsed_time, epoch)
            for k, v in metrics.items():
                self.tb_writer.add_scalar(k, v, epoch)
                
        # Log to CSV
        csv_data = {"epoch": epoch, "learning_rate": lr, "duration_sec": round(elapsed_time, 2)}
        csv_data.update({k.lower().replace(" ", "_"): v for k, v in metrics.items()})
        
        fieldnames = list(csv_data.keys())
        
        # Open CSV file and write
        file_exists = os.path.exists(self.csv_path)
        with open(self.csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists or not self.csv_initialized:
                writer.writeheader()
                self.csv_initialized = True
            writer.writerow(csv_data)
            
    def log_hyperparams(self, config: Dict[str, Any], final_metrics: Dict[str, float]):
        """
        Logs hyperparameter configurations alongside their final metrics to TensorBoard.
        
        Args:
            config (Dict[str, Any]): Flattened configuration dictionary.
            final_metrics (Dict[str, float]): The peak metrics achieved in the experiment.
        """
        if self.tb_writer:
            # Flatten the nested config for TensorBoard hparams tab
            flat_config = self._flatten_dict(config)
            
            # Convert values to strings if they are lists or dicts
            flat_config_str = {}
            for k, v in flat_config.items():
                if isinstance(v, (list, dict, tuple)):
                    flat_config_str[k] = str(v)
                else:
                    flat_config_str[k] = v
                    
            self.tb_writer.add_hparams(
                hparam_dict=flat_config_str,
                metric_dict=final_metrics
            )
            self.info("Saved hyperparameter configuration to TensorBoard.")
            
    def close(self):
        """Closes all loggers and TensorBoard SummaryWriter."""
        if self.tb_writer:
            self.tb_writer.flush()
            self.tb_writer.close()
            self.info("Closed TensorBoard writer.")
        self.info("Experiment logging finished.")
        
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '/') -> Dict[str, Any]:
        """Utility to flatten a nested dictionary for TensorBoard hparams logging."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
