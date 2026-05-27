import os
import time
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np

from src.utils.logger import ExperimentLogger
from src.utils.metrics import compute_metrics
from src.models.model_factory import freeze_backbone, unfreeze_all_parameters, get_parameter_count
from src.training.schedulers import get_warmup_cosine_scheduler

class Trainer:
    """
    Core PyTorch training orchestrator.
    Handles:
        - Two-phase training protocol (Head-only -> Full fine-tuning)
        - Dynamic optimizer/scheduler re-instantiation across phase transitions
        - Mixed Precision (AMP) using GradScaler for optimized T4 GPU performance
        - Value-based Early Stopping and Checkpoint preservation
        - Epoch-level validation and structured metrics tracking
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        logger: ExperimentLogger,
        device: torch.device,
        checkpoint_dir: str,
        early_stopping_patience: int = 7,
        grad_clip_norm: float = 1.0,
        amp: bool = True
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.logger = logger
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.early_stopping_patience = early_stopping_patience
        self.grad_clip_norm = grad_clip_norm
        self.amp = amp
        
        # AMP scaler initialization
        self.scaler = GradScaler(device='cuda') if amp and device.type == 'cuda' else None
        
        # Checkpoint directory setup
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # State variables
        self.best_val_acc = 0.0
        self.early_stopping_counter = 0
        self.optimizer = None
        self.scheduler = None
        
    def train_one_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Runs a single training epoch over the training loader."""
        self.model.train()
        epoch_loss = 0.0
        all_y_true = []
        all_y_pred = []
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with AMP if enabled
            if self.scaler is not None:
                with autocast(device_type='cuda'):
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, labels)
                
                # Backward pass and scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping (unscales gradients first)
                if self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                
                if self.grad_clip_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    
                self.optimizer.step()
                
            # Accumulate metrics
            epoch_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            
            all_y_true.append(labels.detach().cpu().numpy())
            all_y_pred.append(preds.detach().cpu().numpy())
            
        epoch_loss /= len(self.train_loader.dataset)
        
        # Calculate metrics
        y_true = np.concatenate(all_y_true)
        y_pred = np.concatenate(all_y_pred)
        metrics = compute_metrics(y_true, y_pred)
        
        # Prefix keys to differentiate train from val
        train_metrics = {f"Train_{k}": v for k, v in metrics.items()}
        train_metrics["Train_Loss"] = epoch_loss
        
        return epoch_loss, train_metrics
        
    @torch.no_grad()
    def validate(self) -> Tuple[float, Dict[str, float]]:
        """Runs a single validation loop over the validation loader."""
        self.model.eval()
        epoch_loss = 0.0
        all_y_true = []
        all_y_pred = []
        
        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            if self.scaler is not None:
                with autocast(device_type='cuda'):
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                
            epoch_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            
            all_y_true.append(labels.cpu().numpy())
            all_y_pred.append(preds.cpu().numpy())
            
        epoch_loss /= len(self.val_loader.dataset)
        
        # Calculate metrics
        y_true = np.concatenate(all_y_true)
        y_pred = np.concatenate(all_y_pred)
        metrics = compute_metrics(y_true, y_pred)
        
        # Prefix keys
        val_metrics = {f"Val_{k}": v for k, v in metrics.items()}
        val_metrics["Val_Loss"] = epoch_loss
        
        return epoch_loss, val_metrics
        
    def fit(
        self,
        phase1_epochs: int,
        phase2_epochs: int,
        phase1_lr: float,
        phase2_lr: float,
        weight_decay: float = 0.01,
        warmup_epochs: int = 5
    ) -> Dict[str, float]:
        """
        Executes the two-phase transfer learning routine.
        
        Args:
            phase1_epochs (int): Number of epochs for head-only calibration.
            phase2_epochs (int): Number of epochs for global model fine-tuning.
            phase1_lr (float): Initial learning rate for Phase 1.
            phase2_lr (float): Initial learning rate for Phase 2.
            weight_decay (float): Regularization penalty. Defaults to 0.01.
            warmup_epochs (int): LR warmup epochs. Defaults to 5.
            
        Returns:
            Dict[str, float]: Peak metrics achieved during the experiment.
        """
        peak_metrics = {}
        
        # ==========================================
        # PHASE 1: HEAD-ONLY TRAINING
        # ==========================================
        if phase1_epochs > 0:
            self.logger.info("=" * 60)
            self.logger.info("PHASE 1: HEAD-ONLY FEATURE EXTRACTION")
            self.logger.info("=" * 60)
            
            # Freeze the backbone
            freeze_backbone(self.model)
            
            # Extract trainable head parameters
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(
                trainable_params, 
                lr=phase1_lr, 
                weight_decay=weight_decay
            )
            
            self.scheduler = get_warmup_cosine_scheduler(
                self.optimizer,
                warmup_epochs=min(warmup_epochs, phase1_epochs // 2),
                total_epochs=phase1_epochs
            )
            
            total, trainable = get_parameter_count(self.model)
            self.logger.info(f"Phase 1 active parameters: {trainable:,} (of {total:,})")
            
            self._run_phase(
                num_epochs=phase1_epochs, 
                start_epoch=1, 
                phase_name="Phase1"
            )
            
        # ==========================================
        # PHASE 2: GLOBAL backbone FINE-TUNING
        # ==========================================
        if phase2_epochs > 0:
            self.logger.info("\n" + "=" * 60)
            self.logger.info("PHASE 2: GLOBAL BACKBONE FINE-TUNING")
            self.logger.info("=" * 60)
            
            # Unfreeze everything
            unfreeze_all_parameters(self.model)
            
            # Set up Phase 2 optimizer with all parameters
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=phase2_lr, 
                weight_decay=weight_decay
            )
            
            self.scheduler = get_warmup_cosine_scheduler(
                self.optimizer,
                warmup_epochs=warmup_epochs,
                total_epochs=phase2_epochs
            )
            
            total, trainable = get_parameter_count(self.model)
            self.logger.info(f"Phase 2 active parameters: {trainable:,} (of {total:,})")
            
            # Reset validation tracking for Phase 2 to allow complete adaptation
            self.best_val_acc = 0.0
            self.early_stopping_counter = 0
            
            stop_epoch = self._run_phase(
                num_epochs=phase2_epochs, 
                start_epoch=phase1_epochs + 1, 
                phase_name="Phase2"
            )
            
            self.logger.info(f"Phase 2 completed/stopped at epoch {stop_epoch}")
            
        # Gather final peak metrics for hyperparameter log
        checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        if os.path.exists(checkpoint_path):
            try:
                # Load peak metrics recorded in checkpoint
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                peak_metrics = {
                    "Peak_Val_Accuracy": checkpoint.get("accuracy", 0.0),
                    "Peak_Val_F1_Macro": checkpoint.get("f1_macro", 0.0)
                }
            except Exception:
                peak_metrics = {"Peak_Val_Accuracy": self.best_val_acc}
        else:
            peak_metrics = {"Peak_Val_Accuracy": self.best_val_acc}
            
        return peak_metrics
        
    def _run_phase(self, num_epochs: int, start_epoch: int, phase_name: str) -> int:
        """Helper to run a set of epochs for a training phase."""
        for local_epoch in range(num_epochs):
            epoch = start_epoch + local_epoch
            start_time = time.time()
            
            # Train and validate
            train_loss, train_metrics = self.train_one_epoch()
            val_loss, val_metrics = self.validate()
            
            epoch_duration = time.time() - start_time
            curr_lr = self.optimizer.param_groups[0]['lr']
            
            # Merge metrics and log
            combined_metrics = {}
            combined_metrics.update(train_metrics)
            combined_metrics.update(val_metrics)
            
            self.logger.log_metrics(
                epoch=epoch,
                metrics=combined_metrics,
                lr=curr_lr,
                elapsed_time=epoch_duration
            )
            
            # Scheduler Step
            self.scheduler.step()
            
            # Early Stopping and Checkpointing Check
            val_acc = val_metrics["Val_Accuracy"]
            val_f1_macro = val_metrics["Val_F1_Macro"]
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.early_stopping_counter = 0
                self.logger.info(f"[*] New best validation accuracy: {val_acc:.4f}! Saving checkpoint...")
                self._save_checkpoint(epoch, val_acc, val_f1_macro, filename="best_model.pth")
            else:
                self.early_stopping_counter += 1
                
            # Save periodic checkpoint
            if epoch % 5 == 0:
                self._save_checkpoint(epoch, val_acc, val_f1_macro, filename=f"checkpoint_epoch_{epoch}.pth")
                
            # Check for early stopping (patience)
            if self.early_stopping_counter >= self.early_stopping_patience:
                self.logger.warning(
                    f"[!] Early stopping triggered at epoch {epoch}. "
                    f"Accuracy has not improved for {self.early_stopping_patience} epochs."
                )
                return epoch
                
        return start_epoch + num_epochs - 1

    def _save_checkpoint(self, epoch: int, accuracy: float, f1_macro: float, filename: str):
        """Saves PyTorch state dict checkpoint with essential metadata."""
        save_path = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint_dict = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "amp": self.amp
        }
        
        try:
            torch.save(checkpoint_dict, save_path)
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Loads a saved checkpoint and returns the last completed epoch."""
        self.logger.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        if self.optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
        epoch = checkpoint["epoch"]
        self.best_val_acc = checkpoint.get("accuracy", 0.0)
        self.logger.info(f"Successfully loaded checkpoint from epoch {epoch} with Val Acc: {self.best_val_acc:.4f}")
        return epoch
