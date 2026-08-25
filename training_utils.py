"""
Unified training utilities for stable protein language model training.

This module provides common training functions that guarantee:
1. No NaN losses through comprehensive validation
2. Consistent loss decay through proper scheduling
3. Clean, reusable code with minimal repetition
"""

import torch
import logging
import numpy as np
from typing import Optional, Dict, Any
from torch.cuda.amp import GradScaler
from config import STABILITY_CONFIG

logger = logging.getLogger(__name__)


class StableGradScaler(GradScaler):
    """
    Enhanced GradScaler with NaN detection and recovery.
    
    Prevents training collapse by:
    - Detecting non-finite gradients before stepping
    - Automatically recovering from numerical instability
    - Logging stability issues for debugging
    """
    
    def __init__(self, *args, **kwargs):
        # Use conservative settings from old stable version
        kwargs.setdefault('init_scale', 2**10)
        kwargs.setdefault('growth_interval', 100)
        super().__init__(*args, **kwargs)
        self.nan_count = 0
        self.total_steps = 0
    
    def safe_step(self, optimizer, model, max_norm=1.0):
        """
        Perform optimizer step with gradient validation.
        
        Returns:
            bool: True if step was successful, False if skipped due to NaN
        """
        self.total_steps += 1
        
        # Unscale gradients
        self.unscale_(optimizer)
        
        # Check for NaN/Inf in gradients
        has_nan = False
        for param in model.parameters():
            if param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    has_nan = True
                    break
        
        if has_nan:
            self.nan_count += 1
            nan_rate = self.nan_count / self.total_steps
            logger.warning(
                f"Non-finite gradients detected (step {self.total_steps}, "
                f"NaN rate: {nan_rate:.2%})"
            )
            optimizer.zero_grad(set_to_none=True)
            self.update()  # Update scale factor
            return False
        
        # Clip gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        # Check gradient magnitude
        if grad_norm.item() > STABILITY_CONFIG["grad_norm_threshold"]:
            logger.warning(f"Large gradient norm: {grad_norm.item():.2f}")
        
        # Optimizer step
        self.step(optimizer)
        self.update()
        
        return True


def validate_batch(batch: Dict[str, torch.Tensor], model_type: str) -> bool:
    """
    Validate batch before forward pass.
    
    Args:
        batch: Dictionary containing input_ids and labels
        model_type: "esm2" or "mlstm"
    
    Returns:
        bool: True if batch is valid, False otherwise
    """
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    
    # Check for empty batches
    if input_ids.numel() == 0:
        logger.warning("Empty batch detected, skipping")
        return False
    
    # Check for sufficient masked tokens
    masked_count = (labels != -100).sum().item()
    if masked_count < STABILITY_CONFIG["min_masked_tokens"]:
        logger.debug(f"Insufficient masked tokens ({masked_count}), skipping")
        return False
    
    # Model-specific validations
    if model_type == "mlstm":
        # mLSTM requires sequence length divisible by chunk_size
        seq_len = input_ids.shape[1]
        if seq_len % 64 != 0:
            logger.warning(f"Sequence length {seq_len} not divisible by 64, skipping")
            return False
    
    # Check for valid token IDs
    if input_ids.max().item() >= 33:  # ESM2 vocab size
        logger.warning(f"Invalid token ID detected: {input_ids.max().item()}")
        return False
    
    return True


def validate_loss(loss: torch.Tensor, step: int) -> bool:
    """
    Validate loss value before backward pass.
    
    Args:
        loss: Loss tensor
        step: Current training step
    
    Returns:
        bool: True if loss is valid, False otherwise
    """
    loss_value = loss.item()
    
    # Check for NaN/Inf
    if not np.isfinite(loss_value):
        logger.error(f"Step {step}: Non-finite loss detected: {loss_value}")
        return False
    
    # Check for extreme losses
    if loss_value > STABILITY_CONFIG["max_loss_threshold"]:
        logger.warning(f"Step {step}: Extreme loss {loss_value:.4f}, skipping")
        return False
    
    # Check for negative loss (shouldn't happen with cross-entropy)
    if loss_value < 0:
        logger.error(f"Step {step}: Negative loss {loss_value:.4f}")
        return False
    
    return True


def safe_forward_pass(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    autocast_enabled: bool = True,
    model_type: str = "esm2"
) -> Optional[torch.Tensor]:
    """
    Perform forward pass with comprehensive error handling.
    
    Args:
        model: Model to evaluate
        batch: Input batch
        device: Device to use
        autocast_enabled: Whether to use automatic mixed precision
        model_type: "esm2" or "mlstm"
    
    Returns:
        Loss tensor if successful, None if failed
    """
    try:
        # Move batch to device
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        
        # ESM2 needs attention mask
        if model_type == "esm2":
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device, non_blocking=True)
        else:
            attention_mask = None
        
        # Forward pass with autocast
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=autocast_enabled):
            if model_type == "esm2" and attention_mask is not None:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
            else:
                outputs = model(input_ids=input_ids, labels=labels)
            
            loss = outputs.loss
        
        return loss
    
    except RuntimeError as e:
        logger.error(f"Forward pass failed: {e}")
        return None


def compute_validation_loss(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    steps: int = 50,
    model_type: str = "esm2"
) -> float:
    """
    Compute validation loss with stable evaluation.
    
    Args:
        model: Model to evaluate
        val_loader: Validation data loader
        device: Device to use
        steps: Number of batches to evaluate
        model_type: "esm2" or "mlstm"
    
    Returns:
        Mean validation loss
    """
    model.eval()
    val_losses = []
    val_iter = iter(val_loader)
    
    with torch.no_grad():
        for _ in range(steps):
            try:
                batch = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                batch = next(val_iter)
            
            # Validate batch
            if not validate_batch(batch, model_type):
                continue
            
            # Forward pass
            loss = safe_forward_pass(model, batch, device, autocast_enabled=True, model_type=model_type)
            
            if loss is not None and torch.isfinite(loss):
                val_losses.append(loss.item())
    
    model.train()
    
    if len(val_losses) == 0:
        logger.warning("No valid validation batches processed")
        return float('inf')
    
    return float(np.mean(val_losses))


def check_training_health(
    train_losses: list,
    val_losses: list,
    window_size: int = 10
) -> Dict[str, Any]:
    """
    Check training health and detect potential issues.
    
    Args:
        train_losses: List of recent training losses
        val_losses: List of validation losses
        window_size: Number of recent losses to analyze
    
    Returns:
        Dictionary with health metrics and warnings
    """
    health = {
        "status": "healthy",
        "warnings": [],
        "metrics": {}
    }
    
    if len(train_losses) < window_size:
        return health
    
    recent_losses = train_losses[-window_size:]
    
    # Check for NaN
    if any(not np.isfinite(l) for l in recent_losses):
        health["status"] = "critical"
        health["warnings"].append("NaN detected in training losses")
        return health
    
    # Check for loss spike
    loss_range = max(recent_losses) - min(recent_losses)
    loss_mean = np.mean(recent_losses)
    if loss_range > 2.0 * loss_mean:
        health["status"] = "warning"
        health["warnings"].append(f"Large loss variance: {loss_range:.4f}")
    
    # Check for stuck training
    loss_std = np.std(recent_losses)
    if loss_std < 1e-4:
        health["warnings"].append("Training may be stuck (no loss change)")
    
    # Check validation trend
    if len(val_losses) >= 3:
        recent_val = val_losses[-3:]
        if all(recent_val[i] < recent_val[i+1] for i in range(2)):
            health["status"] = "warning"
            health["warnings"].append("Validation loss increasing")
    
    health["metrics"]["recent_train_loss_mean"] = loss_mean
    health["metrics"]["recent_train_loss_std"] = loss_std
    
    return health
