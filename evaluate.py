"""
Post-training evaluation module for cramming-style trained models.

This module performs comprehensive validation AFTER training completes,
following the cramming methodology where validation overhead is minimized
during training.

Features:
- Load trained models from checkpoints
- Comprehensive validation on val set
- Perplexity computation
- Results saved for plotting
- Updates training_data.pkl with validation metrics

Usage:
    python evaluate.py esm2
    python evaluate.py mlstm

Or programmatically:
    from evaluate import evaluate_model
    results = evaluate_model("esm2", checkpoint_path="ESM2_cramming_final.pt")
"""

import logging
import pickle
import torch
import math
import sys
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict, Any

from config import DEVICE, ESM2_CONFIG, MLSTM_CONFIG, BASE_TRAINING_CONFIG
from data_setup import make_streaming_loaders
from training_utils import validate_batch, safe_forward_pass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model_checkpoint(
        model: torch.nn.Module,
        checkpoint_path: str,
        device: torch.device = DEVICE
) -> torch.nn.Module:
    """
    Load model from checkpoint.

    Args:
        model: Model instance (architecture)
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    logger.info(f"Loading checkpoint: {checkpoint_path}")

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("✓ Model state loaded")
    else:
        # Assume checkpoint is just the state dict
        model.load_state_dict(checkpoint)
        logger.info("✓ Model state loaded (legacy format)")

    model.to(device)
    model.eval()

    logger.info(f"✓ Model loaded and moved to {device}\n")
    return model


def evaluate_model(
        model_type: str,
        checkpoint_path: Optional[str] = None,
        val_steps: int = None,
        device: torch.device = DEVICE
) -> Dict[str, Any]:
    """
    Evaluate a trained model on validation set.

    Args:
        model_type: "esm2" or "mlstm"
        checkpoint_path: Path to checkpoint (default: {MODEL}_cramming_final.pt)
        val_steps: Number of validation steps (default: from config)
        device: Device to use

    Returns:
        Dictionary with evaluation results
    """

    # Select configuration
    config = ESM2_CONFIG if model_type == "esm2" else MLSTM_CONFIG
    model_name = "ESM2" if model_type == "esm2" else "mLSTM"

    if val_steps is None:
        val_steps = config.get("val_steps", 100)

    logger.info("="*70)
    logger.info(f"EVALUATING {model_name} (POST-TRAINING)")
    logger.info("="*70)
    logger.info(f"Model type: {model_type}")
    logger.info(f"Validation steps: {val_steps}")
    logger.info(f"Device: {device}")
    logger.info("="*70 + "\n")

    # ========================================================================
    # LOAD MODEL
    # ========================================================================
    if checkpoint_path is None:
        checkpoint_path = f"training_data/{model_name}_cramming_final.pt"

    logger.info("Loading model architecture...")
    if model_type == "esm2":
        from esm2_model import model
    else:
        from mlstm_modelNew import model

    # Load checkpoint
    model = load_model_checkpoint(model, checkpoint_path, device)

    # ========================================================================
    # CREATE VALIDATION LOADER
    # ========================================================================
    logger.info("Creating validation data loader...")
    _, val_loader = make_streaming_loaders(
        train_fasta=str(config["train_fasta"]) if "train_fasta" in config else str(BASE_TRAINING_CONFIG["train_fasta"]),
        val_fasta=str(config["val_fasta"]) if "val_fasta" in config else str(BASE_TRAINING_CONFIG["val_fasta"]),
        model=model_type,
        max_tokens_per_batch=config["max_tokens_per_batch"],
        mask_prob=config["mask_prob"],
        max_len=config["max_len"],
        num_workers=config.get("num_workers", 2),
    )
    logger.info("✓ Validation loader created\n")

    # ========================================================================
    # RUN VALIDATION
    # ========================================================================
    logger.info(f"Running validation ({val_steps} steps)...")
    logger.info("-"*70)

    val_losses = []
    val_iter = iter(val_loader)

    with torch.no_grad():
        pbar = tqdm(range(val_steps), desc="Validating", unit="batch")

        for step_idx in pbar:
            try:
                batch = next(val_iter)
            except StopIteration:
                logger.warning(f"Validation data exhausted at step {step_idx}")
                break

            # Validate batch
            if not validate_batch(batch, model_type):
                continue

            # Forward pass
            loss = safe_forward_pass(
                model, batch, device,
                autocast_enabled=True,
                model_type=model_type
            )

            if loss is not None and torch.isfinite(loss):
                val_losses.append(loss.item())

                # Update progress bar
                current_mean = sum(val_losses) / len(val_losses)
                current_perplexity = math.exp(current_mean)
                pbar.set_postfix({
                    'loss': f'{current_mean:.4f}',
                    'ppl': f'{current_perplexity:.2f}'
                })

    logger.info("\n" + "-"*70)

    # ========================================================================
    # COMPUTE METRICS
    # ========================================================================
    if len(val_losses) == 0:
        logger.error("❌ No valid validation batches processed!")
        return None

    import numpy as np

    mean_val_loss = float(np.mean(val_losses))
    val_perplexity = math.exp(mean_val_loss)
    std_val_loss = float(np.std(val_losses))
    min_val_loss = float(np.min(val_losses))
    max_val_loss = float(np.max(val_losses))

    results = {
        "model_type": model_type,
        "model_name": model_name,
        "checkpoint_path": checkpoint_path,
        "val_loss": mean_val_loss,
        "val_perplexity": val_perplexity,
        "val_loss_std": std_val_loss,
        "val_loss_min": min_val_loss,
        "val_loss_max": max_val_loss,
        "num_val_batches": len(val_losses),
        "all_val_losses": val_losses,
    }

    # ========================================================================
    # PRINT RESULTS
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info(f"{model_name} VALIDATION RESULTS")
    logger.info("="*70)
    logger.info(f"Validation Loss:       {mean_val_loss:.4f} (±{std_val_loss:.4f})")
    logger.info(f"Validation Perplexity: {val_perplexity:.2f}")
    logger.info(f"Loss Range:            [{min_val_loss:.4f}, {max_val_loss:.4f}]")
    logger.info(f"Batches Evaluated:     {len(val_losses)}")
    logger.info("="*70 + "\n")

    # ========================================================================
    # UPDATE TRAINING DATA WITH VALIDATION RESULTS
    # ========================================================================
    training_data_path = f"training_data/{model_type}_cramming_training_data.pkl"

    if Path(training_data_path).exists():
        logger.info(f"Updating training data with validation results...")

        with open(training_data_path, "rb") as f:
            training_data = pickle.load(f)

        # Add validation results
        training_data["final_val_loss"] = mean_val_loss
        training_data["final_val_perplexity"] = val_perplexity
        training_data["val_losses"] = val_losses
        training_data["val_loss_std"] = std_val_loss

        # Create validation steps (just mark at end of training)
        final_step = training_data.get("total_updates", training_data["train_steps"][-1])
        training_data["val_steps"] = [final_step]
        training_data["val_perplexities"] = [val_perplexity]

        # Save updated data
        with open(training_data_path, "wb") as f:
            pickle.dump(training_data, f)

        logger.info(f"✓ Updated: {training_data_path}\n")
    else:
        logger.warning(f"⚠️  Training data not found: {training_data_path}")
        logger.info("   Validation results saved separately\n")

    # ========================================================================
    # SAVE EVALUATION RESULTS
    # ========================================================================
    eval_results_path = f"{model_type}_cramming_eval_results.pkl"
    with open(eval_results_path, "wb") as f:
        pickle.dump(results, f)

    logger.info(f"💾 Evaluation results saved: {eval_results_path}\n")

    # ========================================================================
    # COMPARISON CONTEXT
    # ========================================================================
    logger.info("📊 COMPARISON CONTEXT")
    logger.info("-"*70)
    logger.info(f"Your {model_name} perplexity: {val_perplexity:.2f}")
    logger.info(f"Cramming paper target (67M): 13.72 perplexity")

    # Try to load baseline if available
    baseline_path = "esm2_8m_pretrained_baseline.pkl"
    if Path(baseline_path).exists():
        with open(baseline_path, "rb") as f:
            baseline = pickle.load(f)
        baseline_ppl = baseline["validation_perplexity"]
        logger.info(f"ESM2-8M pretrained baseline: {baseline_ppl:.2f}")

        diff = val_perplexity - baseline_ppl
        pct = (val_perplexity / baseline_ppl - 1) * 100
        logger.info(f"Difference from baseline:    {diff:+.2f} ({pct:+.1f}%)")

    logger.info("="*70 + "\n")

    return results


def evaluate_both_models(val_steps: int = None) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate both ESM2 and mLSTM models.

    Args:
        val_steps: Number of validation steps for each model

    Returns:
        Dictionary with results for both models
    """
    results = {}

    for model_type in ["esm2", "mlstm"]:
        try:
            logger.info(f"\n{'='*70}")
            logger.info(f"EVALUATING {model_type.upper()}")
            logger.info(f"{'='*70}\n")

            result = evaluate_model(model_type, val_steps=val_steps)
            results[model_type] = result

        except Exception as e:
            logger.error(f"❌ Failed to evaluate {model_type}: {e}", exc_info=True)
            results[model_type] = None

    # Print summary
    logger.info("\n" + "="*70)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*70)

    for model_type, result in results.items():
        if result:
            logger.info(f"{model_type.upper():6s}: ✅ Perplexity = {result['val_perplexity']:.2f}")
        else:
            logger.info(f"{model_type.upper():6s}: ❌ Evaluation failed")

    logger.info("="*70 + "\n")

    return results


def main():
    """Main evaluation entry point."""

    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python evaluate.py esm2          # Evaluate ESM2")
        print("  python evaluate.py mlstm         # Evaluate mLSTM")
        print("  python evaluate.py both          # Evaluate both models")
        print("\nOptional: specify validation steps")
        print("  python evaluate.py esm2 200      # 200 validation steps\n")
        sys.exit(1)

    model_arg = sys.argv[1].lower()
    val_steps = int(sys.argv[2]) if len(sys.argv) > 2 else None

    if model_arg == "both":
        results = evaluate_both_models(val_steps=val_steps)
        success = all(r is not None for r in results.values())
    elif model_arg in ["esm2", "mlstm"]:
        result = evaluate_model(model_arg, val_steps=val_steps)
        success = result is not None
    else:
        print(f"❌ Invalid model type: {model_arg}")
        print("   Use 'esm2', 'mlstm', or 'both'")
        sys.exit(1)

    if success:
        logger.info("✅ Evaluation complete!")
        logger.info("\nNext step: python comparison_plots.py")
        return 0
    else:
        logger.error("❌ Evaluation failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
