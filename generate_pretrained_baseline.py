"""
Generate ESM2-8M Pretrained Baseline for Comparison

This script evaluates the pretrained ESM2-8M model (from HuggingFace) on your
validation dataset to establish the baseline performance (green dashed line).

This baseline represents what a fully pretrained model achieves WITHOUT any
training from scratch. It's the target your trained models should approach.

Usage:
    python generate_pretrained_baseline.py

Expected runtime: 1-2 hours on V100-16GB
Output: Validation loss, perplexity, and saved results
"""

# ============================================================================
# CRITICAL: Import config FIRST before any torch/CUDA operations!
# This ensures CUDA_VISIBLE_DEVICES=1 is set before torch initializes CUDA
# ============================================================================
from config import DEVICE, STABILITY_CONFIG, VAL_FASTA, BASE_TRAINING_CONFIG

import os
import logging
import torch
import pickle
import math
from pathlib import Path
from tqdm import tqdm
from transformers import EsmForMaskedLM, AutoTokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_baseline_perplexity(
        model_name: str = "facebook/esm2_t6_8M_UR50D",
        val_fasta: str = None,  # Will use config default if None
        device: torch.device = None,
        num_batches: int = 50000,
        batch_size: int = 16,
        max_len: int = 512,
):
    """
    Evaluate pretrained ESM2-8M on validation set.

    Args:
        model_name: HuggingFace model identifier
        val_fasta: Path to validation FASTA file (uses config default if None)
        device: Device to use (uses config DEVICE if None)
        num_batches: Number of batches to evaluate (200 = ~1600 sequences)
        batch_size: Batch size for evaluation
        max_len: Maximum sequence length

    Returns:
        Dictionary with loss, perplexity, and other metrics
    """

    # Use config defaults if not specified
    if device is None:
        device = DEVICE
    if val_fasta is None:
        val_fasta = str(VAL_FASTA)

    # Verify file exists before proceeding
    if not Path(val_fasta).exists():
        # Try with .gz extension
        if Path(val_fasta + ".gz").exists():
            val_fasta = val_fasta + ".gz"
            logger.info(f"📁 Using gzipped file: {val_fasta}")
        else:
            logger.error(f"❌ Validation file not found: {val_fasta}")
            logger.error(f"   Also checked: {val_fasta}.gz")
            logger.error(f"   Please verify the file path and try again.")
            raise FileNotFoundError(f"Validation file not found: {val_fasta}")

    logger.info("="*70)
    logger.info("ESM2-8M PRETRAINED BASELINE EVALUATION")
    logger.info("="*70)
    logger.info(f"Model: {model_name}")
    logger.info(f"Device: {device}")
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    if torch.cuda.is_available():
        logger.info(f"Physical GPU: GPU {os.environ.get('CUDA_VISIBLE_DEVICES', '0')}")
        logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    logger.info(f"Validation file: {val_fasta}")
    logger.info(f"Evaluation batches: {num_batches} (x{batch_size} = {num_batches * batch_size} sequences)")
    logger.info("")

    # Load pretrained model and tokenizer
    logger.info("Loading pretrained ESM2-8M model from HuggingFace...")
    try:
        model = EsmForMaskedLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"✅ Model loaded successfully")
        logger.info(f"   Vocab size: {tokenizer.vocab_size}")
        logger.info(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise

    # Move to device and set eval mode
    model.to(device)
    model.eval()
    logger.info(f"✅ Model moved to {device} and set to eval mode")
    logger.info("")

    # Create data loader
    logger.info("Creating validation data loader...")
    from data_setup import make_streaming_loaders

    _, val_loader = make_streaming_loaders(
        train_fasta=val_fasta,  # Not used, but required
        val_fasta=val_fasta,
        tokenizer=tokenizer,
        model="esm2",
        max_tokens_per_batch=batch_size * max_len,
        mask_prob=0.25,
        max_len=max_len,
        num_workers=BASE_TRAINING_CONFIG.get('num_workers', 2),
    )

    logger.info("✅ Data loader created")
    logger.info("")

    # Evaluate on validation set
    logger.info("Starting evaluation...")
    logger.info("-"*70)

    val_losses = []
    val_iter = iter(val_loader)

    with torch.no_grad():
        pbar = tqdm(range(num_batches), desc="Evaluating", unit="batch")

        for batch_idx in pbar:
            try:
                batch = next(val_iter)
            except StopIteration:
                logger.warning(f"Data iterator exhausted at batch {batch_idx}")
                break

            # Move to device
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            # Skip batches with no masked tokens
            masked_count = (labels != -100).sum().item()
            if masked_count == 0:
                continue

            # Forward pass
            try:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss

                # Check for valid loss
                if torch.isfinite(loss):
                    val_losses.append(loss.item())

                    # Update progress bar
                    current_mean = sum(val_losses) / len(val_losses)
                    current_perplexity = math.exp(current_mean)
                    pbar.set_postfix({
                        'loss': f'{current_mean:.4f}',
                        'ppl': f'{current_perplexity:.2f}'
                    })

            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error(f"❌ CUDA OOM at batch {batch_idx}")
                    logger.error(f"   Try reducing batch_size or num_batches")
                    raise
                logger.warning(f"Batch {batch_idx} failed: {e}")
                continue

    logger.info("")
    logger.info("-"*70)

    # Calculate final metrics
    if len(val_losses) == 0:
        logger.error("❌ No valid validation batches processed!")
        return None

    mean_loss = sum(val_losses) / len(val_losses)
    perplexity = math.exp(mean_loss)

    # Additional statistics
    import numpy as np
    std_loss = np.std(val_losses)
    min_loss = min(val_losses)
    max_loss = max(val_losses)

    results = {
        "model_name": model_name,
        "validation_loss": mean_loss,
        "validation_perplexity": perplexity,
        "loss_std": std_loss,
        "loss_min": min_loss,
        "loss_max": max_loss,
        "num_batches": len(val_losses),
        "total_sequences": len(val_losses) * batch_size,
        "device": str(device),
        "gpu_used": os.environ.get('CUDA_VISIBLE_DEVICES', '0'),
    }

    # Print results
    logger.info("="*70)
    logger.info("BASELINE RESULTS")
    logger.info("="*70)
    logger.info(f"Model: {model_name}")
    logger.info(f"Validation Loss: {mean_loss:.4f} (±{std_loss:.4f})")
    logger.info(f"Validation Perplexity: {perplexity:.2f}")
    logger.info(f"Loss Range: [{min_loss:.4f}, {max_loss:.4f}]")
    logger.info(f"Batches evaluated: {len(val_losses)}")
    logger.info(f"Total sequences: {len(val_losses) * batch_size:,}")
    logger.info(f"GPU used: {os.environ.get('CUDA_VISIBLE_DEVICES', '0')}")
    logger.info("="*70)
    logger.info("")

    # Save results
    output_file = "training_data/esm2_8m_pretrained_baseline.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)

    logger.info(f"✅ Results saved to: {output_file}")
    logger.info("")

    # Print comparison context
    logger.info("📊 COMPARISON CONTEXT")
    logger.info("-"*70)
    logger.info(f"Your baseline (ESM2-8M pretrained): {perplexity:.2f} perplexity")
    logger.info(f"Cramming paper target (67M model): 13.72 perplexity")
    logger.info("")
    logger.info("This is your GREEN DASHED LINE - the target to approach!")
    logger.info("Your trained models should aim to get close to this perplexity.")
    logger.info("="*70)

    return results


def main():
    """Main function to run baseline evaluation."""

    # Verify GPU configuration
    logger.info("")
    logger.info("🔍 Verifying GPU configuration...")
    logger.info(f"   CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    logger.info(f"   Using device: {DEVICE}")
    if torch.cuda.is_available():
        logger.info(f"   Physical GPU: GPU {os.environ.get('CUDA_VISIBLE_DEVICES', '0')}")
        logger.info(f"   GPU Name: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"   GPU Memory: {mem_gb:.2f} GB")
        if mem_gb < 15:
            logger.warning(f"   ⚠️  GPU has only {mem_gb:.2f} GB - may need to reduce batch_size")
    logger.info("")

    # Check if baseline already exists
    baseline_file = "training_data/esm2_8m_pretrained_baseline.pkl"
    if Path(baseline_file).exists():
        logger.info("")
        logger.info("⚠️  Baseline file already exists!")
        logger.info(f"   File: {baseline_file}")

        # Load and display existing results
        with open(baseline_file, 'rb') as f:
            results = pickle.load(f)

        logger.info("")
        logger.info("Existing baseline results:")
        logger.info(f"  Validation Loss: {results['validation_loss']:.4f}")
        logger.info(f"  Validation Perplexity: {results['validation_perplexity']:.2f}")
        if 'gpu_used' in results:
            logger.info(f"  GPU used: {results['gpu_used']}")
        logger.info("")

        response = input("Do you want to re-run the baseline evaluation? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            logger.info("Using existing baseline results.")
            return results

        logger.info("")
        logger.info("Re-running baseline evaluation...")
        logger.info("")

    # Run baseline evaluation
    try:
        results = compute_baseline_perplexity()

        if results is not None:
            logger.info("")
            logger.info("✅ SUCCESS! Baseline evaluation complete.")
            logger.info("")
            logger.info("Next steps:")
            logger.info("1. Use this perplexity as your target (green line)")
            logger.info("2. Train your models (ESM2 and mLSTM from scratch)")
            logger.info("3. Compare their perplexity to this baseline")
            logger.info("4. Generate comparison plots")
            logger.info("")
        else:
            logger.error("❌ Baseline evaluation failed!")
            return None

    except KeyboardInterrupt:
        logger.info("")
        logger.info("⚠️  Evaluation interrupted by user")
        return None
    except Exception as e:
        logger.error(f"❌ Evaluation failed with error: {e}", exc_info=True)
        return None

    return results


if __name__ == "__main__":
    results = main()