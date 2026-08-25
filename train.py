"""
Unified training script for ESM2 and mLSTM models.

This script provides:
1. Guaranteed stability (no NaN losses)
2. Consistent loss decay through proper scheduling
3. Clean, readable code with no redundancy
4. Fair comparison (both models trained identically)
5. Publication-quality logging and plotting
6. Resume capability for recovering from failures
"""

import gc
import os
import logging
import pickle
import torch
import wandb
import math
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Literal

from resume_training import resume_training

from config import (
    DEVICE, ESM2_CONFIG, MLSTM_CONFIG, WANDB_PROJECT,
    BASE_TRAINING_CONFIG, STABILITY_CONFIG
)
from training_utils import (
    StableGradScaler, validate_batch, validate_loss,
    safe_forward_pass, compute_validation_loss, check_training_health
)
from data_utils import get_optimizer_and_scheduler, init_wandb_run
from data_setup import make_streaming_loaders

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_model(
        model: torch.nn.Module,
        model_type: Literal["esm2", "mlstm"],
        device: torch.device = DEVICE,
        resume_from: str = None, # Optional: explicit checkpoint path
        config: dict = None
) -> None:
    """
    Universal training function for both ESM2 and mLSTM.

    This function implements the stable training recipe from the old version
    while adding comprehensive monitoring and plotting capabilities.

    Args:
        model: Model to train (ESM2 or mLSTM)
        model_type: "esm2" or "mlstm"
        device: Device to use for training
        resume_from: Optional path to checkpoint to resume from
    """
    # Select appropriate configuration
    if config is None:
        config = ESM2_CONFIG if model_type == "esm2" else MLSTM_CONFIG

    experiment_name = config.get("experiment_name")
    if not experiment_name:
        # fallback if not provided
        experiment_name = f"{model_type}_lr{config['lr']:.0e}_w{config['warmup_steps']}".replace("e-0", "e-")


    # Now experiment_name is safe
    #experiment_name = config.get("experiment_name")
    #if not experiment_name:
    #    lr_val = config.get("lr")
    #    warmup = config.get("warmup_steps")
    #    if lr_val is None or warmup is None:
    #        raise KeyError("config must contain 'lr' and 'warmup_steps' to build experiment_name fallback.")
    #    experiment_name = f"{model_type}_lr{lr_val}_w{warmup}"


    model_name = "ESM2" if model_type == "esm2" else "mLSTM"

    logger.info(f"Starting {model_name} training with {config['total_updates']} updates")
    logger.info(f"Device: {device}")

    # Initialize wandb
    wandb_run = init_wandb_run(
        project=WANDB_PROJECT,
        model_name=model_name,
        run_type="training",
        config=config,
        tags=[model_name, "train", "stable"],
    )

    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader = make_streaming_loaders(
        train_fasta=str(config["train_fasta"]) if "train_fasta" in config else str(BASE_TRAINING_CONFIG["train_fasta"]),
        val_fasta=str(config["val_fasta"]) if "val_fasta" in config else str(BASE_TRAINING_CONFIG["val_fasta"]),
        model=model_type,
        max_tokens_per_batch=config["max_tokens_per_batch"],
        mask_prob=config["mask_prob"],
        max_len=config["max_len"],
        num_workers=config["num_workers"],
    )

    # Setup optimizer and scheduler
    optimizer, scheduler = get_optimizer_and_scheduler(
        model,
        total_steps=config["total_updates"],
        warmup_steps=config["warmup_steps"],
        lr=config["lr"],
    )

    # Setup mixed precision training with stability guarantees
    scaler = StableGradScaler(enabled=(device.type == 'cuda'))

    # ========================================================================
    # RESUME FROM CHECKPOINT (MODEL-SPECIFIC)
    # ========================================================================
    start_step = 0

    # Priority 1: Use explicit resume_from parameter if provided
    if resume_from and os.path.exists(resume_from):
        start_step, info = resume_training(resume_from, model, optimizer, scaler, scheduler)
        logger.info(f"✅ Resumed from {resume_from} at step {start_step}")

    # Priority 2: Check environment variable
    elif os.environ.get('RESUME_FROM'):
        resume_path = os.environ.get('RESUME_FROM')
        if os.path.exists(resume_path):
            start_step, info = resume_training(resume_path, model, optimizer, scaler, scheduler)
            logger.info(f"✅ Resumed from {resume_path} at step {start_step}")
        else:
            logger.warning(f"⚠️  RESUME_FROM set but file not found: {resume_path}")
            logger.info("Starting from scratch instead")

    # Priority 3: Auto-detect latest checkpoint for this model
    else:
        from resume_training import find_latest_checkpoint

        # Look for model-specific checkpoints
        checkpoint_prefix = f"{model_name}_step"
        latest_checkpoint = find_latest_checkpoint('checkpoints', prefix=checkpoint_prefix)

        if latest_checkpoint:
            logger.info(f"📁 Found latest {model_name} checkpoint: {latest_checkpoint}")

            # For automated runs, you might want to auto-resume
            # For manual runs, you might want to ask
            auto_resume = os.environ.get('AUTO_RESUME', 'false').lower() == 'true'

            if auto_resume:
                start_step, info = resume_training(latest_checkpoint, model, optimizer, scaler, scheduler)
                logger.info(f"✅ Auto-resumed from step {start_step}")
            else:
                # Interactive mode - ask user
                try:
                    response = input(f"Resume from this checkpoint? (y/n): ")
                    if response.lower() in ['y', 'yes']:
                        start_step, info = resume_training(latest_checkpoint, model, optimizer, scaler, scheduler)
                        logger.info(f"✅ Resumed from step {start_step}")
                except:
                    # If running non-interactively (e.g., in batch mode), skip prompt
                    logger.info("Non-interactive mode, starting from scratch")

    if start_step == 0:
        logger.info("🆕 Starting training from scratch")
    else:
        logger.info(f"🔄 Continuing training from step {start_step}")

    # ========================================================================
    # GRADIENT CHECKPOINTING (Memory vs Speed Trade-off)
    # ========================================================================
    # Gradient checkpointing saves memory but slows training by ~20-30%
    #
    # Enable if:  Getting OOM errors, want larger batches
    # Disable if: Speed is priority, memory is sufficient
    # ========================================================================

    USE_GRADIENT_CHECKPOINTING = False  # ← Set to True if OOM

    if USE_GRADIENT_CHECKPOINTING and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
            if hasattr(model, "config"):
                model.config.use_cache = False
            logger.info("✅ Gradient checkpointing enabled (saves memory, ~20% slower)")
        except Exception as e:
            logger.warning(f"Could not enable gradient checkpointing: {e}")
    else:
        logger.info("⚡ Gradient checkpointing disabled (maximum speed)")

    # Move model to device and set training mode
    model.to(device)
    model.train()

    # Training state
    train_iter = iter(train_loader)
    micro_step = 0
    update_step = start_step  # ← FIXED: Start from checkpoint, not 0!
    accum_loss = 0.0
    grad_accum = config["grad_accum_steps"]

    # Tracking lists for plotting
    train_losses = []
    train_steps = []
    val_losses = []
    val_steps = []

    # Progress bar
    pbar = tqdm(
        desc=f"Training {model_name}",
        total=config["total_updates"],
        initial=start_step,  # ← FIXED: Show correct progress when resuming!
        dynamic_ncols=True,
        unit="update",
    )

    logger.info(f"Starting training loop from step {start_step}...")

    try:
        while update_step < config["total_updates"]:
            # ================================================================
            # FETCH BATCH
            # ================================================================
            try:
                batch = next(train_iter)
            except StopIteration:
                logger.info("Data iterator exhausted; recreating")
                train_iter = iter(train_loader)
                batch = next(train_iter)

            # ================================================================
            # VALIDATE BATCH
            # ================================================================
            if not validate_batch(batch, model_type):
                continue

            # ================================================================
            # FORWARD PASS
            # ================================================================
            loss = safe_forward_pass(
                model, batch, device,
                autocast_enabled=True,
                model_type=model_type
            )

            if loss is None:
                logger.warning(f"Step {update_step}: Forward pass failed, skipping")
                optimizer.zero_grad(set_to_none=True)
                continue

            # Scale loss for gradient accumulation
            loss = loss / grad_accum

            # ================================================================
            # VALIDATE LOSS
            # ================================================================
            if not validate_loss(loss * grad_accum, update_step):
                optimizer.zero_grad(set_to_none=True)
                continue

            # ================================================================
            # BACKWARD PASS
            # ================================================================
            scaler.scale(loss).backward()

            # Track loss
            batch_loss = loss.item() * grad_accum
            accum_loss += batch_loss
            micro_step += 1

            # ================================================================
            # OPTIMIZER STEP (after gradient accumulation)
            # ================================================================
            if micro_step % grad_accum == 0:
                # Safe step with gradient validation
                step_successful = scaler.safe_step(
                    optimizer, model,
                    max_norm=config["grad_clip_norm"]
                )

                if not step_successful:
                    logger.warning(f"Step {update_step}: Gradient step failed, skipping")
                    accum_loss = 0.0
                    continue

                # Scheduler step
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                # Record metrics
                update_step += 1
                avg_loss = accum_loss / grad_accum
                train_losses.append(avg_loss)
                train_steps.append(update_step)

                # Get current learning rate
                lr = scheduler.get_last_lr()[0]

                # Log to wandb
                if update_step % config["log_interval"] == 0:
                    wandb_run.log({
                        "train_loss": avg_loss,
                        "learning_rate": lr,
                        "update_step": update_step,
                    })

                # Update progress bar
                pbar.update(1)
                pbar.set_description(
                    f"Training {model_name} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"LR: {lr:.2e}"
                )

                # ============================================================
                # VALIDATION
                # ============================================================
                if update_step % config["val_interval"] == 0:
                    logger.info(f"Running validation at step {update_step}")
                    val_loss = compute_validation_loss(
                        model, val_loader, device,
                        steps=config["val_steps"],
                        model_type=model_type
                    )
                    val_perplexity = math.exp(val_loss)

                    val_losses.append(val_loss)
                    val_steps.append(update_step)

                    logger.info(f"Validation loss: {val_loss:.4f}")
                    wandb_run.log({
                        "val_loss": val_loss,
                        "val_perplexity": val_perplexity,
                        "update_step": update_step,
                    })

                    # Check training health
                    health = check_training_health(train_losses, val_losses)
                    if health["warnings"]:
                        for warning in health["warnings"]:
                            logger.warning(f"Health check: {warning}")

                # ============================================================
                # CHECKPOINTING
                # ============================================================
                if update_step in config["checkpoint_steps"]:
                    # Create checkpoints directory if it doesn't exist
                    os.makedirs('checkpoints', exist_ok=True)

                    lr_suffix = f"_lr{config['lr']:.0e}".replace("e-0", "e-")
                    #checkpoint_path = f"checkpoints/{model_name}_step{update_step}{lr_suffix}.pt"
                    checkpoint_path = f"checkpoints/{experiment_name}_step{update_step}.pt"

                    torch.save({
                        'step': update_step,  # ← IMPORTANT: Save as 'step' not 'update_step'
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                        'train_steps': train_steps,
                        'val_steps': val_steps,
                        'config': config,
                        'train_loss': avg_loss,
                        'val_loss': val_losses[-1] if val_losses else None,
                        'learning_rate': lr,
                    }, checkpoint_path)

                    wandb_run.save(checkpoint_path)
                    logger.info(f"💾 Saved checkpoint: {checkpoint_path}")

                # ============================================================
                # MEMORY MANAGEMENT
                # ============================================================
                if update_step % 1000 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                # Reset accumulated loss
                accum_loss = 0.0

    except KeyboardInterrupt:
        logger.info("\n⚠️  Training interrupted by user")

        # Save emergency checkpoint
        logger.info("💾 Saving emergency checkpoint...")
        os.makedirs('checkpoints', exist_ok=True)
        lr_suffix = f"_lr{config['lr']:.0e}".replace("e-0", "e-")
        emergency_path = f"checkpoints/{model_name}_emergency_step{update_step}{lr_suffix}.pt"

        torch.save({
            'step': update_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_steps': train_steps,
            'val_steps': val_steps,
            'config': config,
        }, emergency_path)

        logger.info(f"✅ Emergency checkpoint saved: {emergency_path}")
        logger.info(f"   Resume with: RESUME_FROM='{emergency_path}' python train.py")

    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        raise
    finally:
        pbar.close()

    # ========================================================================
    # FINAL VALIDATION
    # ========================================================================
    logger.info("Running final validation...")
    final_val_loss = compute_validation_loss(
        model, val_loader, device,
        steps=config["val_steps"] * 2,
        model_type=model_type
    )
    final_val_perplexity = math.exp(final_val_loss)
    logger.info(f"Final validation loss: {final_val_loss:.4f}")
    logger.info(f"Final validation perplexity: {final_val_perplexity:.2f}")

    # ========================================================================
    # SAVE TRAINING DATA FOR PLOTTING
    # ========================================================================
    # Calculate perplexities for all validation losses
    val_perplexities = [math.exp(loss) for loss in val_losses]
    train_perplexities = [math.exp(loss) for loss in train_losses]

    training_data = {
        "train_steps": train_steps,
        "train_losses": train_losses,
        "train_perplexities": train_perplexities,
        "val_steps": val_steps,
        "val_losses": val_losses,
        "val_perplexities": val_perplexities,
        "final_val_loss": final_val_loss,
        "final_val_perplexity": final_val_perplexity,
        "config": config,
        "model_type": model_type,
        "model_name": model_name,
    }

    lr_suffix = f"_lr{config['lr']:.0e}".replace("e-0", "e-")
    #data_path = f"{model_type}_training_data{lr_suffix}.pkl"
    data_path = f"{experiment_name}_training_data.pkl"

    with open(data_path, "wb") as f:
        pickle.dump(training_data, f)
    logger.info(f"📊 Saved training data: {data_path}")
    wandb_run.save(data_path)

    # ========================================================================
    # SAVE FINAL MODEL
    # ========================================================================
    lr_suffix = f"_lr{config['lr']:.0e}".replace("e-0", "e-")
    #final_model_path = f"{model_name}_final{lr_suffix}.pt"
    final_model_path = f"{experiment_name}_final.pt"

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'final_val_loss': final_val_loss,
        'final_val_perplexity': final_val_perplexity,
        'total_updates': update_step,
    }, final_model_path)
    wandb_run.save(final_model_path)
    logger.info(f"💾 Saved final model: {final_model_path}")

    # Finish wandb run
    wandb_run.finish()

    logger.info("="*70)
    logger.info(f"✅ {model_name} TRAINING COMPLETE!")
    logger.info("="*70)
    logger.info(f"Final validation loss: {final_val_loss:.4f}")
    logger.info(f"Final validation perplexity: {final_val_perplexity:.2f}")
    logger.info(f"Total updates: {update_step}")
    logger.info(f"Training data saved: {data_path}")
    logger.info(f"Final model saved: {final_model_path}")
    logger.info("="*70)


# ===================================================================
# Main
# ===================================================================

def main():
    """
    Minimal runner to keep training script clean and reproducible.
    Uses config.py for experiment definitions.
    """

    # ---- Mode selection (keep it simple) ----
    print("\nSelect mode:")
    print("  1) Baseline: ESM2 vs mLSTM (6-layer)")
    print("  2) Exit")
    choice = input("Enter choice (1/2): ").strip()

    if choice != "1":
        print("Exiting.")
        return

    # ---- Baseline selection ----
    # We keep baseline keys stable like: lr1e-3_w1000, lr4e-4_w2000 ...
    from config import BASELINE_CONFIGS  # requires the clean config.py version I gave you

    keys = list(BASELINE_CONFIGS.keys())
    print("\nAvailable baseline configs:")
    for i, k in enumerate(keys, 1):
        print(f"  {i}) {k}")

    sel = input("Choose a run number or type 'all': ").strip().lower()

    # Import builders (clean model creation)
    from esm2_model import build_esm2_model
    from mlstm_modelNew import build_mlstm_model

    def run_one(key: str):
        pair = BASELINE_CONFIGS[key]
        esm_cfg = pair["esm2"]
        mlstm_cfg = pair["mlstm"]

        print("\n" + "=" * 70)
        print(f"RUNNING BASELINE: {key}")
        print("=" * 70)

        # Build models using same logic as before (from-scratch ESM2, comparable mLSTM)
        esm_model = build_esm2_model(esm_cfg)          # returns model (keeps your verification)
        mlstm_model = build_mlstm_model(mlstm_cfg)     # returns model

        # Train ESM2 then mLSTM (same data pipeline, same logging style)
        train_model(esm_model, "esm2", device=DEVICE, config=esm_cfg)
        train_model(mlstm_model, "mlstm", device=DEVICE, config=mlstm_cfg)

    if sel == "all":
        for k in keys:
            run_one(k)
    else:
        idx = int(sel) - 1
        run_one(keys[idx])


if __name__ == "__main__":
    main()
