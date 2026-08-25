"""
Resume Training Utility

This script helps you resume training from a checkpoint.
Handles both old ('update_step') and new ('step') checkpoint formats.
"""

import torch
import os
from pathlib import Path


def resume_training(checkpoint_path, model, optimizer, scaler=None, scheduler=None):
    """
    Resume training from a checkpoint.

    Handles both old and new checkpoint formats:
    - Old format: uses 'update_step' key
    - New format: uses 'step' key

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into
        scaler: Optional gradient scaler for mixed precision
        scheduler: Optional learning rate scheduler

    Returns:
        start_step: Step number to resume from
        checkpoint_info: Dictionary with checkpoint metadata
    """
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return 0, None

    print("="*70)
    print("RESUMING TRAINING FROM CHECKPOINT")
    print("="*70)
    print(f"Loading: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Model state loaded")

    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"✅ Optimizer state loaded")

    # Load scaler if using mixed precision
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print(f"✅ Gradient scaler state loaded")

    # Load scheduler if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"✅ Scheduler state loaded")

    # Get resume step - handle both old and new formats
    if 'step' in checkpoint:
        current_step = checkpoint['step']
    elif 'update_step' in checkpoint:
        current_step = checkpoint['update_step']
        print(f"⚠️  Old checkpoint format detected (using 'update_step')")
    else:
        print(f"⚠️  No step information in checkpoint, starting from 0")
        current_step = 0

    start_step = current_step + 1

    # Print checkpoint info
    print()
    print("Checkpoint Information:")
    print("-"*70)
    print(f"  Step: {current_step:,}")
    print(f"  Resuming from: {start_step:,}")

    if 'train_loss' in checkpoint:
        print(f"  Training loss: {checkpoint['train_loss']:.4f}")

    if 'val_loss' in checkpoint:
        print(f"  Validation loss: {checkpoint['val_loss']:.4f}")
        if 'val_perplexity' in checkpoint:
            print(f"  Validation perplexity: {checkpoint['val_perplexity']:.2f}")

    if 'learning_rate' in checkpoint:
        print(f"  Learning rate: {checkpoint['learning_rate']:.2e}")

    # Print training history if available
    if 'train_losses' in checkpoint:
        train_losses = checkpoint['train_losses']
        if len(train_losses) > 0:
            print(f"  Steps trained: {len(train_losses):,}")
            print(f"  Best training loss: {min(train_losses):.4f}")
            if len(train_losses) >= 100:
                recent_avg = sum(train_losses[-100:]) / 100
                print(f"  Recent avg loss: {recent_avg:.4f}")

    print("="*70)
    print()

    # Prepare info dict
    checkpoint_info = {
        'step': current_step,
        'train_loss': checkpoint.get('train_loss', None),
        'val_loss': checkpoint.get('val_loss', None),
        'learning_rate': checkpoint.get('learning_rate', None),
    }

    return start_step, checkpoint_info


def find_latest_checkpoint(checkpoint_dir='checkpoints', prefix='checkpoint'):
    """
    Find the latest checkpoint in a directory.

    Args:
        checkpoint_dir: Directory containing checkpoints
        prefix: Prefix of checkpoint files

    Returns:
        Path to latest checkpoint, or None if none found
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        # Also check current directory
        checkpoints = list(Path('.').glob(f"{prefix}_*.pt"))
        if not checkpoints:
            print(f"Checkpoint directory not found: {checkpoint_dir}")
            return None
    else:
        checkpoints = list(checkpoint_dir.glob(f"{prefix}_*.pt"))

    if not checkpoints:
        print(f"No checkpoints found matching pattern: {prefix}_*.pt")
        return None

    # Sort by step number (extract from filename)
    def get_step(path):
        try:
            # Extract number from filename like "checkpoint_5000.pt" or "mLSTM_step5000.pt"
            stem = path.stem
            # Try different patterns
            if '_step' in stem:
                num_str = stem.split('_step')[-1].split('_')[0]
            else:
                # Fallback: find last number in filename
                import re
                numbers = re.findall(r'\d+', stem)
                num_str = numbers[-1] if numbers else '0'
            return int(num_str)
        except:
            return 0

    checkpoints.sort(key=get_step)
    latest = checkpoints[-1]

    print(f"Found {len(checkpoints)} checkpoint(s)")
    print(f"Latest checkpoint: {latest} (step {get_step(latest)})")

    return str(latest)


def list_available_checkpoints(checkpoint_dir='checkpoints', prefix='checkpoint'):
    """
    List all available checkpoints with their info.

    Args:
        checkpoint_dir: Directory containing checkpoints
        prefix: Prefix of checkpoint files
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        # Also check current directory
        checkpoints = list(Path('.').glob(f"{prefix}_*.pt"))
        if not checkpoints:
            print(f"Checkpoint directory not found: {checkpoint_dir}")
            return
    else:
        checkpoints = list(checkpoint_dir.glob(f"{prefix}_*.pt"))

    if not checkpoints:
        print(f"No checkpoints found matching pattern: {prefix}_*.pt")
        return

    # Sort by step number
    def get_step(path):
        try:
            stem = path.stem
            if '_step' in stem:
                num_str = stem.split('_step')[-1].split('_')[0]
            else:
                import re
                numbers = re.findall(r'\d+', stem)
                num_str = numbers[-1] if numbers else '0'
            return int(num_str)
        except:
            return 0

    checkpoints.sort(key=get_step)

    print("="*70)
    print("AVAILABLE CHECKPOINTS")
    print("="*70)
    print(f"{'Step':>8s} | {'File':40s} | {'Size':>10s} | {'Loss':>8s}")
    print("-"*70)

    for ckpt in checkpoints:
        step = get_step(ckpt)
        size_mb = ckpt.stat().st_size / (1024 * 1024)

        # Try to load loss info
        try:
            state = torch.load(ckpt, map_location='cpu')
            loss = state.get('train_loss', state.get('val_loss', None))
            loss_str = f"{loss:.4f}" if loss else "N/A"
        except:
            loss_str = "Error"

        print(f"{step:8,d} | {ckpt.name:40s} | {size_mb:8.2f} MB | {loss_str:>8s}")

    print("="*70)
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Resume training utility')
    parser.add_argument('--list', action='store_true',
                        help='List available checkpoints')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory containing checkpoints')
    parser.add_argument('--prefix', type=str, default='checkpoint',
                        help='Checkpoint file prefix')
    parser.add_argument('--latest', action='store_true',
                        help='Find latest checkpoint')

    args = parser.parse_args()

    if args.list:
        list_available_checkpoints(args.checkpoint_dir, args.prefix)
    elif args.latest:
        latest = find_latest_checkpoint(args.checkpoint_dir, args.prefix)
        if latest:
            print(f"\nLatest checkpoint: {latest}")
            print(f"To resume training, use:")
            print(f"  RESUME_FROM='{latest}' python train.py")
    else:
        print("Usage:")
        print("  python resume_training.py --list              # List all checkpoints")
        print("  python resume_training.py --latest            # Find latest checkpoint")
        print()
        print("In your training script:")
        print("  from resume_training import resume_training")
        print("  start_step, info = resume_training(checkpoint_path, model, optimizer)")