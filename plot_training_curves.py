"""
Training Loss Curves Plotter

Generates publication-quality plots showing training loss progression.
Shows how models learn over time during training.

Usage:
    python plot_training_curves.py
    
Generates:
    - Individual training loss plots
    - Combined comparison plots
    - Training vs validation curves
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from scipy.signal import savgol_filter

# Publication style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10


def load_experiment_data(experiment_name: str) -> Optional[Dict]:
    """Load training data for an experiment."""
    # Try organized structure first
    organized_path = Path("experiments") / experiment_name / "training_data.pkl"
    if organized_path.exists():
        with open(organized_path, 'rb') as f:
            return pickle.load(f)
    
    # Try flat structure
    import glob
    possible_files = glob.glob(f"*{experiment_name}*training_data*.pkl")
    if possible_files:
        with open(possible_files[0], 'rb') as f:
            return pickle.load(f)
    
    return None


def smooth_curve(values, window_length=51, polyorder=3):
    """Smooth curve using Savitzky-Golay filter."""
    if len(values) < window_length:
        window_length = len(values) if len(values) % 2 == 1 else len(values) - 1
    if window_length < 4:
        return values
    return savgol_filter(values, window_length=window_length, polyorder=polyorder)


def plot_baseline_training_curves():
    """Plot training curves for baseline models."""
    
    print("\n" + "="*70)
    print("BASELINE TRAINING CURVES")
    print("="*70)
    
    experiments = [
        ('ESM2 6-layer', 'esm2_lr4e-4_w1000', '#e74c3c'),
        ('mLSTM 6-layer', 'mlstm_lr4e-4_w1000', '#3498db'),
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Training Loss
    for display_name, exp_name, color in experiments:
        data = load_experiment_data(exp_name)
        if data:
            train_steps = data.get('train_steps', [])
            train_losses = data.get('train_losses', [])
            
            if train_steps and train_losses:
                # Smooth for visualization
                smoothed = smooth_curve(train_losses, window_length=51, polyorder=3)
                ax1.plot(train_steps, smoothed, label=display_name, 
                        color=color, linewidth=2, alpha=0.9)
        else:
            print(f"⚠️  Data not found: {exp_name}")
    
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss Progression', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 2: Validation Perplexity
    for display_name, exp_name, color in experiments:
        data = load_experiment_data(exp_name)
        if data:
            val_steps = data.get('val_steps', [])
            val_perplexities = data.get('val_perplexities', [])
            
            if val_steps and val_perplexities:
                # Smooth for visualization
                smoothed = smooth_curve(val_perplexities, window_length=5, polyorder=2)
                ax2.plot(val_steps, smoothed, 'o-', label=display_name,
                        color=color, linewidth=2, markersize=5, alpha=0.9)
    
    ax2.set_xlabel('Training Steps', fontsize=12)
    ax2.set_ylabel('Validation Perplexity', fontsize=12)
    ax2.set_title('Validation Performance', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save
    output_dir = Path("thesis_plots")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "baseline_training_curves.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


def plot_depth_training_curves():
    """Plot training curves for different depths (best hyperparameter only)."""
    
    print("\n" + "="*70)
    print("DEPTH COMPARISON TRAINING CURVES")
    print("="*70)
    
    # Use best hyperparameter for each depth (lr4e-4_w1k)
    experiments = [
        ('mLSTM 1-layer', 'mlstm_1layer_lr4e-4_w1k', '#f39c12'),
        ('mLSTM 6-layer', 'mlstm_6layer_lr4e-4_w1k', '#3498db'),
        ('mLSTM 12-layer', 'mlstm_12layer_lr4e-4_w1k', '#9b59b6'),
        ('ESM2 6-layer', 'esm2_baseline_lr4e-4_w1k', '#e74c3c'),
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Training Loss
    for display_name, exp_name, color in experiments:
        data = load_experiment_data(exp_name)
        if data:
            train_steps = data.get('train_steps', [])
            train_losses = data.get('train_losses', [])
            
            if train_steps and train_losses:
                smoothed = smooth_curve(train_losses, window_length=51, polyorder=3)
                ax1.plot(train_steps, smoothed, label=display_name,
                        color=color, linewidth=2, alpha=0.9)
        else:
            print(f"⚠️  Data not found: {exp_name}")
    
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss by Model Depth', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 2: Validation Perplexity
    for display_name, exp_name, color in experiments:
        data = load_experiment_data(exp_name)
        if data:
            val_steps = data.get('val_steps', [])
            val_perplexities = data.get('val_perplexities', [])
            
            if val_steps and val_perplexities:
                smoothed = smooth_curve(val_perplexities, window_length=5, polyorder=2)
                ax2.plot(val_steps, smoothed, 'o-', label=display_name,
                        color=color, linewidth=2, markersize=5, alpha=0.9)
    
    ax2.set_xlabel('Training Steps', fontsize=12)
    ax2.set_ylabel('Validation Perplexity', fontsize=12)
    ax2.set_title('Validation Performance by Depth', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save
    output_dir = Path("thesis_plots")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "depth_training_curves.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


def plot_hyperparameter_training_curves():
    """Plot training curves for hyperparameter comparison (6-layer mLSTM)."""
    
    print("\n" + "="*70)
    print("HYPERPARAMETER COMPARISON TRAINING CURVES")
    print("="*70)
    
    # All hyperparams for 6-layer mLSTM
    experiments = [
        ('lr=1e-3, warmup=1k', 'mlstm_6layer_lr1e-3_w1k', '#e74c3c'),
        ('lr=1e-3, warmup=2k', 'mlstm_6layer_lr1e-3_w2k', '#f39c12'),
        ('lr=4e-4, warmup=1k', 'mlstm_6layer_lr4e-4_w1k', '#3498db'),
        ('lr=4e-4, warmup=2k', 'mlstm_6layer_lr4e-4_w2k', '#9b59b6'),
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Training Loss
    for display_name, exp_name, color in experiments:
        data = load_experiment_data(exp_name)
        if data:
            train_steps = data.get('train_steps', [])
            train_losses = data.get('train_losses', [])
            
            if train_steps and train_losses:
                smoothed = smooth_curve(train_losses, window_length=51, polyorder=3)
                ax1.plot(train_steps, smoothed, label=display_name,
                        color=color, linewidth=2, alpha=0.9)
        else:
            print(f"⚠️  Data not found: {exp_name}")
    
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss - Hyperparameter Comparison (6-layer mLSTM)', 
                 fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 2: Validation Perplexity
    for display_name, exp_name, color in experiments:
        data = load_experiment_data(exp_name)
        if data:
            val_steps = data.get('val_steps', [])
            val_perplexities = data.get('val_perplexities', [])
            
            if val_steps and val_perplexities:
                smoothed = smooth_curve(val_perplexities, window_length=5, polyorder=2)
                ax2.plot(val_steps, smoothed, 'o-', label=display_name,
                        color=color, linewidth=2, markersize=5, alpha=0.9)
    
    ax2.set_xlabel('Training Steps', fontsize=12)
    ax2.set_ylabel('Validation Perplexity', fontsize=12)
    ax2.set_title('Validation Performance - Hyperparameter Comparison',
                 fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save
    output_dir = Path("thesis_plots")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "hyperparameter_training_curves.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


def plot_training_vs_validation():
    """Plot training loss vs validation loss for best models."""
    
    print("\n" + "="*70)
    print("TRAINING VS VALIDATION CURVES")
    print("="*70)
    
    experiments = [
        ('ESM2 6-layer', 'esm2_lr4e-4_w1000', '#e74c3c'),
        ('mLSTM 6-layer', 'mlstm_6layer_lr4e-4_w1k', '#3498db'),
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, (display_name, exp_name, color) in enumerate(experiments):
        ax = axes[idx]
        data = load_experiment_data(exp_name)
        
        if data:
            # Training loss
            train_steps = data.get('train_steps', [])
            train_losses = data.get('train_losses', [])
            
            # Validation loss
            val_steps = data.get('val_steps', [])
            val_losses = data.get('val_losses', [])
            
            if train_steps and train_losses:
                smoothed_train = smooth_curve(train_losses, window_length=51, polyorder=3)
                ax.plot(train_steps, smoothed_train, label='Training Loss',
                       color=color, linewidth=2, alpha=0.8)
            
            if val_steps and val_losses:
                smoothed_val = smooth_curve(val_losses, window_length=5, polyorder=2)
                ax.plot(val_steps, smoothed_val, 'o-', label='Validation Loss',
                       color='#2ecc71', linewidth=2, markersize=5, alpha=0.9)
            
            ax.set_xlabel('Training Steps', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title(f'{display_name}\nTraining vs Validation',
                        fontsize=13, fontweight='bold')
            ax.legend(loc='upper right', fontsize=11)
            ax.grid(True, alpha=0.3, linestyle='--')
        else:
            print(f"⚠️  Data not found: {exp_name}")
    
    plt.tight_layout()
    
    # Save
    output_dir = Path("thesis_plots")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "training_vs_validation.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


def main():
    """Generate all training curve plots."""
    
    print("\n" + "="*80)
    print("TRAINING CURVES GENERATOR")
    print("="*80)
    print("Generating publication-quality training curve plots...")
    print("="*80 + "\n")
    
    # Create output directory
    output_dir = Path("thesis_plots")
    output_dir.mkdir(exist_ok=True)
    
    # Generate all plots
    plot_baseline_training_curves()
    plot_depth_training_curves()
    plot_hyperparameter_training_curves()
    plot_training_vs_validation()
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\n📁 All plots saved to: {output_dir}/")
    print("  • baseline_training_curves.png       - ESM2 vs mLSTM training")
    print("  • depth_training_curves.png          - 1L vs 6L vs 12L comparison")
    print("  • hyperparameter_training_curves.png - LR/warmup impact")
    print("  • training_vs_validation.png         - Overfitting analysis")
    print("\n✨ Ready for thesis inclusion!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
