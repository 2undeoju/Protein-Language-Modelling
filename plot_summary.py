"""
Layer Comparison Summary: 2×2 Grid

Shows all 4 hyperparameter comparisons in one figure.

Usage: python plot_summary.py

Output: Single summary figure in ./plots_depth_comparison/
"""

import pickle
import math
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from scipy.signal import savgol_filter

plt.style.use('seaborn-v0_8-paper')


def smooth_perplexity(perplexities, window_length=5, polyorder=2):
    """
    Smooth perplexity curve using Savitzky-Golay filter.

    Args:
        perplexities: List of perplexity values
        window_length: Window size (must be odd)
        polyorder: Polynomial order

    Returns:
        Smoothed perplexities
    """
    if len(perplexities) < window_length:
        return perplexities
    return savgol_filter(perplexities, window_length=window_length, polyorder=polyorder)


def load_experiment_data(experiment_name: str) -> Optional[Dict]:
    """Load training data using clean naming convention."""
    filename = f"training_data/{experiment_name}_training_data.pkl"

    if Path(filename).exists():
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None


def load_pretrained_baseline() -> Optional[float]:
    """Load pretrained baseline."""
    if Path("training_data/esm2_8m_pretrained_baseline.pkl").exists():
        with open("training_data/esm2_8m_pretrained_baseline.pkl", 'rb') as f:
            return pickle.load(f).get('validation_perplexity')
    return None


def create_summary_grid(output_dir: str = './plots_depth_comparison'):
    """Create 2×2 grid showing all 4 hyperparameter comparisons."""

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # Four configurations for clean comparison
    hyperparams = [
        ('lr1e-3_w1k', 1e-3, 1000, 'LR=1e-3, Warmup=1000'),
        ('lr4e-4_w1k', 4e-4, 1000, 'LR=4e-4, Warmup=1000'),
        ('lr1e-3_w2k', 1e-3, 2000, 'LR=1e-3, Warmup=2000'),
        ('lr4e-4_w2k', 4e-4, 2000, 'LR=4e-4, Warmup=2000')

    ]

    pretrained_ppl = load_pretrained_baseline()

    for idx, (hyperparam_name, lr, warmup, title) in enumerate(hyperparams):
        ax = axes[idx]

        # Determine warmup short form
        warmup_short = f"{warmup//1000}k" if warmup >= 1000 else str(warmup)

        experiments = {
            'esm2': {
                'name': f'esm2_lr{lr:.0e}_w{warmup}'.replace('e-0', 'e-'),
                'label': 'ESM2~8M 6-Layers',
                'color': '#e74c3c',
                'marker': 'o',
            },
            'mlstm_1': {
                'name': f'mlstm_1layer_lr{lr:.0e}_w{warmup_short}'.replace('e-0', 'e-'),
                'label': 'mLSTM-8M 1-Layer',
                'color': '#f39c12',
                'marker': 's',
            },
            'mlstm_6': {
                'name': f'mlstm_lr{lr:.0e}_w{warmup}'.replace('e-0', 'e-'),
                'label': 'mLSTM-8M 6-Layers',
                'color': '#3498db',
                'marker': '^',
            },
            'mlstm_12': {
                'name': f'mlstm_12layer_lr{lr:.0e}_w{warmup_short}'.replace('e-0', 'e-'),
                'label': 'mLSTM-8M 12-Layers',
                'color': '#9b59b6',
                'marker': 'D',
            },
        }

        # Plot each experiment
        for key in ['esm2', 'mlstm_1', 'mlstm_6', 'mlstm_12']:
            exp = experiments[key]
            data = load_experiment_data(exp['name'])

            if data:
                val_steps = data.get('val_steps', [])
                val_perplexities = data.get('val_perplexities', [])

                if not val_perplexities and data.get('val_losses'):
                    val_perplexities = [math.exp(loss) for loss in data['val_losses']]

                if val_steps and val_perplexities:
                    # Smooth the perplexity curve
                    smoothed_ppl = smooth_perplexity(val_perplexities, window_length=5, polyorder=2)

                    ax.plot(val_steps, smoothed_ppl,
                            label=exp['label'], color=exp['color'],
                            marker=exp['marker'], markersize=4,
                            linewidth=1.5, alpha=0.85)

        # Pretrained baseline
        if pretrained_ppl:
            ax.axhline(y=pretrained_ppl, color='#2ecc71',
                       linestyle='--', linewidth=2,
                       label=f'Pretrained: {pretrained_ppl:.2f}',
                       alpha=0.8)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Training Steps', fontsize=10)
        ax.set_ylabel('Validation Perplexity', fontsize=10)
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')

    plt.suptitle('mLSTM Depth Exploration: All Hyperparameter Configurations',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    # Save
    filepath = output_path / "depth_comparison_summary_grid.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: depth_comparison_summary_grid.png")

    filepath_pdf = output_path / "depth_comparison_summary_grid.pdf"
    plt.savefig(filepath_pdf, bbox_inches='tight')
    print(f"✅ Saved: depth_comparison_summary_grid.pdf")

    plt.close()


if __name__ == "__main__":
    create_summary_grid()