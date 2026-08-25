"""
Generate 4 Layer Comparison Plots

Creates one plot per hyperparameter combination, comparing:
- ESM2 baseline (trained from scratch)
- mLSTM 1-layer, 6-layer, 12-layer
- Pretrained ESM2-8M (green dashed)

Usage: python plot_layers.py

Output: 4 publication-quality plots in ./plots_depth_comparison/
"""

import pickle
import math
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Optional

# Publication style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10


def load_experiment_data(experiment_name: str) -> Optional[Dict]:
    """Load training data for an experiment using clean naming convention."""
    filename = f"training_data/{experiment_name}_training_data.pkl"

    if Path(filename).exists():
        with open(filename, 'rb') as f:
            return pickle.load(f)

    print(f"⚠️  Not found: {filename}")
    return None


def load_pretrained_baseline() -> Optional[float]:
    """Load pretrained ESM2-8M baseline perplexity."""
    baseline_file = "training_data/esm2_8m_pretrained_baseline.pkl"

    if Path(baseline_file).exists():
        with open(baseline_file, 'rb') as f:
            data = pickle.load(f)
        return data.get('validation_perplexity')

    print(f"⚠️  Pretrained baseline not found. Run: python generate_pretrained_baseline.py")
    return None


def create_comparison_plot(
        hyperparam_name: str,
        lr: float,
        warmup: int,
        output_dir: str = './plots_depth_comparison'
):
    """
    Create comparison plot for one hyperparameter combination.

    Args:
        hyperparam_name: e.g., "lr1e-3_w1k"
        lr: Learning rate (for title)
        warmup: Warmup steps (for title)
        output_dir: Output directory for plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Determine warmup short form (1000 -> 1k, 2000 -> 2k)
    warmup_short = f"{warmup//1000}k" if warmup >= 1000 else str(warmup)

    # Load all experiments for this hyperparameter combination
    # Using clean naming convention
    experiments = {
        'esm2': {
            'name': f'esm2_lr{lr:.0e}_w{warmup}'.replace('e-0', 'e-'),
            'label': 'ESM2~8M 6Llayer',
            'color': '#e74c3c',  # Red
            'linestyle': '-',
            'marker': 'o',
        },
        'mlstm_1': {
            'name': f'mlstm_1layer_lr{lr:.0e}_w{warmup_short}'.replace('e-0', 'e-'),
            'label': 'mLSTM-8M 1-Layer',
            'color': '#f39c12',  # Orange
            'linestyle': '-',
            'marker': 's',
        },
        'mlstm_6': {
            'name': f'mlstm_lr{lr:.0e}_w{warmup}'.replace('e-0', 'e-'),
            'label': 'mLSTM-8M 6-Layers',
            'color': '#3498db',  # Blue
            'linestyle': '-',
            'marker': '^',
        },
        'mlstm_12': {
            'name': f'mlstm_12layer_lr{lr:.0e}_w{warmup_short}'.replace('e-0', 'e-'),
            'label': 'mLSTM-8M 12-Layers',
            'color': '#9b59b6',  # Purple
            'linestyle': '-',
            'marker': 'D',
        },
    }

    # Load data
    data_loaded = {}
    for key, exp in experiments.items():
        data = load_experiment_data(exp['name'])
        if data:
            data_loaded[key] = data

    if not data_loaded:
        print(f"❌ No data found for {hyperparam_name}")
        return

    # Load pretrained baseline
    pretrained_ppl = load_pretrained_baseline()

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each experiment
    for key in ['esm2', 'mlstm_1', 'mlstm_6', 'mlstm_12']:
        if key not in data_loaded:
            continue

        data = data_loaded[key]
        exp = experiments[key]

        val_steps = data.get('val_steps', [])
        val_perplexities = data.get('val_perplexities', [])

        if not val_steps or not val_perplexities:
            # Calculate perplexities from losses
            val_losses = data.get('val_losses', [])
            if val_losses:
                val_perplexities = [math.exp(loss) for loss in val_losses]

        if val_steps and val_perplexities:
            ax.plot(
                val_steps,
                val_perplexities,
                label=exp['label'],
                color=exp['color'],
                linestyle=exp['linestyle'],
                marker=exp['marker'],
                markersize=5,
                linewidth=2,
                alpha=0.9,
            )

    # Plot pretrained baseline
    if pretrained_ppl:
        ax.axhline(
            y=pretrained_ppl,
            color='#2ecc71',  # Green
            linestyle='--',
            linewidth=2.5,
            label=f'ESM2-8M Pretrained: {pretrained_ppl:.2f}',
            alpha=0.9,
        )

    # Formatting
    ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation Perplexity', fontsize=12, fontweight='bold')

    # Format learning rate: 4e-4 not 4e-04
    lr_formatted = f"{lr:.0e}".replace('e-0', 'e-')
    title = f'Depth Comparison (LR={lr_formatted}, Warmup={warmup})'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    ax.legend(loc='upper right', framealpha=0.95, edgecolor='gray')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Set reasonable y-axis limits
    if data_loaded:
        all_ppls = []
        for data in data_loaded.values():
            ppls = data.get('val_perplexities', [])
            if ppls:
                all_ppls.extend(ppls)

        if all_ppls:
            min_ppl = min(all_ppls)
            max_ppl = max(all_ppls)
            margin = (max_ppl - min_ppl) * 0.1
            ax.set_ylim(min_ppl - margin, max_ppl + margin)

    plt.tight_layout()

    # Save plot
    filename = f"depth_comparison_{hyperparam_name}.png"
    filepath = output_path / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: {filename}")

    # Also save PDF version
    filepath_pdf = output_path / filename.replace('.png', '.pdf')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Recreate plot for PDF (same code as above)
    for key in ['esm2', 'mlstm_1', 'mlstm_6', 'mlstm_12']:
        if key not in data_loaded:
            continue

        data = data_loaded[key]
        exp = experiments[key]

        val_steps = data.get('val_steps', [])
        val_perplexities = data.get('val_perplexities', [])

        if not val_perplexities and data.get('val_losses'):
            val_perplexities = [math.exp(loss) for loss in data['val_losses']]

        if val_steps and val_perplexities:
            ax.plot(val_steps, val_perplexities, label=exp['label'],
                    color=exp['color'], linestyle=exp['linestyle'],
                    marker=exp['marker'], markersize=5, linewidth=2, alpha=0.9)

    if pretrained_ppl:
        ax.axhline(y=pretrained_ppl, color='#2ecc71', linestyle='--',
                   linewidth=2.5, label=f'ESM2-8M Pretrained: {pretrained_ppl:.2f}', alpha=0.9)

    ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation Perplexity', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper right', framealpha=0.95, edgecolor='gray')
    ax.grid(True, alpha=0.3, linestyle='--')

    if all_ppls:
        ax.set_ylim(min_ppl - margin, max_ppl + margin)

    plt.tight_layout()
    plt.savefig(filepath_pdf, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: {filename.replace('.png', '.pdf')}")


def create_all_comparison_plots(output_dir: str = './plots_depth_comparison'):
    """Generate all comparison plots for depth exploration."""

    print("\n" + "="*80)
    print("GENERATING DEPTH COMPARISON PLOTS")
    print("="*80 + "\n")

    # Four configurations: LR=4e-4/w=1000 LR=1e-3/w=1000, LR=4e-4/w=2000 and LR=1e-3/w=2000
    hyperparams = [
        ('lr1e-3_w1k', 1e-3, 1000),
        ('lr4e-4_w1k', 4e-4, 1000),
        ('lr1e-3_w2k', 1e-3, 2000),
        ('lr4e-4_w2k', 4e-4, 2000)
    ]

    for hyperparam_name, lr, warmup in hyperparams:
        print(f"\n📊 Creating plot for {hyperparam_name}...")
        create_comparison_plot(hyperparam_name, lr, warmup, output_dir)

    print("\n" + "="*80)
    print("✅ ALL PLOTS GENERATED")
    print("="*80)
    print(f"\n📁 Output directory: {output_dir}/")
    print("\nGenerated files:")
    print("  1. depth_comparison_lr1e-3_w1k.png/.pdf")
    print("  2. depth_comparison_lr4e-4_w1k.png/.pdf")
    print("  3. depth_comparison_lr1e-3_w2k.png/.pdf")
    print("  4. depth_comparison_lr4e-4_w2k.png/.pdf")
    print("="*80 + "\n")


if __name__ == "__main__":
    create_all_comparison_plots()