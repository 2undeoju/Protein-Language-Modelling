
"""
Clean plotting utilities for protein language model training visualization.

This module provides functions to create publication-quality plots for:
1. ESM training and validation curves
2. mLSTM training and validation curves
3. Validation loss comparison between models
4. Validation perplexity comparison with paper target reference

Author: Akintunde Ojutiku
Date: 2025-10-28
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
from pathlib import Path
from scipy.signal import savgol_filter


# ============================================================================
# CONFIGURATION
# ============================================================================

# Target perplexity from the cramming paper (Table 1)
PAPER_TARGET_PERPLEXITY = 13.72

# Plot style configuration
PLOT_CONFIG = {
    "figsize": (10, 6),
    "dpi": 300,
    "linewidth": 2,
    "markersize": 6,
    "alpha": 0.8,
    "grid_alpha": 0.3,
    "title_fontsize": 16,
    "label_fontsize": 14,
    "legend_fontsize": 12,
    "tick_fontsize": 12,
}

# Color scheme
COLORS = {
    "esm_train": "blue",
    "esm_val": "red",
    "mlstm_train": "blue",
    "mlstm_val": "red",
    "esm_comparison": "red",
    "mlstm_comparison": "blue",
    "target": "green",
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def loss_to_perplexity(loss: np.ndarray) -> np.ndarray:
    """
    Convert loss values to perplexity.

    Perplexity = exp(loss)

    Args:
        loss: Array of loss values

    Returns:
        Array of perplexity values
    """
    return np.exp(loss)


def save_plot(filename: str, output_dir: str = "./plots") -> None:
    """
    Save current plot in both PDF and PNG formats.

    Args:
        filename: Base filename (without extension)
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pdf_path = output_path / f"{filename}.pdf"
    png_path = output_path / f"{filename}.png"

    plt.savefig(pdf_path, dpi=PLOT_CONFIG["dpi"], bbox_inches="tight")
    plt.savefig(png_path, dpi=PLOT_CONFIG["dpi"], bbox_inches="tight")

    print(f"✅ Saved: {pdf_path}")
    print(f"✅ Saved: {png_path}")


def format_lr(lr: float) -> str:
    """
    Format LR like 4e-4 instead of 4e-04 (matches how it's typically written in code).
    """
    s = f"{lr:.0e}"              # e.g. 4e-04
    s = s.replace("e-0", "e-")   # -> 4e-4
    s = s.replace("e+0", "e+")   # safety for large values
    return s


def format_config_title(base_title: str, lr: float, warmup_steps: int) -> str:
    """
    Consistent plot title formatting across all plots.
    """
    return f"{base_title}\nLR={format_lr(lr)}, Warm-up={warmup_steps}"


# ============================================================================
# PLOT 1: ESM Training and Validation Loss
# ============================================================================

def plot_esm_training(
        train_steps: List[int],
        train_losses: List[float],
        val_steps: List[int],
        val_losses: List[float],
        lr: float,
        warmup_steps: int,
        output_dir: str = "./plots",
        show_plot: bool = True,
) -> None:
    """
    Create training and validation loss plot for ESM model.

    Args:
        train_steps: List of training step numbers
        train_losses: List of training loss values
        val_steps: List of validation step numbers
        val_losses: List of validation loss values
        output_dir: Directory to save plots
        show_plot: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=PLOT_CONFIG["figsize"])

    # Plot training loss
    ax.plot(
        train_steps, train_losses,
        color=COLORS["esm_train"],
        linewidth=PLOT_CONFIG["linewidth"],
        label="Training Loss",
        alpha=PLOT_CONFIG["alpha"]
    )

    # Plot validation loss with markers
    ax.plot(
        val_steps, val_losses,
        "o-",
        color=COLORS["esm_val"],
        linewidth=PLOT_CONFIG["linewidth"],
        markersize=PLOT_CONFIG["markersize"],
        label="Validation Loss",
        alpha=PLOT_CONFIG["alpha"]
    )

    # Styling
    ax.set_xlabel("Training Steps", fontsize=PLOT_CONFIG["label_fontsize"], fontweight="bold")
    ax.set_ylabel("Loss", fontsize=PLOT_CONFIG["label_fontsize"], fontweight="bold")

    ax.set_title(
        format_config_title("ESM2 Training and Validation Loss", lr, warmup_steps),
        fontsize=PLOT_CONFIG["title_fontsize"],
        fontweight="bold"
    )

    ax.legend(fontsize=PLOT_CONFIG["legend_fontsize"], loc="upper right")
    ax.grid(True, alpha=PLOT_CONFIG["grid_alpha"], linestyle="--")
    ax.tick_params(labelsize=PLOT_CONFIG["tick_fontsize"])

    plt.tight_layout()
    save_plot("esm_training_validation_curve", output_dir)

    if show_plot:
        plt.show()
    plt.close()


# ============================================================================
# PLOT 2: mLSTM Training and Validation Loss
# ============================================================================

def plot_mlstm_training(
        train_steps: List[int],
        train_losses: List[float],
        val_steps: List[int],
        val_losses: List[float],
        lr: float,
        warmup_steps: int,
        output_dir: str = "./plots",
        show_plot: bool = True,
) -> None:
    """
    Create training and validation loss plot for mLSTM model.

    Args:
        train_steps: List of training step numbers
        train_losses: List of training loss values
        val_steps: List of validation step numbers
        val_losses: List of validation loss values
        output_dir: Directory to save plots
        show_plot: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=PLOT_CONFIG["figsize"])

    # Plot training loss
    ax.plot(
        train_steps, train_losses,
        color=COLORS["mlstm_train"],
        linewidth=PLOT_CONFIG["linewidth"],
        label="Training Loss",
        alpha=PLOT_CONFIG["alpha"]
    )

    # Plot validation loss with markers
    ax.plot(
        val_steps, val_losses,
        "o-",
        color=COLORS["mlstm_val"],
        linewidth=PLOT_CONFIG["linewidth"],
        markersize=PLOT_CONFIG["markersize"],
        label="Validation Loss",
        alpha=PLOT_CONFIG["alpha"]
    )

    # Styling
    ax.set_xlabel("Training Steps", fontsize=PLOT_CONFIG["label_fontsize"], fontweight="bold")
    ax.set_ylabel("Loss", fontsize=PLOT_CONFIG["label_fontsize"], fontweight="bold")

    ax.set_title(
        format_config_title("mLSTM Training and Validation Loss", lr, warmup_steps),
        fontsize=PLOT_CONFIG["title_fontsize"],
        fontweight="bold"
    )

    ax.legend(fontsize=PLOT_CONFIG["legend_fontsize"], loc="upper right")
    ax.grid(True, alpha=PLOT_CONFIG["grid_alpha"], linestyle="--")
    ax.tick_params(labelsize=PLOT_CONFIG["tick_fontsize"])

    plt.tight_layout()
    save_plot("mlstm_training_validation_curve", output_dir)

    if show_plot:
        plt.show()
    plt.close()


# ============================================================================
# PLOT 3: Validation Loss Comparison
# ============================================================================

def plot_validation_loss_comparison(
        esm_val_steps: List[int],
        esm_val_losses: List[float],
        mlstm_val_steps: List[int],
        mlstm_val_losses: List[float],
        lr: float,
        warmup_steps: int,
        output_dir: str = "./plots",
        show_plot: bool = True,
) -> None:
    """
    Compare validation loss between ESM and mLSTM models.

    Args:
        esm_val_steps: ESM validation step numbers
        esm_val_losses: ESM validation loss values
        mlstm_val_steps: mLSTM validation step numbers
        mlstm_val_losses: mLSTM validation loss values
        output_dir: Directory to save plots
        show_plot: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=PLOT_CONFIG["figsize"])

    # Plot ESM validation loss
    ax.plot(
        esm_val_steps, esm_val_losses,
        "o-",
        color=COLORS["esm_comparison"],
        linewidth=PLOT_CONFIG["linewidth"],
        markersize=PLOT_CONFIG["markersize"],
        label="ESM 8M (6-layer)",
        alpha=PLOT_CONFIG["alpha"]
    )

    # Plot mLSTM validation loss
    ax.plot(
        mlstm_val_steps, mlstm_val_losses,
        "o-",
        color=COLORS["mlstm_comparison"],
        linewidth=PLOT_CONFIG["linewidth"],
        markersize=PLOT_CONFIG["markersize"],
        label="mLSTM 8M (6-layer)",
        alpha=PLOT_CONFIG["alpha"]
    )

    # Styling
    ax.set_xlabel("Training Steps", fontsize=PLOT_CONFIG["label_fontsize"], fontweight="bold")
    ax.set_ylabel("Validation Loss", fontsize=PLOT_CONFIG["label_fontsize"], fontweight="bold")

    ax.set_title(
        format_config_title("Validation Loss Comparison", lr, warmup_steps),
        fontsize=PLOT_CONFIG["title_fontsize"],
        fontweight="bold"
    )

    ax.legend(fontsize=PLOT_CONFIG["legend_fontsize"], loc="upper right")
    ax.grid(True, alpha=PLOT_CONFIG["grid_alpha"], linestyle="--")
    ax.tick_params(labelsize=PLOT_CONFIG["tick_fontsize"])

    plt.tight_layout()
    save_plot("validation_loss_comparison", output_dir)

    if show_plot:
        plt.show()
    plt.close()


# ============================================================================
# PLOT 4: Validation Perplexity Comparison with Target
# ============================================================================

def plot_validation_perplexity_comparison(
        esm_val_steps: List[int],
        esm_val_losses: List[float],
        mlstm_val_steps: List[int],
        mlstm_val_losses: List[float],
        lr: float,
        warmup_steps: int,
        target_perplexity: float = PAPER_TARGET_PERPLEXITY,
        output_dir: str = "./plots",
        show_plot: bool = True,
        target_label: str = "Paper Target",
) -> None:
    """
    Compare validation perplexity between ESM and mLSTM with target reference.

    Args:
        esm_val_steps: ESM validation step numbers
        esm_val_losses: ESM validation loss values
        mlstm_val_steps: mLSTM validation step numbers
        mlstm_val_losses: mLSTM validation loss values
        target_perplexity: Target perplexity (from paper or baseline)
        output_dir: Directory to save plots
        show_plot: Whether to display the plot
        target_label: Label for the target line (e.g., "ESM2-8M Pretrained" or "Paper Target")
    """
    # Convert losses to perplexity
    esm_val_perplexity = loss_to_perplexity(np.array(esm_val_losses))
    mlstm_val_perplexity = loss_to_perplexity(np.array(mlstm_val_losses))

    fig, ax = plt.subplots(figsize=PLOT_CONFIG["figsize"])

    # Smooth the data before plotting (safe window)
    if len(esm_val_perplexity) >= 5:
        esm_smooth = savgol_filter(esm_val_perplexity, window_length=5, polyorder=2)
    else:
        esm_smooth = esm_val_perplexity

    if len(mlstm_val_perplexity) >= 5:
        mlstm_smooth = savgol_filter(mlstm_val_perplexity, window_length=5, polyorder=2)
    else:
        mlstm_smooth = mlstm_val_perplexity

    # Plot smoothed curves
    ax.plot(
        esm_val_steps, esm_smooth,
        "o-",
        color="#d62728",
        label="ESM 8M (6-layer)",
        linewidth=PLOT_CONFIG["linewidth"],
        markersize=PLOT_CONFIG["markersize"]
    )

    ax.plot(
        mlstm_val_steps, mlstm_smooth,
        "o-",
        color="#1f77b4",
        label="mLSTM 8M (6-layer)",
        linewidth=PLOT_CONFIG["linewidth"],
        markersize=PLOT_CONFIG["markersize"]
    )

    # Add target reference line
    ax.axhline(
        y=target_perplexity,
        color=COLORS["target"],
        linestyle="--",
        linewidth=PLOT_CONFIG["linewidth"],
        label=f"{target_label}: {target_perplexity:.2f}",
        alpha=0.7
    )

    # Styling
    ax.set_xlabel("Training Steps", fontsize=PLOT_CONFIG["label_fontsize"], fontweight="bold")
    ax.set_ylabel("Validation Perplexity (↓ better)", fontsize=PLOT_CONFIG["label_fontsize"], fontweight="bold")

    ax.set_title(
        format_config_title("Validation Perplexity Comparison", lr, warmup_steps),
        fontsize=PLOT_CONFIG["title_fontsize"],
        fontweight="bold"
    )

    ax.legend(fontsize=PLOT_CONFIG["legend_fontsize"], loc="upper right")
    ax.grid(True, alpha=PLOT_CONFIG["grid_alpha"], linestyle="--")
    ax.tick_params(labelsize=PLOT_CONFIG["tick_fontsize"])

    plt.tight_layout()
    save_plot("validation_perplexity_comparison", output_dir)

    if show_plot:
        plt.show()
    plt.close()


# ============================================================================
# MAIN PLOTTING FUNCTION
# ============================================================================

def create_all_plots(
        esm_train_steps: List[int],
        esm_train_losses: List[float],
        esm_val_steps: List[int],
        esm_val_losses: List[float],
        mlstm_train_steps: List[int],
        mlstm_train_losses: List[float],
        mlstm_val_steps: List[int],
        mlstm_val_losses: List[float],
        lr: float,
        warmup_steps: int,
        baseline_perplexity: Optional[float] = None,
        output_dir: str = "./plots",
        show_plots: bool = True,
) -> None:
    """
    Generate all four plots in one function call (for ONE config at a time).
    Call this function 4 times for the 4 configs.

    Args:
        esm_train_steps: ESM training step numbers
        esm_train_losses: ESM training loss values
        esm_val_steps: ESM validation step numbers
        esm_val_losses: ESM validation loss values
        mlstm_train_steps: mLSTM training step numbers
        mlstm_train_losses: mLSTM training loss values
        mlstm_val_steps: mLSTM validation step numbers
        mlstm_val_losses: mLSTM validation loss values
        esm_val_perplexities: ESM validation perplexities (optional, calculated if None)
        mlstm_val_perplexities: mLSTM validation perplexities (optional, calculated if None)
        baseline_perplexity: Pretrained baseline perplexity for green line (optional)
        output_dir: Directory to save all plots
        show_plots: Whether to display plots
    """
    print("\n" + "=" * 70)
    print(f"GENERATING ALL PLOTS | LR={format_lr(lr)}, Warm-up={warmup_steps}")
    print("=" * 70 + "\n")

    # Plot 1: ESM Training
    print("📊 Creating Plot 1: ESM Training and Validation Loss...")
    plot_esm_training(
        esm_train_steps, esm_train_losses,
        esm_val_steps, esm_val_losses,
        lr=lr,
        warmup_steps=warmup_steps,
        output_dir=output_dir,
        show_plot=show_plots
    )

    # Plot 2: mLSTM Training
    print("\n📊 Creating Plot 2: mLSTM Training and Validation Loss...")
    plot_mlstm_training(
        mlstm_train_steps, mlstm_train_losses,
        mlstm_val_steps, mlstm_val_losses,
        lr=lr,
        warmup_steps=warmup_steps,
        output_dir=output_dir,
        show_plot=show_plots
    )

    # Plot 3: Validation Loss Comparison
    print("\n📊 Creating Plot 3: Validation Loss Comparison...")
    plot_validation_loss_comparison(
        esm_val_steps, esm_val_losses,
        mlstm_val_steps, mlstm_val_losses,
        lr=lr,
        warmup_steps=warmup_steps,
        output_dir=output_dir,
        show_plot=show_plots
    )

    # Plot 4: Validation Perplexity Comparison
    print("\n📊 Creating Plot 4: Validation Perplexity Comparison...")

    target_perplexity = baseline_perplexity if baseline_perplexity is not None else PAPER_TARGET_PERPLEXITY
    target_label = "ESM2-8M Pretrained" if baseline_perplexity is not None else "Paper Target"

    plot_validation_perplexity_comparison(
        esm_val_steps, esm_val_losses,
        mlstm_val_steps, mlstm_val_losses,
        lr=lr,
        warmup_steps=warmup_steps,
        target_perplexity=target_perplexity,
        output_dir=output_dir,
        show_plot=show_plots,
        target_label=target_label
    )

    print("\n" + "=" * 70)
    print("✅ ALL PLOTS GENERATED SUCCESSFULLY")
    print(f"📁 Saved to: {output_dir}")
    print("=" * 70 + "\n")

# ============================================================================
# EXAMPLE USAGE
# ============================================================================
#if __name__ == "__main__":

