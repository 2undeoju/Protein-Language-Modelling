"""
Generate comparison plots after both ESM2 and mLSTM training complete.

This script creates comparison plots for both training configurations:
  - Config 1: LR=4e-4, Warmup=1k
  - Config 2: LR=1e-3, Warmup=2k

Usage:
    python comparison_plots.py

Requirements:
    - esm2_lr4e-4_w1000_training_data.pkl
    - mlstm_lr4e-4_w1000_training_data.pkl
    - esm2_lr1e-3_w2000_training_data.pkl
    - mlstm_lr1e-3_w2000_training_data.pkl
    - esm2_8m_pretrained_baseline.pkl (optional)
    - plot.py (in same directory)
"""

import pickle
import sys
import math
from pathlib import Path
from plot import create_all_plots


# ============================================================================
# CONFIGURATION - CLEAN NAMING CONVENTION
# ============================================================================
CONFIGS = [
    {
        "name": "LR=4e-4, Warmup=1000",
        "lr": 4e-4,
        "warmup": 1000,
        "esm_file": "training_data/esm2_lr4e-4_w1000_training_data.pkl",
        "mlstm_file": "training_data/mlstm_lr4e-4_w1000_training_data.pkl",
        "output_dir": "./plots_lr4e-4_w1000",
    },
    {
        "name": "LR=4e-4, Warmup=2000",
        "lr": 4e-4,
        "warmup": 2000,
        "esm_file": "training_data/esm2_lr4e-4_w2000_training_data.pkl",
        "mlstm_file": "training_data/mlstm_lr4e-4_w2000_training_data.pkl",
        "output_dir": "./plots_lr4e-4_w2000",
    },
    {
        "name": "LR=1e-3, Warmup=1000",
        "lr": 1e-3,
        "warmup": 1000,
        "esm_file": "training_data/esm2_lr1e-3_w1000_training_data.pkl",
        "mlstm_file": "training_data/mlstm_lr1e-3_w1000_training_data.pkl",
        "output_dir": "./plots_lr1e-3_w1000",
    },
    {
        "name": "LR=1e-3, Warmup=2000",
        "lr": 1e-3,
        "warmup": 2000,
        "esm_file": "training_data/esm2_lr1e-3_w2000_training_data.pkl",
        "mlstm_file": "training_data/mlstm_lr1e-3_w2000_training_data.pkl",
        "output_dir": "./plots_lr1e-3_w2000",
    },
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def load_baseline(filename: str = "training_data/esm2_8m_pretrained_baseline.pkl"):
    """Load pretrained baseline perplexity."""
    try:
        with open(filename, "rb") as f:
            baseline = pickle.load(f)
        baseline_perplexity = baseline["validation_perplexity"]
        print(f"✅ Loaded baseline: {filename}")
        print(f"   Pretrained ESM2-8M (6-layer) perplexity: {baseline_perplexity:.2f}")
        return baseline_perplexity
    except FileNotFoundError:
        print(f"⚠️  Warning: {filename} not found!")
        print(f"   Continuing without baseline reference line...")
        return None
    except Exception as e:
        print(f"⚠️  Warning: Could not load baseline: {e}")
        return None


def load_training_data(filename: str):
    """Load training data from pickle file."""
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
        print(f"✅ Loaded: {filename}")
        return data
    except FileNotFoundError:
        print(f"⚠️  Warning: {filename} not found!")
        return None
    except Exception as e:
        print(f"❌ Error loading {filename}: {e}")
        return None


def validate_data(data: dict, model_name: str) -> bool:
    """Validate that required keys exist in data dictionary."""
    if data is None:
        return False

    required_keys = ["train_steps", "train_losses", "val_steps", "val_losses"]

    for key in required_keys:
        if key not in data:
            print(f"❌ Error: '{key}' not found in {model_name} data!")
            return False

    # Calculate perplexities if not present
    if "train_perplexities" not in data:
        data["train_perplexities"] = [math.exp(loss) for loss in data["train_losses"]]

    if "val_perplexities" not in data:
        data["val_perplexities"] = [math.exp(loss) for loss in data["val_losses"]]

    print(f"✅ {model_name} validated:")
    print(f"   - Training points: {len(data['train_steps'])}")
    print(f"   - Validation points: {len(data['val_steps'])}")
    if "final_val_perplexity" in data:
        print(f"   - Final perplexity: {data['final_val_perplexity']:.2f}")

    return True


def generate_plots_for_config(config: dict, baseline_perplexity: float) -> bool:
    """Generate comparison plots for a single configuration."""
    print("\n" + "="*70)
    print(f"PROCESSING: {config['name']}")
    print("="*70 + "\n")

    # Load ESM2 data
    print(f"📂 Loading ESM2 data...")
    esm_data = load_training_data(config["esm_file"])
    if not validate_data(esm_data, f"ESM2 ({config['name']})"):
        print(f"⚠️  Skipping {config['name']} - ESM2 data not available")
        return False
    print()

    # Load mLSTM data
    print(f"📂 Loading mLSTM data...")
    mlstm_data = load_training_data(config["mlstm_file"])
    if not validate_data(mlstm_data, f"mLSTM ({config['name']})"):
        print(f"⚠️  Skipping {config['name']} - mLSTM data not available")
        return False
    print()

    # Generate plots
    print(f"📊 Generating plots...")
    print(f"   Output directory: {config['output_dir']}")
    print("-" * 70 + "\n")

    create_all_plots(
        esm_train_steps=esm_data["train_steps"],
        esm_train_losses=esm_data["train_losses"],
        esm_val_steps=esm_data["val_steps"],
        esm_val_losses=esm_data["val_losses"],

        mlstm_train_steps=mlstm_data["train_steps"],
        mlstm_train_losses=mlstm_data["train_losses"],
        mlstm_val_steps=mlstm_data["val_steps"],
        mlstm_val_losses=mlstm_data["val_losses"],

        lr=config["lr"],
        warmup_steps=config["warmup"],

        baseline_perplexity=baseline_perplexity,
        output_dir=config["output_dir"],
        show_plots=False,
    )


    # Summary
    print("\n" + "="*70)
    print(f"✅ SUCCESS! Plots generated for {config['name']}")
    print("="*70)
    print(f"\n📁 Plots saved to: {config['output_dir']}/")
    print("\nGenerated files:")
    print("  1. esm_training_validation_curve.pdf/png")
    print("  2. mlstm_training_validation_curve.pdf/png")
    print("  3. validation_loss_comparison.pdf/png")
    print("  4. validation_perplexity_comparison.pdf/png")

    # Perplexity comparison
    if baseline_perplexity is not None:
        print("\n" + "-"*70)
        print(f"PERPLEXITY COMPARISON ({config['name']})")
        print("-"*70)
        print(f"Pretrained ESM2-8M (6-layer baseline): {baseline_perplexity:.2f}")

        if "final_val_perplexity" in esm_data:
            esm_final = esm_data["final_val_perplexity"]
            improvement = ((baseline_perplexity - esm_final) / baseline_perplexity) * 100
            print(f"ESM2~8M (6-layer, trained):            {esm_final:.2f} ({improvement:+.1f}% vs baseline)")

        if "final_val_perplexity" in mlstm_data:
            mlstm_final = mlstm_data["final_val_perplexity"]
            improvement = ((baseline_perplexity - mlstm_final) / baseline_perplexity) * 100
            print(f"mLSTM-8M (6-layer, trained):           {mlstm_final:.2f} ({improvement:+.1f}% vs baseline)")

        print("-"*70)

    return True


# ============================================================================
# MAIN FUNCTION
# ============================================================================
def main():
    """Main function: Generate comparison plots for all configurations."""
    print("\n" + "="*70)
    print("COMPARISON PLOTS GENERATOR")
    print("="*70 + "\n")

    # Check if plot.py exists
    if not Path("plot.py").exists():
        print("❌ Error: plot.py not found in current directory!")
        print("   Please copy plot.py to this directory first.")
        sys.exit(1)

    # Load baseline
    print("📂 Loading pretrained baseline...")
    baseline_perplexity = load_baseline()  # uses training_data/ default
    print()

    # Process each configuration
    successful = []
    skipped = []

    for config in CONFIGS:
        success = generate_plots_for_config(config, baseline_perplexity)
        if success:
            successful.append(config["name"])
        else:
            skipped.append(config["name"])
        print("\n" + "="*70 + "\n")

    # Final summary
    print("="*70)
    print("FINAL SUMMARY")
    print("="*70)

    if successful:
        print(f"\n✅ Successfully generated plots for {len(successful)} configuration(s):")
        for name in successful:
            print(f"   - {name}")

    if skipped:
        print(f"\n⚠️  Skipped {len(skipped)} configuration(s):")
        for name in skipped:
            print(f"   - {name}")
        print("\n💡 Tip: Complete training for missing configurations")

    print("\n" + "="*70 + "\n")

    if not successful:
        print("❌ No plots were generated. Please check that training data files exist.")
        sys.exit(1)


if __name__ == "__main__":
    main()