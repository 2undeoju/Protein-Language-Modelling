"""
UNIFIED Configuration for Protein Language Model Training

This single config handles:
1. Baseline mode: ESM2 vs mLSTM comparison (6-layer)
2. Layer exploration: Full depth comparison (1, 6, 12 layers)

IMPORTANT:
- Baseline configs are stored in BASELINE_CONFIGS with stable keys:
    lr1e-3_w1000, lr1e-3_w2000, lr4e-4_w1000, lr4e-4_w2000
- Backward compatibility:
    ESM2_CONFIG and MLSTM_CONFIG still exist and default to lr4e-4_w1000
"""

import os
import torch
from pathlib import Path
from typing import Dict, List, Tuple

# ============================================================================
# SMALL HELPERS
# ============================================================================

def format_lr(lr: float) -> str:
    """Format LR like 4e-4 instead of 4e-04."""
    s = f"{lr:.0e}"
    return s.replace("e-0", "e-").replace("e+0", "e+")


# ============================================================================
# GPU CONFIGURATION
# ============================================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("GPU CONFIGURATION")
print("=" * 70)
print(f"Using device: {DEVICE}")
print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print("=" * 70 + "\n")

# ============================================================================
# DATA PATHS
# ============================================================================
DATA_DIR = Path("../dataset/uniref50_split")
TRAIN_FASTA = DATA_DIR / "train.fasta"
VAL_FASTA = DATA_DIR / "val.fasta"

# ============================================================================
# BASE TRAINING CONFIGURATION (Shared)
# ============================================================================
BASE_TRAINING_CONFIG: Dict = {
    "vocab_size": 23,
    "train_fasta": str(TRAIN_FASTA),
    "val_fasta": str(VAL_FASTA),

    "total_updates": 10000,
    "batch_size": 12,
    "max_len": 512,
    "grad_accum_steps": 24,

    # Default hyperparams (overridden per experiment below)
    "lr": 4e-4,
    "warmup_steps": 1000,
    "betas": (0.99, 0.98),
    "eps": 1e-12,
    "weight_decay": 0.0,

    "grad_clip_norm": 0.5,
    "mask_prob": 0.25,

    "val_interval": 500,
    "val_steps": 100,

    "num_workers": 4,
    "max_tokens_per_batch": 12 * 512,
    "checkpoint_steps": [100, 500, 1000, 5000, 9000, 10000],
    "log_interval": 10,
}

# ============================================================================
# BASELINE MODEL TEMPLATES (6-layer comparison)
# ============================================================================
ESM2_TEMPLATE: Dict = {
    **BASE_TRAINING_CONFIG,
    "model_type": "esm2",
    "model_name": "facebook/esm2_t6_8M_UR50D",
    "remove_biases": True,
}

MLSTM_TEMPLATE: Dict = {
    **BASE_TRAINING_CONFIG,
    "model_type": "mlstm",
    "embed_dim": 128,
    "hidden_size": 320,
    "num_heads": 8,
    "chunk_size": 64,
    "bidirectional": True,
    "dropout": 0.0,
    "remove_biases": True,
    "num_layers": 6,
    "max_position_embeddings": 2048,
}

# ============================================================================
# BASELINE CONFIGS (4 matched experiments, stable keys)
# ============================================================================
BASELINE_SWEEPS: Dict[str, Dict] = {
    "lr4e-4_w1000": {"lr": 4e-4, "warmup_steps": 1000},
    "lr4e-4_w2000": {"lr": 4e-4, "warmup_steps": 2000},
    "lr1e-3_w1000": {"lr": 1e-3, "warmup_steps": 1000},
    "lr1e-3_w2000": {"lr": 1e-3, "warmup_steps": 2000},
}

BASELINE_CONFIGS: Dict[str, Dict[str, Dict]] = {}

for key, hp in BASELINE_SWEEPS.items():
    lr = hp["lr"]
    warmup = hp["warmup_steps"]

    BASELINE_CONFIGS[key] = {
        "esm2": {
            **ESM2_TEMPLATE,
            **hp,
            "experiment_name": f"esm2_lr{format_lr(lr)}_w{warmup}",
        },
        "mlstm": {
            **MLSTM_TEMPLATE,
            **hp,
            "experiment_name": f"mlstm_lr{format_lr(lr)}_w{warmup}",
        },
    }

# Backward compatibility: keep your old imports working
ESM2_CONFIG: Dict = BASELINE_CONFIGS["lr4e-4_w1000"]["esm2"]
MLSTM_CONFIG: Dict = BASELINE_CONFIGS["lr4e-4_w1000"]["mlstm"]

# ============================================================================
# LAYER EXPLORATION: Depth Variants (1, 6, 12 layers)
# ============================================================================
DEPTH_VARIANTS = {
    "mlstm_1layer": {
        "num_layers": 1,
        "hidden_size": 672,
        "description": "1-layer mLSTM (WIDE)",
    },
    "mlstm_6layer": {
        "num_layers": 6,
        "hidden_size": 320,
        "description": "6-layer mLSTM (BALANCED)",
    },
    "mlstm_12layer": {
        "num_layers": 12,
        "hidden_size": 240,
        "description": "12-layer mLSTM (DEEP)",
    },
}

# ============================================================================
# LAYER EXPLORATION: Hyperparameter Sweeps (your original naming kept)
# ============================================================================
HYPERPARAMETER_SWEEPS = {
    "lr1e-3_w1k": {"lr": 1e-3, "warmup_steps": 1000},
    "lr1e-3_w2k": {"lr": 1e-3, "warmup_steps": 2000},
    "lr4e-4_w1k": {"lr": 4e-4, "warmup_steps": 1000},
    "lr4e-4_w2k": {"lr": 4e-4, "warmup_steps": 2000},
}

# ============================================================================
# LAYER EXPLORATION: Helper Functions (kept same behavior)
# ============================================================================

def get_layer_config(depth_variant: str, hyperparam_sweep: str) -> Dict:
    """
    Get configuration for layer exploration experiment.

    Args:
        depth_variant: "mlstm_1layer", "mlstm_6layer", "mlstm_12layer"
        hyperparam_sweep: "lr1e-3_w1k", "lr1e-3_w2k", "lr4e-4_w1k", "lr4e-4_w2k"

    Returns:
        Complete configuration dictionary
    """
    if depth_variant not in DEPTH_VARIANTS:
        raise ValueError(f"Unknown depth: {depth_variant}")
    if hyperparam_sweep not in HYPERPARAMETER_SWEEPS:
        raise ValueError(f"Unknown hyperparam: {hyperparam_sweep}")

    # Start from base mLSTM template (matches your original intent)
    config = {
        **BASE_TRAINING_CONFIG,
        "model_type": "mlstm",
        "embed_dim": 128,
        "num_heads": 8,
        "chunk_size": 64,
        "bidirectional": True,
        "dropout": 0.0,
        "remove_biases": True,
        "max_position_embeddings": 2048,
    }

    # Add depth variant
    config.update(DEPTH_VARIANTS[depth_variant])

    # Add hyperparameters
    config.update(HYPERPARAMETER_SWEEPS[hyperparam_sweep])

    # Add metadata
    config["experiment_name"] = f"{depth_variant}_{hyperparam_sweep}"
    config["depth_variant"] = depth_variant
    config["hyperparam_sweep"] = hyperparam_sweep

    # File suffix (kept for your tooling)
    lr_str = format_lr(config["lr"])
    config["suffix"] = f"_L{config['num_layers']}_{lr_str}_w{config['warmup_steps']}"

    return config


def get_esm2_baseline_config(hyperparam_sweep: str) -> Dict:
    """Get ESM2 baseline config for layer exploration."""
    if hyperparam_sweep not in HYPERPARAMETER_SWEEPS:
        raise ValueError(f"Unknown hyperparam: {hyperparam_sweep}")

    config = {**BASE_TRAINING_CONFIG}
    config.update(HYPERPARAMETER_SWEEPS[hyperparam_sweep])
    config.update({
        "model_name": "facebook/esm2_t6_8M_UR50D",
        "model_type": "esm2",
        "remove_biases": True,
        "experiment_name": f"esm2_baseline_{hyperparam_sweep}",
        "depth_variant": "esm2_baseline",
        "hyperparam_sweep": hyperparam_sweep,
    })

    lr_str = format_lr(config["lr"])
    config["suffix"] = f"_esm2_{lr_str}_w{config['warmup_steps']}"

    return config


def get_all_layer_experiments() -> List[Tuple[str, str, Dict]]:
    """Get all 12 mLSTM layer experiments."""
    experiments = []
    for depth in DEPTH_VARIANTS.keys():
        for hyperparam in HYPERPARAMETER_SWEEPS.keys():
            cfg = get_layer_config(depth, hyperparam)
            experiments.append((depth, hyperparam, cfg))
    return experiments


def get_all_esm2_baselines() -> List[Dict]:
    """Get all 4 ESM2 baseline experiments."""
    return [get_esm2_baseline_config(hp) for hp in HYPERPARAMETER_SWEEPS.keys()]

# ============================================================================
# WANDB & STABILITY (unchanged in meaning)
# ============================================================================
WANDB_PROJECT = "protein_cramming_unified"
WANDB_ENTITY = None

STABILITY_CONFIG = {
    "check_nan": True,
    "max_loss_threshold": 15.0,
    "min_masked_tokens": 8,
    "grad_norm_threshold": 50.0,
}

# ============================================================================
# GPU OPTIMIZATIONS
# ============================================================================
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("medium")
    torch.cuda.empty_cache()
    print("✅ GPU optimizations enabled\n")

# ============================================================================
# CONFIGURATION SUMMARY (prints baseline keys clearly)
# ============================================================================
print("=" * 70)
print("UNIFIED CONFIGURATION LOADED")
print("=" * 70)
print("Available modes:")
print("  1. Baseline: ESM2 + mLSTM (6-layer)")
print("     Baseline keys in BASELINE_CONFIGS:")
for k in BASELINE_CONFIGS.keys():
    esm_name = BASELINE_CONFIGS[k]["esm2"]["experiment_name"]
    ml_name = BASELINE_CONFIGS[k]["mlstm"]["experiment_name"]
    print(f"       • {k}: {esm_name}  vs  {ml_name}")
print("  2. Layer exploration: 16 experiments")
print("     - 4 ESM2 baselines")
print("     - 12 mLSTM (1/6/12 layers × 4 hyperparams)")
print("=" * 70 + "\n")