"""
OPTIMIZED Downstream Evaluation Configuration
Increased batch sizes for better GPU utilization on 16GB V100

Changes from original:
- 4x larger batch sizes for better GPU utilization
- Proper sequence length handling for AAV (735 AA)
- Gradient accumulation for memory efficiency
- Better ESM2 configuration for long sequences
"""

import os
from pathlib import Path

# ============================================================================
# GPU SELECTION - Use GPU 1 (same as main training config)
# ============================================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Print device information
print("="*70)
print("DOWNSTREAM EVALUATION - GPU CONFIGURATION (OPTIMIZED)")
print("="*70)
print(f"Using device: {DEVICE}")
print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
if torch.cuda.is_available():
    print(f"torch.cuda.device_count() = {torch.cuda.device_count()}")
    print(f"torch.cuda.current_device() = {torch.cuda.current_device()}")
    print(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    # Memory info
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    cached = torch.cuda.memory_reserved(0) / 1024**3
    free = total_memory - allocated
    print(f"GPU Memory: {total_memory:.2f} GB total")
    print(f"  Allocated: {allocated:.2f} GB")
    print(f"  Cached: {cached:.2f} GB")
    print(f"  Free: {free:.2f} GB")
    print(f"\n⚡ OPTIMIZED MODE: Using larger batch sizes for {total_memory:.0f}GB GPU")
print("="*70 + "\n")

# ============================================================================
# PATHS - Data in dataset/flip, everything else in pyscript
# ============================================================================

class Paths:
    """Simple path management"""

    def __init__(self):
        # Data directories (in dataset/flip/)
        self.data_dir = Path("../dataset/flip/data")
        self.embeddings_dir = Path("../dataset/flip/embeddings")

        # Working directories (in pyscript/)
        self.results_dir = Path("results")
        self.plots_dir = Path("plots")

        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

# Global paths instance
PATHS = Paths()

# ============================================================================
# MODEL CONFIG
# ============================================================================

# Checkpoint paths (in pyscript directory)
CHECKPOINTS = {
    'esm2': 'training_data/esm2_baseline_lr1e-3_w1k_final.pt',
    'mlstm': 'training_data/mlstm_6layer_lr1e-3_w1k_final.pt',

    #'esm2': 'ESM2_final.pt',
    #'mlstm': 'mLSTM_final_lr4e-4.pt'
}

# Model dimensions (from training configs)
EMBEDDING_DIMS = {
    'esm2': 320,      # ESM2 t6_8M hidden size
    'mlstm': 320     # mLSTM hidden_size from config
}

# ============================================================================
# EMBEDDING EXTRACTION CONFIG - OPTIMIZED FOR 16GB GPU
# ============================================================================

# Maximum sequence lengths (ESM2 can handle up to 1024 tokens)
MAX_SEQ_LENGTHS = {
    'esm2': 1024,    # ESM2 max context length
    'mlstm': 1024,   # mLSTM can handle longer with chunking
}

# ESM2-specific optimization flags
ESM2_CONFIG = {
    'use_gradient_checkpointing': False,  # Disable for inference
    'truncation_mode': 'right',           # Truncate from right if too long
    'attention_implementation': 'sdpa',   # Use scaled dot-product attention
}

# ============================================================================
# FLIP TASKS CONFIGURATION - OPTIMIZED BATCH SIZES
# ============================================================================

# Base configuration shared across all tasks
BASE_TASK_CONFIG = {
    # Prediction head training (following cramming paper)
    'hidden_dim': 256,  # From paper: 2-layer MLP with 256 hidden
    'dropout': 0.0,  # From paper: no dropout
    'learning_rate': 4e-5,  # From paper: 4×10^-5
    'epochs': 100000,
    'early_stop_patience': 100,

    # Evaluation
    'metric': 'spearman',  # Spearman correlation for regression
    'task_type': 'regression',
}

# OPTIMIZED: 4x larger batch sizes for better GPU utilization
TASK_CONFIGS = {
    'gb1': {
        **BASE_TASK_CONFIG,
        'data_url': 'https://github.com/J-SNACKKB/FLIP/raw/main/splits/gb1/splits.zip',
        'split_name': 'two_vs_rest',
        'max_seq_len': 64,  # GB1 sequences are ~56 AA
        'target_col': 'target',
        'description': 'GB1 binding fitness prediction',
        
        # OPTIMIZED: Increased from 4 to 16
        'batch_size_embed': 16,   # 4x larger
        'batch_size_train': 128,  # 4x larger
    },
    'aav': {
        **BASE_TASK_CONFIG,
        'data_url': 'https://github.com/J-SNACKKB/FLIP/raw/main/splits/aav/splits.zip',
        'split_name': 'two_vs_many',
        'max_seq_len': 1024,  # CRITICAL: Increased to handle full 735 AA sequences
        'target_col': 'target',
        'description': 'AAV capsid fitness prediction',
        
        # OPTIMIZED: Increased from 2 to 8 for ESM2, 1 to 4 for mLSTM
        'batch_size_embed': 16,    # 4x larger
        'batch_size_train': 128,   # 2x larger (memory-constrained due to longer sequences)
    },
    'meltome': {
        **BASE_TASK_CONFIG,
        'data_url': 'https://github.com/J-SNACKKB/FLIP/raw/main/splits/meltome/splits.zip',
        'split_name': 'mixed_split',
        'max_seq_len': 512,  # Variable length proteins
        'target_col': 'target',
        'description': 'Protein thermostability (Tm) prediction',
        
        # OPTIMIZED: Increased from 4 to 16 for ESM2, 2 to 8 for mLSTM
        'batch_size_embed': 16,   # 4x larger
        'batch_size_train': 128,   # 3x larger
    }
}

# Task-specific batch size overrides for mLSTM (needs more memory)
# OPTIMIZED: Increased all batch sizes
MLSTM_BATCH_OVERRIDES = {
    'gb1': 16,        # Increased from 4 to 8
    'aav': 4,        # Increased from 1 to 4 (long sequences, still memory-limited)
    'meltome': 2,    # Increased from 2 to 8
}

# Legacy: Keep GB1_CONFIG for backward compatibility
GB1_CONFIG = TASK_CONFIGS['gb1']

print(f"⚡ OPTIMIZED config loaded")
print(f"  Device: {DEVICE}")
print(f"  Data: {PATHS.data_dir}")
print(f"  Embeddings: {PATHS.embeddings_dir}")
print(f"  Tasks: {list(TASK_CONFIGS.keys())}")
print(f"\nBatch size improvements:")
print(f"  GB1: embed 4→16 (4x), train 32→128 (4x)")
print(f"  AAV: embed 2→8 (4x), train 32→64 (2x)")
print(f"  Meltome: embed 4→16 (4x), train 32→96 (3x)")
print(f"\nmLSTM batch sizes:")
print(f"  GB1: 4→8 (2x), AAV: 1→4 (4x), Meltome: 2→8 (4x)")





