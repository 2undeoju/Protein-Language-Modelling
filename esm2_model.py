"""
ESM2 model loading and initialization - FIXED FOR FROM-SCRATCH TRAINING.

Handles:
- Loading ESM2 architecture with RANDOM initialization (not pretrained)
- Bias removal for memory efficiency (cramming-style)
- Device management
- Initialization verification

CRITICAL CHANGE: Models are now initialized randomly, not from pretrained weights.
This ensures fair comparison with mLSTM in the cramming methodology.
"""

import os
import logging
import math
import torch
import torch.nn as nn
from transformers import EsmTokenizer, EsmForMaskedLM, AutoConfig

from config import ESM2_CONFIG, DEVICE  # keeps your original behavior

logger = logging.getLogger(__name__)

# Model configuration (kept same logic)
MODEL_NAME = os.getenv("ESM2_MODEL_NAME", ESM2_CONFIG.get("model_name", "facebook/esm2_t6_8M_UR50D"))

logger.info(f"Using device: {DEVICE}")
logger.info(f"Using model architecture: {MODEL_NAME}")


def load_tokenizer(model_name: str = MODEL_NAME) -> EsmTokenizer:
    """
    Load ESM2 tokenizer.

    Returns:
        EsmTokenizer instance.
    """
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    logger.info(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
    return tokenizer


def load_model_from_scratch(model_name: str = MODEL_NAME) -> EsmForMaskedLM:
    """
    Load ESM2 architecture with RANDOM initialization (not pretrained).

    Returns:
        EsmForMaskedLM model instance with random weights.
    """
    logger.info("=" * 70)
    logger.info("INITIALIZING ESM2 FROM SCRATCH (RANDOM WEIGHTS)")
    logger.info("=" * 70)
    logger.info(f"Loading architecture: {model_name}")
    logger.info("⚠️  NOT loading pretrained weights - training from scratch")

    config = AutoConfig.from_pretrained(model_name)
    model = EsmForMaskedLM(config)  # RANDOM init

    logger.info("✓ Model initialized with RANDOM weights")
    logger.info(f"  Architecture: {model_name}")
    logger.info("  Config loaded, weights randomized")

    return model


def remove_biases(model: torch.nn.Module) -> None:
    """
    Remove bias parameters from all linear layers.

    This reduces memory footprint and is common in cramming-style training
    where model capacity is limited.

    Args:
        model: Model to modify in-place.
    """
    logger.info("Removing biases from all linear layers")
    removed_count = 0

    for _, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.register_parameter("bias", None)
            removed_count += 1

    logger.info(f"Removed biases from {removed_count} linear layers")


@torch.no_grad()
def verify_random_initialization(model: torch.nn.Module, tokenizer: EsmTokenizer, vocab_size: int) -> bool:
    """
    Verify that model is randomly initialized by checking initial loss.

    A randomly initialized model should have loss close to -log(1/vocab_size).
    For ESM2 with 33 tokens: expected loss ≈ 3.50

    Args:
        model: Model to verify
        tokenizer: Tokenizer for creating test batch
        vocab_size: Vocabulary size

    Returns:
        bool: True if model appears randomly initialized
    """
    logger.info("\n" + "=" * 70)
    logger.info("VERIFYING RANDOM INITIALIZATION")
    logger.info("=" * 70)

    model.eval()

    batch_size = 4
    seq_len = 128

    # Random input
    input_ids = torch.randint(4, vocab_size - 1, (batch_size, seq_len), device=DEVICE)
    labels = input_ids.clone()

    # Mask 25%
    mask_token_id = tokenizer.mask_token_id
    mask = torch.rand(batch_size, seq_len, device=DEVICE) < 0.25
    input_ids[mask] = mask_token_id

    outputs = model(input_ids=input_ids, labels=labels)
    loss = float(outputs.loss)

    expected_loss = math.log(vocab_size)
    difference = abs(loss - expected_loss)

    logger.info(f"Expected initial loss (random): {expected_loss:.3f}")
    logger.info(f"Observed initial loss: {loss:.3f}")
    logger.info(f"Difference: {difference:.3f}")

    # Check if loss is in reasonable range for random initialization
    # Allow some variance (±0.5) due to small sample size
    is_random = difference < 0.5

    if is_random:
        logger.info("✓ Model appears to be randomly initialized")
        logger.info("  Loss is consistent with random guessing")
    else:
        logger.warning("❌ WARNING: Model may NOT be randomly initialized!")
        logger.warning("  Loss is too far from expected random value")
        logger.warning("  This could indicate:")
        logger.warning("    - Pretrained weights were loaded")
        logger.warning("    - Checkpoint was resumed")
        logger.warning("    - Incorrect loss calculation")

    logger.info("=" * 70 + "\n")

    model.train()
    return is_random


def initialize_model(cfg: dict = None) -> EsmForMaskedLM:
    """
    Complete initialization pipeline: load model, remove biases, verify, move to device.

    Returns:
        Initialized ESM2 model ready for training (from scratch).
    """
    # Load model with random weights
    cfg = cfg or ESM2_CONFIG
    model_name = cfg.get("model_name", MODEL_NAME)

    model = load_model_from_scratch(model_name)

    # Remove biases for memory efficiency
    if cfg.get("remove_biases", True):
        remove_biases(model)

    # Move to device
    model.to(DEVICE)
    logger.info(f"Model moved to {DEVICE}")

    # Log model stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Verify random initialization
    tok = load_tokenizer(model_name)
    is_random = verify_random_initialization(model, tok, vocab_size=tok.vocab_size)

    if not is_random:
        logger.error("=" * 70)
        logger.error("CRITICAL: Model initialization verification FAILED!")
        logger.error("The model does not appear to be randomly initialized.")
        logger.error("=" * 70)
        raise ValueError("Model initialization verification failed - not training from scratch")

    return model


# ----------------------------------------------------------------------------
# CLEAN ADDITION: builder function (so you can create models for any baseline key)
# ----------------------------------------------------------------------------
def build_esm2_model(cfg: dict) -> EsmForMaskedLM:
    """
    Clean factory: build an ESM2 model from a config dict.
    Keeps the same exact initialization logic as before.
    """
    return initialize_model(cfg)


# ----------------------------------------------------------------------------
# BACKWARD COMPATIBILITY: keep the original behavior (global tokenizer + model)
# so your existing training code remains consistent.
# ----------------------------------------------------------------------------
logger.info("\n" + "=" * 70)
logger.info("CREATING ESM2 MODEL FOR FROM-SCRATCH TRAINING")
logger.info("=" * 70 + "\n")

tokenizer = load_tokenizer(MODEL_NAME)
model = initialize_model(ESM2_CONFIG)

logger.info("\n" + "=" * 70)
logger.info("ESM2 MODEL READY FOR TRAINING")
logger.info("=" * 70)
logger.info("✓ Weights initialized randomly (NOT pretrained)")
logger.info("✓ Biases removed (cramming-style)")
logger.info("✓ Random initialization verified")
logger.info("✓ Ready for fair comparison with mLSTM")
logger.info("=" * 70 + "\n")



