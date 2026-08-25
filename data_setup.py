"""
Data setup for streaming protein sequence loading.

This module provides streaming data loaders that:
- Load sequences on-the-fly without exhausting RAM
- Support both ESM2 and mLSTM tokenization
- Apply masked language modeling (MLM) correctly
- Group sequences into token-budgeted batches
"""

import logging
import torch
from typing import Tuple
from transformers import EsmTokenizer
from DataLoader import StreamingProteinDataset, TokenBucketDataLoader
from data_utils import vocab_dict, vocab_size, MASK_ID

logger = logging.getLogger(__name__)


def make_streaming_loaders(
    train_fasta: str,
    val_fasta: str,
    model: str = "esm2",
    max_tokens_per_batch: int = 8192,
    mask_prob: float = 0.25,
    max_len: int = 512,
    num_workers: int = 2,
    tokenizer = None,
) -> Tuple[TokenBucketDataLoader, TokenBucketDataLoader]:
    """
    Create streaming data loaders for training and validation.
    
    Args:
        train_fasta: Path to training FASTA file
        val_fasta: Path to validation FASTA file
        model: "esm2" or "mlstm"
        max_tokens_per_batch: Maximum tokens per batch
        mask_prob: Probability of masking each token
        max_len: Maximum sequence length
        num_workers: Number of data loading workers
        tokenizer: Optional pre-loaded tokenizer (for ESM2)
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    logger.info(f"Creating streaming loaders for {model.upper()}")
    
    if model.lower() == "esm2":
        # ESM2 tokenizer
        if tokenizer is None:
            tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        
        def tokenize_fn(seq: str):
            return tokenizer(seq, add_special_tokens=False)["input_ids"]
        
        mask_id = tokenizer.mask_token_id
        task = "mlm"
        
        logger.info(f"ESM2 tokenizer loaded (vocab_size={tokenizer.vocab_size})")
    
    else:
        # mLSTM tokenizer
        mask_id = MASK_ID
        
        def tokenize_fn(seq: str):
            return [vocab_dict.get(aa, vocab_dict["<unk>"]) for aa in seq]
        
        task = "mlm"  # Both models use MLM for fair comparison
        
        logger.info(f"mLSTM tokenizer (vocab_size={vocab_size})")
    
    # Create datasets
    train_ds = StreamingProteinDataset(
        train_fasta,
        tokenize_fn,
        task=task,
        mask_token_id=mask_id,
        mask_prob=mask_prob,
        max_len=max_len,
        shuffle_window=1024,
        seed=123,
    )
    
    val_ds = StreamingProteinDataset(
        val_fasta,
        tokenize_fn,
        task=task,
        mask_token_id=mask_id,
        mask_prob=mask_prob,
        max_len=max_len,
        shuffle_window=1024,
        seed=456,
    )
    
    # Create loaders with token bucketing
    train_loader = TokenBucketDataLoader(
        train_ds,
        max_tokens_per_batch=max_tokens_per_batch,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    val_loader = TokenBucketDataLoader(
        val_ds,
        max_tokens_per_batch=max_tokens_per_batch,
        num_workers=max(1, num_workers // 2),
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    logger.info(f"Streaming loaders created successfully")
    logger.info(f"  Max tokens per batch: {max_tokens_per_batch:,}")
    logger.info(f"  Mask probability: {mask_prob}")
    logger.info(f"  Max sequence length: {max_len}")
    
    return train_loader, val_loader
