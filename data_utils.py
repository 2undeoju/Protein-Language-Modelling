import logging
import datetime
import torch
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm
from Bio import SeqIO
from torch.optim import AdamW


logger = logging.getLogger(__name__)

# ============================================================================
# VOCABULARY & TOKENIZATION
# ============================================================================

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")  # 20 standard amino acids
SPECIAL_TOKENS = ["<pad>", "<unk>", "<mask>"]
vocab = SPECIAL_TOKENS + AMINO_ACIDS
vocab_dict = {aa: i for i, aa in enumerate(vocab)}
vocab_size = len(vocab)

PAD_ID = vocab_dict["<pad>"]      # 0
UNK_ID = vocab_dict["<unk>"]      # 1
MASK_ID = vocab_dict["<mask>"]    # 2

logger.info(
    f"Initialized mLSTM vocabulary: {vocab_size} tokens "
    f"({len(SPECIAL_TOKENS)} special + {len(AMINO_ACIDS)} amino acids)"
)


def tokenize_mlstm(seq: str) -> list:
    """
    Convert protein sequence to token IDs using mLSTM vocabulary.

    Unknown amino acids map to <unk> (token_id=1).

    Args:
        seq: Protein sequence string.

    Returns:
        List of token IDs.

    Raises:
        AssertionError: If any token is out of valid range.
    """
    tokens = [vocab_dict.get(aa, UNK_ID) for aa in seq]

    # Sanity check
    assert all(0 <= t < vocab_size for t in tokens), (
        f"Token out of range! min={min(tokens)}, max={max(tokens)}, vocab_size={vocab_size}"
    )

    return tokens


# ============================================================================
# ESM2 TOKENIZATION (BATCH MODE)
# ============================================================================

def tokenize_esm(
        sequences: list,
        tokenizer,
        mask_prob: float = 0.25,
) -> tuple:
    """
    Batch tokenize sequences with ESM tokenizer and apply MLM masking.

    This is useful for offline preprocessing or non-streaming scenarios.
    For streaming datasets, use the per-sequence tokenizer in data_setup.py instead.

    Args:
        sequences: List of protein sequence strings.
        tokenizer: Huggingface tokenizer (e.g., EsmTokenizer).
        mask_prob: Probability of masking each token.

    Returns:
        Tuple of (input_ids_list, label_ids_list) where:
        - input_ids_list: List of masked input tensors
        - label_ids_list: List of label tensors (original tokens at masked positions)
    """
    import random

    input_ids_list = []
    label_ids_list = []

    logger.info(f"Tokenizing {len(sequences)} sequences with ESM tokenizer")

    # Batch tokenize
    batch_encoding = tokenizer(
        sequences,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512,
    )

    all_input_ids = batch_encoding["input_ids"]

    # Apply masking
    for tokens in tqdm(all_input_ids, desc="Applying MLM masking", dynamic_ncols=True, leave=False):
        tokens = tokens.tolist()
        labels = [-100] * len(tokens)
        masked_tokens = tokens.copy()

        # Mask interior tokens (skip special tokens at boundaries)
        for i in range(1, len(tokens) - 1):
            if random.random() < mask_prob:
                labels[i] = tokens[i]
                masked_tokens[i] = tokenizer.mask_token_id

        # Clamp labels to valid range
        labels = [
            label if (label == -100 or 0 <= label < tokenizer.vocab_size) else -100
            for label in labels
        ]

        input_ids_list.append(torch.tensor(masked_tokens))
        label_ids_list.append(torch.tensor(labels))

    logger.info(f"Tokenization complete: processed {len(input_ids_list)} sequences")
    return input_ids_list, label_ids_list


# ============================================================================
# SEQUENCE LOADING
# ============================================================================

def load_sequences(fasta_file: str) -> list:
    """
    Load all sequences from a FASTA file.

    Args:
        fasta_file: Path to FASTA file.

    Returns:
        List of sequence strings.
    """
    sequences = [str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")]
    logger.info(f"Loaded {len(sequences)} sequences from {fasta_file}")
    return sequences


# ============================================================================
# OPTIMIZER & SCHEDULER
# ============================================================================

def get_optimizer_and_scheduler(
        model: torch.nn.Module,
        total_steps: int = 50000,
        warmup_steps: int = 1000,
        lr: float = 1e-3,
) -> tuple:
    """
    Create AdamW optimizer and learning rate scheduler with warmup + decay.

    Learning rate schedule:
    - Warmup phase (0 to warmup_steps): Linear increase from 0 to lr
    - Decay phase (warmup_steps to total_steps): Linear decay from lr to 0

    Args:
        model: Model to optimize.
        total_steps: Total training steps.
        warmup_steps: Number of warmup steps.
        lr: Peak learning rate.

    Returns:
        Tuple of (optimizer, scheduler).
    """
    # Ensure positive steps
    total_steps = max(int(total_steps), 1)
    warmup_steps = max(int(warmup_steps), 1)

    # AdamW optimizer with aggressive bias/variance correction
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.99, 0.98),
        eps=1e-12,
    )

    # ✅ FIXED: Learning rate schedule with proper warmup + linear decay
    def lr_lambda(step: int) -> float:
        """
        Compute LR multiplier for current step.
        
        Returns a multiplier in [0, 1] that will be applied to the base learning rate.
        """
        if step < warmup_steps:
            # Warmup phase: linear increase from 0 to 1
            return (step + 1) / warmup_steps
        else:
            # Decay phase: linear decrease from 1 to 0
            decay_steps = total_steps - warmup_steps
            return max(0.0, 1.0 - (step - warmup_steps) / decay_steps)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    logger.info(
        f"Optimizer: AdamW(lr={lr}, betas=(0.99, 0.98), eps={1e-12})"
    )
    logger.info(
        f"Scheduler: Warmup for {warmup_steps} steps, then linear decay to 0 "
        f"over {total_steps - warmup_steps} steps (total: {total_steps} steps)"
    )

    return optimizer, scheduler


# ============================================================================
# WANDB INTEGRATION
# ============================================================================

def init_wandb_run(
        project: str,
        model_name: str,
        config: dict = None,
        run_name: str = None,
        run_type: str = None,
        tags: list = None,
) -> wandb.run:
    """
    Initialize a Weights & Biases run with automatic naming.

    Either provide explicit run_name or run_type (which generates a name).

    Args:
        project: Wandb project name.
        model_name: Name of the model (e.g., "ESM2", "mLSTM").
        config: Configuration dict to log.
        run_name: Explicit run name. If provided, run_type is ignored.
        run_type: Type descriptor (e.g., "training", "eval") for auto-generated name.
        tags: Additional tags (model_name is always the first tag).

    Returns:
        Initialized wandb run object.

    Raises:
        ValueError: If neither run_name nor run_type is provided.
    """
    # Generate run name if needed
    if run_type and not run_name:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{model_name.lower()}_{run_type}_{timestamp}"
    elif not run_name:
        raise ValueError("Must provide either run_name or run_type")

    # Build tag list
    combined_tags = [model_name] + (tags if tags else [])

    # Initialize run
    run = wandb.init(
        project=project,
        name=run_name,
        config=config,
        reinit=True,
        save_code=True,
        tags=combined_tags,
    )

    logger.info(f"Initialized wandb run: {run_name} (project={project})")
    return run


def log_batch_loss(
        run: wandb.run,
        loss: float,
        batch_idx: int,
) -> None:
    """
    Log batch-level loss and index to wandb.

    Args:
        run: Wandb run object.
        loss: Loss value for this batch.
        batch_idx: Batch index.
    """
    run.log({"loss": loss, "batch_idx": batch_idx})


def log_metrics(
        step: int,
        loss: float,
        model_name: str = None,
        **kwargs,
) -> None:
    """
    Log training metrics to wandb.

    Args:
        step: Training step or epoch number.
        loss: Loss value.
        model_name: Optional model name.
        **kwargs: Additional metrics to log.
    """
    metrics = {
        "loss": loss,
        "step": step,
    }

    if model_name:
        metrics["model"] = model_name

    metrics.update(kwargs)
    wandb.log(metrics)


def log_summary_plot(
        run: wandb.run,
        losses: list,
        model_name: str,
) -> None:
    """
    Create and log a summary plot of training losses.

    Args:
        run: Wandb run object.
        losses: List of loss values over batches/steps.
        model_name: Name of model (for plot title).
    """
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title(f"Batch-wise Loss for {model_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    run.log({"loss_curve": wandb.Image(plt)})
    plt.close()

    logger.info(f"Logged summary plot for {model_name} ({len(losses)} batches)")