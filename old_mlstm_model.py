"""
mLSTM model implementation - Option A: Bidirectional with memory optimization.

This module implements a full mLSTM-based protein language model with:
- Bidirectional chunkwise processing
- Bias removal (matching ESM2 cramming style)
- Optimized hidden dimensions for fair parameter comparison with ESM2
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from typing import Optional
from data_utils import vocab_size as VOCAB_SIZE
from mlstm_utils import MultiHeadRMSNorm, soft_cap, bias_linspace_init_
from chunkwise import mlstm_chunkwise__native_custbw

logger = logging.getLogger(__name__)

# Device configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"mLSTM using device: {DEVICE}")


class mLSTMAttention(nn.Module):
    """
    mLSTM Attention Module with Multi-Head RMS Normalization and Chunkwise Processing.

    Implements matrix-LSTM attention with forget/input gates, supporting both
    unidirectional and bidirectional processing via chunkwise operations.

    Args:
        hidden_size: Total hidden dimension (must be divisible by num_heads).
        num_heads: Number of attention heads.
        chunk_size: Size of chunks for chunkwise computation (typically 64).
        bidirectional: If True, processes sequence bidirectionally.
        dropout: Dropout rate (default 0.0, can be increased for regularization).
    """

    def __init__(
            self,
            hidden_size: int,
            num_heads: int = 8,
            chunk_size: int = 64,
            bidirectional: bool = True,
            dropout: float = 0.0,
    ):
        super().__init__()

        assert hidden_size % num_heads == 0, (
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
        )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.chunk_size = chunk_size
        self.bidirectional = bidirectional
        self.dropout_p = dropout

        # Projection dimensions: Q,K use head_dim//2 for efficiency
        qk_dim = hidden_size // 2
        v_dim = hidden_size

        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_size, qk_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, qk_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, v_dim, bias=False)

        # Gate projections
        self.i_proj = nn.Linear(hidden_size, num_heads, bias=False)  # Input gate
        self.f_proj = nn.Linear(hidden_size, num_heads, bias=False)  # Forget gate
        self.o_gate = nn.Linear(hidden_size, hidden_size, bias=False)  # Output gate

        # Headwise normalization
        self.norm = MultiHeadRMSNorm(
            num_heads=num_heads,
            head_dim=self.head_dim,
            eps=1e-6,
            use_weight=True,
            use_bias=False,
            force_float32_reductions=True,
        )

        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Dropout
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.to(DEVICE)
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights with gate-specific strategies:
        - Input gate bias: -10 (starts closed)
        - Forget gate bias: linspace(3.0, 6.0) (starts open)
        - Output gate bias: 0
        """
        with torch.no_grad():
            # Zero out all projection weights
            for param in self.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param, gain=0.01)

            # Initialize gate biases if they exist (they shouldn't with bias=False, but for safety)
            if hasattr(self.i_proj, 'bias') and self.i_proj.bias is not None:
                self.i_proj.bias.fill_(-10.0)
            if hasattr(self.f_proj, 'bias') and self.f_proj.bias is not None:
                bias_linspace_init_(self.f_proj.bias, start=3.0, end=6.0)
            if hasattr(self.o_gate, 'bias') and self.o_gate.bias is not None:
                self.o_gate.bias.zero_()

    def forward(
            self,
            x: torch.Tensor,
            return_last_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple]:
        """
        Forward pass of mLSTM attention.

        Args:
            x: Input tensor of shape (B, S, hidden_size).
            return_last_states: Whether to return final states (for sequential generation).

        Returns:
            output: Attention output of shape (B, S, hidden_size).
            states: Last states if return_last_states=True.
        """
        x = x.to(DEVICE)
        B, S, _ = x.shape

        if self.bidirectional:
            return self._forward_bidirectional(x, return_last_states)
        else:
            return self._forward_unidirectional(x, return_last_states)

    def _forward_unidirectional(
            self,
            x: torch.Tensor,
            return_last_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple]:
        """Forward pass for unidirectional (causal) processing."""
        B, S, _ = x.shape
        H = self.num_heads
        D = self.head_dim
        D_qk = D // 2

        # Project to Q, K, V
        q = self.q_proj(x).view(B, S, H, D_qk).transpose(1, 2)  # (B, H, S, D_qk)
        k = self.k_proj(x).view(B, S, H, D_qk).transpose(1, 2)
        v = self.v_proj(x).view(B, S, H, D).transpose(1, 2)      # (B, H, S, D)

        # Project to gates with soft capping
        i = soft_cap(self.i_proj(x), cap_value=15).transpose(1, 2)  # (B, H, S)
        f = soft_cap(self.f_proj(x), cap_value=15).transpose(1, 2)

        # Apply mLSTM chunkwise computation
        out = mlstm_chunkwise__native_custbw(
            q=q,
            k=k,
            v=v,
            i=i,
            f=f,
            chunk_size=self.chunk_size,
            return_last_states=return_last_states,
        )

        if return_last_states:
            out, states = out

        # Reshape for normalization: (B, H, S, D) -> (B, S, H, D)
        out = out.transpose(1, 2).contiguous()

        # Apply headwise normalization
        out = self.norm(out)

        # Apply output gate and projection
        out = out * torch.sigmoid(self.o_gate(x))
        out = self.out_proj(out)

        if self.dropout:
            out = self.dropout(out)

        if return_last_states:
            return out, states
        return out

    def _forward_bidirectional(
            self,
            x: torch.Tensor,
            return_last_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple]:
        """
        Forward pass for bidirectional processing.

        Processes forward and reverse sequences simultaneously, then combines.
        Note: This doubles memory usage briefly (2B batch) but uses single kernel call.
        """
        B, S, _ = x.shape
        H = self.num_heads
        D = self.head_dim
        D_qk = D // 2

        # Create bidirectional batch: [forward, backward]
        x_bidir = torch.cat([x, torch.flip(x, dims=[1])], dim=0)  # (2B, S, hidden_size)

        # Project to Q, K, V for both directions
        q_bidir = self.q_proj(x_bidir).view(2*B, S, H, D_qk).transpose(1, 2)
        k_bidir = self.k_proj(x_bidir).view(2*B, S, H, D_qk).transpose(1, 2)
        v_bidir = self.v_proj(x_bidir).view(2*B, S, H, D).transpose(1, 2)

        # Project to gates
        i_bidir = soft_cap(self.i_proj(x_bidir), cap_value=15).transpose(1, 2)
        f_bidir = soft_cap(self.f_proj(x_bidir), cap_value=15).transpose(1, 2)

        # Apply mLSTM chunkwise computation
        out_bidir = mlstm_chunkwise__native_custbw(
            q=q_bidir,
            k=k_bidir,
            v=v_bidir,
            i=i_bidir,
            f=f_bidir,
            chunk_size=self.chunk_size,
            return_last_states=return_last_states,
        )

        if return_last_states:
            out_bidir, states_bidir = out_bidir

        # Reshape for normalization: (2B, H, S, D) -> (2B, S, H, D)
        out_bidir = out_bidir.transpose(1, 2).contiguous()

        # Apply headwise normalization
        out_bidir = self.norm(out_bidir)

        # Apply output gate
        out_bidir = out_bidir * torch.sigmoid(self.o_gate(x_bidir))

        # Split forward and backward outputs
        out_fw, out_bw = out_bidir[:B], out_bidir[B:]  # Each (B, S, hidden_size)

        # Flip backward output back to original order
        out_bw = torch.flip(out_bw, dims=[1])

        # Combine: element-wise sum of forward and backward
        out_combined = out_fw + out_bw

        # Output projection
        out = self.out_proj(out_combined)

        if self.dropout:
            out = self.dropout(out)

        if return_last_states:
            return out, states_bidir
        return out


class FullmLSTMModel(nn.Module):
    """
    Full mLSTM-based protein language model.

    Architecture:
    - Embedding: vocab_size -> embed_dim
    - Encoder: embed_dim -> hidden_size
    - mLSTM layer: (hidden_size, num_heads, chunk_size, bidirectional)
    - Output head: hidden_size -> vocab_size

    All linear layers have bias=False for memory efficiency (cramming style).
    """

    def __init__(
            self,
            vocab_size: int,
            embed_dim: int = 128,
            hidden_size: int = 320,  # Optimized to ~8M params
            num_heads: int = 8,
            chunk_size: int = 64,
            bidirectional: bool = True,
            dropout: float = 0.0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.bidirectional = bidirectional

        # Embedding layer
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Encoder: project embeddings to hidden dimension
        self.encoder = nn.Linear(embed_dim, hidden_size, bias=False)

        # mLSTM attention layer
        self.mlstm = mLSTMAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            chunk_size=chunk_size,
            bidirectional=bidirectional,
            dropout=dropout,
        )

        # Output head: project to vocab
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

        self.to(DEVICE)
        logger.info(self._get_model_info())

    def _get_model_info(self) -> str:
        """Return model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return (
            f"mLSTM Model (Option A - Bidirectional):\n"
            f"  Vocab size: {self.vocab_size}\n"
            f"  Embedding dim: {self.embed_dim}\n"
            f"  Hidden size: {self.hidden_size}\n"
            f"  Num heads: {self.num_heads}\n"
            f"  Head dim: {self.hidden_size // self.num_heads}\n"
            f"  Chunk size: {self.chunk_size}\n"
            f"  Bidirectional: {self.bidirectional}\n"
            f"  Total parameters: {total_params:,}\n"
            f"  Trainable parameters: {trainable_params:,}"
        )

    def forward(
            self,
            input_ids: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            return_hidden_states: bool = False,
    ) -> SimpleNamespace:
        """
        Forward pass of the mLSTM model.

        Args:
            input_ids: Token IDs of shape (B, S).
            labels: Target token IDs of shape (B, S) for training. Use -100 to ignore positions.

        Returns:
            SimpleNamespace with 'loss' (if labels provided) and 'logits'.
        """
        # Validate input range
        assert input_ids.max().item() < self.vocab_size, (
            f"Token ID {input_ids.max().item()} >= vocab_size {self.vocab_size}"
        )

        # Embedding
        x = self.embed(input_ids)  # (B, S, embed_dim)

        # Encode to hidden dimension
        x = self.encoder(x)  # (B, S, hidden_size)

        # mLSTM attention
        x = self.mlstm(x)  # (B, S, hidden_size)

        if return_hidden_states:
            return SimpleNamespace(hidden_states=x)

        # Output logits
        logits = self.head(x)  # (B, S, vocab_size)

        loss = None
        if labels is not None:
            # Validate sequence length is divisible by chunk_size
            S = input_ids.shape[1]
            assert S % self.chunk_size == 0, (
                f"Sequence length {S} must be divisible by chunk_size {self.chunk_size}"
            )

            # Validate labels are in valid range
            if labels[labels != -100].numel() > 0:
                assert labels.max().item() < self.vocab_size, (
                    f"Label {labels.max().item()} >= vocab_size {self.vocab_size}"
                )

            # Compute cross-entropy loss
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        return SimpleNamespace(loss=loss, logits=logits)


# Initialize model with Option A configuration
logger.info("Initializing mLSTM model (Option A - Bidirectional, Optimized)")

from config import MLSTM_CONFIG

model = FullmLSTMModel(
    vocab_size=VOCAB_SIZE,
    embed_dim=MLSTM_CONFIG["embed_dim"],           # ✅ From config
    hidden_size=MLSTM_CONFIG["hidden_size"],       # ✅ From config (1024!)
    num_heads=MLSTM_CONFIG["num_heads"],           # ✅ From config
    chunk_size=MLSTM_CONFIG["chunk_size"],         # ✅ From config
    bidirectional=MLSTM_CONFIG["bidirectional"],   # ✅ From config
    dropout=MLSTM_CONFIG["dropout"],               # ✅ From config
).to(DEVICE)

#logger.info(model._get_model_info())