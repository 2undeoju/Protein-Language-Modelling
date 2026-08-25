"""
ESM2-comparable mLSTM Protein Language Model

Cleaned-up deeper mLSTM architecture directly comparable to ESM2 (esm2_t6_8M).

Keeps:
- stacked mLSTM blocks
- pre-LN + residual
- FFN after each block
- learned positional embeddings
- final LN + MLM head
- same chunkwise kernel
"""

import math
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from types import SimpleNamespace

from config import MLSTM_CONFIG, DEVICE
from data_utils import vocab_size as VOCAB_SIZE
from chunkwise import mlstm_chunkwise__native_custbw
from mlstm_utils import MultiHeadRMSNorm, soft_cap, bias_linspace_init_

logger = logging.getLogger(__name__)
logger.info("Initializing NEW ESM2-comparable mLSTM model")


# ============================================
# 1. mLSTMAttention CLASS
# ============================================

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

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

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
            if hasattr(self.f_proj, "bias") and self.f_proj.bias is not None:
                bias_linspace_init_(self.f_proj.bias, start=3.0, end=6.0)
            if hasattr(self.o_gate, "bias") and self.o_gate.bias is not None:
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
        q_bidir = self.q_proj(x_bidir).view(2 * B, S, H, D_qk).transpose(1, 2)
        k_bidir = self.k_proj(x_bidir).view(2 * B, S, H, D_qk).transpose(1, 2)
        v_bidir = self.v_proj(x_bidir).view(2 * B, S, H, D).transpose(1, 2)

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

# ============================================
# 2. FeedForward class
# ============================================

class FeedForward(nn.Module):
    """ESM2-style MLP: hidden -> 4*hidden -> hidden."""
    def __init__(self, hidden_size: int, expansion: int = 4, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        inner = hidden_size * expansion
        self.fc1 = nn.Linear(hidden_size, inner, bias=bias)
        self.fc2 = nn.Linear(inner, hidden_size, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

        # small init similar to cramming style
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.5)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)


class mLSTMBlock(nn.Module):
    """Pre-LN residual mLSTM block + FFN."""
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        chunk_size: int,
        bidirectional: bool = True,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size, elementwise_affine=True)
        self.attn = mLSTMAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            chunk_size=chunk_size,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(hidden_size, elementwise_affine=True)
        self.ffn = FeedForward(hidden_size, expansion=4, dropout=dropout, bias=bias)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # mLSTM attention sublayer
        h = self.ln1(x)
        h = self.attn(h) #, attention_mask=attention_mask)
        x = x + self.dropout1(h)

        # FFN sublayer
        h = self.ln2(x)
        h = self.ffn(h)
        x = x + self.dropout2(h)
        return x

# ============================================
# 3. FullmLSTMModel class
# ============================================

class FullmLSTMModel(nn.Module):
    """
    ESM2-comparable stacked mLSTM protein language model.

    Args:
        vocab_size: size of AA vocabulary.
        embed_dim: token embedding size (projected to hidden_size).
        hidden_size: model width (match ESM2 small = 320 by default).
        num_layers: number of stacked mLSTM blocks (match esm2_t6 = 6).
        num_heads: number of heads for mLSTM attention.
        chunk_size: chunk size for chunkwise kernel.
        bidirectional: whether to use forward+backward mLSTM attention.
        dropout: dropout prob.
        max_position_embeddings: max length for learned pos embeddings.
        bias: whether linear layers use bias (keep False for cramming fairness).

    Forward returns a SimpleNamespace with fields:
        logits: (B, S, vocab_size)
        loss: scalar or None
        hidden_states: optional list of per-layer states if output_hidden_states=True
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_size: int = 320,
        num_layers: int = 6,
        num_heads: int = 8,
        chunk_size: int = 64,
        bidirectional: bool = True,
        dropout: float = 0.0,
        max_position_embeddings: int = 2048,
        bias: bool = False,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.max_position_embeddings = max_position_embeddings
        self.bias = bias

        # Token embedding
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim)

        # Learned positional embedding (ESM2 has position info implicitly; we add explicitly here)
        self.embed_positions = nn.Embedding(max_position_embeddings, hidden_size)

        # Project token embeddings to hidden width
        self.encoder = nn.Linear(embed_dim, hidden_size, bias=bias)

        # Stacked blocks
        self.layers = nn.ModuleList([
            mLSTMBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                chunk_size=chunk_size,
                bidirectional=bidirectional,
                dropout=dropout,
                bias=bias,
            )
            for _ in range(num_layers)
        ])

        self.final_ln = nn.LayerNorm(hidden_size, elementwise_affine=True)

        # MLM head
        self.head = nn.Linear(hidden_size, vocab_size, bias=bias)

        self.dropout_layer = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        # Similar init spirit to your cramming setup
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.embed_positions.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.encoder.weight, gain=1.0)
        nn.init.xavier_uniform_(self.head.weight, gain=1.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
    ):
        """
        Args:
            input_ids: (B, S) token ids.
            attention_mask: (B, S) 1 for valid tokens, 0 for padding.
            labels: (B, S) with -100 for ignore.
            return_hidden_states: return list of hidden states.

        Note: chunkwise kernel expects sequence length divisible by chunk_size.
        Your ds_utils pads to chunk multiple already.
        """
        B, S = input_ids.shape
        if S > self.max_position_embeddings:
            raise ValueError(
                f"Sequence length {S} exceeds max_position_embeddings={self.max_position_embeddings}. "
                f"Increase max_position_embeddings if needed."
            )

        # token embeddings -> hidden
        x = self.embed_tokens(input_ids)              # (B, S, embed_dim)
        x = self.encoder(x)                          # (B, S, hidden)
        x = self.dropout_layer(x)

        # add positions
        pos_ids = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, S)
        pos_emb = self.embed_positions(pos_ids)      # (B, S, hidden)
        x = x + pos_emb
        x = self.dropout_layer(x)

        all_hidden_states: List[torch.Tensor] = [] if return_hidden_states else None
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
            if return_hidden_states:
                all_hidden_states.append(x)

        x = self.final_ln(x)
        logits = self.head(x)                        # (B, S, vocab)

        loss = None
        if labels is not None:
            # standard MLM loss
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        return SimpleNamespace(
            logits=logits,
            loss=loss,
            hidden_states=x,
            all_hidden_states=all_hidden_states,
        )

    def get_config(self) -> str:
        """Return model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return (
            f"mLSTM Model (Option B - Bidirectional):\n"
            f"  Vocab size: {self.vocab_size}\n"
            f"  Embedding dim: {self.embed_dim}\n"
            f"  Hidden size: {self.hidden_size}\n"
            f"  Num layers: {self.num_layers}\n"
            f"  Num heads: {self.num_heads}\n"
            f"  Head dim: {self.hidden_size // self.num_heads}\n"
            f"  Chunk size: {self.chunk_size}\n"
            f"  Bidirectional: {self.bidirectional}\n"
            f"  Dropout: {self.dropout}\n"
            f"  Max position embeddings: {self.max_position_embeddings}\n"
            f"  Bias: {self.bias}\n"
            f"  Total parameters: {total_params:,}\n"
            f"  Trainable parameters: {trainable_params:,}"
        )

    def __repr__(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        cfg = self.get_config()
        return (
            "FullmLSTMModel(ESM2-comparable)\n"
            + "\n".join([f"  {k}: {v}" for k, v in cfg.items()])
            + f"\n  total_params: {total_params:,}\n"
            + f"  trainable_params: {trainable_params:,}"
        )

# ----------------------------------------------------------------------------
# CLEAN ADDITION: builder function
# ----------------------------------------------------------------------------
def build_mlstm_model(cfg: dict) -> FullmLSTMModel:
    """
    Clean factory: build mLSTM model from config dict.
    Keeps the same exact architecture choices as your original.
    """
    return FullmLSTMModel(
        vocab_size=VOCAB_SIZE,
        embed_dim=cfg["embed_dim"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        chunk_size=cfg["chunk_size"],
        bidirectional=cfg["bidirectional"],
        dropout=cfg["dropout"],
        max_position_embeddings=cfg["max_position_embeddings"],
        bias=False,
    )


# ----------------------------------------------------------------------------
# BACKWARD COMPATIBILITY: keep global model exactly like before
# ----------------------------------------------------------------------------
model = build_mlstm_model(MLSTM_CONFIG).to(DEVICE)
logger.info(model.get_config())


