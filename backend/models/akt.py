"""Attentive Knowledge Tracing (AKT) Transformer model."""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AKTSelfAttention(nn.Module):
    """Multi-head self-attention with causal (monotonic) mask for AKT."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = q.shape
        Q = self.q_proj(q).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(k).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(v).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)


class AKTBlock(nn.Module):
    """Single transformer block for AKT."""

    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = AKTSelfAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(), nn.Linear(ff_dim, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.attn(x, x, x, mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class AKT(nn.Module):
    """
    Attentive Knowledge Tracing Transformer.
    Uses separate encoders for questions and interactions with monotonic attention.
    """

    def __init__(
        self,
        n_concepts: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 3,
        ff_dim: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.n_concepts = n_concepts
        self.d_model = d_model

        self.concept_embed = nn.Embedding(n_concepts + 1, d_model, padding_idx=0)
        self.interaction_embed = nn.Embedding(2 * n_concepts + 1, d_model, padding_idx=0)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList(
            [AKTBlock(d_model, n_heads, ff_dim, dropout) for _ in range(n_layers)]
        )
        self.output_layer = nn.Linear(d_model, n_concepts)
        self.sigmoid = nn.Sigmoid()

    def forward(self, concepts: torch.Tensor, interactions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            concepts: (batch, seq_len)
            interactions: (batch, seq_len)
        Returns:
            (batch, seq_len, n_concepts) — predicted correctness probabilities
        """
        B, T = concepts.shape
        pos = torch.arange(T, device=concepts.device).unsqueeze(0)

        x = self.interaction_embed(interactions) + self.pos_embed(pos)
        mask = torch.tril(torch.ones(T, T, device=concepts.device)).unsqueeze(0).unsqueeze(0)

        for block in self.blocks:
            x = block(x, mask)

        return self.sigmoid(self.output_layer(x))  # (B, T, n_concepts)

    def get_concept_mastery(
        self, concepts: torch.Tensor, interactions: torch.Tensor
    ) -> dict[int, float]:
        """Get per-concept mastery from final sequence step."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(concepts, interactions)
            last = logits[0, -1, :]
            return {i: float(last[i].item()) for i in range(self.n_concepts)}

    def greedy_path(
        self,
        concepts: torch.Tensor,
        interactions: torch.Tensor,
        available_concepts: list[int],
        n_steps: int = 10,
    ) -> list[int]:
        """AKT-greedy: select concept with highest predicted mastery gain."""
        current = self.get_concept_mastery(concepts, interactions)
        remaining = list(available_concepts)
        path: list[int] = []
        for _ in range(n_steps):
            if not remaining:
                break
            best = max(remaining, key=lambda c: 1.0 - current.get(c, 0.0))
            path.append(best)
            remaining.remove(best)
            current[best] = min(1.0, current.get(best, 0.0) + 0.2)
        return path
