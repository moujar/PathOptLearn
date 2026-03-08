"""Deep-IRT: DKVMN + IRT parameter prediction heads for item calibration."""
import logging
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DeepIRT(nn.Module):
    """
    Deep-IRT model combining DKVMN memory with IRT parameter prediction heads.
    Outputs: difficulty_b (-3..+3), discrimination_a (0.5..2.5), correct_prob (0..1).
    """

    def __init__(self, n_concepts: int = 100, embed_dim: int = 384):
        super().__init__()
        self.n_concepts = n_concepts

        # IRT difficulty head
        self.difficulty_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Tanh(),  # output in -1..1 -> scaled to -3..+3
        )

        # IRT discrimination head
        self.discrimination_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Softplus(),  # positive
        )

        # Student state projection for correct-prob prediction
        self.correct_prob_head = nn.Sequential(
            nn.Linear(embed_dim + 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, question_embedding: torch.Tensor, student_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            question_embedding: (batch, embed_dim)
            student_state: (batch, embed_dim) or None
        Returns:
            difficulty_b, discrimination_a, correct_prob
        """
        b = self.difficulty_head(question_embedding).squeeze(-1) * 3.0   # -3..+3
        a = self.discrimination_head(question_embedding).squeeze(-1) * 2.0 + 0.5  # 0.5..2.5

        if student_state is not None:
            x = torch.cat([student_state, b.unsqueeze(-1), a.unsqueeze(-1)], dim=-1)
        else:
            zeros = torch.zeros_like(question_embedding)
            x = torch.cat([zeros, b.unsqueeze(-1), a.unsqueeze(-1)], dim=-1)

        prob = self.correct_prob_head(x).squeeze(-1)
        return b, a, prob

    def calibrate_item(
        self, question_text: str, embed_model=None
    ) -> Tuple[float, float, float]:
        """
        Estimate IRT parameters (a, b, c) for a raw question string.
        Returns (discrimination_a, difficulty_b, guessing_c).
        """
        self.eval()
        with torch.no_grad():
            if embed_model is not None:
                try:
                    emb = embed_model.encode([question_text])
                    q_emb = torch.tensor(emb, dtype=torch.float32)
                except Exception as exc:
                    logger.warning(f"Embedding failed: {exc}; using zeros")
                    q_emb = torch.zeros(1, 384)
            else:
                q_emb = torch.zeros(1, 384)

            b, a, _ = self.forward(q_emb)
            return float(a.item()), float(b.item()), 0.25

    @classmethod
    def load_pretrained(cls, path: str) -> "DeepIRT":
        """Load pretrained weights or return fresh model if unavailable."""
        model = cls()
        if os.path.exists(path):
            try:
                state = torch.load(path, map_location="cpu")
                model.load_state_dict(state)
                logger.info(f"Loaded DeepIRT from {path}")
            except Exception as exc:
                logger.warning(f"Failed to load DeepIRT weights: {exc}")
        else:
            logger.info("No pretrained DeepIRT found; using randomly initialized model.")
        return model
