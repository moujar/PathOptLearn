"""Deep Knowledge Tracing (DKT) using LSTM for knowledge state prediction."""
import torch
import torch.nn as nn
from typing import Tuple


class DKT(nn.Module):
    """
    Deep Knowledge Tracing model using LSTM.
    Predicts student's probability of correct answer per concept given interaction history.
    """

    def __init__(
        self,
        n_concepts: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_concepts = n_concepts
        self.embed_dim = embed_dim

        # Interaction embedding: concept_id*2 + correct (0/1) -> embed
        self.interaction_embed = nn.Embedding(2 * n_concepts + 1, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.output_layer = nn.Linear(hidden_dim, n_concepts)
        self.sigmoid = nn.Sigmoid()

    def forward(self, interactions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            interactions: (batch, seq_len) — values are concept_id * 2 + correct
        Returns:
            predictions: (batch, seq_len, n_concepts) — P(correct) per concept
            h_n: final LSTM hidden state
        """
        embeds = self.interaction_embed(interactions)
        lstm_out, (h_n, _) = self.lstm(embeds)
        predictions = self.sigmoid(self.output_layer(lstm_out))
        return predictions, h_n

    def predict_next(self, interactions: torch.Tensor, target_concept: int) -> float:
        """Predict P(correct) for target_concept given interaction history."""
        self.eval()
        with torch.no_grad():
            preds, _ = self.forward(interactions.unsqueeze(0))
            return float(preds[0, -1, target_concept].item())

    def get_knowledge_state(self, interactions: torch.Tensor) -> dict[int, float]:
        """Return per-concept mastery dict from current interaction sequence."""
        self.eval()
        with torch.no_grad():
            preds, _ = self.forward(interactions.unsqueeze(0))
            last = preds[0, -1, :]  # (n_concepts,)
            return {i: float(last[i].item()) for i in range(self.n_concepts)}
