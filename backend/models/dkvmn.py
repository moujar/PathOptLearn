"""Dynamic Key-Value Memory Network (DKVMN) for knowledge tracing."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DKVMN(nn.Module):
    """
    Dynamic Key-Value Memory Network.
    Uses static key memory for concept lookup and dynamic value memory for student state.
    """

    def __init__(
        self,
        n_concepts: int,
        key_dim: int = 50,
        value_dim: int = 200,
        summary_dim: int = 50,
        final_fc_dim: int = 50,
    ):
        super().__init__()
        self.n_concepts = n_concepts
        self.key_dim = key_dim
        self.value_dim = value_dim

        # Static key memory — one key per concept
        self.key_memory = nn.Parameter(torch.randn(n_concepts, key_dim))
        # Initial value memory — learned starting state
        self.value_memory = nn.Parameter(torch.randn(n_concepts, value_dim))

        # Embeddings
        self.concept_embed = nn.Embedding(n_concepts + 1, key_dim, padding_idx=0)
        self.interaction_embed = nn.Embedding(2 * n_concepts + 1, value_dim, padding_idx=0)

        # Erase / add memory write gates
        self.erase_linear = nn.Linear(value_dim, value_dim)
        self.add_linear = nn.Linear(value_dim, value_dim)

        # Output MLP
        self.fc1 = nn.Linear(value_dim + key_dim, summary_dim)
        self.fc2 = nn.Linear(summary_dim, final_fc_dim)
        self.fc3 = nn.Linear(final_fc_dim, 1)
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    # ── Memory operations ──────────────────────────────────────────────────

    def _attention(self, q: torch.Tensor) -> torch.Tensor:
        """Softmax attention over key memory given query q."""
        return F.softmax(torch.matmul(q, self.key_memory.T), dim=-1)

    def _read(self, w: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Read weighted sum from value memory."""
        return torch.matmul(w.unsqueeze(1), memory).squeeze(1)

    def _write(self, w: torch.Tensor, memory: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Erase-add write to value memory."""
        erase = self.sigmoid(self.erase_linear(v))
        add = self.tanh(self.add_linear(v))
        memory = memory * (1 - w.unsqueeze(-1) * erase.unsqueeze(1))
        memory = memory + w.unsqueeze(-1) * add.unsqueeze(1)
        return memory

    # ── Forward pass ───────────────────────────────────────────────────────

    def forward(self, concepts: torch.Tensor, interactions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            concepts: (batch, seq_len) — concept ids (1-indexed)
            interactions: (batch, seq_len) — interaction ids (concept*2+correct)
        Returns:
            predictions: (batch, seq_len) — P(correct) for next step
        """
        batch_size, seq_len = concepts.shape
        memory = self.value_memory.unsqueeze(0).expand(batch_size, -1, -1).clone()
        predictions = []

        for t in range(seq_len):
            q = self.concept_embed(concepts[:, t])
            w = self._attention(q)
            r = self._read(w, memory)

            f = torch.cat([r, q], dim=-1)
            f = self.dropout(torch.relu(self.fc1(f)))
            f = self.dropout(torch.relu(self.fc2(f)))
            pred = self.sigmoid(self.fc3(f))
            predictions.append(pred)

            if t < seq_len - 1:
                v = self.interaction_embed(interactions[:, t])
                memory = self._write(w, memory, v)

        return torch.cat(predictions, dim=-1)  # (batch, seq_len)

    def get_knowledge_state(
        self, concepts: torch.Tensor, interactions: torch.Tensor
    ) -> dict[int, float]:
        """Extract per-concept mastery after processing a sequence."""
        self.eval()
        with torch.no_grad():
            batch_size, seq_len = concepts.shape
            memory = self.value_memory.unsqueeze(0).expand(batch_size, -1, -1).clone()

            for t in range(seq_len):
                q = self.concept_embed(concepts[:, t])
                w = self._attention(q)
                v = self.interaction_embed(interactions[:, t])
                memory = self._write(w, memory, v)

            masteries: dict[int, float] = {}
            for i in range(self.n_concepts):
                c = torch.tensor([[i + 1]])
                q = self.concept_embed(c)
                w = self._attention(q)
                r = self._read(w, memory[0:1])
                f = torch.cat([r, q], dim=-1)
                f = torch.relu(self.fc1(f))
                f = torch.relu(self.fc2(f))
                masteries[i] = float(self.sigmoid(self.fc3(f)).item())

            return masteries

    def greedy_path(
        self,
        concepts: torch.Tensor,
        interactions: torch.Tensor,
        available_concepts: list[int],
        n_steps: int = 10,
    ) -> list[int]:
        """Greedy path: always pick the concept with lowest current mastery."""
        state = self.get_knowledge_state(concepts, interactions)
        remaining = list(available_concepts)
        path: list[int] = []
        for _ in range(n_steps):
            if not remaining:
                break
            best = min(remaining, key=lambda c: state.get(c, 0.0))
            path.append(best)
            remaining.remove(best)
            state[best] = min(1.0, state.get(best, 0.0) + 0.15)
        return path
