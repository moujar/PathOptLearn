"""Ebbinghaus forgetting curve module with spaced repetition stability tracking."""
import math
import logging
from datetime import datetime, timezone
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class EbbinghausForgetting:
    """
    Models memory retention using the Ebbinghaus forgetting curve:
        R(t) = e^(-t / S)
    where t = time since last study (hours) and S = stability (hours).

    Stability grows after each review (spaced repetition effect).
    """

    DEFAULT_STABILITY_HOURS: float = 24.0

    def __init__(self) -> None:
        self._stability: Dict[str, float] = {}

    def apply_forgetting(
        self, concept: str, mastery: float, t_hours: float
    ) -> Tuple[float, float]:
        """
        Apply decay to a concept mastery value.

        Returns:
            (new_mastery, delta)  where delta is the loss (negative).
        """
        S = self._stability.get(concept, self.DEFAULT_STABILITY_HOURS)
        retention = math.exp(-t_hours / S)
        new_mastery = max(0.0, mastery * retention)
        delta = new_mastery - mastery
        logger.debug(
            f"Forgetting [{concept}]: {mastery:.3f} -> {new_mastery:.3f} "
            f"(t={t_hours:.1f}h, S={S:.1f}h, R={retention:.3f})"
        )
        return new_mastery, delta

    def update_stability(self, concept: str, mastery_before: float) -> float:
        """
        Increase stability after a review session.
        Higher mastery -> larger stability boost (SM-2 inspired).
        """
        S = self._stability.get(concept, self.DEFAULT_STABILITY_HOURS)
        new_S = S * (1.0 + 0.5 * mastery_before)
        self._stability[concept] = new_S
        return new_S

    def apply_forgetting_to_vector(
        self,
        concept_mastery: Dict[str, float],
        last_studied_at: Dict[str, str],
        current_time_iso: str = "",
        global_t_hours: float = 0.0,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Apply forgetting to every concept in a KnowledgeVector.

        Args:
            concept_mastery: {concept: mastery_0_to_1}
            last_studied_at: {concept: ISO timestamp}
            current_time_iso: current time as ISO string
            global_t_hours: fallback elapsed hours if per-concept timestamps are missing

        Returns:
            (updated_mastery, per_concept_deltas)
        """
        now = (
            datetime.fromisoformat(current_time_iso.replace("Z", "+00:00"))
            if current_time_iso
            else datetime.now(timezone.utc)
        )

        updated: Dict[str, float] = {}
        deltas: Dict[str, float] = {}

        for concept, mastery in concept_mastery.items():
            t_hours = global_t_hours
            if concept in last_studied_at:
                try:
                    last = datetime.fromisoformat(
                        last_studied_at[concept].replace("Z", "+00:00")
                    )
                    t_hours = max(t_hours, (now - last).total_seconds() / 3600.0)
                except Exception:
                    pass

            if t_hours > 0:
                new_m, delta = self.apply_forgetting(concept, mastery, t_hours)
                updated[concept] = round(new_m, 4)
                deltas[concept] = round(delta, 4)
            else:
                updated[concept] = mastery
                deltas[concept] = 0.0

        return updated, deltas


# Module-level singleton
forgetting_module = EbbinghausForgetting()
