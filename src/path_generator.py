"""
Greedy learning path generator: given student history, target skills, and constraints,
output an ordered path of items that maximizes expected learning (MVP = greedy next-step).
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

from .data import get_student_history
from .model import SuccessPredictor


def generate_path(
    predictor: SuccessPredictor,
    interactions: pd.DataFrame,
    items: pd.DataFrame,
    user_id: int,
    target_skills: Optional[List[int]] = None,
    max_steps: int = 10,
    exclude_item_ids: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
) -> Tuple[List[int], List[float]]:
    """
    Generate a learning path for the given user (student).

    Args:
        predictor: Trained SuccessPredictor.
        interactions: Full interaction table (for feature building).
        items: Item pool with item_id, skill_id, difficulty.
        user_id: Student id.
        target_skills: Skills to prioritize (e.g. [0, 1, 2] to master skills 0,1,2). None = all.
        max_steps: Maximum number of items in the path.
        exclude_item_ids: Item ids already in history to avoid repeating (or None to allow repeats).
        random_state: For tie-breaking.

    Returns:
        path_item_ids: Ordered list of item ids.
        path_scores: Predicted P(correct) (or score) for each step.
    """
    rng = np.random.default_rng(random_state)
    history = get_student_history(interactions, user_id)
    all_item_ids = items["item_id"].values
    if exclude_item_ids is not None:
        mask = np.isin(all_item_ids, exclude_item_ids, invert=True)
        candidate_pool = all_item_ids[mask]
    else:
        candidate_pool = all_item_ids

    path_item_ids: List[int] = []
    path_scores: List[float] = []

    for _ in range(max_steps):
        if len(candidate_pool) == 0:
            break
        # Score each candidate given current (simulated) history
        scores = predictor.score_next(
            interactions, items, history,
            candidate_item_ids=candidate_pool,
            target_skills=np.array(target_skills) if target_skills else None,
        )
        # Greedy: pick item with highest score (expected success; target-skill boost applied in score_next)
        best_idx = np.argmax(scores)
        # Tie-breaking
        best_score = scores[best_idx]
        ties = np.where(np.isclose(scores, best_score))[0]
        best_idx = rng.choice(ties)
        next_item = int(candidate_pool[best_idx])
        path_item_ids.append(next_item)
        path_scores.append(float(scores[best_idx]))

        # Append a "virtual" interaction so next step's features use this item
        row = items[items["item_id"] == next_item].iloc[0]
        new_row = pd.DataFrame([{
            "user_id": user_id,
            "item_id": next_item,
            "skill_id": row["skill_id"],
            "correct": 1,  # assume success for path simulation (or use predicted)
            "timestamp": len(history) + len(path_item_ids) - 1,
        }])
        history = pd.concat([history, new_row], ignore_index=True)

        # Optional: remove chosen item from pool to avoid repetition
        candidate_pool = np.delete(candidate_pool, best_idx)

    return path_item_ids, path_scores
