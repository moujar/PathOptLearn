"""
Next-step / success predictor: given student history and candidate items,
predict P(correct) for each candidate. Used by the path generator to pick the next best item.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from typing import Optional, Tuple

from .data import (
    build_training_data,
    build_feature_matrix,
    get_student_history,
)


class SuccessPredictor:
    """Predicts P(correct) for (student history, item) using hand-crafted features + logistic regression."""

    def __init__(self, n_skills: int, C: float = 1.0, max_iter: int = 500):
        self.n_skills = n_skills
        self.clf = LogisticRegression(C=C, max_iter=max_iter, random_state=42)

    def fit(
        self,
        interactions: pd.DataFrame,
        items: pd.DataFrame,
    ) -> "SuccessPredictor":
        X, y = build_training_data(interactions, items, self.n_skills)
        self.clf.fit(X, y)
        return self

    def predict_proba_items(
        self,
        interactions: pd.DataFrame,
        items: pd.DataFrame,
        history: pd.DataFrame,
        candidate_item_ids: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        For a given student history, return P(correct) for each candidate item.
        Returns (proba_correct, item_ids).
        """
        X, item_ids = build_feature_matrix(
            interactions, items, self.n_skills,
            history_per_student=history,
            candidate_item_ids=candidate_item_ids,
        )
        # P(correct) = second column of predict_proba (class 1)
        proba = self.clf.predict_proba(X)[:, 1]
        return proba, item_ids

    def score_next(
        self,
        interactions: pd.DataFrame,
        items: pd.DataFrame,
        history: pd.DataFrame,
        candidate_item_ids: np.ndarray,
        target_skills: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Score each candidate item for "next step" in a path.
        By default: score = P(correct). If target_skills given, boost items whose skill is in target_skills
        (to favor covering target skills).
        """
        proba, _ = self.predict_proba_items(
            interactions, items, history, candidate_item_ids=candidate_item_ids
        )
        if target_skills is None or len(target_skills) == 0:
            return proba
        items_df = items.set_index("item_id")
        boost = np.array([
            1.2 if items_df.loc[iid, "skill_id"] in target_skills else 1.0
            for iid in candidate_item_ids
        ])
        return proba * boost
