"""
Synthetic education data for MVP: students, items (with skills), interactions.
Mimics EdNet/ASSISTments-style columns: user_id, item_id, skill_id, correct, timestamp.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional


def generate_synthetic_data(
    n_students: int = 200,
    n_items: int = 100,
    n_skills: int = 10,
    n_interactions_per_student: int = 50,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic interaction data and item–skill mapping.

    Returns:
        interactions: columns [user_id, item_id, skill_id, correct, timestamp]
        items: columns [item_id, skill_id, difficulty] (difficulty in [0,1])
    """
    rng = np.random.default_rng(seed)

    # Item pool: each item belongs to one skill; difficulty ~ U(0.2, 0.8)
    item_ids = np.arange(n_items)
    item_skill = rng.integers(0, n_skills, size=n_items)
    item_difficulty = rng.uniform(0.2, 0.8, size=n_items)
    items = pd.DataFrame({
        "item_id": item_ids,
        "skill_id": item_skill,
        "difficulty": item_difficulty,
    })

    # Per-student latent skill mastery (evolves with practice)
    student_mastery = rng.uniform(0.2, 0.5, size=(n_students, n_skills))

    rows = []
    for u in range(n_students):
        mastery = student_mastery[u].copy()
        # Random path through items (with replacement for simplicity)
        seq = rng.integers(0, n_items, size=n_interactions_per_student)
        for t, item_id in enumerate(seq):
            skill_id = item_skill[item_id]
            diff = item_difficulty[item_id]
            # P(correct) increases with mastery and decreases with difficulty
            p_correct = mastery[skill_id] * (1 - diff * 0.5) + 0.2
            p_correct = np.clip(p_correct, 0.1, 0.95)
            correct = 1 if rng.random() < p_correct else 0
            # Slight mastery update
            mastery[skill_id] = np.clip(mastery[skill_id] + (0.05 if correct else -0.02), 0, 1)
            rows.append({
                "user_id": u,
                "item_id": item_id,
                "skill_id": skill_id,
                "correct": correct,
                "timestamp": t,
            })

    interactions = pd.DataFrame(rows)
    return interactions, items


def get_student_history(
    interactions: pd.DataFrame,
    user_id: int,
    max_len: Optional[int] = None,
) -> pd.DataFrame:
    """Return chronologically ordered interactions for one student."""
    hist = interactions[interactions["user_id"] == user_id].sort_values("timestamp")
    if max_len is not None:
        hist = hist.tail(max_len)
    return hist.reset_index(drop=True)


def _features_for_item(
    item_id: int,
    items: pd.DataFrame,
    skill_acc: np.ndarray,
    n_interactions: int,
    avg_correct: float,
    n_skills: int,
) -> np.ndarray:
    """Single row of features for one (student state, item) pair."""
    row = items[items["item_id"] == item_id].iloc[0]
    skill_id = int(row["skill_id"])
    diff = float(row["difficulty"])
    feat = [skill_acc[skill_id], diff, avg_correct, n_interactions]
    for s in range(n_skills):
        feat.append(1.0 if s == skill_id else 0.0)
    return np.array(feat, dtype=np.float32)


def build_training_data(
    interactions: pd.DataFrame,
    items: pd.DataFrame,
    n_skills: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) for training a success predictor.
    For each interaction, features are computed from prior history of that user.
    """
    X_list, y_list = [], []
    for user_id in interactions["user_id"].unique():
        hist = get_student_history(interactions, user_id)
        for i in range(len(hist)):
            prior = hist.iloc[:i]
            current = hist.iloc[i]
            item_id = current["item_id"]
            if len(prior) == 0:
                skill_acc = np.full(n_skills, 0.5)
                n_interactions, avg_correct = 0, 0.5
            else:
                skill_correct = prior.groupby("skill_id")["correct"].agg(["sum", "count"]).reindex(
                    range(n_skills), fill_value=0
                )
                skill_acc = (skill_correct["sum"] / (skill_correct["count"] + 1e-6)).values
                n_interactions = len(prior)
                avg_correct = prior["correct"].mean()
            feat = _features_for_item(
                item_id, items, skill_acc, n_interactions, avg_correct, n_skills
            )
            X_list.append(feat)
            y_list.append(current["correct"])
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)


def build_feature_matrix(
    interactions: pd.DataFrame,
    items: pd.DataFrame,
    n_skills: int,
    history_per_student: Optional[pd.DataFrame] = None,
    candidate_item_ids: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build features for success prediction (inference).
    Given a student history, score candidate items. Returns (X, item_ids).
    """
    if history_per_student is None or len(history_per_student) == 0:
        skill_acc = np.full(n_skills, 0.5)
        n_interactions, avg_correct = 0, 0.5
    else:
        hist = history_per_student
        skill_correct = hist.groupby("skill_id")["correct"].agg(["sum", "count"]).reindex(
            range(n_skills), fill_value=0
        )
        skill_acc = (skill_correct["sum"] / (skill_correct["count"] + 1e-6)).values
        n_interactions = len(hist)
        avg_correct = hist["correct"].mean()

    if candidate_item_ids is None:
        candidate_item_ids = items["item_id"].values

    X_list = [
        _features_for_item(iid, items, skill_acc, n_interactions, avg_correct, n_skills)
        for iid in candidate_item_ids
    ]
    return np.array(X_list, dtype=np.float32), np.asarray(candidate_item_ids)
