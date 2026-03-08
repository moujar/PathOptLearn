"""Evaluation metrics for AdaptLearn AI benchmarks."""
import numpy as np
from typing import Sequence


def compute_auc(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    """Compute Area Under ROC Curve."""
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return 0.5


def compute_accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Compute binary classification accuracy."""
    if not y_true:
        return 0.0
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)


def compute_rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Compute Root Mean Squared Error."""
    if not y_true:
        return 1.0
    mse = sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)
    return float(mse ** 0.5)


def compute_recall_at_k(
    recommended: Sequence[str], relevant: Sequence[str], k: int = 5
) -> float:
    """Compute Recall@K for path recommendation."""
    if not relevant:
        return 0.0
    top_k = set(list(recommended)[:k])
    relevant_set = set(relevant)
    return len(top_k & relevant_set) / len(relevant_set)


def compute_les(knowledge_gain: float, time_invested_minutes: float) -> float:
    """
    Learning Efficiency Score = knowledge_gain / time_invested.
    knowledge_gain: total mastery improvement (0-1 scale)
    time_invested_minutes: total time in minutes
    """
    if time_invested_minutes <= 0:
        return 0.0
    return knowledge_gain / time_invested_minutes


def evaluate_path(
    path_steps: list[dict],
    initial_mastery: dict[str, float],
    final_mastery: dict[str, float],
    knowledge_gain: float,
    time_minutes: float,
) -> dict:
    """
    Compute all benchmark metrics for a learning path.
    Returns {auc, accuracy, rmse, recall_at_k, les}.
    """
    concepts_in_path = [s.get("concept", "") for s in path_steps]
    mastered_concepts = [c for c, v in final_mastery.items() if v >= 0.85]

    recall_k = compute_recall_at_k(concepts_in_path, mastered_concepts, k=5)
    les = compute_les(knowledge_gain, time_minutes)

    # Simulate AUC / ACC / RMSE from mastery predictions
    y_true, y_pred_prob = [], []
    for step in path_steps:
        concept = step.get("concept", "")
        predicted_delta = step.get("predicted_mastery_delta", 0.1)
        actual_gain = max(0.0, final_mastery.get(concept, 0) - initial_mastery.get(concept, 0))
        y_true.append(1 if actual_gain > 0.05 else 0)
        y_pred_prob.append(min(1.0, initial_mastery.get(concept, 0.5) + predicted_delta))

    auc = compute_auc(y_true, y_pred_prob) if y_true else 0.5
    acc = compute_accuracy(y_true, [1 if p >= 0.5 else 0 for p in y_pred_prob])
    rmse = compute_rmse(y_true, y_pred_prob)

    return {
        "auc": round(auc, 4),
        "accuracy": round(acc, 4),
        "rmse": round(rmse, 4),
        "recall_at_k": round(recall_k, 4),
        "les": round(les, 6),
    }
