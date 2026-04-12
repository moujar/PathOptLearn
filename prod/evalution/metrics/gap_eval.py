"""
Knowledge Gap Detection Quality — PathOptLearn Evaluation
==========================================================
Evaluates how accurately PathOptLearn's POST /find-gaps endpoint identifies
conceptual weaknesses in a student's understanding.

Metrics
-------
  1. Precision, Recall, F1   — predicted gaps vs. expert-annotated ground-truth
  2. Severity calibration    — correlation between predicted severity and quiz failure
  3. Gap stability           — Jaccard similarity when the same answers are re-evaluated
  4. Over/under-detection    — are too many or too few gaps returned?

Usage
-----
    from metrics.gap_eval import evaluate_gap_detection

    result = evaluate_gap_detection(
        predicted_gaps    = ["gradient descent", "overfitting"],
        ground_truth_gaps = ["overfitting", "regularisation", "learning rate"],
        gap_severities    = ["medium", "high"],
        quiz_outcomes     = [1, 0],   # 1=correct, 0=wrong, aligned with gaps
    )
    print(result)
"""

from __future__ import annotations

import statistics
from typing import Any


# ── Severity encoding ─────────────────────────────────────────────────────────
SEVERITY_SCORE: dict[str | None, int] = {
    "high":   3,
    "medium": 2,
    "low":    1,
    None:     0,
}


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Precision / Recall / F1
# ══════════════════════════════════════════════════════════════════════════════

def precision_recall_f1(
    predicted_gaps: list[str],
    ground_truth_gaps: list[str],
    normalize: bool = True,
) -> dict:
    """
    Token-level precision, recall, and F1 score comparing predicted gap
    concepts against a ground-truth gap set.

    Comparison is case-insensitive and trims whitespace when
    ``normalize=True``.

    Returns
    -------
    {
        "precision":        float,
        "recall":           float,
        "f1":               float,
        "n_predicted":      int,
        "n_ground_truth":   int,
        "n_correct":        int,   # true positives (TP)
        "false_positives":  list[str],
        "false_negatives":  list[str],
        "detection_rate":   str,   # "over" | "under" | "on_target"
    }
    """
    def _norm(s: str) -> str:
        return s.lower().strip() if normalize else s

    pred_set = {_norm(g) for g in predicted_gaps}
    gt_set   = {_norm(g) for g in ground_truth_gaps}

    tp = pred_set & gt_set
    fp = pred_set - gt_set
    fn = gt_set   - pred_set

    precision = len(tp) / max(len(pred_set), 1)
    recall    = len(tp) / max(len(gt_set),   1)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    # Over/under-detection judgement
    ratio = len(pred_set) / max(len(gt_set), 1)
    if ratio > 1.5:
        detection_rate = "over"
    elif ratio < 0.5:
        detection_rate = "under"
    else:
        detection_rate = "on_target"

    return {
        "precision":       round(precision, 4),
        "recall":          round(recall,    4),
        "f1":              round(f1,        4),
        "n_predicted":     len(pred_set),
        "n_ground_truth":  len(gt_set),
        "n_correct":       len(tp),
        "false_positives": sorted(fp),
        "false_negatives": sorted(fn),
        "detection_rate":  detection_rate,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Severity calibration
# ══════════════════════════════════════════════════════════════════════════════

def _pearson_r(xs: list[float], ys: list[float]) -> float:
    """Pure-Python Pearson correlation (fallback when numpy is unavailable)."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num   = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denom = (sum((x - mx) ** 2 for x in xs) *
             sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / denom if denom else 0.0


def severity_calibration(
    gap_severities: list[str | None],
    quiz_outcomes:  list[int],
) -> dict:
    """
    Measure how well gap severity predicts quiz failure.

    A well-calibrated detector should produce:
      - high severity → student fails the corresponding quiz item
      - low / none    → student passes

    We correlate severity scores (high=3, medium=2, low=1, none=0) with
    quiz failure (1 - quiz_outcome) using point-biserial correlation.

    Parameters
    ----------
    gap_severities : severity strings per gap, aligned with quiz_outcomes
    quiz_outcomes  : 0 = wrong, 1 = correct — one per gap

    Returns
    -------
    {
        "point_biserial_r":          float,
        "severity_distribution":     dict,
        "failure_rate_by_severity":  dict,
        "n_samples":                 int,
        "calibration_label":         str,   # "well" | "moderately" | "weakly" | "mis"
    }
    """
    if len(gap_severities) != len(quiz_outcomes) or not gap_severities:
        return {"error": "mismatched or empty inputs", "n_samples": len(gap_severities)}

    sev_nums = [SEVERITY_SCORE.get(s, 0) for s in gap_severities]
    failures = [1 - o for o in quiz_outcomes]

    try:
        import numpy as np
        arr_sev  = np.array(sev_nums,  dtype=float)
        arr_fail = np.array(failures,  dtype=float)
        r = (0.0 if arr_sev.std() == 0 or arr_fail.std() == 0
             else float(np.corrcoef(arr_sev, arr_fail)[0, 1]))
    except ImportError:
        r = _pearson_r([float(x) for x in sev_nums],
                       [float(x) for x in failures])

    # Distributions
    sev_dist: dict[str, int] = {}
    for s in gap_severities:
        k = s or "none"
        sev_dist[k] = sev_dist.get(k, 0) + 1

    fail_by_sev: dict[str, dict[str, int]] = {}
    for sev, outcome in zip(gap_severities, quiz_outcomes):
        k = sev or "none"
        fail_by_sev.setdefault(k, {"total": 0, "fail": 0})
        fail_by_sev[k]["total"] += 1
        if outcome == 0:
            fail_by_sev[k]["fail"] += 1
    fail_rates = {
        k: round(v["fail"] / v["total"], 3)
        for k, v in fail_by_sev.items()
    }

    label = (
        "well_calibrated"       if r >= 0.5 else
        "moderately_calibrated" if r >= 0.2 else
        "weakly_calibrated"     if r >= 0.0 else
        "miscalibrated"
    )

    return {
        "point_biserial_r":         round(r, 4),
        "severity_distribution":    sev_dist,
        "failure_rate_by_severity": fail_rates,
        "n_samples":                len(gap_severities),
        "calibration_label":        label,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Gap stability (reproducibility)
# ══════════════════════════════════════════════════════════════════════════════

def gap_stability(gaps1: list[str], gaps2: list[str]) -> float:
    """
    Jaccard similarity between two gap-detection outputs for the same input.

    A high score (> 0.7) means the system produces consistent results
    when called twice with identical inputs.

    Returns a float in [0, 1].
    """
    set1 = {g.lower().strip() for g in gaps1}
    set2 = {g.lower().strip() for g in gaps2}
    union = set1 | set2
    if not union:
        return 1.0  # both empty → perfectly stable
    return round(len(set1 & set2) / len(union), 4)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Aggregate stability across multiple re-evaluations
# ══════════════════════════════════════════════════════════════════════════════

def batch_stability(repeated_gap_lists: list[list[str]]) -> dict:
    """
    Compute pairwise Jaccard stability across N repeated evaluations
    of the same student quiz.

    Parameters
    ----------
    repeated_gap_lists : list of gap lists, each from a separate /find-gaps call
                         with the same input

    Returns
    -------
    {
        "mean_pairwise_jaccard": float,
        "min_pairwise_jaccard":  float,
        "n_evaluations":         int,
        "stability_label":       str,   # "high" | "medium" | "low"
    }
    """
    n = len(repeated_gap_lists)
    if n < 2:
        return {"mean_pairwise_jaccard": 1.0, "min_pairwise_jaccard": 1.0,
                "n_evaluations": n, "stability_label": "high"}

    scores: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            scores.append(gap_stability(repeated_gap_lists[i], repeated_gap_lists[j]))

    mean_j = statistics.mean(scores)
    label  = (
        "high"   if mean_j >= 0.7 else
        "medium" if mean_j >= 0.4 else
        "low"
    )
    return {
        "mean_pairwise_jaccard": round(mean_j, 4),
        "min_pairwise_jaccard":  round(min(scores), 4),
        "n_evaluations":         n,
        "stability_label":       label,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Unified gap-detection evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_gap_detection(
    predicted_gaps:    list[str],
    ground_truth_gaps: list[str],
    gap_severities:    list[str | None] | None = None,
    quiz_outcomes:     list[int] | None = None,
) -> dict:
    """
    Run the full gap-detection evaluation in a single call.

    Parameters
    ----------
    predicted_gaps    : gap concepts returned by POST /find-gaps
    ground_truth_gaps : expert-annotated correct gaps
    gap_severities    : optional — severity strings from /find-gaps response
                        (aligned with quiz_outcomes for calibration check)
    quiz_outcomes     : optional — 0/1 per quiz item (1 = answered correctly)

    Returns
    -------
    Unified dict combining precision/recall/F1 and, optionally,
    severity calibration.
    """
    result = precision_recall_f1(predicted_gaps, ground_truth_gaps)

    if gap_severities is not None and quiz_outcomes is not None:
        result["severity_calibration"] = severity_calibration(
            gap_severities, quiz_outcomes
        )

    return result
