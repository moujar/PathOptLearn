"""
Learning Effectiveness Metrics — PathOptLearn Evaluation
=========================================================
Computes metrics that measure how effectively PathOptLearn helps students
acquire knowledge. All functions accept simulation summary dicts produced
by evalution/UserSimulation/llm_student.py or equivalent session data
loaded from the PostgreSQL PROGRESS / TOPIC_MASTERY tables.

Simulation summary schema (from llm_student.py)
------------------------------------------------
{
    "profile":        str,          # student profile name
    "topic":          str,
    "diag_score":     int,          # 0-100, diagnostic quiz score
    "diag_level":     str,          # beginner / intermediate / advanced
    "diag_gaps":      list[str],    # gaps identified in diagnostic
    "n_modules":      int,
    "n_passed":       int,
    "pass_rate":      float,        # 0.0 – 1.0
    "avg_score":      float,        # mean final score across modules
    "total_retries":  int,
    "wall_time_s":    float,
    "module_results": [
        {
            "module_id":    int,
            "module_title": str,
            "attempts":     int,    # total attempts made
            "final_score":  float,  # score on the last attempt (0-100)
            "passed":       bool,
        },
        ...
    ]
}
"""

from __future__ import annotations

import statistics
from typing import Any

MASTERY_THRESHOLD = 70.0   # % score required to pass a module


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Normalised Learning Gain (Hake, 1998)
# ══════════════════════════════════════════════════════════════════════════════

def normalized_learning_gain(
    pre_score: float,
    post_score: float,
    max_score: float = 100.0,
) -> float:
    """
    Hake's normalised learning gain:

        g = (post - pre) / (max - pre)

    Interpretation
    --------------
    g >= 0.7  →  high gain
    0.3 <= g  →  medium gain
    g < 0.3   →  low gain
    g < 0     →  regression (post < pre)

    Returns float('nan') when pre_score == max_score (ceiling effect).
    """
    if pre_score >= max_score:
        return float("nan")
    return (post_score - pre_score) / (max_score - pre_score)


def classify_learning_gain(g: float) -> str:
    """Return a human-readable label for a normalised learning gain value."""
    if g != g:          # NaN
        return "ceiling"
    if g >= 0.7:
        return "high"
    if g >= 0.3:
        return "medium"
    if g >= 0.0:
        return "low"
    return "regression"


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Module success rates
# ══════════════════════════════════════════════════════════════════════════════

def module_success_rate(sim_results: list[dict]) -> dict:
    """
    Analyse first-attempt vs. eventual pass rates across a cohort.

    Returns
    -------
    {
        "first_attempt_pass_rate": float,   # passed on attempt 1
        "eventual_pass_rate":      float,   # passed eventually (any attempt)
        "never_passed_rate":       float,   # exhausted patience and failed
        "total_modules":           int,
        "total_students":          int,
    }
    """
    first_pass = 0
    eventual   = 0
    never      = 0
    total      = 0

    for sim in sim_results:
        for mod in sim.get("module_results", []):
            total += 1
            if mod.get("passed"):
                eventual += 1
                if mod.get("attempts", 1) == 1:
                    first_pass += 1
            else:
                never += 1

    if total == 0:
        return {
            "first_attempt_pass_rate": 0.0,
            "eventual_pass_rate":      0.0,
            "never_passed_rate":       0.0,
            "total_modules":           0,
            "total_students":          0,
        }

    return {
        "first_attempt_pass_rate": round(first_pass / total, 4),
        "eventual_pass_rate":      round(eventual   / total, 4),
        "never_passed_rate":       round(never      / total, 4),
        "total_modules":           total,
        "total_students":          len(sim_results),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Quiz improvement rate (per-retry score delta)
# ══════════════════════════════════════════════════════════════════════════════

def quiz_improvement_rate(sim_results: list[dict]) -> dict:
    """
    For modules that required multiple attempts, estimate how much the score
    improved per retry.

    Since llm_student.py only records ``final_score`` (last attempt) and
    ``attempts`` count, we use a conservative proxy for modules that eventually
    passed:

        improvement_per_retry = (final_score - MASTERY_THRESHOLD) / (attempts - 1)

    Returns
    -------
    {
        "mean_improvement_per_retry": float,   # proxy score gain per extra attempt
        "n_multi_attempt_modules":    int,
        "retry_pass_rate":            float,   # of retried modules, fraction that pass
    }
    """
    deltas       = []
    retry_pass   = 0
    retry_total  = 0

    for sim in sim_results:
        for mod in sim.get("module_results", []):
            if mod.get("attempts", 1) > 1:
                retry_total += 1
                if mod.get("passed"):
                    retry_pass += 1
                    delta = (mod["final_score"] - MASTERY_THRESHOLD) / (mod["attempts"] - 1)
                    deltas.append(max(delta, 0.0))

    return {
        "mean_improvement_per_retry": round(statistics.mean(deltas), 3) if deltas else 0.0,
        "n_multi_attempt_modules":    retry_total,
        "retry_pass_rate":            round(retry_pass / max(retry_total, 1), 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Time-to-mastery
# ══════════════════════════════════════════════════════════════════════════════

def time_to_mastery(
    sim_results: list[dict],
    threshold: float = MASTERY_THRESHOLD,
) -> dict:
    """
    Distribution of attempts (and wall-clock time) needed to reach mastery.

    Modules that never reach threshold get a DNF value of patience + 1.

    Returns
    -------
    {
        "mean_attempts":        float,
        "median_attempts":      float,
        "max_attempts":         int,
        "mastered_1st_try_pct": float,   # % modules mastered on attempt 1
        "never_mastered_pct":   float,   # % modules never mastered
        "mean_wall_time_s":     float,   # mean student wall-clock time
    }
    """
    all_attempts: list[int] = []
    never = 0

    for sim in sim_results:
        patience = sim.get("patience", 3)
        for mod in sim.get("module_results", []):
            if mod.get("passed") and mod.get("final_score", 0) >= threshold:
                all_attempts.append(mod.get("attempts", 1))
            else:
                all_attempts.append(patience + 1)
                never += 1

    wall_times = [s.get("wall_time_s", 0) for s in sim_results if s.get("wall_time_s")]

    if not all_attempts:
        return {
            "mean_attempts": 0, "median_attempts": 0, "max_attempts": 0,
            "mastered_1st_try_pct": 0.0, "never_mastered_pct": 0.0,
            "mean_wall_time_s": 0.0,
        }

    return {
        "mean_attempts":        round(statistics.mean(all_attempts), 3),
        "median_attempts":      float(statistics.median(all_attempts)),
        "max_attempts":         int(max(all_attempts)),
        "mastered_1st_try_pct": round(sum(1 for a in all_attempts if a == 1) / len(all_attempts), 4),
        "never_mastered_pct":   round(never / len(all_attempts), 4),
        "mean_wall_time_s":     round(statistics.mean(wall_times), 2) if wall_times else 0.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Completion rate
# ══════════════════════════════════════════════════════════════════════════════

def completion_rate(sim_results: list[dict]) -> dict:
    """
    Fraction of students who complete all modules within their patience limit.

    Returns
    -------
    {
        "full_completion_rate":    float,   # all modules passed
        "partial_completion_rate": float,   # >= 50 % modules passed
        "dropout_rate":            float,   # < 50 % modules passed
        "mean_completion_pct":     float,   # average % of modules completed
    }
    """
    if not sim_results:
        return {
            "full_completion_rate":    0.0,
            "partial_completion_rate": 0.0,
            "dropout_rate":            0.0,
            "mean_completion_pct":     0.0,
        }

    full = partial = dropout = 0
    pcts: list[float] = []

    for sim in sim_results:
        n_mods   = sim.get("n_modules", 0)
        n_passed = sim.get("n_passed",  0)
        pct = n_passed / max(n_mods, 1)
        pcts.append(pct)

        if pct >= 1.0:
            full += 1
        elif pct >= 0.5:
            partial += 1
        else:
            dropout += 1

    n = len(sim_results)
    return {
        "full_completion_rate":    round(full    / n, 4),
        "partial_completion_rate": round(partial / n, 4),
        "dropout_rate":            round(dropout / n, 4),
        "mean_completion_pct":     round(statistics.mean(pcts), 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Learning trajectory
# ══════════════════════════════════════════════════════════════════════════════

def learning_trajectory(sim_result: dict) -> list[float]:
    """
    Return the ordered sequence of quiz scores across the learning journey.

    Format: [diag_score, module_1_final, module_2_final, ...]

    Useful for plotting how a student's performance evolves.
    """
    scores = [float(sim_result.get("diag_score", 0))]
    for mod in sim_result.get("module_results", []):
        scores.append(float(mod.get("final_score", 0)))
    return scores


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Knowledge retention proxy (diagnostic alignment)
# ══════════════════════════════════════════════════════════════════════════════

def knowledge_retention_proxy(sim_results: list[dict]) -> dict:
    """
    Measure alignment between diagnostic score and mean final module score.

    A positive Pearson r means the diagnostic correctly ordered learners —
    students who scored higher on the diagnostic also scored higher on modules.

    Returns
    -------
    {
        "pearson_r":        float,
        "mean_diag_score":  float,
        "mean_final_score": float,
        "mean_uplift":      float,   # avg(final_score) - avg(diag_score)
    }
    """
    diag_scores:  list[float] = []
    final_scores: list[float] = []

    for sim in sim_results:
        mods = sim.get("module_results", [])
        if not mods:
            continue
        diag_scores.append(float(sim.get("diag_score", 50)))
        final_scores.append(
            statistics.mean(m.get("final_score", 0) for m in mods)
        )

    if len(diag_scores) < 2:
        return {"pearson_r": None, "note": "insufficient data (need >= 2 students)"}

    try:
        import numpy as np
        d = np.array(diag_scores,  dtype=float)
        f = np.array(final_scores, dtype=float)
        r = float(np.corrcoef(d, f)[0, 1])
    except ImportError:
        # Pure-Python fallback
        mx = statistics.mean(diag_scores)
        my = statistics.mean(final_scores)
        num   = sum((x - mx) * (y - my) for x, y in zip(diag_scores, final_scores))
        denom = (sum((x - mx) ** 2 for x in diag_scores) *
                 sum((y - my) ** 2 for y in final_scores)) ** 0.5
        r = (num / denom) if denom else 0.0

    return {
        "pearson_r":        round(r, 4),
        "mean_diag_score":  round(statistics.mean(diag_scores),  2),
        "mean_final_score": round(statistics.mean(final_scores), 2),
        "mean_uplift":      round(
            statistics.mean(f - d for f, d in zip(final_scores, diag_scores)), 2
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 8.  Per-profile cohort statistics
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_cohort_stats(sim_results: list[dict]) -> dict:
    """
    High-level statistics across a cohort broken down by student profile.

    Returns
    -------
    {
        "n_students":     int,
        "mean_pass_rate": float,
        "std_pass_rate":  float,
        "mean_avg_score": float,
        "mean_retries":   float,
        "per_profile":    {
            "<profile>": {
                "mean_pass_rate": float,
                "mean_avg_score": float,
                "mean_retries":   float,
                "n_runs":         int,
            }
        }
    }
    """
    if not sim_results:
        return {}

    profiles: dict[str, dict] = {}
    all_pass: list[float] = []
    all_score: list[float] = []
    all_retry: list[int]   = []

    for sim in sim_results:
        p = sim.get("profile", "unknown")
        profiles.setdefault(p, {"pass_rates": [], "avg_scores": [], "retries": []})
        profiles[p]["pass_rates"].append(sim.get("pass_rate", 0.0))
        profiles[p]["avg_scores"].append(sim.get("avg_score", 0.0))
        profiles[p]["retries"].append(sim.get("total_retries", 0))
        all_pass.append(sim.get("pass_rate", 0.0))
        all_score.append(sim.get("avg_score", 0.0))
        all_retry.append(sim.get("total_retries", 0))

    per_profile = {
        p: {
            "mean_pass_rate":  round(statistics.mean(v["pass_rates"]),  4),
            "mean_avg_score":  round(statistics.mean(v["avg_scores"]),  2),
            "mean_retries":    round(statistics.mean(v["retries"]),     2),
            "n_runs":          len(v["pass_rates"]),
        }
        for p, v in profiles.items()
    }

    return {
        "n_students":     len(sim_results),
        "mean_pass_rate": round(statistics.mean(all_pass),  4),
        "std_pass_rate":  round(
            statistics.stdev(all_pass) if len(all_pass) > 1 else 0.0, 4
        ),
        "mean_avg_score": round(statistics.mean(all_score), 2),
        "mean_retries":   round(statistics.mean(all_retry), 2),
        "per_profile":    per_profile,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 9.  Unified metric computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_all_learning_metrics(sim_results: list[dict]) -> dict:
    """
    Compute every learning metric for a cohort in a single call.

    Returns a unified dict suitable for JSON export or experiment tracking.
    """
    # Per-student normalised learning gain (diag → mean final)
    nlg_values: list[float] = []
    nlg_labels: list[str]   = []
    for sim in sim_results:
        mods = sim.get("module_results", [])
        if not mods:
            continue
        post = statistics.mean(m.get("final_score", 0) for m in mods)
        g    = normalized_learning_gain(sim.get("diag_score", 0), post)
        nlg_values.append(g)
        nlg_labels.append(classify_learning_gain(g))

    valid_nlg = [v for v in nlg_values if v == v]  # exclude NaN
    nlg_summary = {
        "mean":   round(statistics.mean(valid_nlg), 4) if valid_nlg else None,
        "labels": {lbl: nlg_labels.count(lbl) for lbl in
                   ("high", "medium", "low", "regression", "ceiling")},
    }

    return {
        "learning_gain":          nlg_summary,
        "module_success":         module_success_rate(sim_results),
        "quiz_improvement":       quiz_improvement_rate(sim_results),
        "time_to_mastery":        time_to_mastery(sim_results),
        "completion":             completion_rate(sim_results),
        "cohort_stats":           aggregate_cohort_stats(sim_results),
        "knowledge_retention":    knowledge_retention_proxy(sim_results),
        "trajectories":           {
            f"{s.get('profile', '?')}_{i}": learning_trajectory(s)
            for i, s in enumerate(sim_results)
        },
    }
