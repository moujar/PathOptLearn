"""
A/B Testing Framework — PathOptLearn Evaluation
================================================
Compares PathOptLearn's full personalized pipeline against three baselines
by running identical student profiles through each condition and measuring
learning outcomes.

Experimental Conditions
-----------------------
  A  PATHOPTLEARN     — Full pipeline: diagnostic → personalised roadmap →
                         adaptive gap-based remediation
  B  STATIC           — No diagnostic; uses a fixed module order; no gap
                         analysis for remediation (simulates a traditional LMS)
  C  NO_REMEDIATION   — Full pipeline but patience=1; no retry resources
                         (tests value of gap-based retry assistance)
  D  SEARCH_ONLY      — No quizzes at all; "learning" is proxied by time
                         spent on resources (baseline for resource-only UX)

Statistical tests
-----------------
  - Mann-Whitney U test (non-parametric, no normality assumption)
  - Cohen's d effect size
  - 95 % bootstrap confidence intervals for mean difference

Usage
-----
  python ab_testing.py \
      --topic "Machine Learning" \
      --api   http://localhost:8000 \
      --profiles beginner intermediate \
      --n-reps 2 \
      --output ab_results.json

  Or from Python:
    from benchmakring.ab_testing import run_ab_experiment, print_ab_report
    results = run_ab_experiment("Machine Learning", ["beginner"], api, model)
    print_ab_report(results)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import requests

# ── Path setup ────────────────────────────────────────────────────────────────
_EVAL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_EVAL_ROOT))
sys.path.insert(0, str(_EVAL_ROOT / "UserSimulation"))


# ══════════════════════════════════════════════════════════════════════════════
# Condition definitions
# ══════════════════════════════════════════════════════════════════════════════

class Condition(str, Enum):
    PATHOPTLEARN    = "pathoptlearn"   # A — full system
    STATIC          = "static"         # B — no personalisation
    NO_REMEDIATION  = "no_remediation" # C — no retry resources
    SEARCH_ONLY     = "search_only"    # D — resource browsing only


# ══════════════════════════════════════════════════════════════════════════════
# Condition runners
# ══════════════════════════════════════════════════════════════════════════════

def _api_post(api: str, path: str, body: dict | None = None,
              params: dict | None = None) -> dict:
    r = requests.post(f"{api}{path}", json=body, params=params, timeout=120)
    r.raise_for_status()
    return r.json()


def _api_get(api: str, path: str, **params) -> dict:
    r = requests.get(f"{api}{path}", params=params, timeout=120)
    r.raise_for_status()
    return r.json()


def run_condition_a(topic: str, profile_name: str, api: str,
                    model: str) -> dict | None:
    """
    Condition A: full PathOptLearn pipeline.
    Delegates to the existing LLMStudent.run() implementation.
    """
    try:
        from llm_student import LLMStudent
        student = LLMStudent(profile_name=profile_name, topic=topic,
                             api=api, model=model)
        return student.run()
    except Exception as e:
        print(f"  [A] Error: {e}")
        return None


def run_condition_b(topic: str, profile_name: str, api: str,
                    model: str) -> dict | None:
    """
    Condition B: static curriculum.
    - No diagnostic — assume intermediate level for everyone
    - Fixed module order from the roadmap (no reordering based on diagnosis)
    - No gap-based remediation resources on failure
    - patience forced to 1 (one attempt per module, no gap analysis)
    """
    try:
        from llm_student import LLMStudent, PROFILES

        class StaticStudent(LLMStudent):
            """Overrides the run() method to skip diagnostics and gap analysis."""

            def run(self) -> dict:
                import time as _time
                t_start = _time.time()
                level   = "intermediate"  # static assumption

                # Get roadmap (no diagnostic first)
                roadmap   = _api_get(self.api, "/roadmap",
                                     topic=self.topic, level=level)
                modules   = roadmap.get("modules", [])
                base_topic = roadmap.get("topic", self.topic)
                self._emit("roadmap", {"n_modules": len(modules), "condition": "static"})

                module_results = []
                for mod in modules:
                    mod_id    = mod["id"]
                    mod_title = mod.get("title", f"Module {mod_id}")
                    # One attempt only — no remediation
                    lesson    = _api_get(self.api, "/lesson",
                                         topic=base_topic, module_id=mod_id,
                                         session_id="")
                    self._read_lesson(lesson.get("content", ""), mod_title)

                    quiz_raw = _api_post(self.api, "/quiz", body={
                        "content":       lesson.get("content", ""),
                        "num_questions": 5,
                    })
                    quiz_qs  = quiz_raw.get("questions", [])
                    quiz_ans = [self._answer_mcq(q) for q in quiz_qs]

                    score_data = _api_post(self.api, "/find-gaps", body={
                        "topic":     mod_title,
                        "questions": quiz_qs,
                        "answers":   quiz_ans,
                    })
                    score  = score_data.get("score", 0)
                    passed = score >= 70

                    module_results.append({
                        "module_id":    mod_id,
                        "module_title": mod_title,
                        "attempts":     1,         # always 1 in static condition
                        "final_score":  score,
                        "passed":       passed,
                    })

                elapsed   = round(_time.time() - t_start, 1)
                n_passed  = sum(1 for m in module_results if m["passed"])
                avg_score = sum(m["final_score"] for m in module_results) / max(len(module_results), 1)

                return {
                    "profile":        self.profile_name,
                    "topic":          self.topic,
                    "diag_level":     level,
                    "diag_score":     50,          # no diagnostic — neutral value
                    "diag_gaps":      [],
                    "n_modules":      len(module_results),
                    "n_passed":       n_passed,
                    "pass_rate":      round(n_passed / max(len(module_results), 1), 3),
                    "avg_score":      round(avg_score, 1),
                    "total_retries":  0,
                    "wall_time_s":    elapsed,
                    "module_results": module_results,
                    "log_entries":    len(self.log),
                    "condition":      "static",
                }

        student = StaticStudent(profile_name=profile_name, topic=topic,
                                api=api, model=model)
        return student.run()
    except Exception as e:
        print(f"  [B] Error: {e}")
        return None


def run_condition_c(topic: str, profile_name: str, api: str,
                    model: str) -> dict | None:
    """
    Condition C: no remediation — full pipeline but patience=1.
    Students get one attempt per module, no gap resources on failure.
    """
    try:
        from llm_student import LLMStudent, PROFILES
        # Override patience to 1 in the profile
        patched_profile = dict(PROFILES.get(profile_name, PROFILES["intermediate"]))
        patched_profile["patience"] = 1

        student = LLMStudent(profile_name=profile_name, topic=topic,
                             api=api, model=model)
        student.profile = patched_profile
        result = student.run()
        if result:
            result["condition"] = "no_remediation"
        return result
    except Exception as e:
        print(f"  [C] Error: {e}")
        return None


def run_condition_d(topic: str, profile_name: str, api: str,
                    model: str) -> dict | None:
    """
    Condition D: search-only (no quizzes).
    Simulates a student who only reads recommended resources.
    We measure hypothetical gain as the number of resources consumed
    and proxy score as a weak random baseline.
    """
    try:
        resources_consumed = 0
        try:
            recs = _api_get(api, "/recommender",
                            gaps=f"introduction to {topic},advanced {topic}",
                            student_id="", limit=5)
            for gap_recs in recs.get("results", {}).values():
                resources_consumed += len(gap_recs)
        except Exception:
            resources_consumed = random.randint(2, 6)

        # No quizzes → proxy score via profile error rate
        from llm_student import PROFILES
        profile = PROFILES.get(profile_name, PROFILES["intermediate"])
        proxy_score = round((1 - profile["error_rate"]) * 100 * random.uniform(0.85, 1.0), 1)

        return {
            "profile":        profile_name,
            "topic":          topic,
            "diag_level":     "unknown",
            "diag_score":     0,
            "diag_gaps":      [],
            "n_modules":      0,
            "n_passed":       0,
            "pass_rate":      0.0,
            "avg_score":      proxy_score,
            "total_retries":  0,
            "wall_time_s":    resources_consumed * 5.0,  # proxy: 5s per resource
            "module_results": [],
            "log_entries":    0,
            "resources_consumed": resources_consumed,
            "condition":      "search_only",
        }
    except Exception as e:
        print(f"  [D] Error: {e}")
        return None


CONDITION_RUNNERS = {
    Condition.PATHOPTLEARN:   run_condition_a,
    Condition.STATIC:         run_condition_b,
    Condition.NO_REMEDIATION: run_condition_c,
    Condition.SEARCH_ONLY:    run_condition_d,
}


# ══════════════════════════════════════════════════════════════════════════════
# Statistical tests
# ══════════════════════════════════════════════════════════════════════════════

def cohen_d(a: list[float], b: list[float]) -> float:
    """Cohen's d effect size (pooled SD)."""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    na, nb  = len(a), len(b)
    ma, mb  = np.mean(a), np.mean(b)
    va, vb  = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled  = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return round(float((ma - mb) / pooled), 4) if pooled > 0 else 0.0


def mann_whitney_u(a: list[float], b: list[float]) -> dict:
    """
    Mann-Whitney U test (non-parametric comparison of two samples).
    Returns U statistic, p-value approximation, and effect size r.
    """
    try:
        from scipy.stats import mannwhitneyu
        stat, p = mannwhitneyu(a, b, alternative="two-sided")
        n       = len(a) * len(b)
        r       = round(float(stat / n - 0.5) * 2, 4)  # rank-biserial r
        return {"u_statistic": round(float(stat), 4),
                "p_value":     round(float(p), 6),
                "effect_r":    r,
                "significant": p < 0.05}
    except ImportError:
        return {"note": "scipy not available — install scipy for p-values",
                "mean_a": round(float(np.mean(a)), 3),
                "mean_b": round(float(np.mean(b)), 3)}


def bootstrap_ci(a: list[float], n_boot: int = 1000,
                 ci: float = 0.95) -> dict:
    """Bootstrap confidence interval for the mean of ``a``."""
    if not a:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    arr = np.array(a)
    boot_means = [np.mean(np.random.choice(arr, len(arr), replace=True))
                  for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    low   = float(np.percentile(boot_means, alpha * 100))
    high  = float(np.percentile(boot_means, (1 - alpha) * 100))
    return {
        "mean":    round(float(arr.mean()), 3),
        "ci_low":  round(low,  3),
        "ci_high": round(high, 3),
        "ci_pct":  int(ci * 100),
    }


# ══════════════════════════════════════════════════════════════════════════════
# A/B experiment runner
# ══════════════════════════════════════════════════════════════════════════════

def run_ab_experiment(
    topic: str,
    profiles: list[str],
    api: str,
    model: str,
    conditions: list[Condition] | None = None,
    n_reps: int = 1,
) -> dict:
    """
    Run all conditions for each profile and return a structured results dict.

    Parameters
    ----------
    topic      : learning topic
    profiles   : list of student profile names
    api        : PathOptLearn API base URL
    model      : Ollama model name
    conditions : conditions to run (default: all four)
    n_reps     : number of repetitions per (profile, condition) pair

    Returns
    -------
    {
        "topic":       str,
        "profiles":    list[str],
        "conditions":  list[str],
        "raw_results": {
            "<condition>": {
                "<profile>": [<result_dict>, ...]
            }
        },
        "comparison":  {   # per (condition_a, condition_b) pair
            "<cond_a> vs <cond_b>": {
                "metric":       str,
                "mann_whitney": dict,
                "cohen_d":      float,
                "ci_a":         dict,
                "ci_b":         dict,
            }
        }
    }
    """
    if conditions is None:
        conditions = list(Condition)

    raw: dict[str, dict[str, list[dict]]] = {c.value: {} for c in conditions}

    total = len(conditions) * len(profiles) * n_reps
    done  = 0

    for cond in conditions:
        runner = CONDITION_RUNNERS[cond]
        for profile in profiles:
            raw[cond.value].setdefault(profile, [])
            for rep in range(n_reps):
                done += 1
                print(f"[A/B] ({done}/{total}) condition={cond.value} "
                      f"profile={profile} rep={rep+1}")
                result = runner(topic, profile, api, model)
                if result:
                    result["condition"] = cond.value
                    raw[cond.value][profile].append(result)
                time.sleep(0.2)

    # Statistical comparison: A vs each other condition
    comparison: dict[str, Any] = {}
    ref_cond = Condition.PATHOPTLEARN.value

    for cond in conditions:
        if cond.value == ref_cond:
            continue
        key = f"{ref_cond} vs {cond.value}"

        # Collect avg_score across all profiles
        a_scores = [
            r.get("avg_score", 0)
            for profile_results in raw[ref_cond].values()
            for r in profile_results
        ]
        b_scores = [
            r.get("avg_score", 0)
            for profile_results in raw[cond.value].values()
            for r in profile_results
        ]
        a_pass = [
            r.get("pass_rate", 0)
            for profile_results in raw[ref_cond].values()
            for r in profile_results
        ]
        b_pass = [
            r.get("pass_rate", 0)
            for profile_results in raw[cond.value].values()
            for r in profile_results
        ]

        comparison[key] = {
            "metric":          "avg_score",
            "mann_whitney":    mann_whitney_u(a_scores, b_scores),
            "cohen_d":         cohen_d(a_scores, b_scores),
            "bootstrap_ci_a":  bootstrap_ci(a_scores),
            "bootstrap_ci_b":  bootstrap_ci(b_scores),
            "pass_rate_ci_a":  bootstrap_ci(a_pass),
            "pass_rate_ci_b":  bootstrap_ci(b_pass),
        }

    return {
        "topic":       topic,
        "profiles":    profiles,
        "conditions":  [c.value for c in conditions],
        "raw_results": raw,
        "comparison":  comparison,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════════════

def print_ab_report(results: dict) -> str:
    """Print and return a human-readable A/B test report."""
    lines = [
        "",
        "═" * 72,
        "  PathOptLearn — A/B Test Report",
        f"  Topic   : {results.get('topic', '?')}",
        f"  Profiles: {', '.join(results.get('profiles', []))}",
        "═" * 72,
        "",
        "  Condition summary (avg score ± CI):",
        "",
    ]

    raw = results.get("raw_results", {})
    for cond_name, profiles in raw.items():
        all_scores = [r.get("avg_score", 0)
                      for rlist in profiles.values() for r in rlist]
        all_pass   = [r.get("pass_rate", 0)
                      for rlist in profiles.values() for r in rlist]
        if not all_scores:
            continue
        ci     = bootstrap_ci(all_scores)
        ci_p   = bootstrap_ci(all_pass)
        marker = " ◀ reference" if cond_name == Condition.PATHOPTLEARN.value else ""
        lines.append(
            f"  {cond_name:<20}  "
            f"avg_score={ci['mean']:>5.1f}  "
            f"[{ci['ci_low']:.1f}–{ci['ci_high']:.1f}]   "
            f"pass={ci_p['mean']:.1%}  "
            f"n={len(all_scores)}{marker}"
        )

    lines += ["", "  Pairwise comparisons (PathOptLearn vs. baseline):", ""]

    for pair, comp in results.get("comparison", {}).items():
        mw  = comp.get("mann_whitney", {})
        cd  = comp.get("cohen_d", 0)
        sig = "✓ significant" if mw.get("significant") else "✗ not significant"
        p   = mw.get("p_value", "?")
        lines.append(f"  {pair}")
        lines.append(f"    Cohen's d = {cd:+.3f}  |  Mann-Whitney p = {p}  "
                     f"|  {sig}")
        ci_a = comp.get("bootstrap_ci_a", {})
        ci_b = comp.get("bootstrap_ci_b", {})
        lines.append(f"    A: {ci_a.get('mean', 0):.1f}  "
                     f"[{ci_a.get('ci_low',0):.1f}–{ci_a.get('ci_high',0):.1f}]   "
                     f"B: {ci_b.get('mean', 0):.1f}  "
                     f"[{ci_b.get('ci_low',0):.1f}–{ci_b.get('ci_high',0):.1f}]")
        lines.append("")

    lines.append("═" * 72 + "\n")
    report = "\n".join(lines)
    print(report)
    return report


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="PathOptLearn A/B testing framework"
    )
    parser.add_argument("--topic",      default="Machine Learning")
    parser.add_argument("--api",        default="http://localhost:8000")
    parser.add_argument("--model",      default="llama3.2:1b")
    parser.add_argument("--profiles",   nargs="+",
                        default=["beginner", "intermediate"])
    parser.add_argument("--conditions", nargs="+",
                        choices=[c.value for c in Condition],
                        default=[c.value for c in Condition])
    parser.add_argument("--n-reps",     type=int, default=1,
                        help="Repetitions per (profile, condition) cell")
    parser.add_argument("--output",     default="ab_results.json")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    conds = [Condition(c) for c in args.conditions]
    results = run_ab_experiment(
        topic      = args.topic,
        profiles   = args.profiles,
        api        = args.api,
        model      = args.model,
        conditions = conds,
        n_reps     = args.n_reps,
    )

    print_ab_report(results)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[A/B] Results saved → {args.output}")


if __name__ == "__main__":
    main()
