"""
Evaluation Runner — PathOptLearn Master Pipeline
=================================================
Orchestrates the full evaluation suite:

  1. Student-simulation evaluation        (LLM student → learning metrics)
  2. LLM content quality evaluation       (lesson + quiz quality)
  3. Resource recommendation evaluation   (relevance, diversity, coverage)
  4. Knowledge gap detection evaluation   (precision, recall, F1)

All results are stored in the ExperimentTracker (SQLite) and written to
``output_dir`` as JSON/CSV files.

Usage
-----
  python eval_runner.py --topic "Machine Learning" --profiles beginner intermediate
  python eval_runner.py --topic "Sorting Algorithms" --no-llm-quality

  Or from Python:
    from pipeline.eval_runner import EvalConfig, EvalRunner
    from pipeline.experiment_tracker import ExperimentTracker

    cfg     = EvalConfig(topic="Machine Learning", profiles=["beginner", "advanced"])
    tracker = ExperimentTracker("eval_output/results.db")
    runner  = EvalRunner(cfg, tracker)
    results = runner.run_all()
    runner.generate_report(results)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

# ── Path setup so sibling modules resolve correctly ───────────────────────────
_EVAL_ROOT = Path(__file__).resolve().parent.parent   # evalution/
sys.path.insert(0, str(_EVAL_ROOT))
sys.path.insert(0, str(_EVAL_ROOT / "metrics"))
sys.path.insert(0, str(_EVAL_ROOT / "UserSimulation"))

from metrics.learning_metrics  import compute_all_learning_metrics
from metrics.llm_quality       import (batch_evaluate_lessons,
                                        evaluate_quiz_quality,
                                        summarize_lesson_quality)
from metrics.resource_eval     import evaluate_resource_list
from metrics.gap_eval          import evaluate_gap_detection
from pipeline.experiment_tracker import ExperimentTracker


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalConfig:
    """All tunable parameters for an evaluation run."""

    # System under test
    api_url: str = "http://localhost:8000"
    topic:   str = "Machine Learning"

    # Simulation settings
    profiles:          list[str] = field(default_factory=lambda: ["beginner", "intermediate", "advanced"])
    n_runs_per_profile: int      = 1
    run_simulation:    bool      = True

    # LLM quality evaluation
    run_llm_quality: bool = True
    eval_n_lessons:  int  = 2     # number of lessons to fetch and evaluate
    ollama_model:    str  = "llama3.2:1b"
    ollama_host:     str  = "http://localhost:11434"

    # Resource evaluation
    run_resource_eval: bool = True

    # Gap detection evaluation
    run_gap_eval: bool = True

    # Output
    output_dir:      str        = "eval_output"
    experiment_name: str | None = None
    seed:            int        = 42


# ══════════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════════

class EvalRunner:
    """Master orchestrator for the PathOptLearn evaluation pipeline."""

    def __init__(self, config: EvalConfig, tracker: ExperimentTracker):
        self.cfg     = config
        self.tracker = tracker
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        # Set environment variables for Ollama so metrics modules pick them up
        os.environ["OLLAMA_HOST"]  = config.ollama_host
        os.environ["OLLAMA_MODEL"] = config.ollama_model

    # ── API helpers ────────────────────────────────────────────────────────────

    def _get(self, path: str, **params) -> dict:
        r = requests.get(f"{self.cfg.api_url}{path}", params=params, timeout=120)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: dict | None = None,
              params: dict | None = None) -> dict:
        r = requests.post(f"{self.cfg.api_url}{path}", json=body,
                          params=params, timeout=120)
        r.raise_for_status()
        return r.json()

    def _api_available(self) -> bool:
        try:
            requests.get(f"{self.cfg.api_url}/docs", timeout=5)
            return True
        except Exception:
            return False

    # ── 1.  Student simulation evaluation ─────────────────────────────────────

    def run_simulation_eval(self, run_id: str) -> dict:
        """
        Run LLM student simulations and compute all learning metrics.

        Imports llm_student.run_batch() dynamically so this module works
        even when Ollama is not available.
        """
        print("\n[EvalRunner] ── Student Simulation Evaluation ──")
        try:
            from llm_student import run_batch, PROFILES
        except ImportError as e:
            print(f"  [WARN] Could not import llm_student: {e}")
            return {"error": str(e)}

        all_summaries: list[dict] = []
        for profile in self.cfg.profiles:
            for _ in range(self.cfg.n_runs_per_profile):
                try:
                    results = run_batch(
                        topic         = self.cfg.topic,
                        api           = self.cfg.api_url,
                        profile_names = [profile],
                        model         = self.cfg.ollama_model,
                        output_path   = None,
                    )
                    all_summaries.extend(results)
                except Exception as e:
                    print(f"  [ERROR] Simulation for '{profile}' failed: {e}")

        if not all_summaries:
            return {"error": "no simulation results"}

        metrics = compute_all_learning_metrics(all_summaries)

        # Log flat scalars
        self.tracker.log_nested_metrics(run_id, metrics, prefix="sim.")

        # Store full simulation summaries as artefact
        self.tracker.log_artifact(run_id, "sim_summaries", all_summaries)

        # Write to file
        out = Path(self.cfg.output_dir) / "sim_results.json"
        out.write_text(json.dumps({"config": {
            "topic": self.cfg.topic,
            "profiles": self.cfg.profiles,
        }, "metrics": metrics, "summaries": all_summaries}, indent=2, default=str))
        print(f"  Simulation results → {out}")

        return metrics

    # ── 2.  LLM content quality evaluation ────────────────────────────────────

    def run_llm_quality_eval(self, run_id: str) -> dict:
        """
        Fetch N lessons from the API and evaluate their quality with Ollama.
        """
        print("\n[EvalRunner] ── LLM Content Quality Evaluation ──")

        # Step 1: get roadmap to discover module IDs
        try:
            roadmap = self._get("/roadmap", topic=self.cfg.topic, level="intermediate")
            modules = roadmap.get("modules", [])[:self.cfg.eval_n_lessons]
        except Exception as e:
            print(f"  [ERROR] Could not fetch roadmap: {e}")
            return {"error": str(e)}

        if not modules:
            print("  [WARN] No modules in roadmap — skipping quality eval")
            return {"error": "empty roadmap"}

        # Step 2: fetch each lesson
        lessons_to_eval: list[dict] = []
        quizzes_to_eval: list[dict] = []

        for mod in modules:
            mod_id = mod.get("id")
            try:
                lesson = self._get("/lesson",
                                   topic=self.cfg.topic,
                                   module_id=mod_id,
                                   session_id="")
                lessons_to_eval.append({
                    "content":   lesson.get("content", ""),
                    "objective": mod.get("objective", ""),
                    "topic":     self.cfg.topic,
                    "level":     "intermediate",
                    "module_id": mod_id,
                })

                # Generate quiz for this module
                quiz_raw = self._post("/quiz", body={
                    "content":       lesson.get("content", ""),
                    "num_questions": 5,
                })
                quizzes_to_eval.append({
                    "module_id": mod_id,
                    "questions": quiz_raw.get("questions", []),
                })
                time.sleep(0.5)
            except Exception as e:
                print(f"  [ERROR] Could not fetch lesson for module {mod_id}: {e}")

        if not lessons_to_eval:
            return {"error": "no lessons fetched"}

        # Step 3: evaluate lesson quality
        lesson_evals   = batch_evaluate_lessons(lessons_to_eval, model=self.cfg.ollama_model)
        lesson_summary = summarize_lesson_quality(lesson_evals)

        # Step 4: evaluate quiz quality
        quiz_eval_results = [
            {"module_id": q["module_id"],
             **evaluate_quiz_quality(q["questions"], model=self.cfg.ollama_model)}
            for q in quizzes_to_eval
        ]
        n_quiz = len(quiz_eval_results)
        quiz_summary = {
            "n_quizzes":                     n_quiz,
            "mean_format_valid_pct":         round(
                sum(r.get("format_valid_pct", 0) for r in quiz_eval_results) / max(n_quiz, 1), 4
            ),
            "mean_distractor_quality_score": round(
                sum(r.get("distractor_quality_score", 0) for r in quiz_eval_results) / max(n_quiz, 1), 3
            ),
        }

        result = {
            "lesson_quality": lesson_summary,
            "quiz_quality":   quiz_summary,
        }

        self.tracker.log_nested_metrics(run_id, result, prefix="quality.")
        self.tracker.log_artifact(run_id, "lesson_evals",   lesson_evals)
        self.tracker.log_artifact(run_id, "quiz_evals",     quiz_eval_results)

        out = Path(self.cfg.output_dir) / "quality_results.json"
        out.write_text(json.dumps(result, indent=2, default=str))
        print(f"  Quality results → {out}")

        return result

    # ── 3.  Resource recommendation evaluation ─────────────────────────────────

    def run_resource_eval(self, run_id: str) -> dict:
        """
        Fetch resources for a set of synthetic gaps and evaluate quality.
        """
        print("\n[EvalRunner] ── Resource Recommendation Evaluation ──")

        # Synthesise representative gaps from the topic
        synthetic_gaps = [
            f"introduction to {self.cfg.topic}",
            f"advanced {self.cfg.topic} concepts",
            f"{self.cfg.topic} practical applications",
        ]

        resources_by_gap: dict[str, list[dict]] = {}
        for gap in synthetic_gaps:
            try:
                rec_data = self._get("/recommender",
                                     gaps=gap,
                                     student_id="",
                                     limit=3)
                # /recommender returns {"results": {"gap": [resources]}}
                for g, recs in rec_data.get("results", {}).items():
                    resources_by_gap[g] = recs
                time.sleep(0.5)
            except Exception as e:
                print(f"  [ERROR] Recommender call for '{gap}' failed: {e}")
                resources_by_gap[gap] = []

        if not any(resources_by_gap.values()):
            return {"error": "no resources returned"}

        used_gaps = list(resources_by_gap.keys())
        result    = evaluate_resource_list(used_gaps, resources_by_gap)

        self.tracker.log_nested_metrics(run_id, result, prefix="resource.")
        self.tracker.log_artifact(run_id, "resource_eval", result)

        out = Path(self.cfg.output_dir) / "resource_results.json"
        out.write_text(json.dumps(result, indent=2, default=str))
        print(f"  Resource results → {out}")

        return result

    # ── 4.  Gap detection evaluation ──────────────────────────────────────────

    def run_gap_eval(self, run_id: str) -> dict:
        """
        Run a controlled gap-detection test using a synthetic quiz where
        we know the ground truth gaps.
        """
        print("\n[EvalRunner] ── Knowledge Gap Detection Evaluation ──")

        # A minimal synthetic quiz for the topic with known ground-truth gaps
        questions = [
            {"question":    f"What is the core idea of {self.cfg.topic}?",
             "options":     ["A. Correct definition", "B. Wrong def 1",
                             "C. Wrong def 2",     "D. Wrong def 3"],
             "answer":      "A",
             "concept":     f"{self.cfg.topic} fundamentals",
             "explanation": ""},
            {"question":    f"Which technique is central to {self.cfg.topic}?",
             "options":     ["A. Unrelated concept", "B. Core technique",
                             "C. Wrong technique",  "D. Partial answer"],
             "answer":      "B",
             "concept":     f"{self.cfg.topic} key techniques",
             "explanation": ""},
        ]
        # Simulate a struggling student (answers wrong on both)
        wrong_answers = ["B", "A"]

        try:
            gap_resp = self._post("/find-gaps", body={
                "topic":     self.cfg.topic,
                "questions": questions,
                "answers":   wrong_answers,
            })
            predicted_gaps = [g["concept"] for g in gap_resp.get("gaps", [])]
            severities     = [g.get("severity") for g in gap_resp.get("gaps", [])]
        except Exception as e:
            print(f"  [ERROR] /find-gaps call failed: {e}")
            return {"error": str(e)}

        # Ground truth: a student who answered both questions wrong should have
        # gaps in the core concept areas
        ground_truth = [
            f"{self.cfg.topic} fundamentals",
            f"{self.cfg.topic} key techniques",
        ]

        # Quiz outcomes (both wrong → 0, 0)
        quiz_outcomes = [0, 0]

        result = evaluate_gap_detection(
            predicted_gaps    = predicted_gaps,
            ground_truth_gaps = ground_truth,
            gap_severities    = severities[:len(quiz_outcomes)],
            quiz_outcomes     = quiz_outcomes,
        )
        result["predicted_gaps"] = predicted_gaps
        result["n_gaps_found"]   = len(predicted_gaps)

        self.tracker.log_nested_metrics(run_id, result, prefix="gap.")
        self.tracker.log_artifact(run_id, "gap_eval", result)

        out = Path(self.cfg.output_dir) / "gap_results.json"
        out.write_text(json.dumps(result, indent=2, default=str))
        print(f"  Gap results → {out}")

        return result

    # ── 5.  Master orchestrator ────────────────────────────────────────────────

    def run_all(self) -> dict:
        """
        Run every enabled evaluation dimension and return a unified results dict.
        """
        name    = self.cfg.experiment_name or f"eval_{self.cfg.topic.replace(' ', '_')}"
        run_id  = self.tracker.create_run(name, {
            "topic":    self.cfg.topic,
            "profiles": self.cfg.profiles,
            "api_url":  self.cfg.api_url,
        })

        if not self._api_available():
            print(f"[EvalRunner] WARNING: API not reachable at {self.cfg.api_url}")
            self.tracker.complete_run(run_id, status="failed")
            return {"error": "API not available"}

        results: dict[str, Any] = {"run_id": run_id}

        if self.cfg.run_simulation:
            results["simulation"] = self.run_simulation_eval(run_id)

        if self.cfg.run_llm_quality:
            results["llm_quality"] = self.run_llm_quality_eval(run_id)

        if self.cfg.run_resource_eval:
            results["resource"] = self.run_resource_eval(run_id)

        if self.cfg.run_gap_eval:
            results["gap_detection"] = self.run_gap_eval(run_id)

        self.tracker.complete_run(run_id)

        # Write master report
        out = Path(self.cfg.output_dir) / "eval_report.json"
        out.write_text(json.dumps(results, indent=2, default=str))
        print(f"\n[EvalRunner] All evaluations complete → {out}")

        return results

    # ── 6.  Report generation ──────────────────────────────────────────────────

    def generate_report(self, results: dict) -> str:
        """
        Print a human-readable summary report and return it as a string.
        """
        lines = [
            "",
            "═" * 70,
            "  PathOptLearn — Evaluation Report",
            f"  Run ID : {results.get('run_id', '?')}",
            f"  Topic  : {self.cfg.topic}",
            "═" * 70,
        ]

        # Simulation
        sim = results.get("simulation", {})
        if sim and "error" not in sim:
            cs = sim.get("cohort_stats", {})
            lines += [
                "",
                "  ── Learning Metrics ──",
                f"  Students     : {cs.get('n_students', '?')}",
                f"  Mean pass    : {cs.get('mean_pass_rate', 0):.1%}",
                f"  Mean score   : {cs.get('mean_avg_score', 0):.1f}%",
                f"  Mean retries : {cs.get('mean_retries', 0):.1f}",
            ]
            ttm = sim.get("time_to_mastery", {})
            lines.append(f"  Mastered 1st try: {ttm.get('mastered_1st_try_pct', 0):.1%}")

        # LLM Quality
        qual = results.get("llm_quality", {})
        if qual and "error" not in qual:
            lq = qual.get("lesson_quality", {})
            qq = qual.get("quiz_quality",   {})
            lines += [
                "",
                "  ── Content Quality ──",
                f"  Lessons evaluated    : {lq.get('n_lessons', 0)}",
                f"  Mean overall score   : {lq.get('mean_overall_score', 0):.2f}/5",
                f"  Hallucination rate   : {lq.get('hallucination_rate', 0):.1%}",
                f"  Mean readability     : Grade {lq.get('mean_readability_grade', 0):.1f}",
                f"  Quizzes evaluated    : {qq.get('n_quizzes', 0)}",
                f"  Quiz distractor qual : {qq.get('mean_distractor_quality_score', 0):.2f}/5",
            ]

        # Resources
        res = results.get("resource", {})
        if res and "error" not in res:
            gc = res.get("gap_coverage", {})
            sd = res.get("source_diversity", {})
            lines += [
                "",
                "  ── Resource Recommendations ──",
                f"  Gaps evaluated       : {res.get('n_gaps', 0)}",
                f"  Total resources      : {res.get('n_total_resources', 0)}",
                f"  Mean relevance score : {res.get('mean_relevance_score', 0):.4f}",
                f"  Coverage rate        : {gc.get('coverage_rate', 0):.1%}",
                f"  Source diversity     : {sd.get('diversity_score', 0):.2f}",
            ]

        # Gap detection
        gd = results.get("gap_detection", {})
        if gd and "error" not in gd:
            lines += [
                "",
                "  ── Gap Detection ──",
                f"  Predicted gaps : {gd.get('n_predicted', 0)}",
                f"  Precision      : {gd.get('precision', 0):.3f}",
                f"  Recall         : {gd.get('recall', 0):.3f}",
                f"  F1             : {gd.get('f1', 0):.3f}",
            ]
            cal = gd.get("severity_calibration", {})
            if cal:
                lines.append(
                    f"  Calibration    : {cal.get('calibration_label', '?')} "
                    f"(r={cal.get('point_biserial_r', 0):.3f})"
                )

        lines.append("\n" + "═" * 70 + "\n")
        report = "\n".join(lines)
        print(report)

        # Save text report
        rpt_path = Path(self.cfg.output_dir) / "report.txt"
        rpt_path.write_text(report)
        return report


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="PathOptLearn evaluation runner"
    )
    parser.add_argument("--topic",    default="Machine Learning")
    parser.add_argument("--api",      default="http://localhost:8000")
    parser.add_argument("--profiles", nargs="+",
                        default=["beginner", "intermediate", "advanced"])
    parser.add_argument("--n-runs",   type=int, default=1,
                        help="Runs per profile")
    parser.add_argument("--model",    default="llama3.2:1b")
    parser.add_argument("--output",   default="eval_output")
    parser.add_argument("--name",     default=None,
                        help="Experiment name (default: auto-generated)")
    parser.add_argument("--no-sim",   action="store_true",
                        help="Skip student simulation")
    parser.add_argument("--no-llm",   action="store_true",
                        help="Skip LLM quality evaluation")
    parser.add_argument("--no-res",   action="store_true",
                        help="Skip resource evaluation")
    parser.add_argument("--no-gap",   action="store_true",
                        help="Skip gap detection evaluation")
    args = parser.parse_args()

    cfg = EvalConfig(
        api_url            = args.api,
        topic              = args.topic,
        profiles           = args.profiles,
        n_runs_per_profile = args.n_runs,
        run_simulation     = not args.no_sim,
        run_llm_quality    = not args.no_llm,
        run_resource_eval  = not args.no_res,
        run_gap_eval       = not args.no_gap,
        ollama_model       = args.model,
        output_dir         = args.output,
        experiment_name    = args.name,
    )

    tracker = ExperimentTracker(str(Path(args.output) / "results.db"))
    runner  = EvalRunner(cfg, tracker)
    results = runner.run_all()
    runner.generate_report(results)
    tracker.print_summary()


if __name__ == "__main__":
    main()
