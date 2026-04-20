"""
PathOptLearn — Benchmark Evaluation
====================================
Evaluates PathOptLearn's recommendation and knowledge-tracing quality
against three standard educational datasets:

  1. Riiid!     — 13M interactions, AUC on answered_correctly
  2. EdNet-KT1  — 95M interactions, AUC/Accuracy on correctness
  3. ASSISTments 2009-2010 / 2015 — classroom KT data

What is evaluated
-----------------
For each dataset we measure how well PathOptLearn's /find-gaps pipeline
approximates a knowledge-tracing (KT) signal — i.e., given a prefix of
a student's interaction history, can the system predict whether the
student will answer the next question correctly?

Evaluation protocol (leave-one-out per student)
-----------------------------------------------
  1. For each student take the last interaction as the test label.
  2. Feed the previous K interactions as "answers" to /find-gaps.
  3. Use the gap severity score as a proxy correctness probability:
       p_correct = 1 − severity_score   (high gap → likely wrong)
  4. Compute AUC, Accuracy, RMSE across all students.

Since PathOptLearn does not expose a raw correctness probability the
benchmark also runs a BASELINE column (logistic regression on features
derivable from the dataset without the API) for comparison.

Usage
-----
  python run_benchmark.py \
      --dataset riiid \
      --data    path/to/train.csv \
      --api     http://localhost:8000 \
      --sample  2000 \
      --output  results.json

  python run_benchmark.py --dataset ednet  --data path/to/KT1/ ...
  python run_benchmark.py --dataset assist --data path/to/skill_builder_data.csv ...

Dependencies: pandas numpy scikit-learn requests tqdm
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

HISTORY_WINDOW = 8          # interactions used as "quiz answers"
MIN_INTERACTIONS = 10       # minimum interactions per student
SEVERITY_MAP = {            # gap severity → probability of being wrong
    "high":   0.85,
    "medium": 0.55,
    "low":    0.25,
    None:     0.10,
}
API_TIMEOUT = 60            # seconds per /find-gaps call


# ══════════════════════════════════════════════════════════════════════════════
# Dataset loaders
# ══════════════════════════════════════════════════════════════════════════════

class RiiidLoader:
    """
    Riiid! Answer Correctness Prediction (Kaggle 2020)

    Required file: train.csv
    Key columns:
      user_id, content_type_id, question_id, answered_correctly,
      prior_question_elapsed_time, timestamp
    """

    DATASET_NAME = "Riiid!"

    def load(self, data_path: str, sample: int) -> pd.DataFrame:
        path = Path(data_path)
        if path.is_dir():
            path = path / "train.csv"
        print(f"[Riiid] Loading {path} …")

        df = pd.read_csv(
            path,
            dtype={
                "user_id": "int32",
                "content_type_id": "int8",
                "answered_correctly": "int8",
            },
            nrows=sample * 50,  # over-sample then filter
        )

        # Keep only question rows (content_type_id == 0)
        df = df[df["content_type_id"] == 0].copy()
        df = df[df["answered_correctly"].isin([0, 1])]
        df.sort_values(["user_id", "timestamp"], inplace=True)

        # Sample students
        users = df["user_id"].unique()
        if len(users) > sample:
            users = np.random.choice(users, sample, replace=False)
        df = df[df["user_id"].isin(users)]

        return df.rename(columns={
            "user_id":            "student_id",
            "question_id":        "item_id",
            "answered_correctly": "correct",
            "prior_question_elapsed_time": "elapsed_ms",
        })

    @staticmethod
    def student_features(group: pd.DataFrame) -> dict:
        """Derive simple features for the baseline model."""
        return {
            "avg_correct":     group["correct"].mean(),
            "n_interactions":  len(group),
            "avg_elapsed_ms":  group.get("elapsed_ms", pd.Series([0])).mean(),
        }


class EdNetLoader:
    """
    EdNet-KT1

    Data layout: one CSV per student named u{student_id}.csv
    OR a merged file with columns: user_id, timestamp, question_id,
       user_answer, elapsed_time, correct (0/1)

    Merged file can be produced with:
      python merge_ednet.py path/to/KT1/ merged_kt1.csv
    """

    DATASET_NAME = "EdNet-KT1"

    def load(self, data_path: str, sample: int) -> pd.DataFrame:
        path = Path(data_path)

        # Accept a pre-merged CSV
        if path.is_file():
            print(f"[EdNet] Loading merged file {path} …")
            df = pd.read_csv(path, nrows=sample * 50)
        else:
            # Per-student directory: read first `sample` files
            files = sorted(path.glob("u*.csv"))[:sample * 5]
            print(f"[EdNet] Reading {len(files)} student files …")
            chunks = []
            for f in tqdm(files, desc="EdNet files"):
                try:
                    chunk = pd.read_csv(f, header=None,
                                        names=["timestamp", "solving_id",
                                               "question_id", "user_answer",
                                               "elapsed_time"])
                    chunk["student_id"] = int(f.stem[1:])  # u12345 → 12345
                    chunks.append(chunk)
                except Exception:
                    pass
            df = pd.concat(chunks, ignore_index=True)

        # Normalise column names
        df = df.rename(columns={
            "user_id":    "student_id",
            "item_id":    "item_id",
            "elapsed_time": "elapsed_ms",
        })

        # Compute correctness if not present
        if "correct" not in df.columns:
            # EdNet uses 'O'/'X' or stores separate correct_answer
            if "user_answer" in df.columns and "correct_answer" in df.columns:
                df["correct"] = (df["user_answer"] == df["correct_answer"]).astype(int)
            else:
                raise ValueError(
                    "EdNet data must contain 'correct' or "
                    "('user_answer', 'correct_answer') columns."
                )

        if "question_id" in df.columns and "item_id" not in df.columns:
            df["item_id"] = df["question_id"]

        df = df[df["correct"].isin([0, 1])].copy()
        df.sort_values(["student_id", "timestamp"], inplace=True)

        users = df["student_id"].unique()
        if len(users) > sample:
            users = np.random.choice(users, sample, replace=False)
        return df[df["student_id"].isin(users)]

    @staticmethod
    def student_features(group: pd.DataFrame) -> dict:
        return {
            "avg_correct":    group["correct"].mean(),
            "n_interactions": len(group),
            "avg_elapsed_ms": group.get("elapsed_ms", pd.Series([0])).mean(),
        }


class ASSISTmentsLoader:
    """
    ASSISTments 2009-2010 or 2015

    Key columns (2009-2010):
      order_id, user_id, problem_id, correct, attempt_count,
      ms_first_response, hint_count, skill_id, skill_name, opportunity

    Key columns (2015):
      log_id, user_id, problem_id, correct, sequence_id
    """

    DATASET_NAME = "ASSISTments"

    def load(self, data_path: str, sample: int) -> pd.DataFrame:
        path = Path(data_path)
        if path.is_dir():
            candidates = list(path.glob("*.csv"))
            if not candidates:
                raise FileNotFoundError(f"No CSV in {path}")
            path = candidates[0]

        print(f"[ASSISTments] Loading {path} …")
        df = pd.read_csv(path, low_memory=False)

        # Normalise names
        rename = {}
        for col in df.columns:
            low = col.lower()
            if "user" in low and "id" in low:
                rename[col] = "student_id"
            elif low in ("problem_id", "item_id", "question_id"):
                rename[col] = "item_id"
            elif low == "correct":
                rename[col] = "correct"
            elif "first_response" in low or "elapsed" in low:
                rename[col] = "elapsed_ms"
            elif "opportunity" == low:
                rename[col] = "opportunity"
            elif "skill_name" in low:
                rename[col] = "skill_name"
            elif "hint_count" in low:
                rename[col] = "hint_count"
        df = df.rename(columns=rename)

        if "item_id" not in df.columns:
            # fallback
            for c in df.columns:
                if "problem" in c.lower():
                    df["item_id"] = df[c]
                    break

        df = df[df["correct"].isin([0, 1])].copy()
        df.sort_values(["student_id"], inplace=True)

        users = df["student_id"].unique()
        if len(users) > sample:
            users = np.random.choice(users, sample, replace=False)
        return df[df["student_id"].isin(users)]

    @staticmethod
    def student_features(group: pd.DataFrame) -> dict:
        return {
            "avg_correct":    group["correct"].mean(),
            "n_interactions": len(group),
            "avg_elapsed_ms": group.get("elapsed_ms", pd.Series([0])).mean(),
            "avg_hints":      group.get("hint_count", pd.Series([0])).mean(),
            "avg_opportunity": group.get("opportunity", pd.Series([0])).mean(),
        }


LOADERS = {
    "riiid":  RiiidLoader,
    "ednet":  EdNetLoader,
    "assist": ASSISTmentsLoader,
}


# ══════════════════════════════════════════════════════════════════════════════
# PathOptLearn API evaluator
# ══════════════════════════════════════════════════════════════════════════════

def _build_synthetic_questions(item_ids: list[int], corrects: list[int],
                                topic: str) -> tuple[list, list]:
    """
    Convert a student's interaction history into the format expected by
    POST /find-gaps:
      questions = [{"question": "...", "answer": "A", "concept": "..."}]
      answers   = ["A"/"B"/"C"/"D"]  (A = correct attempt in MCQ proxy)
    """
    questions = []
    answers   = []
    option_map = {0: "B", 1: "A"}  # A = correct in our synthetic MCQ

    for iid, c in zip(item_ids, corrects):
        questions.append({
            "question":    f"Question about item {iid} in {topic}",
            "options":     ["A. correct answer", "B. wrong answer",
                            "C. distractor 1",  "D. distractor 2"],
            "answer":      "A",      # ground truth is always A
            "concept":     str(iid),
            "explanation": "",
        })
        answers.append(option_map[c])  # A if student was correct, B if wrong

    return questions, answers


def call_find_gaps(api_url: str, topic: str,
                   questions: list, answers: list) -> float:
    """
    Call POST /find-gaps and return a probability-of-being-correct proxy.

    The proxy is:
      p_correct = 1 − weighted_severity_score
    where severity_score aggregates the severity of all returned gaps.
    A student with no gaps → p_correct ≈ 1.0
    A student with all high gaps → p_correct ≈ 0.15
    """
    try:
        r = requests.post(
            f"{api_url}/find-gaps",
            json={
                "topic":     topic,
                "questions": questions,
                "answers":   answers,
            },
            timeout=API_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()

        gaps = data.get("gaps", [])
        if not gaps:
            # No gaps → high predicted correctness
            return 0.90

        # Average severity → correctness proxy
        severities = [SEVERITY_MAP.get(g.get("severity"), 0.55) for g in gaps]
        avg_severity = np.mean(severities)
        return float(1.0 - avg_severity)

    except Exception as e:
        print(f"  [API error] {e}")
        return 0.5  # neutral fallback


# ══════════════════════════════════════════════════════════════════════════════
# Baseline model (no API)
# ══════════════════════════════════════════════════════════════════════════════

def baseline_predict(train_features: list[dict],
                     train_labels:   list[int],
                     test_features:  list[dict]) -> np.ndarray:
    """
    Logistic regression baseline using only student-level aggregated features.
    Trained on the history prefix, evaluated on the held-out last answer.
    """
    feat_keys = sorted({k for f in train_features for k in f})
    X_train = np.array([[f.get(k, 0.0) for k in feat_keys] for f in train_features])
    X_test  = np.array([[f.get(k, 0.0) for k in feat_keys] for f in test_features])
    y_train = np.array(train_labels)

    # Handle degenerate case (all same label)
    if len(set(y_train)) < 2:
        return np.full(len(X_test), float(y_train.mean()))

    clf = LogisticRegression(max_iter=200, C=1.0)
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_test)[:, 1]


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                    name: str) -> dict:
    """Compute AUC, Accuracy, RMSE for a set of predictions."""
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {"model": name}

    try:
        metrics["auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
    except ValueError:
        metrics["auc"] = None

    metrics["accuracy"] = round(float(accuracy_score(y_true, y_pred)), 4)
    metrics["rmse"]     = round(float(np.sqrt(np.mean((y_true - y_prob) ** 2))), 4)
    metrics["n_samples"] = int(len(y_true))
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# Main evaluation loop
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(dataset_name: str, df: pd.DataFrame,
             loader,  # loader instance for feature extraction
             api_url: str, topic: str,
             use_api: bool = True) -> dict:

    students = df["student_id"].unique()
    print(f"\n[Eval] {dataset_name} — {len(students)} students, "
          f"{len(df)} interactions")

    y_true, y_api, train_feats, test_feats = [], [], [], []

    for sid in tqdm(students, desc=f"Evaluating {dataset_name}"):
        group = df[df["student_id"] == sid].reset_index(drop=True)
        if len(group) < MIN_INTERACTIONS:
            continue

        # Split: last interaction = test, previous window = history
        history = group.iloc[-HISTORY_WINDOW - 1 : -1]
        last    = group.iloc[-1]
        label   = int(last["correct"])

        # ── API prediction ─────────────────────────────────────────────────
        api_prob = 0.5
        if use_api:
            item_ids = history["item_id"].tolist()
            corrects = history["correct"].tolist()
            questions, answers = _build_synthetic_questions(
                item_ids, corrects, topic
            )
            api_prob = call_find_gaps(api_url, topic, questions, answers)
            time.sleep(0.05)  # rate limit

        # ── Baseline features ───────────────────────────────────────────────
        feats = loader.student_features(history)

        y_true.append(label)
        y_api.append(api_prob)
        train_feats.append(feats)
        test_feats.append(feats)   # baseline uses same features for test

    y_true_arr = np.array(y_true)
    y_api_arr  = np.array(y_api)

    # Split baseline: 80% train, 20% test by index
    split = int(len(y_true_arr) * 0.8)
    bl_probs = baseline_predict(
        train_feats[:split], list(y_true_arr[:split]),
        test_feats[split:],
    )

    results = {
        "dataset": dataset_name,
        "n_students": len(y_true_arr),
        "models": [],
    }

    # PathOptLearn (API)
    if use_api:
        results["models"].append(
            compute_metrics(y_true_arr, y_api_arr, "PathOptLearn (/find-gaps)")
        )

    # Baseline (logistic regression)
    if len(y_true_arr[split:]) > 1:
        results["models"].append(
            compute_metrics(y_true_arr[split:], bl_probs, "Baseline (LogReg)")
        )

    # Random baseline
    rand_probs = np.random.uniform(0.4, 0.6, len(y_true_arr))
    results["models"].append(
        compute_metrics(y_true_arr, rand_probs, "Random baseline")
    )

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Report printer
# ══════════════════════════════════════════════════════════════════════════════

def print_report(all_results: list[dict]):
    print("\n" + "═" * 72)
    print("  PathOptLearn — Benchmark Evaluation Report")
    print("═" * 72)

    for res in all_results:
        print(f"\n  Dataset : {res['dataset']}")
        print(f"  Students: {res['n_students']}")
        print()
        header = f"  {'Model':<40} {'AUC':>7} {'Acc':>7} {'RMSE':>7} {'N':>7}"
        print(header)
        print("  " + "-" * 66)
        for m in res["models"]:
            auc  = f"{m['auc']:.4f}"  if m.get("auc")      else "  N/A "
            acc  = f"{m['accuracy']:.4f}"
            rmse = f"{m['rmse']:.4f}"
            print(f"  {m['model']:<40} {auc:>7} {acc:>7} {rmse:>7} "
                  f"{m['n_samples']:>7}")

    print("\n" + "═" * 72 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="PathOptLearn benchmark evaluation"
    )
    parser.add_argument("--dataset", required=True,
                        choices=["riiid", "ednet", "assist", "all"],
                        help="Dataset to evaluate against")
    parser.add_argument("--data", required=True,
                        help="Path to dataset file or directory")
    parser.add_argument("--api", default="http://localhost:8000",
                        help="PathOptLearn API base URL")
    parser.add_argument("--topic", default="Mathematics",
                        help="Topic label sent to /find-gaps (should match "
                             "domain of dataset, e.g. 'TOEIC English' for EdNet)")
    parser.add_argument("--sample", type=int, default=500,
                        help="Number of students to sample")
    parser.add_argument("--no-api", action="store_true",
                        help="Skip API calls (baseline only)")
    parser.add_argument("--output", default=None,
                        help="Save JSON results to this path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    # Resolve datasets to evaluate
    datasets = (
        list(LOADERS.keys()) if args.dataset == "all"
        else [args.dataset]
    )

    all_results = []

    for ds_name in datasets:
        loader_cls = LOADERS[ds_name]
        loader     = loader_cls()

        try:
            df = loader.load(args.data, args.sample)
        except FileNotFoundError as e:
            print(f"[SKIP] {ds_name}: {e}")
            continue

        res = evaluate(
            dataset_name=loader_cls.DATASET_NAME,
            df=df,
            loader=loader,
            api_url=args.api,
            topic=args.topic,
            use_api=not args.no_api,
        )
        all_results.append(res)

    print_report(all_results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
