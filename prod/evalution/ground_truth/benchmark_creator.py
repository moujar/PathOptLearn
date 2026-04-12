"""
Benchmark Dataset Creator — PathOptLearn Ground-Truth Generator
===============================================================
Generates a curated benchmark dataset with:

  1. 15 topics across 3 domains (CS, Mathematics, Applied)
  2. 5 ground-truth MCQ questions per topic + level
  3. Expected knowledge gaps per learner level
  4. Validation utilities

The benchmark can be used to evaluate PathOptLearn's:
  - Diagnostic quiz quality (do generated questions match ground truth?)
  - Gap detection recall (does /find-gaps find the expected gaps?)
  - Roadmap relevance (do generated modules cover the expected concepts?)

Ground-truth questions are generated via Ollama and manually annotatable.
The output JSON can be edited by domain experts before use in evaluation.

Usage
-----
  python benchmark_creator.py --output benchmark.json
  python benchmark_creator.py --output benchmark.json --validate
  python benchmark_creator.py --topic "Machine Learning" --level intermediate
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

try:
    import ollama as _ollama_lib
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

OLLAMA_HOST  = os.environ.get("OLLAMA_HOST",  "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")


# ══════════════════════════════════════════════════════════════════════════════
# Curated benchmark topic catalogue
# ══════════════════════════════════════════════════════════════════════════════

BENCHMARK_TOPICS: list[dict] = [
    # ── Computer Science ──────────────────────────────────────────────────────
    {"topic": "Machine Learning",           "domain": "CS",
     "levels": ["beginner", "intermediate", "advanced"],
     "key_concepts": ["supervised learning", "gradient descent", "overfitting",
                      "neural networks", "regularisation"]},

    {"topic": "Data Structures",            "domain": "CS",
     "levels": ["beginner", "intermediate"],
     "key_concepts": ["arrays", "linked lists", "trees", "heaps",
                      "hash tables", "graph traversal"]},

    {"topic": "Algorithms",                 "domain": "CS",
     "levels": ["beginner", "intermediate", "advanced"],
     "key_concepts": ["sorting", "binary search", "dynamic programming",
                      "greedy algorithms", "complexity analysis"]},

    {"topic": "Computer Networks",          "domain": "CS",
     "levels": ["beginner", "intermediate"],
     "key_concepts": ["TCP/IP", "HTTP", "DNS", "routing", "OSI model"]},

    {"topic": "Cybersecurity",              "domain": "CS",
     "levels": ["beginner", "intermediate", "advanced"],
     "key_concepts": ["encryption", "authentication", "OWASP Top 10",
                      "PKI", "threat modelling"]},

    # ── Mathematics ───────────────────────────────────────────────────────────
    {"topic": "Linear Algebra",             "domain": "Mathematics",
     "levels": ["beginner", "intermediate", "advanced"],
     "key_concepts": ["vectors", "matrices", "eigenvalues",
                      "matrix decomposition", "dot product"]},

    {"topic": "Calculus",                   "domain": "Mathematics",
     "levels": ["beginner", "intermediate", "advanced"],
     "key_concepts": ["limits", "derivatives", "integration",
                      "chain rule", "fundamental theorem"]},

    {"topic": "Statistics and Probability", "domain": "Mathematics",
     "levels": ["beginner", "intermediate"],
     "key_concepts": ["probability distributions", "hypothesis testing",
                      "Bayes theorem", "confidence intervals", "p-value"]},

    {"topic": "Discrete Mathematics",       "domain": "Mathematics",
     "levels": ["beginner", "intermediate"],
     "key_concepts": ["sets", "logic", "combinatorics",
                      "graph theory", "proofs by induction"]},

    # ── Applied / Engineering ─────────────────────────────────────────────────
    {"topic": "Database Systems",           "domain": "Applied",
     "levels": ["beginner", "intermediate", "advanced"],
     "key_concepts": ["SQL", "normalisation", "indexing",
                      "transactions", "ACID properties"]},

    {"topic": "Software Engineering",       "domain": "Applied",
     "levels": ["beginner", "intermediate"],
     "key_concepts": ["design patterns", "SOLID principles", "testing",
                      "CI/CD", "refactoring"]},

    {"topic": "System Design",              "domain": "Applied",
     "levels": ["intermediate", "advanced"],
     "key_concepts": ["scalability", "load balancing", "caching",
                      "microservices", "CAP theorem"]},

    {"topic": "Operating Systems",          "domain": "CS",
     "levels": ["beginner", "intermediate"],
     "key_concepts": ["processes", "threads", "memory management",
                      "scheduling", "file systems"]},

    {"topic": "Deep Learning",              "domain": "CS",
     "levels": ["intermediate", "advanced"],
     "key_concepts": ["backpropagation", "CNNs", "RNNs",
                      "attention mechanism", "transfer learning"]},

    {"topic": "Natural Language Processing","domain": "CS",
     "levels": ["intermediate", "advanced"],
     "key_concepts": ["tokenisation", "word embeddings", "transformers",
                      "named entity recognition", "language models"]},
]


# ══════════════════════════════════════════════════════════════════════════════
# Fallback ground-truth questions (static, no LLM needed)
# ══════════════════════════════════════════════════════════════════════════════

_STATIC_QUESTIONS: dict[str, list[dict]] = {
    "Machine Learning": [
        {"question":    "Which of the following best describes supervised learning?",
         "options":     ["A. Learning from labelled examples",
                         "B. Learning without any labels",
                         "C. Reinforcement from rewards only",
                         "D. Clustering similar data points"],
         "answer":      "A",
         "concept":     "supervised learning",
         "difficulty":  "beginner"},
        {"question":    "What is the purpose of a validation set?",
         "options":     ["A. To train the model parameters",
                         "B. To tune hyperparameters and detect overfitting",
                         "C. To evaluate final model performance",
                         "D. To augment the training data"],
         "answer":      "B",
         "concept":     "overfitting",
         "difficulty":  "intermediate"},
        {"question":    "Which optimiser uses adaptive learning rates per parameter?",
         "options":     ["A. SGD", "B. Momentum", "C. Adam", "D. Newton-Raphson"],
         "answer":      "C",
         "concept":     "gradient descent",
         "difficulty":  "intermediate"},
        {"question":    "Regularisation reduces overfitting by:",
         "options":     ["A. Adding more training data",
                         "B. Penalising large model weights",
                         "C. Increasing the learning rate",
                         "D. Removing validation examples"],
         "answer":      "B",
         "concept":     "regularisation",
         "difficulty":  "intermediate"},
        {"question":    "In a neural network, the activation function's main role is to:",
         "options":     ["A. Initialise weights to zero",
                         "B. Compute the loss function",
                         "C. Introduce non-linearity into the model",
                         "D. Normalise the input features"],
         "answer":      "C",
         "concept":     "neural networks",
         "difficulty":  "beginner"},
    ],
}

_DEFAULT_GAPS: dict[str, dict[str, list[str]]] = {
    "Machine Learning": {
        "beginner":     ["supervised learning", "training vs. testing data", "model evaluation"],
        "intermediate": ["gradient descent", "overfitting", "regularisation", "cross-validation"],
        "advanced":     ["backpropagation", "neural network architecture", "hyperparameter tuning"],
    },
    "Data Structures": {
        "beginner":     ["arrays", "linked lists", "stack", "queue"],
        "intermediate": ["binary trees", "heaps", "hash tables"],
    },
    "Linear Algebra": {
        "beginner":     ["vectors", "matrices", "dot product"],
        "intermediate": ["matrix multiplication", "determinants", "eigenvalues"],
        "advanced":     ["SVD", "PCA", "matrix decompositions"],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# LLM-based question generation
# ══════════════════════════════════════════════════════════════════════════════

def _llm_generate_questions(topic: str, level: str,
                             concepts: list[str],
                             n: int = 5) -> list[dict]:
    """
    Use Ollama to generate ``n`` ground-truth MCQ questions for a topic.
    Falls back to static questions if Ollama is unavailable or fails.
    """
    if not OLLAMA_AVAILABLE:
        return _static_questions(topic, level, n)

    prompt = f"""Generate {n} multiple-choice questions to test a {level} student's
knowledge of "{topic}". Focus on these concepts: {', '.join(concepts[:4])}.

Each question must have exactly 4 options (A, B, C, D) with exactly one correct answer.
Vary difficulty: include 1 easy, {n-2} medium, and 1 hard question.

Return ONLY a JSON array in this exact format — no other text:
[
  {{
    "question":   "<question text>",
    "options":    ["A. <option>", "B. <option>", "C. <option>", "D. <option>"],
    "answer":     "A",
    "concept":    "<concept tested>",
    "difficulty": "beginner|intermediate|advanced",
    "explanation": "<why the answer is correct>"
  }}
]"""

    try:
        client   = _ollama_lib.Client(host=OLLAMA_HOST)
        response = client.chat(
            model    = OLLAMA_MODEL,
            messages = [
                {"role": "system",
                 "content": "You are an expert educational content creator. "
                             "Respond only with valid JSON arrays."},
                {"role": "user", "content": prompt},
            ],
        )
        raw  = response["message"]["content"].strip()
        raw  = re.sub(r"```(?:json)?", "", raw).strip()
        data = json.loads(raw)
        if isinstance(data, list) and len(data) > 0:
            return data[:n]
    except Exception as e:
        print(f"  [BenchmarkCreator] LLM failed ({e}), using static fallback")

    return _static_questions(topic, level, n)


def _static_questions(topic: str, level: str, n: int = 5) -> list[dict]:
    """Return static fallback questions for well-known topics."""
    q = _STATIC_QUESTIONS.get(topic, [])
    if q:
        return (q * ((n // len(q)) + 1))[:n]

    # Generic placeholder questions for unknown topics
    return [
        {
            "question":    f"What is the primary goal of {topic}?",
            "options":     [
                "A. To process data efficiently",
                "B. To solve domain-specific problems",
                "C. To optimise resource usage",
                "D. To automate repetitive tasks",
            ],
            "answer":      "B",
            "concept":     f"{topic} fundamentals",
            "difficulty":  level,
            "explanation": f"The primary goal of {topic} is to solve problems within its domain.",
        }
    ] * n


# ══════════════════════════════════════════════════════════════════════════════
# Ground-truth gap annotation
# ══════════════════════════════════════════════════════════════════════════════

def get_ground_truth_gaps(topic: str, level: str,
                          concepts: list[str]) -> list[str]:
    """
    Return expected knowledge gaps for a student at ``level`` on ``topic``.

    Uses static annotations where available, otherwise derives gaps from
    the first half of ``concepts`` (typical for beginner/intermediate).
    """
    static = _DEFAULT_GAPS.get(topic, {}).get(level)
    if static:
        return static

    # Heuristic: beginners typically struggle with foundational concepts
    n_gaps = max(2, len(concepts) // 2)
    return concepts[:n_gaps]


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark generation
# ══════════════════════════════════════════════════════════════════════════════

def create_benchmark(
    topics: list[dict] | None = None,
    output_path: str | None   = None,
    use_llm: bool             = True,
    sleep_between: float      = 0.5,
) -> dict:
    """
    Generate the complete benchmark dataset.

    Parameters
    ----------
    topics       : list of topic dicts (default: BENCHMARK_TOPICS)
    output_path  : if given, write JSON to this path
    use_llm      : if False, use only static fallback questions
    sleep_between: seconds to wait between LLM calls

    Returns
    -------
    A benchmark dict with structure:
    {
        "version": "1.0",
        "n_topics": int,
        "topics": [
            {
                "topic":     str,
                "domain":    str,
                "levels": {
                    "<level>": {
                        "questions":      list[dict],
                        "expected_gaps":  list[str],
                    }
                }
            }
        ]
    }
    """
    if topics is None:
        topics = BENCHMARK_TOPICS

    benchmark: dict[str, Any] = {
        "version":  "1.0",
        "n_topics": len(topics),
        "topics":   [],
    }

    for i, t in enumerate(topics):
        topic_name = t["topic"]
        print(f"[BenchmarkCreator] {i+1}/{len(topics)}: {topic_name}")

        topic_entry: dict[str, Any] = {
            "topic":   topic_name,
            "domain":  t["domain"],
            "levels":  {},
        }

        for level in t.get("levels", ["beginner", "intermediate"]):
            concepts = t.get("key_concepts", [])
            if use_llm and OLLAMA_AVAILABLE:
                questions = _llm_generate_questions(topic_name, level, concepts, n=5)
                time.sleep(sleep_between)
            else:
                questions = _static_questions(topic_name, level, n=5)

            expected_gaps = get_ground_truth_gaps(topic_name, level, concepts)

            topic_entry["levels"][level] = {
                "questions":     questions,
                "expected_gaps": expected_gaps,
                "key_concepts":  concepts,
            }

        benchmark["topics"].append(topic_entry)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(benchmark, f, indent=2)
        print(f"[BenchmarkCreator] Benchmark saved → {output_path}")

    return benchmark


# ══════════════════════════════════════════════════════════════════════════════
# Validation
# ══════════════════════════════════════════════════════════════════════════════

def validate_benchmark(benchmark: dict) -> dict:
    """
    Validate the format and completeness of a benchmark dataset.

    Returns a validation report dict.
    """
    issues: list[str] = []
    stats:  dict[str, int] = {
        "n_topics": 0, "n_levels": 0, "n_questions": 0,
        "n_invalid_questions": 0,
    }

    for t in benchmark.get("topics", []):
        stats["n_topics"] += 1
        for level, data in t.get("levels", {}).items():
            stats["n_levels"] += 1
            for j, q in enumerate(data.get("questions", [])):
                stats["n_questions"] += 1
                ans = str(q.get("answer", "")).upper()
                if ans not in ("A", "B", "C", "D"):
                    stats["n_invalid_questions"] += 1
                    issues.append(
                        f"{t['topic']} [{level}] Q{j+1}: invalid answer '{ans}'"
                    )
                if len(q.get("options", [])) < 4:
                    issues.append(
                        f"{t['topic']} [{level}] Q{j+1}: fewer than 4 options"
                    )
            if not data.get("expected_gaps"):
                issues.append(f"{t['topic']} [{level}]: no expected gaps defined")

    return {
        "valid":   len(issues) == 0,
        "stats":   stats,
        "n_issues": len(issues),
        "issues":  issues[:20],
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="PathOptLearn benchmark creator")
    parser.add_argument("--output",   default="benchmark.json",
                        help="Output JSON path")
    parser.add_argument("--topic",    default=None,
                        help="Generate benchmark for a single topic only")
    parser.add_argument("--level",    default=None,
                        help="Filter to a specific level")
    parser.add_argument("--validate", action="store_true",
                        help="Validate an existing benchmark file instead")
    parser.add_argument("--no-llm",   action="store_true",
                        help="Use static questions only (no Ollama)")
    args = parser.parse_args()

    if args.validate:
        if not Path(args.output).exists():
            print(f"File not found: {args.output}")
            return
        with open(args.output) as f:
            bm = json.load(f)
        report = validate_benchmark(bm)
        print(json.dumps(report, indent=2))
        return

    topics = BENCHMARK_TOPICS
    if args.topic:
        topics = [t for t in BENCHMARK_TOPICS
                  if t["topic"].lower() == args.topic.lower()]
        if not topics:
            # Add as a custom topic
            topics = [{"topic": args.topic, "domain": "Custom",
                       "levels": [args.level or "intermediate"],
                       "key_concepts": []}]

    if args.level and not args.topic:
        for t in topics:
            t = dict(t)
            t["levels"] = [args.level] if args.level in t.get("levels", []) else t["levels"]

    benchmark = create_benchmark(
        topics      = topics,
        output_path = args.output,
        use_llm     = not args.no_llm,
    )

    report = validate_benchmark(benchmark)
    print(f"\nValidation: {'PASS' if report['valid'] else 'ISSUES FOUND'}")
    print(f"  Topics: {report['stats']['n_topics']}  "
          f"Levels: {report['stats']['n_levels']}  "
          f"Questions: {report['stats']['n_questions']}  "
          f"Invalid: {report['stats']['n_invalid_questions']}")
    if report["issues"]:
        print("  Issues:")
        for issue in report["issues"][:10]:
            print(f"    - {issue}")


if __name__ == "__main__":
    main()
