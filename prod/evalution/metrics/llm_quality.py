"""
LLM Output Quality Metrics — PathOptLearn Evaluation
=====================================================
Evaluates the quality of AI-generated lessons and quizzes using:
  - Ollama (llama3.2:1b) as an LLM judge
  - Deterministic readability approximations (no external NLP libs)

Evaluated dimensions
--------------------
  Lessons
    1. Factual accuracy / hallucination risk
    2. Topic relevance (does the lesson address the module objective?)
    3. Coherence and logical structure
    4. Level appropriateness (beginner / intermediate / advanced)
    5. Example quality (concrete, accurate, helpful)
    6. Readability grade (Flesch-Kincaid approximation)

  Quizzes
    7. MCQ format validity
    8. Answer distribution balance
    9. Distractor pedagogical quality (LLM judge)

Usage
-----
    from metrics.llm_quality import evaluate_lesson_quality, evaluate_quiz_quality

    result = evaluate_lesson_quality(
        lesson_content="...",
        module_objective="Understand gradient descent",
        topic="Machine Learning",
        level="intermediate",
    )
    print(result["overall_score"], result["hallucination_risk"])
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

try:
    import ollama as _ollama_lib
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

OLLAMA_HOST  = os.environ.get("OLLAMA_HOST",  "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")


# ── Ollama judge ──────────────────────────────────────────────────────────────

def _judge_call(prompt: str, model: str = OLLAMA_MODEL,
                max_retries: int = 2) -> str:
    """
    Send a structured evaluation prompt to Ollama.
    Returns the raw text response (expected to be JSON).
    Falls back to a neutral placeholder if Ollama is unavailable.
    """
    if not OLLAMA_AVAILABLE:
        return json.dumps({"error": "ollama not installed"})

    for attempt in range(max_retries + 1):
        try:
            client = _ollama_lib.Client(host=OLLAMA_HOST)
            result = client.chat(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert educational content evaluator. "
                            "Always respond with valid JSON only — no markdown, "
                            "no prose, no code fences."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            return result["message"]["content"].strip()
        except Exception as e:
            if attempt == max_retries:
                return json.dumps({"error": str(e)})
            time.sleep(1)
    return "{}"


def _parse_json(text: str) -> dict:
    """Parse JSON from an LLM response, stripping accidental markdown fences."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {"parse_error": True, "raw": text[:300]}


# ══════════════════════════════════════════════════════════════════════════════
# Readability (no external dependencies)
# ══════════════════════════════════════════════════════════════════════════════

def compute_readability(text: str) -> dict:
    """
    Approximate Flesch-Kincaid readability without textstat or NLTK.

    Syllables are counted heuristically (vowel-group method).

    Returns
    -------
    {
        "avg_sentence_length":   float,   # words per sentence
        "avg_word_length":       float,   # characters per word
        "estimated_grade_level": float,   # Flesch-Kincaid Grade Level
        "reading_ease_label":    str,     # "Very Easy" … "Very Difficult"
        "word_count":            int,
        "sentence_count":        int,
    }
    """
    if not text or not text.strip():
        return {
            "avg_sentence_length": 0, "avg_word_length": 0,
            "estimated_grade_level": 0, "reading_ease_label": "N/A",
            "word_count": 0, "sentence_count": 0,
        }

    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip()) > 5]
    n_sentences = max(len(sentences), 1)

    words = re.findall(r"\b[a-zA-Z]+\b", text)
    n_words = max(len(words), 1)

    def _syllables(word: str) -> int:
        word = word.lower()
        count = len(re.findall(r"[aeiouy]+", word))
        if word.endswith("e") and count > 1:
            count -= 1
        return max(count, 1)

    avg_sentence_len = n_words / n_sentences
    avg_word_len     = sum(len(w) for w in words) / n_words
    avg_syllables    = sum(_syllables(w) for w in words) / n_words

    # Flesch-Kincaid Grade Level formula
    fk = max(0.0, round(0.39 * avg_sentence_len + 11.8 * avg_syllables - 15.59, 2))

    label = (
        "Very Easy"      if fk < 6  else
        "Easy"           if fk < 8  else
        "Moderate"       if fk < 10 else
        "Difficult"      if fk < 14 else
        "Very Difficult"
    )

    return {
        "avg_sentence_length":   round(avg_sentence_len, 2),
        "avg_word_length":       round(avg_word_len,     2),
        "estimated_grade_level": fk,
        "reading_ease_label":    label,
        "word_count":            n_words,
        "sentence_count":        n_sentences,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Lesson quality evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_lesson_quality(
    lesson_content: str,
    module_objective: str,
    topic: str,
    level: str = "intermediate",
    model: str = OLLAMA_MODEL,
) -> dict:
    """
    Comprehensive quality evaluation of an AI-generated lesson.

    Uses Ollama as a judge for 5 dimensions (1–5 scale) plus hallucination
    risk flag and missing-concept list.  Readability is computed locally.

    Returns
    -------
    {
        "factual_accuracy":      int,    # 1-5
        "relevance":             int,    # 1-5
        "coherence":             int,    # 1-5
        "level_appropriateness": int,    # 1-5
        "example_quality":       int,    # 1-5
        "overall_score":         float,  # mean of the 5 dimensions
        "hallucination_risk":    bool,
        "hallucination_issues":  list[str],
        "missing_concepts":      list[str],
        "word_count":            int,
        "readability":           dict,
        "topic":                 str,
        "level":                 str,
    }
    """
    excerpt = lesson_content[:2500]  # stay within context window

    prompt = f"""Evaluate this AI-generated educational lesson.

TOPIC:     {topic}
LEVEL:     {level}
OBJECTIVE: {module_objective}

LESSON EXCERPT (first 2500 chars):
---
{excerpt}
---

Score each dimension 1 (very poor) to 5 (excellent):
1. factual_accuracy     — Are all claims factually correct?
2. relevance            — Does the lesson address the stated objective?
3. coherence            — Is the text logically structured and clear?
4. level_appropriateness — Is complexity suited to a {level} learner?
5. example_quality      — Are examples concrete, accurate, and helpful?

Also provide:
- hallucination_risk    : true if you detect probable factual errors
- hallucination_issues  : list specific factual errors (empty list if none)
- missing_concepts      : list concepts the objective mentions that the lesson skips

Respond with ONLY this JSON (no other text):
{{
  "factual_accuracy": <1-5>,
  "relevance": <1-5>,
  "coherence": <1-5>,
  "level_appropriateness": <1-5>,
  "example_quality": <1-5>,
  "hallucination_risk": <true|false>,
  "hallucination_issues": ["..."],
  "missing_concepts": ["..."]
}}"""

    raw    = _judge_call(prompt, model=model)
    result = _parse_json(raw)

    # Compute derived fields
    score_keys = ["factual_accuracy", "relevance", "coherence",
                  "level_appropriateness", "example_quality"]
    scores = [result[k] for k in score_keys
              if isinstance(result.get(k), (int, float))]
    result["overall_score"] = round(sum(scores) / len(scores), 3) if scores else None
    result["word_count"]    = len(lesson_content.split())
    result["readability"]   = compute_readability(lesson_content)
    result["topic"]         = topic
    result["level"]         = level

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Quiz quality evaluation
# ══════════════════════════════════════════════════════════════════════════════

def _evaluate_distractor_quality(questions: list[dict], model: str) -> float:
    """
    LLM judge rates distractor (wrong-option) quality for the first 3 questions.
    Returns a score in [1, 5].
    """
    sample = questions[:3]
    if not sample:
        return 3.0

    q_text = "\n\n".join(
        f"Q{i+1}: {q.get('question', '')}\n"
        + "\n".join(q.get("options", []))
        + f"\nCorrect answer: {q.get('answer', '')}"
        for i, q in enumerate(sample)
    )

    prompt = (
        f"Rate the pedagogical quality of the distractors (wrong options) "
        f"in these MCQ questions:\n\n{q_text}\n\n"
        f"Score 1 (trivially obvious wrong answers) to 5 "
        f"(plausible, educationally valuable distractors).\n"
        f'Respond ONLY with: {{"distractor_quality": <1-5>, "reason": "<one sentence>"}}'
    )

    raw    = _judge_call(prompt, model=model)
    parsed = _parse_json(raw)
    score  = parsed.get("distractor_quality")
    return float(score) if isinstance(score, (int, float)) else 3.0


def evaluate_quiz_quality(
    questions: list[dict],
    model: str = OLLAMA_MODEL,
) -> dict:
    """
    Evaluate the quality of a list of MCQ questions.

    Structural checks are deterministic; distractor quality uses Ollama.

    Expected question schema (from POST /quiz):
    {
        "question":    str,
        "options":     ["A. ...", "B. ...", "C. ...", "D. ..."],
        "answer":      "A" | "B" | "C" | "D",
        "concept":     str,
        "explanation": str,
    }

    Returns
    -------
    {
        "n_questions":              int,
        "format_valid_pct":         float,   # % with correct MCQ structure
        "answer_distribution":      dict,    # {"A": n, "B": n, "C": n, "D": n}
        "distractor_quality_score": float,   # 1-5 (LLM judge)
        "difficulty_balance":       str,     # "balanced" | "answer_biased" | "acceptable"
        "format_issues":            list[str],
    }
    """
    n = len(questions)
    if n == 0:
        return {
            "n_questions": 0,
            "format_valid_pct": 0.0,
            "format_issues": ["No questions provided"],
        }

    issues: list[str] = []
    valid = 0
    answer_dist = {"A": 0, "B": 0, "C": 0, "D": 0}

    for i, q in enumerate(questions):
        q_issues: list[str] = []

        if not str(q.get("question", "")).strip():
            q_issues.append(f"Q{i+1}: missing question text")

        options = q.get("options", [])
        if len(options) < 4:
            q_issues.append(f"Q{i+1}: only {len(options)} options (need 4)")

        ans = str(q.get("answer", "")).strip().upper()
        if ans not in ("A", "B", "C", "D"):
            q_issues.append(f"Q{i+1}: answer '{ans}' is not A/B/C/D")
        else:
            answer_dist[ans] += 1

        if not q_issues:
            valid += 1
        issues.extend(q_issues)

    # Answer balance
    counts    = list(answer_dist.values())
    max_count = max(counts)
    balance   = (
        "balanced"      if max_count - min(counts) <= 1 else
        "answer_biased" if max_count > n * 0.7           else
        "acceptable"
    )

    distractor_score = _evaluate_distractor_quality(questions, model)

    return {
        "n_questions":              n,
        "format_valid_pct":         round(valid / n, 4),
        "answer_distribution":      answer_dist,
        "distractor_quality_score": distractor_score,
        "difficulty_balance":       balance,
        "format_issues":            issues[:10],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Batch evaluation helpers
# ══════════════════════════════════════════════════════════════════════════════

def batch_evaluate_lessons(
    lessons: list[dict],
    model: str = OLLAMA_MODEL,
) -> list[dict]:
    """
    Evaluate multiple lessons sequentially.

    Each item in ``lessons`` must have:
      - content   : str  (lesson text)
      - objective : str  (module objective)
      - topic     : str
      - level     : str  (optional, default "intermediate")

    Returns a list of evaluation dicts (one per lesson), each annotated
    with ``lesson_index``, ``topic``, and ``objective``.
    """
    results: list[dict] = []
    for i, lesson in enumerate(lessons):
        print(f"  [LLM Quality] Evaluating lesson {i + 1}/{len(lessons)}: "
              f"{lesson.get('topic', '?')[:40]}…")
        result = evaluate_lesson_quality(
            lesson_content   = lesson.get("content", ""),
            module_objective = lesson.get("objective", ""),
            topic            = lesson.get("topic", ""),
            level            = lesson.get("level", "intermediate"),
            model            = model,
        )
        result["lesson_index"] = i
        results.append(result)
        time.sleep(0.3)  # avoid saturating Ollama

    return results


def summarize_lesson_quality(eval_results: list[dict]) -> dict:
    """
    Aggregate a list of lesson evaluation results into a summary dict.

    Returns
    -------
    {
        "n_lessons":                int,
        "mean_overall_score":       float,
        "mean_factual_accuracy":    float,
        "mean_relevance":           float,
        "mean_coherence":           float,
        "mean_level_appropriateness": float,
        "mean_example_quality":     float,
        "hallucination_rate":       float,   # fraction of lessons flagged
        "mean_readability_grade":   float,
        "readability_label_dist":   dict,    # {"Easy": 2, "Moderate": 1, ...}
    }
    """
    if not eval_results:
        return {"n_lessons": 0}

    n = len(eval_results)
    dims = [
        "factual_accuracy", "relevance", "coherence",
        "level_appropriateness", "example_quality", "overall_score",
    ]

    summary: dict[str, Any] = {"n_lessons": n}
    for dim in dims:
        vals = [r[dim] for r in eval_results
                if isinstance(r.get(dim), (int, float))]
        summary[f"mean_{dim}"] = round(sum(vals) / len(vals), 3) if vals else None

    summary["hallucination_rate"] = round(
        sum(1 for r in eval_results if r.get("hallucination_risk")) / n, 4
    )

    grades = [
        r.get("readability", {}).get("estimated_grade_level")
        for r in eval_results
        if isinstance(r.get("readability", {}).get("estimated_grade_level"), (int, float))
    ]
    summary["mean_readability_grade"] = round(sum(grades) / len(grades), 2) if grades else None

    label_dist: dict[str, int] = {}
    for r in eval_results:
        lbl = r.get("readability", {}).get("reading_ease_label", "?")
        label_dist[lbl] = label_dist.get(lbl, 0) + 1
    summary["readability_label_dist"] = label_dist

    return summary
