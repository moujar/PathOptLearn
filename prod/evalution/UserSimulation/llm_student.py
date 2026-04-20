"""
PathOptLearn — LLM Student Simulator
======================================
Simulates realistic student behaviour through the full PathOptLearn
learning flow using a local Ollama LLM as the "student brain".

What it simulates
-----------------
The LLM plays the role of a student with a configurable profile
(level, learning style, attention span, error rate). It:

  1. Picks a topic to learn
  2. Receives the diagnostic quiz  → answers each question in character
  3. Receives a module lesson      → reads and forms memory
  4. Receives a module quiz        → answers based on memory + profile
  5. Repeats for each module in the roadmap
  6. Logs every interaction to simulation_log.jsonl

The simulator calls the real PathOptLearn API so the full pipeline
(DeepSearch → Roadmap → Lesson → Quiz → Gaps → Recommender) runs
exactly as it would for a real user.

Metrics collected per simulation run
--------------------------------------
  - per-module quiz score
  - pass rate (≥70%)
  - number of retries per module
  - gaps identified vs. gaps closed (resolved_at populated)
  - total time simulated (wall-clock)

Usage
-----
  # Single student, default profile
  python llm_student.py --topic "Machine Learning" --api http://localhost:8000

  # Batch: 5 students with different profiles
  python llm_student.py --topic "Machine Learning" --api http://localhost:8000 \
      --profiles beginner intermediate advanced struggling fast_learner \
      --runs 5 --output sim_results.jsonl

  # Replay a dataset row (use real student history as initial context)
  python llm_student.py --topic "Mathematics" \
      --history  "Q1: What is a derivative? Q2: Solve integral..." \
      --init-level intermediate

Dependencies: requests ollama (pip install ollama)
"""

import argparse
import json
import os
import re
import random
import time
from datetime import datetime, timezone

import requests

try:
    import ollama as _ollama_lib
    _OLLAMA_CLIENT = _ollama_lib.Client  # keep a direct reference; avoids "possibly unbound"
    OLLAMA_AVAILABLE = True
except ImportError:
    _OLLAMA_CLIENT = None
    OLLAMA_AVAILABLE = False

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_API   = "http://localhost:8000"
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")
OLLAMA_HOST   = os.environ.get("OLLAMA_HOST",  "http://localhost:11434")
API_TIMEOUT   = 120


# ══════════════════════════════════════════════════════════════════════════════
# Student profiles
# ══════════════════════════════════════════════════════════════════════════════

PROFILES = {
    "beginner": {
        "description":   "A complete beginner with no prior knowledge of the topic.",
        "error_rate":    0.65,   # probability of answering wrong on hard questions
        "attention":     "low",  # affects reading thoroughness
        "style":         "passive",
        "patience":      2,      # max retries before giving up on a module
    },
    "intermediate": {
        "description":   "A student with some exposure but significant gaps.",
        "error_rate":    0.35,
        "attention":     "medium",
        "style":         "active",
        "patience":      3,
    },
    "advanced": {
        "description":   "An advanced student who mostly answers correctly.",
        "error_rate":    0.10,
        "attention":     "high",
        "style":         "critical",
        "patience":      2,
    },
    "struggling": {
        "description":   "A student who struggles despite effort; makes systematic errors.",
        "error_rate":    0.75,
        "attention":     "high",  # tries hard but still fails
        "style":         "passive",
        "patience":      4,
    },
    "fast_learner": {
        "description":   "A quick learner who retains content well after one read.",
        "error_rate":    0.15,
        "attention":     "high",
        "style":         "critical",
        "patience":      1,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# LLM wrapper — routes to Ollama (local) or falls back to heuristic
# ══════════════════════════════════════════════════════════════════════════════

def _llm_call(prompt: str, system: str = "", model: str = DEFAULT_MODEL) -> str:
    """Call local Ollama. Falls back to heuristic if unavailable."""
    if not OLLAMA_AVAILABLE:
        return _heuristic_answer(prompt)

    try:
        assert _OLLAMA_CLIENT is not None
        client   = _OLLAMA_CLIENT(host=OLLAMA_HOST)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        result = client.chat(model=model, messages=messages)
        return result["message"]["content"].strip()
    except Exception as e:
        print(f"  [LLM] Ollama unavailable ({e}), using heuristic fallback")
        return _heuristic_answer(prompt)


def _heuristic_answer(prompt: str) -> str:
    """Deterministic fallback when Ollama is not running."""
    # If prompt contains MCQ options A/B/C/D → pick one
    if "A." in prompt and "B." in prompt:
        return random.choice(["A", "B", "C", "D"])
    return "I think the answer relates to the fundamental concepts discussed."


# ══════════════════════════════════════════════════════════════════════════════
# API helpers
# ══════════════════════════════════════════════════════════════════════════════

def _api_get(api: str, path: str, **params):
    r = requests.get(f"{api}{path}", params=params, timeout=API_TIMEOUT)
    r.raise_for_status()
    return r.json()


def _api_post(api: str, path: str, body: dict | None = None, params: dict | None = None):
    r = requests.post(f"{api}{path}", json=body, params=params,
                      timeout=API_TIMEOUT)
    r.raise_for_status()
    return r.json()


# ══════════════════════════════════════════════════════════════════════════════
# Student brain
# ══════════════════════════════════════════════════════════════════════════════

class LLMStudent:
    """
    Simulates one student navigating PathOptLearn.

    The LLM is used for two tasks:
      1. Answering MCQ questions (diagnostic + module quizzes)
      2. Summarising lesson content into working "memory"

    All API calls go to the real PathOptLearn backend so the
    full pipeline (search, KG, gaps, recommendations) is exercised.
    """

    def __init__(self, profile_name: str, topic: str, api: str,
                 model: str = DEFAULT_MODEL,
                 prior_history: str = "",
                 student_id: int | None = None):
        self.profile_name   = profile_name
        self.profile        = PROFILES.get(profile_name, PROFILES["intermediate"])
        self.topic          = topic
        self.api            = api
        self.model          = model
        self.prior_history  = prior_history   # optional seed context
        self.student_id     = student_id
        self.session_id     = None

        # Working memory: accumulated summaries from lessons
        self.memory: list[str] = []
        if prior_history:
            self.memory.append(f"Prior knowledge: {prior_history[:500]}")

        # Telemetry
        self.log: list[dict] = []
        self.module_results: list[dict] = []

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _system_prompt(self) -> str:
        p = self.profile
        mem = "\n".join(self.memory[-4:]) or "No prior knowledge yet."
        return (
            f"You are simulating a student learning about \"{self.topic}\".\n"
            f"Profile: {p['description']}\n"
            f"Learning style: {p['style']} learner, {p['attention']} attention.\n"
            f"Your current knowledge:\n{mem}\n\n"
            f"Answer as this student would — make realistic mistakes when "
            f"the question is hard for your level."
        )

    def _emit(self, event: str, data: dict):
        entry = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **data}
        self.log.append(entry)

    def _answer_mcq(self, question: dict) -> str:
        """Ask the LLM to answer one MCQ in character."""
        options_text = "\n".join(question.get("options", []))
        prompt = (
            f"Question: {question['question']}\n\n"
            f"{options_text}\n\n"
            f"Reply with ONLY the letter (A, B, C, or D) of your answer."
        )
        raw = _llm_call(prompt, system=self._system_prompt(), model=self.model)
        # Extract first A/B/C/D
        import re
        match = re.search(r"\b([A-Da-d])\b", raw)
        letter = match.group(1).upper() if match else "A"
        self._emit("answer_mcq", {
            "question": question.get("question", "")[:80],
            "chosen":   letter,
            "correct":  question.get("answer", "A"),
        })
        return letter

    def _read_lesson(self, content: str, module_title: str):
        """LLM summarises lesson into working memory."""
        attention = self.profile["attention"]
        excerpt   = content[:3000] if attention == "high" else content[:1200]
        prompt = (
            f"You just read a lesson titled \"{module_title}\".\n"
            f"Lesson content (excerpt):\n{excerpt}\n\n"
            f"Write 3-5 bullet points summarising what you understood and "
            f"what you found confusing or unclear. Be honest about gaps."
        )
        summary = _llm_call(prompt, system=self._system_prompt(), model=self.model)
        self.memory.append(f"[Module: {module_title}]\n{summary[:400]}")
        self._emit("lesson_read", {"module": module_title,
                                   "memory_len": len(self.memory)})
        return summary

    # ── Main flow ─────────────────────────────────────────────────────────────

    def run(self) -> dict:
        print(f"\n{'─'*60}")
        print(f"  Student:  {self.profile_name} profile")
        print(f"  Topic:    {self.topic}")
        print(f"  Model:    {self.model}")
        print(f"{'─'*60}")
        t_start = time.time()

        # 1. Deep search (enrich KG before diagnostic)
        print("\n[1/5] Running DeepSearch…")
        try:
            _api_post(self.api, "/deep-search",
                      body={"topic": self.topic,
                            "student_id": self.student_id})
            self._emit("deep_search", {"topic": self.topic})
        except Exception as e:
            print(f"  DeepSearch failed: {e} (continuing)")

        # 2. Diagnostic quiz
        print("[2/5] Taking diagnostic quiz…")
        diag_data = _api_get(self.api, "/assess", topic=self.topic)
        questions = diag_data.get("questions", [])
        answers   = [self._answer_mcq(q) for q in questions]

        eval_data = _api_post(self.api, "/assess/evaluate", body={
            "topic":     self.topic,
            "questions": questions,
            "answers":   answers,
        })
        level       = eval_data.get("level", "beginner")
        level_emoji = eval_data.get("level_emoji", "🟢")
        diag_score  = eval_data.get("score", 0)
        print(f"  Level: {level_emoji} {level} (score {diag_score}%)")
        self._emit("diagnostic", {"level": level, "score": diag_score,
                                  "feedback": eval_data.get("feedback", "")})

        # Find gaps from diagnostic
        gap_data = _api_post(self.api, "/find-gaps", body={
            "topic":      self.topic,
            "questions":  questions,
            "answers":    answers,
            "student_id": self.student_id,
        })
        diag_gaps = [g["concept"] for g in gap_data.get("gaps", [])]
        self._emit("diag_gaps", {"gaps": diag_gaps,
                                 "n_gaps": len(diag_gaps)})

        # 3. Start session + get roadmap
        print("[3/5] Generating roadmap…")
        if self.student_id:
            sess = _api_post(self.api, "/session/start", params={
                "topic": self.topic, "level": level,
                "level_emoji": level_emoji,
                "user_id": self.student_id,
            })
            self.session_id = sess.get("session_id")

        roadmap   = _api_get(self.api, "/roadmap",
                              topic=self.topic, level=level)
        modules   = roadmap.get("modules", [])
        base_topic = roadmap.get("topic", self.topic)
        print(f"  Roadmap: {len(modules)} modules across "
              f"{len(roadmap.get('levels', []))} levels")
        self._emit("roadmap", {
            "n_modules": len(modules),
            "n_levels":  len(roadmap.get("levels", [])),
            "total_duration": roadmap.get("total_duration", ""),
        })

        # 4. Module loop
        print("[4/5] Working through modules…")
        for mod in modules:
            mod_id    = mod["id"]
            mod_title = mod.get("title", f"Module {mod_id}")
            mod_uid   = f"{base_topic}::{mod_id}"
            print(f"\n  → Module {mod_id}: {mod_title}")

            attempts       = 0
            passed         = False
            quiz_score_final = 0

            while not passed and attempts < self.profile["patience"]:
                attempts += 1

                # Get lesson
                lesson = _api_get(self.api, "/lesson",
                                   topic=base_topic,
                                   module_id=mod_id,
                                   session_id=self.session_id or "")
                self._read_lesson(lesson.get("content", ""), mod_title)

                # Get quiz
                quiz_raw = _api_post(self.api, "/quiz", body={
                    "content":       lesson.get("content", ""),
                    "num_questions": 5,
                })
                quiz_qs  = quiz_raw.get("questions", [])
                quiz_ans = [self._answer_mcq(q) for q in quiz_qs]

                # Score via /find-gaps
                score_data = _api_post(self.api, "/find-gaps", body={
                    "topic":      mod_title,
                    "questions":  quiz_qs,
                    "answers":    quiz_ans,
                    "student_id": self.student_id,
                })
                quiz_score_final = score_data.get("score", 0)
                quiz_gaps = [g["concept"] for g in score_data.get("gaps", [])]
                passed    = quiz_score_final >= 70

                print(f"    Attempt {attempts}: {quiz_score_final:.0f}% "
                      f"({'PASS' if passed else 'FAIL'}) "
                      f"gaps={quiz_gaps[:3]}")

                self._emit("module_attempt", {
                    "module_uid":  mod_uid,
                    "module_title": mod_title,
                    "attempt":     attempts,
                    "score":       quiz_score_final,
                    "passed":      passed,
                    "gaps":        quiz_gaps,
                })

                if not passed and quiz_gaps:
                    # Ask recommender for remediation resources
                    try:
                        recs = _api_get(self.api, "/recommender",
                                        gaps=",".join(quiz_gaps[:3]),
                                        student_id=self.student_id or "",
                                        limit=2)
                        self._emit("remediation_resources", {
                            "module_uid": mod_uid,
                            "results":    {k: [r.get("title") for r in v]
                                           for k, v in recs.get("results", {}).items()},
                        })
                    except Exception:
                        pass

            # Persist module progress
            if self.session_id:
                try:
                    _api_post(self.api, "/next", body={
                        "session_id":           self.session_id,
                        "completed_module_uid": mod_uid,
                        "quiz_score":           quiz_score_final,
                        "num_quiz_questions":   5,
                    })
                except Exception:
                    pass

            self.module_results.append({
                "module_id":    mod_id,
                "module_title": mod_title,
                "attempts":     attempts,
                "final_score":  quiz_score_final,
                "passed":       passed,
            })

        # 5. Final recommendation (next-module recommender)
        print("\n[5/5] Requesting final recommendation…")
        completed_mods = [
            {"module_uid": f"{base_topic}::{m['module_id']}",
             "quiz_score":     m["final_score"],
             "time_spent_min": 30}
            for m in self.module_results
        ]
        try:
            rec = _api_post(self.api, "/recommend/next", body={
                "subject": base_topic,
                "historical_user_data": {
                    "session_id":        self.session_id,
                    "completed_modules": completed_mods,
                    "current_level":     level,
                    "preferred_type":    self.profile.get("style", "mixed"),
                },
                "use_graph_db":     True,
                "include_content":  False,
            })
            self._emit("final_recommendation", {
                "recommended_uid": rec.get("recommendation", {}).get("recommended_uid"),
                "reason":          rec.get("recommendation", {}).get("reason", "")[:200],
            })
        except Exception as e:
            print(f"  Recommendation failed: {e}")

        # ── Summary ────────────────────────────────────────────────────────
        elapsed      = round(time.time() - t_start, 1)
        n_modules    = len(self.module_results)
        n_passed     = sum(1 for m in self.module_results if m["passed"])
        avg_score    = (sum(m["final_score"] for m in self.module_results)
                        / max(n_modules, 1))
        total_retries = sum(m["attempts"] - 1 for m in self.module_results)

        summary = {
            "profile":         self.profile_name,
            "topic":           self.topic,
            "diag_level":      level,
            "diag_score":      diag_score,
            "diag_gaps":       diag_gaps,
            "n_modules":       n_modules,
            "n_passed":        n_passed,
            "pass_rate":       round(n_passed / max(n_modules, 1), 3),
            "avg_score":       round(avg_score, 1),
            "total_retries":   total_retries,
            "wall_time_s":     elapsed,
            "module_results":  self.module_results,
            "log_entries":     len(self.log),
        }

        print(f"\n{'─'*60}")
        print(f"  SUMMARY")
        print(f"  Profile:       {self.profile_name}")
        print(f"  Diag level:    {level} ({diag_score}%)")
        print(f"  Modules:       {n_passed}/{n_modules} passed "
              f"(pass rate {summary['pass_rate']:.0%})")
        print(f"  Avg score:     {avg_score:.1f}%")
        print(f"  Retries:       {total_retries}")
        print(f"  Wall time:     {elapsed}s")
        print(f"{'─'*60}")

        return summary


# ══════════════════════════════════════════════════════════════════════════════
# Multi-run batch
# ══════════════════════════════════════════════════════════════════════════════

def run_batch(topic: str, api: str, profile_names: list[str],
              model: str, output_path: str | None,
              prior_history: str = "") -> list[dict]:

    all_summaries = []
    all_logs      = []

    for pname in profile_names:
        if pname not in PROFILES:
            print(f"[WARN] Unknown profile '{pname}', skipping.")
            continue

        student = LLMStudent(
            profile_name=pname,
            topic=topic,
            api=api,
            model=model,
            prior_history=prior_history,
        )
        try:
            summary = student.run()
        except Exception as e:
            print(f"[ERROR] Simulation failed for '{pname}': {e}")
            summary = {"profile": pname, "error": str(e)}

        all_summaries.append(summary)
        all_logs.extend(student.log)

    # Print comparison table
    print("\n" + "═" * 72)
    print("  Simulation Comparison")
    print("═" * 72)
    header = f"  {'Profile':<16} {'Level':<14} {'Pass %':>7} {'AvgScore':>9} {'Retries':>8}"
    print(header)
    print("  " + "-" * 56)
    for s in all_summaries:
        if "error" in s:
            print(f"  {s['profile']:<16} ERROR: {s['error'][:40]}")
            continue
        print(
            f"  {s['profile']:<16} "
            f"{s.get('diag_level','?'):<14} "
            f"{s.get('pass_rate',0)*100:>6.0f}% "
            f"{s.get('avg_score',0):>8.1f}% "
            f"{s.get('total_retries',0):>8}"
        )
    print("═" * 72 + "\n")

    if output_path:
        with open(output_path, "w") as f:
            for entry in all_logs:
                f.write(json.dumps(entry) + "\n")
        print(f"Full event log saved to {output_path}")

    return all_summaries


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="PathOptLearn LLM student simulator"
    )
    parser.add_argument("--topic",    required=True,
                        help="Learning topic (e.g. 'Machine Learning')")
    parser.add_argument("--api",      default=DEFAULT_API,
                        help="PathOptLearn API base URL")
    parser.add_argument("--model",    default=DEFAULT_MODEL,
                        help="Ollama model name")
    parser.add_argument("--profiles", nargs="+",
                        default=["beginner"],
                        choices=list(PROFILES.keys()),
                        help="Student profiles to simulate")
    parser.add_argument("--history",  default="",
                        help="Optional prior knowledge context for the student")
    parser.add_argument("--output",   default=None,
                        help="Path to write JSONL event log")
    args = parser.parse_args()

    run_batch(
        topic=args.topic,
        api=args.api,
        profile_names=args.profiles,
        model=args.model,
        output_path=args.output,
        prior_history=args.history,
    )


if __name__ == "__main__":
    main()
