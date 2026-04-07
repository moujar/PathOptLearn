import json
import re
import time
import uuid
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional

from app.core.llm import llm
from app.core.search import search_web, edu_resources
from app.core.youtube import search_youtube


# ── State ─────────────────────────────────────────────────────────────────────

@dataclass
class LearnFlowState:
    session_id:           str   = field(default_factory=lambda: str(uuid.uuid4()))
    topic:                str   = ""
    assessment_questions: list  = field(default_factory=list)
    assessment_answers:   list  = field(default_factory=list)
    level:                str   = ""
    gaps:                 list  = field(default_factory=list)
    resources:            list  = field(default_factory=list)
    roadmap:              list  = field(default_factory=list)
    current_module:       int   = 0
    content:              str   = ""
    quiz:                 list  = field(default_factory=list)
    quiz_answers:         list  = field(default_factory=list)
    quiz_score:           float = 0.0
    attempts:             int   = 0
    completed:            bool  = False
    db_id:                Optional[int] = None

    def current_module_info(self) -> dict:
        if self.current_module < len(self.roadmap):
            return self.roadmap[self.current_module]
        return {}


# ── In-memory session store ───────────────────────────────────────────────────

_sessions: dict[str, LearnFlowState] = {}
_lock = Lock()


def get_session(session_id: str) -> LearnFlowState:
    with _lock:
        if session_id not in _sessions:
            raise KeyError(f"Session {session_id} not found")
        return _sessions[session_id]


def save_session(state: LearnFlowState):
    with _lock:
        _sessions[state.session_id] = state


def list_sessions() -> list[dict]:
    with _lock:
        return [
            {"session_id": s.session_id, "topic": s.topic,
             "level": s.level, "completed": s.completed,
             "module": s.current_module, "total": len(s.roadmap)}
            for s in _sessions.values()
        ]


# ── Node functions ─────────────────────────────────────────────────────────────

def _parse_json_list(raw: str) -> list:
    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return []


def _parse_json_obj(raw: str) -> dict:
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return {}


def node_generate_assessment(state: LearnFlowState) -> LearnFlowState:
    prompt = f"""Create a diagnostic quiz of exactly 8 questions to assess knowledge of: "{state.topic}"
Rules:
- Mix difficulty: 2 easy, 4 medium, 2 hard
- Include multiple-choice (A/B/C/D) and short-answer questions
- Each question targets a distinct concept

Return ONLY a JSON array:
[{{"q":"...", "type":"mcq"|"short", "options":["A)...","B)...","C)...","D)..."]|null, "answer":"...", "concept":"..."}}]"""

    questions = _parse_json_list(llm(prompt))
    if not questions:
        questions = [
            {"q": f"What is {state.topic}?", "type": "short", "options": None, "answer": "", "concept": "definition"},
            {"q": f"Name a key application of {state.topic}.", "type": "short", "options": None, "answer": "", "concept": "application"},
        ]
    state.assessment_questions = questions[:8]
    save_session(state)
    return state


def node_analyze_level(state: LearnFlowState) -> LearnFlowState:
    qa_pairs = "\n".join(
        f"Q{i+1}: {q['q']}\nGiven: {a}\nExpected: {q.get('answer','N/A')}\nConcept: {q.get('concept','')}"
        for i, (q, a) in enumerate(zip(state.assessment_questions, state.assessment_answers))
    )
    prompt = f"""Evaluate quiz answers for topic "{state.topic}":\n{qa_pairs}
Return ONLY JSON: {{"level":"beginner|intermediate|advanced","gaps":["gap1","gap2","gap3"],"score":0-100}}"""

    result = _parse_json_obj(llm(prompt))
    state.level = result.get("level", "beginner")
    state.gaps  = result.get("gaps", [f"fundamentals of {state.topic}"])
    save_session(state)
    return state


def node_research_resources(state: LearnFlowState) -> LearnFlowState:
    queries = [f"{state.topic} {state.level} tutorial"] + \
              [f"{g} explained" for g in state.gaps[:2]]
    seen, unique = set(), []
    for q in queries:
        for r in search_web(q, max_results=4):
            if r["url"] not in seen:
                seen.add(r["url"])
                unique.append(r)
        time.sleep(0.2)
    state.resources = unique[:12]
    save_session(state)
    return state


def node_build_roadmap(state: LearnFlowState) -> LearnFlowState:
    resource_titles = "\n".join(f"- {r['title']}: {r['snippet'][:100]}" for r in state.resources[:6])
    gaps_str        = "\n".join(f"- {g}" for g in state.gaps)
    prompt = f"""Create a personalised learning roadmap:
Topic: {state.topic} | Level: {state.level}
Gaps:\n{gaps_str}
Resources:\n{resource_titles}

Design 4-6 progressive modules. Return ONLY a JSON array:
[{{"module":1,"title":"...","objective":"...","concepts":["c1","c2"],"duration":"30 min"}}]"""

    roadmap = _parse_json_list(llm(prompt))
    if not roadmap:
        roadmap = [
            {"module": 1, "title": f"Intro to {state.topic}",    "objective": "Understand basics",    "concepts": ["overview"],  "duration": "30 min"},
            {"module": 2, "title": f"Core {state.topic}",        "objective": "Build core knowledge", "concepts": ["core"],      "duration": "45 min"},
            {"module": 3, "title": f"Applied {state.topic}",     "objective": "Practical skills",     "concepts": ["practice"],  "duration": "60 min"},
        ]
    state.roadmap        = roadmap
    state.current_module = 0
    save_session(state)
    return state


def node_generate_content(state: LearnFlowState) -> LearnFlowState:
    module  = state.current_module_info()
    context = "\n\n".join(
        f"{r['title']}: {r['snippet']}"
        for r in state.resources[:5] if r.get("snippet")
    )
    system = "You are an expert tutor. Write clear, engaging lesson content with concrete examples."
    prompt = f"""Write a comprehensive lesson for:
Module: {module.get('title')}
Objective: {module.get('objective')}
Concepts: {', '.join(module.get('concepts', []))}
Learner level: {state.level}

Reference material:
{context[:2000]}

Structure with: ## Overview, ## Key Concepts (with examples), ## Practical Application, ## Summary
Use markdown."""

    state.content  = llm(prompt, system=system)
    state.attempts = 0
    save_session(state)
    return state


def node_generate_quiz(state: LearnFlowState) -> LearnFlowState:
    module = state.current_module_info()
    prompt = f"""Create exactly 5 quiz questions to test understanding of:
Module: {module.get('title')}
Content: {state.content[:1500]}

Mix: 3 multiple-choice (A/B/C/D) + 2 short-answer.
Return ONLY a JSON array:
[{{"q":"...","type":"mcq"|"short","options":["A)...","B)...","C)...","D)..."]|null,"answer":"..."}}]"""

    quiz = _parse_json_list(llm(prompt))
    if not quiz:
        quiz = [{"q": f"Summarise what you learned about {module.get('title')}.",
                 "type": "short", "options": None, "answer": ""}]
    state.quiz = quiz[:5]
    save_session(state)
    return state


def node_evaluate_quiz(state: LearnFlowState) -> dict:
    """Returns evaluation result dict (score, feedback, suggestions)."""
    module   = state.current_module_info()
    qa_pairs = "\n".join(
        f"Q{i+1}: {q['q']}\nGiven: {a}\nCorrect: {q.get('answer','')}"
        for i, (q, a) in enumerate(zip(state.quiz, state.quiz_answers))
    )
    prompt = f"""Grade this quiz for module "{module.get('title')}":
{qa_pairs}
Partial credit for short-answer if the concept is correct.
Return ONLY JSON: {{"score":0-100,"passed":true|false,"feedback":[{{"q":1,"correct":true|false,"explanation":"..."}}]}}
Passing threshold: 70."""

    result      = _parse_json_obj(llm(prompt))
    score       = float(result.get("score", 0))
    feedback    = result.get("feedback", [])

    state.quiz_score = score
    state.attempts  += 1
    save_session(state)

    # Suggestions
    query   = f"{state.topic} {module.get('title', '')} {state.level}"
    videos  = search_youtube(f"{query} explained", max_results=4)
    edu_res = edu_resources(query)

    return {
        "score":    score,
        "passed":   score >= 70 or state.attempts >= 3,
        "attempts": state.attempts,
        "feedback": feedback,
        "videos":   videos,
        "edu_resources": edu_res,
    }


def node_advance(state: LearnFlowState) -> LearnFlowState:
    state.current_module += 1
    state.completed = state.current_module >= len(state.roadmap)
    save_session(state)
    return state
