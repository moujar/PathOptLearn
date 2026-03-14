"""
LearnFlow AI — LangGraph stateful learning loop.
PostgreSQL is used for BOTH relational data AND LangGraph checkpointing.
No Redis required.
"""

from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.config import settings


# ── State definition ────────────────────────────────────────────────────────


class LearningState(TypedDict):
    # Session
    session_id: str
    topic: str
    user_goal: str
    # Assessment
    assessment_questions: list[dict]
    assessment_answers: dict
    score: int
    total_questions: int
    # Level analysis
    detected_level: str
    knowledge_gaps: list[str]
    strengths: list[str]
    # Resources
    vector_resources: list[dict]
    web_resources: list[dict]
    # Roadmap
    roadmap: list[dict]
    current_module_idx: int
    total_modules: int
    # Current module
    module_content: str
    quiz_questions: list[dict]
    quiz_answers: dict
    quiz_score: float
    quiz_passed: bool
    quiz_attempt_count: int
    # Progress
    completed_modules: list[int]
    phase: str


# ── Node functions ──────────────────────────────────────────────────────────


async def generate_assessment(state: LearningState) -> dict:
    """Generate 8-question diagnostic quiz using Claude."""
    from app.llm.prompts import ASSESSMENT_SYSTEM_PROMPT
    from app.llm.client import call_claude_json
    from app.vectordb.queries import find_resources

    resources = await find_resources(state["topic"], "beginner", top_k=5)
    result = await call_claude_json(
        system=ASSESSMENT_SYSTEM_PROMPT,
        user=(
            f"Topic: {state['topic']}\n"
            f"Goal: {state['user_goal']}\n"
            f"Related resources: {resources}"
        ),
    )
    return {
        "assessment_questions": result["questions"],
        "total_questions": len(result["questions"]),
        "phase": "assessment",
    }


async def analyze_level(state: LearningState) -> dict:
    """Score the assessment answers."""
    questions = state["assessment_questions"]
    answers = state.get("assessment_answers", {})
    correct = sum(
        1
        for i, q in enumerate(questions)
        if answers.get(str(i)) == q["correctIndex"]
    )
    return {
        "score": correct,
        "phase": "analyzing",
    }


async def research_resources(state: LearningState) -> dict:
    """Search pgvector + ask Claude for resources."""
    from app.vectordb.queries import find_resources
    from app.llm.prompts import RESOURCE_SYSTEM_PROMPT
    from app.llm.client import call_claude_json

    level = state.get("detected_level", "beginner")
    vector_res = await find_resources(state["topic"], level)
    web_res = await call_claude_json(
        system=RESOURCE_SYSTEM_PROMPT,
        user=(
            f"Topic: {state['topic']}\n"
            f"Level: {level}\n"
            f"Gaps: {state.get('knowledge_gaps', [])}"
        ),
    )
    return {
        "vector_resources": vector_res,
        "web_resources": web_res.get("resources", []),
    }


async def build_roadmap(state: LearningState) -> dict:
    """Generate personalised learning roadmap."""
    from app.llm.prompts import ROADMAP_SYSTEM_PROMPT
    from app.llm.client import call_claude_json

    assessment_summary = [
        {
            "question": q["question"],
            "correct": state.get("assessment_answers", {}).get(str(i))
            == q["correctIndex"],
            "difficulty": q["difficulty"],
        }
        for i, q in enumerate(state["assessment_questions"])
    ]

    result = await call_claude_json(
        system=ROADMAP_SYSTEM_PROMPT,
        user=(
            f"Topic: {state['topic']}\n"
            f"Score: {state.get('score', 0)}/{state.get('total_questions', 8)}\n"
            f"Assessment: {assessment_summary}\n"
            f"Resources found: {state.get('vector_resources', [])}\n"
            f"Web resources: {state.get('web_resources', [])}"
        ),
    )
    return {
        "detected_level": result["level"],
        "knowledge_gaps": result.get("knowledgeGaps", []),
        "strengths": result.get("strengths", []),
        "roadmap": result["roadmap"],
        "total_modules": len(result["roadmap"]),
        "current_module_idx": 0,
        "completed_modules": [],
        "phase": "roadmap",
    }


async def generate_content(state: LearningState) -> dict:
    """Generate lesson content for the current module."""
    from app.llm.prompts import CONTENT_SYSTEM_PROMPT
    from app.llm.client import call_claude_text
    from app.vectordb.queries import find_similar_content
    from app.vectordb.embeddings import store_content

    idx = state["current_module_idx"]
    module = state["roadmap"][idx]
    prev_titles = [
        state["roadmap"][i]["title"] for i in state.get("completed_modules", [])
    ]
    similar = await find_similar_content(module["title"])

    content = await call_claude_text(
        system=CONTENT_SYSTEM_PROMPT.format(level=state.get("detected_level", "beginner")),
        user=(
            f"Topic: {state['topic']}\n"
            f"Module: {module['title']}\n"
            f"Description: {module['description']}\n"
            f"Subtopics: {module.get('topics', [])}\n"
            f"Previous modules: {prev_titles}\n"
            f"Existing similar content (avoid overlap): {similar}"
        ),
    )

    await store_content(state["session_id"], module["title"], content)
    return {
        "module_content": content,
        "quiz_attempt_count": 0,
        "phase": "learning",
    }


async def generate_quiz(state: LearningState) -> dict:
    """Generate a 5-question module quiz."""
    from app.llm.prompts import QUIZ_SYSTEM_PROMPT
    from app.llm.client import call_claude_json
    from app.vectordb.queries import find_user_weak_concepts

    weak = await find_user_weak_concepts(state["session_id"])
    module = state["roadmap"][state["current_module_idx"]]

    result = await call_claude_json(
        system=QUIZ_SYSTEM_PROMPT.format(level=state.get("detected_level", "beginner")),
        user=(
            f"Topic: {state['topic']}\n"
            f"Module: {module['title']}\n"
            f"Content taught:\n{state.get('module_content', '')[:3000]}\n"
            f"Attempt #{state.get('quiz_attempt_count', 0) + 1}\n"
            f"Weak concepts: {weak}"
        ),
    )
    return {
        "quiz_questions": result["questions"],
        "phase": "quiz",
    }


async def evaluate_quiz(state: LearningState) -> dict:
    """Score quiz and update knowledge vectors."""
    from app.vectordb.embeddings import update_user_knowledge

    questions = state["quiz_questions"]
    answers = state.get("quiz_answers", {})
    correct = sum(
        1
        for i, q in enumerate(questions)
        if answers.get(str(i)) == q["correctIndex"]
    )
    score = correct / len(questions) if questions else 0.0
    passed = score >= 0.7

    for i, q in enumerate(questions):
        is_correct = answers.get(str(i)) == q["correctIndex"]
        await update_user_knowledge(
            state["session_id"],
            q.get("concept", q["question"]),
            1.0 if is_correct else 0.3,
        )

    return {
        "quiz_score": score,
        "quiz_passed": passed,
        "quiz_attempt_count": state.get("quiz_attempt_count", 0) + 1,
        "phase": "quiz_result",
    }


async def advance_module(state: LearningState) -> dict:
    """Mark current module done, move to next."""
    completed = list(state.get("completed_modules", []))
    completed.append(state["current_module_idx"])
    return {
        "completed_modules": completed,
        "current_module_idx": state["current_module_idx"] + 1,
    }


# ── Conditional edge routers ────────────────────────────────────────────────


def quiz_router(state: LearningState) -> Literal["advance", "retry"]:
    return "advance" if state.get("quiz_passed") else "retry"


def completion_router(state: LearningState) -> Literal["continue", "completed"]:
    if state["current_module_idx"] >= state["total_modules"]:
        return "completed"
    return "continue"


# ── Graph builder ───────────────────────────────────────────────────────────


def _build_graph(checkpointer):
    graph = StateGraph(LearningState)

    graph.add_node("generate_assessment", generate_assessment)
    graph.add_node("analyze_level", analyze_level)
    graph.add_node("research_resources", research_resources)
    graph.add_node("build_roadmap", build_roadmap)
    graph.add_node("generate_content", generate_content)
    graph.add_node("generate_quiz", generate_quiz)
    graph.add_node("evaluate_quiz", evaluate_quiz)
    graph.add_node("advance_module", advance_module)

    graph.set_entry_point("generate_assessment")
    graph.add_edge("generate_assessment", "analyze_level")
    graph.add_edge("analyze_level", "research_resources")
    graph.add_edge("research_resources", "build_roadmap")
    graph.add_edge("build_roadmap", "generate_content")
    graph.add_edge("generate_content", "generate_quiz")
    graph.add_edge("generate_quiz", "evaluate_quiz")

    graph.add_conditional_edges(
        "evaluate_quiz",
        quiz_router,
        {"advance": "advance_module", "retry": "generate_content"},
    )
    graph.add_conditional_edges(
        "advance_module",
        completion_router,
        {"continue": "generate_content", "completed": END},
    )

    return graph.compile(checkpointer=checkpointer, interrupt_before=[
        "analyze_level",   # wait for assessment answers
        "generate_quiz",   # wait for user to request quiz
        "evaluate_quiz",   # wait for quiz answers
    ])


# ── Singleton that is initialised at app startup ────────────────────────────
# Use `await setup_graph()` in the FastAPI lifespan handler.

_learning_app = None
_checkpointer_conn = None


async def setup_graph():
    """Create the PostgreSQL checkpointer and compile the graph."""
    global _learning_app, _checkpointer_conn
    import psycopg

    conn = await psycopg.AsyncConnection.connect(
        settings.DATABASE_URL, autocommit=True
    )
    _checkpointer_conn = conn
    checkpointer = AsyncPostgresSaver(conn)
    await checkpointer.setup()
    _learning_app = _build_graph(checkpointer)
    return _learning_app


def get_graph():
    if _learning_app is None:
        raise RuntimeError("Graph not initialised — call setup_graph() first")
    return _learning_app
