import uuid

from fastapi import APIRouter, HTTPException

from app.graph.learning_graph import get_graph
from app.models.schemas import (
    AssessmentResponse,
    CreateSessionRequest,
    SessionResponse,
    SubmitAssessmentRequest,
)

router = APIRouter()


@router.post("/create", response_model=SessionResponse)
async def create_session(req: CreateSessionRequest):
    """Start a new learning session. No login required."""
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    graph = get_graph()

    # Run until first interrupt (before analyze_level) — gives us assessment questions
    state = await graph.ainvoke(
        {
            "session_id": session_id,
            "topic": req.topic,
            "user_goal": req.goal,
            "assessment_questions": [],
            "assessment_answers": {},
            "score": 0,
            "total_questions": 0,
            "detected_level": "",
            "knowledge_gaps": [],
            "strengths": [],
            "vector_resources": [],
            "web_resources": [],
            "roadmap": [],
            "current_module_idx": 0,
            "total_modules": 0,
            "module_content": "",
            "quiz_questions": [],
            "quiz_answers": {},
            "quiz_score": 0.0,
            "quiz_passed": False,
            "quiz_attempt_count": 0,
            "completed_modules": [],
            "phase": "landing",
        },
        config,
    )

    return SessionResponse(
        session_id=session_id,
        topic=req.topic,
        assessment_questions=state["assessment_questions"],
        message="Bookmark this session_id to resume later",
    )


@router.post("/{session_id}/assess", response_model=AssessmentResponse)
async def submit_assessment(session_id: str, req: SubmitAssessmentRequest):
    """Submit assessment answers — triggers level analysis + roadmap generation."""
    config = {"configurable": {"thread_id": session_id}}
    graph = get_graph()

    # Inject answers into the graph state
    await graph.aupdate_state(config, {"assessment_answers": req.answers})

    # Resume: analyze_level → research_resources → build_roadmap → (interrupt before generate_quiz)
    state = await graph.ainvoke(None, config)

    return AssessmentResponse(
        level=state["detected_level"],
        knowledge_gaps=state["knowledge_gaps"],
        roadmap=state["roadmap"],
        total_modules=state["total_modules"],
    )


@router.get("/{session_id}")
async def get_session(session_id: str):
    """Resume a session — returns full current state."""
    config = {"configurable": {"thread_id": session_id}}
    graph = get_graph()
    snapshot = await graph.aget_state(config)
    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail="Session not found")
    return snapshot.values
