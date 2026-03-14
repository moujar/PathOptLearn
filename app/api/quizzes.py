from fastapi import APIRouter, HTTPException

from app.graph.learning_graph import get_graph
from app.models.schemas import QuizResponse, QuizResultResponse, SubmitQuizRequest

router = APIRouter()


@router.post("/{session_id}/generate-quiz", response_model=QuizResponse)
async def generate_quiz(session_id: str):
    """Generate a 5-question quiz for the current module."""
    config = {"configurable": {"thread_id": session_id}}
    graph = get_graph()

    snapshot = await graph.aget_state(config)
    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail="Session not found")

    # Resume through generate_quiz node
    state = await graph.ainvoke(None, config)
    return QuizResponse(quiz_questions=state["quiz_questions"])


@router.post("/{session_id}/submit-quiz", response_model=QuizResultResponse)
async def submit_quiz(session_id: str, req: SubmitQuizRequest):
    """Submit quiz answers, score them, advance or retry."""
    config = {"configurable": {"thread_id": session_id}}
    graph = get_graph()

    await graph.aupdate_state(config, {"quiz_answers": req.answers})
    state = await graph.ainvoke(None, config)

    return QuizResultResponse(
        score=state["quiz_score"],
        passed=state["quiz_passed"],
        attempt=state["quiz_attempt_count"],
        current_module_idx=state["current_module_idx"],
        completed_modules=state["completed_modules"],
        is_course_complete=state["current_module_idx"] >= state["total_modules"],
    )
