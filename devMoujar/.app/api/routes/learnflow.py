from fastapi import APIRouter, HTTPException
from app.api.models import (
    StartRequest, StartResponse,
    AssessmentRequest, AssessmentResponse,
    ContentResponse, QuizResponse,
    QuizAnswerRequest, EvaluationResponse,
    AdvanceResponse, SessionRequest,
)
from app.core.learnflow import (
    LearnFlowState,
    get_session, save_session, list_sessions,
    node_generate_assessment,
    node_analyze_level,
    node_research_resources,
    node_build_roadmap,
    node_generate_content,
    node_generate_quiz,
    node_evaluate_quiz,
    node_advance,
)

router = APIRouter(prefix="/api/learnflow", tags=["learnflow"])


@router.post("/start", response_model=StartResponse)
def start(req: StartRequest):
    """Create session → generate diagnostic assessment."""
    state = LearnFlowState(topic=req.topic)
    save_session(state)
    state = node_generate_assessment(state)
    return StartResponse(
        session_id=state.session_id,
        topic=state.topic,
        questions=state.assessment_questions,
    )


@router.post("/assessment", response_model=AssessmentResponse)
def submit_assessment(req: AssessmentRequest):
    """Submit assessment answers → detect level, gaps, build roadmap."""
    state = get_session(req.session_id)
    state.assessment_answers = req.answers
    state = node_analyze_level(state)
    state = node_research_resources(state)
    state = node_build_roadmap(state)
    return AssessmentResponse(
        session_id=state.session_id,
        level=state.level,
        gaps=state.gaps,
        roadmap=state.roadmap,
    )


@router.post("/content", response_model=ContentResponse)
def get_content(req: SessionRequest):
    """Generate lesson content for the current module."""
    state  = get_session(req.session_id)
    if state.completed:
        raise HTTPException(400, "All modules completed.")
    state  = node_generate_content(state)
    module = state.current_module_info()
    return ContentResponse(
        session_id=state.session_id,
        module_idx=state.current_module,
        module_title=module.get("title", ""),
        module_obj=module.get("objective", ""),
        total_modules=len(state.roadmap),
        content=state.content,
    )


@router.post("/quiz", response_model=QuizResponse)
def get_quiz(req: SessionRequest):
    """Generate quiz questions for the current module."""
    state = get_session(req.session_id)
    state = node_generate_quiz(state)
    return QuizResponse(session_id=state.session_id, questions=state.quiz)


@router.post("/evaluate", response_model=EvaluationResponse)
def evaluate(req: QuizAnswerRequest):
    """Evaluate quiz answers → score + YouTube/web suggestions."""
    state = get_session(req.session_id)
    state.quiz_answers = req.answers
    save_session(state)
    result = node_evaluate_quiz(state)
    return EvaluationResponse(
        session_id=req.session_id,
        **result,
    )


@router.post("/advance", response_model=AdvanceResponse)
def advance(req: SessionRequest):
    """Advance to the next module (call after passing quiz)."""
    state = get_session(req.session_id)
    state = node_advance(state)
    next_title = (
        state.roadmap[state.current_module].get("title")
        if not state.completed else None
    )
    return AdvanceResponse(
        session_id=state.session_id,
        completed=state.completed,
        current_module=state.current_module,
        total_modules=len(state.roadmap),
        next_title=next_title,
    )


@router.get("/sessions")
def sessions():
    return list_sessions()
