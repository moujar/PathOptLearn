from pydantic import BaseModel
from typing import Any


class CreateSessionRequest(BaseModel):
    topic: str
    goal: str = "Master this topic"


class SubmitAssessmentRequest(BaseModel):
    answers: dict[str, int]  # {"0": 2, "1": 0, ...}


class SubmitQuizRequest(BaseModel):
    answers: dict[str, int]  # {"0": 1, "1": 3, ...}


class SessionResponse(BaseModel):
    session_id: str
    topic: str
    assessment_questions: list[dict[str, Any]]
    message: str


class AssessmentResponse(BaseModel):
    level: str
    knowledge_gaps: list[str]
    roadmap: list[dict[str, Any]]
    total_modules: int


class ContentResponse(BaseModel):
    module_idx: int
    title: str
    content: str


class QuizResponse(BaseModel):
    quiz_questions: list[dict[str, Any]]


class QuizResultResponse(BaseModel):
    score: float
    passed: bool
    attempt: int
    current_module_idx: int
    completed_modules: list[int]
    is_course_complete: bool
