from pydantic import BaseModel
from typing import Optional


# ── Search ────────────────────────────────────────────────────────────────────

class QuickSearchRequest(BaseModel):
    topic: str

class QuickSearchResponse(BaseModel):
    topic:   str
    summary: str
    results: list[dict]

class YoutubeSearchRequest(BaseModel):
    query:       str
    max_results: int = 4

class YoutubeSearchResponse(BaseModel):
    query:  str
    videos: list[dict]


# ── LearnFlow ─────────────────────────────────────────────────────────────────

class StartRequest(BaseModel):
    topic: str

class StartResponse(BaseModel):
    session_id: str
    topic:      str
    questions:  list[dict]

class AssessmentRequest(BaseModel):
    session_id: str
    answers:    list[str]

class AssessmentResponse(BaseModel):
    session_id: str
    level:      str
    gaps:       list[str]
    roadmap:    list[dict]

class ContentResponse(BaseModel):
    session_id:     str
    module_idx:     int
    module_title:   str
    module_obj:     str
    total_modules:  int
    content:        str

class QuizResponse(BaseModel):
    session_id: str
    questions:  list[dict]

class QuizAnswerRequest(BaseModel):
    session_id: str
    answers:    list[str]

class EvaluationResponse(BaseModel):
    session_id:    str
    score:         float
    passed:        bool
    attempts:      int
    feedback:      list[dict]
    videos:        list[dict]
    edu_resources: list[dict]

class AdvanceResponse(BaseModel):
    session_id:     str
    completed:      bool
    current_module: int
    total_modules:  int
    next_title:     Optional[str] = None

class SessionRequest(BaseModel):
    session_id: str
