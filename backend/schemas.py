"""Shared Pydantic models for AdaptLearn AI — used across all FastAPI routers."""
from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timezone


class GoalOutput(BaseModel):
    goal_id: str
    user_id: str = ""
    raw_goal: str = ""
    topic: str
    sub_topics: List[str]
    difficulty_hint: str  # 'beginner' | 'intermediate' | 'advanced'
    kg_query_terms: List[str]
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class HarvestedDoc(BaseModel):
    doc_id: str
    url: str
    title: str
    content_text: str
    source_type: str  # 'web' | 'youtube'
    goal_id: str


class QuizItem(BaseModel):
    item_id: str
    question: str
    options: List[str]
    correct_index: int
    difficulty_b: float = Field(description="IRT difficulty parameter (-3 to +3)")
    discrimination_a: float = Field(description="IRT discrimination parameter (0.5 to 2.5)")
    guessing_c: float = 0.25
    bloom_level: str  # remember | understand | apply | analyze
    concept: str
    source_url: str = ""


class SessionResponse(BaseModel):
    item_id: str
    answer_index: int
    response_time_ms: int
    correct: Optional[bool] = None


class KnowledgeVector(BaseModel):
    student_id: str
    theta: float = Field(description="Overall IRT ability estimate")
    concept_mastery: Dict[str, float] = Field(description="Concept -> mastery 0.0-1.0")
    confidence_interval: List[float] = Field(description="[lower, upper] 95% CI for theta")
    assessed_at: str
    last_studied_at: Optional[Dict[str, str]] = Field(default_factory=dict)


class KGNode(BaseModel):
    node_id: str
    concept: str
    difficulty: float
    embedding: List[float] = Field(default_factory=list)
    definition: str = ""


class KGEdge(BaseModel):
    from_node: str
    to_node: str
    relation: str  # 'prerequisite' | 'semantic'
    weight: float


class PathStep(BaseModel):
    step: int
    concept: str
    resource_type: str  # 'video' | 'article' | 'quiz'
    url: str
    duration_min: int
    predicted_mastery_delta: float
    node_id: str = ""


class LearningPath(BaseModel):
    path_id: str
    student_id: str
    goal_id: str
    algorithm: str  # 'DRL-PPO' | 'AKT-greedy' | 'DKVMN-greedy' | 'BKT-greedy'
    steps: List[PathStep]
    energy_score: float = Field(description="Lower = more efficient path")
    predicted_completion_sessions: int
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class BenchmarkResult(BaseModel):
    algorithm: str
    auc: float
    accuracy: float
    rmse: float
    recall_at_k: float
    les: float = Field(description="Learning Efficiency Score = knowledge_gain / time_invested")
    path: Optional[LearningPath] = None


class NextItemResponse(BaseModel):
    next_item: Optional[QuizItem]
    current_theta: float
    items_remaining_estimate: int
    should_stop: bool
