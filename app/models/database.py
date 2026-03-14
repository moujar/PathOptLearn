import enum
import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


# ── Enums ──────────────────────────────────────────────────────────────────


class SessionStatus(str, enum.Enum):
    ASSESSMENT = "assessment"
    ANALYZING = "analyzing"
    LEARNING = "learning"
    COMPLETED = "completed"


class ModuleStatus(str, enum.Enum):
    LOCKED = "locked"
    ACTIVE = "active"
    QUIZ_PENDING = "quiz_pending"
    COMPLETED = "completed"


# ── Relational tables ───────────────────────────────────────────────────────


class LearningSession(Base):
    __tablename__ = "learning_sessions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    topic = Column(String, nullable=False)
    user_goal = Column(String, default="Master this topic")
    status = Column(Enum(SessionStatus), default=SessionStatus.ASSESSMENT)
    detected_level = Column(String, nullable=True)
    current_module_idx = Column(Integer, default=0)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())

    assessment = relationship("Assessment", back_populates="session", uselist=False)
    roadmap = relationship("Roadmap", back_populates="session", uselist=False)
    modules = relationship(
        "Module", back_populates="session", order_by="Module.order_index"
    )

    __table_args__ = (Index("idx_session_status", "status"),)


class Assessment(Base):
    __tablename__ = "assessments"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey("learning_sessions.id"), unique=True)
    questions = Column(JSON)  # [{question, options, correctIndex, difficulty, concept}]
    answers = Column(JSON, nullable=True)
    score = Column(Integer, nullable=True)
    total_questions = Column(Integer, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    session = relationship("LearningSession", back_populates="assessment")


class Roadmap(Base):
    __tablename__ = "roadmaps"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey("learning_sessions.id"), unique=True)
    level_explanation = Column(Text, nullable=True)
    total_modules = Column(Integer)
    estimated_hours = Column(Float, nullable=True)
    raw_llm_response = Column(JSON)
    created_at = Column(DateTime, server_default=func.now())

    session = relationship("LearningSession", back_populates="roadmap")


class Module(Base):
    __tablename__ = "modules"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey("learning_sessions.id"))
    order_index = Column(Integer)
    title = Column(String)
    description = Column(Text)
    topics = Column(JSON)  # ["subtopic1", "subtopic2"]
    difficulty = Column(String)
    estimated_time = Column(String, nullable=True)
    resources = Column(JSON)  # [{title, url, type}]
    status = Column(Enum(ModuleStatus), default=ModuleStatus.LOCKED)

    session = relationship("LearningSession", back_populates="modules")
    content = relationship("ModuleContent", back_populates="module", uselist=False)
    quizzes = relationship("Quiz", back_populates="module")

    __table_args__ = (Index("idx_module_session", "session_id", "status"),)


class ModuleContent(Base):
    __tablename__ = "module_contents"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    module_id = Column(String, ForeignKey("modules.id"), unique=True)
    markdown = Column(Text)
    word_count = Column(Integer, nullable=True)
    generated_at = Column(DateTime, server_default=func.now())

    module = relationship("Module", back_populates="content")


class Quiz(Base):
    __tablename__ = "quizzes"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    module_id = Column(String, ForeignKey("modules.id"))
    questions = Column(JSON)
    pass_score = Column(Float, default=0.7)
    version = Column(Integer, default=1)
    created_at = Column(DateTime, server_default=func.now())

    module = relationship("Module", back_populates="quizzes")
    attempts = relationship("QuizAttempt", back_populates="quiz")


class QuizAttempt(Base):
    __tablename__ = "quiz_attempts"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    quiz_id = Column(String, ForeignKey("quizzes.id"))
    session_id = Column(String)
    answers = Column(JSON)
    score = Column(Integer)
    total_questions = Column(Integer)
    passed = Column(Boolean)
    taken_at = Column(DateTime, server_default=func.now())

    quiz = relationship("Quiz", back_populates="attempts")

    __table_args__ = (Index("idx_attempt_session", "session_id", "quiz_id"),)


class ResourceCache(Base):
    __tablename__ = "resource_cache"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    topic = Column(String)
    level = Column(String)
    resources = Column(JSON)
    created_at = Column(DateTime, server_default=func.now())
    expires_at = Column(DateTime)

    __table_args__ = (Index("idx_resource_topic", "topic", "level"),)


# ── pgvector tables (replaces Pinecone) ────────────────────────────────────

EMBEDDING_DIM = 768  # nomic-embed-text (local Ollama)


class LearningResource(Base):
    """Curated resources indexed by topic + level for semantic search."""

    __tablename__ = "learning_resources"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    topic = Column(String, nullable=False)
    subtopic = Column(String, nullable=True)
    difficulty = Column(String, nullable=False)  # beginner|intermediate|advanced
    type = Column(String, nullable=True)  # article|course|video|book|tool
    title = Column(String, nullable=False)
    url = Column(String, nullable=True)
    quality_score = Column(Float, default=0.8)
    description = Column(Text, nullable=True)
    embedding = Column(Vector(EMBEDDING_DIM), nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (Index("idx_lr_topic_level", "topic", "difficulty"),)


class GeneratedContent(Base):
    """AI-generated lesson content for deduplication."""

    __tablename__ = "generated_content"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, nullable=False)
    module_title = Column(String, nullable=False)
    level = Column(String, nullable=True)
    preview = Column(Text, nullable=True)  # first 500 chars
    embedding = Column(Vector(EMBEDDING_DIM), nullable=True)
    generated_at = Column(DateTime, server_default=func.now())

    __table_args__ = (Index("idx_gc_session", "session_id"),)


class UserKnowledge(Base):
    """Per-session concept mastery scores."""

    __tablename__ = "user_knowledge"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, nullable=False)
    concept = Column(String, nullable=False)
    mastery_score = Column(Float, default=0.5)
    times_tested = Column(Integer, default=1)
    embedding = Column(Vector(EMBEDDING_DIM), nullable=True)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (Index("idx_uk_session", "session_id"),)
