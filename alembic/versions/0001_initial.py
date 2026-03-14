"""Initial schema — PostgreSQL + pgvector

Revision ID: 0001
Revises:
Create Date: 2026-03-14
"""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

EMBEDDING_DIM = 768  # nomic-embed-text (local Ollama)


def upgrade() -> None:
    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ── learning_sessions ─────────────────────────────────────────────────
    op.create_table(
        "learning_sessions",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("topic", sa.String, nullable=False),
        sa.Column("user_goal", sa.String, default="Master this topic"),
        sa.Column(
            "status",
            sa.Enum(
                "assessment", "analyzing", "learning", "completed",
                name="sessionstatus",
            ),
            default="assessment",
        ),
        sa.Column("detected_level", sa.String, nullable=True),
        sa.Column("current_module_idx", sa.Integer, default=0),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, onupdate=sa.func.now()),
    )
    op.create_index("idx_session_status", "learning_sessions", ["status"])

    # ── assessments ───────────────────────────────────────────────────────
    op.create_table(
        "assessments",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("session_id", sa.String, sa.ForeignKey("learning_sessions.id"), unique=True),
        sa.Column("questions", sa.JSON),
        sa.Column("answers", sa.JSON, nullable=True),
        sa.Column("score", sa.Integer, nullable=True),
        sa.Column("total_questions", sa.Integer, nullable=True),
        sa.Column("completed_at", sa.DateTime, nullable=True),
    )

    # ── roadmaps ──────────────────────────────────────────────────────────
    op.create_table(
        "roadmaps",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("session_id", sa.String, sa.ForeignKey("learning_sessions.id"), unique=True),
        sa.Column("level_explanation", sa.Text, nullable=True),
        sa.Column("total_modules", sa.Integer),
        sa.Column("estimated_hours", sa.Float, nullable=True),
        sa.Column("raw_llm_response", sa.JSON),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )

    # ── modules ───────────────────────────────────────────────────────────
    op.create_table(
        "modules",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("session_id", sa.String, sa.ForeignKey("learning_sessions.id")),
        sa.Column("order_index", sa.Integer),
        sa.Column("title", sa.String),
        sa.Column("description", sa.Text),
        sa.Column("topics", sa.JSON),
        sa.Column("difficulty", sa.String),
        sa.Column("estimated_time", sa.String, nullable=True),
        sa.Column("resources", sa.JSON),
        sa.Column(
            "status",
            sa.Enum("locked", "active", "quiz_pending", "completed", name="modulestatus"),
            default="locked",
        ),
    )
    op.create_index("idx_module_session", "modules", ["session_id", "status"])

    # ── module_contents ───────────────────────────────────────────────────
    op.create_table(
        "module_contents",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("module_id", sa.String, sa.ForeignKey("modules.id"), unique=True),
        sa.Column("markdown", sa.Text),
        sa.Column("word_count", sa.Integer, nullable=True),
        sa.Column("generated_at", sa.DateTime, server_default=sa.func.now()),
    )

    # ── quizzes ───────────────────────────────────────────────────────────
    op.create_table(
        "quizzes",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("module_id", sa.String, sa.ForeignKey("modules.id")),
        sa.Column("questions", sa.JSON),
        sa.Column("pass_score", sa.Float, default=0.7),
        sa.Column("version", sa.Integer, default=1),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )

    # ── quiz_attempts ─────────────────────────────────────────────────────
    op.create_table(
        "quiz_attempts",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("quiz_id", sa.String, sa.ForeignKey("quizzes.id")),
        sa.Column("session_id", sa.String),
        sa.Column("answers", sa.JSON),
        sa.Column("score", sa.Integer),
        sa.Column("total_questions", sa.Integer),
        sa.Column("passed", sa.Boolean),
        sa.Column("taken_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("idx_attempt_session", "quiz_attempts", ["session_id", "quiz_id"])

    # ── resource_cache ────────────────────────────────────────────────────
    op.create_table(
        "resource_cache",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("topic", sa.String),
        sa.Column("level", sa.String),
        sa.Column("resources", sa.JSON),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("expires_at", sa.DateTime),
    )
    op.create_index("idx_resource_topic", "resource_cache", ["topic", "level"])

    # ── pgvector tables ────────────────────────────────────────────────────
    op.create_table(
        "learning_resources",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("topic", sa.String, nullable=False),
        sa.Column("subtopic", sa.String, nullable=True),
        sa.Column("difficulty", sa.String, nullable=False),
        sa.Column("type", sa.String, nullable=True),
        sa.Column("title", sa.String, nullable=False),
        sa.Column("url", sa.String, nullable=True),
        sa.Column("quality_score", sa.Float, default=0.8),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("idx_lr_topic_level", "learning_resources", ["topic", "difficulty"])

    op.create_table(
        "generated_content",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("session_id", sa.String, nullable=False),
        sa.Column("module_title", sa.String, nullable=False),
        sa.Column("level", sa.String, nullable=True),
        sa.Column("preview", sa.Text, nullable=True),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=True),
        sa.Column("generated_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("idx_gc_session", "generated_content", ["session_id"])

    op.create_table(
        "user_knowledge",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("session_id", sa.String, nullable=False),
        sa.Column("concept", sa.String, nullable=False),
        sa.Column("mastery_score", sa.Float, default=0.5),
        sa.Column("times_tested", sa.Integer, default=1),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=True),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("idx_uk_session", "user_knowledge", ["session_id"])


def downgrade() -> None:
    op.drop_table("user_knowledge")
    op.drop_table("generated_content")
    op.drop_table("learning_resources")
    op.drop_table("resource_cache")
    op.drop_table("quiz_attempts")
    op.drop_table("quizzes")
    op.drop_table("module_contents")
    op.drop_table("modules")
    op.drop_table("roadmaps")
    op.drop_table("assessments")
    op.drop_table("learning_sessions")
    op.execute("DROP TYPE IF EXISTS sessionstatus")
    op.execute("DROP TYPE IF EXISTS modulestatus")
    op.execute("DROP EXTENSION IF EXISTS vector")
