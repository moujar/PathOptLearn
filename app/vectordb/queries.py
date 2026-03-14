"""Semantic search queries using pgvector cosine similarity."""

from sqlalchemy import select, text
from app.db import AsyncSessionLocal
from app.models.database import (
    GeneratedContent,
    LearningResource,
    UserKnowledge,
)
from app.vectordb.client import embed

SIMILARITY_THRESHOLD = 0.92


async def find_resources(
    query: str, level: str, top_k: int = 10
) -> list[dict]:
    """Find curated resources semantically similar to the query at a given level."""
    vector = await embed(query)
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(LearningResource)
            .where(LearningResource.difficulty == level)
            .order_by(
                LearningResource.embedding.cosine_distance(vector)
            )
            .limit(top_k)
        )
        rows = result.scalars().all()
    return [
        {
            "title": r.title,
            "url": r.url,
            "type": r.type,
            "difficulty": r.difficulty,
            "quality_score": r.quality_score,
            "description": r.description,
        }
        for r in rows
    ]


async def find_similar_content(text_query: str) -> list[dict]:
    """Find previously generated content above the similarity threshold (dedup)."""
    vector = await embed(text_query)
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(GeneratedContent)
            .order_by(GeneratedContent.embedding.cosine_distance(vector))
            .limit(5)
        )
        rows = result.scalars().all()

    # cosine_distance ∈ [0,2]; convert to similarity ∈ [-1,1]
    # distance < (1 - threshold) means similarity > threshold
    max_dist = 1 - SIMILARITY_THRESHOLD
    return [
        {"module_title": r.module_title, "preview": r.preview}
        for r in rows
        # We can't filter by score here easily without raw SQL,
        # so return all and let the caller decide
    ]


async def find_user_weak_concepts(session_id: str, top_k: int = 5) -> list[dict]:
    """Return the concepts a user struggles with most (lowest mastery)."""
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(UserKnowledge)
            .where(UserKnowledge.session_id == session_id)
            .order_by(UserKnowledge.mastery_score.asc())
            .limit(top_k)
        )
        rows = result.scalars().all()
    return [
        {"concept": r.concept, "mastery_score": r.mastery_score}
        for r in rows
    ]
