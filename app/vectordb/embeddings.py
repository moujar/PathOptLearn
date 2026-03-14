"""Store embeddings for generated content and user knowledge."""

from sqlalchemy import select
from app.db import AsyncSessionLocal
from app.models.database import GeneratedContent, UserKnowledge
from app.vectordb.client import embed


async def store_content(session_id: str, title: str, content: str) -> None:
    """Embed and store generated module content for deduplication."""
    vector = await embed(content[:2000])
    async with AsyncSessionLocal() as db:
        record = GeneratedContent(
            session_id=session_id,
            module_title=title,
            preview=content[:500],
            embedding=vector,
        )
        db.add(record)
        await db.commit()


async def update_user_knowledge(
    session_id: str, concept: str, mastery: float
) -> None:
    """Upsert a concept mastery score for a session."""
    vector = await embed(concept)
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(UserKnowledge).where(
                UserKnowledge.session_id == session_id,
                UserKnowledge.concept == concept,
            )
        )
        existing = result.scalar_one_or_none()
        if existing:
            existing.mastery_score = mastery
            existing.times_tested = (existing.times_tested or 0) + 1
            existing.embedding = vector
        else:
            db.add(
                UserKnowledge(
                    session_id=session_id,
                    concept=concept,
                    mastery_score=mastery,
                    times_tested=1,
                    embedding=vector,
                )
            )
        await db.commit()
