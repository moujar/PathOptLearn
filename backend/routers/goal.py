"""Stage 1 — Goal Elicitation & Intent Parsing."""
import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/goal", tags=["Goal Elicitation"])
logger = logging.getLogger(__name__)


# ── Request models ────────────────────────────────────────────────────────

class GoalParseRequest(BaseModel):
    user_id: str
    raw_goal: str


class HarvestRequest(BaseModel):
    goal_id: str


# ── Dependencies ──────────────────────────────────────────────────────────

async def get_redis():
    from backend.config import get_settings
    import redis.asyncio as aioredis
    settings = get_settings()
    try:
        r = aioredis.from_url(settings.redis_url, decode_responses=True)
        await r.ping()
        return r
    except Exception:
        return None


async def get_openai():
    from backend.config import get_settings
    from openai import OpenAI
    settings = get_settings()
    return OpenAI(api_key=settings.openai_api_key)


# ── Endpoints ─────────────────────────────────────────────────────────────

@router.post("/parse")
async def parse_goal(
    request: GoalParseRequest,
    redis_client=Depends(get_redis),
    openai_client=Depends(get_openai),
):
    """
    Parse a natural language learning goal into structured GoalOutput.
    Uses GPT-4 to extract topic tree, sub-topics, difficulty hint, and KG query terms.
    Stores result in Redis with TTL 3600s.
    """
    from backend.config import get_settings
    settings = get_settings()
    goal_id = str(uuid.uuid4())[:12]

    system_prompt = (
        "You are an educational topic parser. Given a student learning goal, extract: "
        "(a) main topic, (b) 4-8 sub-topics in logical learning order, "
        "(c) difficulty_hint: beginner|intermediate|advanced, "
        "(d) kg_query_terms: 6-10 keywords for knowledge graph construction. "
        'Return ONLY valid JSON: {"topic":str,"sub_topics":[str],"difficulty_hint":str,"kg_query_terms":[str]}'
    )

    try:
        resp = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.raw_goal},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        parsed = json.loads(resp.choices[0].message.content)
    except Exception as exc:
        logger.error(f"GPT-4 goal parse error: {exc}")
        raw = request.raw_goal.lower().replace("i want to learn", "").replace("from scratch", "").strip()
        parsed = {
            "topic": raw.title(),
            "sub_topics": [
                f"{raw.title()} Fundamentals",
                f"{raw.title()} Core Concepts",
                f"{raw.title()} Intermediate Topics",
                f"{raw.title()} Advanced Applications",
            ],
            "difficulty_hint": "beginner" if "scratch" in request.raw_goal.lower() else "intermediate",
            "kg_query_terms": raw.split()[:6],
        }

    goal_output = {
        "goal_id": goal_id,
        "user_id": request.user_id,
        "raw_goal": request.raw_goal,
        "topic": parsed.get("topic", "Unknown"),
        "sub_topics": parsed.get("sub_topics", []),
        "difficulty_hint": parsed.get("difficulty_hint", "beginner"),
        "kg_query_terms": parsed.get("kg_query_terms", []),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    if redis_client:
        await redis_client.set(f"goal:{goal_id}", json.dumps(goal_output), ex=3600)

    logger.info(f"Goal parsed: {goal_id} topic='{goal_output['topic']}'")
    return goal_output


@router.post("/harvest")
async def harvest_content(
    request: HarvestRequest,
    redis_client=Depends(get_redis),
):
    """
    Trigger web + YouTube content harvesting for a parsed goal.
    Returns list of HarvestedDocs cached in Redis.
    """
    from backend.config import get_settings
    from backend.services.content_harvester import harvest
    settings = get_settings()

    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available.")

    goal_data = await redis_client.get(f"goal:{request.goal_id}")
    if not goal_data:
        raise HTTPException(status_code=404, detail="Goal not found. Run /goal/parse first.")

    goal = json.loads(goal_data)

    # Check cache
    cached = await redis_client.get(f"harvest:{request.goal_id}")
    if cached:
        docs = json.loads(cached)
        return {"goal_id": request.goal_id, "n_docs": len(docs), "docs": docs, "cached": True}

    docs_objs = await harvest(
        goal_id=request.goal_id,
        sub_topics=goal.get("sub_topics", []),
        tavily_api_key=settings.tavily_api_key,
        youtube_api_key=settings.youtube_api_key,
    )
    docs = [d.dict() for d in docs_objs]
    await redis_client.set(f"harvest:{request.goal_id}", json.dumps(docs), ex=3600)

    return {"goal_id": request.goal_id, "n_docs": len(docs), "docs": docs}
