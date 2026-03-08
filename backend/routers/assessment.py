"""Stage 3 — CAT-IRT Knowledge State Assessment."""
import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/assessment", tags=["Knowledge Assessment"])
logger = logging.getLogger(__name__)


class AssessmentRunRequest(BaseModel):
    student_id: str
    quiz_id: str
    responses: list[dict]  # [{item_id, answer_index, response_time_ms}]


class NextItemRequest(BaseModel):
    student_id: str
    quiz_id: str
    current_responses: list[dict] = []


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


@router.post("/run")
async def run_assessment(
    request: AssessmentRunRequest,
    redis_client=Depends(get_redis),
):
    """
    Batch CAT-IRT assessment: score all responses and return KnowledgeVector.
    Stores result in Redis with TTL 86400s.
    """
    from backend.services.cat_engine import run_cat_batch

    quiz_raw = await redis_client.get(f"quiz:{request.quiz_id}") if redis_client else None
    if not quiz_raw:
        raise HTTPException(status_code=404, detail=f"Quiz {request.quiz_id} not found.")

    quiz = json.loads(quiz_raw)
    quiz_items = quiz.get("items", [])
    items_map = {item["item_id"]: item for item in quiz_items}

    # Enrich responses with correct flag
    enriched = [
        {**r, "correct": r.get("answer_index") == items_map[r["item_id"]].get("correct_index")}
        for r in request.responses
        if r.get("item_id") in items_map
    ]

    result = run_cat_batch(quiz_items, enriched)
    knowledge_vector = {
        "student_id": request.student_id,
        "theta": result["theta"],
        "concept_mastery": result["concept_mastery"],
        "confidence_interval": result["confidence_interval"],
        "assessed_at": result["assessed_at"],
        "last_studied_at": {},
        "quiz_id": request.quiz_id,
    }

    if redis_client:
        await redis_client.set(f"ks:{request.student_id}", json.dumps(knowledge_vector), ex=86400)

    logger.info(f"Assessment: student={request.student_id} theta={result['theta']}")
    return knowledge_vector


@router.post("/next-item")
async def get_next_item(
    request: NextItemRequest,
    redis_client=Depends(get_redis),
):
    """
    Real-time adaptive quiz: return the next item using Fisher Information maximisation.
    """
    from backend.services.cat_engine import get_next_item_response

    quiz_raw = await redis_client.get(f"quiz:{request.quiz_id}") if redis_client else None
    if not quiz_raw:
        raise HTTPException(status_code=404, detail=f"Quiz {request.quiz_id} not found.")

    quiz = json.loads(quiz_raw)
    quiz_items = quiz.get("items", [])
    items_map = {item["item_id"]: item for item in quiz_items}

    enriched = [
        {**r, "correct": r.get("answer_index") == items_map[r["item_id"]].get("correct_index")}
        for r in request.current_responses
        if r.get("item_id") in items_map
    ]

    return get_next_item_response(quiz_items, enriched)
