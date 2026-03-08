"""Stage 7 — LLM Explanation of Recommended Learning Path."""
import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/explain", tags=["Explanation"])
logger = logging.getLogger(__name__)


class ExplainRequest(BaseModel):
    student_id: str
    path_id: str
    explanation_type: str = "full"  # full | brief | motivational


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
    return OpenAI(api_key=get_settings().openai_api_key)


@router.post("/path")
async def explain_path(
    request: ExplainRequest,
    redis_client=Depends(get_redis),
    openai_client=Depends(get_openai),
):
    """
    Generate a RAG-augmented GPT-4 explanation for a recommended learning path.
    Explains WHY the path was chosen based on the student's knowledge state.
    """
    from backend.config import get_settings
    from backend.services.rag_engine import get_rag_engine
    from backend.services.llm_explainer import explain_path as _explain

    settings = get_settings()

    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available.")

    # Load data
    ks_raw = await redis_client.get(f"ks:{request.student_id}")
    if not ks_raw:
        raise HTTPException(status_code=404, detail="Knowledge state not found.")

    path_raw = await redis_client.get(f"path:{request.path_id}")
    if not path_raw:
        raise HTTPException(status_code=404, detail=f"Path {request.path_id} not found.")

    ks = json.loads(ks_raw)
    path = json.loads(path_raw)
    goal_id = path.get("goal_id", "")

    # Retrieve top RAG chunks for weakest concept
    mastery = ks.get("concept_mastery", {})
    weak_concept = min(mastery.items(), key=lambda x: x[1])[0] if mastery else "core concepts"

    rag = get_rag_engine(goal_id, settings.embed_model_name)
    rag_chunks = await rag.retrieve_async(weak_concept, k=3)

    result = await _explain(
        knowledge_vector=ks,
        learning_path=path,
        rag_chunks=rag_chunks,
        openai_client=openai_client,
        explanation_type=request.explanation_type,
        model=settings.openai_model,
    )

    return result
