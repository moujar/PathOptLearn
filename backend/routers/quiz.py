"""Stage 2 — Adaptive Quiz Generation using LangChain RAG + GPT-4."""
import asyncio
import json
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/quiz", tags=["Quiz Generation"])
logger = logging.getLogger(__name__)


class QuizGenerateRequest(BaseModel):
    goal_id: str
    n_questions: int = 12
    bloom_distribution: dict = {
        "remember": 2, "understand": 4, "apply": 4, "analyze": 2
    }


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


_BLOOM_DESC = {
    "remember": "recall definitions, facts, basic terminology",
    "understand": "explain concepts in own words, summarise, classify",
    "apply": "use knowledge to solve problems, demonstrate procedures",
    "analyze": "break down information, compare, distinguish, examine relationships",
}


@router.post("/generate")
async def generate_quiz(
    request: QuizGenerateRequest,
    redis_client=Depends(get_redis),
    openai_client=Depends(get_openai),
):
    """
    Generate an adaptive quiz from harvested content via RAG + GPT-4.
    Items are Bloom-tagged and IRT-calibrated using Deep-IRT.
    """
    from backend.config import get_settings
    from backend.services.rag_engine import get_rag_engine
    from backend.models.deep_irt import DeepIRT

    settings = get_settings()
    quiz_id = str(uuid.uuid4())[:12]

    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available.")

    docs_raw = await redis_client.get(f"harvest:{request.goal_id}")
    if not docs_raw:
        raise HTTPException(
            status_code=404,
            detail="No harvested content. Run /goal/harvest first.",
        )
    docs = json.loads(docs_raw)

    # Build or reuse FAISS index
    rag = get_rag_engine(request.goal_id, settings.embed_model_name)
    if not rag._index:
        await asyncio.to_thread(rag.build_index, docs)

    # Deep-IRT calibrator
    deep_irt = DeepIRT.load_pretrained(
        settings.drl_model_path.replace("drl_ppo.zip", "deep_irt.pt")
    )

    all_items: list[dict] = []
    item_counter = 0

    for bloom_level, n_items in request.bloom_distribution.items():
        if n_items <= 0:
            continue
        level_desc = _BLOOM_DESC.get(bloom_level, bloom_level)
        chunks = await rag.retrieve_async(f"{bloom_level}: {level_desc}", k=5)
        context = "\n\n".join(c.get("text", "")[:800] for c in chunks)

        system_prompt = (
            f"You are an expert quiz designer. Generate {n_items} multiple-choice questions "
            f"at Bloom's taxonomy level '{bloom_level}' ({level_desc}). "
            "Each question must be answerable from the provided context only, "
            "have exactly 4 options with one correct answer (0-indexed), "
            "include the source concept name, and an estimated difficulty from -3 to +3. "
            'Return ONLY JSON: {"questions":[{"question":str,"options":[str,str,str,str],'
            '"correct_index":int,"concept":str,"difficulty_hint":float}]}'
        )
        user_prompt = f"Context:\n{context}\n\nGenerate {n_items} {bloom_level}-level questions."

        items_raw: list[dict] = []
        try:
            resp = await asyncio.to_thread(
                openai_client.chat.completions.create,
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.5,
            )
            raw = json.loads(resp.choices[0].message.content)
            items_raw = raw.get("questions", raw.get("items", []))
            if isinstance(items_raw, dict):
                items_raw = list(items_raw.values())[0]
        except Exception as exc:
            logger.error(f"GPT-4 quiz generation failed (bloom={bloom_level}): {exc}")

        for item_data in items_raw[:n_items]:
            item_counter += 1
            item_id = f"q{item_counter:03d}_{quiz_id}"

            try:
                a, b, c = deep_irt.calibrate_item(item_data.get("question", ""))
            except Exception:
                b = float(item_data.get("difficulty_hint", 0.0))
                a, c = 1.0, 0.25

            all_items.append(
                {
                    "item_id": item_id,
                    "question": item_data.get("question", ""),
                    "options": item_data.get("options", ["A", "B", "C", "D"]),
                    "correct_index": int(item_data.get("correct_index", 0)),
                    "difficulty_b": round(b, 3),
                    "discrimination_a": round(a, 3),
                    "guessing_c": round(c, 3),
                    "bloom_level": bloom_level,
                    "concept": item_data.get("concept", "general"),
                    "source_url": chunks[0].get("url", "") if chunks else "",
                }
            )

    quiz = {
        "quiz_id": quiz_id,
        "goal_id": request.goal_id,
        "items": all_items,
        "estimated_duration_min": len(all_items) * 2,
    }

    await redis_client.set(f"quiz:{quiz_id}", json.dumps(quiz), ex=3600)
    logger.info(f"Quiz {quiz_id} created: {len(all_items)} items for goal {request.goal_id}")
    return quiz
