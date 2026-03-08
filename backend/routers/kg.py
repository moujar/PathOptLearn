"""Stage 4 — Dynamic Knowledge Graph Construction."""
import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/kg", tags=["Knowledge Graph"])
logger = logging.getLogger(__name__)


class KGBuildRequest(BaseModel):
    goal_id: str
    student_id: str


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


@router.post("/build")
async def build_kg(
    request: KGBuildRequest,
    redis_client=Depends(get_redis),
    openai_client=Depends(get_openai),
):
    """
    Build a dynamic knowledge graph from harvested content.
    Uses GraphRAG entity extraction + GAT embeddings + Neo4j storage.
    """
    from backend.config import get_settings
    from backend.services.kg_builder import build_knowledge_graph
    from sentence_transformers import SentenceTransformer

    settings = get_settings()

    docs_raw = await redis_client.get(f"harvest:{request.goal_id}") if redis_client else None
    if not docs_raw:
        raise HTTPException(
            status_code=404, detail="No harvested content. Run /goal/harvest first."
        )
    docs = json.loads(docs_raw)

    # Return cached KG if available
    if redis_client:
        cached = await redis_client.get(f"kg:{request.goal_id}")
        if cached:
            return {**json.loads(cached), "cached": True}

    embedder = SentenceTransformer(settings.embed_model_name)

    neo4j_driver = None
    try:
        from neo4j import AsyncGraphDatabase
        neo4j_driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
        )
    except Exception as exc:
        logger.warning(f"Neo4j not available: {exc}")

    kg = await build_knowledge_graph(
        goal_id=request.goal_id,
        docs=docs,
        openai_client=openai_client,
        embedder=embedder,
        neo4j_driver=neo4j_driver,
    )

    response_data = {
        "kg_id": kg["kg_id"],
        "goal_id": request.goal_id,
        "student_id": request.student_id,
        "n_nodes": len(kg["nodes"]),
        "n_edges": len(kg["edges"]),
        "nodes": kg["nodes"],
        "edges": kg["edges"],
        "built_at": datetime.now(timezone.utc).isoformat(),
    }

    if redis_client:
        await redis_client.set(f"kg:{request.goal_id}", json.dumps(response_data), ex=3600)
        await redis_client.set(f"kg_id:{kg['kg_id']}", json.dumps(response_data), ex=3600)

    if neo4j_driver:
        await neo4j_driver.close()

    logger.info(f"KG built: {kg['kg_id']} nodes={len(kg['nodes'])} edges={len(kg['edges'])}")
    return response_data


@router.get("/{kg_id}/visualize")
async def visualize_kg(
    kg_id: str,
    redis_client=Depends(get_redis),
):
    """Return KG data formatted for pyvis visualisation in Streamlit."""
    kg_raw = await redis_client.get(f"kg_id:{kg_id}") if redis_client else None
    if not kg_raw:
        raise HTTPException(status_code=404, detail=f"KG {kg_id} not found.")

    kg = json.loads(kg_raw)

    vis_nodes = [
        {
            "id": n["node_id"],
            "label": n["concept"],
            "title": n.get("definition", n["concept"]),
            "value": n["difficulty"],
            "color": _diff_color(n["difficulty"]),
        }
        for n in kg.get("nodes", [])
    ]
    vis_edges = [
        {
            "from": e["from_node"],
            "to": e["to_node"],
            "label": e["relation"],
            "arrows": "to" if e["relation"] == "prerequisite" else "",
            "dashes": e["relation"] == "semantic",
            "width": round(e["weight"] * 3, 1),
        }
        for e in kg.get("edges", [])
    ]
    return {"kg_id": kg_id, "nodes": vis_nodes, "edges": vis_edges}


def _diff_color(difficulty: float) -> str:
    if difficulty < 0.33:
        return "#2ecc71"
    elif difficulty < 0.66:
        return "#f39c12"
    return "#e74c3c"
