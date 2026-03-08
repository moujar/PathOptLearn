"""Stage 6 ★ — Adaptive Learning Loop (Continual KT + Ebbinghaus Forgetting)."""
import json
import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/loop", tags=["Learning Loop"])
logger = logging.getLogger(__name__)


class LoopUpdateRequest(BaseModel):
    student_id: str
    path_id: str
    session_id: str
    session_responses: list[dict]  # [{item_id, answer_index, response_time_ms, correct}]
    time_since_last_session_hours: float = 24.0


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


@router.post("/update")
async def update_loop(
    request: LoopUpdateRequest,
    redis_client=Depends(get_redis),
):
    """
    Post-session update:
    1. Apply Ebbinghaus forgetting decay to all concepts
    2. Update concept mastery from session responses (AKT-style)
    3. Re-optimise learning path for remaining concepts
    Returns updated KnowledgeVector + new LearningPath + session metrics.
    """
    from backend.config import get_settings
    from backend.models.forgetting import forgetting_module
    from backend.services.cat_engine import run_cat_batch
    from backend.models.drl_agent import DRLAgent, LearningEnv

    settings = get_settings()

    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available.")

    # Load current knowledge state
    ks_raw = await redis_client.get(f"ks:{request.student_id}")
    if not ks_raw:
        raise HTTPException(status_code=404, detail="Knowledge state not found. Run /assessment/run first.")
    ks = json.loads(ks_raw)

    # Load current path
    path_raw = await redis_client.get(f"path:{request.path_id}")
    if not path_raw:
        raise HTTPException(status_code=404, detail=f"Path {request.path_id} not found.")
    current_path = json.loads(path_raw)

    concept_mastery: dict[str, float] = ks.get("concept_mastery", {})
    last_studied_at: dict[str, str] = ks.get("last_studied_at", {})

    # ── 1. Apply Ebbinghaus forgetting ────────────────────────────────
    now_iso = datetime.now(timezone.utc).isoformat()
    updated_mastery, forgetting_deltas = forgetting_module.apply_forgetting_to_vector(
        concept_mastery=concept_mastery,
        last_studied_at=last_studied_at,
        current_time_iso=now_iso,
        global_t_hours=request.time_since_last_session_hours,
    )

    # ── 2. Update mastery from session responses ───────────────────────
    # Simple heuristic: for each correct response, boost mastery for that concept
    session_concept_gains: dict[str, float] = {}
    items_in_session: dict[str, str] = {}  # item_id -> concept

    # Try to load quiz from path context
    goal_id = current_path.get("goal_id", "")
    quiz_items: list[dict] = []
    # Look for any quiz associated with the goal
    if redis_client:
        # Scan isn't ideal but workable for demo
        pass  # quiz items resolved from session_responses concept field

    for resp in request.session_responses:
        concept = resp.get("concept", "unknown")
        correct = resp.get("correct", False)
        if correct:
            gain = 0.15 * (1.0 - updated_mastery.get(concept, 0.0))
            session_concept_gains[concept] = session_concept_gains.get(concept, 0.0) + gain
            updated_mastery[concept] = min(1.0, updated_mastery.get(concept, 0.0) + gain)
        else:
            # Wrong answer slightly reduces mastery
            updated_mastery[concept] = max(0.0, updated_mastery.get(concept, 0.0) - 0.02)

        last_studied_at[concept] = now_iso
        forgetting_module.update_stability(concept, updated_mastery.get(concept, 0.0))

    # Recompute overall theta
    avg_mastery = sum(updated_mastery.values()) / max(len(updated_mastery), 1)
    import math
    new_theta = math.log(max(avg_mastery, 1e-6) / max(1.0 - avg_mastery, 1e-6))

    # ── 3. Update KnowledgeVector in Redis ────────────────────────────
    updated_ks = {
        **ks,
        "theta": round(new_theta, 4),
        "concept_mastery": {k: round(v, 4) for k, v in updated_mastery.items()},
        "confidence_interval": [round(new_theta - 0.4, 4), round(new_theta + 0.4, 4)],
        "assessed_at": now_iso,
        "last_studied_at": last_studied_at,
    }

    if redis_client:
        await redis_client.set(f"ks:{request.student_id}", json.dumps(updated_ks), ex=86400)

    # ── 4. Path re-optimisation ────────────────────────────────────────
    steps = current_path.get("steps", [])
    completed_concepts = {
        s["concept"] for s in steps if updated_mastery.get(s["concept"], 0) >= 0.85
    }

    # Load KG for re-optimisation
    kg_raw = None
    if redis_client:
        kg_raw = await redis_client.get(f"kg:{goal_id}")
    docs_raw = await redis_client.get(f"harvest:{goal_id}") if redis_client else None
    docs = json.loads(docs_raw) if docs_raw else []

    path_changes: list[str] = []
    new_path = current_path

    if kg_raw:
        kg = json.loads(kg_raw)
        nodes = kg.get("nodes", [])
        edges = kg.get("edges", [])
        concept_list = [n["concept"] for n in nodes]
        prereq_map: dict[str, list[str]] = {}
        for e in edges:
            if e.get("relation") == "prerequisite":
                to_n = next((n["concept"] for n in nodes if n["node_id"] == e["to_node"]), None)
                from_n = next((n["concept"] for n in nodes if n["node_id"] == e["from_node"]), None)
                if to_n and from_n:
                    prereq_map.setdefault(to_n, []).append(from_n)

        env = LearningEnv(
            concept_list=concept_list,
            initial_mastery=updated_mastery,
            prerequisite_graph=prereq_map,
            concept_difficulty={n["concept"]: n["difficulty"] for n in nodes},
        )
        agent = DRLAgent(model_path=settings.drl_model_path)
        path_actions = agent.generate_path(env)

        from backend.routers.recommend import _build_learning_path
        new_path = _build_learning_path(
            path_actions, nodes, docs,
            request.student_id, goal_id, "DRL-PPO"
        )

        # Identify changes
        old_concepts = [s["concept"] for s in steps]
        new_concepts = [s["concept"] for s in new_path["steps"]]
        for c in new_concepts:
            if c not in old_concepts:
                path_changes.append(f"added: {c}")
        for c in old_concepts:
            if c not in new_concepts:
                path_changes.append(f"removed: {c}")
        for c in completed_concepts:
            path_changes.append(f"completed: {c}")

        if redis_client:
            await redis_client.set(f"path:{new_path['path_id']}", json.dumps(new_path), ex=3600)

    # ── 5. Session metrics ─────────────────────────────────────────────
    total_gain = sum(session_concept_gains.values())
    n_items = len(request.session_responses)
    time_min = n_items * 2.5  # ~2.5 min per item
    efficiency = total_gain / max(time_min, 1)

    all_concepts = list(updated_mastery.keys())
    mastered = sum(1 for v in updated_mastery.values() if v >= 0.85)
    progress_pct = 100.0 * mastered / max(len(all_concepts), 1)
    sessions_remaining = max(1, int((len(all_concepts) - mastered) / max(mastered + 1, 1)))

    # Store session history
    session_record = {
        "session_id": request.session_id,
        "timestamp": now_iso,
        "mastery_before": concept_mastery,
        "mastery_after": updated_mastery,
        "forgetting_delta": forgetting_deltas,
        "learning_delta": session_concept_gains,
        "efficiency_score": round(efficiency, 4),
        "n_responses": n_items,
    }
    if redis_client:
        history_key = f"history:{request.student_id}"
        raw_history = await redis_client.get(history_key)
        history = json.loads(raw_history) if raw_history else []
        history.append(session_record)
        await redis_client.set(history_key, json.dumps(history), ex=86400 * 30)

    return {
        "updated_knowledge_vector": updated_ks,
        "forgetting_applied": {k: round(v, 4) for k, v in forgetting_deltas.items() if v != 0},
        "new_path_recommendation": new_path,
        "path_changes": path_changes,
        "session_learning_efficiency": round(efficiency, 4),
        "cumulative_progress_pct": round(progress_pct, 2),
        "estimated_sessions_remaining": sessions_remaining,
    }


@router.get("/history/{student_id}")
async def get_history(
    student_id: str,
    redis_client=Depends(get_redis),
):
    """Return all learning sessions for a student (for dashboard + thesis plots)."""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available.")

    raw = await redis_client.get(f"history:{student_id}")
    if not raw:
        return {"student_id": student_id, "sessions": []}

    return {"student_id": student_id, "sessions": json.loads(raw)}
