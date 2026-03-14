from fastapi import APIRouter, HTTPException

from app.graph.learning_graph import get_graph
from app.models.schemas import ContentResponse

router = APIRouter()


@router.post("/{session_id}/generate-content", response_model=ContentResponse)
async def generate_content(session_id: str):
    """Generate learning content for the current module."""
    config = {"configurable": {"thread_id": session_id}}
    graph = get_graph()

    snapshot = await graph.aget_state(config)
    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail="Session not found")

    # Resume graph from current interrupt point through generate_content
    state = await graph.ainvoke(None, config)

    idx = state["current_module_idx"]
    return ContentResponse(
        module_idx=idx,
        title=state["roadmap"][idx]["title"],
        content=state["module_content"],
    )
