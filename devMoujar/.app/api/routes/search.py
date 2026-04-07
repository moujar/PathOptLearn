from fastapi import APIRouter
from app.api.models import (
    QuickSearchRequest, QuickSearchResponse,
    YoutubeSearchRequest, YoutubeSearchResponse,
)
from app.core.llm import llm
from app.core.search import quick_search_summary
from app.core.youtube import search_youtube

router = APIRouter(prefix="/api/search", tags=["search"])


@router.post("/quick", response_model=QuickSearchResponse)
def quick_search(req: QuickSearchRequest):
    results, summary = quick_search_summary(req.topic, llm)
    return QuickSearchResponse(topic=req.topic, summary=summary, results=results)


@router.post("/youtube", response_model=YoutubeSearchResponse)
def youtube_search(req: YoutubeSearchRequest):
    videos = search_youtube(req.query, max_results=req.max_results)
    return YoutubeSearchResponse(query=req.query, videos=videos)
