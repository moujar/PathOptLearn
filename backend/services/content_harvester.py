"""Content harvesting service — Tavily Search API + YouTube Data API v3."""
import asyncio
import hashlib
import json
import logging
from dataclasses import asdict, dataclass

import httpx

logger = logging.getLogger(__name__)


@dataclass
class HarvestResult:
    doc_id: str
    url: str
    title: str
    content_text: str
    source_type: str  # 'web' | 'youtube'
    goal_id: str

    def dict(self) -> dict:
        return asdict(self)


# ── Web harvesting ─────────────────────────────────────────────────────────

async def harvest_web(
    sub_topic: str, goal_id: str, tavily_api_key: str, n_results: int = 5
) -> list[HarvestResult]:
    """Fetch top web pages for a sub-topic via Tavily Search API."""
    results: list[HarvestResult] = []
    if not tavily_api_key:
        logger.warning("TAVILY_API_KEY not set; skipping web harvest.")
        return results

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": tavily_api_key,
                    "query": sub_topic,
                    "max_results": n_results,
                    "include_raw_content": True,
                },
            )
            resp.raise_for_status()
            for item in resp.json().get("results", []):
                doc_id = hashlib.md5(item.get("url", "").encode()).hexdigest()[:12]
                content = item.get("raw_content") or item.get("content", "")
                results.append(
                    HarvestResult(
                        doc_id=doc_id,
                        url=item.get("url", ""),
                        title=item.get("title", sub_topic),
                        content_text=content[:8000],
                        source_type="web",
                        goal_id=goal_id,
                    )
                )
    except Exception as exc:
        logger.warning(f"Web harvest failed for '{sub_topic}': {exc}")

    return results


# ── YouTube harvesting ─────────────────────────────────────────────────────

async def _get_transcript(video_id: str) -> str:
    """Fetch YouTube transcript via youtube-transcript-api (runs in thread)."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        loop = asyncio.get_event_loop()
        transcript_list = await loop.run_in_executor(
            None,
            lambda: YouTubeTranscriptApi.get_transcript(video_id, languages=["en"]),
        )
        return " ".join(t["text"] for t in transcript_list)
    except Exception as exc:
        logger.warning(f"Transcript unavailable for {video_id}: {exc}")
        return f"[Transcript unavailable for video {video_id}]"


async def harvest_youtube(
    sub_topic: str, goal_id: str, youtube_api_key: str, n_results: int = 3
) -> list[HarvestResult]:
    """Fetch top YouTube videos + transcripts for a sub-topic."""
    results: list[HarvestResult] = []
    if not youtube_api_key:
        logger.warning("YOUTUBE_API_KEY not set; skipping YouTube harvest.")
        return results

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                "https://www.googleapis.com/youtube/v3/search",
                params={
                    "key": youtube_api_key,
                    "q": sub_topic,
                    "part": "snippet",
                    "type": "video",
                    "maxResults": n_results,
                    "relevanceLanguage": "en",
                },
            )
            resp.raise_for_status()
            items = resp.json().get("items", [])

        for item in items:
            video_id = item["id"].get("videoId", "")
            if not video_id:
                continue
            title = item["snippet"]["title"]
            url = f"https://www.youtube.com/watch?v={video_id}"
            transcript = await _get_transcript(video_id)
            doc_id = hashlib.md5(url.encode()).hexdigest()[:12]
            results.append(
                HarvestResult(
                    doc_id=doc_id,
                    url=url,
                    title=title,
                    content_text=transcript[:8000],
                    source_type="youtube",
                    goal_id=goal_id,
                )
            )
    except Exception as exc:
        logger.warning(f"YouTube harvest failed for '{sub_topic}': {exc}")

    return results


# ── Main harvest entry point ───────────────────────────────────────────────

async def harvest(
    goal_id: str,
    sub_topics: list[str],
    tavily_api_key: str,
    youtube_api_key: str,
) -> list[HarvestResult]:
    """
    Harvest web + YouTube content for all sub-topics of a goal in parallel.
    Returns combined de-duplicated list of HarvestResult objects.
    """
    tasks: list = []
    for topic in sub_topics:
        tasks.append(harvest_web(topic, goal_id, tavily_api_key))
        tasks.append(harvest_youtube(topic, goal_id, youtube_api_key))

    gathered = await asyncio.gather(*tasks, return_exceptions=True)

    seen_urls: set[str] = set()
    docs: list[HarvestResult] = []
    for batch in gathered:
        if isinstance(batch, list):
            for doc in batch:
                if doc.url not in seen_urls:
                    seen_urls.add(doc.url)
                    docs.append(doc)

    logger.info(f"Harvested {len(docs)} unique documents for goal {goal_id}")
    return docs
