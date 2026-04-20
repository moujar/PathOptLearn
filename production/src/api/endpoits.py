"""
Search and YouTube helper functions used by the LearnFlow graph and UI pages.
"""
import re
import time
import sys
import os

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
import yt_dlp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import MAX_SEARCH_RESULTS, MAX_CHUNK_CHARS, REQUEST_TIMEOUT, YT_MAX_RESULTS


# ── Web Search ───────────────────────────────────────────────────────────────

def search_web(query: str, max_results: int = MAX_SEARCH_RESULTS) -> list[dict]:
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title":   r.get("title", ""),
                    "url":     r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
    except Exception as e:
        print(f"[Search error] {e}")
    return results


def fetch_page(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; PathOptLearn/1.0)"}
    try:
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header",
                         "aside", "form", "iframe", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return re.sub(r"\s+", " ", text)
    except Exception:
        return ""


def chunk_text(text: str, chunk_size: int = MAX_CHUNK_CHARS) -> list[str]:
    chunks = []
    step = int(chunk_size * 0.8)
    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size]
        if len(chunk) > 200:
            chunks.append(chunk)
    return chunks


def quick_search_summary(topic: str, llm_fn) -> tuple[list[dict], str]:
    """Return (results, summary_text)."""
    results = search_web(topic, max_results=10)
    snippets = "\n\n".join(
        f"**{r['title']}** ({r['url']})\n{r['snippet']}"
        for r in results if r["snippet"]
    )
    system = "You are a research assistant. Summarize search results into a clear, structured answer."
    prompt = f"Topic: {topic}\n\nSearch results:\n{snippets}\n\nWrite a comprehensive summary:"
    summary = llm_fn(prompt, system=system)
    return results, summary


def edu_resources(query: str) -> list[dict]:
    """Search educational sites only."""
    seen, out = set(), []
    filters = [
        "site:coursera.org OR site:edx.org OR site:khanacademy.org",
        "site:mit.edu OR site:stanford.edu OR site:ocw.mit.edu",
    ]
    for f in filters:
        for r in search_web(f"{query} {f}", max_results=3):
            if r["url"] not in seen:
                seen.add(r["url"])
                out.append(r)
        time.sleep(0.2)
    if not out:
        out = search_web(f"{query} course lecture tutorial", max_results=5)
    return out[:6]


# ── YouTube ──────────────────────────────────────────────────────────────────

def search_youtube(query: str, max_results: int = YT_MAX_RESULTS) -> list[dict]:
    ydl_opts = {"quiet": True, "no_warnings": True, "extract_flat": True}
    videos = []
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"ytsearch{max_results}:{query}", download=False)
            for entry in info.get("entries", []):
                d      = int(entry.get("duration") or 0)
                vid_id = entry.get("id", "")
                videos.append({
                    "id":       vid_id,
                    "title":    entry.get("title", ""),
                    "channel":  entry.get("channel") or entry.get("uploader", ""),
                    "duration": f"{d // 60}:{d % 60:02d}",
                    "views":    entry.get("view_count"),
                    "desc":     (entry.get("description") or "")[:200],
                    "url":      f"https://www.youtube.com/watch?v={vid_id}",
                    "thumb":    f"https://img.youtube.com/vi/{vid_id}/hqdefault.jpg",
                })
    except Exception as e:
        print(f"[YouTube error] {e}")
    return videos
