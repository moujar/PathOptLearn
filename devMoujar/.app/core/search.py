import re
import time

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

from app.config import MAX_SEARCH_RESULTS, MAX_CHUNK_CHARS, REQUEST_TIMEOUT


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
    headers = {"User-Agent": "Mozilla/5.0 (compatible; DeepSearch/1.0)"}
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
