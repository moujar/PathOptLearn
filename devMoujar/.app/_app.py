import json
import re
import time
from typing import Optional

import ollama
import psycopg2
import requests
import yt_dlp
from bs4 import BeautifulSoup
from ddgs import DDGS
from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from neo4j import GraphDatabase
from psycopg2.extras import RealDictCursor
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="DeepSearch API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ────────────────────────────────────────────────────────────────────

MODEL            = "llama3.2:1b"
MAX_SEARCH_RESULTS = 6
MAX_SUB_QUERIES  = 3
MAX_CHUNK_CHARS  = 1500
TOP_CHUNKS       = 8
REQUEST_TIMEOUT  = 10
YT_MAX_RESULTS   = 6

DB = dict(
    host="localhost", port=5432,
    dbname="deepsearch", user="deepsearch", password="deepsearch",
)

NEO4J_URI  = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "deepsearch"

# ── DB helpers ────────────────────────────────────────────────────────────────

def get_conn():
    return psycopg2.connect(**DB)

def init_db():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS searches (
                    id          SERIAL PRIMARY KEY,
                    topic       TEXT NOT NULL,
                    search_type TEXT NOT NULL,
                    created_at  TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE TABLE IF NOT EXISTS web_results (
                    id        SERIAL PRIMARY KEY,
                    search_id INT REFERENCES searches(id) ON DELETE CASCADE,
                    title     TEXT, url TEXT, snippet TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE TABLE IF NOT EXISTS youtube_results (
                    id          SERIAL PRIMARY KEY,
                    search_id   INT REFERENCES searches(id) ON DELETE CASCADE,
                    video_id    TEXT, title TEXT, channel TEXT,
                    duration    TEXT, views BIGINT, description TEXT,
                    url TEXT, thumbnail TEXT,
                    created_at  TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE TABLE IF NOT EXISTS reports (
                    id        SERIAL PRIMARY KEY,
                    search_id INT REFERENCES searches(id) ON DELETE CASCADE,
                    content   TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
        conn.commit()

def _new_search(topic: str, kind: str) -> int:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO searches (topic, search_type) VALUES (%s,%s) RETURNING id",
                (topic, kind),
            )
            sid = cur.fetchone()[0]
        conn.commit()
    return sid

def _save_web(sid: int, results: list):
    if not results:
        return
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO web_results (search_id,title,url,snippet) VALUES (%s,%s,%s,%s)",
                [(sid, r["title"], r["url"], r["snippet"]) for r in results],
            )
        conn.commit()

def _save_yt(sid: int, videos: list):
    if not videos:
        return
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO youtube_results
                   (search_id,video_id,title,channel,duration,views,description,url,thumbnail)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                [(sid, v["id"], v["title"], v["channel"], v["duration"],
                  v["views"], v["desc"], v["url"], v["thumb"]) for v in videos],
            )
        conn.commit()

def _save_report(sid: int, text: str):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO reports (search_id,content) VALUES (%s,%s)", (sid, text)
            )
        conn.commit()

# ── Neo4j helpers ─────────────────────────────────────────────────────────────

def get_neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

def init_graph_schema(driver):
    with driver.session() as s:
        s.run("CREATE CONSTRAINT topic_unique   IF NOT EXISTS FOR (t:Topic)   REQUIRE t.name IS UNIQUE")
        s.run("CREATE CONSTRAINT module_unique  IF NOT EXISTS FOR (m:Module)  REQUIRE m.uid  IS UNIQUE")
        s.run("CREATE CONSTRAINT concept_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE")
        s.run("CREATE CONSTRAINT level_unique   IF NOT EXISTS FOR (l:Level)   REQUIRE l.name IS UNIQUE")

def save_roadmap_to_graph(driver, roadmap: dict):
    """Persist roadmap as a property graph (Topic · Level · Module · Concept)."""
    topic = roadmap["topic"]
    level = roadmap["level"]
    with driver.session() as s:
        s.run(
            "MERGE (t:Topic {name:$n}) SET t.level=$lv, t.emoji=$em",
            n=topic, lv=level, em=roadmap["level_emoji"],
        )
        s.run("MERGE (l:Level {name:$n})", n=level)
        s.run(
            "MATCH (t:Topic{name:$t}),(l:Level{name:$l}) MERGE (t)-[:TARGET_LEVEL]->(l)",
            t=topic, l=level,
        )
        for mod in roadmap["modules"]:
            uid = f"{topic}::{mod['id']}"
            s.run(
                """MERGE (m:Module {uid:$uid})
                   SET m.title=$title, m.objective=$obj,
                       m.duration_minutes=$dur, m.order=$n, m.topic=$topic""",
                uid=uid, title=mod["title"], obj=mod["objective"],
                dur=mod["duration_minutes"], n=mod["id"], topic=topic,
            )
            s.run(
                "MATCH (t:Topic{name:$t}),(m:Module{uid:$u}) MERGE (t)-[:HAS_MODULE]->(m)",
                t=topic, u=uid,
            )
            for cname in mod.get("concepts", []):
                s.run("MERGE (c:Concept {name:$n})", n=cname)
                s.run(
                    "MATCH (m:Module{uid:$u}),(c:Concept{name:$n}) MERGE (m)-[:TEACHES]->(c)",
                    u=uid, n=cname,
                )
            for pid in mod.get("prerequisites", []):
                pre_uid = f"{topic}::{pid}"
                s.run(
                    "MATCH (pre:Module{uid:$p}),(m:Module{uid:$u}) MERGE (pre)-[:PREREQUISITE_FOR]->(m)",
                    p=pre_uid, u=uid,
                )

def _recommend_modules(driver, topic: str, completed_uids: list) -> list:
    """
    Cypher-based recommender: returns modules whose prerequisites are all
    satisfied, ranked by concepts_taught DESC then module order ASC.
    """
    with driver.session() as s:
        result = s.run(
            """
            MATCH (t:Topic {name:$topic})-[:HAS_MODULE]->(m:Module)
            WHERE NOT m.uid IN $done
              AND NOT EXISTS {
                  MATCH (pre:Module)-[:PREREQUISITE_FOR]->(m)
                  WHERE NOT pre.uid IN $done
              }
            OPTIONAL MATCH (m)-[:TEACHES]->(c:Concept)
            WITH m, count(c) AS concepts_taught
            RETURN m.uid               AS uid,
                   m.title             AS title,
                   m.objective         AS objective,
                   m.duration_minutes  AS duration,
                   m.order             AS module_order,
                   concepts_taught
            ORDER BY concepts_taught DESC, module_order ASC
            """,
            topic=topic,
            done=completed_uids,
        )
        return [dict(r) for r in result]

# ── Core utilities ────────────────────────────────────────────────────────────

def llm(prompt: str, system: str = "", stream: bool = False):
    """Call Ollama. Returns full string or a streaming generator."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    if stream:
        def _gen():
            for chunk in ollama.chat(model=MODEL, messages=messages, stream=True):
                yield chunk["message"]["content"]
        return _gen()

    result = ollama.chat(model=MODEL, messages=messages)
    return result["message"]["content"]

def llm_str(prompt: str, system: str = "") -> str:
    """Non-streaming LLM call, always returns a string."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return ollama.chat(model=MODEL, messages=messages)["message"]["content"]

def search_web(query: str, max_results: int = MAX_SEARCH_RESULTS) -> list:
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
        pass
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

def chunk_text(text: str, chunk_size: int = MAX_CHUNK_CHARS) -> list:
    chunks = []
    step = int(chunk_size * 0.8)
    for i in range(0, len(text), step):
        chunk = text[i : i + chunk_size]
        if len(chunk) > 200:
            chunks.append(chunk)
    return chunks

def search_youtube(query: str, max_results: int = YT_MAX_RESULTS) -> list:
    ydl_opts = {"quiet": True, "no_warnings": True, "extract_flat": True}
    videos = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(f"ytsearch{max_results}:{query}", download=False)
        for entry in info.get("entries", []):
            d = int(entry.get("duration") or 0)
            vid_id = entry.get("id", "")
            videos.append({
                "id":       vid_id,
                "title":    entry.get("title", ""),
                "channel":  entry.get("channel") or entry.get("uploader", ""),
                "duration": f"{d // 60}:{d % 60:02d}",
                "views":    entry.get("view_count"),
                "desc":     (entry.get("description") or "")[:250],
                "url":      f"https://www.youtube.com/watch?v={vid_id}",
                "thumb":    f"https://img.youtube.com/vi/{vid_id}/hqdefault.jpg",
            })
    return videos

# ── DeepSearch pipeline ───────────────────────────────────────────────────────

def to_search_query(topic: str) -> str:
    """
    Convert a natural-language question or sentence into a concise,
    keyword-optimised educational search query for DuckDuckGo.
    e.g. "i want to learn about physics electromagnetism"
         → "electromagnetism tutorial explained for beginners"
    """
    raw = llm_str(
        f'Convert the following into a short educational search query (max 8 words) '
        f'targeting learning resources like tutorials, courses, and explanations. '
        f'Return ONLY the query, no explanation, no quotes.\n\nInput: {topic}'
    )
    query = raw.strip().splitlines()[0].strip().strip('"').strip("'")
    return query or topic


# Trusted educational sources across all subjects
EDUCATIONAL_SITES = (
    # General
    "site:wikipedia.org OR site:britannica.com OR "
    # Maths & Sciences
    "site:khanacademy.org OR site:mathworld.wolfram.com OR site:brilliant.org OR "
    "site:physicsclassroom.com OR site:chemguide.co.uk OR site:chemlibre.libretexts.org OR "
    "site:math.libretexts.org OR site:physics.libretexts.org OR site:bio.libretexts.org OR "
    # University / MOOCs
    "site:mit.edu OR site:ocw.mit.edu OR site:coursera.org OR site:edx.org OR "
    "site:openlearn.open.ac.uk OR site:scholar.harvard.edu OR "
    # CS / Engineering
    "site:geeksforgeeks.org OR site:cs.stanford.edu OR "
    # Data science / AI
    "site:towardsdatascience.com OR site:deeplearning.ai OR "
    # Community / Blogs
    "site:medium.com OR site:stackexchange.com OR site:quora.com"
)

def generate_sub_queries(topic: str) -> list:
    clean = to_search_query(topic)
    prompt = (
        f'Generate {MAX_SUB_QUERIES} diverse educational search queries to find '
        f'tutorials, explanations, and learning resources about:\n'
        f'"{clean}"\n\n'
        f'Focus on: beginner guides, concept explanations, worked examples, and courses.\n'
        f'Return ONLY the queries, one per line, no numbering, no explanation.'
    )
    raw = llm_str(prompt)
    queries = [q.strip() for q in raw.strip().splitlines() if q.strip()][:MAX_SUB_QUERIES]
    if clean not in queries:
        queries.insert(0, clean)
    # Append a site-filtered query to always pull from trusted educational sources
    queries.append(f"{clean} {EDUCATIONAL_SITES}")
    return queries

def gather_sources(queries: list) -> list:
    seen, all_results = set(), []
    for q in queries:
        for r in search_web(q):
            if r["url"] and r["url"] not in seen:
                seen.add(r["url"])
                all_results.append(r)
        time.sleep(0.5)
    return all_results

def scrape_and_rank(sources: list, topic: str) -> list:
    scored = []
    for src in sources:
        page_text = fetch_page(src["url"]) or src.get("snippet", "")
        if not page_text:
            continue
        for chunk in chunk_text(page_text)[:4]:
            try:
                score_str = llm_str(
                    f'Rate 0-10 how useful this text is as educational content for learning "{topic}". '
                    f'Prefer explanations, definitions, examples, and tutorials over news or ads. '
                    f'Return ONLY a number.\nText: {chunk[:500]}'
                )
                score = float(re.search(r"\d+\.?\d*", score_str).group())
            except Exception:
                score = 5.0
            scored.append((score, chunk, src["title"], src["url"]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:TOP_CHUNKS]

def extract_learning_roadmap(topic: str, report: str) -> dict:
    prompt = f"""You are a curriculum designer.
Based on the research report about "{topic}", create a personalised learning roadmap.

Return ONLY valid JSON — no markdown fences, no extra text:
{{
  "topic": "{topic}",
  "level": "beginner",
  "level_emoji": "🟢",
  "gaps": ["gap 1", "gap 2"],
  "modules": [
    {{
      "id": 1,
      "title": "Module title",
      "objective": "Clear learning objective",
      "concepts": ["c1", "c2"],
      "duration_minutes": 60,
      "prerequisites": []
    }}
  ]
}}

Rules:
• level: "beginner" | "intermediate" | "advanced"
• level_emoji: 🟢 beginner · 🟡 intermediate · 🔴 advanced
• 4–7 modules, ordered foundational → advanced
• prerequisites: list of module ids that must be done first
• duration_minutes: 30 / 45 / 60 / 90 / 120
• gaps: 2–4 key knowledge areas this roadmap addresses

Report (excerpt):
{report[:3500]}"""

    raw = llm_str(prompt)
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {
        "topic": topic,
        "level": "beginner",
        "level_emoji": "🟢",
        "gaps": [f"fundamentals of {topic}"],
        "modules": [{
            "id": 1,
            "title": f"{topic}: Introduction",
            "objective": f"Understand the core concepts of {topic}",
            "concepts": ["overview", "fundamentals"],
            "duration_minutes": 60,
            "prerequisites": [],
        }],
    }

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def read_root():
    return {"status": "ok", "model": MODEL}


@app.get("/deepSearch")
def deep_search(
    topic: str = Query(..., description="Research topic", example="i want to learn about physics electromagnetism"),
):
    """
    Full deep-search pipeline: sub-queries → web scrape → rank → LLM synthesis.
    Streams the final report as plain text.
    """
    queries    = generate_sub_queries(topic)
    sources    = gather_sources(queries)
    top_chunks = scrape_and_rank(sources, topic)

    context = "\n\n".join(
        f"--- Source: {title} ({url}) ---\n{chunk}"
        for _, chunk, title, url in top_chunks
    )
    system = (
        "You are an expert educator capable of teaching any subject — "
        "mathematics, physics, chemistry, biology, computer science, history, economics, and more. "
        "Your goal is to make the topic clear and accessible. "
        "Write a structured educational lesson in markdown following this layout:\n"
        "1. **Introduction** — what this topic is and why it matters\n"
        "2. **Key Concepts** — core ideas explained simply, with analogies\n"
        "3. **Worked Examples or Illustrations** — concrete examples, formulas, or diagrams described in text\n"
        "4. **Common Misconceptions** — pitfalls beginners often encounter\n"
        "5. **Summary** — bullet-point recap of what was learned\n"
        "6. **Further Reading** — cite the sources used so the student can explore more.\n"
        "Adapt the depth and language to a curious learner encountering this subject for the first time."
    )
    prompt = (
        f"Teach me about: **{topic}**\n\n"
        f"Use the following educational sources to build the lesson:\n\n{context}\n\n"
        f"Write the lesson now:"
    )

    return StreamingResponse(llm(prompt, system=system, stream=True), media_type="text/plain")


@app.get("/deepSearchWebsite")
def quick_search(
    topic: str = Query(..., description="Research topic", example="i want to learn about physics electromagnetism"),
):
    """
    Fast search using DuckDuckGo snippets only (no page scraping).
    Streams the LLM summary.
    """
    query    = to_search_query(topic)
    results  = search_web(query, max_results=10)
    snippets = "\n\n".join(
        f"**{r['title']}** ({r['url']})\n{r['snippet']}"
        for r in results if r["snippet"]
    )
    system = "You are a research assistant. Summarize search results into a clear, structured answer."
    prompt = f"Topic: {topic}\n\nSearch results:\n{snippets}\n\nWrite a comprehensive summary:"

    return StreamingResponse(llm(prompt, system=system, stream=True), media_type="text/plain")


@app.get("/deepSearchYoutube")
def youtube_search(
    topic: str = Query(..., description="Search query", example="machine learning tutorial for beginners"),
):
    """Search YouTube and return video metadata."""
    query  = to_search_query(topic)
    videos = search_youtube(query)
    return {"topic": topic, "query": query, "count": len(videos), "videos": videos}


@app.get("/fullResearch")
def full_research(
    topic: str = Query(..., description="Research topic", example="large language models 2025"),
):
    """
    Combined: quick web summary (streamed) + YouTube videos.
    Returns YouTube results as JSON header, then streams the web report.
    """
    query    = to_search_query(topic)
    videos   = search_youtube(query)
    results  = search_web(query, max_results=10)
    snippets = "\n\n".join(
        f"**{r['title']}**\n{r['snippet']}" for r in results if r["snippet"]
    )
    system = "Summarize into a clear structured answer."
    prompt = f"Topic: {topic}\n\nResults:\n{snippets}\n\nSummary:"

    def _gen():
        # Emit YouTube results as JSON first line, then stream the report
        yield json.dumps({"videos": videos}) + "\n---\n"
        yield from llm(prompt, system=system, stream=True)

    return StreamingResponse(_gen(), media_type="text/plain")


@app.get("/roadmap")
def get_roadmap(
    topic: str = Query(..., description="Learning topic", example="Machine Learning"),
    completed: str = Query("", description="Comma-separated completed module IDs", example="1,2"),
):
    """
    Run deep-search on the topic, extract a personalised learning roadmap via LLM,
    persist it to Neo4j, and return recommendations.
    """
    # 1. Deep search
    queries    = generate_sub_queries(topic)
    sources    = gather_sources(queries)
    top_chunks = scrape_and_rank(sources, topic)
    context    = "\n\n".join(
        f"--- {title} ({url}) ---\n{chunk}" for _, chunk, title, url in top_chunks
    )
    system = (
        "You are a thorough research assistant. Write a comprehensive markdown report "
        "based on the provided sources."
    )
    report = llm_str(
        f"Write a deep-dive report on:\n**{topic}**\n\nSOURCES:\n{context}\n\nReport:",
        system=system,
    )

    # 2. Extract roadmap from report
    roadmap = extract_learning_roadmap(topic, report)

    # 3. Store in Neo4j & compute recommendations
    recommendations = []
    try:
        completed_ids  = [int(x) for x in completed.split(",") if x.strip().isdigit()]
        completed_uids = [f"{topic}::{i}" for i in completed_ids]
        drv = get_neo4j_driver()
        init_graph_schema(drv)
        save_roadmap_to_graph(drv, roadmap)
        recommendations = _recommend_modules(drv, topic, completed_uids)
        drv.close()
    except Exception as e:
        recommendations = []
        roadmap["_neo4j_error"] = str(e)

    # 4. Compute total duration
    total_min  = sum(m["duration_minutes"] for m in roadmap["modules"])
    h, m_rem   = divmod(total_min, 60)
    total_str  = f"{h}h {m_rem:02d}min" if h else f"{m_rem}min"

    return {
        "topic":           roadmap["topic"],
        "level":           roadmap["level"],
        "level_emoji":     roadmap["level_emoji"],
        "gaps":            roadmap["gaps"],
        "modules":         roadmap["modules"],
        "total_duration":  total_str,
        "recommendations": recommendations,
        "report_excerpt":  report[:500],
    }


@app.get("/recommend")
def recommend(
    topic: str = Query(..., description="Learning topic", example="Machine Learning"),
    completed: str = Query("", description="Comma-separated completed module IDs", example="1,2,3"),
):
    """Return next-best module recommendations from the Neo4j graph."""
    try:
        completed_ids  = [int(x) for x in completed.split(",") if x.strip().isdigit()]
        completed_uids = [f"{topic}::{i}" for i in completed_ids]
        drv = get_neo4j_driver()
        recs = _recommend_modules(drv, topic, completed_uids)
        drv.close()
        return {"topic": topic, "recommendations": recs}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Neo4j unavailable: {e}")


@app.get("/showDB")
def show_db(limit: int = Query(20, description="Max rows to return", example=10)):
    """Return the history of all past searches stored in PostgreSQL."""
    try:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT s.id, s.topic, s.search_type, s.created_at,
                           COUNT(DISTINCT w.id) AS web_count,
                           COUNT(DISTINCT y.id) AS yt_count,
                           COUNT(DISTINCT r.id) AS report_count
                    FROM searches s
                    LEFT JOIN web_results     w ON w.search_id = s.id
                    LEFT JOIN youtube_results y ON y.search_id = s.id
                    LEFT JOIN reports         r ON r.search_id = s.id
                    GROUP BY s.id ORDER BY s.created_at DESC LIMIT %s
                    """,
                    (limit,),
                )
                rows = cur.fetchall()
        return {"searches": [dict(r) for r in rows]}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"DB unavailable: {e}")


@app.get("/report/{search_id}")
def get_report(
    search_id: int = Path(..., description="Search ID returned by a previous search", example=1),
):
    """Retrieve a stored research report by its search ID."""
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT content FROM reports WHERE search_id=%s ORDER BY id DESC LIMIT 1",
                    (search_id,),
                )
                row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"No report for search_id={search_id}")
        return {"search_id": search_id, "content": row[0]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"DB unavailable: {e}")


@app.get("/videos/{search_id}")
def get_videos(
    search_id: int = Path(..., description="Search ID returned by a YouTube or combined search", example=2),
):
    """Retrieve stored YouTube results by search ID."""
    try:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM youtube_results WHERE search_id=%s ORDER BY id",
                    (search_id,),
                )
                rows = cur.fetchall()
        return {"search_id": search_id, "videos": [dict(r) for r in rows]}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"DB unavailable: {e}")


class QuizRequest(BaseModel):
    search_id: Optional[int] = None      # load content from DB (web report or YT transcript)
    content:   Optional[str] = None      # raw text when no search_id given
    num_questions: int = 5               # 1–10


def _fetch_transcript(video_id: str) -> str:
    """Fetch YouTube auto/manual subtitles as plain text."""
    api = YouTubeTranscriptApi()
    try:
        segments = api.fetch(video_id, languages=["en", "en-US", "en-GB"])
        return " ".join(s.text for s in segments)
    except Exception:
        # Fallback: try any available language
        try:
            t = next(iter(api.list(video_id)))
            segments = t.fetch()
            return " ".join(s.text for s in segments)
        except Exception:
            return ""


def _content_from_db(search_id: int) -> tuple[str, str]:
    """
    Return (content, source_label) for a given search_id.
    - web / quick / full → report text (fallback: joined snippets)
    - youtube            → concatenated transcripts
    - combined           → report + transcripts
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Get search metadata
            cur.execute("SELECT topic, search_type FROM searches WHERE id = %s", (search_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail=f"search_id={search_id} not found")
            search_type = row["search_type"]

            parts  = []
            labels = []

            # ── Web/report content ────────────────────────────────────────────
            if search_type in ("quick", "full", "combined"):
                cur.execute(
                    "SELECT content FROM reports WHERE search_id=%s ORDER BY id DESC LIMIT 1",
                    (search_id,),
                )
                rep = cur.fetchone()
                if rep and rep["content"]:
                    parts.append(rep["content"])
                    labels.append("web report")
                else:
                    # Fallback to raw snippets
                    cur.execute(
                        "SELECT title, snippet FROM web_results WHERE search_id=%s",
                        (search_id,),
                    )
                    snippets = "\n\n".join(
                        f"{r['title']}\n{r['snippet']}"
                        for r in cur.fetchall() if r["snippet"]
                    )
                    if snippets:
                        parts.append(snippets)
                        labels.append("web snippets")

            # ── YouTube transcripts ───────────────────────────────────────────
            if search_type in ("youtube", "combined"):
                cur.execute(
                    "SELECT video_id, title FROM youtube_results WHERE search_id=%s",
                    (search_id,),
                )
                videos = cur.fetchall()
                for v in videos:
                    transcript = _fetch_transcript(v["video_id"])
                    if transcript:
                        parts.append(f"[Video: {v['title']}]\n{transcript}")
                        labels.append(f"transcript:{v['video_id']}")

    if not parts:
        raise HTTPException(
            status_code=422,
            detail=f"No content found for search_id={search_id} (type={search_type}). "
                   "Run the search first to populate the DB.",
        )

    source_label = f"DB search_id={search_id} ({', '.join(labels)})"
    return "\n\n".join(parts), source_label


def _build_quiz(content: str, num_questions: int) -> list:
    num_questions = max(1, min(num_questions, 10))
    prompt = f"""You are a quiz generator. Based on the content below, create {num_questions} multiple-choice questions.

Return ONLY valid JSON — no markdown fences, no extra text:
[
  {{
    "question": "Question text?",
    "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
    "answer": "A",
    "explanation": "Brief explanation of why A is correct."
  }}
]

Rules:
• Exactly {num_questions} questions
• 4 options per question labelled A, B, C, D
• "answer" is the letter only (A / B / C / D)
• Questions must be answerable from the content
• Vary difficulty: mix recall, comprehension, and application

Content:
{content[:4000]}"""

    raw   = llm_str(prompt)
    match = re.search(r"\[[\s\S]*\]", raw)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    raise HTTPException(status_code=500, detail="LLM returned invalid JSON — please retry.")


@app.post(
    "/quiz",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "from_web_search": {
                            "summary": "Quiz from stored web report (search_id=1)",
                            "value": {"search_id": 1, "num_questions": 5},
                        },
                        "from_youtube": {
                            "summary": "Quiz from YouTube search — auto-transcribes (search_id=3)",
                            "value": {"search_id": 3, "num_questions": 4},
                        },
                        "raw_content": {
                            "summary": "Quiz from raw text content",
                            "value": {
                                "content": (
                                    "Machine learning is a subset of artificial intelligence. "
                                    "Supervised learning uses labelled data to train models. "
                                    "Unsupervised learning finds hidden patterns without labels. "
                                    "Neural networks are inspired by the human brain and consist "
                                    "of layers of interconnected nodes called neurons."
                                ),
                                "num_questions": 3,
                            },
                        },
                    }
                }
            }
        }
    },
)
def generate_quiz(req: QuizRequest):
    """
    Generate a multiple-choice quiz.

    **Option 1 — from DB** (`search_id` provided):
    - Web / quick / full search → uses the stored report text
    - YouTube search            → fetches video transcripts automatically
    - Combined search           → report + transcripts

    **Option 2 — raw content** (`content` provided, no `search_id`):
    - Uses the text you supply directly

    Response includes each question, 4 options (A–D), the correct answer letter, and an explanation.
    """
    if req.search_id is not None:
        content, source = _content_from_db(req.search_id)
    elif req.content:
        content = req.content
        source  = "inline content"
    else:
        raise HTTPException(
            status_code=422,
            detail="Provide either 'search_id' (load from DB) or 'content' (raw text).",
        )

    questions = _build_quiz(content, req.num_questions)
    return {
        "source":        source,
        "num_questions": len(questions),
        "questions":     questions,
    }


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
def on_startup():
    try:
        init_db()
        print("✅ PostgreSQL schema ready")
    except Exception as e:
        print(f"⚠️  PostgreSQL unavailable: {e}")
