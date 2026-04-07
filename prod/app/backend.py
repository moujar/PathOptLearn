import hashlib
import json
import re
import secrets as _secrets
import time
from datetime import datetime, timedelta
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
import os

MODEL              = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")
MAX_SEARCH_RESULTS = int(os.environ.get("MAX_SEARCH_RESULTS", 6))
MAX_SUB_QUERIES    = int(os.environ.get("MAX_SUB_QUERIES", 3))
MAX_CHUNK_CHARS    = int(os.environ.get("MAX_CHUNK_CHARS", 1500))
TOP_CHUNKS         = int(os.environ.get("TOP_CHUNKS", 8))
REQUEST_TIMEOUT    = int(os.environ.get("REQUEST_TIMEOUT", 10))
YT_MAX_RESULTS     = int(os.environ.get("YT_MAX_RESULTS", 6))

DB = dict(
    host     = os.environ.get("POSTGRES_HOST", "localhost"),
    port     = int(os.environ.get("POSTGRES_PORT", 5432)),
    dbname   = os.environ.get("POSTGRES_DB",   "pathoptlearn"),
    user     = os.environ.get("POSTGRES_USER",  "pathoptlearn"),
    password = os.environ.get("POSTGRES_PASSWORD", "pathoptlearn"),
)

NEO4J_URI  = os.environ.get("NEO4J_URI",  "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASS = os.environ.get("NEO4J_PASS", "pathoptlearn")

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
                CREATE TABLE IF NOT EXISTS users (
                    id         SERIAL PRIMARY KEY,
                    username   TEXT NOT NULL UNIQUE,
                    email      TEXT NOT NULL UNIQUE,
                    full_name  TEXT,
                    level      TEXT NOT NULL DEFAULT 'beginner',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE TABLE IF NOT EXISTS learning_sessions (
                    id         SERIAL PRIMARY KEY,
                    user_id    INT REFERENCES users(id) ON DELETE SET NULL,
                    topic      TEXT NOT NULL,
                    level      TEXT NOT NULL DEFAULT 'beginner',
                    level_emoji TEXT NOT NULL DEFAULT '🟢',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                -- Add user_id column to existing learning_sessions if it doesn't exist yet
                DO $$ BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name='learning_sessions' AND column_name='user_id'
                    ) THEN
                        ALTER TABLE learning_sessions
                            ADD COLUMN user_id INT REFERENCES users(id) ON DELETE SET NULL;
                    END IF;
                END $$;
                CREATE TABLE IF NOT EXISTS session_progress (
                    id           SERIAL PRIMARY KEY,
                    session_id   INT REFERENCES learning_sessions(id) ON DELETE CASCADE,
                    module_uid   TEXT NOT NULL,
                    module_title TEXT,
                    quiz_score   FLOAT,
                    completed_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE TABLE IF NOT EXISTS lessons (
                    id          SERIAL PRIMARY KEY,
                    session_id  INT REFERENCES learning_sessions(id) ON DELETE CASCADE,
                    module_uid  TEXT NOT NULL,
                    module_title TEXT,
                    content     TEXT,
                    created_at  TIMESTAMPTZ DEFAULT NOW()
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

# ── DB cache helpers ──────────────────────────────────────────────────────────

def _cached_search(topic: str, kind: str) -> Optional[dict]:
    """
    Return the most recent cached search for (topic, kind) if it exists and has content.
    Returns: {"search_id": int, "report": str|None, "web": list, "videos": list}
    or None if no usable cache exists.
    """
    try:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Match case-insensitively on a cleaned topic
                cur.execute(
                    """SELECT s.id, r.content AS report
                       FROM searches s
                       LEFT JOIN reports r ON r.search_id = s.id
                       WHERE LOWER(s.topic) = LOWER(%s) AND s.search_type = %s
                       ORDER BY s.created_at DESC LIMIT 1""",
                    (topic, kind),
                )
                row = cur.fetchone()
                if not row:
                    return None
                sid = row["id"]

                cur.execute(
                    "SELECT title, url, snippet FROM web_results WHERE search_id = %s",
                    (sid,),
                )
                web = [dict(r) for r in cur.fetchall()]

                cur.execute(
                    """SELECT video_id AS id, title, channel, duration,
                              views, description AS desc, url, thumbnail AS thumb
                       FROM youtube_results WHERE search_id = %s""",
                    (sid,),
                )
                videos = [dict(r) for r in cur.fetchall()]

        # Only return a cache hit when there is actually something useful
        if row["report"] or web or videos:
            return {
                "search_id": sid,
                "report":    row["report"],
                "web":       web,
                "videos":    videos,
            }
    except Exception:
        pass
    return None


def _cached_lesson(session_id: int, module_uid: str) -> Optional[str]:
    """Return cached lesson content for this session + module, or None."""
    try:
        schema = _progress_schema(session_id)
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f'SELECT content FROM "{schema}".lessons '
                    f'WHERE session_id=%s AND module_uid=%s ORDER BY id DESC LIMIT 1',
                    (session_id, module_uid),
                )
                row = cur.fetchone()
        return row[0] if row and row[0] else None
    except Exception:
        return None


# ── Per-user schema helpers ────────────────────────────────────────────────────

def _user_schema(user_id: Optional[int]) -> str:
    """Schema name for a user's private history. Falls back to 'public' for anonymous."""
    return f"user_{user_id}" if user_id else "public"


_USER_HISTORY_SQL = """
    CREATE TABLE IF NOT EXISTS "{s}".session_progress (
        id           SERIAL PRIMARY KEY,
        session_id   INT NOT NULL,
        module_uid   TEXT NOT NULL,
        module_title TEXT,
        quiz_score   FLOAT,
        completed_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS "{s}".lessons (
        id           SERIAL PRIMARY KEY,
        session_id   INT NOT NULL,
        module_uid   TEXT NOT NULL,
        module_title TEXT,
        content      TEXT,
        created_at   TIMESTAMPTZ DEFAULT NOW()
    );
"""


def _create_user_schema(user_id: int):
    """Create a private PostgreSQL schema for the user with their history tables."""
    schema = _user_schema(user_id)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
            cur.execute(_USER_HISTORY_SQL.format(s=schema))
        conn.commit()


def _progress_schema(session_id: int) -> str:
    """Look up which schema owns the progress/lessons for a given session_id."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id FROM learning_sessions WHERE id = %s", (session_id,)
            )
            row = cur.fetchone()
    user_id = row[0] if row and row[0] else None
    return _user_schema(user_id)


# ── Neo4j helpers ─────────────────────────────────────────────────────────────

def get_neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

def init_graph_schema(driver):
    with driver.session() as s:
        s.run("CREATE CONSTRAINT topic_unique    IF NOT EXISTS FOR (t:Topic)    REQUIRE t.name IS UNIQUE")
        s.run("CREATE CONSTRAINT module_unique   IF NOT EXISTS FOR (m:Module)   REQUIRE m.uid  IS UNIQUE")
        s.run("CREATE CONSTRAINT concept_unique  IF NOT EXISTS FOR (c:Concept)  REQUIRE c.name IS UNIQUE")
        s.run("CREATE CONSTRAINT level_unique    IF NOT EXISTS FOR (l:Level)    REQUIRE l.name IS UNIQUE")
        s.run("CREATE CONSTRAINT resource_unique IF NOT EXISTS FOR (r:Resource) REQUIRE r.url  IS UNIQUE")

def save_roadmap_to_graph(driver, roadmap: dict):
    """Persist roadmap as a property graph (Topic · LevelGroup · Module · Concept)."""
    topic = roadmap["topic"]
    level = roadmap.get("level", roadmap.get("current_level", "beginner"))
    with driver.session() as s:
        s.run(
            "MERGE (t:Topic {name:$n}) SET t.level=$lv, t.emoji=$em",
            n=topic, lv=level, em=roadmap.get("level_emoji", "🟢"),
        )
        # Store each level group as a LevelGroup node linked to the topic
        for lg in roadmap.get("levels", []):
            lg_name = f"{topic}::L{lg['level_num']}"
            s.run(
                """MERGE (lg:LevelGroup {uid:$uid})
                   SET lg.level_num=$n, lg.level_name=$name, lg.emoji=$em, lg.topic=$topic""",
                uid=lg_name, n=lg["level_num"], name=lg["level_name"],
                em=lg["emoji"], topic=topic,
            )
            s.run(
                "MATCH (t:Topic{name:$t}),(lg:LevelGroup{uid:$u}) MERGE (t)-[:HAS_LEVEL]->(lg)",
                t=topic, u=lg_name,
            )
            for mod in lg["modules"]:
                uid = f"{topic}::{mod['id']}"
                s.run(
                    """MERGE (m:Module {uid:$uid})
                       SET m.title=$title, m.objective=$obj, m.duration_minutes=$dur,
                           m.order=$n, m.topic=$topic, m.level_num=$lnum, m.level_name=$lname""",
                    uid=uid, title=mod["title"], obj=mod["objective"],
                    dur=mod["duration_minutes"], n=mod["id"], topic=topic,
                    lnum=mod.get("level_num", 1), lname=mod.get("level_name", "Beginner"),
                )
                s.run(
                    "MATCH (lg:LevelGroup{uid:$lg}),(m:Module{uid:$u}) MERGE (lg)-[:HAS_MODULE]->(m)",
                    lg=lg_name, u=uid,
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
        # Fallback: if no levels key (old format), persist flat modules
        if not roadmap.get("levels"):
            for mod in roadmap.get("modules", []):
                uid = f"{topic}::{mod['id']}"
                s.run(
                    """MERGE (m:Module {uid:$uid})
                       SET m.title=$title, m.objective=$obj, m.duration_minutes=$dur,
                           m.order=$n, m.topic=$topic""",
                    uid=uid, title=mod["title"], obj=mod["objective"],
                    dur=mod["duration_minutes"], n=mod["id"], topic=topic,
                )
                s.run(
                    "MATCH (t:Topic{name:$t}),(m:Module{uid:$u}) MERGE (t)-[:HAS_MODULE]->(m)",
                    t=topic, u=uid,
                )

def _save_resources_to_graph(driver, topic: str, web_results: list, videos: list,
                              search_type: str = "search"):
    """
    Persist web articles and YouTube videos as Resource nodes linked to a Topic.
    Graph: (Topic)-[:HAS_RESOURCE]->(Resource)
    Resource properties: url, title, type ('article'|'video'), snippet/channel/duration.
    """
    with driver.session() as s:
        # Ensure topic node exists
        s.run("MERGE (t:Topic {name:$n})", n=topic)

        for r in web_results:
            if not r.get("url"):
                continue
            s.run(
                """MERGE (res:Resource {url: $url})
                   SET res.title       = $title,
                       res.type        = 'article',
                       res.snippet     = $snippet,
                       res.search_type = $st,
                       res.kg_source   = $kg_source
                   WITH res
                   MATCH (t:Topic {name: $topic})
                   MERGE (t)-[:HAS_RESOURCE]->(res)""",
                url=r["url"], title=r.get("title", ""), snippet=r.get("snippet", ""),
                st=search_type, topic=topic,
                kg_source=r.get("source", "web"),
            )

        for v in videos:
            if not v.get("url"):
                continue
            s.run(
                """MERGE (res:Resource {url: $url})
                   SET res.title       = $title,
                       res.type        = 'video',
                       res.video_id    = $vid,
                       res.channel     = $channel,
                       res.duration    = $duration,
                       res.views       = $views,
                       res.thumbnail   = $thumb,
                       res.search_type = $st
                   WITH res
                   MATCH (t:Topic {name: $topic})
                   MERGE (t)-[:HAS_RESOURCE]->(res)""",
                url=v["url"], title=v.get("title", ""), vid=v.get("id", ""),
                channel=v.get("channel", ""), duration=v.get("duration", ""),
                views=v.get("views", 0) or 0, thumb=v.get("thumb", ""),
                st=search_type, topic=topic,
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

_ollama_client = ollama.Client(host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))


def llm(prompt: str, system: str = "", stream: bool = False):
    """Call Ollama. Returns full string or a streaming generator."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    if stream:
        def _gen():
            for chunk in _ollama_client.chat(model=MODEL, messages=messages, stream=True):
                yield chunk["message"]["content"]
        return _gen()

    result = _ollama_client.chat(model=MODEL, messages=messages)
    return result["message"]["content"]

def llm_str(prompt: str, system: str = "") -> str:
    """Non-streaming LLM call, always returns a string."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return _ollama_client.chat(model=MODEL, messages=messages)["message"]["content"]

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

# ── Open Knowledge Databases ─────────────────────────────────────────────────

_KG_TIMEOUT = 5   # per-request HTTP timeout for all KG calls
_KG_BUDGET  = 10  # max seconds to wait for all KG fetches combined

def _kg_wikipedia(topic: str) -> list[dict]:
    """Wikipedia article intro via MediaWiki API (single batched request)."""
    results = []
    try:
        search_url = "https://en.wikipedia.org/w/api.php"
        # One request: search + fetch extracts together using generator
        resp = requests.get(search_url, params={
            "action": "query", "list": "search", "srsearch": topic,
            "srlimit": 1, "format": "json",
        }, timeout=_KG_TIMEOUT).json()
        hits = resp.get("query", {}).get("search", [])
        if not hits:
            return results
        pageid = hits[0]["pageid"]
        ext = requests.get(search_url, params={
            "action": "query", "pageids": pageid,
            "prop": "extracts", "exintro": True,
            "explaintext": True, "exchars": 1500, "format": "json",
        }, timeout=_KG_TIMEOUT).json()
        page = next(iter(ext["query"]["pages"].values()))
        extract = page.get("extract", "").strip()
        page_title = page.get("title", topic)
        if extract:
            results.append({
                "title":   f"Wikipedia: {page_title}",
                "url":     f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}",
                "snippet": extract,
                "source":  "wikipedia",
            })
    except Exception:
        pass
    return results


def _kg_wikidata(topic: str) -> list[dict]:
    """Wikidata entity search + description only (no SPARQL to avoid timeouts)."""
    results = []
    try:
        search = requests.get(
            "https://www.wikidata.org/w/api.php",
            params={"action": "wbsearchentities", "search": topic,
                    "language": "en", "limit": 3, "format": "json"},
            timeout=_KG_TIMEOUT,
        ).json()
        hits = search.get("search", [])
        facts = []
        for hit in hits[:3]:
            label = hit.get("label", "")
            desc  = hit.get("description", "")
            qid   = hit.get("id", "")
            if label and desc:
                facts.append(f"**{label}** ({qid}): {desc}")
        if facts:
            results.append({
                "title":   f"Wikidata: {topic}",
                "url":     f"https://www.wikidata.org/w/index.php?search={topic.replace(' ', '+')}",
                "snippet": "\n".join(facts),
                "source":  "wikidata",
            })
    except Exception:
        pass
    return results


def _kg_openalex(topic: str, max_results: int = 3) -> list[dict]:
    """OpenAlex open-access academic papers with reconstructed abstracts."""
    results = []
    try:
        resp = requests.get(
            "https://api.openalex.org/works",
            params={
                "search": topic,
                "filter": "is_oa:true",
                "sort": "cited_by_count:desc",
                "per-page": max_results,
                "select": "title,abstract_inverted_index,primary_location,cited_by_count,publication_year",
            },
            headers={"User-Agent": "DeepTutor/1.0 (mailto:research@deeptutor.ai)"},
            timeout=_KG_TIMEOUT,
        ).json()
        for work in resp.get("results", []):
            title = work.get("title", "")
            year  = work.get("publication_year", "")
            cited = work.get("cited_by_count", 0)
            url   = (work.get("primary_location") or {}).get("landing_page_url", "")
            inv   = work.get("abstract_inverted_index") or {}
            if inv:
                positions = {}
                for word, pos_list in inv.items():
                    for p in pos_list:
                        positions[p] = word
                abstract = " ".join(positions[k] for k in sorted(positions))
            else:
                abstract = ""
            if title and abstract:
                results.append({
                    "title":   f"[Paper {year}] {title} ({cited} citations)",
                    "url":     url or f"https://openalex.org/works?search={topic}",
                    "snippet": abstract[:1200],
                    "source":  "openalex",
                })
    except Exception:
        pass
    return results


def _kg_arxiv(topic: str, max_results: int = 3) -> list[dict]:
    """arXiv preprints — CS, math, physics, bio (Atom XML feed)."""
    results = []
    try:
        from xml.etree import ElementTree as ET
        resp = requests.get(
            "https://export.arxiv.org/api/query",
            params={
                "search_query": f"all:{topic}",
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
            },
            timeout=_KG_TIMEOUT,
        )
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(resp.text)
        for entry in root.findall("atom:entry", ns):
            title   = (entry.findtext("atom:title", "", ns) or "").strip().replace("\n", " ")
            summary = (entry.findtext("atom:summary", "", ns) or "").strip().replace("\n", " ")
            link_el = entry.find("atom:id", ns)
            url     = link_el.text.strip() if link_el is not None else ""
            if title and summary:
                results.append({
                    "title":   f"[arXiv] {title}",
                    "url":     url,
                    "snippet": summary[:1200],
                    "source":  "arxiv",
                })
    except Exception:
        pass
    return results


def gather_open_knowledge(topic: str) -> list[dict]:
    """
    Query open knowledge databases in parallel with a strict time budget.
    Sources: Wikipedia · Wikidata · OpenAlex · arXiv
    Returns whatever finishes within _KG_BUDGET seconds; slow/failing sources are skipped.
    """
    import concurrent.futures
    fetchers = [_kg_wikipedia, _kg_wikidata, _kg_openalex, _kg_arxiv]
    all_sources = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_map = {executor.submit(fn, topic): fn.__name__ for fn in fetchers}
        done, _ = concurrent.futures.wait(
            future_map, timeout=_KG_BUDGET,
            return_when=concurrent.futures.ALL_COMPLETED,
        )
        for future in done:
            try:
                all_sources.extend(future.result(timeout=0))
            except Exception:
                pass
    return all_sources


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

def gather_sources(queries: list, topic: str = "") -> list:
    seen, all_results = set(), []
    # 1. Regular web search across all sub-queries
    for q in queries:
        for r in search_web(q):
            if r["url"] and r["url"] not in seen:
                seen.add(r["url"])
                all_results.append(r)
        time.sleep(0.3)
    # 2. Open knowledge databases (Wikipedia, Wikidata, OpenAlex, arXiv, DBpedia)
    if topic:
        for r in gather_open_knowledge(topic):
            if r["url"] and r["url"] not in seen:
                seen.add(r["url"])
                all_results.append(r)
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

def _clean_topic(raw: str) -> str:
    """Extract the real academic subject from natural language input.
    e.g. 'i want to learn Master ML' → 'Machine Learning'
    """
    prompt = (
        f'Extract the main academic subject name from this phrase. '
        f'Return ONLY 2–5 words, the subject name only, no extra text.\n'
        f'Examples: "i want to learn calculus" → "Calculus"  |  '
        f'"teach me ML" → "Machine Learning"\n'
        f'Phrase: "{raw}"\nSubject:'
    )
    result = llm_str(prompt).strip().strip('"\'.,').strip()
    # Sanity check: if output is too long or empty, return original
    if not result or len(result) > 80:
        return raw
    return result


def _generate_level_modules(topic: str, level_num: int, level_name: str,
                             start_id: int, all_prev: list) -> list:
    """Ask the LLM to produce exactly 5 modules for one level of the roadmap."""
    prereqs = [m["id"] for m in all_prev[-2:]] if all_prev else []
    prompt = (
        f'Create 5 learning modules for "{topic}" at {level_name} level.\n'
        f'Reply ONLY with a JSON array, no markdown fences:\n'
        f'[{{"id":{start_id},"title":"...","objective":"...","concepts":["c1","c2"],'
        f'"duration_minutes":60,"prerequisites":{prereqs}}}]\n'
        f'Module ids must be {start_id} to {start_id+4}. '
        f'Build on previous levels. Be specific to "{topic}".'
    )
    raw = llm_str(prompt)
    parsed = [q for q in (_parse_json_array(raw) or []) if isinstance(q, dict)]

    # Normalize and fill gaps
    result = []
    for i in range(5):
        m = parsed[i] if i < len(parsed) else {}
        m["id"]               = start_id + i
        m.setdefault("title",            f"{topic} — {level_name} {i + 1}")
        m.setdefault("objective",        f"Learn {level_name.lower()} {topic} concepts, part {i + 1}")
        m.setdefault("concepts",         [f"{level_name.lower()}", f"{topic.lower()} basics"])
        m.setdefault("duration_minutes", 60)
        m["prerequisites"]    = ([start_id - 1] if i == 0 and prereqs else
                                  [start_id + i - 1] if i > 0 else [])
        m["level_num"]        = level_num
        m["level_name"]       = level_name
        if not isinstance(m.get("concepts"), list):
            m["concepts"] = [f"{level_name.lower()} concept"]
        result.append(m)
    return result


_LEVEL_PALETTE = [
    ("Beginner",     "🟢"),
    ("Intermediate", "🟡"),
    ("Advanced",     "🔴"),
    ("Expert",       "🔵"),
    ("Master",       "🟣"),
]


def _decide_levels(topic: str) -> list[tuple[int, str, str]]:
    """
    Ask the LLM how many learning levels this topic requires (2–5) and their names.
    Returns a list of (level_num, level_name, emoji).
    Falls back to 3 standard levels on any failure.
    """
    prompt = (
        f'How many learning levels does mastering "{topic}" require? '
        f'Reply with ONLY a JSON array of level names, 2–5 items, e.g. '
        f'["Beginner","Intermediate","Advanced"] or '
        f'["Foundations","Core","Advanced","Expert"].\n'
        f'Topic: "{topic}". Array:'
    )
    raw    = llm_str(prompt)
    parsed = _parse_json_array(raw)

    # Validate: must be a list of 2–5 strings
    if (parsed and isinstance(parsed, list) and 2 <= len(parsed) <= 5
            and all(isinstance(n, str) for n in parsed)):
        names = parsed
    else:
        names = ["Beginner", "Intermediate", "Advanced"]

    return [
        (i + 1, name, _LEVEL_PALETTE[min(i, len(_LEVEL_PALETTE) - 1)][1])
        for i, name in enumerate(names)
    ]


def extract_multilevel_roadmap(topic: str, report: str, user_level: str = "beginner") -> dict:
    """
    Generate an N-level × 5-module learning roadmap.
    Number of levels (2–5) is decided by the LLM based on topic complexity.
    """
    levels_meta = _decide_levels(topic)   # [(1, "Beginner", "🟢"), ...]

    all_modules: list  = []
    level_groups: list = []

    for level_num, level_name, emoji in levels_meta:
        start_id = (level_num - 1) * 5 + 1
        modules  = _generate_level_modules(topic, level_num, level_name, start_id, all_modules)
        all_modules.extend(modules)
        level_groups.append({
            "level_num":  level_num,
            "level_name": level_name,
            "emoji":      emoji,
            "modules":    modules,
        })

    total_min  = sum(m.get("duration_minutes", 60) for m in all_modules)
    h, m_rem   = divmod(total_min, 60)
    user_emoji = {"beginner": "🟢", "intermediate": "🟡", "advanced": "🔴"}.get(user_level, "🟢")

    return {
        "topic":          topic,
        "current_level":  user_level,
        "level_emoji":    user_emoji,
        "level":          user_level,
        "gaps":           [f"fundamentals of {topic}", f"applied {topic}", f"mastery of {topic}"],
        "levels":         level_groups,
        "modules":        all_modules,
        "total_modules":  len(all_modules),
        "total_duration": f"{h}h {m_rem:02d}min" if h else f"{m_rem}min",
    }


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

    _fallback = {
        "topic":       topic,
        "level":       "beginner",
        "level_emoji": "🟢",
        "gaps":        [f"fundamentals of {topic}"],
        "modules": [{
            "id": 1,
            "title": f"{topic}: Introduction",
            "objective": f"Understand the core concepts of {topic}",
            "concepts": ["overview", "fundamentals"],
            "duration_minutes": 60,
            "prerequisites": [],
        }],
    }

    def _to_str_list(val, fallback: list) -> list:
        """Ensure a value is a list of plain strings (the LLM sometimes returns dicts)."""
        if not isinstance(val, list) or not val:
            return fallback
        result = []
        for item in val:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                # e.g. {"gap": "some text"} or {"name": "some text"}
                result.append(next(iter(item.values()), str(item)))
            else:
                result.append(str(item))
        return result or fallback

    def _normalise(parsed: dict) -> dict:
        """Guarantee every required key exists at top-level and inside each module."""
        parsed.setdefault("topic",       topic)
        parsed.setdefault("level",       "beginner")
        parsed.setdefault("level_emoji", "🟢")
        parsed["gaps"] = _to_str_list(parsed.get("gaps"), [f"fundamentals of {topic}"])
        parsed.setdefault("modules",     _fallback["modules"])
        for i, mod in enumerate(parsed["modules"]):
            mod.setdefault("id",               i + 1)
            mod.setdefault("title",            f"Module {i + 1}")
            mod.setdefault("objective",        f"Learn module {i + 1}")
            mod["concepts"] = _to_str_list(mod.get("concepts"), [])
            mod.setdefault("duration_minutes", 60)
            mod.setdefault("prerequisites",    [])
        return parsed

    raw = llm_str(prompt)
    # Strip markdown fences, then try reversed matches so we get the outermost object
    stripped = re.sub(r"```(?:json)?\s*([\s\S]*?)```", r"\1", raw).strip()
    for text in (stripped, raw):
        matches = list(re.finditer(r"\{[\s\S]*\}", text))
        for m in reversed(matches):
            candidate = re.sub(r",\s*([}\]])", r"\1", m.group())  # fix trailing commas
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict) and "modules" in parsed:
                    return _normalise(parsed)
            except (json.JSONDecodeError, ValueError):
                continue
    return _fallback

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
    Checks the DB cache first — if a report for this topic already exists, streams it directly.
    Persists sources + report to PostgreSQL and Neo4j on a cache miss.
    """
    # ── Cache hit: stream the stored report directly ───────────────────────────
    cache = _cached_search(topic, "deepSearch")
    if cache and cache["report"]:
        print(f"[deepSearch] cache hit for '{topic}' (search_id={cache['search_id']})")
        def _cached_gen():
            yield f"*(from cache — search #{cache['search_id']})*\n\n"
            yield cache["report"]
        return StreamingResponse(_cached_gen(), media_type="text/plain")

    # ── Cache miss: run full pipeline ─────────────────────────────────────────
    queries    = generate_sub_queries(topic)
    sources    = gather_sources(queries, topic)
    top_chunks = scrape_and_rank(sources, topic)
    sid        = _new_search(topic, "deepSearch")
    _save_web(sid, sources)

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

    def _gen():
        full = []
        for chunk in llm(prompt, system=system, stream=True):
            full.append(chunk)
            yield chunk
        _save_report(sid, "".join(full))
        try:
            drv = get_neo4j_driver()
            init_graph_schema(drv)
            _save_resources_to_graph(drv, topic, sources, [], search_type="deepSearch")
            drv.close()
        except Exception:
            pass

    return StreamingResponse(_gen(), media_type="text/plain")


@app.get("/deepSearchWebsite")
def quick_search(
    topic: str = Query(..., description="Research topic", example="i want to learn about physics electromagnetism"),
):
    """
    Fast search using DuckDuckGo snippets only (no page scraping).
    Checks the DB cache first. Streams from cache or runs fresh pipeline.
    """
    cache = _cached_search(topic, "quickSearch")
    if cache and cache["report"]:
        print(f"[quickSearch] cache hit for '{topic}'")
        def _cached_gen():
            yield f"*(from cache — search #{cache['search_id']})*\n\n"
            yield cache["report"]
        return StreamingResponse(_cached_gen(), media_type="text/plain")

    query   = to_search_query(topic)
    results = search_web(query, max_results=10)
    # Enrich with open knowledge databases
    kg_sources = gather_open_knowledge(topic)
    results    = results + kg_sources
    sid     = _new_search(topic, "quickSearch")
    _save_web(sid, results)

    snippets = "\n\n".join(
        f"**{r['title']}** ({r['url']})\n{r['snippet']}"
        for r in results if r.get("snippet")
    )
    system = "You are a research assistant. Summarize search results into a clear, structured answer."
    prompt = f"Topic: {topic}\n\nSearch results:\n{snippets}\n\nWrite a comprehensive summary:"

    def _gen():
        full = []
        for chunk in llm(prompt, system=system, stream=True):
            full.append(chunk)
            yield chunk
        _save_report(sid, "".join(full))
        try:
            drv = get_neo4j_driver()
            init_graph_schema(drv)
            _save_resources_to_graph(drv, topic, results, [], search_type="quickSearch")
            drv.close()
        except Exception:
            pass

    return StreamingResponse(_gen(), media_type="text/plain")


@app.get("/deepSearchYoutube")
def youtube_search(
    topic: str = Query(..., description="Search query", example="machine learning tutorial for beginners"),
):
    """
    Search YouTube and return video metadata.
    Returns cached results if available; otherwise fetches and persists.
    """
    cache = _cached_search(topic, "youtube")
    if cache and cache["videos"]:
        print(f"[youtube] cache hit for '{topic}'")
        return {
            "topic": topic, "query": topic,
            "search_id": cache["search_id"],
            "count": len(cache["videos"]),
            "videos": cache["videos"],
            "cached": True,
        }

    query  = to_search_query(topic)
    videos = search_youtube(query)
    sid    = _new_search(topic, "youtube")
    _save_yt(sid, videos)
    try:
        drv = get_neo4j_driver()
        init_graph_schema(drv)
        _save_resources_to_graph(drv, topic, [], videos, search_type="youtube")
        drv.close()
    except Exception:
        pass

    return {"topic": topic, "query": query, "search_id": sid, "count": len(videos), "videos": videos}


@app.get("/fullResearch")
def full_research(
    topic: str = Query(..., description="Research topic", example="large language models 2025"),
):
    """
    Combined: quick web summary (streamed) + YouTube videos.
    Returns YouTube results as JSON header, then streams the web report.
    Checks DB cache first; only runs full pipeline on a cache miss.
    """
    cache = _cached_search(topic, "fullResearch")
    if cache and cache["report"] and cache["videos"]:
        print(f"[fullResearch] cache hit for '{topic}'")
        def _cached_gen():
            yield json.dumps({"videos": cache["videos"], "search_id": cache["search_id"], "cached": True}) + "\n---\n"
            yield f"*(from cache — search #{cache['search_id']})*\n\n"
            yield cache["report"]
        return StreamingResponse(_cached_gen(), media_type="text/plain")

    query   = to_search_query(topic)
    videos  = search_youtube(query)
    results = search_web(query, max_results=10)
    # Enrich with open knowledge databases
    kg_sources = gather_open_knowledge(topic)
    results    = results + kg_sources
    sid     = _new_search(topic, "fullResearch")
    _save_web(sid, results)
    _save_yt(sid, videos)

    snippets = "\n\n".join(
        f"**{r['title']}**\n{r.get('snippet', '')}" for r in results if r.get("snippet")
    )
    system = "Summarize into a clear structured answer."
    prompt = f"Topic: {topic}\n\nResults:\n{snippets}\n\nSummary:"

    def _gen():
        yield json.dumps({"videos": videos, "search_id": sid}) + "\n---\n"
        full = []
        for chunk in llm(prompt, system=system, stream=True):
            full.append(chunk)
            yield chunk
        _save_report(sid, "".join(full))
        try:
            drv = get_neo4j_driver()
            init_graph_schema(drv)
            _save_resources_to_graph(drv, topic, results, videos, search_type="fullResearch")
            drv.close()
        except Exception:
            pass

    return StreamingResponse(_gen(), media_type="text/plain")


@app.get("/roadmap")
def get_roadmap(
    topic:     str = Query(...,  description="Learning topic (natural language OK)", example="i want to learn Machine Learning"),
    level:     str = Query("beginner", description="User level from /assess", example="beginner"),
    completed: str = Query("",  description="Comma-separated completed module IDs", example="1,2"),
):
    """
    Generates a **3-level × 5-module** personalised learning roadmap.

    Levels: 🟢 Beginner (modules 1–5) · 🟡 Intermediate (6–10) · 🔴 Advanced (11–15)

    - Cleans the topic (extracts subject from natural language input)
    - Deep-searches to build context
    - Generates modules per level via LLM
    - Stores full hierarchy in Neo4j
    - Persists sources to PostgreSQL
    """
    # 0. Clean topic from natural language
    clean_topic = _clean_topic(topic)

    # 0b. Check Neo4j cache — if roadmap already built, reconstruct from graph
    try:
        _drv = get_neo4j_driver()
        with _drv.session() as _s:
            _lg_rows = _s.run(
                """
                MATCH (t:Topic {name:$topic})-[:HAS_LEVEL]->(lg:LevelGroup)
                RETURN lg.level_num AS level_num, lg.level_name AS level_name,
                       lg.emoji AS emoji
                ORDER BY lg.level_num
                """,
                topic=clean_topic,
            ).data()
            _mod_rows = _s.run(
                """
                MATCH (t:Topic {name:$topic})-[:HAS_MODULE]->(m:Module)
                OPTIONAL MATCH (m)-[:TEACHES]->(c:Concept)
                RETURN m.uid AS uid, m.title AS title, m.objective AS objective,
                       m.duration AS duration, m.order AS order,
                       collect(c.name) AS concepts
                ORDER BY m.order
                """,
                topic=clean_topic,
            ).data()
        _drv.close()
        if _lg_rows and _mod_rows:
            completed_ids  = [int(x) for x in completed.split(",") if x.strip().isdigit()]
            completed_uids = [f"{clean_topic}::{i}" for i in completed_ids]
            _drv2 = get_neo4j_driver()
            _recs = _recommend_modules(_drv2, clean_topic, completed_uids)
            _drv2.close()
            _levels_out = []
            for _lg in _lg_rows:
                _lmods = [
                    {
                        "id":        int(r["uid"].rsplit("::",1)[1]) if "::" in r["uid"] else r["order"],
                        "uid":       r["uid"],
                        "title":     r["title"],
                        "objective": r["objective"],
                        "duration_minutes": r["duration"] or 30,
                        "concepts":  r["concepts"] or [],
                        "level":     _lg["level_num"],
                    }
                    for r in _mod_rows
                    if int(r["uid"].rsplit("::",1)[1]) in range(
                        (_lg["level_num"] - 1) * 5 + 1,
                        _lg["level_num"] * 5 + 1,
                    )
                ] if _lg_rows else []
                _levels_out.append({
                    "level_num":  _lg["level_num"],
                    "level_name": _lg["level_name"],
                    "emoji":      _lg["emoji"],
                    "modules":    _lmods,
                })
            _all_mods = [m for lg in _levels_out for m in lg["modules"]]
            return {
                "topic":           clean_topic,
                "raw_topic":       topic,
                "level":           "beginner",
                "level_emoji":     _levels_out[0]["emoji"] if _levels_out else "🟢",
                "gaps":            [],
                "levels":          _levels_out,
                "modules":         _all_mods,
                "total_modules":   len(_all_mods),
                "total_duration":  sum(m["duration_minutes"] for m in _all_mods),
                "recommendations": _recs,
                "report_excerpt":  f"(cached from graph — {len(_all_mods)} modules across {len(_levels_out)} levels)",
                "_cached": True,
            }
    except Exception:
        pass  # cache miss or Neo4j unavailable — fall through to full pipeline

    # 1. Deep search on the cleaned topic
    queries    = generate_sub_queries(clean_topic)
    sources    = gather_sources(queries, clean_topic)
    top_chunks = scrape_and_rank(sources, clean_topic)
    context    = "\n\n".join(
        f"--- {title} ({url}) ---\n{chunk}" for _, chunk, title, url in top_chunks
    )
    report = llm_str(
        f"Write a brief educational overview of **{clean_topic}** using these sources:\n\n{context}\n\nOverview:",
        system="You are an expert educator. Be concise.",
    )

    # Save sources + report to PostgreSQL
    sid = _new_search(clean_topic, "roadmap")
    _save_web(sid, sources)
    _save_report(sid, report)

    # 2. Generate multi-level roadmap (3 LLM calls, one per level)
    roadmap = extract_multilevel_roadmap(clean_topic, report, user_level=level)

    # 3. Store in Neo4j & compute recommendations
    recommendations = []
    try:
        completed_ids  = [int(x) for x in completed.split(",") if x.strip().isdigit()]
        completed_uids = [f"{clean_topic}::{i}" for i in completed_ids]
        drv = get_neo4j_driver()
        init_graph_schema(drv)
        save_roadmap_to_graph(drv, roadmap)
        _save_resources_to_graph(drv, clean_topic, sources, [], search_type="roadmap")
        recommendations = _recommend_modules(drv, clean_topic, completed_uids)
        drv.close()
    except Exception as e:
        recommendations = []
        roadmap["_neo4j_error"] = str(e)

    return {
        "topic":           clean_topic,
        "raw_topic":       topic,
        "level":           roadmap["current_level"],
        "level_emoji":     roadmap["level_emoji"],
        "gaps":            roadmap["gaps"],
        "levels":          roadmap["levels"],        # NEW: grouped by level
        "modules":         roadmap["modules"],        # flat list (backward compat)
        "total_modules":   roadmap["total_modules"],
        "total_duration":  roadmap["total_duration"],
        "recommendations": recommendations,
        "report_excerpt":  report[:400],
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


@app.get("/knowledge", summary="Query open knowledge databases for a topic (Wikipedia · Wikidata · OpenAlex · arXiv · DBpedia)")
def get_open_knowledge(
    topic: str = Query(..., description="Topic to look up", example="Machine Learning"),
):
    """
    Queries 5 open knowledge databases in parallel and returns structured results:
    - **Wikipedia** — article summary
    - **Wikidata** — entity descriptions and labels
    - **OpenAlex** — open-access academic papers with abstracts
    - **arXiv** — preprints (CS, math, physics, bio)

    Results are also saved to Neo4j as Resource nodes linked to the topic.
    """
    sources = gather_open_knowledge(topic)
    # Persist to Neo4j so they show up in /resources
    try:
        drv = get_neo4j_driver()
        init_graph_schema(drv)
        _save_resources_to_graph(drv, topic, sources, [], search_type="knowledge_graph")
        drv.close()
    except Exception:
        pass
    # Group by source database
    grouped: dict[str, list] = {}
    for s in sources:
        db = s.get("source", "web")
        grouped.setdefault(db, []).append({
            "title":   s["title"],
            "url":     s["url"],
            "snippet": s.get("snippet", "")[:500],
        })
    return {
        "topic":   topic,
        "total":   len(sources),
        "sources": grouped,
    }


@app.get("/resources", summary="Query resources for a topic from Neo4j graph DB")
def get_resources(
    topic: str  = Query(..., description="Topic name", example="Machine Learning"),
    rtype: str  = Query("all", description="Filter by type: all | article | video", example="all"),
    limit: int  = Query(50, ge=1, le=200, description="Max results", example=20),
):
    """
    Returns all Resource nodes linked to the given topic in Neo4j.
    Each resource has `url`, `title`, `type` (article/video), and type-specific fields.
    """
    try:
        drv = get_neo4j_driver()
        with drv.session() as s:
            type_filter = "" if rtype == "all" else "AND res.type = $rtype"
            result = s.run(
                f"""MATCH (t:Topic {{name: $topic}})-[:HAS_RESOURCE]->(res:Resource)
                    WHERE res.url IS NOT NULL {type_filter}
                    RETURN res
                    ORDER BY res.type, res.title
                    LIMIT $limit""",
                topic=topic, rtype=rtype, limit=limit,
            )
            resources = [dict(r["res"]) for r in result]
        drv.close()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Neo4j unavailable: {e}")

    articles = [r for r in resources if r.get("type") == "article"]
    videos   = [r for r in resources if r.get("type") == "video"]
    return {
        "topic":    topic,
        "total":    len(resources),
        "articles": articles,
        "videos":   videos,
    }


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

    # Trim content so the small model isn't overwhelmed
    snippet = content[:2000]

    # ── Attempt 1: compact single-line format (easiest for small models) ──────
    prompt1 = (
        f'Create {num_questions} MCQ questions about this text. '
        f'Reply with ONLY a JSON array, no markdown:\n'
        f'[{{"question":"...?","options":["A. ...","B. ...","C. ...","D. ..."],"answer":"A","explanation":"..."}}]\n\n'
        f'Text: {snippet}'
    )
    raw = llm_str(prompt1)
    print(f"[quiz raw] {raw[:300]}")
    questions = [q for q in (_parse_json_array(raw) or []) if isinstance(q, dict)]

    # ── Attempt 2: ask one question at a time and collect ─────────────────────
    if not questions:
        questions = []
        for i in range(min(num_questions, 3)):
            p = (
                f'Write 1 multiple-choice question about: "{snippet[:500]}"\n'
                f'Reply ONLY with this JSON (no markdown):\n'
                f'{{"question":"...?","options":["A. ...","B. ...","C. ...","D. ..."],"answer":"A","explanation":"..."}}'
            )
            r = llm_str(p)
            parsed = _parse_json_array(r)
            if parsed:
                questions.extend(q for q in parsed if isinstance(q, dict))

    # ── Attempt 3: very minimal prompt ───────────────────────────────────────
    if not questions:
        p = (
            f'JSON array of 3 quiz questions about "{snippet[:300]}":\n'
            f'[{{"question":"?","options":["A. x","B. y","C. z","D. w"],"answer":"A","explanation":"x"}}]'
        )
        raw = llm_str(p)
        questions = [q for q in (_parse_json_array(raw) or []) if isinstance(q, dict)]

    # ── Fallback: extract key phrases from content and build generic Qs ───────
    if not questions:
        # Pull first sentences as "facts" and manufacture questions from them
        sentences = [s.strip() for s in re.split(r'[.!?]', content) if len(s.strip()) > 30][:num_questions]
        for i, sent in enumerate(sentences):
            questions.append({
                "question":    f"Which of the following best describes: '{sent[:80]}…'?",
                "options":     ["A. It is correct as stated", "B. It is false", "C. It is partially true", "D. It is unrelated"],
                "answer":      "A",
                "explanation": sent[:200],
            })

    # ── Final fallback if content had no usable sentences ────────────────────
    if not questions:
        questions = [{
            "question":    "What is the main topic covered in this lesson?",
            "options":     ["A. The subject described above", "B. An unrelated topic", "C. A historical event", "D. None of the above"],
            "answer":      "A",
            "explanation": "This question is based on the lesson content.",
        }]

    # Normalise all questions
    normed = []
    for i, q in enumerate(questions[:num_questions]):
        if not isinstance(q, dict):
            continue
        q.setdefault("question",    f"Question {i+1}")
        q.setdefault("options",     ["A. ?", "B. ?", "C. ?", "D. ?"])
        q.setdefault("answer",      "A")
        q.setdefault("explanation", "")
        normed.append(q)

    # normed is always a list of dicts; fall through to sentence fallback is already handled above
    return normed


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


# ── Assess helpers ────────────────────────────────────────────────────────────

def _clean_json(text: str) -> str:
    """Remove trailing commas and strip markdown fences."""
    text = re.sub(r"```(?:json)?\s*([\s\S]*?)```", r"\1", text).strip()
    return re.sub(r",\s*([}\]])", r"\1", text)


def _try_parse_list(text: str) -> list | None:
    """Attempt to parse text as a JSON list; return None on failure."""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        # LLM returned a dict — check for a nested list value
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list) and v:
                    return v
            # Single question object → wrap in list
            if {"question", "options", "answer"}.issubset(parsed):
                return [parsed]
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _recover_truncated(text: str) -> str:
    """Close unclosed braces/brackets so a truncated JSON can be parsed."""
    t = text.rstrip().rstrip(",")
    t += "}" * max(0, t.count("{") - t.count("}"))
    t += "]" * max(0, t.count("[") - t.count("]"))
    return t


def _parse_json_array(raw: str) -> list | None:
    """
    Robust JSON-array extractor for small-model output.
    Handles: markdown fences · single object · nested list · trailing commas ·
             truncated output · multiple bare objects without an array wrapper.
    """
    # Build candidate strings from most to least processed
    fenced    = re.sub(r"```(?:json)?\s*([\s\S]*?)```", r"\1", raw).strip()
    arr_block = (re.search(r"\[[\s\S]*\]", raw) or re.search(r"\[[\s\S]*\]", fenced))

    candidates = [raw.strip(), fenced]
    if arr_block:
        candidates.append(arr_block.group())

    for raw_cand in candidates:
        for text in (raw_cand, _clean_json(raw_cand)):
            result = _try_parse_list(text)
            if result:
                return result
            # Truncation recovery
            result = _try_parse_list(_clean_json(_recover_truncated(text)))
            if result:
                return result

    # Last resort: find ALL {...} objects and collect them into a list
    objects = re.findall(r"\{[^{}]*\}", raw)
    if objects:
        collected = []
        for obj_str in objects:
            result = _try_parse_list(_clean_json(obj_str))
            if result:
                collected.extend(result)
        if collected:
            return collected

    return None


def _generate_diagnostic_questions(topic: str) -> list:
    """Generate MCQ questions to gauge the learner's prior knowledge level."""
    prompt = f"""You are an expert educator. Generate 6 multiple-choice questions to assess prior knowledge of "{topic}".

2 basic (facts), 2 intermediate (concepts), 2 advanced (application).

Respond with ONLY a JSON array, no other text, no markdown:
[{{"id":1,"level":"basic","question":"...?","options":["A. ...","B. ...","C. ...","D. ..."],"answer":"A","explanation":"..."}},...]

Topic: {topic}"""

    raw = llm_str(prompt)
    print(f"[assess raw] {raw[:300]}")   # visible in uvicorn logs for debugging

    questions = _parse_json_array(raw)

    # Validate each question has the required keys; drop malformed ones
    required = {"question", "options", "answer"}
    if questions:
        questions = [q for q in questions if isinstance(q, dict) and required.issubset(q)]

    if questions:
        # Normalise: ensure id and level exist
        levels = ["basic", "basic", "intermediate", "intermediate", "advanced", "advanced"]
        for i, q in enumerate(questions):
            q.setdefault("id", i + 1)
            q.setdefault("level", levels[i] if i < len(levels) else "intermediate")
            q.setdefault("explanation", "")
        return questions

    # ── Hardcoded fallback so the endpoint never 500s ─────────────────────────
    print(f"[assess] LLM parse failed — using fallback questions for '{topic}'")
    return [
        {"id": 1, "level": "basic",
         "question": f"Which of the following best describes {topic}?",
         "options": ["A. A programming language", "B. A field of study or practice",
                     "C. A physical device",    "D. A mathematical theorem"],
         "answer": "B", "explanation": "This is a general orientation question."},
        {"id": 2, "level": "basic",
         "question": f"Have you studied {topic} before?",
         "options": ["A. Never", "B. A little (self-study)",
                     "C. Formally (course/book)", "D. I use it professionally"],
         "answer": "A", "explanation": "Self-reported experience check."},
        {"id": 3, "level": "intermediate",
         "question": f"What is a key challenge in {topic}?",
         "options": ["A. Lack of data",        "B. Complexity and abstraction",
                     "C. Hardware limitations", "D. Language barriers"],
         "answer": "B", "explanation": "Most fields involve managing complexity."},
        {"id": 4, "level": "intermediate",
         "question": f"Which skill is most useful when learning {topic}?",
         "options": ["A. Memorisation",   "B. Critical thinking",
                     "C. Speed reading",  "D. Drawing"],
         "answer": "B", "explanation": "Critical thinking aids deep understanding."},
        {"id": 5, "level": "advanced",
         "question": f"How would you apply {topic} to solve a real-world problem?",
         "options": ["A. By reading a textbook",        "B. By following a fixed recipe",
                     "C. By adapting principles to context", "D. By avoiding uncertainty"],
         "answer": "C", "explanation": "Application requires contextual adaptation."},
        {"id": 6, "level": "advanced",
         "question": f"What distinguishes an expert in {topic} from a beginner?",
         "options": ["A. Knowing more vocabulary",   "B. Having more tools",
                     "C. Deeper conceptual models",  "D. Working faster"],
         "answer": "C", "explanation": "Experts build richer mental models."},
    ]


def _evaluate_level(topic: str, questions: list, answers: list) -> dict:
    """
    Score the diagnostic answers and determine the learner's level.
    Returns {score, level, level_emoji, feedback}.
    """
    qa_pairs = []
    for i, q in enumerate(questions):
        user_ans = answers[i].strip().upper() if i < len(answers) else "?"
        correct  = q["answer"].strip().upper()
        qa_pairs.append(
            f"Q{i+1} [{q['level']}]: {q['question']}\n"
            f"  Correct: {correct} | User: {user_ans} | {'✅' if user_ans == correct else '❌'}"
        )

    prompt = f"""You assessed a student on "{topic}". Here are their results:

{chr(10).join(qa_pairs)}

Based on these results, determine:
1. Overall score (0-100)
2. Level: "beginner" if score < 40, "intermediate" if 40-70, "advanced" if > 70
3. Short personalised feedback (2 sentences)

Return ONLY valid JSON:
{{
  "score": 65,
  "level": "intermediate",
  "level_emoji": "🟡",
  "feedback": "You have a solid grasp of the basics but some advanced concepts need work."
}}"""

    raw   = llm_str(prompt)
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # Fallback: compute score manually
    correct = sum(
        1 for i, q in enumerate(questions)
        if i < len(answers) and answers[i].strip().upper() == q["answer"].strip().upper()
    )
    score = round(correct / len(questions) * 100) if questions else 0
    if score < 40:
        level, emoji = "beginner", "🟢"
    elif score <= 70:
        level, emoji = "intermediate", "🟡"
    else:
        level, emoji = "advanced", "🔴"
    return {"score": score, "level": level, "level_emoji": emoji,
            "feedback": f"You answered {correct}/{len(questions)} correctly."}


# ── Lesson helpers ─────────────────────────────────────────────────────────────

def _rank_resources(module_title: str, sources: list, videos: list) -> list:
    """
    Ask the LLM to pick the top 3 best resources (web + video) for this module
    and return them with a short reason.
    """
    web_lines = "\n".join(
        f"[WEB {i+1}] {s['title']} — {s['url']}\n  {s['snippet'][:120]}"
        for i, s in enumerate(sources[:10])
    )
    vid_lines = "\n".join(
        f"[VID {i+1}] {v['title']} ({v['channel']}, {v['duration']}) — {v['url']}"
        for i, v in enumerate(videos[:6])
    )
    prompt = f"""You are a curriculum advisor. A student is learning about "{module_title}".
Below are available web articles and videos. Pick the TOP 3 best resources for a learner.

WEB ARTICLES:
{web_lines}

VIDEOS:
{vid_lines}

Return ONLY valid JSON — no markdown fences:
[
  {{
    "type": "web",
    "title": "...",
    "url": "...",
    "reason": "One sentence why this is the best resource."
  }}
]

Rules:
• Exactly 3 items
• Mix web and video if both are good
• "type" is "web" or "video"
• Prefer beginner-friendly, comprehensive, trusted sources"""

    raw    = llm_str(prompt)
    parsed = _parse_json_array(raw)
    if parsed:
        return parsed
    # Fallback: return first web + first video
    fallback = []
    if sources:
        fallback.append({"type": "web",   "title": sources[0]["title"], "url": sources[0]["url"], "reason": "Top web result."})
    if videos:
        fallback.append({"type": "video", "title": videos[0]["title"],  "url": videos[0]["url"],  "reason": "Top video result."})
    return fallback


def _generate_lesson(module_title: str, module_objective: str,
                     concepts: list, topic: str, level: str) -> dict:
    """
    Deep-search the module topic, gather web sources + YouTube videos,
    synthesise an educational lesson, and return everything together.

    Returns:
        {content, sources, videos, recommended}
    """
    query   = to_search_query(f"{module_title} {' '.join(concepts)}")
    sources = gather_sources(generate_sub_queries(query), topic or module_title)
    chunks  = scrape_and_rank(sources, module_title)

    # YouTube videos for this module
    videos = search_youtube(f"{module_title} tutorial explained")

    context = "\n\n".join(
        f"--- {title} ({url}) ---\n{chunk}" for _, chunk, title, url in chunks
    )
    system = (
        "You are an expert educator capable of teaching any subject. "
        f"The student is at {level} level. Adapt depth and language accordingly. "
        "Structure the lesson in markdown:\n"
        "1. **Introduction** — what this topic is and why it matters\n"
        "2. **Key Concepts** — core ideas with simple language and analogies\n"
        "3. **Worked Examples** — concrete examples, formulas, or step-by-step illustrations\n"
        "4. **Common Misconceptions** — pitfalls to avoid\n"
        "5. **Summary** — bullet-point recap"
    )
    prompt = (
        f"Teach: **{module_title}**\n"
        f"Objective: {module_objective}\n"
        f"Key concepts: {', '.join(concepts)}\n\n"
        f"Sources:\n{context}\n\nWrite the lesson:"
    )
    content = llm_str(prompt, system=system)

    # Rank best resources
    recommended = _rank_resources(module_title, sources, videos)

    return {
        "content":     content,
        "sources":     [{"title": s["title"], "url": s["url"], "snippet": s["snippet"]} for s in sources],
        "videos":      videos,
        "recommended": recommended,
    }


def _get_module_from_graph(driver, module_uid: str) -> dict:
    """Fetch module metadata from Neo4j."""
    with driver.session() as s:
        result = s.run(
            """
            MATCH (m:Module {uid: $uid})
            OPTIONAL MATCH (m)-[:TEACHES]->(c:Concept)
            WITH m, collect(c.name) AS concepts
            RETURN m.uid AS uid, m.title AS title, m.objective AS objective,
                   m.duration_minutes AS duration, m.order AS order,
                   m.topic AS topic, concepts
            """,
            uid=module_uid,
        )
        row = result.single()
    if not row:
        raise HTTPException(status_code=404, detail=f"Module '{module_uid}' not found in graph.")
    return dict(row)


def _get_completed_uids(session_id: int) -> list:
    """Return list of completed module UIDs for a session."""
    schema = _progress_schema(session_id)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f'SELECT module_uid FROM "{schema}".session_progress WHERE session_id=%s ORDER BY completed_at',
                (session_id,),
            )
            return [r[0] for r in cur.fetchall()]


# ── Pydantic models ────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    username:  str
    email:     str
    full_name: Optional[str] = None
    level:     str = "beginner"   # beginner | intermediate | advanced


class UserUpdate(BaseModel):
    username:  Optional[str] = None
    email:     Optional[str] = None
    full_name: Optional[str] = None
    level:     Optional[str] = None


class UserOut(BaseModel):
    id:         int
    username:   str
    email:      str
    full_name:  Optional[str]
    level:      str
    created_at: str
    updated_at: str


class AssessEvaluateRequest(BaseModel):
    topic:     str
    questions: list   # the questions list returned by GET /assess
    answers:   list   # list of answer letters, e.g. ["A", "C", "B", "D", "A", "B"]


class LessonRequest(BaseModel):
    pass  # query params only


class NextRequest(BaseModel):
    session_id:          int
    completed_module_uid: str
    quiz_score:          float = 0.0   # 0–100, score on the just-finished quiz
    num_quiz_questions:  int  = 5


# ── /assess ────────────────────────────────────────────────────────────────────

@app.get(
    "/assess",
    summary="Step 2 — Diagnostic quiz to determine learner level",
    openapi_extra={
        "requestBody": None,
        "parameters": [
            {
                "name": "topic",
                "in": "query",
                "required": True,
                "schema": {"type": "string", "example": "Machine Learning"},
                "description": "The subject to assess prior knowledge for",
            }
        ],
    },
)
def assess(
    topic: str = Query(..., description="Subject to assess", example="Machine Learning"),
):
    """
    **Step 2 of the learning flow.**

    Generates 6 diagnostic multiple-choice questions (2 basic · 2 intermediate · 2 advanced)
    to gauge the learner's prior knowledge before building the roadmap.

    Pass the returned `questions` list and the learner's `answers` to `POST /assess/evaluate`
    to get their level (beginner / intermediate / advanced).
    """
    questions = _generate_diagnostic_questions(topic)
    return {
        "topic":        topic,
        "num_questions": len(questions),
        "instructions": "Answer each question then POST to /assess/evaluate with your answers.",
        "questions":    questions,
    }


@app.post(
    "/assess/evaluate",
    summary="Step 2b — Score diagnostic answers and return learner level",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "ml_beginner": {
                            "summary": "Machine Learning — mostly wrong answers (beginner)",
                            "value": {
                                "topic": "Machine Learning",
                                "questions": [],
                                "answers": ["A", "B", "A", "C", "A", "B"],
                            },
                        },
                    }
                }
            }
        }
    },
)
def assess_evaluate(req: AssessEvaluateRequest):
    """
    **Step 2b of the learning flow.**

    Scores the learner's answers to the diagnostic quiz returned by `GET /assess`
    and determines their level: **beginner 🟢 · intermediate 🟡 · advanced 🔴**.

    Use the returned `level` when calling `GET /roadmap` to get a tailored learning path.
    Optionally create a learning session by calling `POST /session/start` with topic + level.
    """
    if not req.questions:
        raise HTTPException(status_code=422, detail="'questions' list is required.")
    if not req.answers:
        raise HTTPException(status_code=422, detail="'answers' list is required.")

    result = _evaluate_level(req.topic, req.questions, req.answers)
    return {
        "topic":       req.topic,
        "score":       result["score"],
        "level":       result["level"],
        "level_emoji": result["level_emoji"],
        "feedback":    result["feedback"],
        "next_step":   "Call GET /roadmap?topic=... to build your personalised learning path.",
    }


# ── /users ─────────────────────────────────────────────────────────────────────

@app.post(
    "/users",
    summary="Create a new user",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "alice": {
                            "summary": "Create Alice",
                            "value": {"username": "alice", "email": "alice@example.com", "full_name": "Alice Dupont", "level": "beginner"},
                        },
                        "bob": {
                            "summary": "Create Bob (intermediate)",
                            "value": {"username": "bob", "email": "bob@example.com", "full_name": "Bob Martin", "level": "intermediate"},
                        },
                    }
                }
            }
        }
    },
)
def create_user(req: UserCreate):
    """Create a user manually. `username` and `email` must be unique."""
    try:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """INSERT INTO users (username, email, full_name, level)
                       VALUES (%s, %s, %s, %s) RETURNING *""",
                    (req.username, req.email, req.full_name, req.level),
                )
                row = dict(cur.fetchone())
            conn.commit()
    except psycopg2.errors.UniqueViolation:
        raise HTTPException(status_code=409, detail="username or email already exists.")
    _create_user_schema(row["id"])
    row["created_at"] = str(row["created_at"])
    row["updated_at"] = str(row["updated_at"])
    return row


@app.get("/users", summary="List all users")
def list_users(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM users ORDER BY id LIMIT %s OFFSET %s",
                (limit, offset),
            )
            rows = cur.fetchall()
    result = []
    for r in rows:
        r = dict(r)
        r["created_at"] = str(r["created_at"])
        r["updated_at"] = str(r["updated_at"])
        result.append(r)
    return {"total": len(result), "users": result}


@app.get("/users/{user_id}", summary="Get a user by ID")
def get_user(user_id: int = Path(..., description="User ID", example=1)):
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
            row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found.")
    row = dict(row)
    row["created_at"] = str(row["created_at"])
    row["updated_at"] = str(row["updated_at"])
    return row


@app.put(
    "/users/{user_id}",
    summary="Update a user",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "promote": {
                            "summary": "Promote to intermediate",
                            "value": {"level": "intermediate"},
                        },
                        "rename": {
                            "summary": "Update full name",
                            "value": {"full_name": "Alice Smith"},
                        },
                    }
                }
            }
        }
    },
)
def update_user(
    req: UserUpdate,
    user_id: int = Path(..., description="User ID", example=1),
):
    """Partial update — only supply the fields you want to change."""
    fields = {k: v for k, v in req.model_dump().items() if v is not None}
    if not fields:
        raise HTTPException(status_code=422, detail="No fields to update.")
    set_clause = ", ".join(f"{k} = %s" for k in fields)
    values = list(fields.values()) + [user_id]
    try:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    f"UPDATE users SET {set_clause}, updated_at = NOW() WHERE id = %s RETURNING *",
                    values,
                )
                row = cur.fetchone()
            conn.commit()
    except psycopg2.errors.UniqueViolation:
        raise HTTPException(status_code=409, detail="username or email already exists.")
    if not row:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found.")
    row = dict(row)
    row["created_at"] = str(row["created_at"])
    row["updated_at"] = str(row["updated_at"])
    return row


@app.delete("/users/{user_id}", summary="Delete a user")
def delete_user(user_id: int = Path(..., description="User ID", example=1)):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM users WHERE id = %s RETURNING id", (user_id,))
            deleted = cur.fetchone()
        conn.commit()
    if not deleted:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found.")
    return {"deleted": True, "user_id": user_id}


@app.get("/users/{user_id}/sessions", summary="Get all learning sessions for a user")
def get_user_sessions(user_id: int = Path(..., description="User ID", example=1)):
    """Returns all learning sessions linked to the given user, newest first."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Verify user exists
            cur.execute("SELECT id FROM users WHERE id = %s", (user_id,))
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail=f"User {user_id} not found.")
            schema = _user_schema(user_id)
            cur.execute(
                f"""SELECT ls.*, COUNT(sp.id) AS modules_completed
                   FROM learning_sessions ls
                   LEFT JOIN "{schema}".session_progress sp ON sp.session_id = ls.id
                   WHERE ls.user_id = %s
                   GROUP BY ls.id
                   ORDER BY ls.created_at DESC""",
                (user_id,),
            )
            rows = cur.fetchall()
    result = []
    for r in rows:
        r = dict(r)
        r["created_at"] = str(r["created_at"])
        result.append(r)
    return {"user_id": user_id, "total": len(result), "sessions": result}


@app.get(
    "/users/{user_id}/history",
    summary="Full learning history for a user — sessions + per-module progress",
)
def get_user_history(user_id: int = Path(..., description="User ID", example=1)):
    """
    Returns every learning session for the user with the list of completed modules,
    quiz scores, and aggregate stats per session.  Useful for building a per-user
    learning history table.
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Verify user exists
            cur.execute("SELECT id, username, level FROM users WHERE id = %s", (user_id,))
            user_row = cur.fetchone()
            if not user_row:
                raise HTTPException(status_code=404, detail=f"User {user_id} not found.")

            # All sessions for this user
            cur.execute(
                """SELECT id, topic, level, level_emoji, created_at
                   FROM learning_sessions
                   WHERE user_id = %s
                   ORDER BY created_at DESC""",
                (user_id,),
            )
            sessions = cur.fetchall()

            history = []
            for sess in sessions:
                sess = dict(sess)
                sess_id = sess["id"]

                # Modules completed in this session
                sp_schema = _progress_schema(sess_id)
                cur.execute(
                    f"""SELECT module_uid, module_title, quiz_score,
                              completed_at, ROW_NUMBER() OVER (ORDER BY completed_at) AS step
                       FROM "{sp_schema}".session_progress
                       WHERE session_id = %s
                       ORDER BY completed_at""",
                    (sess_id,),
                )
                modules = [dict(r) for r in cur.fetchall()]
                for m in modules:
                    m["completed_at"] = str(m["completed_at"])
                    m["quiz_score"]   = float(m["quiz_score"]) if m["quiz_score"] is not None else None

                scores = [m["quiz_score"] for m in modules if m["quiz_score"] is not None]
                history.append({
                    "session_id":        sess_id,
                    "topic":             sess["topic"],
                    "level":             sess["level"],
                    "level_emoji":       sess["level_emoji"],
                    "started_at":        str(sess["created_at"]),
                    "modules_completed": len(modules),
                    "avg_score":         round(sum(scores) / len(scores), 1) if scores else None,
                    "best_score":        max(scores) if scores else None,
                    "modules":           modules,
                })

    return {
        "user_id":  user_id,
        "username": user_row["username"],
        "level":    user_row["level"],
        "total_sessions": len(history),
        "history":  history,
    }


# ── /session ───────────────────────────────────────────────────────────────────

@app.post(
    "/session/start",
    summary="Create a new learning session (persists topic + level to DB)",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "ml_beginner": {
                            "summary": "Start ML session at beginner level",
                            "value": {"topic": "Machine Learning", "level": "beginner", "level_emoji": "🟢"},
                        },
                        "physics_intermediate": {
                            "summary": "Start Physics session at intermediate level",
                            "value": {"topic": "Electromagnetism", "level": "intermediate", "level_emoji": "🟡"},
                        },
                    }
                }
            }
        }
    },
)
def session_start(
    topic:       str = Query(..., description="Learning topic",  example="Machine Learning"),
    level:       str = Query("beginner", description="beginner | intermediate | advanced", example="beginner"),
    level_emoji: str = Query("🟢",       description="Level emoji", example="🟢"),
    user_id:     Optional[int] = Query(None, description="Optional user ID to link session to a user", example=1),
):
    """
    Creates a persistent learning session in PostgreSQL.
    Cleans the topic (extracts subject from natural language) before storing.
    Returns a `session_id` used by `GET /lesson` and `POST /next`.
    """
    clean = _clean_topic(topic)   # "i want to learn ML" → "Machine Learning"
    if user_id is not None:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM users WHERE id = %s", (user_id,))
                if not cur.fetchone():
                    raise HTTPException(status_code=404, detail=f"User {user_id} not found.")
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO learning_sessions (topic, level, level_emoji, user_id) VALUES (%s,%s,%s,%s) RETURNING id",
                (clean, level, level_emoji, user_id),
            )
            sid = cur.fetchone()[0]
        conn.commit()
    return {
        "session_id":  sid,
        "user_id":     user_id,
        "topic":       clean,
        "raw_topic":   topic,
        "level":       level,
        "level_emoji": level_emoji,
        "next_step":   f"Call GET /lesson?topic={clean}&module_id=1&session_id={sid} to start your first lesson.",
    }


# ── /lesson ────────────────────────────────────────────────────────────────────

@app.get(
    "/lesson",
    summary="Step 7 — Generate lesson content for a roadmap module",
)
def get_lesson(
    topic:      str = Query(...,  description="Learning topic",   example="Machine Learning"),
    module_id:  int = Query(...,  description="Module number from the roadmap (e.g. 1)", example=1),
    session_id: Optional[int] = Query(None, description="Session ID (from /session/start) to persist lesson", example=1),
):
    """
    **Step 7 of the learning flow.**

    Fetches the module from Neo4j, runs a deep educational search on its concepts,
    and returns a fully structured lesson (intro · concepts · examples · misconceptions · summary).

    If `session_id` is provided the lesson is persisted to PostgreSQL so it can
    be retrieved later and used by `POST /quiz` or `POST /next`.
    """
    module_uid = f"{topic}::{module_id}"

    # 0. Check lesson cache (per-user schema)
    if session_id:
        _cached_content = _cached_lesson(session_id, module_uid)
        if _cached_content:
            # Still need module metadata from Neo4j for the response shape
            try:
                _drv = get_neo4j_driver()
                _mod = _get_module_from_graph(_drv, module_uid)
                _drv.close()
            except Exception:
                _mod = {"title": module_uid, "objective": "", "concepts": [], "duration": 30}
            return {
                "module_uid":   module_uid,
                "module_title": _mod["title"],
                "objective":    _mod.get("objective", ""),
                "concepts":     _mod.get("concepts", []),
                "duration_min": _mod.get("duration", 30),
                "level":        "beginner",
                "lesson_id":    None,
                "content":      _cached_content,
                "sources":      [],
                "videos":       [],
                "recommended":  [],
                "next_step":    "Study this lesson then POST to /quiz with the content to test yourself, or POST to /next to advance.",
                "_cached":      True,
            }

    # 1. Get module metadata from graph
    try:
        drv    = get_neo4j_driver()
        module = _get_module_from_graph(drv, module_uid)
        drv.close()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Neo4j unavailable: {e}")

    # 2. Determine learner level for depth calibration
    level = "beginner"
    if session_id:
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT level FROM learning_sessions WHERE id=%s", (session_id,))
                    row = cur.fetchone()
                    if row:
                        level = row[0]
        except Exception:
            pass

    # 3. Generate lesson via deep search (returns content + sources + videos + recommended)
    lesson = _generate_lesson(
        module_title=module["title"],
        module_objective=module["objective"],
        concepts=module["concepts"],
        topic=topic,
        level=level,
    )

    # 4. Persist lesson if session provided
    lesson_id = None
    if session_id:
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    schema = _progress_schema(session_id)
                    cur.execute(
                        f'INSERT INTO "{schema}".lessons (session_id, module_uid, module_title, content) VALUES (%s,%s,%s,%s) RETURNING id',
                        (session_id, module_uid, module["title"], lesson["content"]),
                    )
                    lesson_id = cur.fetchone()[0]
                conn.commit()
        except Exception:
            pass

    return {
        "module_uid":   module_uid,
        "module_title": module["title"],
        "objective":    module["objective"],
        "concepts":     module["concepts"],
        "duration_min": module["duration"],
        "level":        level,
        "lesson_id":    lesson_id,
        "content":      lesson["content"],
        "sources":      lesson["sources"],
        "videos":       lesson["videos"],
        "recommended":  lesson["recommended"],
        "next_step":    "Study this lesson then POST to /quiz with the content to test yourself, or POST to /next to advance.",
    }


# ── /next ──────────────────────────────────────────────────────────────────────

@app.post(
    "/next",
    summary="Step 9 — Mark module done, get next lesson + quiz (the learning loop)",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "advance_after_quiz": {
                            "summary": "Finished module 1 with 80% score, get module 2",
                            "value": {
                                "session_id": 1,
                                "completed_module_uid": "Machine Learning::1",
                                "quiz_score": 80.0,
                                "num_quiz_questions": 5,
                            },
                        },
                        "low_score": {
                            "summary": "Finished module 2 with 40% score (still advances)",
                            "value": {
                                "session_id": 1,
                                "completed_module_uid": "Machine Learning::2",
                                "quiz_score": 40.0,
                                "num_quiz_questions": 5,
                            },
                        },
                    }
                }
            }
        }
    },
)
def next_module(req: NextRequest):
    """
    **Step 9 of the learning flow — the loop engine.**

    1. Saves the completed module + quiz score to `session_progress` in PostgreSQL.
    2. Queries Neo4j for the next best module (prerequisites satisfied, most concepts covered).
    3. Generates the lesson content for that module via deep search.
    4. Generates a quiz on that lesson.
    5. Returns everything in one response so the frontend can display lesson → quiz → next.

    When `next_module` is `null` the learner has completed the entire roadmap 🎉.
    """
    # 1. Persist progress
    try:
        schema = _progress_schema(req.session_id)
        with get_conn() as conn:
            with conn.cursor() as cur:
                # Get module title for readability
                cur.execute(
                    f'SELECT module_title FROM "{schema}".lessons WHERE session_id=%s AND module_uid=%s ORDER BY id DESC LIMIT 1',
                    (req.session_id, req.completed_module_uid),
                )
                row = cur.fetchone()
                module_title = row[0] if row else req.completed_module_uid

                cur.execute(
                    f'INSERT INTO "{schema}".session_progress (session_id, module_uid, module_title, quiz_score) VALUES (%s,%s,%s,%s)',
                    (req.session_id, req.completed_module_uid, module_title, req.quiz_score),
                )
            conn.commit()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"DB unavailable: {e}")

    # 2. Get session info
    try:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT topic, level FROM learning_sessions WHERE id=%s", (req.session_id,))
                session = cur.fetchone()
        if not session:
            raise HTTPException(status_code=404, detail=f"session_id={req.session_id} not found.")
        topic = session["topic"]
        level = session["level"]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"DB unavailable: {e}")

    # 3. Get all completed module UIDs for this session
    completed_uids = _get_completed_uids(req.session_id)

    # 4. Find next module — try Neo4j first, fall back to sequential order
    next_rec  = None
    next_mod  = None
    neo4j_ok  = False
    try:
        drv  = get_neo4j_driver()
        recs = _recommend_modules(drv, topic, completed_uids)
        if recs:
            next_rec = recs[0]
            next_mod = _get_module_from_graph(drv, next_rec["uid"])
        neo4j_ok = True
        drv.close()
    except Exception:
        pass   # Neo4j unavailable or module not found — use sequential fallback

    # Sequential fallback: extract completed module numbers and pick the next integer
    if next_mod is None:
        done_nums = set()
        for uid in completed_uids:
            parts = uid.rsplit("::", 1)
            if len(parts) == 2 and parts[1].isdigit():
                done_nums.add(int(parts[1]))
        # Also include the module just completed (might not be persisted yet)
        cur_parts = req.completed_module_uid.rsplit("::", 1)
        if len(cur_parts) == 2 and cur_parts[1].isdigit():
            done_nums.add(int(cur_parts[1]))

        next_id = min((n for n in range(1, 26) if n not in done_nums), default=None)

        if next_id is None:
            # All 25 possible modules done → completed
            avg_score = 0.0
            try:
                with get_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            f'SELECT AVG(quiz_score) FROM "{schema}".session_progress WHERE session_id=%s',
                            (req.session_id,),
                        )
                        row = cur.fetchone()
                        avg_score = round(row[0] or 0, 1)
            except Exception:
                pass
            return {
                "completed": True,
                "message":   f"🎉 Congratulations! You finished all modules in '{topic}'.",
                "avg_score": avg_score,
                "next_module": None, "lesson": None, "quiz": None,
            }

        next_uid = f"{topic}::{next_id}"
        # Try to get from Neo4j; build a minimal stub if unavailable
        try:
            drv      = get_neo4j_driver()
            next_mod = _get_module_from_graph(drv, next_uid)
            drv.close()
        except Exception:
            next_mod = {
                "uid":       next_uid,
                "title":     f"{topic} — Module {next_id}",
                "objective": f"Continue learning {topic}",
                "concepts":  [],
                "duration":  60,
                "topic":     topic,
            }
        next_rec = {"uid": next_uid}

    # Check if truly completed via Neo4j (no recs and neo4j was OK)
    if neo4j_ok and not recs and next_mod is None:
        avg_score = 0.0
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f'SELECT AVG(quiz_score) FROM "{schema}".session_progress WHERE session_id=%s',
                        (req.session_id,),
                    )
                    row = cur.fetchone()
                    avg_score = round(row[0] or 0, 1)
        except Exception:
            pass
        return {
            "completed": True,
            "message":   f"🎉 Congratulations! You finished all modules in '{topic}'.",
            "avg_score": avg_score,
            "next_module": None, "lesson": None, "quiz": None,
        }

    lesson = _generate_lesson(
        module_title=next_mod["title"],
        module_objective=next_mod["objective"],
        concepts=next_mod["concepts"],
        topic=topic,
        level=level,
    )

    # 6. Persist lesson
    lesson_id = None
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f'INSERT INTO "{schema}".lessons (session_id, module_uid, module_title, content) VALUES (%s,%s,%s,%s) RETURNING id',
                    (req.session_id, next_rec["uid"], next_mod["title"], lesson["content"]),
                )
                lesson_id = cur.fetchone()[0]
            conn.commit()
    except Exception:
        pass

    # 7. Generate quiz on new lesson
    try:
        quiz = _build_quiz(lesson["content"], req.num_quiz_questions)
    except HTTPException:
        quiz = []

    return {
        "completed":       False,
        "previous_module": req.completed_module_uid,
        "previous_score":  req.quiz_score,
        "next_module": {
            "uid":          next_rec["uid"],
            "title":        next_mod["title"],
            "objective":    next_mod["objective"],
            "concepts":     next_mod["concepts"],
            "duration_min": next_mod["duration"],
            "lesson_id":    lesson_id,
            "content":      lesson["content"],
            "sources":      lesson["sources"],
            "videos":       lesson["videos"],
            "recommended":  lesson["recommended"],
        },
        "quiz": {
            "num_questions": len(quiz),
            "questions":     quiz,
        },
        "remaining_modules": max(0, len(recs) - 1) if neo4j_ok and recs else None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LEARNING-PATH SCHEMA & ENDPOINTS  (DbSchema.md · endpoint.md)
#
# New tables : students · courses · assessments · assessment_answers ·
#              knowledge_gaps · roadmap_modules · topic_mastery ·
#              concept_resources · learning_progress
#
# New routes : POST /students
#              POST /students/{id}/verify
#              POST /deep-search          ← topic → per-URL summary → KG + DB
#              POST /generate-quiz        ← topic(s) → MCQ using KG context
#              POST /find-gaps            ← MCQ + answers → gaps persisted to DB
#              GET  /recommender          ← gaps → ranked resources
# ═══════════════════════════════════════════════════════════════════════════════


# ── Schema initialisation ─────────────────────────────────────────────────────

def init_learning_schema():
    """Create all learning-path tables (idempotent — safe on every startup)."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS students (
                    id                SERIAL PRIMARY KEY,
                    username          TEXT UNIQUE NOT NULL,
                    email             TEXT UNIQUE NOT NULL,
                    password_hash     TEXT NOT NULL,
                    verified          INTEGER DEFAULT 0,
                    verification_code TEXT,
                    code_expiry       TIMESTAMPTZ,
                    created_at        TIMESTAMPTZ DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS courses (
                    id           SERIAL PRIMARY KEY,
                    student_id   INT REFERENCES students(id) ON DELETE CASCADE,
                    name         TEXT NOT NULL,
                    description  TEXT,
                    playlist_url TEXT,
                    created_at   TIMESTAMPTZ DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS assessments (
                    id         SERIAL PRIMARY KEY,
                    student_id INT REFERENCES students(id) ON DELETE CASCADE,
                    course_id  INT REFERENCES courses(id) ON DELETE SET NULL,
                    level      TEXT,
                    score      INTEGER,
                    taken_at   TIMESTAMPTZ DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS assessment_answers (
                    id             SERIAL PRIMARY KEY,
                    assessment_id  INT REFERENCES assessments(id) ON DELETE CASCADE,
                    question       TEXT,
                    concept        TEXT,
                    student_answer TEXT,
                    correct_answer TEXT,
                    is_correct     INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS knowledge_gaps (
                    id            SERIAL PRIMARY KEY,
                    student_id    INT REFERENCES students(id) ON DELETE CASCADE,
                    course_id     INT REFERENCES courses(id) ON DELETE SET NULL,
                    topic_name    TEXT NOT NULL,
                    severity      TEXT DEFAULT 'medium',
                    source        TEXT DEFAULT 'assessment',
                    identified_at TIMESTAMPTZ DEFAULT NOW(),
                    resolved_at   TIMESTAMPTZ
                );

                CREATE TABLE IF NOT EXISTS roadmap_modules (
                    id            SERIAL PRIMARY KEY,
                    student_id    INT REFERENCES students(id) ON DELETE CASCADE,
                    course_id     INT REFERENCES courses(id) ON DELETE SET NULL,
                    module_number INTEGER,
                    title         TEXT,
                    objective     TEXT,
                    concepts_json TEXT,
                    duration      TEXT,
                    status        TEXT DEFAULT 'pending'
                );

                CREATE TABLE IF NOT EXISTS topic_mastery (
                    id            SERIAL PRIMARY KEY,
                    student_id    INT REFERENCES students(id) ON DELETE CASCADE,
                    course_id     INT REFERENCES courses(id) ON DELETE SET NULL,
                    topic_name    TEXT NOT NULL,
                    mastery_score FLOAT DEFAULT 0.0,
                    attempt_count INTEGER DEFAULT 0,
                    pass_count    INTEGER DEFAULT 0,
                    last_updated  TIMESTAMPTZ DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS concept_resources (
                    id            SERIAL PRIMARY KEY,
                    student_id    INT REFERENCES students(id) ON DELETE CASCADE,
                    course_id     INT REFERENCES courses(id) ON DELETE SET NULL,
                    concept_name  TEXT NOT NULL,
                    resource_type TEXT,
                    url           TEXT,
                    title         TEXT,
                    channel       TEXT,
                    duration      TEXT,
                    views         BIGINT,
                    summary       TEXT,
                    metadata_json TEXT,
                    fetched_at    TIMESTAMPTZ DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS learning_progress (
                    id             SERIAL PRIMARY KEY,
                    student_id     INT REFERENCES students(id) ON DELETE CASCADE,
                    course_id      INT REFERENCES courses(id) ON DELETE SET NULL,
                    module_id      INT REFERENCES roadmap_modules(id) ON DELETE SET NULL,
                    video_id       TEXT,
                    title          TEXT,
                    score          INTEGER,
                    total          INTEGER,
                    passed         INTEGER DEFAULT 0,
                    attempt_number INTEGER DEFAULT 1,
                    timestamp      TIMESTAMPTZ DEFAULT NOW()
                );
            """)
        conn.commit()


# Run once at import time (same pattern as init_db above)
try:
    init_learning_schema()
except Exception as _e:
    print(f"[init_learning_schema] {_e}")


# ── Auth helpers ──────────────────────────────────────────────────────────────

def _hash_password(password: str) -> str:
    salt = _secrets.token_hex(16)
    h    = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
    return f"{salt}:{h}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt, h = stored.split(":", 1)
        return hashlib.sha256(f"{salt}{password}".encode()).hexdigest() == h
    except Exception:
        return False


# ── Concept-resource helpers ──────────────────────────────────────────────────

def _get_cached_concept_resources(concept_name: str) -> list[dict]:
    """Return all stored resources for a concept, newest first."""
    try:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """SELECT id, resource_type, url, title, channel, duration,
                              views, summary, metadata_json, fetched_at
                       FROM concept_resources
                       WHERE LOWER(concept_name) = LOWER(%s)
                       ORDER BY fetched_at DESC""",
                    (concept_name,),
                )
                rows = cur.fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _save_concept_resource(
    concept_name: str,
    resource_type: str,
    url: str,
    title: str,
    summary: str,
    metadata: dict,
    student_id: Optional[int] = None,
    course_id:  Optional[int] = None,
    channel:    str           = "",
    duration:   str           = "",
    views:      Optional[int] = None,
) -> None:
    """Upsert a concept resource row (skip duplicates by url + concept_name)."""
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO concept_resources
                       (student_id, course_id, concept_name, resource_type,
                        url, title, channel, duration, views, summary, metadata_json)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT DO NOTHING""",
                    (
                        student_id, course_id, concept_name, resource_type,
                        url, title, channel, duration, views,
                        summary, json.dumps(metadata),
                    ),
                )
            conn.commit()
    except Exception as e:
        print(f"[concept_resources] save error: {e}")


def _summarise_url(url: str, title: str, concept: str, fallback: str = "") -> str:
    """
    Fetch the page and ask the LLM for a 3–5 sentence educational summary
    focused on what this resource teaches about `concept`.
    Falls back to summarising the snippet/description if the page is unreachable.
    """
    page_text = fetch_page(url)
    source    = (page_text or fallback)[:3000]
    if not source:
        return ""
    return llm_str(
        f'You are an educational summarizer. Write a concise 3–5 sentence summary '
        f'of what this resource teaches about "{concept}". Be specific.\n\n'
        f'Page title: {title}\n\nContent:\n{source}\n\nSummary:'
    )


def _enrich_and_store_resources(
    concept_name: str,
    web_results:  list,
    videos:       list,
    student_id:   Optional[int] = None,
    course_id:    Optional[int] = None,
) -> list[dict]:
    """
    For every web result and YouTube video:
      1. Fetch page / use description as context
      2. Generate a focused LLM summary for the concept
      3. Persist to concept_resources
    Returns the enriched list ready for the API response.
    """
    enriched: list[dict] = []

    for r in web_results[:6]:
        url     = r.get("url", "")
        title   = r.get("title", "")
        snippet = r.get("snippet", "")
        if not url:
            continue
        summary  = _summarise_url(url, title, concept_name, snippet)
        metadata = {"snippet": snippet, "source": r.get("source", "web")}
        _save_concept_resource(
            concept_name=concept_name, resource_type="web",
            url=url, title=title, summary=summary, metadata=metadata,
            student_id=student_id, course_id=course_id,
        )
        enriched.append({
            "resource_type": "web",
            "url": url, "title": title,
            "summary": summary, "metadata": metadata,
        })

    for v in videos[:4]:
        url   = v.get("url", "")
        title = v.get("title", "")
        if not url:
            continue
        desc    = v.get("desc", "")
        # For videos: use description as fallback; only fetch page if desc is empty
        summary = _summarise_url(url, title, concept_name, desc) if desc else llm_str(
            f'In 2–3 sentences, explain what a YouTube video titled '
            f'"{title}" by "{v.get("channel","")}" likely teaches about "{concept_name}".'
        )
        metadata = {
            "channel":  v.get("channel", ""),
            "duration": v.get("duration", ""),
            "views":    v.get("views"),
            "thumb":    v.get("thumb", ""),
            "desc":     desc,
        }
        _save_concept_resource(
            concept_name=concept_name, resource_type="youtube",
            url=url, title=title, summary=summary, metadata=metadata,
            student_id=student_id, course_id=course_id,
            channel=v.get("channel", ""),
            duration=v.get("duration", ""),
            views=v.get("views"),
        )
        enriched.append({
            "resource_type": "youtube",
            "url": url, "title": title,
            "channel":  v.get("channel"),
            "duration": v.get("duration"),
            "views":    v.get("views"),
            "summary":  summary,
            "metadata": metadata,
        })

    return enriched


# ── Pydantic models ───────────────────────────────────────────────────────────

class StudentCreate(BaseModel):
    username: str
    email:    str
    password: str


class DeepSearchConceptRequest(BaseModel):
    topic:      str
    student_id: Optional[int] = None
    course_id:  Optional[int] = None


class GenerateQuizConceptRequest(BaseModel):
    topics:        list[str]
    student_id:    Optional[int] = None
    num_questions: int = 8
    use_kg:        bool = True   # use stored concept_resources as context


class FindGapsRequest(BaseModel):
    topic:         str
    questions:     list          # MCQ list returned by /generate-quiz
    answers:       list          # student answer letters, e.g. ["A","C","B"]
    student_id:    Optional[int] = None
    course_id:     Optional[int] = None
    assessment_id: Optional[int] = None


# ── POST /students ────────────────────────────────────────────────────────────

@app.post("/students", summary="CreateUser — register a new student account")
def create_student(req: StudentCreate):
    """
    Register a new student. Password is hashed (sha-256 + random salt) before
    storage — the plain-text password is never persisted.

    Returns a 6-digit `verification_code` (valid 15 min) that must be submitted
    to `POST /students/{id}/verify` to activate the account.
    In production, send this code by email instead of returning it in the response.
    """
    if not req.username.strip() or not req.email.strip() or not req.password.strip():
        raise HTTPException(status_code=422, detail="All fields are required.")
    if "@" not in req.email or "." not in req.email.split("@")[-1]:
        raise HTTPException(status_code=422, detail="Invalid email address.")
    if len(req.password) < 6:
        raise HTTPException(status_code=422, detail="Password must be at least 6 characters.")

    pwd_hash = _hash_password(req.password)
    code     = str(_secrets.randbelow(900_000) + 100_000)   # 6-digit
    expiry   = datetime.utcnow() + timedelta(minutes=15)

    try:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """INSERT INTO students
                       (username, email, password_hash, verified, verification_code, code_expiry)
                       VALUES (%s,%s,%s,0,%s,%s)
                       RETURNING id, username, email, created_at""",
                    (req.username.strip(), req.email.strip().lower(),
                     pwd_hash, code, expiry),
                )
                row = dict(cur.fetchone())
            conn.commit()
    except psycopg2.errors.UniqueViolation as exc:
        detail = ("Username already taken."
                  if "username" in str(exc) else "Email already registered.")
        raise HTTPException(status_code=409, detail=detail)

    return {
        "student_id":        row["id"],
        "username":          row["username"],
        "email":             row["email"],
        "created_at":        str(row["created_at"]),
        "verification_code": code,   # ← send via email in production
        "next_step": (
            f"Verify your account: "
            f"POST /students/{row['id']}/verify?code={code}"
        ),
    }


@app.post(
    "/students/{student_id}/verify",
    summary="Verify student email with the 6-digit code",
)
def verify_student_account(
    student_id: int  = Path(..., description="Student ID from /students"),
    code:       str  = Query(..., description="6-digit verification code"),
):
    """Activate the student account. The code expires after 15 minutes."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT verification_code, code_expiry, verified FROM students WHERE id=%s",
                (student_id,),
            )
            row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Student not found.")
    stored_code, expiry, verified = row

    if verified:
        return {"verified": True, "message": "Account already verified."}
    if datetime.utcnow() > expiry:
        raise HTTPException(status_code=400, detail="Code expired. Register again or request a new code.")
    if code.strip() != stored_code:
        raise HTTPException(status_code=400, detail="Wrong verification code.")

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE students SET verified=1 WHERE id=%s", (student_id,))
        conn.commit()

    return {"verified": True, "student_id": student_id,
            "message": "Account activated. You can now log in."}


# ── POST /deep-search ─────────────────────────────────────────────────────────

@app.post(
    "/deep-search",
    summary="DeepSearch — search a topic, summarise every URL, store in KG + DB",
)
def deep_search_concept(req: DeepSearchConceptRequest):
    """
    **Full DeepSearch pipeline with per-URL summarisation.**

    Flow:
    1. Check `concept_resources` table — return immediately on cache hit.
    2. Cache miss → run DuckDuckGo sub-queries + open-knowledge DBs + YouTube.
    3. For **every** result URL: fetch page content → LLM summary focused on the concept.
    4. Persist `{url, title, resource_type, summary, metadata_json}` to `concept_resources`.
    5. Mirror Resource nodes to Neo4j so `/resources` and `/recommend` can use them.

    Returns all enriched resources with per-URL `summary` fields.
    """
    concept = _clean_topic(req.topic)

    # ── Cache hit ─────────────────────────────────────────────────────────────
    cached = _get_cached_concept_resources(concept)
    if cached:
        return {
            "concept":   concept,
            "cached":    True,
            "total":     len(cached),
            "resources": cached,
        }

    # ── Cache miss: full pipeline ─────────────────────────────────────────────
    query   = to_search_query(concept)
    queries = generate_sub_queries(concept)
    web_res = gather_sources(queries, concept)
    kg_res  = gather_open_knowledge(concept)
    all_web = web_res + kg_res
    videos  = search_youtube(f"{query} tutorial explained")

    enriched = _enrich_and_store_resources(
        concept_name=concept,
        web_results=all_web,
        videos=videos,
        student_id=req.student_id,
        course_id=req.course_id,
    )

    # Mirror to Neo4j knowledge graph
    try:
        drv = get_neo4j_driver()
        init_graph_schema(drv)
        _save_resources_to_graph(drv, concept, all_web, videos, search_type="deep-search")
        drv.close()
    except Exception:
        pass

    return {
        "concept":   concept,
        "cached":    False,
        "total":     len(enriched),
        "resources": enriched,
    }


# ── POST /generate-quiz ───────────────────────────────────────────────────────

@app.post(
    "/generate-quiz",
    summary="GenerateQuiz — build MCQ from topic(s) using stored KG summaries",
)
def generate_quiz_from_concepts(req: GenerateQuizConceptRequest):
    """
    **Generate a diagnostic or module quiz grounded in stored knowledge.**

    If `use_kg=true` (default):
    - Loads `concept_resources` summaries from the DB for each topic.
    - Uses those summaries as context so the LLM generates questions tied to
      real sources, not hallucinated facts.
    - Falls back to pure LLM generation if no resources are stored yet
      (run `/deep-search` first for best results).

    Each question includes `concept` (which topic it tests) so `/find-gaps`
    can map wrong answers directly to knowledge gaps.
    """
    num_q = max(1, min(req.num_questions, 10))
    context_blocks: list[str] = []

    if req.use_kg:
        for topic in req.topics[:5]:
            clean     = _clean_topic(topic)
            resources = _get_cached_concept_resources(clean)
            summaries = "\n\n".join(
                f"[{r['resource_type'].upper()}] {r['title']}\n{r['summary']}"
                for r in resources[:4] if r.get("summary")
            )
            if summaries:
                context_blocks.append(f"=== {clean} ===\n{summaries}")

    if context_blocks:
        context_text = "\n\n".join(context_blocks)
        prompt = (
            f'You are an expert educator. Using the knowledge base below, '
            f'create exactly {num_q} multiple-choice questions that test deep '
            f'understanding of: {", ".join(req.topics)}.\n\n'
            f'Knowledge base:\n{context_text[:4500]}\n\n'
            f'Reply ONLY with a JSON array, no markdown fences:\n'
            f'[{{"question":"...?","options":["A. ...","B. ...","C. ...","D. ..."],'
            f'"answer":"A","explanation":"...","concept":"topic_name"}}]'
        )
    else:
        topics_str = ", ".join(req.topics)
        prompt = (
            f'Create {num_q} MCQ questions to test knowledge of: "{topics_str}". '
            f'Mix easy / medium / hard difficulties. '
            f'Reply ONLY with a JSON array, no markdown fences:\n'
            f'[{{"question":"...?","options":["A. ...","B. ...","C. ...","D. ..."],'
            f'"answer":"A","explanation":"...","concept":"topic_name"}}]'
        )

    raw       = llm_str(prompt)
    questions = [q for q in (_parse_json_array(raw) or []) if isinstance(q, dict) and "question" in q]

    # Per-topic fallback when bulk generation fails
    if not questions:
        for topic in req.topics[:num_q]:
            p = (
                f'Write 1 MCQ about "{topic}". '
                f'Reply ONLY with JSON (no markdown):\n'
                f'{{"question":"...?","options":["A. ...","B. ...","C. ...","D. ..."],'
                f'"answer":"A","explanation":"...","concept":"{topic}"}}'
            )
            parsed = _parse_json_array(llm_str(p)) or []
            questions.extend(q for q in parsed if isinstance(q, dict))

    # Normalise
    default_concept = req.topics[0] if req.topics else "general"
    for i, q in enumerate(questions[:num_q]):
        q.setdefault("question",    f"Question {i + 1}")
        q.setdefault("options",     ["A. ?", "B. ?", "C. ?", "D. ?"])
        q.setdefault("answer",      "A")
        q.setdefault("explanation", "")
        q.setdefault("concept",     default_concept)

    return {
        "topics":        req.topics,
        "num_questions": len(questions[:num_q]),
        "used_kg":       bool(context_blocks),
        "questions":     questions[:num_q],
    }


# ── POST /find-gaps ───────────────────────────────────────────────────────────

@app.post(
    "/find-gaps",
    summary="FindGaps — score MCQ answers and identify knowledge gaps",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "ml_quiz": {
                            "summary": "ML quiz — 2 wrong answers",
                            "value": {
                                "topic": "Machine Learning",
                                "questions": [],   # pass the list from /generate-quiz
                                "answers": ["A", "B", "A", "C", "A", "B", "D", "A"],
                                "student_id": 1,
                            },
                        }
                    }
                }
            }
        }
    },
)
def find_gaps(req: FindGapsRequest):
    """
    **Evaluate MCQ answers and identify knowledge gaps.**

    Steps:
    1. Compare each answer against the correct answer — compute raw score.
    2. Send the full Q&A table to the LLM: it identifies *which concepts* are
       weak (not just which questions were wrong) and assigns severity:
       `low` (1 slip) · `medium` (conceptual confusion) · `high` (fundamental gap).
    3. Persist each gap to `knowledge_gaps` (if `student_id` provided).
    4. Persist per-question detail to `assessment_answers` (if `assessment_id` provided).

    Pass the returned gap `concept` names to `GET /recommender` to get targeted resources.
    """
    if not req.questions or not req.answers:
        raise HTTPException(status_code=422, detail="'questions' and 'answers' are required.")

    # ── Score answers ─────────────────────────────────────────────────────────
    qa_pairs: list[dict] = []
    for i, q in enumerate(req.questions):
        user_ans    = req.answers[i].strip().upper() if i < len(req.answers) else "?"
        correct_ans = q.get("answer", "A").strip().upper()
        qa_pairs.append({
            "question":      q.get("question", ""),
            "concept":       q.get("concept", req.topic),
            "user_answer":   user_ans,
            "correct_answer": correct_ans,
            "is_correct":    int(user_ans == correct_ans),
        })

    correct_count = sum(p["is_correct"] for p in qa_pairs)
    score = round(correct_count / len(req.questions) * 100) if req.questions else 0

    # ── LLM gap analysis ──────────────────────────────────────────────────────
    qa_text = "\n".join(
        f"Q{i+1} [concept: {p['concept']}]: {p['question']}\n"
        f"  Student: {p['user_answer']}  Correct: {p['correct_answer']}  "
        f"{'✅' if p['is_correct'] else '❌'}"
        for i, p in enumerate(qa_pairs)
    )
    prompt = (
        f'A student answered a quiz on "{req.topic}". Analyse and identify gaps:\n\n'
        f'{qa_text}\n\n'
        f'Return ONLY a JSON array of gaps (empty array if all correct):\n'
        f'[{{"concept":"...","severity":"low|medium|high",'
        f'"explanation":"why this is a gap — 1 sentence"}}]'
    )
    raw  = llm_str(prompt)
    gaps = [g for g in (_parse_json_array(raw) or [])
            if isinstance(g, dict) and "concept" in g]

    # ── Persist gaps to DB ────────────────────────────────────────────────────
    gap_ids: list[int] = []
    if req.student_id and gaps:
        with get_conn() as conn:
            with conn.cursor() as cur:
                for g in gaps:
                    cur.execute(
                        """INSERT INTO knowledge_gaps
                           (student_id, course_id, topic_name, severity, source)
                           VALUES (%s,%s,%s,%s,'quiz') RETURNING id""",
                        (req.student_id, req.course_id,
                         g["concept"], g.get("severity", "medium")),
                    )
                    gap_ids.append(cur.fetchone()[0])
            conn.commit()

    # ── Persist per-question answers ──────────────────────────────────────────
    if req.assessment_id:
        with get_conn() as conn:
            with conn.cursor() as cur:
                for p in qa_pairs:
                    cur.execute(
                        """INSERT INTO assessment_answers
                           (assessment_id, question, concept,
                            student_answer, correct_answer, is_correct)
                           VALUES (%s,%s,%s,%s,%s,%s)""",
                        (req.assessment_id, p["question"], p["concept"],
                         p["user_answer"], p["correct_answer"], p["is_correct"]),
                    )
            conn.commit()

    level = ("advanced"     if score >= 70
             else "intermediate" if score >= 40
             else "beginner")

    return {
        "topic":          req.topic,
        "score":          score,
        "correct":        correct_count,
        "total":          len(req.questions),
        "level":          level,
        "gaps_found":     len(gaps),
        "gaps":           gaps,
        "gap_ids":        gap_ids,
        "answers_detail": qa_pairs,
        "next_step": (
            "Call GET /recommender?gaps="
            + ",".join(g["concept"] for g in gaps)
            + " to get targeted learning resources."
            if gaps else "No gaps found — great job!"
        ),
    }


# ── GET /recommender ──────────────────────────────────────────────────────────

@app.get(
    "/recommender",
    summary="Recommender — return best resources for identified knowledge gaps",
)
def recommender(
    gaps:       str           = Query(..., description="Comma-separated gap concept names",
                                      example="binary search,recursion,dynamic programming"),
    student_id: Optional[int] = Query(None, description="Student ID to personalise search"),
    limit:      int           = Query(3, ge=1, le=10, description="Max resources per gap"),
):
    """
    **Resource recommender for knowledge gaps.**

    For each gap concept:
    1. Queries `concept_resources` table (populated by `/deep-search`).
    2. If no resources are cached, runs a targeted search + stores them.
    3. Uses the LLM to rank the top `limit` resources and explain why each helps.

    Returns a per-gap dict of ranked resources with LLM-generated `reason` fields.
    Run `/deep-search` on a topic first for the richest recommendations.
    """
    gap_list = [g.strip() for g in gaps.split(",") if g.strip()]
    if not gap_list:
        raise HTTPException(status_code=422, detail="Provide at least one gap concept.")

    results: dict[str, list] = {}

    for gap in gap_list[:6]:   # cap at 6 gaps per call
        resources = _get_cached_concept_resources(gap)

        # Nothing stored yet — run a focused search and cache it
        if not resources:
            web = search_web(f"{gap} tutorial explained for beginners", max_results=4)
            yt  = search_youtube(f"{gap} explained", max_results=3)
            _enrich_and_store_resources(gap, web, yt, student_id=student_id)
            resources = _get_cached_concept_resources(gap)

        if not resources:
            results[gap] = []
            continue

        # Ask LLM to rank the best ones
        res_lines = "\n".join(
            f"[{i+1}] ({r['resource_type']}) {r['title']} — {r['url']}\n"
            f"  Summary: {(r.get('summary') or '')[:200]}"
            for i, r in enumerate(resources[:10])
        )
        prompt = (
            f'A student has a knowledge gap in "{gap}". '
            f'From the resources below, pick the TOP {limit} most helpful ones '
            f'for a learner trying to close this gap.\n\n'
            f'{res_lines}\n\n'
            f'Return ONLY valid JSON:\n'
            f'[{{"rank":1,"title":"...","url":"...","resource_type":"web|youtube",'
            f'"reason":"1 sentence why this helps with {gap}"}}]'
        )
        raw    = llm_str(prompt)
        ranked = [r for r in (_parse_json_array(raw) or [])
                  if isinstance(r, dict) and "url" in r]

        # Fallback: return top resources directly without LLM ranking
        if not ranked:
            ranked = [
                {
                    "rank":          i + 1,
                    "title":         r["title"],
                    "url":           r["url"],
                    "resource_type": r["resource_type"],
                    "reason":        (r.get("summary") or "")[:150],
                }
                for i, r in enumerate(resources[:limit])
            ]

        results[gap] = ranked[:limit]

    return {
        "gaps":    gap_list,
        "results": results,
    }


# ══════════════════════════════════════════════════════════════════════════════
# RECOMMENDATION ENGINE  POST /recommend/next
# Inputs : subject · historical_user_data · graph_db_resources
# Output : scored candidate modules + the single best "next content"
# ══════════════════════════════════════════════════════════════════════════════

class CompletedModule(BaseModel):
    module_uid:     str
    quiz_score:     float = 0.0      # 0-100
    time_spent_min: int   = 0        # minutes actually spent


class HistoricalUserData(BaseModel):
    session_id:      Optional[int]         = None   # load history from DB if provided
    completed_modules: list[CompletedModule] = []   # or supply directly
    weak_concepts:   list[str]             = []     # concepts the user struggled with
    preferred_type:  str                   = "mixed"  # "web" | "video" | "mixed"
    current_level:   str                   = "beginner"


class RecommendNextRequest(BaseModel):
    subject:              str
    historical_user_data: HistoricalUserData
    use_graph_db:         bool = True    # query Neo4j for graph-aware candidates
    include_content:      bool = False   # also generate lesson for the top pick


# ── Graph-aware candidate retrieval ──────────────────────────────────────────

def _graph_candidates(subject: str, completed_uids: list, weak_concepts: list) -> list:
    """
    Cypher query that returns candidate modules with a composite score:
      - prerequisites satisfied (filter)
      - concepts_taught         (breadth bonus)
      - weak_concept_matches    (remediation bonus: concepts user struggled with)
      - module_order            (tie-break: earlier modules first)
    """
    try:
        drv = get_neo4j_driver()
        with drv.session() as s:
            result = s.run(
                """
                MATCH (t:Topic {name:$subject})-[:HAS_MODULE]->(m:Module)
                WHERE NOT m.uid IN $done
                  AND NOT EXISTS {
                      MATCH (pre:Module)-[:PREREQUISITE_FOR]->(m)
                      WHERE NOT pre.uid IN $done
                  }
                OPTIONAL MATCH (m)-[:TEACHES]->(c:Concept)
                WITH m,
                     count(c)                                         AS concepts_taught,
                     count(CASE WHEN c.name IN $weak THEN 1 END)     AS weak_matches
                RETURN m.uid              AS uid,
                       m.title            AS title,
                       m.objective        AS objective,
                       m.duration_minutes AS duration,
                       m.order            AS module_order,
                       concepts_taught,
                       weak_matches,
                       (concepts_taught * 1.0 + weak_matches * 2.0)  AS graph_score
                ORDER BY graph_score DESC, module_order ASC
                LIMIT 10
                """,
                subject=subject,
                done=completed_uids,
                weak=weak_concepts,
            )
            candidates = [dict(r) for r in result]
        drv.close()
        return candidates
    except Exception as e:
        print(f"[recommend] Neo4j unavailable: {e}")
        return []


# ── History loader ────────────────────────────────────────────────────────────

def _load_history_from_db(session_id: int) -> tuple[list[CompletedModule], list[str]]:
    """
    Load completed modules + quiz scores from session_progress.
    Derive weak_concepts: concepts taught by modules where quiz_score < 60.
    Returns (completed_list, weak_concept_names).
    """
    completed, weak_concepts = [], []
    try:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                schema = _progress_schema(session_id)
                cur.execute(
                    f'SELECT module_uid, quiz_score FROM "{schema}".session_progress '
                    "WHERE session_id=%s ORDER BY completed_at",
                    (session_id,),
                )
                rows = cur.fetchall()
        completed = [
            CompletedModule(module_uid=r["module_uid"],
                            quiz_score=float(r["quiz_score"] or 0))
            for r in rows
        ]
        # Identify weak modules (score < 60) and fetch their concepts from Neo4j
        weak_uids = [c.module_uid for c in completed if c.quiz_score < 60]
        if weak_uids:
            try:
                drv = get_neo4j_driver()
                with drv.session() as s:
                    result = s.run(
                        "MATCH (m:Module)-[:TEACHES]->(c:Concept) "
                        "WHERE m.uid IN $uids RETURN DISTINCT c.name AS name",
                        uids=weak_uids,
                    )
                    weak_concepts = [r["name"] for r in result]
                drv.close()
            except Exception:
                pass
    except Exception as e:
        print(f"[recommend] DB load error: {e}")
    return completed, weak_concepts


# ── LLM re-ranker ─────────────────────────────────────────────────────────────

def _llm_rerank(subject: str, candidates: list, history: HistoricalUserData,
                completed_uids: list) -> dict:
    """
    Give the top graph candidates + full user context to the LLM and ask it
    to pick the single best next module with a personalised explanation.
    """
    if not candidates:
        return {}

    cand_lines = "\n".join(
        f"  [{i+1}] uid={c['uid']} | {c['title']} | "
        f"concepts={c.get('concepts_taught',0)} | weak_matches={c.get('weak_matches',0)} | "
        f"graph_score={c.get('graph_score',0):.1f}"
        for i, c in enumerate(candidates[:5])
    )

    completed_lines = "\n".join(
        f"  - {c.module_uid} | score={c.quiz_score:.0f}% | time={c.time_spent_min}min"
        for c in history.completed_modules
    ) or "  None yet"

    weak_line = ", ".join(history.weak_concepts) or "none identified"

    prompt = f"""You are a personalised learning advisor for the subject "{subject}".

LEARNER PROFILE:
  Level      : {history.current_level}
  Completed  : {len(completed_uids)} modules
{completed_lines}
  Weak areas : {weak_line}
  Preference : {history.preferred_type} content

CANDIDATE NEXT MODULES (graph-ranked):
{cand_lines}

Choose the SINGLE best next module for this learner. Return ONLY valid JSON:
{{
  "recommended_uid": "...",
  "title": "...",
  "reason": "2-3 sentence personalised explanation of why this module is best right now.",
  "learning_tip": "One concrete study tip for this specific learner on this topic."
}}"""

    raw   = llm_str(prompt)
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        try:
            result = json.loads(re.sub(r",\s*([}\]])", r"\1", match.group()))
            # attach full candidate metadata
            chosen = next((c for c in candidates if c["uid"] == result.get("recommended_uid")), candidates[0])
            result.update({k: chosen.get(k) for k in ("uid","title","objective","duration","module_order") if k in chosen})
            return result
        except (json.JSONDecodeError, ValueError):
            pass
    # fallback: top graph candidate
    top = candidates[0]
    return {
        "recommended_uid": top["uid"],
        "title":           top["title"],
        "objective":       top.get("objective",""),
        "reason":          "This module has the highest graph relevance score for your profile.",
        "learning_tip":    "Take notes and test yourself after each section.",
    }


# ── /recommend/next ───────────────────────────────────────────────────────────

@app.post(
    "/recommend/next",
    summary="Smart recommendation: graph traversal + weakness scoring + LLM re-ranking",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "from_session": {
                            "summary": "Load history from DB session",
                            "value": {
                                "subject": "Machine Learning",
                                "historical_user_data": {
                                    "session_id": 1,
                                    "preferred_type": "video",
                                    "current_level": "beginner",
                                },
                                "use_graph_db": True,
                                "include_content": False,
                            },
                        },
                        "inline_history": {
                            "summary": "Provide history directly",
                            "value": {
                                "subject": "Machine Learning",
                                "historical_user_data": {
                                    "completed_modules": [
                                        {"module_uid": "Machine Learning::1", "quiz_score": 85, "time_spent_min": 45},
                                        {"module_uid": "Machine Learning::2", "quiz_score": 42, "time_spent_min": 90},
                                    ],
                                    "weak_concepts": ["gradient descent", "backpropagation"],
                                    "preferred_type": "mixed",
                                    "current_level": "beginner",
                                },
                                "use_graph_db": True,
                                "include_content": False,
                            },
                        },
                        "with_content": {
                            "summary": "Get recommendation + generate lesson for top pick",
                            "value": {
                                "subject": "Electromagnetism",
                                "historical_user_data": {
                                    "session_id": 2,
                                    "current_level": "intermediate",
                                },
                                "use_graph_db": True,
                                "include_content": True,
                            },
                        },
                    }
                }
            }
        }
    },
)
def recommend_next(req: RecommendNextRequest):
    """
    **Smart recommendation engine.**

    Pipeline:
    1. **Load history** — from DB (`session_id`) or inline `completed_modules`
    2. **Graph traversal** — Neo4j Cypher finds all unlocked modules
       (prerequisites satisfied) and scores them by concept coverage + weak-concept remediation
    3. **LLM re-ranking** — top graph candidates are passed to the LLM with full
       user context (level, scores, weak areas, preference) for a personalised final pick
    4. **Content generation** *(optional)* — if `include_content=true`, runs a deep
       search on the recommended module and returns the full lesson + quiz

    Response fields:
    - `recommendation` — the best next module with `reason` and `learning_tip`
    - `candidates` — all scored alternatives (so frontend can show a choice)
    - `lesson` + `quiz` — only present when `include_content=true`
    """
    hist = req.historical_user_data

    # 1. Load / merge history
    db_completed, db_weak = [], []
    if hist.session_id:
        db_completed, db_weak = _load_history_from_db(hist.session_id)

    # Merge DB history with any inline history provided
    all_completed  = {c.module_uid: c for c in db_completed}
    all_completed.update({c.module_uid: c for c in hist.completed_modules})
    completed_list = list(all_completed.values())
    completed_uids = [c.module_uid for c in completed_list]

    weak_concepts  = list(set(db_weak + hist.weak_concepts))

    # Rebuild a merged HistoricalUserData for the LLM
    merged_hist = HistoricalUserData(
        session_id=hist.session_id,
        completed_modules=completed_list,
        weak_concepts=weak_concepts,
        preferred_type=hist.preferred_type,
        current_level=hist.current_level,
    )

    # 2. Graph-DB candidates
    candidates = []
    if req.use_graph_db:
        candidates = _graph_candidates(req.subject, completed_uids, weak_concepts)

    if not candidates:
        return {
            "subject":        req.subject,
            "recommendation": None,
            "candidates":     [],
            "message":        "No unlocked modules found. Either the roadmap is complete or "
                              "no roadmap exists yet for this subject — call GET /roadmap first.",
        }

    # 3. LLM re-ranking
    recommendation = _llm_rerank(req.subject, candidates, merged_hist, completed_uids)

    # 4. Optional: generate lesson + quiz for the top pick
    lesson_data = None
    quiz_data   = None
    if req.include_content and recommendation.get("recommended_uid"):
        try:
            drv    = get_neo4j_driver()
            module = _get_module_from_graph(drv, recommendation["recommended_uid"])
            drv.close()
            lesson = _generate_lesson(
                module_title=module["title"],
                module_objective=module["objective"],
                concepts=module["concepts"],
                topic=req.subject,
                level=hist.current_level,
            )
            lesson_data = {
                "content":     lesson["content"],
                "sources":     lesson["sources"],
                "videos":      lesson["videos"],
                "recommended": lesson["recommended"],
            }
            quiz_data = {"questions": _build_quiz(lesson["content"], 5)}
        except Exception as e:
            lesson_data = {"error": str(e)}

    return {
        "subject":        req.subject,
        "completed_count": len(completed_uids),
        "weak_concepts":   weak_concepts,
        "recommendation":  recommendation,
        "candidates": [
            {
                "uid":           c["uid"],
                "title":         c["title"],
                "objective":     c.get("objective",""),
                "duration":      c.get("duration", 60),
                "concepts_taught": c.get("concepts_taught", 0),
                "weak_matches":  c.get("weak_matches", 0),
                "graph_score":   round(c.get("graph_score", 0), 2),
            }
            for c in candidates
        ],
        **({"lesson": lesson_data, "quiz": quiz_data} if req.include_content else {}),
    }


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
def on_startup():
    try:
        init_db()
        print("✅ PostgreSQL schema (core) ready")
    except Exception as e:
        print(f"⚠️  PostgreSQL unavailable: {e}")
    try:
        init_learning_schema()
        print("✅ PostgreSQL schema (learning-path) ready")
    except Exception as e:
        print(f"⚠️  Learning schema init failed: {e}")
