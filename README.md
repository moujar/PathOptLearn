# PathOptLearn

**AI adaptive learning platform** that builds personalized educational pathways using local LLMs, knowledge graphs, and real-time web research.

---

## Demo

[Watch the demo](demo/demo.mp4)

---

## Overview

PathOptLearn guides learners through any subject by dynamically assessing their knowledge level, identifying gaps, and generating a structured roadmap of AI-synthesized lessons — all running locally via Docker.

**Learning flow:**

1. Enter any topic (e.g., "Machine Learning", "Calculus", "Distributed Systems")
2. Platform runs a deep web + YouTube search to build context
3. Diagnostic quiz (6 questions) determines your level: beginner / intermediate / advanced
4. LLM generates a 15-module roadmap (3 levels × 5 modules each)
5. For each module: read an AI-generated lesson → take a 5-question quiz
6. Score ≥ 70%: advance. Score < 70%: gaps are identified → targeted resources → retry
7. Track your progress on an interactive knowledge graph dashboard

---

## Architecture

```text
Topic Input
    ↓
[Deep Search]  DuckDuckGo + Wikipedia + arXiv + YouTube → cached in PostgreSQL
    ↓
[Assessment]   LLM generates 6-question diagnostic → score → level
    ↓
[Roadmap]      LLM creates 15 modules → stored in Neo4j knowledge graph
    ↓
[Per-Module Loop]
  ├── Lesson: synthesized from web research + resource links
  ├── Quiz: 5 questions → score
  ├── < 70%: LLM identifies gaps → recommends resources → retry
  └── ≥ 70%: advance to next module
    ↓
[Completion]   Final stats, full history, graph visualization
```

**Services:**

| Service           | Role                                        | Port         |
| ----------------- | ------------------------------------------- | ------------ |
| FastAPI backend   | REST API, LLM orchestration                 | 8000         |
| Streamlit frontend| Web UI                                      | 8501         |
| PostgreSQL        | Users, sessions, assessments, gaps          | 5432         |
| Neo4j             | Knowledge graph (topics, modules, concepts) | 7474 / 7687  |
| Ollama            | Local LLM inference (`llama3.2:1b`)         | 11434        |

---

## Quick Start

**Prerequisites:** Docker and Docker Compose

```bash
# Clone the repo
git clone https://github.com/yourusername/PathOptLearn.git
cd PathOptLearn

# Start all services (first run pulls the LLM model ~30s)
docker compose up --build

# Watch model initialization
docker compose logs -f ollama-init
```

**Access:**

- Frontend UI → <http://localhost:8501>
- API (Swagger) → <http://localhost:8000/docs>
- Neo4j Browser → <http://localhost:7474>

---

## Features

- **Adaptive leveling** — Diagnostic quiz classifies beginner / intermediate / advanced before starting
- **Knowledge gap tracking** — Wrong answers are analyzed by LLM to identify concept-level weaknesses (high/medium/low severity), not just scored
- **Dynamic roadmap** — 15 modules generated per topic, stored in Neo4j and reused across students
- **Multi-source research** — Web pages, Wikipedia, arXiv, OpenAlex, and YouTube transcripts scraped and ranked per topic
- **Smart caching** — All searches and lessons are cached in PostgreSQL; repeat topics load instantly
- **Knowledge graph visualization** — Interactive Neo4j graph of topics, modules, concepts, and resources
- **Streaming responses** — Deep search and lesson synthesis stream progress to the UI in real time
- **Local-first** — All LLM inference runs via Ollama; no external API keys required

---

## Project Structure

```text
PathOptLearn/
├── app/
│   ├── backend.py          # FastAPI API (~4,300 lines): LLM, search, Neo4j, PostgreSQL
│   ├── frontend.py         # Streamlit UI (~1,140 lines): 12 pages
│   ├── requirements.txt
│   └── Dockerfile
├── evalution/
│   ├── benchmakring/
│   │   └── run_benchmark.py    # Benchmark against Riiid!, EdNet-KT1, ASSISTments datasets
│   └── UserSimulation/
│       └── llm_student.py      # LLM-based student simulator (5 student profiles)
├── docker-compose.yml
├── DbSchema.md             # PostgreSQL + Neo4j schema diagrams
├── endpoint.md             # API endpoint flow diagram
└── instruction.md          # Setup notes
```

---

## API Reference

Key endpoints (full docs at `/docs`):

| Endpoint                   | Method | Description                                  |
| -------------------------- | ------ | -------------------------------------------- |
| `POST /students`           | POST   | Register a new user                          |
| `POST /deep-search`        | POST   | Web + YouTube research for a topic           |
| `GET /assess`              | GET    | Generate 6-question diagnostic quiz          |
| `POST /assess/evaluate`    | POST   | Score answers → determine level + gaps       |
| `GET /roadmap`             | GET    | Build 3-level × 5-module learning roadmap    |
| `GET /lesson`              | GET    | Generate AI lesson for a module              |
| `POST /quiz`               | POST   | Generate 5-question module quiz              |
| `POST /find-gaps`          | POST   | Score quiz + identify knowledge gaps         |
| `GET /recommender`         | GET    | LLM-ranked resources per gap                 |
| `POST /next`               | POST   | Mark module complete + get next module       |
| `GET /users/{id}/progress` | GET    | Full progress report (assessments, mastery)  |
| `GET /graph`               | GET    | Export Neo4j knowledge graph                 |

---

## Database Schema

**PostgreSQL** — Users, sessions, quizzes, gaps, resources, search history  
**Neo4j** — Nodes: `Topic`, `Module`, `Concept`, `Resource`, `LevelGroup` with edges like `HAS_MODULE`, `TEACHES`, `REQUIRES`, `REFERENCES`

See [DbSchema.md](DbSchema.md) for full ER and graph diagrams.

---

## Evaluation

### Benchmarking

Tests the platform's knowledge-tracing accuracy against standard educational datasets:

```bash
cd evalution
pip install -r requirements.txt
python benchmakring/run_benchmark.py
```

Measures AUC, accuracy, and RMSE against Riiid!, EdNet-KT1, and ASSISTments.

### Student Simulation

Simulates full learning sessions with LLM-driven student profiles:

```bash
python UserSimulation/llm_student.py
```

Profiles: `beginner`, `intermediate`, `advanced`, `struggling`, `fast_learner`  
Output: `simulation_log.jsonl` with per-module pass rates, retry counts, gap open/close events.

---

## Tech Stack

- **Backend:** Python, FastAPI, Uvicorn
- **Frontend:** Streamlit, streamlit-agraph
- **Databases:** PostgreSQL 16, Neo4j 5
- **LLM:** Ollama (`llama3.2:1b` by default — swap for any Ollama-supported model)
- **Search:** DuckDuckGo (`ddgs`), BeautifulSoup4, yt-dlp, youtube-transcript-api
- **Evaluation:** pandas, scikit-learn, numpy

---

## License

MIT License — Copyright 2026 Abderrahmane Moujar
