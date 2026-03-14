# ◈ LearnFlow AI

Adaptive AI-powered learning platform. Enter any topic, get a personalised diagnostic quiz, a custom roadmap, AI-generated lessons, and module quizzes — all running **100% locally** with no API keys required.

```
FastAPI  ·  Streamlit  ·  LangGraph  ·  PostgreSQL  ·  pgvector  ·  Ollama
```

---

## How it works

```
You enter a topic
      │
      ▼
Diagnostic quiz (8 questions)
      │
      ▼
AI analyses your level + gaps
      │
      ▼
Personalised roadmap (5–8 modules)
      │
      ▼
AI-generated lesson  →  5-question quiz  →  pass (70%) → next module
                                         └→  fail → review & retry
      │
      ▼
Course complete 🏆
```

State is checkpointed to PostgreSQL so you can close the browser and resume later with your session ID.

---

## Prerequisites

| Tool | Version | Install |
|---|---|---|
| Docker + Docker Compose | v2+ | https://docs.docker.com/get-docker |
| Ollama | latest | https://ollama.com |
| Python | 3.11+ | https://python.org (local-only mode) |

---

## Option A — Docker (recommended)

Everything runs in containers. Ollama models are pulled automatically on first start.

### 1. Clone and configure

```bash
git clone <your-repo-url>
cd learnflow
cp .env.example .env
```

The default `.env` works out of the box for Docker. No API keys needed.

### 2. Start all services

```bash
docker compose up -d
```

This starts:
- **PostgreSQL 16** with the pgvector extension
- **Ollama** server (downloads `llama3.2` + `nomic-embed-text` on first run — ~3 GB, one-time)
- **FastAPI** backend on port `8000`
- **Streamlit** frontend on port `8501`

> First startup takes a few minutes while Ollama pulls the models.

### 3. Run database migrations

```bash
docker compose exec api alembic upgrade head
```

### 4. Open the app

| Service | URL |
|---|---|
| Streamlit app | http://localhost:8501 |
| FastAPI docs (Swagger) | http://localhost:8000/docs |

### Useful commands

```bash
# View logs
docker compose logs -f api
docker compose logs -f ollama

# Stop everything
docker compose down

# Stop and delete database data
docker compose down -v

# Rebuild after code changes
docker compose up -d --build api
```

---

## Option B — Local (no Docker)

Run each service directly on your machine.

### 1. Install and start Ollama

Download from https://ollama.com, then pull the required models:

```bash
ollama pull llama3.2          # ~2 GB — chat model
ollama pull nomic-embed-text  # ~270 MB — embeddings
```

Verify Ollama is running:
```bash
curl http://localhost:11434/api/tags
```

### 2. Start PostgreSQL with pgvector

The easiest way is still Docker for just the database:

```bash
docker run -d \
  --name learnflow-db \
  -e POSTGRES_DB=learnflow \
  -e POSTGRES_USER=user \
  -e POSTGRES_PASSWORD=pass \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

Or use an existing PostgreSQL 16+ instance and install the extension manually:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` if your Postgres or Ollama settings differ from the defaults:

```env
DATABASE_URL=postgresql://user:pass@localhost:5432/learnflow
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
OLLAMA_EMBED_MODEL=nomic-embed-text
```

### 4. Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 5. Run database migrations

```bash
alembic upgrade head
```

### 6. Start the FastAPI backend

```bash
uvicorn app.main:app --reload --port 8000
```

API docs available at http://localhost:8000/docs

### 7. Start the Streamlit frontend

Open a second terminal:

```bash
source .venv/bin/activate
streamlit run streamlit_app/app.py
```

App available at http://localhost:8501

---

## Changing the LLM model

Edit `.env` and set `OLLAMA_MODEL` to any model you have pulled:

```env
OLLAMA_MODEL=mistral          # fast, good quality
OLLAMA_MODEL=qwen2.5          # strong reasoning
OLLAMA_MODEL=gemma2           # Google's model
OLLAMA_MODEL=llama3.2         # default
```

Pull a model with:
```bash
ollama pull mistral
```

Browse all available models at https://ollama.com/library

---

## Project structure

```
learnflow/
├── app/                        # FastAPI backend
│   ├── main.py                 # App entry point + lifespan
│   ├── config.py               # Settings (reads .env)
│   ├── db.py                   # SQLAlchemy async engine
│   ├── api/
│   │   ├── sessions.py         # POST /api/sessions/create, /assess, GET /{id}
│   │   ├── modules.py          # POST /api/modules/{id}/generate-content
│   │   └── quizzes.py          # POST /api/quizzes/{id}/generate-quiz, /submit-quiz
│   ├── graph/
│   │   └── learning_graph.py   # LangGraph 8-node learning loop
│   ├── llm/
│   │   ├── client.py           # Ollama chat wrapper
│   │   └── prompts.py          # All system prompts
│   ├── models/
│   │   ├── database.py         # SQLAlchemy models (relational + pgvector)
│   │   └── schemas.py          # Pydantic request/response schemas
│   └── vectordb/
│       ├── client.py           # Ollama embeddings wrapper
│       ├── embeddings.py       # Store content + knowledge vectors
│       └── queries.py          # Semantic search queries
├── streamlit_app/
│   └── app.py                  # Full UI — 7 phases
├── alembic/
│   ├── env.py
│   └── versions/
│       └── 0001_initial.py     # Full schema + pgvector tables
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.streamlit
├── requirements.txt
└── .env.example
```

---

## Architecture overview

```
┌─────────────────────┐        HTTP        ┌──────────────────────┐
│   Streamlit UI      │ ◄────────────────► │   FastAPI backend    │
│   :8501             │                    │   :8000              │
└─────────────────────┘                    └──────────┬───────────┘
                                                      │
                          ┌───────────────────────────┼──────────────────┐
                          │                           │                  │
                   ┌──────▼──────┐           ┌────────▼───────┐  ┌──────▼──────┐
                   │  LangGraph  │           │  PostgreSQL 16  │  │   Ollama    │
                   │  workflow   │           │  + pgvector     │  │  :11434     │
                   │  (8 nodes)  │           │                 │  │             │
                   └─────────────┘           │ • sessions      │  │ • llama3.2  │
                          │                 │ • roadmaps      │  │   (chat)    │
                   PostgresSaver            │ • modules       │  │             │
                   (checkpointing)          │ • quizzes       │  │ • nomic-    │
                          │                 │ • embeddings    │  │   embed-text│
                          └────────────────►│   (pgvector)    │  │   (768-dim) │
                                            └─────────────────┘  └─────────────┘
```

**No external services. No API keys. Everything runs on your machine.**

---

## Troubleshooting

**Ollama models not responding**
```bash
ollama list                    # check pulled models
ollama run llama3.2 "hello"   # test the model directly
```

**Database connection error**
```bash
docker compose ps postgres     # check it's healthy
docker compose logs postgres
```

**pgvector extension missing**
```bash
docker compose exec postgres psql -U user -d learnflow -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

**Port already in use**
```bash
# Change ports in docker-compose.yml or run:
lsof -i :8000   # find what's using port 8000
lsof -i :8501   # find what's using port 8501
```

**Alembic migration fails**
```bashl
# Reset and re-run
docker compose exec api alembic downgrade base
docker compose exec api alembic upgrade head


# 1. Install & start Ollama  →  https://ollama.com
ollama pull llama3.2
ollama pull nomic-embed-text

# 2. Copy env (no API keys needed)
cp .env.example .env

# 3. Start Postgres, then run
docker compose up postgres -d
pip install -r requirements.txt
uvicorn app.main:app --reload &
streamlit run streamlit_app/app.py

```
