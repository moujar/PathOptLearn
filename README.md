# AdaptLearn AI

**Adaptive AI System for Personalised Learning Path Recommendation & Knowledge State Prediction**

Master's Thesis Implementation · Paris-Saclay University

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AdaptLearn AI Pipeline                      │
├─────────┬─────────┬─────────┬──────────┬──────────┬────────────────┤
│ Stage 1 │ Stage 2 │ Stage 3 │ Stage 4  │ Stage 5★ │  Stage 6★      │
│  Goal   │  Quiz   │  CAT-   │  KG      │  DRL-PPO │  Adaptive      │
│ Parsing │  Gen    │  IRT    │  Build   │  Path    │  Loop +        │
│  GPT-4  │ RAG+GPT │  3PL    │ GraphRAG │  Rec.    │  Forgetting    │
├─────────┴─────────┴─────────┴──────────┴──────────┴────────────────┤
│              FastAPI Backend  ·  Redis  ·  Neo4j  ·  FAISS          │
├─────────────────────────────────────────────────────────────────────┤
│                    Streamlit Frontend (port 8501)                    │
└─────────────────────────────────────────────────────────────────────┘
★ = Main thesis contribution
```

## Quick Start

### 1. Prerequisites

```bash
# Python 3.11+
python --version

# Redis (macOS)
brew install redis && brew services start redis

# Neo4j (optional — KG storage)
brew install neo4j && neo4j start
```

### 2. Install Dependencies

```bash
cd adaptlearn
pip install -r requirements.txt

# Optional: torch-geometric for GAT embeddings
pip install torch-geometric
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys:
#   OPENAI_API_KEY, TAVILY_API_KEY, YOUTUBE_API_KEY
```

### 4. Start the Backend

```bash
uvicorn backend.main:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

### 5. Start the Frontend

```bash
streamlit run frontend/app.py --server.port 8501
# Frontend: http://localhost:8501
```

---

## API Endpoint Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/goal/parse` | Parse learning goal with GPT-4 |
| POST | `/api/v1/goal/harvest` | Harvest web + YouTube content |
| POST | `/api/v1/quiz/generate` | Generate IRT-calibrated quiz |
| POST | `/api/v1/assessment/run` | Run batch CAT-IRT assessment |
| POST | `/api/v1/assessment/next-item` | Get next adaptive item |
| POST | `/api/v1/kg/build` | Build knowledge graph |
| GET  | `/api/v1/kg/{kg_id}/visualize` | KG visualisation data |
| POST | `/api/v1/recommend/path` | ★ DRL-PPO path recommendation |
| POST | `/api/v1/recommend/benchmark` | Benchmark all 4 algorithms |
| POST | `/api/v1/loop/update` | ★ Update state + re-optimise path |
| GET  | `/api/v1/loop/history/{student_id}` | Session history |
| POST | `/api/v1/explain/path` | GPT-4 RAG explanation |
| GET  | `/health` | Health check |

---

## Running Benchmarks

```bash
# Download ASSISTments 2009 dataset first:
# https://sites.google.com/site/assistmentsdata/home/assistment-2009-2010-data

python -m backend.benchmarks.run_benchmark \
    --data data/benchmarks/assistments_2009.csv \
    --output results/benchmark_table.csv \
    --n_students 500
```

Output: LaTeX table for thesis + `results/benchmark_table.csv`

---

## Training the DRL Agent

```bash
python scripts/train_drl.py \
    --data data/benchmarks/assistments_2009.csv \
    --timesteps 200000 \
    --output models/checkpoints/drl_ppo
```

---

## Running Tests

```bash
pytest tests/test_pipeline.py -v
```

---

## Novel Contributions (Thesis)

1. **End-to-end pipeline**: goal parsing → content harvest → CAT-IRT assessment → dynamic KG → DRL-PPO recommendation → continual adaptation
2. **DRL-PPO over dynamic KG**: MDP formulation with mastery gain + difficulty matching + prerequisite satisfaction reward
3. **Forgetting-aware continual KT**: Ebbinghaus decay with spaced repetition stability tracking integrated into path re-optimisation
4. **Zero pre-built item bank**: all quiz items generated live from harvested content via RAG + GPT-4
5. **GraphRAG + GAT**: domain KG built dynamically from harvested text with Graph Attention Network node embeddings

---

## Project Structure

```
adaptlearn/
├── backend/
│   ├── main.py                  # FastAPI app
│   ├── schemas.py               # Shared Pydantic models
│   ├── config.py                # Settings (pydantic-settings)
│   ├── routers/
│   │   ├── goal.py              # Stage 1 — Goal Elicitation
│   │   ├── quiz.py              # Stage 2 — Quiz Generation
│   │   ├── assessment.py        # Stage 3 — CAT-IRT
│   │   ├── kg.py                # Stage 4 — Knowledge Graph
│   │   ├── recommend.py         # Stage 5 ★ — DRL-PPO Path
│   │   ├── loop.py              # Stage 6 ★ — Adaptive Loop
│   │   └── explain.py           # Stage 7 — LLM Explanation
│   ├── models/
│   │   ├── dkt.py               # Deep Knowledge Tracing (LSTM)
│   │   ├── dkvmn.py             # Dynamic Key-Value Memory Network
│   │   ├── akt.py               # Attentive Knowledge Tracing
│   │   ├── deep_irt.py          # Deep-IRT item calibration
│   │   ├── drl_agent.py         # PPO agent + LearningEnv (gym)
│   │   └── forgetting.py        # Ebbinghaus forgetting module
│   ├── services/
│   │   ├── content_harvester.py # Tavily + YouTube scraper
│   │   ├── rag_engine.py        # FAISS + sentence-transformers
│   │   ├── kg_builder.py        # GraphRAG + GAT + Neo4j
│   │   ├── cat_engine.py        # IRT-CAT 3PL scoring
│   │   └── llm_explainer.py     # RAG-augmented GPT-4 explanations
│   └── benchmarks/
│       ├── run_benchmark.py     # ASSISTments benchmark runner
│       └── metrics.py           # AUC, ACC, RMSE, Recall@K, LES
├── frontend/
│   ├── app.py                   # Streamlit main app
│   └── pages/
│       ├── 1_goal.py            # Goal input + harvest
│       ├── 2_quiz.py            # Adaptive quiz UI
│       ├── 3_assessment.py      # Knowledge state visualisation
│       ├── 4_path.py            # KG + path recommendation
│       └── 5_loop.py            # Learning loop + progress
├── data/benchmarks/             # ASSISTments, EdNet, Riiid! CSVs
├── scripts/
│   └── train_drl.py             # DRL-PPO training script
├── tests/
│   └── test_pipeline.py         # Integration + unit tests
├── results/                     # Benchmark outputs
├── requirements.txt
├── .env.example
└── README.md
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI · Uvicorn · Pydantic v2 |
| Frontend | Streamlit · pyvis · plotly |
| LLM | GPT-4 · LangChain · sentence-transformers |
| ML | PyTorch · Stable-Baselines3 (PPO) · gymnasium |
| KT Models | DKT (LSTM) · DKVMN · AKT · Deep-IRT |
| Knowledge Graph | Neo4j · torch_geometric (GAT) · FAISS |
| Data | Tavily Search · YouTube Data API v3 |
| Cache | Redis |

---

*Generated for Master's Thesis — AdaptLearn AI v1.0 · Paris-Saclay 2026*
