# PathOptLearn

**Modeling Learning Dynamics for Optimal Learning Pathways**

AI Learning Path Generation: given a student's history and a target (e.g. master a skill set), generate an **optimal learning path** (ordered sequence of items/exercises) that maximizes learning gain and efficiency.

**Inspired by [DeepTutor](https://github.com/HKUDS/DeepTutor)** — AI-Powered Personalized Learning Assistant. PathOptLearn uses a similar layout: **FastAPI backend** + **tabbed UI** (Dashboard, Path Generator, Chat, About).

---

## Quick start

```bash
# From repo root
pip install -r requirements.txt

# Option A: Gradio only (in-process model)
python app.py
# → Open http://127.0.0.1:7860

# Option B: Backend + Gradio (DeepTutor-style)
python scripts/start_web.py
# → Backend http://127.0.0.1:8001, UI http://127.0.0.1:7860

# Option C: Backend only (API)
python scripts/start_web.py --api
# → http://127.0.0.1:8001 (see /docs for Swagger)
```

---

## UI (DeepTutor-like tabs)

| Tab | Description |
|-----|-------------|
| **Dashboard** | Overview, reload data & retrain (when in-process). |
| **Path Generator** | Student ID, target skills, max steps → **Generate path** → table + student history. |
| **Chat** | Q&A about PathOptLearn, path generation, and the last path (e.g. “What is PathOptLearn?”, “Generate path for student 5”, “Explain the path”). |
| **About** | Short description, link to DeepTutor, link to docs. |

---

## API (FastAPI)

When running the backend (e.g. `python scripts/start_web.py --api` or `uvicorn src.api.main:app --port 8001`):

| Endpoint | Description |
|----------|-------------|
| **GET /health** | Service health. |
| **GET /api/v1/stats** | Dataset stats (n_students, n_items, n_skills, n_interactions). |
| **POST /api/v1/path/generate** | Body: `user_id`, `target_skills` (optional), `max_steps`. Returns path steps + summary + history. |
| **POST /api/v1/chat** | Body: `message`. Returns `reply`. |

Interactive docs: **http://127.0.0.1:8001/docs**

---

## Project layout (DeepTutor-like)

```
PathOptLearn/
├── app.py                 # Gradio UI (tabs: Dashboard, Path Generator, Chat, About)
├── run.py                 # CLI demo
├── requirements.txt
├── config/
│   └── app.yaml           # App and data config
├── scripts/
│   └── start_web.py       # Start backend + Gradio (--api, --ui, or both)
├── src/
│   ├── data.py            # Synthetic data, feature building
│   ├── model.py           # SuccessPredictor (sklearn)
│   ├── path_generator.py  # Greedy path generation
│   └── api/
│       ├── main.py        # FastAPI routes
│       ├── state.py      # Shared state (model, data, last path)
│       └── run_server.py # uvicorn entry
├── docs/
│   ├── LLM_approach.md    # Objectives, research papers, roadmap
│   └── ARCHITECTURE.md   # DeepTutor-like architecture and comparison
└── README.md
```

---

## MVP (core)

- **Synthetic data**: EdNet/ASSISTments-style interactions (user, item, skill, correct, timestamp).
- **Success predictor**: Logistic regression on hand-crafted features → P(correct) for (student history, item).
- **Path generator**: Greedy next-step selection (P(correct) + optional target-skill boost).

---

## Documentation

- **`docs/LLM_approach.md`**: Objectives, AI Learning Path Generation, optimal path criteria, phased approach (LLM + path generation), **research papers (bibliography)**.
- **`docs/ARCHITECTURE.md`**: Comparison with DeepTutor, layout, API, roadmap (Knowledge Base, Guided Learning, LLM, Next.js).

---

## License

See [LICENSE](LICENSE).
