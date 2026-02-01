# PathOptLearn Architecture (DeepTutor-like)

PathOptLearn is structured similarly to **[DeepTutor](https://github.com/HKUDS/DeepTutor)** (AI-Powered Personalized Learning Assistant) while focusing on **optimal learning path generation**.

---

## Comparison with DeepTutor

| Aspect | DeepTutor | PathOptLearn |
|--------|-----------|--------------|
| **Focus** | All-in-one tutoring: RAG Q&A, guided learning, question gen, deep research, idea gen | **Optimal learning path generation** (sequence of items for a student) |
| **Backend** | FastAPI (Python) | FastAPI (Python) |
| **Frontend** | Next.js + React | Gradio (MVP); optional Next.js later |
| **Data** | Knowledge bases (PDF, docs), notebooks | Synthetic / EdNet/ASSISTments-style interactions |
| **Core modules** | Smart Solver, Question Gen, Guided Learning, Co-Writer, Deep Research, IdeaGen | Path Generator, Chat Q&A, Dashboard, About |
| **Config** | `.env`, `config/agents.yaml` | `config/app.yaml`, env (BACKEND_PORT, FRONTEND_PORT) |
| **Scripts** | `scripts/start_web.py`, `scripts/install_all.py` | `scripts/start_web.py`, `app.py`, `run.py` |

---

## Current layout

```
PathOptLearn/
├── app.py                 # Gradio UI (tabs: Dashboard, Path Generator, Chat, About)
├── run.py                 # CLI demo
├── requirements.txt
├── config/
│   └── app.yaml           # App and data config
├── scripts/
│   └── start_web.py       # Start backend + Gradio (DeepTutor-style)
├── src/
│   ├── data.py            # Synthetic data, feature building
│   ├── model.py           # SuccessPredictor (sklearn)
│   ├── path_generator.py  # Greedy path generation
│   └── api/
│       ├── main.py        # FastAPI: /health, /api/v1/path/generate, /api/v1/chat, /api/v1/stats
│       ├── state.py      # Shared state (model, data, last path)
│       └── run_server.py  # uvicorn entry
├── docs/
│   ├── LLM_approach.md    # Objectives, research papers, roadmap
│   └── ARCHITECTURE.md    # This file
└── README.md
```

---

## User flow (DeepTutor-like)

1. **Dashboard** — Overview, reload model (when in-process), quick links to Path Generator and Chat.
2. **Path Generator** — Student ID, target skills, max steps → **Generate path** → table + student history.
3. **Chat** — Q&A about PathOptLearn, path generation, and the last path (e.g. “Explain the path”, “Generate path for student 5”).
4. **About** — Short description, link to DeepTutor, link to `docs/LLM_approach.md`.

---

## API (FastAPI)

- **GET /health** — Service health.
- **GET /api/v1/stats** — Dataset stats (n_students, n_items, n_skills, n_interactions).
- **POST /api/v1/path/generate** — Body: `user_id`, `target_skills` (optional), `max_steps`. Returns path steps + summary + history.
- **POST /api/v1/chat** — Body: `message`. Returns `reply`.

When running `python scripts/start_web.py` (no `--api`/`--ui`), the backend runs on port 8001 and Gradio on 7860; Gradio uses `API_BASE=http://127.0.0.1:8001` to call the backend.

---

## Roadmap (toward DeepTutor-like features)

- [ ] **Knowledge Base** — Upload documents (e.g. textbooks); use RAG to inform path content (skills, prerequisites).
- [ ] **Guided Learning** — Step-by-step interactive pages (like DeepTutor’s Guided Learning) driven by the generated path.
- [ ] **LLM integration** — Use an LLM for item/skill descriptions, explanations, and chat (see `docs/LLM_approach.md`).
- [ ] **Next.js frontend** — Replace or complement Gradio with a React/Next.js UI (like DeepTutor).
- [ ] **Notebook / session history** — Persist paths and chat per user/session.
- [ ] **Real datasets** — EdNet, ASSISTments, Riiid (see `docs/LLM_approach.md`).

---

## References

- [DeepTutor](https://github.com/HKUDS/DeepTutor) — AI-Powered Personalized Learning Assistant (HKUDS).
- PathOptLearn **objectives and research papers**: `docs/LLM_approach.md`.
