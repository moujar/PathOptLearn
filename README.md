

### Personalized Learning Assistant



[] Massive Document Knowledge Q&A


[] Knowledge Reinforcement with Practice Exercise Generator
• **Intelligent Exercise Creation**: Generate targeted quizzes, practice problems, and customized assessments tailored to your current knowledge level and specific learning objectives.<br>
• **Authentic Exam Simulation**: Upload reference exams to generate practice questions that perfectly match the original style, format, and difficulty—giving you realistic preparation for the actual test.

[] Deep Research & Idea Generation
• **Comprehensive Research & Literature Review**: Conduct in-depth topic exploration with systematic analysis. Identify patterns, connect related concepts across disciplines, and synthesize existing research findings.<br>
• **Novel Insight Discovery**: Generate structured learning materials and uncover knowledge gaps. Identify promising new research directions through intelligent cross-domain knowledge synthesis.


## 📋 Todo
> 🌟 Star to follow our future updates!
- [ x ] Multi-linguistic support
- [ x ] Video & Audio file support
- [ x-50% ] Atomic RAG pipeline customize
- [ - ] Incremental Knowledge-base Edit
- [ - ] Personalized Workspace
- [ - ] DataBase Visualization
- [ - ] Online Demo


```bash
git clone https://github.com/moujar/PathOptLearn
cd PathOptLearn
```


```bash
cp .env.example .env
```

```bash
docker compose up                  # Build and start (~11 min first run on mac mini M4)
docker compose build --no-cache    # Clear cache and rebuild after pull the newest repo
```


```bash
# Works on all platforms - Docker auto-detects your architecture
docker run -d --name PathOptLearn \
  -p 8001:8001 -p 3782:3782 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/config:/app/config:ro \
  ghcr.io/ParisSaclay/PathOptLearn:latest

# Windows PowerShell: use ${PWD} instead of $(pwd)
```

**Common Commands**:

```bash
docker compose up -d      # Start
docker compose down       # Stop
docker compose logs -f    # View logs
docker compose up --build # Rebuild after changes
```


```bash
docker run -d --name PathOptLearn \
  -p 8001:8001 -p 3782:3782 \
  -e NEXT_PUBLIC_API_BASE_EXTERNAL=https://your-server.com:8001 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  ghcr.io/ParisSaclay/PathOptLearn:latest
```



```bash
docker run -d --name PathOptLearn \
  -p 9001:9001 -p 3000:3000 \
  -e BACKEND_PORT=9001 \
  -e FRONTEND_PORT=3000 \
  -e NEXT_PUBLIC_API_BASE_EXTERNAL=https://your-server.com:9001 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  ghcr.io/ParisSaclay/PathOptLearn:latest
```

</details>

---


```bash
# Using conda (Recommended)
conda create -n PathOptLearn python=3.10 && conda activate PathOptLearn

# Or using venv
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
```

**2. Install Dependencies**:

```bash
# One-click installation (Recommended)
python scripts/install_all.py
# Or: bash scripts/install_all.sh

# Or manual installation
pip install -r requirements.txt
npm install --prefix web
```

**3. Launch**:

```bash
python scripts/start_web.py    # Start frontend + backend
# Or: python scripts/start.py  # CLI only
# Stop: Ctrl+C
```


```bash
python src/api/run_server.py
# Or: uvicorn src.api.main:app --host 0.0.0.0 --port 8001 --reload
```

**Frontend** (Next.js):
```bash
cd web && npm install && npm run dev -- -p 3782
```

**Note**: Create `web/.env.local`:
```bash
NEXT_PUBLIC_API_BASE=http://localhost:8001
```

| Service | Default Port |
|:---:|:---:|
| Backend | `8001` |
| Frontend | `3782` |

</details>

### Access URLs

| Service | URL | Description |
|:---:|:---|:---|
| **Frontend** | http://localhost:3782 | Main web interface |
| **API Docs** | http://localhost:8001/docs | Interactive API documentation |

---

## 📂 Data Storage

All user content and system data are stored in the `data/` directory:

```
data/
├── knowledge_bases/              # Knowledge base storage
└── user/                         # User activity data
    ├── solve/                    # Problem solving results and artifacts
    ├── question/                 # Generated questions
    ├── research/                 # Research reports and cache
    ├── co-writer/                # Interactive IdeaGen documents and audio files
    ├── notebook/                 # Notebook records and metadata
    ├── guide/                    # Guided learning sessions
    ├── logs/                     # System logs
    └── run_code_workspace/       # Code execution workspace
```