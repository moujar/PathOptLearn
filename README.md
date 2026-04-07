# PathOptLearn

An AI-powered adaptive learning platform that builds a personalised learning roadmap from a single topic input. It diagnoses your knowledge level, identifies gaps, generates structured lessons, and validates understanding through quizzes — all driven by an LLM.

---

## Learning Flow

```text
You enter a topic
      │
      ▼
  [1] Check topic on Knowledge Graph  OR  [2] Search DuckDuckGo to enrich context
      │
      ▼
Diagnostic Quiz — 8 questions (MCQ + short-answer, mixed difficulty)
      │
      ▼
Submit answers → LLM analyses level + identifies gaps
      │
      ▼
  [1] Fetch resources from KG if they exist  OR  [2] Deep web search (DuckDuckGo + edu sites + YouTube)
      │
      ▼
Store enriched resources in Knowledge Graph
      │
      ▼
Personalised roadmap generated (4–6 modules, progressive)
      │
      ▼
For each module:
  AI-generated lesson  →  5-question quiz  →  score ≥ 70%  →  next module
                                           └→  score < 70%  →  find gaps
                                                                    │
                                                                    ▼
                                                         suggest best resources for gaps
                                                                    │
                                                                    ▼
                                                              review & retry
      │
      ▼
 All modules complete 🏆
```

---

## Stack

| Layer | Technology |
| --- | --- |
| UI | Streamlit |
| LLM | Groq (default: `llama-3.3-70b-versatile`) or Ollama (local) |
| Web search | DuckDuckGo (`ddgs`) |
| Video search | `yt-dlp` |
| Database | SQLite |
| Auth | `bcrypt` + email verification |

---

## Project Structure

```text
production/
├── src/
│   ├── main.py              # Streamlit entry point
│   ├── config.py            # Environment config
│   ├── db.py                # SQLite helpers (auth, courses, progress)
│   ├── mailer.py            # Email verification sender
│   ├── graph/
│   │   └── graph.py         # LearnFlow state machine — all graph nodes
│   ├── api/
│   │   └── endpoits.py      # Web search & YouTube helpers
│   └── pages/
│       ├── 1_Login.py       # Auth page (login / register / verify)
│       ├── 2_Dashboard.py   # Student dashboard
│       ├── 3_Learning.py    # Course-based learning (video + quiz)
│       └── 4_LearnFlow.py   # Adaptive LearnFlow (main flow)
├── db/
│   └── center.db            # SQLite database
├── .env-example
├── requirements.txt
└── docker-compose.yml
```

---

## Setup

### 1. Environment variables

Copy `.env-example` to `.env` and fill in the values:

```env
LLM_PROVIDER=groq          # "groq" or "ollama"
GROQ_API_KEY=your_key
GROQ_MODEL=llama-3.3-70b-versatile

OLLAMA_MODEL=llama3.2:1b   # only if LLM_PROVIDER=ollama

SENDER_EMAIL=your@gmail.com
SENDER_PASSWORD=app_password

COLAB_API_URL=             # optional remote inference endpoint
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

For local LLM (Ollama), uncomment `ollama>=0.2.0` in `requirements.txt`.

### 3. Run

```bash
streamlit run production/src/main.py
```

Or with Docker:

```bash
docker-compose up --build
```

---

## Graph Node API

These are the internal functions in [production/src/graph/graph.py](production/src/graph/graph.py) that drive the learning flow. Each node takes a `LearnFlowState` and returns an updated state.

---

### `node_generate_assessment(state)`

**Stage:** Topic → Diagnostic Quiz

Generates 8 diagnostic questions for the given topic.

**Input state fields used:**

- `state.topic` — the subject to learn

**Output state fields set:**

- `state.assessment_questions` — list of up to 8 question objects

**Question object schema:**

```json
{
  "q": "Question text",
  "type": "mcq | short",
  "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
  "answer": "correct answer",
  "concept": "concept being tested"
}
```

**Mix:** 2 easy, 4 medium, 2 hard. MCQ and short-answer combined.

---

### `node_analyze_level(state)`

**Stage:** Quiz answers → Level + Gaps

Sends the answered quiz to the LLM for evaluation. Determines learner level and identifies knowledge gaps.

**Input state fields used:**

- `state.assessment_questions`
- `state.assessment_answers`
- `state.topic`

**Output state fields set:**

- `state.level` — `"beginner"`, `"intermediate"`, or `"advanced"`
- `state.gaps` — list of gap strings (e.g. `["gradient descent", "overfitting"]`)

**LLM response schema:**

```json
{
  "level": "beginner | intermediate | advanced",
  "gaps": ["gap1", "gap2", "gap3"],
  "score": 0
}
```

---

### `node_research_resources(state)`

**Stage:** Gaps → Web resources

Searches the web for resources targeted at the learner's level and their specific gaps using DuckDuckGo. Also queries educational domains (Coursera, edX, Khan Academy, MIT OCW, Stanford).

**Input state fields used:**

- `state.topic`
- `state.level`
- `state.gaps`

**Output state fields set:**

- `state.resources` — list of up to 12 resource objects

**Resource object schema:**

```json
{
  "title": "Page title",
  "url": "https://...",
  "snippet": "Short description"
}
```

---

### `node_build_roadmap(state)`

**Stage:** Resources + Gaps → Personalised roadmap

Builds a structured learning roadmap of 4–6 progressive modules tailored to the learner's level and gaps.

**Input state fields used:**

- `state.topic`
- `state.level`
- `state.gaps`
- `state.resources`

**Output state fields set:**

- `state.roadmap` — list of module objects
- `state.current_module` — reset to `0`

**Module object schema:**

```json
{
  "module": 1,
  "title": "Introduction to ...",
  "objective": "Understand the core concepts of ...",
  "concepts": ["concept1", "concept2"],
  "duration": "30 min"
}
```

---

### `node_generate_content(state)`

**Stage:** Module → Lesson content

Generates a full markdown lesson for the current module using the LLM, grounded in the researched resources.

**Input state fields used:**

- `state.current_module` (index into `state.roadmap`)
- `state.roadmap`
- `state.resources`
- `state.level`

**Output state fields set:**

- `state.content` — markdown string
- `state.attempts` — reset to `0`

**Lesson structure:**

```text
## Overview
## Key Concepts (with examples)
## Practical Application
## Summary
```

---

### `node_generate_quiz(state)`

**Stage:** Lesson → Module quiz

Generates 5 questions to test understanding of the lesson just delivered.

**Input state fields used:**

- `state.current_module`
- `state.roadmap`
- `state.content`

**Output state fields set:**

- `state.quiz` — list of up to 5 question objects (same schema as assessment questions)

**Mix:** 3 multiple-choice + 2 short-answer.

---

### `node_evaluate_quiz(state) → dict`

**Stage:** Quiz answers → Score + Feedback + Resources

Grades the module quiz, fetches YouTube videos and educational resources relevant to the module.

**Input state fields used:**

- `state.quiz`
- `state.quiz_answers`
- `state.topic`
- `state.level`
- `state.current_module`

**Output state fields set:**

- `state.quiz_score`
- `state.attempts` (incremented)

**Returns a result dict** (not stored in state directly):

```json
{
  "score": 85,
  "passed": true,
  "attempts": 1,
  "feedback": [
    { "q": 1, "correct": true, "explanation": "..." }
  ],
  "videos": [
    {
      "id": "abc123",
      "title": "Video title",
      "channel": "Channel name",
      "duration": "12:34",
      "views": 100000,
      "url": "https://www.youtube.com/watch?v=abc123",
      "thumb": "https://img.youtube.com/vi/abc123/hqdefault.jpg"
    }
  ],
  "edu_resources": [
    { "title": "...", "url": "...", "snippet": "..." }
  ]
}
```

**Passing threshold:** 70/100. After 3 failed attempts the module is force-passed.

---

### `node_advance(state)`

**Stage:** Pass → Next module (or completion)

Advances the learner to the next module. If all modules are done, marks the session as complete.

**Input state fields used:**

- `state.current_module`
- `state.roadmap`

**Output state fields set:**

- `state.current_module` — incremented by 1
- `state.completed` — `True` if all modules finished

---

## Session Store

Sessions are kept in memory (Python dict, thread-safe) keyed by `session_id` (UUID). Each session is a `LearnFlowState` dataclass.

| Function | Description |
| --- | --- |
| `get_session(session_id)` | Retrieve a session by ID, raises `KeyError` if not found |
| `save_session(state)` | Persist the updated state |
| `list_sessions()` | Return a summary list of all active sessions |

---

## Search & YouTube Helpers

Located in [production/src/api/endpoits.py](production/src/api/endpoits.py).

| Function | Description |
| --- | --- |
| `search_web(query, max_results=6)` | DuckDuckGo text search, returns list of `{title, url, snippet}` |
| `fetch_page(url)` | Fetches and strips a web page to plain text |
| `chunk_text(text, chunk_size=1500)` | Splits text into overlapping chunks for LLM context |
| `quick_search_summary(topic, llm_fn)` | Searches + summarises results via LLM |
| `edu_resources(query)` | Searches educational platforms only (Coursera, edX, MIT, Stanford…) |
| `search_youtube(query, max_results=4)` | Returns YouTube video metadata via `yt-dlp` |

---

## LLM Configuration

Located in [production/src/llm/llm.py](production/src/llm/llm.py).

The `llm(prompt, system="")` function is the single entry point for all LLM calls. It routes based on `LLM_PROVIDER`:

- **`groq`** (default) — calls `https://api.groq.com/openai/v1/chat/completions`, requires `GROQ_API_KEY`
- **`ollama`** — calls a local Ollama instance, requires the `ollama` Python package

---

## Database Schema

Managed by [production/src/db.py](production/src/db.py) using SQLite.

### `students`

| Column | Type | Description |
| --- | --- | --- |
| `id` | INTEGER PK | Auto-increment |
| `username` | TEXT UNIQUE | Login name |
| `email` | TEXT UNIQUE | Used for verification |
| `password_hash` | TEXT | bcrypt hash |
| `verified` | INTEGER | 0 = unverified, 1 = verified |
| `verification_code` | TEXT | 6-digit code |
| `code_expiry` | TEXT | 15-minute expiry |
| `created_at` | TEXT | ISO timestamp |

### `courses`

| Column | Type | Description |
| --- | --- | --- |
| `id` | INTEGER PK | |
| `student_id` | INTEGER FK | |
| `name` | TEXT | Course / topic name |
| `created_at` | TEXT | |

### `progress`

| Column | Type | Description |
| --- | --- | --- |
| `id` | INTEGER PK | |
| `student_id` | INTEGER FK | |
| `course_id` | INTEGER FK | |
| `video_id` | TEXT | YouTube video ID |
| `title` | TEXT | Lesson/video title |
| `score` | INTEGER | Raw score |
| `total` | INTEGER | Total possible |
| `passed` | INTEGER | 0 or 1 |
| `timestamp` | TEXT | ISO timestamp |
