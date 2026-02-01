"""
PathOptLearn FastAPI backend — like DeepTutor API layer.

Endpoints:
  GET  /health
  POST /api/v1/path/generate
  POST /api/v1/chat
  GET  /api/v1/stats
"""

import re
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.api.state import get_state, init_model, N_STUDENTS, N_ITEMS, N_SKILLS
from src.data import get_student_history
from src.path_generator import generate_path


# --- Request/Response models ---

class PathGenerateRequest(BaseModel):
    user_id: int = Field(0, ge=0, description="Student ID")
    target_skills: Optional[list[int]] = Field(None, description="Skill IDs to prioritize")
    max_steps: int = Field(8, ge=1, le=20, description="Max path length")


class PathStep(BaseModel):
    step: int
    item_id: int
    skill_id: int
    difficulty: float
    pred_p_correct: float


class PathGenerateResponse(BaseModel):
    user_id: int
    path: list[PathStep]
    summary: str
    history: list[dict]


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    reply: str


# --- App ---

app = FastAPI(
    title="PathOptLearn API",
    description="AI Learning Path Generation — optimal learning pathways",
    version="0.1.0",
)


@app.on_event("startup")
def startup():
    init_model()


@app.get("/health")
def health():
    return {"status": "ok", "service": "pathoptlearn"}


@app.get("/api/v1/stats")
def stats():
    s = get_state()
    if s["interactions"] is None:
        raise HTTPException(503, "Model not loaded")
    return {
        "n_students": N_STUDENTS,
        "n_items": N_ITEMS,
        "n_skills": N_SKILLS,
        "n_interactions": len(s["interactions"]),
    }


@app.post("/api/v1/path/generate", response_model=PathGenerateResponse)
def api_path_generate(req: PathGenerateRequest):
    s = get_state()
    if s["predictor"] is None:
        raise HTTPException(503, "Model not loaded")
    path_item_ids, path_scores = generate_path(
        s["predictor"],
        s["interactions"],
        s["items"],
        user_id=req.user_id,
        target_skills=req.target_skills,
        max_steps=req.max_steps,
        exclude_item_ids=None,
        random_state=42,
    )
    items = s["items"]
    path_steps = []
    path_rows = []
    for i, (item_id, score) in enumerate(zip(path_item_ids, path_scores), 1):
        row = items[items["item_id"] == item_id].iloc[0]
        path_steps.append(PathStep(
            step=i,
            item_id=int(item_id),
            skill_id=int(row["skill_id"]),
            difficulty=round(float(row["difficulty"]), 2),
            pred_p_correct=round(score, 3),
        ))
        path_rows.append({
            "Step": i,
            "Item ID": int(item_id),
            "Skill ID": int(row["skill_id"]),
            "Difficulty": round(float(row["difficulty"]), 2),
            "Pred. P(correct)": round(score, 3),
        })
    path_df = pd.DataFrame(path_rows)
    s["last_path_df"] = path_df
    s["last_path_item_ids"] = path_item_ids
    s["last_user_id"] = req.user_id
    s["last_summary"] = f"Path for user {req.user_id}: {len(path_item_ids)} items."
    hist = get_student_history(s["interactions"], req.user_id, max_len=10)
    history = hist.to_dict("records") if len(hist) > 0 else []
    return PathGenerateResponse(
        user_id=req.user_id,
        path=path_steps,
        summary=s["last_summary"],
        history=history,
    )


def _chat_reply(message: str) -> str:
    """Same logic as app.py chat_reply, returns only the reply text."""
    msg = (message or "").strip().lower()
    s = get_state()
    reply = ""

    if "path for student" in msg or "path for user" in msg or "generate path" in msg:
        m = re.search(r"(?:student|user)\s*(\d+)", msg, re.I)
        uid = int(m.group(1)) if m else 0
        if s["predictor"] is None:
            reply = "Please load the model first. Then I can generate a path for you."
        else:
            path_item_ids, path_scores = generate_path(
                s["predictor"], s["interactions"], s["items"],
                user_id=min(max(0, uid), N_STUDENTS - 1),
                target_skills=None, max_steps=8, exclude_item_ids=None, random_state=42,
            )
            items = s["items"]
            lines = [f"**Path for student {uid}** ({len(path_item_ids)} steps):"]
            path_rows = []
            for i, (item_id, score) in enumerate(zip(path_item_ids, path_scores), 1):
                row = items[items["item_id"] == item_id].iloc[0]
                lines.append(f"  Step {i}: item {item_id}, skill {int(row['skill_id'])}, difficulty {row['difficulty']:.2f}, P(correct)={score:.3f}")
                path_rows.append({"Step": i, "Item ID": int(item_id), "Skill ID": int(row["skill_id"]), "Difficulty": round(float(row["difficulty"]), 2), "Pred. P(correct)": round(score, 3)})
            reply = "\n".join(lines)
            s["last_path_df"] = pd.DataFrame(path_rows)
            s["last_path_item_ids"] = path_item_ids
            s["last_user_id"] = uid
            s["last_summary"] = f"Path for user {uid}: {len(path_item_ids)} items."

    elif "what is pathoptlearn" in msg or "what is this app" in msg or "what is this" in msg:
        reply = (
            "**PathOptLearn** is an AI system that generates **optimal learning paths** for students. "
            "Given a student's history and a target (e.g. master certain skills), "
            "it outputs an ordered sequence of items that maximizes learning gain and efficiency. "
            "You can generate a path in the **Path Generator** tab."
        )
    elif "how does it work" in msg or "how does path" in msg or "how do you" in msg:
        reply = (
            "**How it works:**\n"
            "1. **Success predictor**: Predicts P(correct) for each (student history, item).\n"
            "2. **Path generation**: **Greedy** selection — at each step, pick the item with highest predicted value (with boost for target skills).\n"
            "3. Path is personalized from the student's prior interactions."
        )
    elif "explain the path" in msg or "explain this path" in msg or "why this path" in msg:
        if s["last_path_df"] is not None and len(s["last_path_df"]) > 0:
            reply = (
                f"The last path was for **student {s['last_user_id']}** and has {len(s['last_path_df'])} steps. "
                "Each step was chosen to maximize predicted P(correct), with preference for target skills."
            )
        else:
            reply = "No path generated yet. Generate a path in the **Path Generator** tab, then ask me to explain it."
    elif "explain step" in msg:
        m = re.search(r"step\s*(\d+)", msg, re.I)
        step_num = int(m.group(1)) if m else None
        if s["last_path_df"] is not None and step_num and 1 <= step_num <= len(s["last_path_df"]):
            row = s["last_path_df"].iloc[step_num - 1]
            reply = f"**Step {step_num}**: Item {row.get('Item ID')}, Skill {row.get('Skill ID')}, difficulty {row.get('Difficulty')}, P(correct)={row.get('Pred. P(correct)'):.3f}. Chosen for best predicted learning value."
        else:
            reply = "Generate a path first, then ask e.g. 'Explain step 1'."
    elif "how many student" in msg or "how many user" in msg:
        reply = f"There are **{N_STUDENTS}** students (user_id 0 to {N_STUDENTS - 1}) in the synthetic dataset."
    elif "how many item" in msg:
        reply = f"There are **{N_ITEMS}** items and **{N_SKILLS}** skills in the synthetic dataset."
    else:
        reply = (
            "I can answer questions about **PathOptLearn**, **path generation**, and the **last path**. "
            "Try: \"What is PathOptLearn?\", \"How does it work?\", \"Generate path for student 5\", \"Explain the path\"."
        )
    return reply


@app.post("/api/v1/chat", response_model=ChatResponse)
def api_chat(req: ChatRequest):
    reply = _chat_reply(req.message)
    return ChatResponse(reply=reply)
