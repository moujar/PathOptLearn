#!/usr/bin/env python3
"""
PathOptLearn — Gradio UI (DeepTutor-like: Dashboard, Path Generator, Chat, About).

Usage:
  python app.py                    # In-process model (default)
  API_BASE=http://localhost:8001 python app.py   # Use FastAPI backend

Or run backend + UI together:
  python scripts/start_web.py
"""

import os
import re

import pandas as pd
import gradio as gr

# Optional: use shared state from API module when backend is used
try:
    from src.api.state import get_state, N_STUDENTS, N_ITEMS, N_SKILLS
    _api_get_state = get_state
except Exception:
    _api_get_state = None
    N_STUDENTS, N_ITEMS, N_SKILLS = 200, 100, 10

from src.data import get_student_history
from src.model import SuccessPredictor
from src.path_generator import generate_path

# When running standalone (python app.py), we use local state
LOCAL_STATE = {
    "predictor": None,
    "interactions": None,
    "items": None,
    "n_skills": N_SKILLS,
    "last_path_df": None,
    "last_path_item_ids": None,
    "last_user_id": None,
    "last_summary": None,
}

API_BASE = os.environ.get("API_BASE", "").rstrip("/")


def _get_state():
    if API_BASE and _api_get_state:
        return _api_get_state()
    return LOCAL_STATE


def init_model():
    """Load data and train predictor (in-process only)."""
    if API_BASE:
        return f"Using API at {API_BASE}. No local model."
    from src.data import generate_synthetic_data
    interactions, items = generate_synthetic_data(
        n_students=N_STUDENTS, n_items=N_ITEMS, n_skills=N_SKILLS,
        n_interactions_per_student=40, seed=42,
    )
    predictor = SuccessPredictor(n_skills=N_SKILLS)
    predictor.fit(interactions, items)
    LOCAL_STATE["predictor"] = predictor
    LOCAL_STATE["interactions"] = interactions
    LOCAL_STATE["items"] = items
    LOCAL_STATE["n_skills"] = N_SKILLS
    return f"Ready: {len(interactions)} interactions, {N_STUDENTS} students, {N_ITEMS} items, {N_SKILLS} skills."


def generate_path_ui(user_id: int, target_skills_str: str, max_steps: int):
    """Generate path — via API if API_BASE set, else in-process."""
    if API_BASE:
        try:
            import requests
            target_skills = None
            if target_skills_str.strip():
                try:
                    target_skills = [int(x.strip()) for x in target_skills_str.split(",") if x.strip()]
                except ValueError:
                    pass
            r = requests.post(
                f"{API_BASE}/api/v1/path/generate",
                json={"user_id": int(user_id), "target_skills": target_skills, "max_steps": max_steps},
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
            path_df = pd.DataFrame([{"Step": s["step"], "Item ID": s["item_id"], "Skill ID": s["skill_id"], "Difficulty": s["difficulty"], "Pred. P(correct)": s["pred_p_correct"]} for s in data["path"]])
            history_df = pd.DataFrame(data["history"]) if data.get("history") else pd.DataFrame()
            return path_df, history_df, data.get("summary", "")
        except Exception as e:
            return None, None, f"API error: {e}"
    s = _get_state()
    if s["predictor"] is None:
        return None, None, "Click **Reload data & retrain** (Dashboard) or start the API first."
    target_skills = None
    if target_skills_str.strip():
        try:
            target_skills = [int(x.strip()) for x in target_skills_str.split(",") if x.strip()]
        except ValueError:
            pass
    path_item_ids, path_scores = generate_path(
        s["predictor"], s["interactions"], s["items"],
        user_id=int(user_id), target_skills=target_skills, max_steps=max_steps,
        exclude_item_ids=None, random_state=42,
    )
    items = s["items"]
    rows = []
    for i, (item_id, score) in enumerate(zip(path_item_ids, path_scores), 1):
        row = items[items["item_id"] == item_id].iloc[0]
        rows.append({"Step": i, "Item ID": int(item_id), "Skill ID": int(row["skill_id"]), "Difficulty": round(float(row["difficulty"]), 2), "Pred. P(correct)": round(score, 3)})
    path_df = pd.DataFrame(rows)
    s["last_path_df"] = path_df
    s["last_path_item_ids"] = path_item_ids
    s["last_user_id"] = user_id
    s["last_summary"] = f"Path for user {user_id}: {len(path_item_ids)} items."
    hist = get_student_history(s["interactions"], int(user_id), max_len=10)
    history_df = hist[["item_id", "skill_id", "correct", "timestamp"]].copy() if len(hist) > 0 else pd.DataFrame()
    history_df.columns = ["Item ID", "Skill ID", "Correct", "Timestamp"]
    return path_df, history_df, s["last_summary"]


def chat_reply(message: str, history: list) -> tuple:
    """Answer user — via API if API_BASE set, else in-process. Returns (new_history, '')."""
    if API_BASE:
        try:
            import requests
            r = requests.post(f"{API_BASE}/api/v1/chat", json={"message": message}, timeout=15)
            r.raise_for_status()
            reply = r.json().get("reply", "")
        except Exception as e:
            reply = f"API error: {e}"
    else:
        reply = _chat_reply_inprocess(message)
    new_history = history + [(message, reply)]
    return new_history, ""


def _chat_reply_inprocess(message: str) -> str:
    """In-process chat logic (same as before)."""
    msg = (message or "").strip().lower()
    s = _get_state()
    if "path for student" in msg or "path for user" in msg or "generate path" in msg:
        m = re.search(r"(?:student|user)\s*(\d+)", msg, re.I)
        uid = int(m.group(1)) if m else 0
        if s["predictor"] is None:
            return "Please load the model first (Dashboard → Reload)."
        path_item_ids, path_scores = generate_path(
            s["predictor"], s["interactions"], s["items"],
            user_id=min(max(0, uid), N_STUDENTS - 1), target_skills=None, max_steps=8,
            exclude_item_ids=None, random_state=42,
        )
        items = s["items"]
        lines = [f"**Path for student {uid}** ({len(path_item_ids)} steps):"]
        path_rows = []
        for i, (item_id, score) in enumerate(zip(path_item_ids, path_scores), 1):
            row = items[items["item_id"] == item_id].iloc[0]
            lines.append(f"  Step {i}: item {item_id}, skill {int(row['skill_id'])}, difficulty {row['difficulty']:.2f}, P(correct)={score:.3f}")
            path_rows.append({"Step": i, "Item ID": int(item_id), "Skill ID": int(row["skill_id"]), "Difficulty": round(float(row["difficulty"]), 2), "Pred. P(correct)": round(score, 3)})
        s["last_path_df"] = pd.DataFrame(path_rows)
        s["last_path_item_ids"] = path_item_ids
        s["last_user_id"] = uid
        s["last_summary"] = f"Path for user {uid}: {len(path_item_ids)} items."
        return "\n".join(lines)
    if "what is pathoptlearn" in msg or "what is this app" in msg or "what is this" in msg:
        return "**PathOptLearn** is an AI system that generates **optimal learning paths** for students. Given history and target skills, it outputs an ordered sequence of items that maximizes learning gain. Use the **Path Generator** tab to generate a path."
    if "how does it work" in msg or "how does path" in msg or "how do you" in msg:
        return "**How it works:** 1) **Success predictor** predicts P(correct) for (student history, item). 2) **Path generation** uses **greedy** selection: at each step, pick the item with highest predicted value (with boost for target skills). 3) Path is personalized from prior interactions."
    if "explain the path" in msg or "explain this path" in msg or "why this path" in msg:
        if s["last_path_df"] is not None and len(s["last_path_df"]) > 0:
            return f"The last path was for **student {s['last_user_id']}** and has {len(s['last_path_df'])} steps. Each step was chosen to maximize P(correct), with preference for target skills."
        return "No path generated yet. Generate a path in the **Path Generator** tab, then ask me to explain it."
    if "explain step" in msg:
        m = re.search(r"step\s*(\d+)", msg, re.I)
        step_num = int(m.group(1)) if m else None
        if s["last_path_df"] is not None and step_num and 1 <= step_num <= len(s["last_path_df"]):
            row = s["last_path_df"].iloc[step_num - 1]
            return f"**Step {step_num}**: Item {row.get('Item ID')}, Skill {row.get('Skill ID')}, difficulty {row.get('Difficulty')}, P(correct)={row.get('Pred. P(correct)'):.3f}. Chosen for best predicted learning value."
        return "Generate a path first, then ask e.g. 'Explain step 1'."
    if "how many student" in msg or "how many user" in msg:
        return f"There are **{N_STUDENTS}** students (user_id 0 to {N_STUDENTS - 1}) in the synthetic dataset."
    if "how many item" in msg:
        return f"There are **{N_ITEMS}** items and **{N_SKILLS}** skills in the synthetic dataset."
    return "I can answer questions about **PathOptLearn**, **path generation**, and the **last path**. Try: \"What is PathOptLearn?\", \"How does it work?\", \"Generate path for student 5\", \"Explain the path\"."


def build_ui(initial_status: str = "Loading…"):
    with gr.Blocks(title="PathOptLearn — Optimal Learning Path", theme=gr.themes.Soft(), css=".gradio-container { max-width: 900px; }") as demo:
        gr.Markdown("# PathOptLearn — AI Learning Path Generation")
        gr.Markdown("Generate **optimal learning paths** for students. *(DeepTutor-like layout)*")

        status = gr.Markdown(initial_status, visible=not bool(API_BASE))
        load_btn = gr.Button("Reload data & retrain model", variant="primary", visible=not bool(API_BASE))

        with gr.Tabs():
            # --- Dashboard ---
            with gr.TabItem("Dashboard", id=0):
                gr.Markdown("### Overview")
                load_btn.click(fn=init_model, outputs=[status])
                gr.Markdown("- **Path Generator**: Generate a personalized path for a student.\n- **Chat**: Ask questions about PathOptLearn and the current path.\n- **About**: Documentation and roadmap.")

            # --- Path Generator ---
            with gr.TabItem("Path Generator", id=1):
                gr.Markdown("### Generate optimal learning path")
                with gr.Row():
                    user_id = gr.Number(value=0, minimum=0, maximum=N_STUDENTS - 1, step=1, label="Student ID", precision=0)
                    target_skills_str = gr.Textbox(value="0, 1, 2", label="Target skills (comma-separated). Empty = all.")
                    max_steps = gr.Slider(1, 20, value=8, step=1, label="Max path length")
                gen_btn = gr.Button("Generate path", variant="primary")
                summary_out = gr.Markdown()
                path_table = gr.Dataframe(label="Generated path", interactive=False)
                history_table = gr.Dataframe(label="Student prior history (last 10)", interactive=False)
                gen_btn.click(fn=generate_path_ui, inputs=[user_id, target_skills_str, max_steps], outputs=[path_table, history_table, summary_out])

            # --- Chat ---
            with gr.TabItem("Chat", id=2):
                gr.Markdown("### Questions & Answers")
                chatbot = gr.Chatbot(label="Chat", height=380)
                msg_in = gr.Textbox(placeholder="e.g. What is PathOptLearn? / Generate path for student 5 / Explain the path", show_label=False, container=False)
                def submit_chat(message, history):
                    if not (message or "").strip():
                        return history, ""
                    return chat_reply(message, history)
                msg_in.submit(submit_chat, inputs=[msg_in, chatbot], outputs=[chatbot, msg_in])
                gr.Examples(["What is PathOptLearn?", "How does it work?", "Generate path for student 5", "Explain the path", "Explain step 1"], inputs=msg_in, label="Example questions")

            # --- About ---
            with gr.TabItem("About", id=3):
                gr.Markdown("### PathOptLearn")
                gr.Markdown(
                    "**Modeling Learning Dynamics for Optimal Learning Pathways**\n\n"
                    "MVP: Synthetic data, logistic regression success predictor, greedy path generation. "
                    "See `docs/LLM_approach.md` for objectives, research papers, and roadmap.\n\n"
                    "**Inspired by [DeepTutor](https://github.com/HKUDS/DeepTutor)**: AI-powered personalized learning assistant (RAG, multi-agent, guided learning). "
                    "PathOptLearn focuses on **optimal learning path generation**; future: Knowledge Base, Guided Learning, LLM integration."
                )

    return demo


if __name__ == "__main__":
    if not API_BASE:
        initial_status = init_model()
    else:
        initial_status = f"Using API at {API_BASE}"
    demo = build_ui(initial_status=initial_status)
    port = int(os.environ.get("FRONTEND_PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)
