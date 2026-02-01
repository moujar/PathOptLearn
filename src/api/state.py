"""
Shared app state: model, data, last path. Used by both FastAPI and Gradio.
"""

import pandas as pd
from typing import Any, Optional

# Config
N_STUDENTS = 200
N_ITEMS = 100
N_SKILLS = 10
N_INTERACTIONS_PER_STUDENT = 40
SEED = 42

_state: dict[str, Any] = {
    "predictor": None,
    "interactions": None,
    "items": None,
    "n_skills": N_SKILLS,
    "last_path_df": None,
    "last_path_item_ids": None,
    "last_user_id": None,
    "last_summary": None,
}


def get_state() -> dict:
    return _state


def init_model() -> str:
    from src.data import generate_synthetic_data
    from src.model import SuccessPredictor

    interactions, items = generate_synthetic_data(
        n_students=N_STUDENTS,
        n_items=N_ITEMS,
        n_skills=N_SKILLS,
        n_interactions_per_student=N_INTERACTIONS_PER_STUDENT,
        seed=SEED,
    )
    predictor = SuccessPredictor(n_skills=N_SKILLS)
    predictor.fit(interactions, items)
    _state["predictor"] = predictor
    _state["interactions"] = interactions
    _state["items"] = items
    _state["n_skills"] = N_SKILLS
    return f"Ready: {len(interactions)} interactions, {N_STUDENTS} students, {N_ITEMS} items, {N_SKILLS} skills."
