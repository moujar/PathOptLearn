#!/usr/bin/env python3
"""
PathOptLearn MVP: generate optimal learning paths from synthetic data.

Usage:
  pip install -r requirements.txt
  python run.py
"""

import numpy as np
import pandas as pd

from src.data import generate_synthetic_data, get_student_history
from src.model import SuccessPredictor
from src.path_generator import generate_path


def main():
    print("PathOptLearn MVP — Optimal Learning Path Generation\n")

    # 1. Generate synthetic data (EdNet/ASSISTments-style)
    n_students, n_items, n_skills = 200, 100, 10
    interactions, items = generate_synthetic_data(
        n_students=n_students,
        n_items=n_items,
        n_skills=n_skills,
        n_interactions_per_student=40,
        seed=42,
    )
    print(f"Data: {len(interactions)} interactions, {n_students} students, {n_items} items, {n_skills} skills")

    # 2. Train success predictor (P(correct) given history + item)
    predictor = SuccessPredictor(n_skills=n_skills)
    predictor.fit(interactions, items)
    print("Success predictor trained (logistic regression on hand-crafted features).")

    # 3. Pick a demo student and generate a path
    demo_user = 0
    target_skills = [0, 1, 2]  # prioritize mastering skills 0, 1, 2
    max_steps = 8

    path_item_ids, path_scores = generate_path(
        predictor,
        interactions,
        items,
        user_id=demo_user,
        target_skills=target_skills,
        max_steps=max_steps,
        exclude_item_ids=None,
        random_state=42,
    )

    print(f"\nGenerated path for user_id={demo_user} (target_skills={target_skills}, max_steps={max_steps}):")
    print("-" * 50)
    for i, (item_id, score) in enumerate(zip(path_item_ids, path_scores), 1):
        skill_id = items.loc[items["item_id"] == item_id, "skill_id"].iloc[0]
        diff = items.loc[items["item_id"] == item_id, "difficulty"].iloc[0]
        print(f"  Step {i}: item_id={item_id}, skill_id={skill_id}, difficulty={diff:.2f}, predicted P(correct)={score:.3f}")
    print("-" * 50)
    print(f"Path length: {len(path_item_ids)} items.")

    # 4. Optional: show student's prior history
    hist = get_student_history(interactions, demo_user, max_len=5)
    print(f"\nDemo student prior history (last 5): {len(hist)} interactions.")
    if len(hist) > 0:
        print(hist[["item_id", "skill_id", "correct"]].to_string(index=False))

    print("\nMVP run complete.")


if __name__ == "__main__":
    main()
