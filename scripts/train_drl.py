"""
Train the DRL-PPO agent on ASSISTments 2009 student trajectories.

Usage:
    python scripts/train_drl.py --timesteps 200000 --output models/checkpoints/drl_ppo
"""
import argparse
import csv
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_env_from_assistments(csv_path: str):
    """Build a LearningEnv from ASSISTments data."""
    from backend.models.drl_agent import LearningEnv

    concepts: set[str] = set()
    mastery: dict[str, list[float]] = {}

    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                skill = row.get("skill_name", row.get("skill", "unknown"))
                correct = float(row.get("correct", 0))
                concepts.add(skill)
                mastery.setdefault(skill, []).append(correct)
    except FileNotFoundError:
        logger.warning(f"File not found: {csv_path}. Using synthetic concepts.")
        concepts = {f"Concept_{i}" for i in range(15)}
        mastery = {c: [0.5] for c in concepts}

    concept_list = sorted(concepts)
    initial_mastery = {
        c: round(sum(vals) / len(vals), 3) for c, vals in mastery.items() if vals
    }

    # Basic prerequisite graph (linear order as default)
    prereq_map: dict[str, list[str]] = {}
    for i, c in enumerate(concept_list[1:], 1):
        prereq_map[c] = [concept_list[i - 1]]

    return LearningEnv(
        concept_list=concept_list,
        initial_mastery=initial_mastery,
        prerequisite_graph=prereq_map,
    )


def main():
    parser = argparse.ArgumentParser(description="Train AdaptLearn DRL-PPO Agent")
    parser.add_argument("--data", default="data/benchmarks/assistments_2009.csv")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--output", default="models/checkpoints/drl_ppo")
    args = parser.parse_args()

    logger.info("Building training environment from ASSISTments data…")
    env = build_env_from_assistments(args.data)
    logger.info(f"Environment: {env.n_concepts} concepts")

    logger.info(f"Training PPO for {args.timesteps:,} timesteps…")
    from backend.models.drl_agent import DRLAgent
    agent = DRLAgent()
    agent.train(env, total_timesteps=args.timesteps, save_path=args.output)

    logger.info(f"Training complete. Model saved to {args.output}")

    # Quick eval
    logger.info("Running quick evaluation (10 episodes)…")
    total_reward = 0.0
    for episode in range(10):
        path_actions = agent.generate_path(env)
        ep_reward = sum(a.get("reward", 0) for a in path_actions)
        total_reward += ep_reward
        logger.info(f"  Episode {episode+1}: steps={len(path_actions)}, reward={ep_reward:.3f}")

    logger.info(f"Avg reward over 10 episodes: {total_reward/10:.3f}")


if __name__ == "__main__":
    main()
