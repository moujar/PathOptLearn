"""DRL-PPO agent and Gym environment for learning path recommendation."""
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class LearningEnv:
    """
    MDP Environment for adaptive learning path recommendation.

    State  : concept mastery vector  (n_concepts,)  float32
    Action : select next concept to study  Discrete(n_concepts)
    Reward : mastery_gain + difficulty_match + prereq_satisfaction - redundancy
    """

    def __init__(
        self,
        concept_list: List[str],
        initial_mastery: Dict[str, float],
        prerequisite_graph: Dict[str, List[str]],
        concept_difficulty: Optional[Dict[str, float]] = None,
        target_mastery: float = 0.85,
        max_steps: int = 20,
        lambdas: Tuple[float, float, float, float] = (0.5, 0.2, 0.2, 0.1),
    ):
        self.concept_list = concept_list
        self.n_concepts = len(concept_list)
        self.initial_mastery = initial_mastery
        self.prerequisite_graph = prerequisite_graph
        self.concept_difficulty = concept_difficulty or {}
        self.target_mastery = target_mastery
        self.max_steps = max_steps
        self.l1, self.l2, self.l3, self.l4 = lambdas

        try:
            import gymnasium as gym
            self.observation_space = gym.spaces.Box(
                low=0.0, high=1.0, shape=(self.n_concepts,), dtype=np.float32
            )
            self.action_space = gym.spaces.Discrete(self.n_concepts)
        except ImportError:
            self.observation_space = None
            self.action_space = None

        self.mastery: np.ndarray = np.zeros(self.n_concepts, dtype=np.float32)
        self.theta: float = 0.0
        self.step_count: int = 0
        self.visited: set = set()

    def reset(self) -> np.ndarray:
        """Reset environment to initial student state."""
        self.mastery = np.array(
            [self.initial_mastery.get(c, 0.0) for c in self.concept_list], dtype=np.float32
        )
        self.theta = float(np.mean(self.mastery))
        self.step_count = 0
        self.visited = set()
        return self.mastery.copy()

    def _prerequisite_score(self, action: int) -> float:
        concept = self.concept_list[action]
        prereqs = self.prerequisite_graph.get(concept, [])
        if not prereqs:
            return 1.0
        name_to_idx = {c: i for i, c in enumerate(self.concept_list)}
        satisfied = sum(
            1
            for p in prereqs
            if p in name_to_idx and float(self.mastery[name_to_idx[p]]) >= 0.6
        )
        return satisfied / len(prereqs)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step: study concept at index `action`."""
        concept = self.concept_list[action]
        difficulty = float(self.concept_difficulty.get(concept, 0.5))

        mastery_before = float(self.mastery[action])
        difficulty_match = 1.0 - abs(difficulty - self.theta)
        mastery_gain = 0.15 * difficulty_match * (1.0 - mastery_before)
        self.mastery[action] = min(1.0, self.mastery[action] + mastery_gain)

        prereq_score = self._prerequisite_score(action)
        redundancy = 1.0 if action in self.visited else 0.0

        reward = (
            self.l1 * mastery_gain
            + self.l2 * difficulty_match
            + self.l3 * prereq_score
            - self.l4 * redundancy
        )

        self.visited.add(action)
        self.theta = float(np.mean(self.mastery))
        self.step_count += 1

        done = bool(np.all(self.mastery >= self.target_mastery))
        truncated = self.step_count >= self.max_steps

        info = {
            "concept": concept,
            "mastery_gain": mastery_gain,
            "prereq_score": prereq_score,
            "difficulty_match": difficulty_match,
        }
        return self.mastery.copy(), reward, done, truncated, info


class DRLAgent:
    """PPO-based agent for learning path recommendation using Stable-Baselines3."""

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path
        self._try_load()

    def _try_load(self) -> None:
        """Attempt to load pre-trained PPO model."""
        if not self.model_path or not os.path.exists(self.model_path):
            logger.info("No pretrained DRL model found; will use greedy fallback.")
            return
        try:
            from stable_baselines3 import PPO
            self.model = PPO.load(self.model_path)
            logger.info(f"Loaded DRL-PPO from {self.model_path}")
        except ImportError:
            logger.warning("stable_baselines3 not installed.")
        except Exception as exc:
            logger.warning(f"DRL model load failed: {exc}")

    def train(
        self,
        env: LearningEnv,
        total_timesteps: int = 100_000,
        save_path: Optional[str] = None,
    ) -> None:
        """Train PPO on the learning environment with 4 parallel envs."""
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        import gymnasium as gym

        class GymWrapper(gym.Env):
            def __init__(self, inner: LearningEnv):
                super().__init__()
                self._inner = inner
                self.observation_space = inner.observation_space
                self.action_space = inner.action_space

            def reset(self, seed=None, options=None):
                return self._inner.reset(), {}

            def step(self, action):
                return self._inner.step(action)

        vec_env = DummyVecEnv([lambda: GymWrapper(env)] * 4)
        self.model = PPO(
            "MlpPolicy", vec_env,
            n_steps=2048, batch_size=64, n_epochs=10,
            learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
            verbose=1,
        )
        self.model.learn(total_timesteps=total_timesteps)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.model.save(save_path)
            logger.info(f"DRL model saved to {save_path}")

    def generate_path(self, env: LearningEnv) -> List[Dict[str, Any]]:
        """
        Roll out the PPO policy (or greedy fallback) to generate a learning path.
        Returns list of step dicts: {action, concept, mastery_gain, reward}.
        """
        obs = env.reset()
        path_actions: List[Dict[str, Any]] = []

        for _ in range(env.max_steps):
            if self.model is not None:
                action, _ = self.model.predict(obs, deterministic=True)
                action = int(action)
            else:
                # Greedy fallback: lowest mastery first
                action = int(np.argmin(obs))

            obs, reward, done, truncated, info = env.step(action)
            path_actions.append(
                {
                    "action": action,
                    "concept": info.get("concept", ""),
                    "mastery_gain": info.get("mastery_gain", 0.0),
                    "reward": float(reward),
                    "prereq_score": info.get("prereq_score", 1.0),
                }
            )

            if done or truncated:
                break

        return path_actions
