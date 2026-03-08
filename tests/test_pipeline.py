"""
Integration tests for the AdaptLearn AI pipeline.
Mocks OpenAI, Tavily, and YouTube APIs.

Run: pytest tests/test_pipeline.py -v
"""
import json
import sys
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def mock_openai_response():
    """Mock OpenAI chat completion response."""
    mock = MagicMock()
    mock.choices[0].message.content = json.dumps({
        "topic": "Machine Learning",
        "sub_topics": ["Supervised Learning", "Neural Networks", "Optimization", "Evaluation"],
        "difficulty_hint": "beginner",
        "kg_query_terms": ["machine learning", "neural networks", "gradient descent", "classification"],
    })
    return mock


@pytest.fixture
def mock_quiz_response():
    mock = MagicMock()
    mock.choices[0].message.content = json.dumps({
        "questions": [
            {
                "question": "What is supervised learning?",
                "options": ["Learning with labels", "Learning without labels", "Reinforcement", "Clustering"],
                "correct_index": 0,
                "concept": "Supervised Learning",
                "difficulty_hint": -1.5,
            }
        ]
    })
    return mock


@pytest.fixture
def sample_knowledge_vector():
    return {
        "student_id": "test_student",
        "theta": 0.5,
        "concept_mastery": {
            "Supervised Learning": 0.7,
            "Neural Networks": 0.4,
            "Optimization": 0.3,
        },
        "confidence_interval": [0.1, 0.9],
        "assessed_at": "2026-03-08T10:00:00+00:00",
        "last_studied_at": {},
    }


@pytest.fixture
def sample_path():
    return {
        "path_id": "test_path_001",
        "student_id": "test_student",
        "goal_id": "test_goal_001",
        "algorithm": "DRL-PPO",
        "steps": [
            {
                "step": 1,
                "concept": "Optimization",
                "resource_type": "article",
                "url": "https://example.com/optimization",
                "duration_min": 15,
                "predicted_mastery_delta": 0.15,
                "node_id": "node_001",
            },
            {
                "step": 2,
                "concept": "Neural Networks",
                "resource_type": "video",
                "url": "https://youtube.com/watch?v=example",
                "duration_min": 20,
                "predicted_mastery_delta": 0.12,
                "node_id": "node_002",
            },
        ],
        "energy_score": 45.0,
        "predicted_completion_sessions": 3,
        "created_at": "2026-03-08T10:00:00+00:00",
    }


# ── Schema Tests ──────────────────────────────────────────────────────────

class TestSchemas:
    def test_knowledge_vector_schema(self, sample_knowledge_vector):
        from backend.schemas import KnowledgeVector
        kv = KnowledgeVector(**sample_knowledge_vector)
        assert kv.student_id == "test_student"
        assert kv.theta == 0.5
        assert "Supervised Learning" in kv.concept_mastery
        assert len(kv.confidence_interval) == 2

    def test_learning_path_schema(self, sample_path):
        from backend.schemas import LearningPath
        lp = LearningPath(**sample_path)
        assert lp.path_id == "test_path_001"
        assert lp.algorithm == "DRL-PPO"
        assert len(lp.steps) == 2
        assert lp.steps[0].step == 1

    def test_quiz_item_schema(self):
        from backend.schemas import QuizItem
        item = QuizItem(
            item_id="q001",
            question="What is ML?",
            options=["A", "B", "C", "D"],
            correct_index=0,
            difficulty_b=-1.0,
            discrimination_a=1.5,
            bloom_level="remember",
            concept="Machine Learning",
        )
        assert item.guessing_c == 0.25
        assert item.bloom_level == "remember"


# ── CAT Engine Tests ──────────────────────────────────────────────────────

class TestCATEngine:
    def test_irt_3pl(self):
        from backend.services.cat_engine import irt_3pl
        # P(correct) should be between c and 1
        p = irt_3pl(theta=0.0, a=1.0, b=0.0, c=0.25)
        assert 0.25 <= p <= 1.0
        # High ability -> high probability
        p_high = irt_3pl(theta=3.0, a=1.0, b=0.0, c=0.25)
        p_low = irt_3pl(theta=-3.0, a=1.0, b=0.0, c=0.25)
        assert p_high > p_low

    def test_fisher_info(self):
        from backend.services.cat_engine import fisher_info
        fi = fisher_info(theta=0.0, a=1.0, b=0.0, c=0.25)
        assert fi >= 0.0

    def test_estimate_theta(self):
        from backend.services.cat_engine import estimate_theta
        items_map = {
            "q1": {"item_id": "q1", "discrimination_a": 1.0, "difficulty_b": 0.0, "guessing_c": 0.25},
            "q2": {"item_id": "q2", "discrimination_a": 1.5, "difficulty_b": 0.5, "guessing_c": 0.25},
        }
        responses = [
            {"item_id": "q1", "correct": True},
            {"item_id": "q2", "correct": True},
        ]
        theta, se = estimate_theta(responses, items_map)
        assert -4.0 <= theta <= 4.0
        assert se > 0.0
        # Correct answers → theta should be positive
        assert theta > 0.0

    def test_select_next_item(self):
        from backend.services.cat_engine import select_next_item
        items = [
            {"item_id": "q1", "discrimination_a": 1.0, "difficulty_b": 0.0, "guessing_c": 0.25},
            {"item_id": "q2", "discrimination_a": 2.0, "difficulty_b": 0.1, "guessing_c": 0.25},
        ]
        next_item = select_next_item(theta=0.0, quiz_items=items, administered_ids=set())
        assert next_item is not None
        assert next_item["item_id"] in {"q1", "q2"}

    def test_run_cat_batch(self):
        from backend.services.cat_engine import run_cat_batch
        items = [
            {
                "item_id": f"q{i}",
                "discrimination_a": 1.0,
                "difficulty_b": float(i - 5) / 3,
                "guessing_c": 0.25,
                "concept": "Neural Networks" if i % 2 == 0 else "Optimization",
            }
            for i in range(10)
        ]
        responses = [
            {"item_id": f"q{i}", "correct": i % 3 != 0}
            for i in range(10)
        ]
        result = run_cat_batch(items, responses)
        assert "theta" in result
        assert "concept_mastery" in result
        assert "confidence_interval" in result
        assert len(result["confidence_interval"]) == 2
        for concept, mastery in result["concept_mastery"].items():
            assert 0.0 <= mastery <= 1.0


# ── Forgetting Module Tests ───────────────────────────────────────────────

class TestForgetting:
    def test_apply_forgetting(self):
        from backend.models.forgetting import EbbinghausForgetting
        f = EbbinghausForgetting()
        new_m, delta = f.apply_forgetting("concept_A", mastery=0.8, t_hours=24.0)
        assert 0.0 <= new_m <= 0.8
        assert delta <= 0.0  # forgetting always decreases mastery

    def test_update_stability(self):
        from backend.models.forgetting import EbbinghausForgetting
        f = EbbinghausForgetting()
        s1 = f.update_stability("concept_A", mastery_before=0.5)
        assert s1 > f.DEFAULT_STABILITY_HOURS

    def test_zero_time_no_forgetting(self):
        from backend.models.forgetting import EbbinghausForgetting
        f = EbbinghausForgetting()
        new_m, delta = f.apply_forgetting("X", 0.7, t_hours=0.0)
        # exp(-0/S) = 1 → no change
        assert abs(new_m - 0.7) < 1e-6

    def test_full_vector_forgetting(self):
        from backend.models.forgetting import EbbinghausForgetting
        f = EbbinghausForgetting()
        mastery = {"A": 0.8, "B": 0.6, "C": 0.4}
        updated, deltas = f.apply_forgetting_to_vector(
            mastery, {}, current_time_iso="", global_t_hours=48.0
        )
        for c, m in updated.items():
            assert m <= mastery[c]


# ── DRL Agent Tests ───────────────────────────────────────────────────────

class TestDRLAgent:
    def test_learning_env_reset(self):
        from backend.models.drl_agent import LearningEnv
        env = LearningEnv(
            concept_list=["A", "B", "C"],
            initial_mastery={"A": 0.5, "B": 0.3, "C": 0.7},
            prerequisite_graph={"B": ["A"]},
        )
        obs = env.reset()
        assert len(obs) == 3
        assert all(0.0 <= v <= 1.0 for v in obs)

    def test_learning_env_step(self):
        from backend.models.drl_agent import LearningEnv
        env = LearningEnv(
            concept_list=["A", "B", "C"],
            initial_mastery={"A": 0.5, "B": 0.3, "C": 0.7},
            prerequisite_graph={},
        )
        env.reset()
        obs, reward, done, truncated, info = env.step(0)
        assert len(obs) == 3
        assert isinstance(reward, float)
        assert "concept" in info

    def test_drl_agent_greedy_fallback(self):
        from backend.models.drl_agent import DRLAgent, LearningEnv
        env = LearningEnv(
            concept_list=["A", "B", "C"],
            initial_mastery={"A": 0.5, "B": 0.3, "C": 0.7},
            prerequisite_graph={},
            max_steps=5,
        )
        agent = DRLAgent(model_path=None)  # No pretrained model
        path = agent.generate_path(env)
        assert len(path) > 0
        assert all("concept" in step for step in path)


# ── Prerequisites Order Tests ─────────────────────────────────────────────

class TestPathOrder:
    def test_path_respects_prerequisites(self):
        """DRL path should generally satisfy prerequisite ordering."""
        from backend.models.drl_agent import DRLAgent, LearningEnv

        concepts = ["Basics", "Intermediate", "Advanced"]
        prereqs = {"Intermediate": ["Basics"], "Advanced": ["Intermediate"]}
        mastery = {"Basics": 0.2, "Intermediate": 0.0, "Advanced": 0.0}

        env = LearningEnv(concepts, mastery, prereqs, max_steps=10)
        agent = DRLAgent()
        path = agent.generate_path(env)

        path_concepts = [step["concept"] for step in path]
        # Basics should appear before Advanced in the path
        if "Basics" in path_concepts and "Advanced" in path_concepts:
            assert path_concepts.index("Basics") < path_concepts.index("Advanced")


# ── Benchmark Metrics Tests ───────────────────────────────────────────────

class TestBenchmarkMetrics:
    def test_compute_auc(self):
        from backend.benchmarks.metrics import compute_auc
        y_true = [1, 0, 1, 0, 1]
        y_prob = [0.9, 0.2, 0.8, 0.3, 0.7]
        auc = compute_auc(y_true, y_prob)
        assert 0.5 <= auc <= 1.0

    def test_compute_recall_at_k(self):
        from backend.benchmarks.metrics import compute_recall_at_k
        recommended = ["A", "B", "C", "D", "E"]
        relevant = ["A", "C", "E", "F"]
        recall = compute_recall_at_k(recommended, relevant, k=5)
        assert 0.0 <= recall <= 1.0

    def test_compute_les(self):
        from backend.benchmarks.metrics import compute_les
        les = compute_les(knowledge_gain=0.5, time_invested_minutes=30.0)
        assert les > 0.0
        # Higher gain → higher LES
        les2 = compute_les(0.8, 30.0)
        assert les2 > les


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
