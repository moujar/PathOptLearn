"""Computerized Adaptive Testing (CAT) engine — 3PL IRT model."""
import math
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


# ── IRT helpers ────────────────────────────────────────────────────────────

def irt_3pl(theta: float, a: float, b: float, c: float) -> float:
    """3-Parameter Logistic IRT: P(correct | θ, a, b, c)."""
    return c + (1.0 - c) / (1.0 + math.exp(-a * (theta - b)))


def fisher_info(theta: float, a: float, b: float, c: float) -> float:
    """Fisher information of an item at ability theta."""
    p = irt_3pl(theta, a, b, c)
    q = 1.0 - p
    if p <= c or q <= 0:
        return 0.0
    return (a ** 2 * (p - c) ** 2 * q) / ((1.0 - c) ** 2 * p)


# ── Theta estimation (Newton-Raphson MLE) ──────────────────────────────────

def estimate_theta(
    responses: list[dict],
    items_map: dict[str, dict],
    init_theta: float = 0.0,
) -> tuple[float, float]:
    """
    Maximum-Likelihood Estimation of theta via Newton-Raphson.
    Returns (theta, standard_error).
    """
    theta = init_theta
    for _ in range(60):
        L1 = 0.0  # first derivative of log-likelihood
        L2 = 0.0  # second derivative
        for r in responses:
            item = items_map.get(r.get("item_id", ""))
            if item is None:
                continue
            a = item.get("discrimination_a", 1.0)
            b = item.get("difficulty_b", 0.0)
            c = item.get("guessing_c", 0.25)
            p = irt_3pl(theta, a, b, c)
            q = 1.0 - p
            u = int(r.get("correct", False))
            if p <= c or q <= 0:
                continue
            p_star = p - c
            denom = p * (1.0 - c)
            L1 += a * (p_star / denom) * (u - p)
            L2 -= (a ** 2 * p_star ** 2 * q) / (denom ** 2)

        if abs(L2) < 1e-8:
            break
        delta = L1 / L2
        theta -= delta
        if abs(delta) < 1e-5:
            break

    theta = max(-4.0, min(4.0, theta))
    se = 1.0 / math.sqrt(max(1e-6, abs(L2) if L2 != 0 else 1e-6))
    return round(theta, 4), round(se, 4)


# ── Item selection ─────────────────────────────────────────────────────────

def select_next_item(
    theta: float,
    quiz_items: list[dict],
    administered_ids: set[str],
) -> Optional[dict]:
    """Pick the unadministered item that maximises Fisher information at theta."""
    best_item: Optional[dict] = None
    best_fi = -1.0
    for item in quiz_items:
        if item["item_id"] in administered_ids:
            continue
        fi = fisher_info(
            theta,
            item.get("discrimination_a", 1.0),
            item.get("difficulty_b", 0.0),
            item.get("guessing_c", 0.25),
        )
        if fi > best_fi:
            best_fi = fi
            best_item = item
    return best_item


# ── Batch CAT scoring ──────────────────────────────────────────────────────

def run_cat_batch(
    quiz_items: list[dict],
    responses: list[dict],
    max_items: int = 15,
    se_threshold: float = 0.30,
) -> dict:
    """
    Score a completed (or partial) CAT session.
    Returns dict with theta, concept_mastery, CI, assessed_at.
    """
    items_map = {item["item_id"]: item for item in quiz_items}
    theta, se = estimate_theta(responses, items_map)

    # Per-concept theta sub-estimates
    concept_responses: dict[str, list[dict]] = {}
    for r in responses:
        item = items_map.get(r.get("item_id", ""))
        if item:
            concept_responses.setdefault(item.get("concept", "unknown"), []).append(r)

    concept_mastery: dict[str, float] = {}
    for concept, c_resps in concept_responses.items():
        c_theta, _ = estimate_theta(c_resps, items_map, init_theta=theta)
        mastery = 1.0 / (1.0 + math.exp(-c_theta))  # sigmoid
        concept_mastery[concept] = round(mastery, 4)

    ci_half = 1.96 * se
    return {
        "theta": theta,
        "concept_mastery": concept_mastery,
        "confidence_interval": [round(theta - ci_half, 4), round(theta + ci_half, 4)],
        "assessed_at": datetime.now(timezone.utc).isoformat(),
        "se": se,
    }


# ── Real-time item-by-item mode ────────────────────────────────────────────

def get_next_item_response(
    quiz_items: list[dict],
    current_responses: list[dict],
    max_items: int = 15,
    se_threshold: float = 0.30,
) -> dict:
    """
    Determine next adaptive item for real-time CAT.
    Returns {next_item, current_theta, items_remaining_estimate, should_stop}.
    """
    items_map = {item["item_id"]: item for item in quiz_items}
    administered = {r["item_id"] for r in current_responses}

    theta, se = 0.0, 999.0
    if current_responses:
        theta, se = estimate_theta(current_responses, items_map)

    enough_items = len(administered) >= max_items
    low_se = se < se_threshold and len(administered) >= 5
    should_stop = enough_items or low_se

    next_item = None
    if not should_stop:
        next_item = select_next_item(theta, quiz_items, administered)
        if next_item is None:
            should_stop = True

    remaining = max(0, max_items - len(administered))
    return {
        "next_item": next_item,
        "current_theta": round(theta, 4),
        "items_remaining_estimate": remaining,
        "should_stop": should_stop,
    }
