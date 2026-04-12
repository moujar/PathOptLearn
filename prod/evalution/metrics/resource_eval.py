"""
Resource Recommendation Quality — PathOptLearn Evaluation
==========================================================
Evaluates how relevant, diverse, and comprehensive the resources returned
by PathOptLearn's GET /recommender endpoint are for a given set of
knowledge gaps.

Metrics
-------
  1. TF-IDF relevance score   — cosine similarity between resource text and gap query
  2. Source diversity          — unique domains and resource types
  3. Gap coverage rate         — fraction of gaps with ≥ 1 relevant resource
  4. Mean reciprocal rank      — for ranked resource lists vs. ground-truth

All computations use sklearn (already in requirements.txt) and urllib
(stdlib) only — no additional NLP dependencies required.

Usage
-----
    from metrics.resource_eval import evaluate_resource_list

    gaps = ["gradient descent", "backpropagation"]
    resources_by_gap = {
        "gradient descent": [
            {"title": "Understanding Gradient Descent",
             "summary": "An explanation of gradient descent optimization",
             "url": "https://example.com/gd",
             "resource_type": "article"},
        ],
        "backpropagation": [...],
    }
    report = evaluate_resource_list(gaps, resources_by_gap)
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ── Internal helpers ──────────────────────────────────────────────────────────

def _extract_domain(url: str) -> str:
    """Extract the registered domain (last two URL parts) from a resource URL."""
    try:
        netloc = urlparse(url).netloc
        parts  = netloc.split(".")
        return ".".join(parts[-2:]) if len(parts) >= 2 else netloc
    except Exception:
        return "unknown"


def _resource_text(resource: dict) -> str:
    """Concatenate the textual fields of a resource for TF-IDF representation."""
    return " ".join(filter(None, [
        resource.get("title",   ""),
        resource.get("summary", ""),
        resource.get("channel", ""),
        resource.get("concept_name", ""),
    ]))


# ══════════════════════════════════════════════════════════════════════════════
# 1.  TF-IDF relevance
# ══════════════════════════════════════════════════════════════════════════════

def compute_tfidf_relevance(
    resources: list[dict],
    query: str,
) -> list[float]:
    """
    Compute TF-IDF cosine similarity between each resource and the query.

    Parameters
    ----------
    resources : list of resource dicts (must contain "title" or "summary")
    query     : gap concept name or topic string

    Returns
    -------
    List of similarity scores in [0, 1], aligned with the ``resources`` list.
    Returns [0.0, ...] when the corpus is empty or the query is blank.
    """
    if not resources or not query.strip():
        return [0.0] * len(resources)

    texts  = [_resource_text(r) for r in resources]
    corpus = texts + [query]

    try:
        vec   = TfidfVectorizer(stop_words="english", min_df=1)
        tfidf = vec.fit_transform(corpus)
        sims  = cosine_similarity(tfidf[:-1], tfidf[-1]).flatten()
        return [round(float(s), 4) for s in sims]
    except ValueError:
        # Raised when vocabulary is empty
        return [0.0] * len(resources)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Source diversity
# ══════════════════════════════════════════════════════════════════════════════

def compute_source_diversity(resources: list[dict]) -> dict:
    """
    Measure the diversity of a recommended resource set.

    Returns
    -------
    {
        "n_resources":         int,
        "n_unique_domains":    int,
        "n_unique_types":      int,
        "domain_distribution": dict,   # {"youtube.com": 2, "arxiv.org": 1, ...}
        "type_distribution":   dict,   # {"video": 2, "article": 1, ...}
        "diversity_score":     float,  # harmonic mean of domain & type diversity (0–1)
    }
    """
    if not resources:
        return {
            "n_resources": 0, "n_unique_domains": 0, "n_unique_types": 0,
            "domain_distribution": {}, "type_distribution": {},
            "diversity_score": 0.0,
        }

    domains: dict[str, int] = {}
    types:   dict[str, int] = {}

    for r in resources:
        d = _extract_domain(r.get("url", ""))
        t = r.get("resource_type", "unknown")
        domains[d] = domains.get(d, 0) + 1
        types[t]   = types.get(t, 0) + 1

    n           = len(resources)
    domain_div  = len(domains) / n
    type_div    = len(types)   / n
    h_mean = (
        2 * domain_div * type_div / (domain_div + type_div)
        if (domain_div + type_div) > 0 else 0.0
    )

    return {
        "n_resources":         n,
        "n_unique_domains":    len(domains),
        "n_unique_types":      len(types),
        "domain_distribution": domains,
        "type_distribution":   types,
        "diversity_score":     round(h_mean, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Gap coverage
# ══════════════════════════════════════════════════════════════════════════════

def gap_coverage_score(
    gaps: list[str],
    resources_by_gap: dict[str, list[dict]],
    relevance_threshold: float = 0.10,
) -> dict:
    """
    Fraction of knowledge gaps that have at least one relevant resource.

    A resource is considered relevant if its TF-IDF similarity to the gap
    concept is >= ``relevance_threshold``.

    Returns
    -------
    {
        "coverage_rate":  float,   # fraction of gaps with ≥ 1 relevant resource
        "mean_relevance": float,   # mean top-1 relevance score across gaps
        "gap_detail":     dict,    # per-gap breakdown
    }
    """
    if not gaps:
        return {"coverage_rate": 1.0, "mean_relevance": 0.0, "gap_detail": {}}

    covered   = 0
    top_scores: list[float] = []
    detail: dict[str, dict] = {}

    for gap in gaps:
        recs   = resources_by_gap.get(gap, [])
        scores = compute_tfidf_relevance(recs, gap) if recs else []
        top    = max(scores) if scores else 0.0
        is_covered = top >= relevance_threshold
        if is_covered:
            covered += 1
        top_scores.append(top)
        detail[gap] = {
            "n_resources":   len(recs),
            "top_relevance": round(top, 4),
            "covered":       is_covered,
        }

    return {
        "coverage_rate":  round(covered / len(gaps), 4),
        "mean_relevance": round(sum(top_scores) / len(top_scores), 4) if top_scores else 0.0,
        "gap_detail":     detail,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Mean Reciprocal Rank (for ranked evaluation)
# ══════════════════════════════════════════════════════════════════════════════

def mean_reciprocal_rank(
    resources: list[dict],
    query: str,
    ground_truth_urls: list[str],
) -> float:
    """
    Compute MRR for a ranked resource list against a ground-truth URL set.

    Returns the reciprocal rank of the first ground-truth resource found
    in the TF-IDF-ranked list. Returns 0.0 if none found.

    Parameters
    ----------
    resources           : list of resource dicts (in any order)
    query               : the gap concept used to rank them
    ground_truth_urls   : URLs of known-relevant resources
    """
    if not resources or not ground_truth_urls:
        return 0.0

    scores = compute_tfidf_relevance(resources, query)
    ranked = sorted(zip(resources, scores), key=lambda x: x[1], reverse=True)
    gt_set = set(u.lower().strip() for u in ground_truth_urls)

    for rank, (resource, _) in enumerate(ranked, start=1):
        url = resource.get("url", "").lower().strip()
        if url in gt_set:
            return round(1.0 / rank, 4)
    return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Unified evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_resource_list(
    gaps: list[str],
    resources_by_gap: dict[str, list[dict]],
    relevance_threshold: float = 0.10,
) -> dict:
    """
    Full resource-recommendation evaluation combining all sub-metrics.

    Parameters
    ----------
    gaps             : list of gap concept names
    resources_by_gap : dict mapping gap_concept → list of resource dicts

    Returns
    -------
    {
        "n_gaps":                int,
        "n_total_resources":     int,
        "mean_relevance_score":  float,
        "gap_coverage":          dict,   # from gap_coverage_score()
        "source_diversity":      dict,   # from compute_source_diversity()
    }
    """
    all_resources = [r for recs in resources_by_gap.values() for r in recs]
    diversity     = compute_source_diversity(all_resources)
    coverage      = gap_coverage_score(gaps, resources_by_gap, relevance_threshold)

    # Aggregate per-gap relevance
    all_scores: list[float] = []
    for gap, recs in resources_by_gap.items():
        all_scores.extend(compute_tfidf_relevance(recs, gap))

    mean_relevance = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0

    return {
        "n_gaps":               len(gaps),
        "n_total_resources":    len(all_resources),
        "mean_relevance_score": mean_relevance,
        "gap_coverage":         coverage,
        "source_diversity":     diversity,
    }
