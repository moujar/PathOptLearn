"""LLM explanation service — RAG-augmented GPT-4 path explanations."""
import asyncio
import logging

logger = logging.getLogger(__name__)

_STYLE_MAP = {
    "full": (
        "Explain in 4-5 sentences why this learning path was recommended. "
        "Reference the student's specific knowledge gaps and how each step addresses them. "
        "Be clear, educational, and avoid jargon."
    ),
    "brief": (
        "Explain in 1-2 sentences why this path was recommended. Be concise."
    ),
    "motivational": (
        "Explain in 3-4 sentences why this path was recommended. "
        "Be encouraging and motivating. Highlight the student's potential and how each step builds on the last."
    ),
}


async def explain_path(
    knowledge_vector: dict,
    learning_path: dict,
    rag_chunks: list[dict],
    openai_client,
    explanation_type: str = "full",
    model: str = "gpt-4",
) -> dict:
    """
    Generate a RAG-augmented GPT-4 explanation for a recommended learning path.

    Args:
        knowledge_vector: Student KnowledgeVector dict
        learning_path: LearningPath dict
        rag_chunks: Retrieved context chunks
        openai_client: OpenAI client
        explanation_type: 'full' | 'brief' | 'motivational'
        model: GPT model name

    Returns:
        {explanation, key_reasons, next_action}
    """
    # Summarise student profile
    mastery = knowledge_vector.get("concept_mastery", {})
    weak_concepts = sorted(mastery.items(), key=lambda x: x[1])[:3]
    weak_str = ", ".join(f"{c} ({v:.0%})" for c, v in weak_concepts) or "unknown gaps"
    theta = knowledge_vector.get("theta", 0.0)

    # Summarise path
    steps = learning_path.get("steps", [])
    path_str = "\n".join(
        f"Step {s.get('step', i+1)}: {s.get('concept','?')} "
        f"({s.get('resource_type','resource')}, {s.get('duration_min', 10)} min)"
        for i, s in enumerate(steps[:8])
    )
    algorithm = learning_path.get("algorithm", "DRL-PPO")

    # RAG context
    context = "\n\n".join(
        f"[{c.get('title', 'Source')}] {c.get('text', '')[:500]}"
        for c in rag_chunks[:3]
    )

    style = _STYLE_MAP.get(explanation_type, _STYLE_MAP["full"])
    system_prompt = (
        f"You are an encouraging AI tutor. {style} "
        "Do not use jargon. Reference the student's specific gaps and path steps by name."
    )
    user_prompt = (
        f"Student's weakest concepts: {weak_str}\n"
        f"Overall ability (theta): {theta:.2f}\n\n"
        f"Recommended path (algorithm: {algorithm}):\n{path_str}\n\n"
        f"Educational context:\n{context[:2000]}"
    )

    try:
        resp = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=600,
        )
        explanation = resp.choices[0].message.content.strip()
    except Exception as exc:
        logger.error(f"GPT-4 explanation failed: {exc}")
        explanation = (
            f"This personalised learning path targets your weakest areas ({weak_str}) "
            f"and uses the {algorithm} algorithm to optimise the order of topics for your "
            f"current ability level (θ={theta:.2f})."
        )

    key_reasons = [
        f"Identified knowledge gaps: {weak_str}",
        f"Path algorithm: {algorithm}",
        f"Estimated completion: {learning_path.get('predicted_completion_sessions', '?')} sessions",
    ]
    if steps:
        key_reasons.append(f"First priority: '{steps[0].get('concept', 'first topic')}'")

    next_action = (
        f"Start with '{steps[0].get('concept', 'the first topic')}' — click the resource link."
        if steps
        else "Begin your learning journey!"
    )

    return {"explanation": explanation, "key_reasons": key_reasons, "next_action": next_action}
