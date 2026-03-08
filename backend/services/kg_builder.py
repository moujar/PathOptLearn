"""Knowledge Graph builder — GraphRAG-style extraction + GAT embeddings + Neo4j."""
import asyncio
import hashlib
import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── GPT-4 entity / relation extraction ────────────────────────────────────

async def extract_concepts(
    text_chunks: list[str], openai_client, model: str = "gpt-4"
) -> list[dict]:
    """Extract educational concepts from text via GPT-4."""
    combined = "\n\n".join(text_chunks[:5])[:6000]
    system_prompt = (
        "Extract all educational concepts, skills, and topics from this text. "
        "For each concept output: name (short, unique), definition (1 sentence), "
        "difficulty (0.0-1.0). "
        'Return JSON: {"concepts": [{"name": str, "definition": str, "difficulty": float}]}'
    )
    try:
        resp = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": combined},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        data = json.loads(resp.choices[0].message.content)
        return data.get("concepts", [])
    except Exception as exc:
        logger.error(f"Concept extraction failed: {exc}")
        return []


async def identify_prerequisites(
    concepts: list[str], openai_client, model: str = "gpt-4"
) -> list[dict]:
    """Identify prerequisite pairs using GPT-4."""
    concept_str = ", ".join(concepts[:30])
    system_prompt = (
        "Given these educational concepts, identify prerequisite pairs. "
        'Format: [{"from": "A", "to": "B", "confidence": 0.9}]. '
        "A->B means: must understand A before learning B. "
        'Return JSON: {"prerequisites": [...]}'
    )
    try:
        resp = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Concepts: {concept_str}"},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        data = json.loads(resp.choices[0].message.content)
        return data.get("prerequisites", [])
    except Exception as exc:
        logger.error(f"Prerequisite identification failed: {exc}")
        return []


# ── Semantic edges ─────────────────────────────────────────────────────────

def compute_semantic_edges(
    concepts: list[dict], embedder, threshold: float = 0.6
) -> list[dict]:
    """Compute cosine-similarity edges between all concept pairs above threshold."""
    import numpy as np

    if not concepts:
        return []

    names = [c["name"] for c in concepts]
    texts = [f"{c['name']}: {c.get('definition', '')}" for c in concepts]
    embs = embedder.encode(texts, show_progress_bar=False)

    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs_norm = embs / (norms + 1e-8)
    sim = embs_norm @ embs_norm.T

    edges: list[dict] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            s = float(sim[i, j])
            if s > threshold:
                edges.append(
                    {"from": names[i], "to": names[j], "relation": "semantic", "weight": round(s, 4)}
                )
    return edges


# ── GAT embeddings ─────────────────────────────────────────────────────────

def run_gat_embeddings(
    concepts: list[dict], edges: list[dict]
) -> dict[str, list[float]]:
    """
    Produce 128-d GAT embeddings for each concept node.
    Falls back to a linear projection if torch_geometric is unavailable.
    """
    import torch
    import torch.nn.functional as F
    from sentence_transformers import SentenceTransformer

    if not concepts:
        return {}

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    names = [c["name"] for c in concepts]
    texts = [f"{c['name']}: {c.get('definition', '')}" for c in concepts]
    x = torch.tensor(embedder.encode(texts, show_progress_bar=False), dtype=torch.float32)
    input_dim = x.shape[1]  # 384 for all-MiniLM

    name_to_idx = {n: i for i, n in enumerate(names)}
    src, dst = [], []
    for e in edges:
        fn, tn = e.get("from", ""), e.get("to", "")
        if fn in name_to_idx and tn in name_to_idx:
            src += [name_to_idx[fn], name_to_idx[tn]]
            dst += [name_to_idx[tn], name_to_idx[fn]]

    try:
        from torch_geometric.nn import GATConv

        edge_index = (
            torch.tensor([src, dst], dtype=torch.long)
            if src
            else torch.zeros((2, 0), dtype=torch.long)
        )
        gat1 = GATConv(input_dim, 64, heads=4, concat=True)
        gat2 = GATConv(256, 128, heads=1, concat=False)
        with torch.no_grad():
            h = F.elu(gat1(x, edge_index))
            h = gat2(h, edge_index)
        return {names[i]: h[i].tolist() for i in range(len(names))}

    except ImportError:
        # Fallback: simple linear projection to 128d
        proj = torch.nn.Linear(input_dim, 128, bias=False)
        torch.nn.init.orthogonal_(proj.weight)
        with torch.no_grad():
            h = proj(x)
        return {names[i]: h[i].tolist() for i in range(len(names))}

    except Exception as exc:
        logger.warning(f"GAT embedding failed: {exc}; using random.")
        import random
        return {c["name"]: [random.gauss(0, 0.1) for _ in range(128)] for c in concepts}


# ── Neo4j storage ──────────────────────────────────────────────────────────

async def store_in_neo4j(
    nodes: list[dict], edges: list[dict], neo4j_driver: Any
) -> None:
    """Persist KG nodes/edges in Neo4j (skips gracefully if driver is None)."""
    if neo4j_driver is None:
        return

    try:
        async with neo4j_driver.session() as session:
            for n in nodes:
                await session.run(
                    "MERGE (c:Concept {id: $id}) "
                    "SET c.concept = $concept, c.difficulty = $difficulty",
                    id=n["node_id"], concept=n["concept"], difficulty=n["difficulty"],
                )
            for e in edges:
                rel = e["relation"].upper()
                await session.run(
                    f"MATCH (a:Concept {{id: $fid}}), (b:Concept {{id: $tid}}) "
                    f"MERGE (a)-[r:{rel}]->(b) SET r.weight = $w",
                    fid=e["from_node"], tid=e["to_node"], w=e["weight"],
                )
        logger.info(f"Neo4j: stored {len(nodes)} nodes, {len(edges)} edges")
    except Exception as exc:
        logger.error(f"Neo4j storage failed: {exc}")


# ── Main pipeline ──────────────────────────────────────────────────────────

async def build_knowledge_graph(
    goal_id: str,
    docs: list[dict],
    openai_client,
    embedder,
    neo4j_driver: Optional[Any] = None,
) -> dict:
    """
    Full KG build pipeline:
    1. Extract concepts via GPT-4
    2. Identify prerequisites + semantic edges
    3. Run GAT embeddings
    4. Store in Neo4j
    Returns {'kg_id', 'nodes', 'edges'}.
    """
    import uuid

    # Step 1 — Extract concepts
    texts = [d.get("content_text", "") for d in docs]
    raw_concepts = await extract_concepts(texts, openai_client)

    # Deduplicate by name
    seen: set[str] = set()
    unique: list[dict] = []
    for c in raw_concepts:
        name = c.get("name", "").strip()
        if name and name.lower() not in seen:
            seen.add(name.lower())
            unique.append(c)

    concept_names = [c["name"] for c in unique]

    # Step 2 — Edges
    prereq_pairs = await identify_prerequisites(concept_names, openai_client)
    semantic_raw = compute_semantic_edges(unique, embedder)
    all_edges_for_gat = prereq_pairs + semantic_raw

    # Step 3 — GAT embeddings
    gat_embs = run_gat_embeddings(unique, all_edges_for_gat)

    # Build node/edge objects
    kg_id = str(uuid.uuid4())[:8]
    name_to_id: dict[str, str] = {}
    nodes: list[dict] = []
    for c in unique:
        nid = hashlib.md5(c["name"].encode()).hexdigest()[:12]
        name_to_id[c["name"]] = nid
        nodes.append(
            {
                "node_id": nid,
                "concept": c["name"],
                "difficulty": float(c.get("difficulty", 0.5)),
                "embedding": gat_embs.get(c["name"], []),
                "definition": c.get("definition", ""),
            }
        )

    edges: list[dict] = []
    for p in prereq_pairs:
        fn, tn = p.get("from", ""), p.get("to", "")
        if fn in name_to_id and tn in name_to_id:
            edges.append(
                {
                    "from_node": name_to_id[fn],
                    "to_node": name_to_id[tn],
                    "relation": "prerequisite",
                    "weight": float(p.get("confidence", 0.8)),
                }
            )
    for e in semantic_raw:
        fn, tn = e.get("from", ""), e.get("to", "")
        if fn in name_to_id and tn in name_to_id:
            edges.append(
                {
                    "from_node": name_to_id[fn],
                    "to_node": name_to_id[tn],
                    "relation": "semantic",
                    "weight": float(e.get("weight", 0.7)),
                }
            )

    # Step 4 — Neo4j
    await store_in_neo4j(nodes, edges, neo4j_driver)

    return {"kg_id": kg_id, "nodes": nodes, "edges": edges}
