"""RAG engine — sentence-transformers + FAISS for retrieval-augmented generation."""
import asyncio
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    Lightweight RAG engine:
    1. Chunks harvested docs into 512-token windows (64-token overlap)
    2. Embeds chunks with sentence-transformers
    3. Stores in FAISS IndexFlatIP for cosine similarity search
    """

    def __init__(self, embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embed_model_name = embed_model_name
        self._index = None
        self._chunks: list[dict] = []
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embed_model_name)
        return self._embedder

    def build_index(self, docs: list[dict]) -> None:
        """
        Build FAISS index from a list of doc dicts with 'content_text', 'url', 'title'.
        Uses 512-word chunks with 64-word overlap.
        """
        import faiss

        self._chunks = []
        texts: list[str] = []

        for doc in docs:
            content = doc.get("content_text", "")
            url = doc.get("url", "")
            title = doc.get("title", "")
            source_type = doc.get("source_type", "web")

            words = content.split()
            chunk_size, overlap = 512, 64
            step = chunk_size - overlap
            for i in range(0, max(1, len(words)), step):
                chunk_words = words[i : i + chunk_size]
                if chunk_words:
                    chunk_text = " ".join(chunk_words)
                    self._chunks.append(
                        {"text": chunk_text, "url": url, "title": title, "source_type": source_type}
                    )
                    texts.append(chunk_text)

        if not texts:
            logger.warning("RAGEngine: no text chunks to index.")
            return

        embedder = self._get_embedder()
        embeddings = embedder.encode(texts, show_progress_bar=False, batch_size=32).astype(
            np.float32
        )

        dim = embeddings.shape[1]
        faiss.normalize_L2(embeddings)
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)
        logger.info(f"FAISS index built: {len(texts)} chunks, dim={dim}")

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """Return top-k chunks most similar to query."""
        import faiss

        if self._index is None or not self._chunks:
            logger.warning("RAGEngine index not built.")
            return []

        embedder = self._get_embedder()
        q_emb = embedder.encode([query], show_progress_bar=False).astype(np.float32)
        faiss.normalize_L2(q_emb)

        distances, indices = self._index.search(q_emb, min(k, len(self._chunks)))
        results: list[dict] = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self._chunks):
                chunk = dict(self._chunks[idx])
                chunk["score"] = float(dist)
                results.append(chunk)
        return results

    async def retrieve_async(self, query: str, k: int = 5) -> list[dict]:
        """Async wrapper around retrieve()."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.retrieve, query, k)


# Per-goal singleton cache
_rag_cache: dict[str, RAGEngine] = {}


def get_rag_engine(
    goal_id: str,
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> RAGEngine:
    """Return (or create) a RAGEngine for the given goal."""
    if goal_id not in _rag_cache:
        _rag_cache[goal_id] = RAGEngine(embed_model_name)
    return _rag_cache[goal_id]
