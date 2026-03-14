"""Async wrappers around a local Ollama instance."""

import json
import re

from ollama import AsyncClient

from app.config import settings

_client = AsyncClient(host=settings.OLLAMA_BASE_URL)


async def call_claude_json(system: str, user: str) -> dict:
    """Call the local LLM and parse the JSON response."""
    response = await _client.chat(
        model=settings.OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={"temperature": 0.3},
    )
    raw = response.message.content.strip()
    # Strip markdown fences if the model wraps its output
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


async def call_claude_text(system: str, user: str) -> str:
    """Call the local LLM and return plain text."""
    response = await _client.chat(
        model=settings.OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={"temperature": 0.7},
    )
    return response.message.content.strip()
