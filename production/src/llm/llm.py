import sys
import os
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import LLM_PROVIDER, OLLAMA_MODEL, GROQ_API_KEY, GROQ_MODEL


def llm(prompt: str, system: str = "") -> str:
    """Unified LLM call. Routes to Ollama or Groq based on LLM_PROVIDER config."""
    if LLM_PROVIDER == "ollama":
        return _ollama(prompt, system)
    return _groq(prompt, system)


def _ollama(prompt: str, system: str) -> str:
    try:
        import ollama as ol
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        result = ol.chat(model=OLLAMA_MODEL, messages=messages)
        return result["message"]["content"]
    except Exception as e:
        return f"[Ollama error: {e}]"


def _groq(prompt: str, system: str) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": messages,
                "max_tokens": 2048,
                "temperature": 0.7,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[Groq error: {e}]"
