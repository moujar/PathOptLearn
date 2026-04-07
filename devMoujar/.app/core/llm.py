import ollama
from app.config import MODEL


def llm(prompt: str, system: str = "") -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    result = ollama.chat(model=MODEL, messages=messages)
    return result["message"]["content"]
