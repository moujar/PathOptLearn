import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)

_SRC_DIR  = Path(__file__).parent
_ROOT_DIR = _SRC_DIR.parent

# ── LLM ─────────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")   # "ollama" or "groq"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ── Search ───────────────────────────────────────────────────
MAX_SEARCH_RESULTS = 6
MAX_CHUNK_CHARS    = 1500
REQUEST_TIMEOUT    = 10
YT_MAX_RESULTS     = 4

# ── Database ─────────────────────────────────────────────────
DB_PATH = str(_ROOT_DIR / "db" / "center.db")

# ── Email (Brevo SMTP — free tier: 300 emails/day) ───────────
SMTP_HOST       = "smtp-relay.brevo.com"
SMTP_PORT       = 587
SENDER_EMAIL    = os.getenv("SENDER_EMAIL", "")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD", "")

# ── External API (optional Colab endpoint) ───────────────────
COLAB_API_URL = os.getenv("COLAB_API_URL", "")
