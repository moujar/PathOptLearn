#!/usr/bin/env python3
"""Run PathOptLearn FastAPI backend (like DeepTutor backend)."""

import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("BACKEND_PORT", "8001"))
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=port,
        reload=os.environ.get("RELOAD", "").lower() in ("1", "true", "yes"),
    )
