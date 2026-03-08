"""AdaptLearn AI — FastAPI application entry point."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.routers import goal, quiz, assessment, kg, recommend, loop, explain
from backend.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise connections on startup; clean up on shutdown."""
    settings = get_settings()
    logger.info("AdaptLearn AI starting up…")

    # Redis
    try:
        import redis.asyncio as aioredis
        app.state.redis = aioredis.from_url(settings.redis_url, decode_responses=True)
        await app.state.redis.ping()
        logger.info(f"Redis connected: {settings.redis_url}")
    except Exception as exc:
        logger.warning(f"Redis not available: {exc}")
        app.state.redis = None

    yield

    # Shutdown
    if app.state.redis:
        await app.state.redis.aclose()
    logger.info("AdaptLearn AI shut down.")


# ── App ───────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AdaptLearn AI",
    description=(
        "Adaptive AI System for Personalised Learning Path Recommendation & "
        "Knowledge State Prediction — Master's Thesis Implementation"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Auth middleware (stub) ────────────────────────────────────────────────

@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    """Verify Bearer token for all routes except /health, /docs, /redoc, /openapi.json."""
    exempt = {"/health", "/docs", "/redoc", "/openapi.json"}
    if request.url.path in exempt or request.url.path.startswith("/docs"):
        return await call_next(request)

    auth = request.headers.get("Authorization", "")
    settings = get_settings()
    if auth and auth.startswith("Bearer "):
        token = auth.split(" ", 1)[1]
        if token == settings.api_key:
            return await call_next(request)

    # Allow unauthenticated in dev mode (api_key = 'dev-secret')
    if settings.api_key == "dev-secret":
        return await call_next(request)

    return JSONResponse({"detail": "Unauthorized", "code": "AUTH_REQUIRED"}, status_code=401)


# ── Routers ───────────────────────────────────────────────────────────────

PREFIX = "/api/v1"

app.include_router(goal.router,       prefix=PREFIX)
app.include_router(quiz.router,       prefix=PREFIX)
app.include_router(assessment.router, prefix=PREFIX)
app.include_router(kg.router,         prefix=PREFIX)
app.include_router(recommend.router,  prefix=PREFIX)
app.include_router(loop.router,       prefix=PREFIX)
app.include_router(explain.router,    prefix=PREFIX)


# ── Health ────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Utility"])
async def health():
    """Service health check."""
    return {"status": "ok", "service": "AdaptLearn AI", "version": "1.0.0"}
