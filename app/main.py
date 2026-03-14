from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db import init_db
from app.graph.learning_graph import setup_graph
from app.api import sessions, modules, quizzes


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialise PostgreSQL schema + pgvector extension
    await init_db()
    # Initialise LangGraph with PostgreSQL checkpointer
    await setup_graph()
    yield


app = FastAPI(
    title="LearnFlow AI",
    description="Adaptive learning platform — FastAPI + LangGraph + PostgreSQL + pgvector",
    version="2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sessions.router, prefix="/api/sessions", tags=["sessions"])
app.include_router(modules.router, prefix="/api/modules", tags=["modules"])
app.include_router(quizzes.router, prefix="/api/quizzes", tags=["quizzes"])


@app.get("/health")
async def health():
    return {"status": "ok", "service": "learnflow", "version": "2.0"}
