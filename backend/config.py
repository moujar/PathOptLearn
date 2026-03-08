"""Configuration via pydantic-settings — loads from .env file."""
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4"

    # External APIs
    tavily_api_key: str = ""
    youtube_api_key: str = ""

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # Auth stub
    api_key: str = "dev-secret"

    # ML models
    model_checkpoint_dir: str = "models/checkpoints"
    faiss_index_path: str = "data/faiss"
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    drl_model_path: str = "models/checkpoints/drl_ppo.zip"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Return cached Settings singleton."""
    return Settings()
