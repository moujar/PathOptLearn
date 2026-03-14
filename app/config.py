from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database
    DATABASE_URL: str = "postgresql://user:pass@localhost:5432/learnflow"

    # Ollama — local LLM + embeddings (no API key needed)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"          # chat model
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"  # 768-dim embeddings
    EMBEDDING_DIM: int = 768

    # LangSmith observability (optional)
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "learnflow"

    # App
    FASTAPI_PORT: int = 8000
    STREAMLIT_PORT: int = 8501

    @property
    def async_database_url(self) -> str:
        return self.DATABASE_URL.replace(
            "postgresql://", "postgresql+asyncpg://"
        )

    @property
    def sync_database_url(self) -> str:
        """Used by LangGraph PostgresSaver (needs psycopg)."""
        return self.DATABASE_URL.replace(
            "postgresql://", "postgresql+psycopg://"
        )


settings = Settings()
