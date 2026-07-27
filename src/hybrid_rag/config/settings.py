"""Application configuration via pydantic-settings."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnv(StrEnum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class VectorStoreBackend(StrEnum):
    FAISS = "faiss"
    CHROMA = "chroma"


class RetrieverStrategy(StrEnum):
    VECTOR_STORE = "vector_store"
    MULTI_QUERY = "multi_query"
    SELF_QUERY = "self_query"
    PARENT_DOCUMENT = "parent_document"


class Settings(BaseSettings):
    """Central configuration. Loaded from env vars / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM
    openai_api_key: SecretStr = Field(..., description="OpenAI API key")
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=1024, ge=64, le=8192)
    llm_timeout_seconds: int = Field(default=60, ge=5)
    llm_max_retries: int = Field(default=3, ge=0, le=10)

    # Retrieval
    default_vector_store: VectorStoreBackend = VectorStoreBackend.FAISS
    default_retriever: RetrieverStrategy = RetrieverStrategy.VECTOR_STORE
    top_k: int = Field(default=4, ge=1, le=50)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    chunk_size: int = Field(default=500, ge=100)
    chunk_overlap: int = Field(default=100, ge=0)
    parent_chunk_size: int = Field(default=2000, ge=200)
    child_chunk_size: int = Field(default=400, ge=50)

    # Paths
    data_dir: Path = Path("./data")
    storage_dir: Path = Path("./storage")
    faiss_index_dir: Path = Path("./storage/faiss")
    chroma_persist_dir: Path = Path("./storage/chroma")
    docstore_dir: Path = Path("./storage/docstore")
    collection_name: str = "hybrid_rag_docs"

    # App
    app_env: AppEnv = AppEnv.DEVELOPMENT
    log_level: str = "INFO"
    log_json: bool = False
    enable_metrics: bool = True
    gradio_server_name: str = "0.0.0.0"
    gradio_server_port: int = Field(default=7860, ge=1, le=65535)
    gradio_share: bool = False
    cache_ttl_seconds: int = Field(default=300, ge=0)

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    @property
    def is_production(self) -> bool:
        return self.app_env == AppEnv.PRODUCTION

    def ensure_directories(self) -> None:
        for path in (
            self.data_dir,
            self.storage_dir,
            self.faiss_index_dir,
            self.chroma_persist_dir,
            self.docstore_dir,
            Path("logs"),
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()  # type: ignore[call-arg]
    settings.ensure_directories()
    return settings
