"""Application settings loaded from .env via Pydantic.

Single source of truth for configuration. Import `settings` from this module
anywhere you need a config value instead of reading environment variables directly.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM provider
    llm_provider: str = Field(default="deepseek", description="LLM backend")
    llm_model: str = Field(default="deepseek-chat")
    deepseek_api_key: str = Field(default="", description="DeepSeek API key")
    anthropic_api_key: str = Field(default="")
    openai_api_key: str = Field(default="")

    # Embeddings
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5")

    # Vector store
    chroma_persist_dir: Path = Field(default=Path("./chroma_db"))
    collection_name: str = Field(default="aws_lambda_docs")

    # Data
    docs_dir: Path = Field(default=Path("./data/docs/aws_lambda"))

    # Retrieval
    top_k_retrieval: int = Field(default=8, ge=1, le=20)
    vector_top_k: int = Field(default=20, ge=1, le=50, description="Candidates from vector search")
    bm25_top_k: int = Field(default=20, ge=1, le=50, description="Candidates from BM25")
    rrf_k: int = Field(default=60, ge=1, le=200, description="RRF constant")

    # Chunking
    chunk_size: int = Field(default=500, ge=100, le=2000)
    chunk_overlap: int = Field(default=50, ge=0, le=500)

    # Generation
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, ge=100, le=4000)

    # Logging
    log_level: str = Field(default="INFO")


settings = Settings()
