"""Runtime configuration, read from environment variables (prefix ``VERITY_``) or a .env file."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VERITY_", env_file=".env", extra="ignore")

    # --- LLM (any OpenAI-compatible endpoint: OpenAI, Groq, OpenRouter, Ollama, vLLM...)
    llm_base_url: str | None = None  # None -> api.openai.com
    llm_api_key: str | None = None
    llm_model: str = "gpt-4.1-mini"
    llm_timeout_s: float = 60.0

    # --- embeddings
    embed_provider: Literal["lsa", "openai"] = "lsa"  # "lsa" runs fully offline
    embed_model: str = "text-embedding-3-small"
    embed_base_url: str | None = None
    embed_api_key: str | None = None
    lsa_dims: int = 256

    # --- data
    kb_dir: Path = Path("knowledge_base")
    index_dir: Path = Path(".verity_index")
    chunk_strategy: Literal["fixed", "recursive", "markdown"] = "markdown"
    chunk_size: int = 700
    chunk_overlap: int = 120

    # --- retrieval
    retrieval_mode: Literal["bm25", "dense", "hybrid"] = "hybrid"
    top_k: int = 8
    candidate_k: int = 30
    use_query_expansion: bool = True
    use_rerank: bool = True
    use_mmr: bool = True

    # --- agent
    max_iterations: int = 2  # corrective retrieval rounds
    verify_answers: bool = True

    # --- observability
    trace_file: Path | None = Path(".verity_traces.jsonl")


def get_settings() -> Settings:
    return Settings()
