"""Settings, loaded from env (prefix GOVRAG_) or a .env file. Nothing secret has a default."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GOVRAG_", env_file=".env", extra="ignore")

    # --- Postgres ---
    db_host: str = "127.0.0.1"
    db_port: int = 55432
    db_name: str = "govrag"
    db_owner_user: str = "govrag_owner"
    db_owner_password: SecretStr = SecretStr("")
    db_app_password: SecretStr
    db_vault_password: SecretStr
    db_ingest_password: SecretStr
    # Pool sizes are the DB's backpressure: when a pool is empty, acquire() waits at most
    # db_acquire_timeout_s, then the request gets a 503. Never an unbounded wait.
    db_app_pool_max: int = 20
    db_vault_pool_max: int = 5
    db_ingest_pool_max: int = 4
    db_acquire_timeout_s: float = 2.0
    db_statement_timeout_ms: int = 5_000

    # --- Keys ---
    # Comma-separated Fernet keys. The first one encrypts; all of them decrypt (MultiFernet).
    # They wrap the per-tenant keys stored in the DB (envelope encryption).
    master_keys: SecretStr

    # --- Auth ---
    jwt_public_key_path: Path = Path("keys/jwt_public.pem")
    jwt_issuer: str = "https://idp.govrag.local"
    jwt_audience: str = "govrag-api"
    jwt_leeway_s: int = 30
    jwt_max_lifetime_s: int = 3600

    # --- Retrieval ---
    top_k_default: int = 5
    top_k_max: int = 20
    hnsw_ef_search: int = 100
    # Upper bound on tuples an iterative scan may visit before giving up. Caps the cost of
    # a query for a role that can see almost nothing.
    hnsw_max_scan_tuples: int = 20_000

    # --- Limits (backpressure) ---
    max_inflight_queries: int = 64
    max_inflight_per_tenant: int = 16
    max_inflight_ingests: int = 2
    max_question_chars: int = 2_000
    max_document_chars: int = 500_000
    max_section_chars: int = 100_000
    request_deadline_s: float = 30.0
    max_context_chars: int = 24_000
    llm_max_tokens: int = 800

    # --- PII ---
    # Each Presidio worker holds its own spaCy model (~700 MB RSS). Pool size is bounded
    # by RAM, not cores.
    pii_ingest_workers: int = 2
    pii_query_workers: int = 1
    spacy_model: str = "en_core_web_lg"
    pii_score_threshold: float = 0.4

    # --- Embeddings ---
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384
    embed_concurrency: int = 2

    # --- Policy cache ---
    policy_cache_ttl_s: float = 30.0

    # --- LLM ---
    llm_backend: str = Field(default="fake", pattern="^(fake|vertex)$")
    vertex_project: str = ""
    llm_timeout_s: float = 20.0
    llm_concurrency: int = 16
    breaker_failures: int = 3
    breaker_cooldown_s: float = 30.0

    # --- Tracing ---
    otlp_endpoint: str = ""  # e.g. http://127.0.0.1:4318/v1/traces. Empty = no export.

    @field_validator("db_app_pool_max", "db_vault_pool_max", "db_ingest_pool_max")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("pool size must be >= 1")
        return v

    def dsn(self, user: str, password: SecretStr) -> str:
        return (
            f"postgresql://{user}:{password.get_secret_value()}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # required fields come from env
