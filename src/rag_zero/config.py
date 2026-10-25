"""Configuração central (variáveis de ambiente com prefixo RAGZERO_)."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAGZERO_", env_file=".env", extra="ignore")

    # Armazenamento local. SQLite + FTS5 + vetores em BLOB (ver ADR-0001).
    db_path: str = "data/rag_zero.db"
    data_dir: str = "data"

    # Identificação da aplicação nas requisições ao PNCP (educação com APIs públicas).
    user_agent: str = "rag-zero/0.0.1 (+https://github.com/BlackYuriJDU/rag-zero)"
    min_request_interval: float = 0.7

    # Recuperação.
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dim: int = 384
    rrf_k: int = 60
    top_k: int = 10

    # Portão de recusa: cobertura IDF dos termos da pergunta no corpus.
    # Calibrado no golden set (ver evals/README.md).
    min_evidence_coverage: float = 0.55
    # Similaridade mínima do melhor chunk denso, quando o modo usa embeddings.
    min_dense_similarity: float = 0.20

    @property
    def db_file(self) -> Path:
        return Path(self.db_path)


settings = Settings()
