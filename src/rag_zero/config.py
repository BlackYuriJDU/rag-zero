"""Configuração central (variáveis de ambiente com prefixo RAGZERO_)."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAGZERO_", env_file=".env", extra="ignore")

    database_url: str | None = None
    user_agent: str = "rag-zero/0.0.1 (+https://github.com/BlackYuriJDU/rag-zero)"
    # Intervalo mínimo entre requisições ao PNCP, em segundos (educação com APIs públicas).
    min_request_interval: float = 0.7
    data_dir: str = "data"


settings = Settings()
