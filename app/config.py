from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ConfigError(RuntimeError):
    """Raised when the application is misconfigured."""


class Settings(BaseSettings):
    """Environment-driven configuration.

    ``.env`` is resolved relative to the project root so scripts and uvicorn can
    be started from any working directory.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Application
    app_name: str = "SmartDesk AI"
    app_env: Literal["development", "production", "test"] = "development"
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    data_dir: Path = Path("./data")
    auto_seed: bool = True
    admin_token: SecretStr = SecretStr("")
    demo_customer_id: str = "C001"
    access_username: str = "demo"
    access_password: SecretStr = SecretStr("")

    # LLM
    llm_provider: Literal["deepseek", "mock"] = "mock"
    llm_base_url: str = "https://api.deepseek.com"
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = "deepseek-v4-flash"
    llm_thinking: Literal["enabled", "disabled", "unset"] = "disabled"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 1200
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2

    # Embeddings
    embedding_provider: Literal["hash", "openai_compatible"] = "hash"
    embedding_base_url: str = "https://api.siliconflow.cn/v1"
    embedding_api_key: SecretStr = SecretStr("")
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 512
    embedding_timeout_seconds: float = 30.0

    # Mock ITSM integration
    itsm_base_url: str = ""
    itsm_api_key: SecretStr = SecretStr("demo-itsm-key")
    itsm_timeout_seconds: float = 10.0

    # Limits / retrieval
    max_upload_mb: int = 5
    max_pdf_pages: int = 20
    chunk_size: int = 420
    chunk_overlap: int = 80
    retrieval_top_k: int = 6
    retrieval_candidates: int = 20
    rrf_k: int = 60
    max_agent_steps: int = 6
    max_tool_calls_per_step: int = 3
    max_tickets_per_session: int = 3
    history_messages: int = 8
    session_max_chars: int = 2000
    max_tool_result_chars: int = 4000
    max_concurrent_streams: int = 4
    sse_heartbeat_seconds: float = 15.0

    @model_validator(mode="after")
    def _resolve_paths_and_validate(self) -> Settings:
        if not self.data_dir.is_absolute():
            self.data_dir = (PROJECT_ROOT / self.data_dir).resolve()
        if self.chunk_size < 120:
            raise ValueError("CHUNK_SIZE 不能小于 120")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("CHUNK_OVERLAP 必须小于 CHUNK_SIZE 且不小于 0")
        if self.retrieval_top_k < 1:
            raise ValueError("RETRIEVAL_TOP_K 必须大于 0")
        if self.retrieval_candidates < self.retrieval_top_k:
            raise ValueError("RETRIEVAL_CANDIDATES 不能小于 RETRIEVAL_TOP_K")
        if self.embedding_dim < 64:
            raise ValueError("EMBEDDING_DIM 不能小于 64")
        if self.max_agent_steps < 1:
            raise ValueError("MAX_AGENT_STEPS 必须大于 0")
        return self

    @property
    def db_path(self) -> Path:
        return self.data_dir / "smartdesk.db"

    @property
    def seed_dir(self) -> Path:
        return PROJECT_ROOT / "seed"

    @property
    def web_dir(self) -> Path:
        return Path(__file__).resolve().parent / "web"

    @property
    def resolved_itsm_base_url(self) -> str:
        if self.itsm_base_url.strip():
            return self.itsm_base_url.rstrip("/")
        return f"http://127.0.0.1:{self.app_port}"

    @property
    def is_mock_llm(self) -> bool:
        return self.llm_provider == "mock"

    @property
    def llm_api_key_value(self) -> str:
        return self.llm_api_key.get_secret_value()

    @property
    def embedding_api_key_value(self) -> str:
        return self.embedding_api_key.get_secret_value()

    @property
    def itsm_api_key_value(self) -> str:
        return self.itsm_api_key.get_secret_value()

    @property
    def admin_token_value(self) -> str:
        return self.admin_token.get_secret_value()

    @property
    def access_password_value(self) -> str:
        return self.access_password.get_secret_value()

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def validate_runtime(self) -> None:
        """Fail fast with actionable messages for real-provider misconfiguration."""
        if self.llm_provider == "deepseek" and (
            not self.llm_api_key_value
            or self.llm_api_key_value.strip() in {"sk-your-key-here", "your-key"}
        ):
            raise ConfigError(
                "LLM_PROVIDER=deepseek 但 LLM_API_KEY 未填写。请在 .env 中填入自己的 Key，"
                "或设置 LLM_PROVIDER=mock 使用离线演示模式。"
            )
        if (
            self.embedding_provider == "openai_compatible"
            and not self.embedding_api_key_value
        ):
            raise ConfigError(
                "EMBEDDING_PROVIDER=openai_compatible 但 EMBEDDING_API_KEY 为空。"
                "请填写 Key，或使用默认的 EMBEDDING_PROVIDER=hash。"
            )
        if self.app_env == "production" and not self.admin_token_value:
            raise ConfigError(
                "生产环境必须设置 ADMIN_TOKEN，否则文档管理端点没有鉴权。"
            )

    def safe_summary(self) -> dict[str, object]:
        """Configuration that is safe to expose in health/stats endpoints."""
        return {
            "app": self.app_name,
            "env": self.app_env,
            "llm_provider": self.llm_provider,
            "llm_model": "mock" if self.is_mock_llm else self.llm_model,
            "embedding_provider": self.embedding_provider,
            "embedding_model": (
                f"hash-ngram-{self.embedding_dim}"
                if self.embedding_provider == "hash"
                else self.embedding_model
            ),
            "demo_customer_id": self.demo_customer_id,
            "admin_auth_enabled": bool(self.admin_token_value),
            "access_auth_enabled": bool(self.access_password_value),
        }


__all__ = ["ConfigError", "PROJECT_ROOT", "Settings"]
