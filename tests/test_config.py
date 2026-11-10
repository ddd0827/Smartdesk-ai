from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from app.config import ConfigError, Settings


def make_settings(**overrides) -> Settings:
    values = dict(
        _env_file=None,
        app_env="test",
        data_dir=Path("data"),
        llm_provider="mock",
        embedding_provider="hash",
        admin_token=SecretStr(""),
    )
    values.update(overrides)
    return Settings(**values)


def test_defaults_and_resolved_paths() -> None:
    settings = make_settings()
    assert settings.is_mock_llm is True
    assert settings.data_dir.is_absolute()
    assert settings.db_path.name == "smartdesk.db"
    validation = settings.safe_summary()
    assert validation["llm_provider"] == "mock"
    assert validation["admin_auth_enabled"] is False


def test_deepseek_requires_a_real_key() -> None:
    settings = make_settings(
        llm_provider="deepseek", llm_api_key=SecretStr("")
    )
    with pytest.raises(ConfigError, match="LLM_API_KEY"):
        settings.validate_runtime()


def test_deepseek_placeholder_key_is_rejected() -> None:
    settings = make_settings(
        llm_provider="deepseek", llm_api_key=SecretStr("sk-your-key-here")
    )
    with pytest.raises(ConfigError, match="LLM_API_KEY"):
        settings.validate_runtime()


def test_production_requires_admin_token() -> None:
    settings = make_settings(app_env="production")
    with pytest.raises(ConfigError, match="ADMIN_TOKEN"):
        settings.validate_runtime()


def test_remote_embedding_requires_key() -> None:
    settings = make_settings(
        embedding_provider="openai_compatible",
        embedding_api_key=SecretStr(""),
    )
    with pytest.raises(ConfigError, match="EMBEDDING_API_KEY"):
        settings.validate_runtime()


def test_invalid_chunk_configuration_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_settings(chunk_size=100, chunk_overlap=10)


def test_secret_values_are_not_in_safe_summary() -> None:
    settings = make_settings(
        llm_api_key=SecretStr("super-secret-value"),
        itsm_api_key=SecretStr("itsm-secret"),
    )
    text = str(settings.safe_summary())
    assert "super-secret-value" not in text
    assert "itsm-secret" not in text
