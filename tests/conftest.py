from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings  # noqa: E402
from app.llm.fake import FakeLLM  # noqa: E402
from app.main import create_app  # noqa: E402
from app.rag.index import RetrievalIndex  # noqa: E402
from app.rag.ingest import DocumentService  # noqa: E402
from app.rag.vectors import HashNgramVectorizer  # noqa: E402
from app.store.db import Database  # noqa: E402
from app.store.repo import Repository  # noqa: E402


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        app_env="test",
        data_dir=tmp_path / "data",
        auto_seed=False,
        llm_provider="mock",
        admin_token=SecretStr(""),
        embedding_provider="hash",
        embedding_dim=128,
        itsm_base_url="http://testserver",
        itsm_api_key=SecretStr("demo-itsm-key"),
        sse_heartbeat_seconds=5,
    )


@pytest.fixture
async def db(settings: Settings) -> Database:
    database = Database(settings.db_path)
    await database.init()
    return database


@pytest.fixture
def repo(db: Database) -> Repository:
    return Repository(db)


@pytest.fixture
def embedder(settings: Settings) -> HashNgramVectorizer:
    return HashNgramVectorizer(settings.embedding_dim)


@pytest.fixture
async def seeded(
    settings: Settings, repo: Repository, embedder: HashNgramVectorizer
) -> SimpleNamespace:
    index = RetrievalIndex(repo, embedder, settings)
    service = DocumentService(repo, embedder, index, settings)
    result = await service.seed(force=False)
    await index.ensure_ready()
    return SimpleNamespace(
        settings=settings,
        repo=repo,
        embedder=embedder,
        index=index,
        service=service,
        seed_result=result,
    )


@pytest.fixture
def live_settings(settings: Settings) -> Settings:
    return settings.model_copy(update={"auto_seed": True})


@pytest.fixture
def app(live_settings: Settings):
    return create_app(live_settings, llm=FakeLLM())


@pytest.fixture
def client(app):
    with TestClient(app) as test_client:
        yield test_client
