from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from app.agent.runner import AgentRunner
from app.config import Settings
from app.llm.base import LLMClient
from app.llm.deepseek import DeepSeekClient
from app.llm.fake import FakeLLM
from app.rag.index import RetrievalIndex
from app.rag.ingest import DocumentService
from app.rag.vectors import Embedder, build_embedder
from app.store.db import Database
from app.store.repo import Repository
from app.tools.kb import SearchKnowledgeBaseTool
from app.tools.orders import LookupOrderTool
from app.tools.registry import ToolRegistry
from app.tools.tickets import CreateTicketTool

logger = logging.getLogger(__name__)


def create_llm(settings: Settings) -> LLMClient:
    if settings.llm_provider == "mock":
        logger.warning("LLM_PROVIDER=mock：使用离线规则模型，仅用于演示和测试")
        return FakeLLM()
    return DeepSeekClient(settings)


@dataclass
class AppContainer:
    settings: Settings
    db: Database
    repo: Repository
    embedder: Embedder
    index: RetrievalIndex
    llm: LLMClient
    tools: ToolRegistry
    agent: AgentRunner
    documents: DocumentService
    stream_semaphore: asyncio.Semaphore
    session_locks: dict[str, asyncio.Lock] = field(default_factory=dict)

    def session_lock(self, session_id: str) -> asyncio.Lock:
        lock = self.session_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self.session_locks[session_id] = lock
        return lock

    async def close(self) -> None:
        await self.tools.close()
        await self.llm.close()
        await self.embedder.close()


async def build_container(
    settings: Settings, *, llm: LLMClient | None = None
) -> AppContainer:
    settings.validate_runtime()
    settings.ensure_dirs()
    db = Database(settings.db_path)
    await db.init()
    repo = Repository(db)

    embedder = build_embedder(settings)
    index = RetrievalIndex(repo, embedder, settings)
    llm_client = llm or create_llm(settings)

    kb_tool = SearchKnowledgeBaseTool(index)
    order_tool = LookupOrderTool(repo)
    ticket_tool = CreateTicketTool(
        repo,
        base_url=settings.resolved_itsm_base_url,
        api_key=settings.itsm_api_key_value,
        timeout=settings.itsm_timeout_seconds,
        max_tickets_per_session=settings.max_tickets_per_session,
    )
    registry = ToolRegistry(
        [kb_tool, order_tool, ticket_tool],
        max_result_chars=settings.max_tool_result_chars,
    )
    agent = AgentRunner(
        settings=settings,
        llm=llm_client,
        tools=registry,
        repo=repo,
    )
    documents = DocumentService(repo, embedder, index, settings)

    if settings.auto_seed:
        try:
            seed_result = await documents.seed(force=False)
            logger.info("启动时种子数据：%s", seed_result.summary)
        except Exception:
            logger.exception("启动时导入种子数据失败，服务仍可启动")
            await llm_client.close()
            await embedder.close()
            raise

    return AppContainer(
        settings=settings,
        db=db,
        repo=repo,
        embedder=embedder,
        index=index,
        llm=llm_client,
        tools=registry,
        agent=agent,
        documents=documents,
        stream_semaphore=asyncio.Semaphore(settings.max_concurrent_streams),
    )


__all__ = ["AppContainer", "build_container", "create_llm"]
