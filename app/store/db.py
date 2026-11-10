from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiosqlite

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class Database:
    """Small aiosqlite wrapper.

    A connection is opened per operation and closed immediately. That keeps the
    demo free of pool/event-loop pitfalls while WAL + ``busy_timeout`` keep
    concurrent readers and the short write transactions safe. Writes are
    serialized inside one process with an ``asyncio.Lock``.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._write_lock = asyncio.Lock()

    async def init(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        schema = SCHEMA_PATH.read_text(encoding="utf-8")
        async with aiosqlite.connect(self.path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA synchronous=NORMAL")
            await db.executescript(schema)
            await db.execute(
                "INSERT INTO schema_meta(key, value) VALUES('schema_version', '1') "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value"
            )
            await db.commit()

    async def _connect(self) -> aiosqlite.Connection:
        db = await aiosqlite.connect(self.path, timeout=5.0)
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA foreign_keys=ON")
        await db.execute("PRAGMA busy_timeout=5000")
        return db

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[aiosqlite.Connection]:
        db = await self._connect()
        try:
            yield db
        finally:
            await db.close()

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[aiosqlite.Connection]:
        """Exclusive short write transaction.

        Never await an LLM/HTTP call while inside this context.
        """
        async with self._write_lock:
            db = await self._connect()
            try:
                yield db
                await db.commit()
            except BaseException:
                await db.rollback()
                raise
            finally:
                await db.close()

    async def fetch_all(
        self, sql: str, params: Sequence[Any] | None = None
    ) -> list[aiosqlite.Row]:
        async with self.connection() as db, db.execute(sql, tuple(params or ())) as cursor:
            return list(await cursor.fetchall())

    async def fetch_one(
        self, sql: str, params: Sequence[Any] | None = None
    ) -> aiosqlite.Row | None:
        async with self.connection() as db, db.execute(sql, tuple(params or ())) as cursor:
            return await cursor.fetchone()

    async def execute(self, sql: str, params: Sequence[Any] | None = None) -> int:
        async with self.transaction() as db:
            cursor = await db.execute(sql, tuple(params or ()))
            return cursor.rowcount

    async def ping(self) -> bool:
        row = await self.fetch_one("SELECT 1 AS ok")
        return bool(row and row["ok"] == 1)


__all__ = ["Database"]
