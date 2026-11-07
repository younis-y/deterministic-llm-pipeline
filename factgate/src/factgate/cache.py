"""The answer cache: the model's raw answer, keyed so it is never served stale.

The key names the item's text, a fingerprint of everything that decides the
answer (the prompt, the schema, `max_tokens`, `temperature`), the backend and
the model. Checking and deciding run on every read, so a rule or resolver
change reaches cached items with no call (rolescan ADR-0008).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import aiosqlite

from factgate.items import Item

#: Bumped by hand only when what a cached row holds changes meaning.
CACHE_VERSION = "v1"


def fingerprint(system: str, schema_json: str, max_tokens: int, temperature: float | None) -> str:
    blob = json.dumps([CACHE_VERSION, system, schema_json, max_tokens, temperature])
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def cache_key(item: Item, fp: str, backend: str, model: str) -> str:
    return f"{item.content_hash}:{fp}:{backend}:{model}"


class Cache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._db: aiosqlite.Connection | None = None

    async def __aenter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.path)
        await self._db.execute(
            "CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, raw TEXT, written TEXT)"
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._db is not None:
            await self._db.commit()
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("use the cache as `async with Cache(path) as cache`")
        return self._db

    async def get(self, key: str) -> dict[str, Any] | None:
        async with self.db.execute("SELECT raw FROM answers WHERE key = ?", (key,)) as cur:
            row = await cur.fetchone()
        return None if row is None else dict(json.loads(row[0]))

    async def put(self, key: str, raw: dict[str, Any]) -> None:
        written = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
        await self.db.execute(
            "INSERT OR REPLACE INTO answers VALUES (?, ?, ?)", (key, json.dumps(raw), written)
        )
        await self.db.commit()
