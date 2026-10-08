"""What a failed open or an interrupted write leaves behind (2.5.8).

Three defects from the 2026-10-07 storage audit that 2.5.7 left open: a file
that is not a database failed on the first PRAGMA, outside the guard that
closes the connection, so the process hung after printing its traceback; an
exception inside `async with Store(...)` still COMMITTED on the way out, so an
interrupted `record_all` left a prefix of its rows marked seen and in no
digest; and a file written by a newer rolescan opened silently under older
code."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from contextlib import closing
from pathlib import Path

import pytest

import rolescan.store as store_module
from rolescan.models import Job, ScoredJob
from rolescan.store import Store, StoreTooNewError


def _workers() -> list[str]:
    """aiosqlite's per-connection worker threads still alive."""
    return [t.name for t in threading.enumerate() if "_connection_worker" in t.name]


async def _wait_for_no_workers() -> list[str]:
    for _ in range(50):  # a closed connection's worker exits on its own
        if not _workers():
            break
        await asyncio.sleep(0.02)
    return _workers()


def _version(path: Path) -> int:
    with closing(sqlite3.connect(path)) as conn:
        return int(conn.execute("PRAGMA user_version").fetchone()[0])


async def test_a_file_that_is_not_a_database_fails_without_a_live_worker(
    tmp_path: Path,
) -> None:
    garbage = tmp_path / "garbage.db"
    garbage.write_bytes(b"\x81&_\xdcLc" * 700)

    with pytest.raises(sqlite3.DatabaseError):
        async with Store(garbage):
            pass

    assert await _wait_for_no_workers() == []


async def test_an_interrupted_record_all_records_nothing(tmp_path: Path) -> None:
    items = [
        ScoredJob(
            job=Job(
                source="t",
                company=f"Acme {i}",
                title="Analyst",
                url=f"https://acme.example/{i}",
            )
        )
        for i in range(10)
    ]
    path = tmp_path / "s.db"
    with pytest.raises(KeyboardInterrupt):
        async with Store(path) as store:
            real = store.record
            calls = 0

            async def flaky(item: ScoredJob, *, reason: str = "") -> None:
                nonlocal calls
                calls += 1
                if calls == 4:
                    raise KeyboardInterrupt
                await real(item, reason=reason)

            store.record = flaky  # type: ignore[method-assign, assignment]
            await store.record_all(items)

    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("SELECT count(*) FROM seen").fetchone()[0] == 0


async def test_a_clean_exit_still_commits_what_was_not_committed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "s.db"
    async with Store(path) as store:
        await store.db.execute(
            "INSERT INTO source_counts VALUES ('k', '2026-10-08T00:00:00+00:00', 3)"
        )
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("SELECT count(*) FROM source_counts").fetchone()[0] == 1


async def test_a_newer_file_is_refused_and_left_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "s.db"
    base = store_module._MIGRATIONS
    with monkeypatch.context() as m:
        m.setattr(
            store_module,
            "_MIGRATIONS",
            (*base, "ALTER TABLE seen ADD COLUMN from_the_future TEXT;"),
        )
        async with Store(path):
            pass

    with pytest.raises(StoreTooNewError, match="newer rolescan"):
        async with Store(path):
            pass

    assert _version(path) == len(base) + 1
    assert await _wait_for_no_workers() == []
