"""`seen` at scale and over time (2.5.8).

`filter_new` bound one SQL variable per posting, so a scan bigger than
SQLite's limit (999 before 3.32) failed outright; `seen.last_seen` never moved,
so nothing could say when a role was last on a board; and `unsee` looked a
posting up by url with a full scan of `seen`."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import httpx
import respx

from conftest import scan_and_record
from rolescan.config import Config
from rolescan.models import Job, ScoredJob
from rolescan.pipeline import run_scan
from rolescan.store import _MIGRATIONS, Store

OLD = "2026-09-01T00:00:00+00:00"
BOARD = {
    "jobs": [
        {
            "id": 1,
            "title": "Graduate Data Scientist, Energy Trading",
            "location": {"name": "London, UK"},
            "absolute_url": "https://boards.greenhouse.io/acme/jobs/1",
            "content": "<p>Python, energy, trading, forecasting.</p>",
            "updated_at": "2026-08-20T10:00:00Z",
        }
    ]
}


def _posting(i: int) -> ScoredJob:
    return ScoredJob(
        job=Job(
            source="t",
            company=f"Acme {i}",
            title="Data Analyst",
            location="London",
            url=f"https://acme.example/{i}",
        )
    )


async def _last_seen(store: Store) -> dict[str, str]:
    rows = await store.db.execute_fetchall("SELECT uid, last_seen FROM seen")
    return {str(r[0]): str(r[1]) for r in rows}


async def test_filter_new_works_under_the_oldest_sqlite_variable_limit(
    tmp_path: Path,
) -> None:
    """SQLite before 3.32 binds at most 999 variables per statement; the
    limit is lowered on this connection to reproduce it on any build."""
    batch = [_posting(i) for i in range(1500)]
    async with Store(tmp_path / "s.db") as store:
        # aiosqlite's worker thread owns the sqlite3 connection, so the limit
        # is set there (`_execute` is how aiosqlite runs a call on it).
        await store.db._execute(  # type: ignore[no-untyped-call]
            store.db._conn.setlimit, sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999
        )
        await store.record_all(batch[:700])
        fresh = await store.filter_new(batch)

    assert [s.job.uid for s in fresh] == [s.job.uid for s in batch[700:]]


async def test_touch_refreshes_last_seen_for_a_role_still_listed(
    tmp_path: Path,
) -> None:
    listed, gone = _posting(1), _posting(2)
    async with Store(tmp_path / "s.db") as store:
        await store.record_all([listed, gone])
        await store.db.execute("UPDATE seen SET last_seen = ?", (OLD,))
        await store.db.commit()

        assert await store.filter_new([listed], touch=True) == []

        last_seen = await _last_seen(store)
    assert last_seen[listed.job.uid] > "2026-10-01"
    assert last_seen[gone.job.uid] == OLD, "not listed this run: unchanged"


async def test_filter_new_without_touch_writes_nothing(tmp_path: Path) -> None:
    posting = _posting(1)
    async with Store(tmp_path / "s.db") as store:
        await store.record_all([posting])
        await store.db.execute("UPDATE seen SET last_seen = ?", (OLD,))
        await store.db.commit()

        await store.filter_new([posting])

        assert await _last_seen(store) == {posting.job.uid: OLD}


async def test_a_url_lookup_uses_an_index_and_the_duplicate_index_is_gone(
    tmp_path: Path,
) -> None:
    async with Store(tmp_path / "s.db") as store:
        plan = await store.db.execute_fetchall(
            "EXPLAIN QUERY PLAN SELECT uid FROM seen WHERE url = ?", ("x",)
        )
        indexes = {
            str(r[0])
            for r in await store.db.execute_fetchall(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }
    assert any("seen_url" in str(row[-1]) for row in plan), plan
    assert "source_counts_key" not in indexes
    assert len(_MIGRATIONS) == 8


@respx.mock
async def test_a_dry_run_leaves_last_seen_alone_and_a_real_run_moves_it(
    config: Config,
) -> None:
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(200, json=BOARD)
    )
    db = config.resolve(config.output.db_path)
    await scan_and_record(config)
    async with Store(db) as store:
        await store.db.execute("UPDATE seen SET last_seen = ?", (OLD,))
        await store.db.commit()

    await run_scan(config, dry_run=True)
    async with Store(db) as store:
        assert list((await _last_seen(store)).values()) == [OLD]

    await run_scan(config)
    async with Store(db) as store:
        [moved] = (await _last_seen(store)).values()
    assert moved > "2026-10-01"
