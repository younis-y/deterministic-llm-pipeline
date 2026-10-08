"""Retention: every cache is trimmed, nothing the user needs is (2.5.8).

Before 2.5.8 only `verdicts` could be trimmed, by hand, and nothing called
it: `postings` was 83% of a real store after two weeks, `deferred` rows
were never removed, and `source_counts` kept every run for ever."""

from __future__ import annotations

import asyncio
import sqlite3
from contextlib import closing
from pathlib import Path

import httpx
import pytest
import respx
from typer.testing import CliRunner

from conftest import plain
from rolescan.cli import app
from rolescan.config import Config, RetentionConfig
from rolescan.store import PruneReport, Store
from rolescan.storefile import run_lock

OLD = "2020-01-01T00:00:00+00:00"
NEW = "2999-01-01T00:00:00+00:00"


async def _seed(store: Store) -> None:
    db = store.db
    for url, fetched in (
        ("https://old.example/plain", OLD),
        ("https://old.example/applied", OLD),
        ("https://old.example/apply-role", OLD),
        ("https://old.example/skipped-role", OLD),
        ("https://new.example/fresh", NEW),
    ):
        await db.execute("INSERT INTO postings VALUES (?, '', '{}', ?)", (url, fetched))
    await db.execute(
        "INSERT INTO applications VALUES "
        "('https://old.example/applied', 'shortlist', '', '', '2020-01-01')"
    )
    await db.execute(
        "INSERT INTO seen (uid, company, title, url, verdict, first_seen, last_seen) "
        "VALUES ('u1', 'Acme', 'Analyst', 'https://old.example/apply-role', "
        "'apply', ?, ?)",
        (OLD, NEW),
    )
    await db.execute(
        "INSERT INTO seen (uid, company, title, url, verdict, first_seen, last_seen) "
        "VALUES ('u2', 'Acme', 'Clerk', 'https://old.example/skipped-role', "
        "'skip', ?, ?)",
        (OLD, NEW),
    )
    await db.execute(
        "INSERT INTO deferred VALUES ('d-old', 'thin', 1, ?, ?)", (OLD, OLD)
    )
    await db.execute(
        "INSERT INTO deferred VALUES ('d-new', 'thin', 1, ?, ?)", (OLD, NEW)
    )
    await db.execute("INSERT INTO verdicts VALUES ('v-old', '{}', ?)", (OLD,))
    await db.execute("INSERT INTO verdicts VALUES ('v-new', '{}', ?)", (NEW,))
    await db.execute("INSERT INTO source_counts VALUES ('k', ?, 3)", (OLD,))
    await db.execute("INSERT INTO source_counts VALUES ('k', ?, 4)", (NEW,))
    await db.commit()


async def _column(store: Store, sql: str) -> set[str]:
    return {str(r[0]) for r in await store.db.execute_fetchall(sql)}


async def test_prune_all_trims_each_cache_and_keeps_what_is_still_needed(
    tmp_path: Path,
) -> None:
    async with Store(tmp_path / "s.db") as store:
        await _seed(store)

        report = await store.prune_all(
            postings_days=90, deferred_days=45, verdicts_days=180
        )

        assert report == PruneReport(
            verdicts=1, postings=2, deferred=1, source_counts=1, vacuumed=False
        )
        assert await _column(store, "SELECT url FROM postings") == {
            "https://old.example/applied",
            "https://old.example/apply-role",
            "https://new.example/fresh",
        }
        assert await _column(store, "SELECT uid FROM deferred") == {"d-new"}
        assert await _column(store, "SELECT content_hash FROM verdicts") == {"v-new"}
        assert await _column(store, "SELECT uid FROM seen") == {"u1", "u2"}
        assert await _column(store, "SELECT url FROM applications") == {
            "https://old.example/applied"
        }


async def test_zero_days_keeps_a_table_whole(tmp_path: Path) -> None:
    async with Store(tmp_path / "s.db") as store:
        await _seed(store)

        report = await store.prune_all(
            postings_days=0, deferred_days=0, verdicts_days=0
        )

        assert (report.postings, report.deferred, report.verdicts) == (0, 0, 0)
        assert len(await _column(store, "SELECT url FROM postings")) == 5


async def test_a_large_trim_compacts_the_file(tmp_path: Path) -> None:
    async with Store(tmp_path / "s.db") as store:
        await store.db.executemany(
            "INSERT INTO postings VALUES (?, '', ?, ?)",
            [(f"https://old.example/{i}", "x" * 4000, OLD) for i in range(300)],
        )
        await store.db.commit()
        [(before,)] = await store.db.execute_fetchall("PRAGMA page_count")

        report = await store.prune_all(
            postings_days=90, deferred_days=45, verdicts_days=180
        )

        [(after,)] = await store.db.execute_fetchall("PRAGMA page_count")
        [(free,)] = await store.db.execute_fetchall("PRAGMA freelist_count")
    assert report.postings == 300
    assert report.vacuumed
    assert free == 0
    assert after < before / 10


def test_retention_defaults_and_bounds() -> None:
    assert RetentionConfig() == RetentionConfig(postings=90, deferred=45, verdicts=180)
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        Config.model_validate({"output": {"retention_days": {"postings": -1}}})


_CONFIG = """
profile:
  keywords: {energy: 6, python: 4}
  min_keyword_score: 4
  min_report_score: 0
llm:
  enabled: false
sources:
  - {kind: greenhouse, slug: acme, label: Acme Energy}
output:
  dir: digests
  db_path: seen.db
"""
_BOARD = {
    "jobs": [
        {
            "id": 1,
            "title": "Energy Data Analyst",
            "location": {"name": "London, UK"},
            "absolute_url": "https://boards.greenhouse.io/acme/jobs/1",
            "content": "<p>Python and energy markets.</p>",
            "updated_at": "2026-08-20T10:00:00Z",
        }
    ]
}


def _old_posting(path: Path) -> None:
    """A store holding one posting fetched long ago. Synchronous, because the
    CLI tests below run `asyncio.run` themselves."""

    async def go() -> None:
        async with Store(path) as store:
            await store.db.execute(
                "INSERT INTO postings VALUES ('https://old.example/x', '', '{}', ?)",
                (OLD,),
            )
            await store.db.commit()

    asyncio.run(go())


def _has_old_posting(path: Path) -> bool:
    with closing(sqlite3.connect(path)) as conn:
        row = conn.execute(
            "SELECT 1 FROM postings WHERE url = 'https://old.example/x'"
        ).fetchone()
    return row is not None


@pytest.mark.parametrize(("flags", "trimmed"), [([], True), (["--dry"], False)])
@respx.mock
def test_a_real_scan_trims_old_caches_and_a_dry_one_does_not(
    tmp_path: Path, flags: list[str], trimmed: bool
) -> None:
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(200, json=_BOARD)
    )
    config = tmp_path / "config.yaml"
    config.write_text(_CONFIG)
    _old_posting(tmp_path / "seen.db")

    result = CliRunner().invoke(app, ["scan", "-c", str(config), "--no-email", *flags])

    assert result.exit_code == 0, result.output
    assert _has_old_posting(tmp_path / "seen.db") is not trimmed


def test_prune_command_reports_every_table(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(_CONFIG)
    _old_posting(tmp_path / "seen.db")

    result = CliRunner().invoke(app, ["prune", "-c", str(config)])

    assert result.exit_code == 0, result.output
    assert "1 cached postings" in " ".join(plain(result.output).split())


def test_prune_while_a_scan_holds_the_lock_exits_4_and_deletes_nothing(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(_CONFIG)
    _old_posting(tmp_path / "seen.db")

    with run_lock(tmp_path / "seen.db"):
        result = CliRunner().invoke(app, ["prune", "-c", str(config)])

    assert result.exit_code == 4
    assert "another rolescan run" in " ".join(plain(result.output).split())
    assert _has_old_posting(tmp_path / "seen.db")


def test_prune_days_must_be_at_least_one(tmp_path: Path) -> None:
    """`--days 0` dropped every cached verdict in 2.5.7, while 0 in
    `retention_days` now means keep them all; the flag refuses it rather
    than mean the opposite of the config."""
    config = tmp_path / "config.yaml"
    config.write_text(_CONFIG)

    result = CliRunner().invoke(app, ["prune", "-c", str(config), "--days", "0"])

    assert result.exit_code == 2
