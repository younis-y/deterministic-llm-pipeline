"""Shrunk and cut-short sources (2.5.8).

The quiet alarm (2.5.7) fires only when a source returns nothing. Three of
the nine silent defects in the 2026-10-07 audit returned rows all the same:
paging stuck on its first page, a Workday board read to 40 because only the
first page states the total, and a 400-row cap on a board of 1,412. Each
run read ok. `source_counts` now keeps the board's own total beside what was
read; a source that falls under 30% of its recent median is named, and so is
one that says it stopped short."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

import rolescan.store as store_module
from rolescan.config import Config
from rolescan.pipeline import (
    Coverage,
    ScanResult,
    SourceReport,
    _check_coverage,
    run_scan,
)
from rolescan.store import _MIGRATIONS, Store

#: `_source_key` of `_report`'s source.
KEY = "workday:acme:Acme"


def _report(
    count: int, *, total: int | None = None, truncated: str = "", error: str = ""
) -> SourceReport:
    return SourceReport(
        kind="workday",
        slug="acme",
        label="Acme",
        count=count,
        total=total,
        truncated=truncated,
        error=error,
    )


async def _history(store: Store, *counts: int, days_ago: int = 1) -> None:
    """One earlier run per count, an hour apart, the newest `days_ago` days
    back."""
    newest = datetime.now(UTC) - timedelta(days=days_ago)
    await store.db.executemany(
        "INSERT INTO source_counts (source_key, ran, count) VALUES (?, ?, ?)",
        [
            (KEY, (newest - timedelta(hours=i)).isoformat(timespec="seconds"), n)
            for i, n in enumerate(counts)
        ],
    )
    await store.db.commit()


def _rows(path: Path) -> list[Any]:
    with closing(sqlite3.connect(path)) as conn:
        return conn.execute(
            "SELECT source_key, count, total FROM source_counts ORDER BY source_key"
        ).fetchall()


async def test_an_older_store_gains_total_and_its_rows_read_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "s.db"
    with monkeypatch.context() as m:
        m.setattr(store_module, "_MIGRATIONS", _MIGRATIONS[:7])
        async with Store(path) as store:
            await _history(store, 40)

    async with Store(path):
        pass

    assert _rows(path) == [(KEY, 40, 0)]
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 8


async def test_migration_8_skips_a_total_column_that_is_already_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store left half-migrated: the column added, the version not moved
    (the old runner ran the script and set the version as two statements)."""
    path = tmp_path / "s.db"
    with monkeypatch.context() as m:
        m.setattr(store_module, "_MIGRATIONS", _MIGRATIONS[:7])
        async with Store(path):
            pass
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(
            "ALTER TABLE source_counts ADD COLUMN total INTEGER NOT NULL DEFAULT 0"
        )
        conn.commit()

    async with Store(path) as store:
        await store.record_source_counts({KEY: (40, 2000)})

    assert _rows(path) == [(KEY, 40, 2000)]


async def test_counts_keep_the_boards_total_and_a_plain_count_still_works(
    tmp_path: Path,
) -> None:
    async with Store(tmp_path / "s.db") as store:
        await store.record_source_counts({"a": (500, 2000), "b": 7, "c": (5, None)})
    assert _rows(tmp_path / "s.db") == [("a", 500, 2000), ("b", 7, 0), ("c", 5, 0)]


async def test_recent_counts_are_the_non_zero_runs_in_the_window_newest_first(
    tmp_path: Path,
) -> None:
    async with Store(tmp_path / "s.db") as store:
        await _history(store, 30, 0, 20, 10)
        await _history(store, 99, days_ago=20)
        assert await store.source_counts_recent(KEY) == [30, 20, 10]
        assert await store.source_counts_recent(KEY, days=30) == [30, 20, 10, 99]
        assert await store.source_counts_recent("unknown") == []


@pytest.mark.parametrize(
    ("count", "history", "shrunk"),
    [
        (10, (400, 400, 400, 400, 400), [("Acme", 10, 400.0)]),  # a 97% drop
        (119, (400, 400, 400, 400, 400), [("Acme", 119, 400.0)]),
        (120, (400, 400, 400, 400, 400), []),  # exactly 30%
        (10, (400, 400), []),  # two earlier runs are not a trend
        (10, (400, 400, 400), [("Acme", 10, 400.0)]),  # three are
        (2, (9, 9, 9, 9, 9), []),  # a median under 10 is noise
        (2, (10, 10, 10), [("Acme", 2, 10.0)]),
        (3, (10, 10, 10), []),  # 30% of 10 exactly, not 3.0000000000000004
        (10, (400, 400, 0, 0, 0), []),  # zero runs are no baseline
    ],
)
async def test_a_collapse_in_rows_is_named(
    tmp_path: Path,
    count: int,
    history: tuple[int, ...],
    shrunk: list[tuple[str, int, float]],
) -> None:
    async with Store(tmp_path / "s.db") as store:
        await _history(store, *history)
        found = await _check_coverage([_report(count)], store)
    assert found == Coverage(shrunk=shrunk)


async def test_runs_older_than_14_days_are_no_baseline(tmp_path: Path) -> None:
    async with Store(tmp_path / "s.db") as store:
        await _history(store, 400, 400, 400, 400, 400, days_ago=15)
        assert await _check_coverage([_report(10)], store) == Coverage()


async def test_nothing_at_all_is_quiet_not_shrunk(tmp_path: Path) -> None:
    """The quiet alarm keeps zero; the shrink alarm starts above it."""
    async with Store(tmp_path / "s.db") as store:
        await _history(store, 400, 400, 400)
        found = await _check_coverage([_report(0)], store)
    assert found == Coverage(quiet=[("Acme", 400)])


async def test_a_run_is_never_its_own_baseline_and_a_dry_one_records_nothing(
    tmp_path: Path,
) -> None:
    async with Store(tmp_path / "s.db") as store:
        await _history(store, 400, 400, 400)
        for _ in range(3):
            dry = await _check_coverage([_report(10)], store, record=False)
            assert dry.shrunk == [("Acme", 10, 400.0)]
        assert await store.source_counts_recent(KEY) == [400, 400, 400]

        real = await _check_coverage([_report(10)], store)

        assert real.shrunk == [("Acme", 10, 400.0)]
        assert await store.source_counts_recent(KEY) == [10, 400, 400, 400]


async def test_a_source_that_says_it_stopped_short_is_listed_and_its_total_kept(
    tmp_path: Path,
) -> None:
    why = "stopped at 500 of 2000 postings (max_rows 500)"
    async with Store(tmp_path / "s.db") as store:
        found = await _check_coverage([_report(500, total=2000, truncated=why)], store)
    assert found == Coverage(truncated=[("Acme", 500, 2000, why)])
    assert _rows(tmp_path / "s.db") == [(KEY, 500, 2000)]


async def test_fewer_than_the_total_with_no_reason_is_not_called_cut_short(
    tmp_path: Path,
) -> None:
    """A posting skipped as unreadable, or a repeat dropped, leaves the count
    under the board's total on every run; a source names its own stops."""
    async with Store(tmp_path / "s.db") as store:
        assert await _check_coverage([_report(44, total=45)], store) == Coverage()


async def test_a_failed_source_is_in_no_list_and_records_nothing(
    tmp_path: Path,
) -> None:
    async with Store(tmp_path / "s.db") as store:
        await _history(store, 400, 400, 400)
        broke = _report(3, truncated="stopped early", error="HTTP 500")
        assert await _check_coverage([broke], store) == Coverage()
        assert await store.source_counts_recent(KEY) == [400, 400, 400]


def test_notes_come_from_sources_that_ran() -> None:
    result = ScanResult(
        reports=[
            SourceReport("workday", "acme", "Acme", 9, note="skipped 1 posting(s)"),
            SourceReport("lever", "b", "B", error="HTTP 500", note="never shown"),
            SourceReport("lever", "c", "C", 3),
        ]
    )
    assert result.notes == [("Acme", "skipped 1 posting(s)")]


@respx.mock
async def test_a_scan_carries_the_cut_short_and_shrunk_lists(tmp_path: Path) -> None:
    def page(request: httpx.Request) -> httpx.Response:
        offset = json.loads(request.content)["offset"]
        rows = [
            {"title": f"Analyst {i}", "externalPath": f"/job/X/J_{i}"}
            for i in range(offset, offset + 20)
        ]
        return httpx.Response(
            200, json={"total": 2000 if offset == 0 else 0, "jobPostings": rows}
        )

    respx.post("https://acme.wd3.myworkdayjobs.com/wday/cxs/acme/Careers/jobs").mock(
        side_effect=page
    )
    cfg = Config.model_validate(
        {
            "llm": {"enabled": False},
            "sources": [
                {
                    "kind": "workday",
                    "slug": "acme",
                    "label": "Acme",
                    "site": "Careers",
                    "host": "wd3",
                    "details": False,
                }
            ],
            "output": {"dir": str(tmp_path), "db_path": str(tmp_path / "s.db")},
        }
    )
    async with Store(tmp_path / "s.db") as store:
        await _history(store, 4000, 4000, 4000)

    result = await run_scan(cfg, dry_run=True)

    assert result.truncated_sources == [
        ("Acme", 500, 2000, "stopped at 500 of 2000 postings (max_rows 500)")
    ]
    assert result.shrunk_sources == [("Acme", 500, 4000.0)]
