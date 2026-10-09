"""An incremental `structured` source across scans (2.6.0).

The source reads only what the sitemap says changed since the last scan that
read the whole board. Where that "last scan" is kept, and when it moves, is
what these tests pin: the store holds one mark per source; the pipeline hands
it to the source before the read and keeps the source's new one only once a
real scan's results are recorded, with nothing left waiting for a later scan.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from rolescan import store as store_module
from rolescan.config import Config
from rolescan.pipeline import (
    ScanResult,
    SourceReport,
    _check_coverage,
    _marks_to_keep,
    _read_shape,
    _source_key,
    record_scan,
    run_scan,
)
from rolescan.sources.structured import Structured
from rolescan.store import Store

HOST = "https://jobs.example.test"
SITEMAP = f"{HOST}/sitemap.xml"

_CONFIG = """
profile:
  keywords: {energy: 6, python: 4}
  min_keyword_score: 4
  min_report_score: 0
llm:
  enabled: false
output:
  dir: digests
  db_path: seen.db
  max_roles: {max_roles}
sources:
  - kind: structured
    slug: acme
    label: Acme Energy
    sitemap: https://jobs.example.test/sitemap.xml
    url_pattern: /job/
    delay: 0
    incremental: true
"""


def _days_ago(days: int) -> str:
    return (datetime.now(UTC).date() - timedelta(days=days)).isoformat()


def _url(n: int) -> str:
    return f"{HOST}/job/{n}/energy-analyst"


def _page(n: int) -> str:
    return (
        '<html><head><script type="application/ld+json">'
        '{"@context":"http://schema.org","@type":"JobPosting",'
        f'"title":"Energy Analyst {n}","datePosted":"2026-10-01",'
        f'"identifier":{{"@type":"PropertyValue","value":"{n}"}},'
        '"hiringOrganization":{"@type":"Organization","name":"Acme Energy"},'
        '"jobLocation":{"@type":"Place","address":{"@type":"PostalAddress",'
        '"addressLocality":"London","addressCountry":"GB"}},'
        '"description":"Energy markets and Python."}'
        "</script></head><body>page</body></html>"
    )


def _sitemap(*entries: tuple[int, str]) -> str:
    urls = "".join(
        f"<url><loc>{_url(n)}</loc><lastmod>{lm}</lastmod></url>" for n, lm in entries
    )
    return f'<?xml version="1.0"?><urlset>{urls}</urlset>'


def _serve(*entries: tuple[int, str]) -> dict[int, respx.Route]:
    respx.get(SITEMAP).mock(return_value=httpx.Response(200, text=_sitemap(*entries)))
    return {
        n: respx.get(_url(n)).mock(return_value=httpx.Response(200, text=_page(n)))
        for n, _ in entries
    }


def _config(
    tmp_path: Path, max_roles: int = 40, max_pages: int | None = None
) -> Config:
    path = tmp_path / "config.yaml"
    text = _CONFIG.replace("{max_roles}", str(max_roles))
    if max_pages is not None:
        text += f"    max_pages: {max_pages}\n"
    path.write_text(text)
    return Config.load(path)


async def _scan(cfg: Config, **kw: Any) -> ScanResult:
    result = await run_scan(cfg, **kw)
    await record_scan(cfg, result)
    return result


async def _mark(cfg: Config) -> str:
    """The mark the store holds for the config's one source."""
    report = SourceReport(
        kind="structured",
        slug="acme",
        label="Acme Energy",
        read_shape=_read_shape(cfg.enabled_sources[0]),
    )
    async with Store(cfg.resolve(cfg.output.db_path)) as store:
        return await store.source_mark(_source_key(report))


# --- the store -------------------------------------------------------------


async def test_a_store_keeps_one_mark_per_source(tmp_path: Path) -> None:
    async with Store(tmp_path / "s.db") as store:
        assert await store.source_mark("a") == ""

        await store.set_source_mark("a", "2026-10-01T00:00:00+00:00|abc")
        await store.set_source_mark("b", "other")
        await store.set_source_mark("a", "2026-10-02T00:00:00+00:00|abc")

        assert await store.source_mark("a") == "2026-10-02T00:00:00+00:00|abc"
        assert await store.source_mark("b") == "other"


async def test_the_marks_table_is_a_migration_and_opening_twice_is_harmless(
    tmp_path: Path,
) -> None:
    path = tmp_path / "s.db"
    async with Store(path) as store:
        await store.set_source_mark("a", "x")
    async with Store(path) as store:
        assert await store.source_mark("a") == "x"
    with closing(sqlite3.connect(path)) as conn:
        tables = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert "source_marks" in tables


async def test_migration_10_adds_the_marks_table_to_a_store_at_version_9(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "s.db"
    with monkeypatch.context() as m:
        m.setattr(store_module, "_MIGRATIONS", store_module._MIGRATIONS[:9])
        async with Store(path) as store:
            await store.record_source_counts({"k": 3})
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 9
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
    assert "source_marks" not in names

    async with Store(path) as store:
        await store.set_source_mark("a", "x")
        assert await store.source_mark("a") == "x"
        assert await store.source_counts_recent("k") == [3], "nothing else moved"
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 10


# --- the pipeline ----------------------------------------------------------


@respx.mock
async def test_the_second_scan_reads_only_what_changed(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    first_pages = _serve((1, _days_ago(40)), (2, _days_ago(30)))
    first = await _scan(cfg)
    assert first.reports[0].ok and first.reports[0].incremental
    assert {n: r.call_count for n, r in first_pages.items()} == {1: 1, 2: 1}
    assert await _mark(cfg), "a whole real scan keeps a mark"

    respx.clear()
    second_pages = _serve((1, _days_ago(40)), (2, _days_ago(30)), (3, _days_ago(0)))
    second = await _scan(cfg)

    assert [r.call_count for r in second_pages.values()] == [0, 0, 1]
    assert second.fetched == 1
    assert second.reports[0].total == 3, "the board's count, not what was new"


@respx.mock
async def test_a_dry_run_keeps_no_mark(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    _serve((1, _days_ago(40)))

    await _scan(cfg, dry_run=True)

    assert await _mark(cfg) == ""


@respx.mock
async def test_a_dry_run_leaves_the_next_real_scan_the_whole_board(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    _serve((1, _days_ago(40)))
    await _scan(cfg, dry_run=True)

    result = await _scan(cfg)

    assert result.fetched == 1
    assert [s.job.raw_id for s, _ in result.to_record] == ["1"]


@respx.mock
async def test_a_scan_that_left_postings_deferred_keeps_the_old_mark(
    tmp_path: Path,
) -> None:
    """A deferred posting is read again by the next scan only if its source
    hands it over again, which an incremental read does not."""
    cfg = _config(tmp_path, max_roles=1)
    _serve((1, _days_ago(1)), (2, _days_ago(1)), (3, _days_ago(1)))

    result = await _scan(cfg)

    assert result.deferred, "max_roles leaves the others for the next scan"
    assert await _mark(cfg) == ""


@respx.mock
async def test_a_scan_whose_model_failed_keeps_the_old_mark(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    _serve((1, _days_ago(1)))
    result = await run_scan(cfg)
    result.llm_errors = 2

    await record_scan(cfg, result)

    assert await _mark(cfg) == ""


@respx.mock
async def test_a_read_cut_at_max_pages_keeps_the_old_mark_until_a_whole_read(
    tmp_path: Path,
) -> None:
    """A read that stopped at `max_pages` left postings in the window unread.
    Moving the mark would put them behind it for good, so it stays, and the
    next scans repeat the cut read until `max_pages` covers the window."""
    cfg = _config(tmp_path, max_pages=2)
    entry = cfg.enabled_sources[0]
    fingerprint = Structured(entry, None)._fingerprint()  # type: ignore[arg-type]
    old = f"{_days_ago(5)}T06:00:00+00:00|{fingerprint}"
    report = SourceReport("structured", "acme", "Acme Energy")
    report.read_shape = _read_shape(entry)
    async with Store(cfg.resolve(cfg.output.db_path)) as store:
        await store.set_source_mark(_source_key(report), old)
    window = (
        (1, _days_ago(1)),
        (2, _days_ago(2)),
        (3, _days_ago(3)),
        (4, _days_ago(4)),
    )
    recorded: list[str] = []

    for _ in range(2):  # a cut read repeats, and keeps the mark each time
        respx.clear()
        _serve(*window)
        result = await _scan(cfg)
        assert result.reports[0].truncated
        assert result.fetched == 2
        assert await _mark(cfg) == old
        recorded += [s.job.raw_id for s, _ in result.to_record]

    # Raised so that it covers the window: a whole read, and the mark moves.
    respx.clear()
    _serve(*window)
    cfg = _config(tmp_path, max_pages=4)
    result = await _scan(cfg)
    recorded += [s.job.raw_id for s, _ in result.to_record]

    assert not result.reports[0].truncated
    assert result.fetched == 4
    assert sorted(recorded) == ["1", "2", "3", "4"], "nothing is skipped for good"
    assert await _mark(cfg) not in ("", old)


@respx.mock
async def test_the_source_is_given_the_mark_the_last_scan_kept(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    _serve((1, _days_ago(40)))
    await _scan(cfg)
    stored = await _mark(cfg)
    assert stored

    respx.clear()
    _serve((1, _days_ago(40)))
    result = await run_scan(cfg)

    assert result.fetched == 0, "nothing in the sitemap is newer than the mark"
    assert result.reports[0].next_mark


def test_a_report_that_says_it_was_cut_short_keeps_no_new_mark() -> None:
    """Whatever the source, a read cut short offered only part of what it
    holds, so the pipeline does not move the mark on its word."""
    whole = SourceReport("structured", "acme", "Acme Energy", incremental=True)
    whole.next_mark = "2026-10-02T06:00:00+00:00|abc"
    cut = SourceReport("structured", "beta", "Beta Energy", incremental=True)
    cut.next_mark = "2026-10-02T06:00:00+00:00|abc"
    cut.truncated = "read the newest 2 of 4 urls in scope (max_pages 2)"

    kept = _marks_to_keep(ScanResult(reports=[whole, cut]))

    assert list(kept) == [_source_key(whole)]


# --- the alarms ------------------------------------------------------------


async def test_an_incremental_source_cannot_go_quiet_or_shrink(
    tmp_path: Path,
) -> None:
    """Nothing new since the last scan is the normal quiet day, and a handful
    of changes after a first read of thousands is not a collapse."""
    async with Store(tmp_path / "s.db") as store:
        key = "structured:acme:Acme Energy"
        for hours, count in enumerate((400, 380, 420), start=1):
            ran = (datetime.now(UTC) - timedelta(hours=hours)).isoformat()
            await store.db.execute(
                "INSERT INTO source_counts (source_key, ran, count) VALUES (?, ?, ?)",
                (key, ran, count),
            )
        await store.db.commit()

        quiet = SourceReport("structured", "acme", "Acme Energy", count=0)
        few = SourceReport("structured", "acme", "Acme Energy", count=20)
        for report in (quiet, few):
            report.incremental = True
        found = await _check_coverage([quiet], store, record=False)
        assert found.quiet == []
        found = await _check_coverage([few], store, record=False)
        assert found.shrunk == []

        quiet.incremental = few.incremental = False
        assert (await _check_coverage([quiet], store, record=False)).quiet
        assert (await _check_coverage([few], store, record=False)).shrunk
