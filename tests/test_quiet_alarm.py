"""2.5.7: the "went quiet" alarm forgot a dead source after five zero runs
(the sixth digest said nothing), and the Markdown digest dropped the alarm
when it was the only problem (`_failures` returned [] before reaching it;
the Markdown digest is what downstream tooling reads). Both found 2026-10-07."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from rolescan.digest import render_markdown
from rolescan.pipeline import ScanResult
from rolescan.store import Store


async def _insert(store: Store, key: str, days_ago: int, count: int) -> None:
    ran = (datetime.now(UTC) - timedelta(days=days_ago)).isoformat(timespec="seconds")
    await store.db.execute(
        "INSERT OR REPLACE INTO source_counts (source_key, ran, count) VALUES (?, ?, ?)",
        (key, ran, count),
    )
    await store.db.commit()


async def test_high_water_survives_many_zero_runs(tmp_path: Path) -> None:
    async with Store(tmp_path / "s.db") as store:
        await _insert(store, "greenhouse/acme", 10, 40)
        # One row per distinct second (the primary key is source_key + ran), so
        # each zero run needs its own day to count as a run.
        for days_ago in (6, 5, 4, 3, 2, 1):
            await _insert(store, "greenhouse/acme", days_ago, 0)
        await store.record_source_counts({"greenhouse/acme": 0})
        assert await store.source_high_water("greenhouse/acme") == 40


async def test_high_water_forgets_after_the_window(tmp_path: Path) -> None:
    async with Store(tmp_path / "s.db") as store:
        await _insert(store, "greenhouse/acme", 20, 40)
        assert await store.source_high_water("greenhouse/acme") == 0


def test_markdown_shows_the_quiet_alarm_when_nothing_else_failed() -> None:
    result = ScanResult(quiet_sources=[("Acme board", 40)])
    text = render_markdown(result)
    assert "went quiet" in text and "Acme board" in text
