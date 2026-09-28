from __future__ import annotations

from pathlib import Path

import pytest

import rolescan.store as store_module
from rolescan.models import (
    Confidence,
    CVVariant,
    FitVerdict,
    Job,
    ScoredJob,
    Verdict,
)
from rolescan.store import _MIGRATIONS, Store

VERDICT = FitVerdict(
    fit_score=88,
    verdict=Verdict.APPLY,
    confidence=Confidence.HIGH,
    reason="Strong overlap with the day-ahead forecasting project.",
    cv_variant=CVVariant.ENERGY,
    tailoring=["Lead with the LP battery dispatch result."],
)


async def test_roundtrip_and_dedup(tmp_path: Path, energy_job: Job) -> None:
    db = tmp_path / "seen.db"
    scored = ScoredJob(job=energy_job, keyword_score=40)
    async with Store(db) as store:
        assert await store.is_new(energy_job)
        await store.record_all([scored])
        assert not await store.is_new(energy_job)
        assert await store.count() == 1

    async with Store(db) as store:
        assert not await store.is_new(energy_job), "state must survive reopening"


async def test_filter_new_partitions_in_one_pass(
    tmp_path: Path, energy_job: Job, gated_job: Job
) -> None:
    a, b = ScoredJob(job=energy_job), ScoredJob(job=gated_job)
    async with Store(tmp_path / "s.db") as store:
        await store.record_all([a])
        fresh = await store.filter_new([a, b])
        assert [s.job.uid for s in fresh] == [b.job.uid]


async def test_record_is_idempotent(tmp_path: Path, energy_job: Job) -> None:
    async with Store(tmp_path / "s.db") as store:
        await store.record_all([ScoredJob(job=energy_job, keyword_score=10)])
        await store.record_all([ScoredJob(job=energy_job, keyword_score=90)])
        assert await store.count() == 1


async def test_verdict_cache_roundtrip(tmp_path: Path, energy_job: Job) -> None:
    async with Store(tmp_path / "s.db") as store:
        assert await store.get_verdict(energy_job.content_hash, 30) is None
        await store.put_verdict(energy_job.content_hash, VERDICT)
        got = await store.get_verdict(energy_job.content_hash, 30)
        assert got is not None
        assert got.fit_score == 88
        assert got.cv_variant is CVVariant.ENERGY
        assert got.tailoring == VERDICT.tailoring


async def test_edited_posting_misses_the_cache(tmp_path: Path, energy_job: Job) -> None:
    edited = energy_job.model_copy(update={"description": "rewritten posting"})
    async with Store(tmp_path / "s.db") as store:
        await store.put_verdict(energy_job.content_hash, VERDICT)
        assert await store.get_verdict(edited.content_hash, 30) is None


async def test_corrupt_cache_row_is_dropped_not_raised(
    tmp_path: Path, energy_job: Job
) -> None:
    """A schema change must not brick the tool."""
    async with Store(tmp_path / "s.db") as store:
        await store.db.execute(
            "INSERT INTO verdicts VALUES (?,?,?)",
            (
                energy_job.content_hash,
                '{"nonsense": true}',
                "2026-08-01T00:00:00+00:00",
            ),
        )
        await store.db.commit()
        assert await store.get_verdict(energy_job.content_hash, 30) is None
        assert await store.get_verdict(energy_job.content_hash, 30) is None


async def test_prune_drops_old_verdicts(tmp_path: Path, energy_job: Job) -> None:
    async with Store(tmp_path / "s.db") as store:
        await store.db.execute(
            "INSERT INTO verdicts VALUES (?,?,?)",
            (
                energy_job.content_hash,
                VERDICT.model_dump_json(),
                "2020-01-01T00:00:00+00:00",
            ),
        )
        await store.db.commit()
        assert await store.prune(days=30) == 1
        assert await store.get_verdict(energy_job.content_hash, 30) is None


async def test_migrations_are_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    for _ in range(3):
        async with Store(db) as store:
            assert await store.count() == 0


# --- posting cache: sitemap lastmod is the only invalidation signal --------
# Verified 2026-08-25 that ADNOC and ACWA Power ignore If-Modified-Since and
# return 200 with the full body, so conditional GETs cannot be used and the
# sitemap's lastmod is what gates a refetch.


def _job(url: str, title: str = "Analyst") -> Job:
    return Job(source="structured", company="ADNOC", title=title, url=url)


async def test_posting_cache_miss_returns_none(tmp_path: Path) -> None:
    async with Store(tmp_path / "s.db") as store:
        assert await store.get_posting("https://x/job/1") is None


async def test_posting_cache_round_trips_lastmod_and_job(tmp_path: Path) -> None:
    job = _job("https://x/job/1", "Senior Analyst")
    async with Store(tmp_path / "s.db") as store:
        await store.put_posting("https://x/job/1", "2026-08-25", job)
        hit = await store.get_posting("https://x/job/1")
    assert hit is not None
    lastmod, cached = hit
    assert lastmod == "2026-08-25"
    assert cached.title == "Senior Analyst"
    assert cached.uid == job.uid, "the cached job must keep its identity"


async def test_posting_cache_overwrites_on_a_new_lastmod(tmp_path: Path) -> None:
    async with Store(tmp_path / "s.db") as store:
        await store.put_posting(
            "https://x/job/1", "2026-08-01", _job("https://x/job/1")
        )
        await store.put_posting(
            "https://x/job/1", "2026-08-25", _job("https://x/job/1", "Retitled")
        )
        hit = await store.get_posting("https://x/job/1")
    assert hit is not None
    assert hit[0] == "2026-08-25"
    assert hit[1].title == "Retitled"


async def test_unreadable_cached_posting_is_dropped_not_raised(
    tmp_path: Path,
) -> None:
    """A model change must degrade to a refetch, never crash the scan."""
    async with Store(tmp_path / "s.db") as store:
        await store.put_posting(
            "https://x/job/1", "2026-08-25", _job("https://x/job/1")
        )
        await store.db.execute(
            "UPDATE postings SET payload='{not json' WHERE url=?", ("https://x/job/1",)
        )
        assert await store.get_posting("https://x/job/1") is None


async def test_posting_cache_survives_reopening_the_file(tmp_path: Path) -> None:
    path = tmp_path / "s.db"
    async with Store(path) as store:
        await store.put_posting(
            "https://x/job/1", "2026-08-25", _job("https://x/job/1")
        )
    async with Store(path) as store:
        assert await store.get_posting("https://x/job/1") is not None


# --- application state -------------------------------------------------


async def test_mark_and_read_application_state(tmp_path: Path) -> None:
    async with Store(tmp_path / "s.sqlite3") as store:
        await store.mark("https://x/1", "applied", company="Glencore", title="Analytics")
        assert await store.application_state("https://x/1") == "applied"
        assert await store.application_state("https://x/2") is None


async def test_mark_is_idempotent_and_updates_state(tmp_path: Path) -> None:
    async with Store(tmp_path / "s.sqlite3") as store:
        await store.mark("https://x/1", "shortlist", company="C", title="T")
        await store.mark("https://x/1", "applied", company="C", title="T")
        assert await store.application_state("https://x/1") == "applied"
        assert len(await store.shortlist()) == 0


async def test_shortlist_returns_only_shortlisted(tmp_path: Path) -> None:
    async with Store(tmp_path / "s.sqlite3") as store:
        await store.mark("https://x/1", "shortlist", company="A", title="One")
        await store.mark("https://x/2", "applied", company="B", title="Two")
        rows = await store.shortlist()
        assert rows == [("https://x/1", "A", "One")]


async def test_dismissed_urls(tmp_path: Path) -> None:
    async with Store(tmp_path / "s.sqlite3") as store:
        await store.mark("https://x/9", "dismissed", company="C", title="T")
        assert await store.dismissed_urls() == {"https://x/9"}


async def test_invalid_state_is_rejected(tmp_path: Path) -> None:
    async with Store(tmp_path / "s.sqlite3") as store:
        with pytest.raises(ValueError):
            await store.mark("https://x/1", "maybe")


async def test_mark_does_not_wipe_company_or_title_with_blanks(
    tmp_path: Path,
) -> None:
    """A later mark() with default empty company/title must not clobber
    values an earlier call already stored."""
    async with Store(tmp_path / "s.sqlite3") as store:
        await store.mark("https://x/1", "shortlist", company="C", title="T")
        await store.mark("https://x/1", "applied")
        rows = list(
            await store.db.execute_fetchall(
                "SELECT company, title FROM applications WHERE url = ?",
                ("https://x/1",),
            )
        )
        assert rows[0] == ("C", "T")


async def test_applications_table_is_created_on_a_fresh_store(
    tmp_path: Path,
) -> None:
    """A brand-new store (current code, all migrations applied together)
    ends up with a working applications table. This does NOT exercise the
    upgrade path of an old, already-populated store — see
    test_applications_table_migrates_onto_a_pre_existing_store for that."""
    path = tmp_path / "s.db"
    job = _job("https://x/job/1")
    scored = ScoredJob(job=job)
    async with Store(path) as store:
        await store.record_all([scored])

    async with Store(path) as store:
        assert await store.count() == 1
        assert not await store.is_new(job)
        await store.mark("https://x/2", "shortlist", company="A", title="One")
        assert await store.shortlist() == [("https://x/2", "A", "One")]


async def test_applications_table_migrates_onto_a_pre_existing_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, energy_job: Job
) -> None:
    """The real scenario: a store created before the `applications` table
    existed, sitting at PRAGMA user_version=2, meeting the new migration
    for the first time on open. Data written under the old schema (the
    `seen` row that drives deduplication) must survive the upgrade."""
    path = tmp_path / "s.db"

    with monkeypatch.context() as m:
        # Pin the module to only the first two migrations, so this open
        # genuinely stops at user_version=2 — the pre-applications state a
        # real user's rolescan.sqlite3 would be in.
        m.setattr(store_module, "_MIGRATIONS", store_module._MIGRATIONS[:2])
        async with Store(path) as store:
            await store.record_all([ScoredJob(job=energy_job)])
            cur = await store.db.execute("PRAGMA user_version")
            row = await cur.fetchone()
            assert row is not None
            assert int(row[0]) == 2, "fixture must start below the new migration"

    # Reopen with the real, unpatched Store: this is the actual upgrade path.
    async with Store(path) as store:
        cur = await store.db.execute("PRAGMA user_version")
        row = await cur.fetchone()
        assert row is not None
        assert int(row[0]) == len(_MIGRATIONS), (
            "user_version must end at the migration count, not a literal - "
            "hardcoding it means every new migration breaks this test"
        )
        assert int(row[0]) > 2, "the new migration must have run"

        tables = list(
            await store.db.execute_fetchall(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='applications'"
            )
        )
        assert tables, "applications table must exist after the upgrade"

        assert await store.count() == 1
        assert not await store.is_new(
            energy_job
        ), "seen data written under the old schema must survive the upgrade"

        await store.mark("https://x/2", "shortlist", company="A", title="One")
        assert await store.shortlist() == [("https://x/2", "A", "One")]
