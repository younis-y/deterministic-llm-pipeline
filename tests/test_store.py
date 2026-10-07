from __future__ import annotations

import asyncio
from pathlib import Path

import aiosqlite
import pytest

import rolescan.store as store_module
from rolescan.models import (
    Confidence,
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


async def _insert_old_seen_row(store: Store, job: Job) -> None:
    """Write a `seen` row the way a store from before the `reason` column did.

    `Store.record` names `reason`, so a test that pins the schema below
    migration 5 cannot use it; it inserts the columns that schema has."""
    await store.db.execute(
        "INSERT INTO seen (uid, company, title, url, first_seen, last_seen) "
        "VALUES (?,?,?,?,?,?)",
        (
            job.uid,
            job.company,
            job.title,
            job.url,
            "2026-09-01T00:00:00+00:00",
            "2026-09-01T00:00:00+00:00",
        ),
    )


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
            await _insert_old_seen_row(store, energy_job)
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


async def test_record_keeps_a_reason_and_unsee_removes_the_row(
    tmp_path: Path, energy_job: Job
) -> None:
    db = tmp_path / "seen.db"
    async with Store(db) as store:
        await store.record_all(
            [(ScoredJob(job=energy_job, keyword_score=4), "prefilter")]
        )
        rows = await store.db.execute_fetchall("SELECT reason FROM seen")
        assert [r[0] for r in rows] == ["prefilter"]
        assert await store.unsee(energy_job.url) == 1
        assert await store.is_new(energy_job)
        assert await store.unsee(energy_job.url) == 0


async def test_record_all_still_accepts_bare_scored_jobs(
    tmp_path: Path, energy_job: Job
) -> None:
    async with Store(tmp_path / "seen.db") as store:
        await store.record_all([ScoredJob(job=energy_job, keyword_score=4)])
        rows = await store.db.execute_fetchall("SELECT reason FROM seen")
        assert [r[0] for r in rows] == [""]


def test_migration_count_is_six() -> None:
    assert len(_MIGRATIONS) == 6


async def test_unsee_matches_on_uid_as_well_as_url(
    tmp_path: Path, energy_job: Job
) -> None:
    """`unsee` takes either key: the digest prints the url, but a row can also
    be addressed by its uid."""
    async with Store(tmp_path / "seen.db") as store:
        await store.record(ScoredJob(job=energy_job, keyword_score=4), reason="judged")
        assert await store.unsee(energy_job.uid) == 1
        assert await store.is_new(energy_job)


async def test_reason_column_migrates_onto_a_version_four_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, energy_job: Job
) -> None:
    """A user's existing `seen.db` sits at `user_version` 4, with rows that
    predate the `reason` column. Opening it with the new code must add the
    column in place, keep those rows, and give them '' (the old behaviour, a
    row with no recorded reason), not fail or drop them."""
    path = tmp_path / "s.db"

    with monkeypatch.context() as m:
        # Pin the module to the first four migrations so this open genuinely
        # stops at the pre-`reason` schema.
        m.setattr(store_module, "_MIGRATIONS", store_module._MIGRATIONS[:4])
        async with Store(path) as store:
            await _insert_old_seen_row(store, energy_job)
            cur = await store.db.execute("PRAGMA user_version")
            row = await cur.fetchone()
            assert row is not None
            assert int(row[0]) == 4, "fixture must start below the new migration"
            columns = [
                r[1] for r in await store.db.execute_fetchall("PRAGMA table_info(seen)")
            ]
            assert "reason" not in columns

    # Reopen with the real, unpatched Store: the actual upgrade path.
    async with Store(path) as store:
        cur = await store.db.execute("PRAGMA user_version")
        row = await cur.fetchone()
        assert row is not None
        assert int(row[0]) == len(_MIGRATIONS)
        columns = [
            r[1] for r in await store.db.execute_fetchall("PRAGMA table_info(seen)")
        ]
        assert "reason" in columns
        reasons = await store.db.execute_fetchall("SELECT reason FROM seen")
        assert [r[0] for r in reasons] == [""]
        assert not await store.is_new(energy_job), "the old row must survive"
        await store.record(ScoredJob(job=energy_job, keyword_score=4), reason="judged")
        reasons = await store.db.execute_fetchall("SELECT reason FROM seen")
        assert [r[0] for r in reasons] == ["judged"]


# --- migrations are atomic with their version bump (2.5.7) -------------------
# Migration 5 is `ALTER TABLE ... ADD COLUMN`, which SQLite cannot make
# `IF NOT EXISTS`. The runner used to execute the script and then set
# `user_version` as two separate statements, so two processes opening a
# version-4 store at once both ran the ALTER (the loser died with "duplicate
# column name: reason"), and a crash between the ALTER and the pragma left a
# store that could never be opened again.


async def _make_version_four_store(
    path: Path, monkeypatch: pytest.MonkeyPatch, job: Job, *, with_reason: bool
) -> None:
    """Create `path` at `user_version` 4. `with_reason` adds the migration-5
    column by hand, as a crash between the ALTER and the pragma would have."""
    with monkeypatch.context() as m:
        m.setattr(store_module, "_MIGRATIONS", store_module._MIGRATIONS[:4])
        async with Store(path) as store:
            await _insert_old_seen_row(store, job)
            if with_reason:
                await store.db.execute(
                    "ALTER TABLE seen ADD COLUMN reason TEXT NOT NULL DEFAULT ''"
                )
            await store.db.commit()


async def _user_version_and_columns(path: Path) -> tuple[int, list[str]]:
    async with Store(path) as store:
        cur = await store.db.execute("PRAGMA user_version")
        row = await cur.fetchone()
        assert row is not None
        columns = [
            r[1] for r in await store.db.execute_fetchall("PRAGMA table_info(seen)")
        ]
        return int(row[0]), columns


async def test_a_store_that_already_has_the_reason_column_still_opens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, energy_job: Job
) -> None:
    """The crash case: the ALTER committed but `user_version` is still 4.
    Opening must not die on "duplicate column name"; it finishes the upgrade."""
    path = tmp_path / "s.db"
    await _make_version_four_store(path, monkeypatch, energy_job, with_reason=True)

    version, columns = await _user_version_and_columns(path)

    assert version == len(_MIGRATIONS)
    assert columns.count("reason") == 1


async def test_two_stores_opening_a_version_four_file_at_once_both_succeed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, energy_job: Job
) -> None:
    """Two processes opening the same old store: one wins the lock and runs
    migration 5, the other must see version 5 under the lock and skip it."""
    path = tmp_path / "s.db"
    await _make_version_four_store(path, monkeypatch, energy_job, with_reason=False)

    async def open_and_close() -> int:
        async with Store(path) as store:
            return await store.count()

    counts = await asyncio.gather(*(open_and_close() for _ in range(4)))

    assert counts == [1, 1, 1, 1]
    version, columns = await _user_version_and_columns(path)
    assert version == len(_MIGRATIONS)
    assert columns.count("reason") == 1


async def test_a_store_that_loses_the_migration_race_skips_the_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, energy_job: Job
) -> None:
    """Deterministic version of the race: a rival holds the write lock, the
    store reads version 4 and waits, then the rival finishes migration 5 and
    commits. The store must re-read the version under the lock and skip the
    step, not run the ALTER a second time."""
    path = tmp_path / "s.db"
    await _make_version_four_store(path, monkeypatch, energy_job, with_reason=False)

    async with aiosqlite.connect(path) as rival:
        await rival.execute("BEGIN IMMEDIATE")
        opening = asyncio.ensure_future(_user_version_and_columns(path))
        await asyncio.sleep(0.3)
        assert not opening.done(), "the store must be waiting on the rival's lock"
        await rival.execute(
            "ALTER TABLE seen ADD COLUMN reason TEXT NOT NULL DEFAULT ''"
        )
        await rival.execute("PRAGMA user_version=5")
        await rival.commit()
        version, columns = await opening

    assert version == len(_MIGRATIONS)
    assert columns.count("reason") == 1


async def test_a_failing_migration_rolls_back_and_leaves_the_version_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A script that dies half way must not leave its first statement behind
    or bump `user_version`: the store reopens at the same version, ready to be
    migrated again once the fault is fixed."""
    path = tmp_path / "s.db"
    broken = (
        *_MIGRATIONS,
        "CREATE TABLE half_done (x INTEGER); SELECT * FROM no_such_table;",
    )
    with monkeypatch.context() as m:
        m.setattr(store_module, "_MIGRATIONS", broken)
        with pytest.raises(aiosqlite.OperationalError):
            async with Store(path):
                pass

    async with Store(path) as store:
        cur = await store.db.execute("PRAGMA user_version")
        row = await cur.fetchone()
        assert row is not None and int(row[0]) == len(_MIGRATIONS)
        tables = await store.db.execute_fetchall(
            "SELECT name FROM sqlite_master WHERE name='half_done'"
        )
        assert list(tables) == []


# --- deferred: how many runs a posting has been held back without text -------


async def test_bump_deferred_counts_each_sighting(tmp_path: Path) -> None:
    async with Store(tmp_path / "s.db") as store:
        assert await store.bump_deferred(["a", "b"], "thin") == {"a": 1, "b": 1}
        assert await store.bump_deferred(["a"], "thin") == {"a": 2}
        assert await store.bump_deferred(["a", "b", "c"], "thin") == {
            "a": 3,
            "b": 2,
            "c": 1,
        }
        assert await store.bump_deferred([], "thin") == {}


async def test_deferred_counts_survive_reopening_the_file(tmp_path: Path) -> None:
    path = tmp_path / "s.db"
    async with Store(path) as store:
        await store.bump_deferred(["a"], "thin")
    async with Store(path) as store:
        assert await store.bump_deferred(["a"], "thin") == {"a": 2}


async def test_forget_deferred_restarts_the_count(tmp_path: Path) -> None:
    async with Store(tmp_path / "s.db") as store:
        await store.bump_deferred(["a", "b"], "thin")
        assert await store.forget_deferred(["a", "never-deferred"]) == 1
        assert await store.bump_deferred(["a", "b"], "thin") == {"a": 1, "b": 2}
        assert await store.forget_deferred([]) == 0


async def test_deferred_handles_more_uids_than_one_query_can_bind(
    tmp_path: Path,
) -> None:
    uids = [f"u{i}" for i in range(1200)]
    async with Store(tmp_path / "s.db") as store:
        assert set((await store.bump_deferred(uids, "thin")).values()) == {1}
        assert await store.forget_deferred(uids) == 1200
