"""Persistence: what has been seen, and what the LLM already decided.

Tables with different jobs:

  seen     keyed on Job.uid, so a role is reported once and never again.
  deferred keyed on Job.uid, counting the runs a posting was held back without
           being judged (no text yet), so a source that never sends text
           cannot defer a posting for ever.
  verdicts keyed on Job.content_hash, so re-running costs nothing for postings
           whose text has not changed. This is what makes it safe to run the
           scan several times a day.
  postings keyed on URL, holding the PARSED job plus the sitemap lastmod it was
           built from. The structured source re-fetches a detail page only when
           lastmod moves. ADNOC and ACWA Power both ignore If-Modified-Since and
           answer 200 with the full body (verified 2026-08-25), so lastmod is the
           only invalidation signal available and this table is what makes it
           usable.

Schema changes go through `_MIGRATIONS`; the file survives upgrades.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import TracebackType
from typing import ClassVar, Self, TypeVar, overload

import aiosqlite
from pydantic import BaseModel, ValidationError

from rolescan.models import FitVerdict, Job, ScoredJob

__all__ = ["PruneReport", "Store", "StoreTooNewError"]

log = logging.getLogger(__name__)

#: `get_verdict`/`put_verdict` hold whatever a caller's scoring mode decided
#: to cache - a `FitVerdict` in judge mode, a `PostingFacts` in facts mode
#: (see `rolescan.scoring.llm.cache_key`, which keeps the two out of each
#: other's rows). Bound to `BaseModel` rather than `FitVerdict` because the
#: row is opaque JSON to this module either way: it is validated back into
#: whatever type the caller asks for, never inspected here.
#:
#: No default on this TypeVar (PEP 696 needs `typing_extensions` on this
#: Python floor, and core takes no new runtime dependency for it) - the two
#: `@overload`s on `get_verdict` below give every no-`model=` call site
#: `FitVerdict` under mypy instead, stdlib-only.
_T = TypeVar("_T", bound=BaseModel)

_MIGRATIONS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS seen (
        uid         TEXT PRIMARY KEY,
        company     TEXT NOT NULL,
        title       TEXT NOT NULL,
        location    TEXT NOT NULL DEFAULT '',
        url         TEXT NOT NULL DEFAULT '',
        source      TEXT NOT NULL DEFAULT '',
        score       INTEGER NOT NULL DEFAULT 0,
        verdict     TEXT NOT NULL DEFAULT '',
        first_seen  TEXT NOT NULL,
        last_seen   TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS seen_company ON seen(company);
    CREATE TABLE IF NOT EXISTS verdicts (
        content_hash TEXT PRIMARY KEY,
        payload      TEXT NOT NULL,
        created      TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS postings (
        url      TEXT PRIMARY KEY,
        lastmod  TEXT NOT NULL,
        payload  TEXT NOT NULL,
        fetched  TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS applications (
        url     TEXT PRIMARY KEY,
        state   TEXT NOT NULL CHECK (state IN ('shortlist','applied','dismissed')),
        company TEXT NOT NULL DEFAULT '',
        title   TEXT NOT NULL DEFAULT '',
        updated TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS applications_state ON applications(state);
    """,
    """
    CREATE TABLE IF NOT EXISTS source_counts (
        source_key TEXT NOT NULL,
        ran        TEXT NOT NULL,
        count      INTEGER NOT NULL,
        PRIMARY KEY (source_key, ran)
    );
    CREATE INDEX IF NOT EXISTS source_counts_key ON source_counts(source_key, ran);
    """,
    """
    ALTER TABLE seen ADD COLUMN reason TEXT NOT NULL DEFAULT '';
    """,
    """
    CREATE TABLE IF NOT EXISTS deferred (
        uid        TEXT PRIMARY KEY,
        reason     TEXT NOT NULL,
        times      INTEGER NOT NULL DEFAULT 1,
        first_seen TEXT NOT NULL,
        last_seen  TEXT NOT NULL
    );
    """,
    # 2.5.8: `unsee` and `mark` look a posting up by url, which scanned the
    # whole of `seen`; and `source_counts_key` duplicated the primary key's
    # own index exactly, so it was pure write cost.
    """
    CREATE INDEX IF NOT EXISTS seen_url ON seen(url);
    DROP INDEX IF EXISTS source_counts_key;
    """,
)


#: `ALTER TABLE <table> ADD COLUMN <column> ...` at the start of a statement.
_ADD_COLUMN = re.compile(
    r"\s*ALTER\s+TABLE\s+(?P<table>\w+)\s+ADD\s+COLUMN\s+(?P<column>\w+)",
    re.IGNORECASE,
)


def _chunks(items: list[str], size: int = 500) -> Iterable[list[str]]:
    """`items` in slices small enough for SQLite's bound-variable limit."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _statements(script: str) -> list[str]:
    """Split a migration script into its complete statements, in order.

    Uses SQLite's own `complete_statement`, so a `;` inside a string or a
    trigger body does not split a statement in two."""
    out: list[str] = []
    start = 0
    for m in re.finditer(";", script):
        chunk = script[start : m.end()]
        if sqlite3.complete_statement(chunk):
            out.append(chunk.strip())
            start = m.end()
    tail = script[start:].strip()
    if tail:
        out.append(tail)
    return out


#: `source_counts` rows older than this are trimmed by `prune_all`. The
#: quiet-source alarm reads 14 days (`source_high_water`), so this only ever
#: removes rows nothing reads.
_SOURCE_COUNTS_DAYS = 90

#: `prune_all` rebuilds the file (VACUUM) once more than this share of its
#: pages are free: a DELETE alone frees pages inside the file but never
#: shrinks it (measured 2026-10-07: 21.35 MB before and after a trim, 9.64 MB
#: after VACUUM, which took 0.03 s).
_VACUUM_FREE_SHARE = 0.2


@dataclass(frozen=True, slots=True)
class PruneReport:
    """Rows `Store.prune_all` removed per table, and whether it vacuumed."""

    verdicts: int = 0
    postings: int = 0
    deferred: int = 0
    source_counts: int = 0
    vacuumed: bool = False


class StoreTooNewError(RuntimeError):
    """The file was written by a newer rolescan than the one opening it.

    Opening it anyway ran old code against a schema it does not know: the
    first write that disagreed failed half way through a scan (2.5.8).
    """


class Store:
    """Async SQLite store. Use as an async context manager."""

    #: The only states an application row may hold.
    STATES: ClassVar[tuple[str, ...]] = ("shortlist", "applied", "dismissed")

    def __init__(self, path: Path) -> None:
        self.path = path
        self._db: aiosqlite.Connection | None = None

    async def __aenter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.path)
        try:
            # Before any PRAGMA that writes: `journal_mode=WAL` rewrites the
            # header of a rollback-journal file, so a newer file refused after
            # it would be refused and changed (2.5.8).
            await self._refuse_a_newer_file()
            # WAL lets a long scan run while you read the digest from another
            # shell.
            await self._db.execute("PRAGMA journal_mode=WAL")
            await self._db.execute("PRAGMA foreign_keys=ON")
            await self._migrate()
        except BaseException:
            # `__aexit__` does not run when `__aenter__` raises, and
            # aiosqlite's worker is a non-daemon thread: a connection left
            # open kept the process alive after the traceback. 2.5.7 closed it
            # for a failed migration only; a file that is not a database fails
            # on the first PRAGMA, before the migration (2.5.8).
            await self._db.close()
            self._db = None
            raise
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._db is None:
            return
        try:
            # Commit on a clean exit only. Committing on the way out of an
            # exception left a PREFIX of `record_all`'s rows behind when it
            # was interrupted: marked seen, and in no digest (2.5.8).
            if exc_type is None:
                await self._db.commit()
            else:
                await self._db.rollback()
        finally:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            msg = "Store must be used as an async context manager"
            raise RuntimeError(msg)
        return self._db

    async def _user_version(self) -> int:
        cur = await self.db.execute("PRAGMA user_version")
        row = await cur.fetchone()
        return int(row[0]) if row else 0

    async def _refuse_a_newer_file(self) -> None:
        """Raise `StoreTooNewError` for a file from a newer rolescan.

        Older code used to open such a file silently and fail at the first
        write the newer schema disagreed with, half way through a scan
        (2.5.8). Called first thing after the connection opens, with nothing
        written: the refused file is left byte for byte as it was.
        """
        version = await self._user_version()
        if version > len(_MIGRATIONS):
            msg = (
                f"{self.path} is at schema version {version}, and this rolescan "
                f"knows versions up to {len(_MIGRATIONS)}: it was written by a "
                "newer rolescan. Upgrade rolescan, or restore a copy of the "
                "store taken before the upgrade."
            )
            raise StoreTooNewError(msg)

    async def _migrate(self) -> None:
        """Bring the file up to `len(_MIGRATIONS)`, one step at a time.

        Each step runs inside its own `BEGIN IMMEDIATE` ... `COMMIT`, together
        with its `user_version` bump, so a step is either wholly applied and
        counted or not applied at all. The version is read again once the
        write lock is held: a second process that opened the same old file at
        the same moment waits on the lock, then sees the winner's version and
        skips the step.

        Migrations 1 to 4 are `CREATE ... IF NOT EXISTS` and could simply be
        re-run. Migrations after 4 may not be re-runnable statements (5 is
        `ALTER TABLE ... ADD COLUMN`, which SQLite cannot make `IF NOT
        EXISTS`), so the runner guards them: it skips an `ADD COLUMN` whose
        column is already there (a store left half-migrated by a crash under
        the old runner, which executed the script and set the version as two
        separate statements).

        Statements are executed one at a time, not through `executescript`,
        because `executescript` commits first and so cannot join a
        transaction.

        A file at a higher version than this code knows is refused before
        this runs (`_refuse_a_newer_file`).
        """
        version = await self._user_version()
        for i in range(version, len(_MIGRATIONS)):
            script = _MIGRATIONS[i]
            await self.db.execute("BEGIN IMMEDIATE")
            try:
                if await self._user_version() > i:
                    # Lost the race: another opener applied this step while
                    # this one waited for the lock.
                    await self.db.rollback()
                    continue
                for statement in _statements(script):
                    if not await self._column_exists(statement):
                        await self.db.execute(statement)
                await self.db.execute(f"PRAGMA user_version={i + 1}")
                await self.db.commit()
            except BaseException:
                await self.db.rollback()
                raise

    async def _column_exists(self, statement: str) -> bool:
        """True when `statement` is an `ALTER TABLE ... ADD COLUMN` whose
        column is already on the table. Anything else is False."""
        m = _ADD_COLUMN.match(statement)
        if m is None:
            return False
        # `table` is `\w+` from this module's own migration text, not input.
        info = await self.db.execute_fetchall(f"PRAGMA table_info({m['table']})")
        return any(row[1].lower() == m["column"].lower() for row in info)

    # -- seen ---------------------------------------------------------------

    async def is_new(self, job: Job) -> bool:
        cur = await self.db.execute("SELECT 1 FROM seen WHERE uid=?", (job.uid,))
        return await cur.fetchone() is None

    async def filter_new(
        self, scored: list[ScoredJob], *, touch: bool = False
    ) -> list[ScoredJob]:
        """The postings not yet in `seen`, looked up 500 uids per query.

        One placeholder per posting used to go into a single `IN (...)`, and
        SQLite refuses more than its bound-variable limit: 999 before 3.32,
        32,766 after, so a scan of a real store's ~2,850 postings failed outright
        on an older SQLite and a 10x board list would fail on any (2.5.8).

        `touch` refreshes `last_seen` on the postings found already seen, so
        "when did a scan last see this role" can be answered: before 2.5.8
        the only write to `last_seen` was an upsert branch `filter_new` made
        unreachable, and 4,616 of 5,048 live rows had `first_seen ==
        last_seen`. Only postings that reach this call are touched: one the
        scan drops earlier (stale, or merged into a near-duplicate) keeps its
        older `last_seen` even though a board still lists it. Off by default,
        so the call stays a pure read for any caller that does not ask; a scan
        passes `touch=not dry_run`.

        The only thing interpolated into the SQL is a run of `?` placeholders,
        whose length comes from the chunk and nothing else. Every value is
        bound. Written down because bandit's S608 flags the shape on sight.
        """
        if not scored:
            return []
        uids = list(dict.fromkeys(s.job.uid for s in scored))
        known: set[str] = set()
        for chunk in _chunks(uids):
            placeholders = ",".join("?" * len(chunk))
            rows = await self.db.execute_fetchall(
                f"SELECT uid FROM seen WHERE uid IN ({placeholders})", chunk
            )
            known.update(str(row[0]) for row in rows)
        if touch and known:
            now = datetime.now(UTC).isoformat(timespec="seconds")
            for chunk in _chunks(sorted(known)):
                placeholders = ",".join("?" * len(chunk))
                await self.db.execute(
                    f"UPDATE seen SET last_seen = ? WHERE uid IN ({placeholders})",
                    [now, *chunk],
                )
            await self.db.commit()
        return [s for s in scored if s.job.uid not in known]

    async def record(self, scored: ScoredJob, *, reason: str = "") -> None:
        now = datetime.now(UTC).isoformat(timespec="seconds")
        job = scored.job
        await self.db.execute(
            """
            INSERT INTO seen
                (uid, company, title, location, url, source, score, verdict,
                 first_seen, last_seen, reason)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(uid) DO UPDATE SET
                last_seen=excluded.last_seen,
                score=excluded.score,
                verdict=excluded.verdict,
                reason=excluded.reason
            """,
            (
                job.uid,
                job.company,
                job.title,
                job.location,
                job.url,
                job.source,
                scored.score,
                scored.verdict.value,
                now,
                now,
                reason,
            ),
        )

    async def record_all(
        self, items: Iterable[ScoredJob | tuple[ScoredJob, str]]
    ) -> None:
        """Record each posting with the reason it was assessed (2.5.7:
        `prefilter`, `judged`, a rule name, `blocked`), so `rolescan unsee`
        and a reader of the table can tell a keyword reject from a judged
        skip. A bare `ScoredJob` records with no reason, for callers that
        predate the column."""
        for item in items:
            scored, reason = item if isinstance(item, tuple) else (item, "")
            await self.record(scored, reason=reason)
        await self.db.commit()

    # -- deferred -----------------------------------------------------------

    async def bump_deferred(self, uids: list[str], reason: str) -> dict[str, int]:
        """Count one more run for each posting that was held back unjudged.

        A posting with no description is deferred, not recorded, so the next
        run can try again; a source that never sends the text would defer it
        for ever. This counts the sightings so the caller can stop holding
        it after N. Upserts: a new uid starts at 1, a known one is
        incremented. Returns the new count per uid."""
        wanted = list(dict.fromkeys(uids))
        if not wanted:
            return {}
        now = datetime.now(UTC).isoformat(timespec="seconds")
        await self.db.executemany(
            """
            INSERT INTO deferred (uid, reason, times, first_seen, last_seen)
            VALUES (?, ?, 1, ?, ?)
            ON CONFLICT(uid) DO UPDATE SET
                times=times + 1,
                reason=excluded.reason,
                last_seen=excluded.last_seen
            """,
            [(uid, reason, now, now) for uid in wanted],
        )
        await self.db.commit()
        counts: dict[str, int] = {}
        for chunk in _chunks(wanted):
            placeholders = ",".join("?" * len(chunk))
            rows = await self.db.execute_fetchall(
                f"SELECT uid, times FROM deferred WHERE uid IN ({placeholders})",
                chunk,
            )
            counts.update({str(r[0]): int(r[1]) for r in rows})
        return counts

    async def forget_deferred(self, uids: Iterable[str]) -> int:
        """Drop the deferral counts for postings that no longer need one (they
        were recorded as seen). Returns the rows removed."""
        removed = 0
        for chunk in _chunks(list(dict.fromkeys(uids))):
            placeholders = ",".join("?" * len(chunk))
            cur = await self.db.execute(
                f"DELETE FROM deferred WHERE uid IN ({placeholders})", chunk
            )
            removed += cur.rowcount
        await self.db.commit()
        return removed

    async def unsee(self, key: str) -> int:
        """Forget a posting by url or uid so it can be reported again.

        The one recovery path for a role hidden by a wrong rule or term; until
        2.5.7 it was a hand-written DELETE. Returns the rows removed."""
        cur = await self.db.execute(
            "DELETE FROM seen WHERE uid = ? OR url = ?", (key, key)
        )
        await self.db.commit()
        return cur.rowcount

    async def count(self) -> int:
        cur = await self.db.execute("SELECT COUNT(*) FROM seen")
        row = await cur.fetchone()
        return int(row[0]) if row else 0

    # -- verdict cache ------------------------------------------------------

    @overload
    async def get_verdict(
        self, content_hash: str, max_age_days: int
    ) -> FitVerdict | None: ...

    @overload
    async def get_verdict(
        self, content_hash: str, max_age_days: int, model: type[_T]
    ) -> _T | None: ...

    async def get_verdict(
        self,
        content_hash: str,
        max_age_days: int,
        model: type[BaseModel] = FitVerdict,
    ) -> BaseModel | None:
        """The cached payload for this key, validated as `model`, or None.

        `model` is how a caller reads back its OWN cached shape - a
        `FitVerdict` in judge mode, a `PostingFacts` in facts mode. A model
        that forbids extra keys writes a payload an unrelated schema refuses,
        and refusing it here silently deletes the row and re-scores - a whole
        cache lost to a schema that was never wrong. A caller passes back
        whatever type it stored.

        Two `@overload`s above, rather than a generic default, give every
        existing call with no `model=` argument `FitVerdict | None` under
        mypy - the implementation signature itself stays non-generic
        (`type[BaseModel]`), which is what lets the default value type-check
        with no `# type: ignore` needed.
        """
        cur = await self.db.execute(
            "SELECT payload, created FROM verdicts WHERE content_hash=?",
            (content_hash,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        if max_age_days > 0:
            try:
                created = datetime.fromisoformat(row[1])
            except ValueError:
                return None
            if datetime.now(UTC) - created > timedelta(days=max_age_days):
                return None
        try:
            return model.model_validate_json(row[0])
        except ValidationError:
            # A schema change invalidates old rows. Drop and re-score rather
            # than crashing on a cache the current code cannot read.
            log.debug("dropping unreadable cached verdict %s", content_hash)
            await self.db.execute(
                "DELETE FROM verdicts WHERE content_hash=?", (content_hash,)
            )
            return None

    async def put_verdict(self, content_hash: str, verdict: BaseModel) -> None:
        """Cache any pydantic model under this key - a `FitVerdict` in judge
        mode, a `PostingFacts` in facts mode. See `get_verdict`."""
        await self.db.execute(
            """
            INSERT INTO verdicts (content_hash, payload, created) VALUES (?,?,?)
            ON CONFLICT(content_hash) DO UPDATE SET
                payload=excluded.payload, created=excluded.created
            """,
            (
                content_hash,
                json.dumps(verdict.model_dump(mode="json")),
                datetime.now(UTC).isoformat(timespec="seconds"),
            ),
        )
        await self.db.commit()

    # -- posting cache ------------------------------------------------------

    async def get_posting(self, url: str) -> tuple[str, Job] | None:
        """The stored (lastmod, job) for a URL, or None to fetch it."""
        cur = await self.db.execute(
            "SELECT lastmod, payload FROM postings WHERE url=?", (url,)
        )
        row = await cur.fetchone()
        if row is None:
            return None
        try:
            return str(row[0]), Job.model_validate_json(row[1])
        except ValidationError:
            # A model change must degrade to a refetch, not crash the scan.
            log.debug("dropping unreadable cached posting %s", url)
            await self.db.execute("DELETE FROM postings WHERE url=?", (url,))
            return None

    async def put_posting(self, url: str, lastmod: str, job: Job) -> None:
        await self.db.execute(
            """
            INSERT INTO postings (url, lastmod, payload, fetched) VALUES (?,?,?,?)
            ON CONFLICT(url) DO UPDATE SET
                lastmod=excluded.lastmod,
                payload=excluded.payload,
                fetched=excluded.fetched
            """,
            (
                url,
                lastmod,
                job.model_dump_json(),
                datetime.now(UTC).isoformat(timespec="seconds"),
            ),
        )
        await self.db.commit()

    # -- application state ---------------------------------------------------

    async def mark(
        self, url: str, state: str, company: str = "", title: str = ""
    ) -> None:
        """Record what the user did with a posting.

        The newest `state` always wins. `company` and `title` do not:
        a blank one leaves whatever is already stored in place, so
        re-marking a posting from a context that has no metadata (the
        digest prints a bare url) cannot erase the labels an earlier
        mark captured.
        """
        if state not in self.STATES:
            msg = f"state must be one of {self.STATES}, got {state!r}"
            raise ValueError(msg)
        await self.db.execute(
            """
            INSERT INTO applications (url, state, company, title, updated)
            VALUES (?, ?, ?, ?, datetime('now'))
            ON CONFLICT(url) DO UPDATE SET
                state = excluded.state,
                company = CASE WHEN excluded.company != '' THEN excluded.company
                               ELSE applications.company END,
                title = CASE WHEN excluded.title != '' THEN excluded.title
                             ELSE applications.title END,
                updated = excluded.updated
            """,
            (url, state, company, title),
        )
        await self.db.commit()

    async def application_state(self, url: str) -> str | None:
        async with self.db.execute(
            "SELECT state FROM applications WHERE url = ?", (url,)
        ) as cur:
            row = await cur.fetchone()
        return str(row[0]) if row else None

    async def shortlist(self) -> list[tuple[str, str, str]]:
        async with self.db.execute(
            "SELECT url, company, title FROM applications "
            "WHERE state = 'shortlist' ORDER BY updated DESC"
        ) as cur:
            rows = await cur.fetchall()
        return [(str(r[0]), str(r[1]), str(r[2])) for r in rows]

    async def dismissed_urls(self) -> set[str]:
        async with self.db.execute(
            "SELECT url FROM applications WHERE state = 'dismissed'"
        ) as cur:
            rows = await cur.fetchall()
        return {str(r[0]) for r in rows}

    async def record_source_counts(self, counts: dict[str, int]) -> None:
        """Remember what each source returned, so a silent zero is detectable.

        Every serious defect in this project has been a component that stopped
        working while the run still exited 0 - a hardcoded first page, a
        concurrency default, an empty query list, options nested one level too
        deep. None of them raised. All of them were obvious the moment you
        compared a source against what it returned yesterday, which is the one
        thing nothing was keeping.
        """
        now = datetime.now(UTC).isoformat(timespec="seconds")
        await self.db.executemany(
            "INSERT OR REPLACE INTO source_counts (source_key, ran, count) "
            "VALUES (?, ?, ?)",
            [(key, now, n) for key, n in counts.items()],
        )
        await self.db.commit()

    async def source_high_water(self, key: str, *, days: int = 14) -> int:
        """The most this source has returned in the last `days` days.

        A high-water mark rather than a mean or a median: the question is not
        "is today typical" but "has this source ever worked", and one good run
        is enough to prove it can. That makes a single fluke unable to raise
        the bar permanently, while a source that has only ever returned zero
        never trips the alarm.

        A window in days, not runs (2.5.7): five runs of zero used to push the
        good run out of the window, so a dead source dropped out of the digest
        on its sixth run, which is the opposite of an alarm.
        """
        since = (datetime.now(UTC) - timedelta(days=days)).isoformat(timespec="seconds")
        rows = await self.db.execute_fetchall(
            "SELECT count FROM source_counts WHERE source_key = ? AND ran >= ?",
            (key, since),
        )
        return max((int(r[0]) for r in rows), default=0)

    async def prune_all(
        self, *, postings_days: int, deferred_days: int, verdicts_days: int
    ) -> PruneReport:
        """Trim every cache to its retention, then VACUUM if it is worth it.

        A `*_days` of 0 keeps that table whole. `seen` and `applications` are
        never touched here: `seen` is the "listed once" record, and deleting a
        row re-lists the role. A cached posting is kept past its age while it
        has an `applications` row, or while it is an apply/consider role whose
        `seen.last_seen` is within `postings_days`: `rolescan mark` reads such
        a page back to fill in the company and title, and the structured
        source's cache reads a page back instead of fetching it again.
        """
        verdicts = await self.prune(verdicts_days) if verdicts_days else 0
        counts: dict[str, int] = {}
        for table, days, sql in (
            (
                "postings",
                postings_days,
                """
                DELETE FROM postings
                WHERE fetched < :cutoff
                  AND url NOT IN (SELECT url FROM applications)
                  AND url NOT IN (SELECT url FROM seen
                                  WHERE verdict IN ('apply', 'consider')
                                    AND last_seen >= :cutoff)
                """,
            ),
            (
                "deferred",
                deferred_days,
                "DELETE FROM deferred WHERE last_seen < :cutoff",
            ),
            (
                "source_counts",
                _SOURCE_COUNTS_DAYS,
                "DELETE FROM source_counts WHERE ran < :cutoff",
            ),
        ):
            if not days:
                counts[table] = 0
                continue
            cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat(
                timespec="seconds"
            )
            cur = await self.db.execute(sql, {"cutoff": cutoff})
            counts[table] = cur.rowcount or 0
        await self.db.commit()
        return PruneReport(
            verdicts=verdicts,
            postings=counts["postings"],
            deferred=counts["deferred"],
            source_counts=counts["source_counts"],
            vacuumed=await self._vacuum_if_worth_it(),
        )

    async def _vacuum_if_worth_it(self) -> bool:
        """VACUUM when more than `_VACUUM_FREE_SHARE` of the pages are free."""
        [(free,)] = await self.db.execute_fetchall("PRAGMA freelist_count")
        [(pages,)] = await self.db.execute_fetchall("PRAGMA page_count")
        if not pages or free <= pages * _VACUUM_FREE_SHARE:
            return False
        await self.db.execute("VACUUM")
        await self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return True

    async def prune(self, days: int = 180) -> int:
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat(
            timespec="seconds"
        )
        cur = await self.db.execute("DELETE FROM verdicts WHERE created < ?", (cutoff,))
        await self.db.commit()
        return cur.rowcount or 0
