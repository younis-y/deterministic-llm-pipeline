"""Persistence: what has been seen, and what the LLM already decided.

Two tables with different jobs:

  seen     keyed on Job.uid, so a role is reported once and never again.
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
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import TracebackType
from typing import ClassVar, Self, TypeVar, overload

import aiosqlite
from pydantic import BaseModel, ValidationError

from rolescan.models import FitVerdict, Job, ScoredJob

__all__ = ["Store"]

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
)


#: `ALTER TABLE <table> ADD COLUMN <column> ...` at the start of a statement.
_ADD_COLUMN = re.compile(
    r"\s*ALTER\s+TABLE\s+(?P<table>\w+)\s+ADD\s+COLUMN\s+(?P<column>\w+)",
    re.IGNORECASE,
)


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
        # WAL lets a long scan run while you read the digest from another shell.
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        try:
            await self._migrate()
        except BaseException:
            # A failed migration used to leave the connection open, and its
            # worker thread kept the process alive after the error.
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
        if self._db is not None:
            await self._db.commit()
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
        """
        for i in range(await self._user_version(), len(_MIGRATIONS)):
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

    async def filter_new(self, scored: list[ScoredJob]) -> list[ScoredJob]:
        """Partition in one query rather than N."""
        if not scored:
            return []
        uids = [s.job.uid for s in scored]
        # The only thing interpolated is a run of `?` placeholders, whose
        # length comes from len(uids) and nothing else. Every value is bound.
        # Checked because bandit's S608 flags this shape on sight and the
        # answer should be written down rather than rediscovered: there is no
        # SQL builder here and no posting text anywhere near the statement.
        placeholders = ",".join("?" * len(uids))
        cur = await self.db.execute(
            f"SELECT uid FROM seen WHERE uid IN ({placeholders})",
            uids,
        )
        known = {row[0] for row in await cur.fetchall()}
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

    async def prune(self, days: int = 180) -> int:
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat(
            timespec="seconds"
        )
        cur = await self.db.execute("DELETE FROM verdicts WHERE created < ?", (cutoff,))
        await self.db.commit()
        return cur.rowcount or 0
