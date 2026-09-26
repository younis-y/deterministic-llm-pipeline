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
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import TracebackType
from typing import ClassVar, Self

import aiosqlite
from pydantic import ValidationError

from rolescan.models import FitVerdict, Job, ScoredJob

__all__ = ["Store"]

log = logging.getLogger(__name__)

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
)


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
        await self._migrate()
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

    async def _migrate(self) -> None:
        cur = await self.db.execute("PRAGMA user_version")
        row = await cur.fetchone()
        version = int(row[0]) if row else 0
        for i, script in enumerate(_MIGRATIONS[version:], start=version):
            await self.db.executescript(script)
            await self.db.execute(f"PRAGMA user_version={i + 1}")
        await self.db.commit()

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

    async def record(self, scored: ScoredJob) -> None:
        now = datetime.now(UTC).isoformat(timespec="seconds")
        job = scored.job
        await self.db.execute(
            """
            INSERT INTO seen
                (uid, company, title, location, url, source, score, verdict,
                 first_seen, last_seen)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(uid) DO UPDATE SET
                last_seen=excluded.last_seen,
                score=excluded.score,
                verdict=excluded.verdict
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
            ),
        )

    async def record_all(self, items: list[ScoredJob]) -> None:
        for s in items:
            await self.record(s)
        await self.db.commit()

    async def count(self) -> int:
        cur = await self.db.execute("SELECT COUNT(*) FROM seen")
        row = await cur.fetchone()
        return int(row[0]) if row else 0

    # -- verdict cache ------------------------------------------------------

    async def get_verdict(
        self, content_hash: str, max_age_days: int
    ) -> FitVerdict | None:
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
            return FitVerdict.model_validate_json(row[0])
        except ValidationError:
            # A schema change invalidates old rows. Drop and re-score rather
            # than crashing on a cache the current code cannot read.
            log.debug("dropping unreadable cached verdict %s", content_hash)
            await self.db.execute(
                "DELETE FROM verdicts WHERE content_hash=?", (content_hash,)
            )
            return None

    async def put_verdict(self, content_hash: str, verdict: FitVerdict) -> None:
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

    async def prune(self, days: int = 180) -> int:
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat(
            timespec="seconds"
        )
        cur = await self.db.execute("DELETE FROM verdicts WHERE created < ?", (cutoff,))
        await self.db.commit()
        return cur.rowcount or 0
