"""Three more aggregators: Reed (the UK), Jooble (60-odd countries) and
Workable's cross-employer search.

Reed and Jooble are built from the providers' public API documentation and were
NOT run against the live services when they were written, for want of a key.
They have fixtures in the documented shapes and nothing more: a first real scan
is what would show a field the documentation did not mention. Workable's search
was checked with one request on 2026-10-09 (see `WorkableSearch`).

Like Adzuna, Reed and Jooble need a key, taken from the entry or the
environment, and with none the source is skipped with one line, not failed.
Workable's search needs none. None has a company slug, so `slug` is only a label
for the entry.

    sources:
      - kind: reed
        slug: uk                    # a label: Reed covers the UK only
        api_key: "..."              # or REED_API_KEY in the environment
        queries: [energy analyst, data scientist]
        where: London               # optional: a place to search near
        distance: 15                # optional: miles from `where`
        graduate: true              # optional: graduate roles only
        direct_employer_only: true  # optional: not recruitment agencies
        max_pages: 5                # pages read per query (default 5)
        results_per_page: 100       # rows a page asks for (1 to 100)

      - kind: jooble
        slug: gb                    # a label
        api_key: "..."              # or JOOBLE_API_KEY in the environment
        queries: [energy analyst]
        where: London               # optional: a place
        radius: 25                  # optional: km from `where`
        salary: 40000               # optional: the least pay wanted
        max_pages: 5

      - kind: workable_search
        slug: gb                    # a label
        queries: [energy analyst]
        where: London               # optional: a place
        max_pages: 5
        delay: 2                    # seconds between requests (default 2)

Reed's search returns a shortened description, which is what the keyword and
model scoring read. Reed has no "newer than" parameter, so age is left to
`profile.max_age_days`. Jooble's key is part of the request URL, which its API
requires: the source never puts it in an error, a note or a log line of its
own, and a filter on httpx's and the fetcher's loggers replaces it in the
request URLs they log (`--verbose` shows them).

A query is read page by page until a page comes back short or empty, the API's
own count is reached, or `max_pages` pages are read. A read cut at `max_pages`
says so (`Source.truncated`), with the API's count as `Source.total` when
there is one query: several queries overlap, so none is claimed.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import os
import re
from datetime import date
from typing import Any, ClassVar

from rolescan.http import FetchError, hide_in_logs
from rolescan.models import Job
from rolescan.sources.base import Source, SourceSkipped, register, strip_html

__all__ = ["Jooble", "Reed", "WorkableSearch"]

log = logging.getLogger(__name__)

_DEFAULT_MAX_PAGES = 5
#: Jooble's endpoint as an error or a log line may show it: the key is a path
#: segment of the real URL.
_JOOBLE_SHOWN = "https://jooble.org/api/<key>"
_UK_DATE = re.compile(r"^(\d{2})/(\d{2})/(\d{4})$")


class _Aggregator(Source):
    """The paging, the key and the failure handling the three share.

    A subclass names its key variable (none, for a service that needs no key),
    says how to ask for one page of one query (`_page`) and how to read a row
    (`_row`)."""

    #: Where the key is read when the entry has none; "" for no key at all.
    env_var: ClassVar[str] = ""
    #: True when a page is asked for by the token the last one returned, and the
    #: read ends when none comes back, not by counting rows.
    cursor_paged: ClassVar[bool] = False
    #: The service's name, as it reads in a message.
    service: ClassVar[str] = ""
    #: The most rows a page may ask for; None when the page size is the
    #: service's own and not ours to set.
    max_per_page: ClassVar[int | None] = None
    #: Seconds between requests for a service that is read one request at a
    #: time, its queries one after another; None for one that reads its queries
    #: together. The entry's `delay` option replaces it.
    default_delay: ClassVar[float | None] = None

    # -- options ----------------------------------------------------------------

    @property
    def api_key(self) -> str:
        if not self.env_var:
            return ""
        return str(
            self.entry.options.get("api_key") or os.environ.get(self.env_var, "")
        )

    @property
    def queries(self) -> list[str]:
        raw = self.entry.options.get("queries") or []
        if isinstance(raw, str):  # one search written as a plain value
            raw = [raw]
        return [str(q) for q in raw]

    def _whole_number(
        self, name: str, default: int | None, high: int | None = None
    ) -> int | None:
        """An option that must be a whole number of at least 1 (and at most
        `high`), or an error that names it; a bool is not a number here. None
        when it is absent and has no default."""
        raw = self.entry.options.get(name, default)
        if raw is None:
            return None
        bound = f"from 1 to {high}" if high is not None else "of at least 1"
        if (
            isinstance(raw, bool)
            or not isinstance(raw, int)
            or raw < 1
            or (high is not None and raw > high)
        ):
            msg = (
                f"{self.name} option {name} must be a whole number {bound}, not {raw!r}"
            )
            raise ValueError(msg)
        return raw

    def _flag(self, name: str) -> bool:
        raw = self.entry.options.get(name, False)
        if not isinstance(raw, bool):
            msg = f"{self.name} option {name} must be true or false, not {raw!r}"
            raise ValueError(msg)
        return raw

    @property
    def max_pages(self) -> int:
        """Pages read per query before the read is cut and says so."""
        return self._whole_number("max_pages", _DEFAULT_MAX_PAGES) or 0

    @property
    def delay(self) -> float | None:
        """Seconds to wait between requests, or None when queries run together.
        0 turns the wait off. An unset option is the default."""
        if self.default_delay is None:
            return None
        raw = self.entry.options.get("delay")
        if raw is None:
            return self.default_delay
        if (
            isinstance(raw, bool)
            or not isinstance(raw, int | float)
            or not math.isfinite(raw)
            or raw < 0
        ):
            msg = (
                f"{self.name} option delay must be a number of seconds, "
                f"0 or more, not {raw!r}"
            )
            raise ValueError(msg)
        return float(raw)

    async def _pause(self) -> None:
        """Wait out `delay` before a request that follows another."""
        if self.delay:
            await asyncio.sleep(self.delay)

    @property
    def results_per_page(self) -> int | None:
        """Rows a page asks for, or None where the service sets it."""
        if self.max_per_page is None:
            return None
        return self._whole_number(
            "results_per_page", self.max_per_page, self.max_per_page
        )

    def _check_options(self) -> None:
        """Read every option once before the first request, so a bad one fails
        the source with its own message and not half way through a read."""
        _ = self.max_pages, self.results_per_page, self.delay

    # -- what a subclass says -----------------------------------------------------

    async def _page(
        self, query: str, page: int, per_page: int | None, cursor: str
    ) -> tuple[list[Any], int | None, str]:
        """(the rows of page `page` of `query`, the service's count for it, the
        token for the page after it or ""). `cursor` is the token the last page
        returned, "" for the first."""
        raise NotImplementedError

    def _row(self, row: dict[str, Any]) -> Job:
        """A row as a Job; a ValueError means it cannot be one."""
        raise NotImplementedError

    # -- the read -----------------------------------------------------------------

    async def fetch(self) -> list[Job]:
        # These raise rather than returning [] so that `discover` reports
        # SKIPPED, and a bad config reads as one, not as a quiet source.
        if self.env_var and not self.api_key:
            msg = f"no key (set {self.env_var}, or api_key on the entry)"
            raise SourceSkipped(msg)
        if not self.queries:
            msg = f"{self.name} needs at least one entry in `queries`"
            raise ValueError(msg)
        self._check_options()
        cap, per_page = self.max_pages, self.results_per_page
        self.total, self.truncated, self.note = None, "", ""
        skipped: list[int] = [0]

        pages = await self._run_queries(cap, per_page, skipped)
        out: list[Job] = []
        totals: list[int | None] = []
        cut: list[str] = []
        for q, page in zip(self.queries, pages, strict=False):
            if isinstance(page, BaseException):
                log.warning("%s %s/%s: %s", self.name, self.slug, q, page)
                continue
            jobs, total, why = page
            out.extend(jobs)
            totals.append(total)
            if why:
                cut.append(why)
        # One query: the API's own count. Several overlap, so none is claimed.
        self.total = totals[0] if len(self.queries) == 1 and totals else None
        self.truncated = "; ".join(cut)
        if skipped[0]:
            self.note = (
                f"{skipped[0]} {self.service} rows had no title or link and "
                "were skipped"
            )
        failures = [p for p in pages if isinstance(p, BaseException)]
        if failures and len(failures) == len(pages):
            first = failures[0]
            url, reason = (
                (first.url, first.detail)
                if isinstance(first, FetchError)
                else ("", str(first))
            )
            msg = f"every {self.service} query failed; first: {reason}"
            raise FetchError(url, msg) from None
        if failures:
            log.warning(
                "%s %s: %d of %d queries failed",
                self.name,
                self.slug,
                len(failures),
                len(pages),
            )
        return out

    async def _run_queries(
        self, cap: int, per_page: int | None, skipped: list[int]
    ) -> list[Any]:
        """What each query read, in order: its result, or the exception it
        raised. Together unless the service is paced (`default_delay`); then
        one at a time, a pause apart, because a service that has not said how
        much traffic it takes is not asked for several searches at once."""
        if self.delay is None:
            return list(
                await asyncio.gather(
                    *(self._search(q, cap, per_page, skipped) for q in self.queries),
                    return_exceptions=True,
                )
            )
        done: list[Any] = []
        for i, q in enumerate(self.queries):
            if i:
                await self._pause()
            try:
                done.append(await self._search(q, cap, per_page, skipped))
            except Exception as e:
                done.append(e)  # gather's return_exceptions, one at a time
        return done

    async def _search(
        self, query: str, cap: int, per_page: int | None, skipped: list[int]
    ) -> tuple[list[Job], int | None, str]:
        """(the jobs one query read, the service's count for it, why the read
        stopped short or "").

        The read ends on a short or empty page (the last one), on reaching the
        stated count, or at `max_pages`; only the last of these is a cut, and
        says so. A page after the first that fails keeps what was read and
        says so too: the first page failing fails the query."""
        out: list[Job] = []
        total: int | None = None
        read = 0
        cursor = ""
        asked: set[str] = set()  # the tokens sent so far
        firsts: set[str] = set()  # the first row of each page read
        for page in range(1, cap + 1):
            if page > 1:
                await self._pause()
            if cursor:
                asked.add(cursor)
            try:
                rows, count, cursor = await self._page(query, page, per_page, cursor)
            except FetchError as e:
                if page == 1:
                    raise
                log.warning(
                    "%s %s/%s page %d: %s", self.name, self.slug, query, page, e
                )
                known = f" of {total}" if total is not None else ""
                return (
                    out,
                    total,
                    f'query "{query}": page {page} failed ({e.detail}), '
                    f"{read}{known} read",
                )
            if total is None and isinstance(count, int) and not isinstance(count, bool):
                total = max(count, 0)
            # A paging parameter the service does not honour answers the first
            # page again; the repeat is dropped, not listed twice.
            if self.cursor_paged and _begins_again(rows, firsts):
                return out, total, _stuck(query, read, "a page repeated rows")
            read += len(rows)
            for row in rows:
                try:
                    out.append(self._row(row))
                except (ValueError, AttributeError, TypeError):
                    skipped[0] += 1
            short = per_page is not None and len(rows) < per_page
            last_token = self.cursor_paged and not cursor
            if not rows or short or last_token or (total is not None and read >= total):
                return out, total, ""
            if cursor in asked:
                return out, total, _stuck(query, read, "the same page token came back")
        stated = f"{read} of {total}" if total is not None else f"{read}"
        none = "" if total is not None else ", the service stated no count"
        return (
            out,
            total,
            f'query "{query}": stopped at {stated} results{none} (max_pages {cap})',
        )


def _begins_again(rows: list[Any], firsts: set[str]) -> bool:
    """Whether a page opens with a row an earlier page opened with; if not, its
    first row is remembered."""
    if not rows:
        return False
    first = json.dumps(rows[0], sort_keys=True, default=str)
    if first in firsts:
        return True
    firsts.add(first)
    return False


def _stuck(query: str, read: int, what: str) -> str:
    """Why a token-paged read stopped before `max_pages` with a next page on
    offer: the service is not moving on, so reading further would only repeat."""
    return (
        f'query "{query}": {what}, so paging stopped after {read} results '
        "(the paging parameter may be wrong)"
    )


def _uk_date(raw: object) -> Any:
    """Reed's day-first `dd/mm/yyyy` as a date; anything else untouched, for
    `Job` to read as ISO or drop."""
    m = _UK_DATE.match(str(raw or "").strip())
    if not m:
        return raw
    try:
        return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    except ValueError:
        return None


@register
class Reed(_Aggregator):
    """https://www.reed.co.uk/api/1.0/search, the Reed.co.uk jobseeker API.

    HTTP Basic with the key as the user name and no password. Paged with
    `resultsToTake` (at most 100) and `resultsToSkip`."""

    name = "reed"
    slug_hint = (
        "slug is a label (Reed covers the UK only), e.g. uk. "
        "Key from reed.co.uk/developers"
    )
    env_var = "REED_API_KEY"
    service = "Reed"
    max_per_page = 100

    def _check_options(self) -> None:
        super()._check_options()
        _ = (
            self._whole_number("distance", None),
            self._flag("graduate"),
            self._flag("direct_employer_only"),
        )

    async def _page(
        self,
        query: str,
        page: int,
        per_page: int | None,
        cursor: str,  # noqa: ARG002 - paged by offset
    ) -> tuple[list[Any], int | None, str]:
        take = per_page or 100
        params: dict[str, Any] = {
            "keywords": query,
            "resultsToTake": take,
            "resultsToSkip": (page - 1) * take,
        }
        if where := self.entry.options.get("where"):
            params["locationName"] = str(where)
        if (distance := self._whole_number("distance", None)) is not None:
            params["distanceFromLocation"] = distance
        if self._flag("graduate"):
            params["graduate"] = "true"
        if self._flag("direct_employer_only"):
            params["postedByDirectEmployer"] = "true"
        token = base64.b64encode(f"{self.api_key}:".encode()).decode()
        url = "https://www.reed.co.uk/api/1.0/search"
        data = await self.fetcher.fetch_json(
            url, params=params, headers={"Authorization": f"Basic {token}"}
        )
        if not isinstance(data, dict):
            raise FetchError(url, "unexpected response: not a JSON object")
        return list(data.get("results") or []), data.get("totalResults"), ""

    def _row(self, row: dict[str, Any]) -> Job:
        title = str(row.get("jobTitle") or "")
        location = str(row.get("locationName") or "")
        return Job(
            source=f"{self.name}:{self.slug}",
            company=str(row.get("employerName") or "") or "unknown",
            title=title,
            location=location,
            url=str(row.get("jobUrl") or ""),
            description=strip_html(row.get("jobDescription", "")),
            posted=_uk_date(row.get("date")),
            remote="remote" in f"{title} {location}".casefold(),
            raw_id=str(row.get("jobId") or ""),
        )


@register
class Jooble(_Aggregator):
    """https://jooble.org/api/<key>, the Jooble search API.

    A POST with a JSON body, paged with `page`. The service sets the page
    size, so a read ends on an empty page or on the stated count."""

    name = "jooble"
    slug_hint = (
        "slug is a label, e.g. gb; set `where` to a place. "
        "Key from jooble.org/api/about"
    )
    env_var = "JOOBLE_API_KEY"
    service = "Jooble"

    def _check_options(self) -> None:
        super()._check_options()
        _ = self._whole_number("radius", None), self._whole_number("salary", None)

    async def _page(
        self,
        query: str,
        page: int,
        per_page: int | None,  # noqa: ARG002 - the service sets the page size
        cursor: str,  # noqa: ARG002 - paged by number
    ) -> tuple[list[Any], int | None, str]:
        body: dict[str, Any] = {"keywords": query, "page": page}
        if where := self.entry.options.get("where"):
            body["location"] = str(where)
        if (radius := self._whole_number("radius", None)) is not None:
            body["radius"] = str(radius)
        if (salary := self._whole_number("salary", None)) is not None:
            body["salary"] = str(salary)
        # The URL carries the key, and httpx and the fetcher log the URL.
        hide_in_logs(self.api_key)
        try:
            data = await self.fetcher.fetch_json(
                f"https://jooble.org/api/{self.api_key}",
                method="POST",
                json_body=body,
            )
        except FetchError as e:
            # The key is in the path, and a FetchError carries the URL into the
            # digest and the log. `from None`: the original keeps the URL too.
            raise FetchError(_JOOBLE_SHOWN, e.detail) from None
        if not isinstance(data, dict):
            detail = "unexpected response: not a JSON object"
            raise FetchError(_JOOBLE_SHOWN, detail)
        return list(data.get("jobs") or []), data.get("totalCount"), ""

    def _row(self, row: dict[str, Any]) -> Job:
        location = str(row.get("location") or "")
        return Job(
            source=f"{self.name}:{self.slug}",
            company=str(row.get("company") or "") or "unknown",
            title=strip_html(row.get("title", "")),
            location=location,
            url=str(row.get("link") or ""),
            description=strip_html(row.get("snippet", "")),
            posted=row.get("updated"),
            remote="remote" in location.casefold(),
            raw_id=str(row.get("id") or ""),
        )


@register
class WorkableSearch(_Aggregator):
    """https://jobs.workable.com/api/v1/jobs, Workable's search across the
    employers that post on it. No key.

    A GET with `query` and `location`, answering `{"title", "totalSize",
    "jobs": [...], "nextPageToken", "autoAppliedFilters"}`; a page is asked for
    again with the token the last one returned. One live request on 2026-10-09
    (after reading robots.txt, which has no rule against this path and no
    crawl delay) answered 200 with those top-level keys. It did not show the
    paging parameter's name: `pageToken` is the service's own site's, and the
    first thing to check against a second page of a real answer.

    Unlike the per-employer Workable board (`workable`), this reads other
    people's postings, so `posted` is when the advert went up.

    Its queries run one after another and every request follows the last by
    `delay` seconds (default 2). A read stops, and says so, when a page opens
    with a row an earlier page opened with or the answer's token is one it
    has already been sent: either means the paging parameter is not honoured,
    and following it would only repeat the first page."""

    name = "workable_search"
    slug_hint = "slug is a label, e.g. gb; set `where` to a place. No key."
    service = "Workable search"
    cursor_paged = True
    #: Workable has not said how much traffic its search takes.
    default_delay = 2.0

    async def _page(
        self,
        query: str,
        page: int,  # noqa: ARG002 - paged by token
        per_page: int | None,  # noqa: ARG002 - the service sets the page size
        cursor: str,
    ) -> tuple[list[Any], int | None, str]:
        params: dict[str, Any] = {"query": query}
        if where := self.entry.options.get("where"):
            params["location"] = str(where)
        if cursor:
            params["pageToken"] = cursor
        url = "https://jobs.workable.com/api/v1/jobs"
        data = await self.fetcher.fetch_json(url, params=params)
        if not isinstance(data, dict):
            detail = "unexpected response: not a JSON object"
            raise FetchError(url, detail)
        return (
            list(data.get("jobs") or []),
            data.get("totalSize"),
            str(data.get("nextPageToken") or ""),
        )

    def _row(self, row: dict[str, Any]) -> Job:
        title = str(row.get("title") or "")
        location = self._place(row)
        body = " ".join(
            strip_html(row.get(k, "")) for k in ("description", "requirementsSection")
        )
        company = row.get("company")
        return Job(
            source=f"{self.name}:{self.slug}",
            company=(
                str(company.get("title") or "") if isinstance(company, dict) else ""
            )
            or "unknown",
            title=title,
            location=location,
            url=str(row.get("url") or ""),
            description=body,
            posted=row.get("created"),
            remote=str(row.get("workplace") or "").casefold() == "remote"
            or "remote" in f"{title} {location}".casefold(),
            raw_id=str(row.get("id") or ""),
        )

    @staticmethod
    def _place(row: dict[str, Any]) -> str:
        """ "City, region, country" from the structured location, else the first
        of the plain `locations` strings, else nothing."""
        where = row.get("location")
        if isinstance(where, dict):
            parts = (where.get(k) for k in ("city", "subregion", "countryName"))
            if joined := ", ".join(str(p) for p in parts if p):
                return joined
        plain = row.get("locations")
        if isinstance(plain, list) and plain:
            return str(plain[0])
        return ""
