"""Adzuna, the first aggregator in the mix (Reed and Jooble are in aggregators.py).

Free tier is 1,000 calls a month. A query reads up to `max_pages` pages (five
by default) and each page is one call, so a scan makes at most queries x
`max_pages` calls: six queries is up to 30 a scan, 660 over 22 weekdays.

Known limitation, and the reason the curated employer list carries the Gulf on
its own: Adzuna covers 20 countries and the UAE is not one of them. It is a
UK and Europe instrument. Configure it for `gb` and do not expect Abu Dhabi
roles to appear here.

Config differs from the ATS sources because there is no company slug:

    sources:
      - kind: adzuna
        slug: gb                 # country code
        app_id: "..."            # or ADZUNA_APP_ID in the environment
        app_key: "..."           # or ADZUNA_APP_KEY
        queries: [energy data scientist, power market analyst]
        max_days_old: 7
        max_pages: 5             # pages read per query (default 5)
        results_per_page: 50     # rows a page asks for (1 to 50, default 50)

A query is read page by page until a page comes back short, the API's own
`count` is reached, or `max_pages` pages are read. A read cut at `max_pages`
says so (`Source.truncated`), with the API's count as `Source.total` when
there is one query: several queries overlap, so no single figure is the
board's.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from rolescan.http import FetchError, hide_in_logs
from rolescan.models import Job
from rolescan.sources.base import Source, SourceSkipped, register, strip_html

__all__ = ["Adzuna"]

log = logging.getLogger(__name__)

#: Adzuna refuses a page larger than this.
_MAX_PER_PAGE = 50
_DEFAULT_MAX_PAGES = 5

SUPPORTED = frozenset(
    {
        "at",
        "au",
        "be",
        "br",
        "ca",
        "ch",
        "de",
        "es",
        "fr",
        "in",
        "it",
        "mx",
        "nl",
        "nz",
        "pl",
        "ru",
        "sg",
        "us",
        "za",
        "gb",
    }
)


@register
class Adzuna(Source):
    name = "adzuna"
    slug_hint = "slug is a country code, e.g. gb. Key from developer.adzuna.com"

    @property
    def country(self) -> str:
        return self.slug.lower()

    @property
    def app_id(self) -> str:
        return str(
            self.entry.options.get("app_id") or os.environ.get("ADZUNA_APP_ID", "")
        )

    @property
    def app_key(self) -> str:
        return str(
            self.entry.options.get("app_key") or os.environ.get("ADZUNA_APP_KEY", "")
        )

    @property
    def queries(self) -> list[str]:
        raw = self.entry.options.get("queries") or []
        return [str(q) for q in raw]

    def _whole_number(self, name: str, default: int, high: int | None = None) -> int:
        """An option that must be a whole number of at least 1 (and at most
        `high`), or an error that names it; a bool is not a number here."""
        raw = self.entry.options.get(name, default)
        bound = f"from 1 to {high}" if high is not None else "of at least 1"
        if (
            isinstance(raw, bool)
            or not isinstance(raw, int)
            or raw < 1
            or (high is not None and raw > high)
        ):
            msg = f"adzuna option {name} must be a whole number {bound}, not {raw!r}"
            raise ValueError(msg)
        return raw

    @property
    def max_pages(self) -> int:
        """Pages read per query before the read is cut and says so."""
        return self._whole_number("max_pages", _DEFAULT_MAX_PAGES)

    @property
    def results_per_page(self) -> int:
        return self._whole_number("results_per_page", _MAX_PER_PAGE, _MAX_PER_PAGE)

    async def fetch(self) -> list[Job]:
        # These raise rather than returning [] so that `discover` reports
        # SKIPPED. Returning an empty list here would render as a successful
        # zero-result probe, which is a lie: nothing was tested.
        if not (self.app_id and self.app_key):
            msg = "no credentials (set ADZUNA_APP_ID and ADZUNA_APP_KEY)"
            raise SourceSkipped(msg)
        if self.country not in SUPPORTED:
            msg = (
                f"country {self.country!r} not covered by Adzuna "
                "(the UAE is not among its 20 countries)"
            )
            raise SourceSkipped(msg)
        # The credentials travel as query parameters, and httpx and the fetcher
        # log the URL under --verbose (2.6.0).
        hide_in_logs(self.app_id)
        hide_in_logs(self.app_key)

        # Read before the first request, so a bad one fails the source with its
        # own message and not half way through a read.
        cap, per_page = self.max_pages, self.results_per_page
        self.total, self.truncated, self.note = None, "", ""

        pages = await asyncio.gather(
            *(self._search(q, cap, per_page) for q in self.queries),
            return_exceptions=True,
        )
        out: list[Job] = []
        totals: list[int | None] = []
        cut: list[str] = []
        for q, page in zip(self.queries, pages, strict=False):
            if isinstance(page, BaseException):
                log.warning("adzuna %s/%s: %s", self.country, q, page)
                continue
            jobs, total, why = page
            out.extend(jobs)
            totals.append(total)
            if why:
                cut.append(why)
        # One query: the API's own count. Several overlap, so none is claimed.
        self.total = totals[0] if len(self.queries) == 1 and totals else None
        self.truncated = "; ".join(cut)
        failures = [p for p in pages if isinstance(p, BaseException)]
        if failures and len(failures) == len(pages):
            first = failures[0]
            # FetchError is (url, detail) and renders "url: detail", so reuse
            # the failed request's own url and detail rather than printing the
            # same url twice in the digest line.
            url, reason = (
                (first.url, first.detail)
                if isinstance(first, FetchError)
                else ("", str(first))
            )
            msg = f"every Adzuna query failed; first: {reason}"
            raise FetchError(url, msg) from first
        if failures:
            log.warning(
                "adzuna %s: %d of %d queries failed",
                self.country,
                len(failures),
                len(pages),
            )
        return out

    async def _search(
        self, query: str, cap: int, per_page: int
    ) -> tuple[list[Job], int | None, str]:
        """(the jobs one query read, the API's count for it, why the read
        stopped short or "").

        Pages are read in order. The read ends on a short page (the last one),
        on reaching the stated `count`, or at `max_pages`; only the last of
        these is a cut, and says so. A page after the first that fails keeps
        what was read and says so too: the first page failing fails the query,
        as it always did."""
        params: dict[str, Any] = {
            "app_id": self.app_id,
            "app_key": self.app_key,
            "what": query,
            "results_per_page": per_page,
            "max_days_old": int(self.entry.options.get("max_days_old", 7)),
            "content-type": "application/json",
        }
        where = self.entry.options.get("where")
        if where:
            params["where"] = str(where)

        out: list[Job] = []
        total: int | None = None
        for page in range(1, cap + 1):
            try:
                data = await self.fetcher.fetch_json(
                    f"https://api.adzuna.com/v1/api/jobs/{self.country}/search/{page}",
                    params=params,
                )
            except FetchError as e:
                if page == 1:
                    raise
                log.warning("adzuna %s/%s page %d: %s", self.country, query, page, e)
                known = f" of {total}" if total is not None else ""
                return (
                    out,
                    total,
                    f'query "{query}": page {page} failed ({e.detail}), '
                    f"{len(out)}{known} read",
                )
            rows = data.get("results") or []
            count = data.get("count")
            if total is None and isinstance(count, int) and not isinstance(count, bool):
                total = max(count, 0)
            out.extend(self._parse(p) for p in rows)
            if len(rows) < per_page or (total is not None and len(out) >= total):
                return out, total, ""
        stated = f"{len(out)} of {total}" if total is not None else f"{len(out)}"
        none = "" if total is not None else ", the API stated no count"
        return (
            out,
            total,
            f'query "{query}": stopped at {stated} results{none} (max_pages {cap})',
        )

    def _parse(self, p: dict[str, Any]) -> Job:
        location = (p.get("location") or {}).get("display_name", "")
        return Job(
            source=f"{self.name}:{self.country}",
            company=(p.get("company") or {}).get("display_name") or "unknown",
            title=p.get("title", ""),
            location=location,
            url=p.get("redirect_url", ""),
            description=strip_html(p.get("description", "")),
            posted=p.get("created"),
            remote="remote" in location.casefold(),
            raw_id=str(p.get("id") or ""),
        )
