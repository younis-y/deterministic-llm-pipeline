"""Adzuna, the one aggregator in the mix.

Free tier is 1,000 calls a month, which is far more than a weekday scan needs.

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
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from rolescan.http import FetchError
from rolescan.models import Job
from rolescan.sources.base import Source, SourceSkipped, register, strip_html

__all__ = ["Adzuna"]

log = logging.getLogger(__name__)

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

        pages = await asyncio.gather(
            *(self._search(q) for q in self.queries), return_exceptions=True
        )
        out: list[Job] = []
        for q, page in zip(self.queries, pages, strict=False):
            if isinstance(page, BaseException):
                log.warning("adzuna %s/%s: %s", self.country, q, page)
                continue
            out.extend(page)
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

    async def _search(self, query: str) -> list[Job]:
        params: dict[str, Any] = {
            "app_id": self.app_id,
            "app_key": self.app_key,
            "what": query,
            "results_per_page": 50,
            "max_days_old": int(self.entry.options.get("max_days_old", 7)),
            "content-type": "application/json",
        }
        where = self.entry.options.get("where")
        if where:
            params["where"] = str(where)

        try:
            data = await self.fetcher.fetch_json(
                f"https://api.adzuna.com/v1/api/jobs/{self.country}/search/1",
                params=params,
            )
        except FetchError:
            raise

        out: list[Job] = []
        for p in data.get("results") or []:
            location = (p.get("location") or {}).get("display_name", "")
            out.append(
                Job(
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
            )
        return out
