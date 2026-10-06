"""Workday.

Worth the extra effort: ADNOC and most large Gulf and UK industrial employers
run Workday, and none of them appear in any free aggregator. The public careers
site is a React app talking to a "CXS" JSON endpoint, which is what this hits.

Unlike the other ATS feeds it needs three coordinates, because one Workday
tenant can host several branded career sites:

    sources:
      - kind: workday
        slug: adnoc            # tenant
        site: ADNOC_Careers    # career site id
        host: wd3              # wd1..wd103, from the careers URL
"""

from __future__ import annotations

import asyncio
from typing import Any

from rolescan.http import FetchError
from rolescan.models import Job
from rolescan.sources.base import Source, register, strip_html

__all__ = ["Workday"]

_PAGE = 20
_MAX_PAGES = 25
_DETAIL_CONCURRENCY = 4


@register
class Workday(Source):
    name = "workday"
    slug_hint = (
        "<tenant>.wd3.myworkdayjobs.com/<Site> -> slug: <tenant>, "
        "site: <Site>, host: wd3"
    )

    @property
    def site(self) -> str:
        return str(self.entry.options.get("site") or self.slug)

    @property
    def host(self) -> str:
        return str(self.entry.options.get("host") or "wd3")

    @property
    def fetch_details(self) -> bool:
        """Descriptions need one extra request each. Worth it, because the
        listing payload has no body text and the LLM scorer would be judging
        titles alone without it."""
        return bool(self.entry.options.get("details", True))

    @property
    def _base(self) -> str:
        return (
            f"https://{self.slug}.{self.host}.myworkdayjobs.com"
            f"/wday/cxs/{self.slug}/{self.site}"
        )

    async def fetch(self) -> list[Job]:
        postings: list[dict[str, Any]] = []
        offset = 0
        # Workday reports `total` on the first page only and 0 on every later
        # one (live, Shell 2026-10-06: 140, then 0, 0), so the first page's
        # figure is the one to page against; a per-page check stopped at 40.
        total = 0
        for _ in range(_MAX_PAGES):
            try:
                data = await self.fetcher.fetch_json(
                    f"{self._base}/jobs",
                    method="POST",
                    json_body={
                        "appliedFacets": {},
                        "limit": _PAGE,
                        "offset": offset,
                        "searchText": "",
                    },
                    headers={"Content-Type": "application/json"},
                )
            except FetchError as e:
                # 404 and 422 mean different things and point at different
                # fixes, so say which. Verified against the live API:
                #
                #     tenant=adnoc                 bogus site -> 422
                #     tenant=zzznonsensetenant999  bogus site -> 422
                #     tenant=centrica              bogus site -> 404
                #
                # `*.myworkdayjobs.com` is a wildcard record, so every hostname
                # connects and Workday answers 422 when no tenant sits behind
                # it. A 404 is the CLOSER miss: the tenant is real and rejected
                # the site id, which is the one part a company directory cannot
                # supply and the one worth retrying.
                if "422" in e.detail:
                    msg = (
                        f"HTTP 422: no Workday tenant {self.slug!r} on "
                        f"{self.host!r}. The slug or host is wrong, not the "
                        "career page. Read both from the careers URL: "
                        "<tenant>.<host>.myworkdayjobs.com"
                    )
                    raise FetchError(e.url, msg) from e
                if "404" in e.detail:
                    msg = (
                        f"HTTP 404: tenant {self.slug!r} exists but rejected "
                        f"site {self.site!r}. Check the segment after /en-US/ "
                        "in the careers URL."
                    )
                    raise FetchError(e.url, msg) from e
                raise
            page = data.get("jobPostings") or []
            postings.extend(page)
            offset += len(page)
            total = max(total, int(data.get("total") or 0))
            if not page or offset >= total:
                break

        jobs = [self._parse(p) for p in postings]
        if self.fetch_details:
            jobs = await self._enrich(jobs, postings)
        return jobs

    def _parse(self, p: dict[str, Any]) -> Job:
        path = str(p.get("externalPath") or "")
        location = str(p.get("locationsText") or "")
        return Job(
            source=self.name,
            company=self.label,
            title=str(p.get("title") or ""),
            location=location,
            url=(
                f"https://{self.slug}.{self.host}.myworkdayjobs.com"
                f"/en-US/{self.site}{path}"
            ),
            description="",
            posted=None,  # Workday gives "Posted Today", not a date.
            remote="remote" in location.casefold(),
            raw_id=path,
        )

    async def _enrich(
        self, jobs: list[Job], postings: list[dict[str, Any]]
    ) -> list[Job]:
        sem = asyncio.Semaphore(_DETAIL_CONCURRENCY)

        async def one(job: Job, posting: dict[str, Any]) -> Job:
            path = str(posting.get("externalPath") or "")
            if not path:
                return job
            async with sem:
                try:
                    detail = await self.fetcher.fetch_json(f"{self._base}{path}")
                except FetchError:
                    return job
            info = detail.get("jobPostingInfo") or {}
            # model_copy(update=...) does NOT re-validate, so Workday's raw
            # startDate string would sit in a `date` field until something
            # called .isoformat() on it and the whole scan died in the digest.
            # Re-validating is the only way an update reaches the parsers.
            return Job.model_validate(
                job.model_dump()
                | {
                    "description": strip_html(info.get("jobDescription", "")),
                    "posted": info.get("startDate") or None,
                }
            )

        results = await asyncio.gather(
            *(one(j, p) for j, p in zip(jobs, postings, strict=False)),
            return_exceptions=True,
        )
        return [
            r if isinstance(r, Job) else j for r, j in zip(results, jobs, strict=False)
        ]
