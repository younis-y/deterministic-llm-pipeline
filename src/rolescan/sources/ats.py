"""The keyless ATS feeds.

Every one of these is the public JSON that backs a company's own careers page.
No key, no scraping, no terms-of-service grey area: it is the same data the
careers site renders, in the form the careers site consumes it.
"""

from __future__ import annotations

from typing import Any

from rolescan.http import FetchError
from rolescan.models import Job
from rolescan.sources.base import Source, first_str, register, strip_html

__all__ = ["Ashby", "Greenhouse", "Lever", "SmartRecruiters", "Workable"]

_REMOTE_HINTS = ("remote", "anywhere", "work from home")

#: Lever hosts EU customers on a separate domain, and a board on the wrong
#: one answers 404 - indistinguishable from a mistyped slug. Trying both
#: costs one extra request only when the first misses, and keeps a genuinely
#: wrong slug failing loudly: it 404s on both hosts and the error propagates.
_LEVER_HOSTS = ("api.lever.co", "api.eu.lever.co")


def _is_remote(*fields: str) -> bool:
    blob = " ".join(fields).casefold()
    return any(h in blob for h in _REMOTE_HINTS)


@register
class SmartRecruiters(Source):
    """https://api.smartrecruiters.com/v1/companies/{slug}/postings

    Paginated at 100. Masdar, and much of the Gulf energy sector, sits here.

    Caveat that shapes `discover`: this endpoint answers HTTP 200 with
    totalFound=0 for a company that does not exist, identically to a real
    employer with no current vacancies. A nonsense slug looks exactly like a
    quiet board, so a zero count here proves nothing.
    """

    name = "smartrecruiters"
    slug_hint = "jobs.smartrecruiters.com/<Slug> -> slug: <Slug> (case sensitive)"
    ambiguous_when_empty = True

    async def fetch(self) -> list[Job]:
        url = f"https://api.smartrecruiters.com/v1/companies/{self.slug}/postings"
        jobs: list[Job] = []
        offset = 0
        while True:
            data = await self.fetcher.fetch_json(
                url, params={"limit": 100, "offset": offset}
            )
            content = data.get("content") or []
            jobs.extend(self._parse(p) for p in content)
            offset += len(content)
            if not content or offset >= int(data.get("totalFound") or 0):
                break
        return jobs

    def _parse(self, p: dict[str, Any]) -> Job:
        loc = p.get("location") or {}
        location = ", ".join(str(x) for x in (loc.get("city"), loc.get("country")) if x)
        # The list endpoint returns a summary; jobAd carries the sections.
        ad = p.get("jobAd") or {}
        sections = (ad.get("sections") or {}) if isinstance(ad, dict) else {}
        body = " ".join(
            strip_html((sections.get(k) or {}).get("text", ""))
            for k in ("companyDescription", "jobDescription", "qualifications")
        )
        return Job(
            source=self.name,
            company=self.label,
            title=first_str(p, "name", "title"),
            location=location,
            url=f"https://jobs.smartrecruiters.com/{self.slug}/{p.get('id')}",
            description=body or strip_html(p.get("jobAdText", "")),
            posted=p.get("releasedDate") or p.get("createdOn"),
            remote=bool(loc.get("remote")) or _is_remote(location),
            raw_id=str(p.get("id") or ""),
        )


@register
class Greenhouse(Source):
    """https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true"""

    name = "greenhouse"
    slug_hint = "boards.greenhouse.io/<slug> -> slug: <slug>"

    async def fetch(self) -> list[Job]:
        data = await self.fetcher.fetch_json(
            f"https://boards-api.greenhouse.io/v1/boards/{self.slug}/jobs",
            params={"content": "true"},
        )
        out: list[Job] = []
        for p in data.get("jobs") or []:
            location = (p.get("location") or {}).get("name", "")
            out.append(
                Job(
                    source=self.name,
                    company=self.label,
                    title=p.get("title", ""),
                    location=location,
                    url=p.get("absolute_url", ""),
                    description=strip_html(p.get("content", "")),
                    posted=first_str(p, "first_published", "updated_at"),
                    remote=_is_remote(location),
                    raw_id=str(p.get("id") or ""),
                )
            )
        return out


@register
class Lever(Source):
    """https://api.lever.co/v0/postings/{slug}?mode=json (or the EU host)"""

    name = "lever"
    slug_hint = "jobs.lever.co/<slug> -> slug: <slug>"

    async def fetch(self) -> list[Job]:
        data: Any = None
        for host in _LEVER_HOSTS:
            try:
                data = await self.fetcher.fetch_json(
                    f"https://{host}/v0/postings/{self.slug}", params={"mode": "json"}
                )
            except FetchError:
                if host == _LEVER_HOSTS[-1]:
                    raise
                continue
            break
        out: list[Job] = []
        for p in data or []:
            cats = p.get("categories") or {}
            location = str(cats.get("location") or "")
            out.append(
                Job(
                    source=self.name,
                    company=self.label,
                    title=p.get("text", ""),
                    location=location,
                    url=first_str(p, "hostedUrl", "applyUrl"),
                    description=strip_html(
                        p.get("descriptionPlain") or p.get("description", "")
                    ),
                    posted=p.get("createdAt"),
                    remote=str(cats.get("commitment") or "").casefold() == "remote"
                    or _is_remote(location),
                    raw_id=str(p.get("id") or ""),
                )
            )
        return out


@register
class Ashby(Source):
    """https://api.ashbyhq.com/posting-api/job-board/{slug}"""

    name = "ashby"
    slug_hint = "jobs.ashbyhq.com/<slug> -> slug: <slug>"

    async def fetch(self) -> list[Job]:
        data = await self.fetcher.fetch_json(
            f"https://api.ashbyhq.com/posting-api/job-board/{self.slug}",
            params={"includeCompensation": "true"},
        )
        return [
            Job(
                source=self.name,
                company=self.label,
                title=p.get("title", ""),
                location=p.get("location", ""),
                url=first_str(p, "jobUrl", "applyUrl"),
                description=strip_html(
                    p.get("descriptionHtml") or p.get("descriptionPlain", "")
                ),
                posted=p.get("publishedAt"),
                remote=bool(p.get("isRemote")) or _is_remote(p.get("location", "")),
                raw_id=str(p.get("id") or ""),
            )
            for p in data.get("jobs") or []
        ]


@register
class Workable(Source):
    """https://apply.workable.com/api/v1/widget/accounts/{slug}?details=true"""

    name = "workable"
    slug_hint = "apply.workable.com/<slug> -> slug: <slug>"

    async def fetch(self) -> list[Job]:
        data = await self.fetcher.fetch_json(
            f"https://apply.workable.com/api/v1/widget/accounts/{self.slug}",
            params={"details": "true"},
        )
        out: list[Job] = []
        for p in data.get("jobs") or []:
            location = ", ".join(str(x) for x in (p.get("city"), p.get("country")) if x)
            out.append(
                Job(
                    source=self.name,
                    company=self.label,
                    title=p.get("title", ""),
                    location=location,
                    url=first_str(p, "url", "shortlink", "application_url"),
                    description=strip_html(p.get("description", "")),
                    posted=p.get("published_on"),
                    remote=bool(p.get("telecommuting")) or _is_remote(location),
                    raw_id=str(p.get("shortcode") or p.get("id") or ""),
                )
            )
        return out
