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

Three optional keys shape the read (2.5.8):

        max_rows: 2000         # postings read per pass (default 500)
        applied_facets: {Country_and_Jurisdiction: ["<value id>"]}
        search_text: [data, analyst]   # one pass per text

A read that stops short of the board's own total says so in `truncated`.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from pydantic import ValidationError

from rolescan.http import FetchError
from rolescan.models import Job
from rolescan.sources.base import Source, register, strip_html

__all__ = ["Workday"]

log = logging.getLogger(__name__)

_PAGE = 20
#: Postings read per pass unless `max_rows` says otherwise: 25 pages of 20,
#: the fixed cap before 2.5.8.
_MAX_ROWS = 500
_DETAIL_CONCURRENCY = 4


def _json_option(name: str, text: str) -> object:
    """A JSON-encoded option value (how a plugin's string-only params carry a
    mapping or a list), or a ValueError that names the option."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        msg = f"workday option {name} is not valid JSON: {e}"
        raise ValueError(msg) from e


@register
class Workday(Source):
    name = "workday"
    #: An employer's own Workday site: the date read is `startDate` from a
    #: posting's detail, when the requisition opened (the listing carries only
    #: "Posted Today", so `posted` is None until the detail is read), and its
    #: presence on the board is the freshness signal (2.6.0; it inherited the
    #: aggregator default before).
    dates_are_freshness = False
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
    def max_rows(self) -> int:
        """Postings read per listing pass (option `max_rows`, default 500).

        2.5.7 stopped every board after 25 pages of 20 and said nothing:
        four large boards were each read as exactly 500 on 2026-10-06, against
        live totals of 2,000 to 3,891. The default stays 500, because each
        posting past it costs a detail request; a board that holds more is
        now reported as cut short, and this lifts the cap for one worth it.
        A numeric string is accepted: plugins pass options through from
        their own config, where every value is a string (2.5.8)."""
        raw = self.entry.options.get("max_rows", _MAX_ROWS)
        try:
            rows = int(str(raw))
        except ValueError:
            rows = 0
        if rows < 1:
            msg = (
                "workday option max_rows must be a whole number of 1 or more, "
                f"not {raw!r}"
            )
            raise ValueError(msg)
        return rows

    @property
    def applied_facets(self) -> dict[str, list[str]]:
        """Workday's own filters, sent as `appliedFacets` (option
        `applied_facets`, 2.5.8): facet parameter -> the value ids to keep.

        Narrows a board the cap would otherwise cut: one bank's board lists
        2,000 postings worldwide and offers a country facet with 50 values.
        The ids are in the `facets` block of the board's own listing answer.
        A mapping in YAML, or the same as a JSON object string; a single id
        stands for a one-item list."""
        raw = self.entry.options.get("applied_facets") or {}
        if isinstance(raw, str):
            raw = _json_option("applied_facets", raw)
        if not isinstance(raw, dict):
            msg = f"workday option applied_facets must be a mapping, not {raw!r}"
            raise ValueError(msg)
        facets: dict[str, list[str]] = {}
        for facet, ids in raw.items():
            values = [ids] if isinstance(ids, str) else ids
            if not isinstance(values, list) or not all(
                isinstance(v, str) for v in values
            ):
                msg = (
                    f"workday option applied_facets: {facet!r} needs a list "
                    f"of value ids, not {ids!r}"
                )
                raise ValueError(msg)
            facets[str(facet)] = values
        return facets

    @property
    def search_texts(self) -> list[str]:
        """The listing passes to read, one per text (option `search_text`,
        2.5.8): a string, a list, or a JSON list string. The default is one
        pass with no text: the whole board, or the whole faceted board.

        A pass per text reaches postings that one capped pass would cut. The
        passes overlap, so a posting is kept once, by its `externalPath`."""
        raw = self.entry.options.get("search_text", "")
        if isinstance(raw, str):
            if not raw.strip().startswith("["):
                return [raw.strip()]
            raw = _json_option("search_text", raw)
        if not isinstance(raw, list) or not all(isinstance(t, str) for t in raw):
            msg = (
                "workday option search_text must be a string or a list of "
                f"strings, not {raw!r}"
            )
            raise ValueError(msg)
        return list(dict.fromkeys(t.strip() for t in raw)) or [""]

    @property
    def _base(self) -> str:
        return (
            f"https://{self.slug}.{self.host}.myworkdayjobs.com"
            f"/wday/cxs/{self.slug}/{self.site}"
        )

    async def fetch(self) -> list[Job]:
        # All three options are read before the first request, so a bad one
        # fails the source with its own message, not half way through a read.
        texts, facets, cap = self.search_texts, self.applied_facets, self.max_rows
        # What a read says about itself is that read's own: a source object
        # fetched twice must not report the first read's note on the second.
        self.total, self.truncated, self.note = None, "", ""
        postings: list[dict[str, Any]] = []
        paths: set[str] = set()
        totals: list[int] = []
        cut: list[str] = []
        for text in texts:
            rows, total = await self._read_pass(text, facets, cap)
            totals.append(total)
            if len(rows) < total:
                cut.append(self._shortfall(len(rows), total, text, cap))
            for row in rows:
                path = str(row.get("externalPath") or "")
                if path and path in paths:
                    continue
                paths.add(path)
                postings.append(row)
        # One pass: the board's own count. Several passes overlap, so no one
        # figure is the board's count, and none is claimed.
        self.total = totals[0] if len(totals) == 1 else None
        self.truncated = "; ".join(cut)

        pairs = self._parse_all(postings)
        jobs = [job for job, _ in pairs]
        if self.fetch_details:
            jobs = await self._enrich(jobs, [row for _, row in pairs])
        return jobs

    async def _read_pass(
        self, text: str, facets: dict[str, list[str]], cap: int
    ) -> tuple[list[dict[str, Any]], int]:
        """(the postings one listing pass read, the total the board stated).

        Workday reports `total` on the first page only and 0 on every later
        one (live, Shell 2026-10-06: 140, then 0, 0), so the first page's
        figure is the one to page against; a per-page check stopped at 40."""
        rows: list[dict[str, Any]] = []
        total = 0
        while len(rows) < cap:
            data = await self._list_page(len(rows), text, facets)
            page = data.get("jobPostings") or []
            total = max(total, int(data.get("total") or 0))
            rows.extend(page[: cap - len(rows)])
            if not page or len(rows) >= total:
                break
        return rows, total

    async def _list_page(
        self, offset: int, text: str, facets: dict[str, list[str]]
    ) -> dict[str, Any]:
        try:
            data: dict[str, Any] = await self.fetcher.fetch_json(
                f"{self._base}/jobs",
                method="POST",
                json_body={
                    "appliedFacets": facets,
                    "limit": _PAGE,
                    "offset": offset,
                    "searchText": text,
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
        return data

    @staticmethod
    def _shortfall(read: int, total: int, text: str, cap: int) -> str:
        """Why one pass read `read` of the `total` postings the board stated."""
        where = f' for search "{text}"' if text else ""
        if read >= cap:
            return f"stopped at {read} of {total} postings{where} (max_rows {cap})"
        return f"the board lists {total} postings{where} but its pages ended at {read}"

    def _parse_all(
        self, postings: list[dict[str, Any]]
    ) -> list[tuple[Job, dict[str, Any]]]:
        """Each posting that makes a valid `Job`, paired with its listing row.

        On 2026-10-07 a live board failed whole: one posting had no title,
        `Job` refused it, and the exception took every other posting on the
        board with it. A posting that fails validation is now skipped,
        logged, and counted in `note` (2.5.8)."""
        pairs: list[tuple[Job, dict[str, Any]]] = []
        bad: list[str] = []
        for row in postings:
            try:
                pairs.append((self._parse(row), row))
            except ValidationError as e:
                path = str(row.get("externalPath") or "") or "(no path)"
                bad.append(f"{path}: {e.errors()[0]['msg']}")
        if bad:
            self.note = (
                f"skipped {len(bad)} posting(s) that are not valid jobs "
                f"(first: {bad[0]})"
            )
            log.warning("workday %s: %s", self.slug, self.note)
        return pairs

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
