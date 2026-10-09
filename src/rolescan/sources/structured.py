"""Generic schema.org source: sitemap for discovery, extruct for extraction.

One adapter for every careers platform that server-renders schema.org
JobPosting, in whichever syntax it happens to use. Verified 2026-08-25:

    ADNOC       Phenom People    JSON-LD    169 postings
    ACWA Power  SuccessFactors   MICRODATA   21 postings
    National Grid SuccessFactors MICRODATA  157 postings

Asking only for JSON-LD would have written off two of the three: SuccessFactors
emits the same vocabulary as microdata attributes and carries no ld+json script
at all. extruct reads JSON-LD, microdata and RDFa in one pass, so the syntax
stops being something this file has to care about.

WHY THERE ARE NO CONDITIONAL REQUESTS HERE
Neither host sends ETag or Last-Modified, and both answer a future
`If-Modified-Since` with 200 and the whole body -- 1.25MB for ADNOC. A
conditional GET would therefore cost a full download per page per run and save
nothing. The sitemap's <lastmod> is the only invalidation signal that works, and
it arrives for every URL in a single request, before anything is downloaded.

A LARGE SITEMAP (2.6.0)
A job board's latest-jobs sitemap lists thousands of URLs, and four options
keep reading one polite and bounded:

  max_sitemap_urls  refuse a sitemap that lists more than this many URLs
                    (default 50,000) with an error, instead of working on it.
  max_age_days      skip a URL whose `lastmod` is more than this many days old.
  incremental       read only the URLs whose `lastmod` is newer than the last
                    scan that read its whole window (a read cut at `max_pages`
                    holds the mark and repeats); see `_since_date`. Only
                    `lastmod` is compared, so a URL first listed with one older
                    than the mark is not read until a scan with `incremental`
                    off sweeps the window.
  (robots.txt)      a `Crawl-delay` there is a floor under `delay`.

None of them guesses: a URL with no readable `lastmod` is never skipped for
age, because unknown is not old.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from datetime import UTC, date, datetime, timedelta
from html import unescape
from typing import Any
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser
from xml.etree import ElementTree

import extruct

from rolescan.config import SourceEntry
from rolescan.http import Fetcher, FetchError
from rolescan.models import Job
from rolescan.sources.base import (
    PostingCache,
    ProbeResult,
    ProbeStatus,
    Source,
    register,
    strip_html,
)

__all__ = ["Structured"]

log = logging.getLogger(__name__)

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")

#: A sitemap that lists more URLs than this is refused unless `max_sitemap_urls`
#: says otherwise: the protocol's own limit is 50,000 per file.
_DEFAULT_MAX_SITEMAP_URLS = 50_000

#: The longest `Crawl-delay` honoured, in seconds. A site asking for more is
#: read at this pace and the note says so: with `max_pages` pages a scan would
#: otherwise be held for hours by one line of a text file.
_MAX_CRAWL_DELAY = 30.0

#: The product token robots.txt groups are matched against.
_ROBOTS_AGENT = "rolescan"

#: An incremental read opens its window this many days before the last mark.
#: The mark is when the sitemap was read, a page edited after that read and
#: before midnight carries a `lastmod` of the day before the mark's date, and
#: a date is all most sitemaps give.
_SLACK_DAYS = 1

#: What a detail page answers when the posting is gone for good. Any other
#: failure may pass, so an incremental read holds its mark back for it.
_GONE = frozenset({"HTTP 404", "HTTP 410"})

#: Whole <script>/<style> elements, contents included. strip_html only removes
#: the tags, so without this a page's JavaScript ends up in the description.
_SCRIPT_STYLE = re.compile(
    r"<(script|style|noscript)\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL
)

#: Body-text fallback ceiling. The LLM scorer trims to `description_chars`
#: anyway; this only stops a pathological page filling the store.
_MAX_BODY_CHARS = 12000

#: ISO-3166 alpha-2 -> the name a human would search for. SuccessFactors puts a
#: bare code in streetAddress ("SA"), and the profile's location list is matched
#: by substring, so a code neither matches "saudi" nor can safely be added to
#: the list: "sa" is inside "usa" and "Sale". Expanding it here is the only
#: place with enough context to do it correctly.
_COUNTRY = {
    "AE": "United Arab Emirates",
    "AR": "Argentina",
    "AT": "Austria",
    "AU": "Australia",
    "AZ": "Azerbaijan",
    "BE": "Belgium",
    "BH": "Bahrain",
    "BR": "Brazil",
    "CA": "Canada",
    "CH": "Switzerland",
    "CN": "China",
    "DE": "Germany",
    "DK": "Denmark",
    "EG": "Egypt",
    "ES": "Spain",
    "FR": "France",
    "GB": "United Kingdom",
    "GR": "Greece",
    "ID": "Indonesia",
    "IE": "Ireland",
    "IN": "India",
    "IT": "Italy",
    "JO": "Jordan",
    "KW": "Kuwait",
    "MA": "Morocco",
    "MX": "Mexico",
    "MY": "Malaysia",
    "NL": "Netherlands",
    "NO": "Norway",
    "NZ": "New Zealand",
    "OM": "Oman",
    "PH": "Philippines",
    "PL": "Poland",
    "PT": "Portugal",
    "QA": "Qatar",
    "SA": "Saudi Arabia",
    "SE": "Sweden",
    "SG": "Singapore",
    "TH": "Thailand",
    "TR": "Turkey",
    "US": "United States",
    "UZ": "Uzbekistan",
    "VN": "Vietnam",
    "ZA": "South Africa",
}
_JOB_POSTING = "jobposting"
_SYNTAXES = ["json-ld", "microdata", "rdfa"]
_LOC = re.compile(r"<loc>\s*([^<]+?)\s*</loc>", re.IGNORECASE)

#: Java's Date.toString, which SuccessFactors emits for datePosted and
#: validThrough: "Wed Aug 05 00:00:00 UTC 2026". Not ISO, so Job._parse_date
#: discards it and every posting arrives with posted=None.
_JAVA_DATE = re.compile(
    r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2})\b.*?\b(\d{4})\b",
    re.IGNORECASE,
)
_MONTHS = {
    m: i
    for i, m in enumerate(
        [
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
            "oct",
            "nov",
            "dec",
        ],
        start=1,
    )
}


def _typename(value: object) -> str:
    """Last path segment of a schema.org type, lowercased.

    JSON-LD says "JobPosting"; microdata says "http://schema.org/JobPosting";
    RDFa may say "https://schema.org/JobPosting". Same thing three ways.
    """
    if isinstance(value, list):
        return " ".join(_typename(v) for v in value)
    return str(value).rstrip("/").rsplit("/", 1)[-1].casefold()


def _first(value: object) -> Any:
    """schema.org permits a bare value or a list of them, everywhere."""
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _text(value: object) -> str:
    """A property that may be a string, a list, or a nested node with a name."""
    value = _first(value)
    if isinstance(value, dict):
        for key in ("name", "@value", "value", "text"):
            if isinstance(value.get(key), str) and value[key].strip():
                return str(value[key]).strip()
        props = value.get("properties")
        if isinstance(props, dict):
            return _text(props.get("name"))
        return ""
    return str(value).strip() if value is not None else ""


def _expand_country(value: str) -> str:
    """A bare alpha-2 code to its country name; anything else untouched."""
    if len(value) == 2 and value.isalpha():
        return _COUNTRY.get(value.upper(), value)
    return value


def _props(node: object) -> dict[str, Any]:
    """Microdata nests real fields under `properties`; JSON-LD does not."""
    node = _first(node)
    if not isinstance(node, dict):
        return {}
    inner = node.get("properties")
    return inner if isinstance(inner, dict) else node


def _lastmod_date(raw: str) -> date | None:
    """The calendar date a sitemap `lastmod` (or a stored mark) starts with, or
    None. The W3C forms all begin YYYY-MM-DD; the time and zone after it are
    ignored, since a day is the grain every comparison here is made at."""
    text = raw.strip()
    if not _ISO_DATE.match(text):
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


@register
class Structured(Source):
    """Any careers site that publishes schema.org JobPosting and a sitemap."""

    name = "structured"
    #: A careers site's own sitemap: a JobPosting's `datePosted` is when the
    #: requisition was opened, and the page still being listed is the
    #: freshness signal (2.6.0; it inherited the aggregator default before).
    dates_are_freshness = False
    slug_hint = (
        "sitemap: <careers host>/sitemap.xml, url_pattern: a substring or regex "
        "matching job detail URLs, e.g. /job/"
    )

    def __init__(
        self,
        entry: SourceEntry,
        fetcher: Fetcher,
        cache: PostingCache | None = None,
    ) -> None:
        super().__init__(entry, fetcher, cache)
        #: Crawl-delay floor per origin, so robots.txt is read once a host.
        self._crawl_floors: dict[str, float] = {}

    @property
    def sitemap_url(self) -> str:
        return str(self.entry.options.get("sitemap") or "")

    @property
    def url_pattern(self) -> str:
        return str(self.entry.options.get("url_pattern") or "/job")

    @property
    def exclude_pattern(self) -> str:
        """Optional regex; matching URLs are dropped before any fetch.

        For boards that mix regions on one instance. National Grid runs UK and
        US hiring on a single SuccessFactors site, where 108 of 164 sitemap
        URLs are US roles the profile's location list penalises anyway. The
        regex is supplied in config rather than inferred, because only the
        reader knows which half of a board they want.
        """
        return str(self.entry.options.get("exclude_pattern") or "")

    @property
    def delay(self) -> float:
        """Seconds between detail fetches. A sitemap hands you every URL at
        once, which makes it very easy to hammer one host."""
        return float(self.entry.options.get("delay", 0.5))

    @property
    def max_pages(self) -> int:
        return int(self.entry.options.get("max_pages", 500))

    def _whole_number(self, name: str, default: int | None) -> int | None:
        """An option that must be a whole number of at least 1, or an error
        that names it; a bool is not a number here. None when it is absent
        and has no default."""
        raw = self.entry.options.get(name, default)
        if raw is None:
            return None
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
            msg = (
                f"structured option {name} must be a whole number of at least 1, "
                f"not {raw!r}"
            )
            raise ValueError(msg)
        return raw

    @property
    def max_sitemap_urls(self) -> int:
        """The most URLs a sitemap may list before it is refused."""
        return self._whole_number("max_sitemap_urls", _DEFAULT_MAX_SITEMAP_URLS) or 0

    @property
    def max_age_days(self) -> int | None:
        """Skip a URL whose `lastmod` is more than this many days old; None
        (the default) skips none."""
        return self._whole_number("max_age_days", None)

    @property
    def incremental(self) -> bool:
        """Read only what the sitemap says changed since the last whole scan."""
        raw = self.entry.options.get("incremental", False)
        if not isinstance(raw, bool):
            msg = f"structured option incremental must be true or false, not {raw!r}"
            raise ValueError(msg)
        return raw

    # -- what to read ---------------------------------------------------------

    def _entries(self, xml: str) -> list[tuple[str, str]]:
        """The sitemap's (url, lastmod) pairs, or an error when it lists more
        than `max_sitemap_urls`.

        Counted from the text first, so an oversized file is refused before it
        is parsed into anything."""
        limit = self.max_sitemap_urls
        listed = xml.count("<loc>")
        entries = [] if listed > limit else self._parse_sitemap(xml)
        listed = max(listed, len(entries))
        if listed > limit:
            msg = (
                f"the sitemap lists {listed} urls, more than max_sitemap_urls "
                f"({limit}): narrow the read with url_pattern or max_age_days, "
                "or raise max_sitemap_urls if this really is one careers board"
            )
            raise FetchError(self.sitemap_url, msg)
        return entries

    def _select(self, entries: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """The sitemap entries in scope: matching `url_pattern`, not matching
        `exclude_pattern`, and not older than `max_age_days`. Not yet cut to
        `max_pages`."""
        pattern = re.compile(self.url_pattern)
        wanted = [(u, lm) for u, lm in entries if pattern.search(u)]
        if self.exclude_pattern:
            drop = re.compile(self.exclude_pattern)
            kept = [(u, lm) for u, lm in wanted if not drop.search(u)]
            if len(kept) != len(wanted):
                log.info(
                    "structured %s: excluded %d of %d urls via exclude_pattern",
                    self.slug,
                    len(wanted) - len(kept),
                    len(wanted),
                )
            wanted = kept
        if (days := self.max_age_days) is not None:
            oldest = datetime.now(UTC).date() - timedelta(days=days)
            wanted = [
                (u, lm)
                for u, lm in wanted
                if (when := _lastmod_date(lm)) is None or when >= oldest
            ]
        return wanted

    def _fingerprint(self) -> str:
        """Which read a mark belongs to: the sitemap and the patterns that pick
        URLs from it. A mark made under other settings says nothing about what
        these would read, so `_since_date` ignores it."""
        blob = json.dumps([self.sitemap_url, self.url_pattern, self.exclude_pattern])
        return hashlib.sha256(blob.encode()).hexdigest()[:8]

    def _since_date(self) -> date | None:
        """The day the last whole scan read this board, from the mark the
        pipeline handed back in `since`; None when there is none, it is not
        this read's, or it is unreadable (everything in scope is then read)."""
        stamp, _, fingerprint = self.since.partition("|")
        if not stamp or fingerprint != self._fingerprint():
            return None
        return _lastmod_date(stamp)

    def _window(
        self, wanted: list[tuple[str, str]]
    ) -> tuple[list[tuple[str, str]], int]:
        """(the entries an incremental read still has to read, newest first,
        and how many of the rest had no lastmod to compare).

        An entry with no readable lastmod cannot be shown to be old, so it is
        always read. Not incremental: the entries as they are, in sitemap
        order, and 0."""
        if not self.incremental:
            return wanted, 0
        since = self._since_date()
        floor = since - timedelta(days=_SLACK_DAYS) if since else None
        undated = sum(1 for _, lm in wanted if _lastmod_date(lm) is None)
        kept = [
            (u, lm)
            for u, lm in wanted
            if floor is None or (when := _lastmod_date(lm)) is None or when >= floor
        ]
        kept.sort(key=lambda e: _lastmod_date(e[1]) or date.min, reverse=True)
        return kept, undated

    async def probe(self) -> ProbeResult:
        """Sitemap plus ONE sample page, never the whole board.

        The inherited probe calls fetch(), which here means a detail request
        per posting: 170 requests and roughly 212MB for ADNOC every time
        `rolescan discover` runs. The sitemap already carries the count, and a
        single sample is enough to prove the pages actually carry markup.
        """
        if not self.sitemap_url:
            return ProbeResult(
                ProbeStatus.FAIL, count=-1, detail="no `sitemap:` url configured"
            )
        try:
            xml = await self.fetcher.fetch_text(self.sitemap_url)
        except FetchError as e:
            return ProbeResult(ProbeStatus.FAIL, count=-1, detail=e.detail[:110])

        try:
            entries = self._entries(xml)
        except FetchError as e:
            return ProbeResult(ProbeStatus.FAIL, count=-1, detail=e.detail[:110])
        if not entries:
            return ProbeResult(ProbeStatus.EMPTY, detail="sitemap lists no urls")

        matched = [u for u, _ in self._select(entries)][: self.max_pages]
        if not matched:
            aged = (
                f" within max_age_days {self.max_age_days}"
                if self.max_age_days is not None
                else ""
            )
            return ProbeResult(
                ProbeStatus.UNKNOWN,
                detail=(
                    f"{len(entries)} urls in the sitemap, none match "
                    f"url_pattern {self.url_pattern!r}{aged}, "
                    f"e.g. {entries[0][0][:48]}"
                ),
            )

        try:
            sample = await self.fetcher.fetch_text(matched[0])
        except FetchError as e:
            return ProbeResult(
                ProbeStatus.FAIL, count=-1, detail=f"sample page: {e.detail}"[:110]
            )
        if self._to_job(sample, matched[0]) is None:
            return ProbeResult(
                ProbeStatus.UNKNOWN,
                count=len(matched),
                detail=(
                    f"{len(matched)} urls found, but the sample page carries no "
                    "schema.org JobPosting markup"
                ),
            )
        return ProbeResult(ProbeStatus.OK, count=len(matched))

    async def fetch(self) -> list[Job]:
        if not self.sitemap_url:
            msg = f"structured source {self.slug!r} needs a `sitemap:` url"
            # EM101 misreads this: "" is FetchError's url argument, and the
            # message is already in `msg`, which is what the rule asks for.
            raise FetchError("", msg)  # noqa: EM101

        # Every option is read before the first request, so a bad one fails the
        # source with its own message and not half way through a read.
        incremental = self.incremental
        self.total, self.truncated, self.note, self.next_mark = None, "", "", ""
        read_at = datetime.now(UTC).isoformat(timespec="seconds")

        entries = self._entries(await self.fetcher.fetch_text(self.sitemap_url))
        wanted, notes = self._plan(entries)
        jobs, unreachable = await self._read_pages(wanted, notes)

        if incremental:
            # A mark says "everything up to here was read". A read cut at
            # `max_pages` left URLs in the window unread, so it holds the mark
            # whatever the cause: the next scan opens the same window, and
            # repeats the cut until `max_pages` or `max_age_days` covers it.
            if unreachable:
                notes.append(
                    f"{unreachable} pages could not be read this time and "
                    "will be tried again next scan"
                )
            elif self.truncated:
                notes.append(
                    "the mark did not move, so the next scan opens the same "
                    "window: raise max_pages, or set max_age_days, to cover it"
                )
            else:
                self.next_mark = f"{read_at}|{self._fingerprint()}"
        self.note = "; ".join(notes)
        return jobs

    def _plan(
        self, entries: list[tuple[str, str]]
    ) -> tuple[list[tuple[str, str]], list[str]]:
        """(the entries to read this run, notes for the digest), and the
        board's count and any cut at `max_pages` on `total` and `truncated`."""
        cap = self.max_pages
        in_scope = self._select(entries)
        window, undated = self._window(in_scope)
        wanted = window[:cap]
        self.total = len(in_scope)
        notes: list[str] = []
        if len(window) > len(wanted):
            order = "newest " if self.incremental else "first "
            self.truncated = (
                f"read the {order}{len(wanted)} of {len(window)} urls in scope "
                f"(max_pages {cap})"
            )
        if self.incremental and undated:
            notes.append(
                f"{undated} urls carry no lastmod, so incremental cannot tell "
                "whether they changed and reads them every run"
            )
        return wanted, notes

    async def _read_pages(
        self, wanted: list[tuple[str, str]], notes: list[str]
    ) -> tuple[list[Job], int]:
        """(the postings read, how many pages failed in a way that may pass).

        A page that is gone (404, 410) or carries no markup is lost for good
        and costs nothing more; any other failure is counted, because an
        incremental read must come back for it."""
        jobs: list[Job] = []
        hits = misses = failures = unreachable = 0
        fetched_any = False

        for url, lastmod in wanted:
            cached = await self._cached(url, lastmod)
            if cached is not None:
                jobs.append(cached)
                hits += 1
                continue
            # Space out only the requests that actually go out, so a run that
            # is entirely cache hits costs no wall-clock at all.
            if fetched_any:
                gap = max(self.delay, await self._crawl_floor(url, notes))
                if gap > 0:
                    await asyncio.sleep(gap)
            fetched_any = True
            try:
                html = await self.fetcher.fetch_text(url)
            except FetchError as e:
                # One expired posting must not cost the other 168.
                log.debug("structured %s: %s", self.slug, e)
                failures += 1
                if e.detail not in _GONE:
                    unreachable += 1
                continue
            job = self._to_job(html, url)
            if job is None:
                failures += 1
                continue
            jobs.append(job)
            misses += 1
            if self.cache is not None and lastmod:
                await self.cache.put_posting(url, lastmod, job)

        log.info(
            "structured %s: %d postings (%d cached, %d fetched, %d unusable)",
            self.slug,
            len(jobs),
            hits,
            misses,
            failures,
        )
        return jobs, unreachable

    async def _crawl_floor(self, url: str, notes: list[str]) -> float:
        """The `Crawl-delay` the host of `url` asks for in its robots.txt, in
        seconds, or 0.0 when it asks for none or cannot be read.

        Asked once per host, and only when a second request is about to go
        out: a gap needs two requests. Best effort by design: a robots.txt
        that is missing, unreadable or odd never fails a source. A delay past
        `_MAX_CRAWL_DELAY` is held to it, and a note says so."""
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin in self._crawl_floors:
            return self._crawl_floors[origin]
        floor = 0.0
        try:
            parser = RobotFileParser()
            parser.parse(
                (await self.fetcher.fetch_text(f"{origin}/robots.txt")).splitlines()
            )
            parser.modified()
            asked = parser.crawl_delay(_ROBOTS_AGENT)
        except Exception as e:  # robots.txt must never fail a source
            log.debug("structured %s: no robots.txt for %s: %s", self.slug, origin, e)
            asked = None
        if asked:
            floor = min(float(asked), _MAX_CRAWL_DELAY)
            log.info(
                "structured %s: %s asks for a crawl delay of %gs",
                self.slug,
                origin,
                floor,
            )
            if float(asked) > _MAX_CRAWL_DELAY:
                notes.append(
                    f"robots.txt asks for a {asked}s crawl delay; pages are read "
                    f"{_MAX_CRAWL_DELAY:g}s apart at most"
                )
        self._crawl_floors[origin] = floor
        return floor

    # -- discovery ----------------------------------------------------------

    @staticmethod
    def _parse_sitemap(xml: str) -> list[tuple[str, str]]:
        """(url, lastmod) pairs. Falls back to a regex, because a third-party
        sitemap is exactly the kind of file that arrives subtly malformed."""
        try:
            root = ElementTree.fromstring(xml)
        except ElementTree.ParseError:
            return [(u, "") for u in _LOC.findall(xml)]
        out: list[tuple[str, str]] = []
        for node in root.iter():
            if node.tag.rsplit("}", 1)[-1] != "url":
                continue
            loc = lastmod = ""
            for child in node:
                tag = child.tag.rsplit("}", 1)[-1]
                if tag == "loc":
                    loc = (child.text or "").strip()
                elif tag == "lastmod":
                    lastmod = (child.text or "").strip()
            if loc:
                out.append((loc, lastmod))
        return out

    async def _cached(self, url: str, lastmod: str) -> Job | None:
        """The stored job, when the sitemap says the page has not moved.

        A hit still returns the posting rather than dropping it. Dropping would
        hide any posting that was fetched but never recorded -- a --dry run, or
        a crash before record_all -- for ever, since its lastmod never moves
        again and nothing would trigger a refetch.
        """
        if self.cache is None or not lastmod:
            return None
        stored = await self.cache.get_posting(url)
        if stored is None or stored[0] != lastmod:
            return None
        return stored[1]

    # -- extraction ---------------------------------------------------------

    def _to_job(self, html: str, url: str) -> Job | None:
        try:
            data = extruct.extract(html, base_url=url, syntaxes=_SYNTAXES, uniform=True)
        except Exception as e:  # a malformed page is not a crash
            log.debug("structured %s: could not parse %s: %s", self.slug, url, e)
            return None

        node = self._find_posting(data)
        if node is None:
            log.debug("structured %s: no JobPosting markup at %s", self.slug, url)
            return None

        p = _props(node)
        title = _text(p.get("title")) or _text(p.get("name"))
        if not title:
            return None

        location = self._location(p.get("jobLocation"))
        try:
            return Job(
                source=self.name,
                company=self._company(p.get("hiringOrganization")),
                title=title,
                location=location,
                url=url,
                description=(
                    self._description(p.get("description")) or self._body_text(html)
                ),
                posted=self._date(p.get("datePosted")),
                remote="remote" in f"{title} {location}".casefold(),
                raw_id=_text(_props(p.get("identifier")).get("value"))
                or _text(p.get("identifier")),
            )
        except ValueError as e:
            log.debug("structured %s: unusable posting at %s: %s", self.slug, url, e)
            return None

    @staticmethod
    def _find_posting(data: dict[str, list[Any]]) -> Any:
        """First JobPosting in any syntax. JSON-LD wins only by being first in
        _SYNTAXES; a page carrying both should agree with itself."""
        for syntax in _SYNTAXES:
            for item in data.get(syntax) or []:
                if not isinstance(item, dict):
                    continue
                if _JOB_POSTING in _typename(item.get("@type") or item.get("type")):
                    return item
        return None

    @staticmethod
    def _description(raw: object) -> str:
        """Plain text, whatever the platform did to it on the way out.

        ADNOC escapes the HTML *inside* a JSON string inside a <script>, so the
        bytes carry `&lt;p&gt;`. strip_html alone is not enough: it unescapes
        AFTER removing tags, so entity-encoded tags survive it and the LLM
        scorer would be handed literal `<p>`. Unescaping first turns them back
        into real tags for the parser to remove.
        """
        text = _text(raw)
        if not text:
            return ""
        return strip_html(unescape(text))

    def _company(self, raw: object) -> str:
        """The employer name, preferring a real one over a tenant id.

        National Grid's SuccessFactors instance reports hiringOrganization as
        "natgridProd", which then appears as the employer on all 161 postings.
        ADNOC reports "ADNOC Logistics & Services", which is more informative
        than the configured label and must survive. A single unspaced token is
        the tell: real employer names in this data always carry a space unless
        they are the label already.
        """
        name = _text(raw)
        if not name:
            return self.label
        if " " in name or name.casefold() == self.label.casefold():
            return name
        return self.label or name

    @staticmethod
    def _date(raw: object) -> str | None:
        """ISO date string, or None. Job._parse_date handles real ISO itself;
        this only has to rescue the Java form SuccessFactors uses."""
        text = _text(raw)
        if not text:
            return None
        if _ISO_DATE.match(text):
            return text
        m = _JAVA_DATE.search(text)
        if not m:
            return None
        month = _MONTHS[m.group(1).casefold()]
        return f"{int(m.group(3)):04d}-{month:02d}-{int(m.group(2)):02d}"

    @staticmethod
    def _body_text(html: str) -> str:
        """Readable page text, for markup that declares a JobPosting but leaves
        the description empty.

        TAQA's Harbour ATS does exactly that: title and dates in the JSON-LD,
        `"description": null`, and the real job text only in the HTML body. An
        empty description scores zero on keywords and is dropped at the
        prefilter, so the source would look configured and return nothing.
        """
        stripped = _SCRIPT_STYLE.sub(" ", html)
        return strip_html(stripped)[:_MAX_BODY_CHARS]

    @staticmethod
    def _location(raw: object) -> str:
        """City and country out of a nested Place > PostalAddress."""
        place = _props(raw)
        address = _props(place.get("address")) or place
        parts = [
            _text(address.get(k))
            for k in ("addressLocality", "addressRegion", "addressCountry")
        ]
        joined = ", ".join(p for p in parts if p)
        # SuccessFactors publishes nothing but streetAddress, and puts a bare
        # country code in it ("SA"). Poor, but an empty location is worse: the
        # profile's location_penalty punishes every posting that has none.
        fallback = (
            _text(address.get("streetAddress"))
            or _text(place.get("name"))
            or _text(raw)
        )
        return joined or _expand_country(fallback)
