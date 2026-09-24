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
"""

from __future__ import annotations

import asyncio
import logging
import re
from html import unescape
from typing import Any
from xml.etree import ElementTree

import extruct

from rolescan.http import FetchError
from rolescan.models import Job
from rolescan.sources.base import (
    ProbeResult,
    ProbeStatus,
    Source,
    register,
    strip_html,
)

__all__ = ["Structured"]

log = logging.getLogger(__name__)

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")

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


@register
class Structured(Source):
    """Any careers site that publishes schema.org JobPosting and a sitemap."""

    name = "structured"
    slug_hint = (
        "sitemap: <careers host>/sitemap.xml, url_pattern: a substring or regex "
        "matching job detail URLs, e.g. /job/"
    )

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

    def _select(self, entries: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """The sitemap entries this source will actually fetch."""
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
        return wanted[: self.max_pages]

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

        entries = self._parse_sitemap(xml)
        if not entries:
            return ProbeResult(ProbeStatus.EMPTY, detail="sitemap lists no urls")

        matched = [u for u, _ in self._select(entries)]
        if not matched:
            return ProbeResult(
                ProbeStatus.UNKNOWN,
                detail=(
                    f"{len(entries)} urls in the sitemap, none match "
                    f"url_pattern {self.url_pattern!r}, e.g. {entries[0][0][:48]}"
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

        entries = self._parse_sitemap(await self.fetcher.fetch_text(self.sitemap_url))
        wanted = self._select(entries)

        jobs: list[Job] = []
        hits = misses = failures = 0
        fetched_any = False

        for url, lastmod in wanted:
            cached = await self._cached(url, lastmod)
            if cached is not None:
                jobs.append(cached)
                hits += 1
                continue
            # Space out only the requests that actually go out, so a run that
            # is entirely cache hits costs no wall-clock at all.
            if fetched_any and self.delay > 0:
                await asyncio.sleep(self.delay)
            fetched_any = True
            try:
                html = await self.fetcher.fetch_text(url)
            except FetchError as e:
                # One expired posting must not cost the other 168.
                log.debug("structured %s: %s", self.slug, e)
                failures += 1
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
        return jobs

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
