"""Source plugin contract and registry.

A source is any object that can turn a configured slug into a list of Jobs.
Adding one means subclassing Source and decorating it with @register, or
shipping it from another package under the `rolescan.sources` entry-point group.
Neither route requires touching pipeline code.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from html import unescape
from html.parser import HTMLParser
from importlib.metadata import entry_points
from typing import Any, ClassVar, Protocol

from rolescan.config import SourceEntry
from rolescan.http import Fetcher, FetchError
from rolescan.models import Job

__all__ = [
    "PostingCache",
    "ProbeResult",
    "ProbeStatus",
    "Source",
    "SourceSkipped",
    "available",
    "get_source",
    "load_plugins",
    "register",
    "strip_html",
]

log = logging.getLogger(__name__)

_REGISTRY: dict[str, type[Source]] = {}
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


class PostingCache(Protocol):
    """What a source needs from the store to skip an unchanged detail page.

    Narrower than Store on purpose: a source has no business reading the seen
    table or the verdict cache, and a test can satisfy this with anything.
    """

    async def get_posting(self, url: str) -> tuple[str, Job] | None: ...

    async def put_posting(self, url: str, lastmod: str, job: Job) -> None: ...


class SourceSkipped(RuntimeError):  # noqa: N818 - a signal, not an error
    """The source declined to run: no credentials, unsupported region.

    Distinct from a failure. A skipped source proves nothing about the slug,
    and must never be reported as a working one.
    """


class ProbeStatus(StrEnum):
    """What `rolescan discover` can honestly conclude about a configured slug.

    The three-way split exists because job board APIs disagree about how to
    signal "no such company". Greenhouse, Lever, Ashby and Workable all 404.
    SmartRecruiters returns HTTP 200 with totalFound=0, exactly as it does for
    a real employer with no current vacancies, so a zero count there is
    genuinely ambiguous and saying OK would be a lie.
    """

    OK = "OK"
    """Returned at least one posting. The slug is real."""
    EMPTY = "EMPTY"
    """Reachable and returned zero, on an API that 404s unknown slugs. The
    board exists; it just has no openings right now."""
    UNKNOWN = "UNKNOWN"
    """Returned zero on an API that cannot distinguish a bad slug from an
    empty board. Verify by hand."""
    SKIPPED = "SKIPPED"
    """Not attempted. Proves nothing."""
    FAIL = "FAIL"
    """Errored. The slug, or the config around it, is wrong."""


@dataclass(slots=True)
class ProbeResult:
    status: ProbeStatus
    count: int = 0
    detail: str = ""

    @property
    def trustworthy(self) -> bool:
        """True only when the probe actually proved the slug is good."""
        return self.status in {ProbeStatus.OK, ProbeStatus.EMPTY}


class Source(ABC):
    """Base class for a job board integration."""

    name: ClassVar[str] = ""
    #: Human-readable hint shown by `rolescan discover` when a slug fails.
    slug_hint: ClassVar[str] = ""
    #: Whether this source's `posted` date means "this advert went up", which
    #: is what an age cutoff assumes. True for aggregators. False for an
    #: employer's own ATS board, where the date is when the requisition was
    #: opened and the listing's presence is itself the freshness signal: Jane
    #: Street's board carries a live req dated 2019, and a 90-day cutoff threw
    #: away 145 of its 228 openings on the grounds that they were old rather
    #: than filled.
    dates_are_freshness: ClassVar[bool] = True
    #: Set True when the API returns 200 with an empty collection for a slug
    #: that does not exist, so a zero count cannot be trusted as verification.
    ambiguous_when_empty: ClassVar[bool] = False

    def __init__(
        self,
        entry: SourceEntry,
        fetcher: Fetcher,
        cache: PostingCache | None = None,
    ) -> None:
        self.entry = entry
        self.fetcher = fetcher
        #: Optional per-URL posting cache. Only the sources that fetch one page
        #: per posting have anything to gain from it; the JSON APIs return a
        #: whole board in one request and ignore it.
        self.cache = cache

    async def probe(self) -> ProbeResult:
        """Test this slug and classify the outcome honestly."""
        try:
            jobs = await self.fetch()
        except SourceSkipped as e:
            return ProbeResult(ProbeStatus.SKIPPED, detail=str(e))
        except FetchError as e:
            return ProbeResult(ProbeStatus.FAIL, count=-1, detail=e.detail[:110])
        except Exception as e:
            return ProbeResult(
                ProbeStatus.FAIL, count=-1, detail=f"{type(e).__name__}: {e}"[:110]
            )
        if jobs:
            return ProbeResult(ProbeStatus.OK, count=len(jobs))
        if self.ambiguous_when_empty:
            return ProbeResult(
                ProbeStatus.UNKNOWN,
                detail="200 with zero rows; this API does not 404 unknown slugs",
            )
        return ProbeResult(ProbeStatus.EMPTY, detail="board reachable, no openings")

    @property
    def slug(self) -> str:
        return self.entry.slug

    @property
    def label(self) -> str:
        return self.entry.label

    @abstractmethod
    async def fetch(self) -> list[Job]: ...

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.slug}>"


def register(cls: type[Source]) -> type[Source]:
    """Class decorator that adds a source to the registry."""
    if not cls.name:
        msg = f"{cls.__name__} must set a class-level `name`"
        raise ValueError(msg)
    if cls.name in _REGISTRY and _REGISTRY[cls.name] is not cls:
        log.warning("source %r re-registered by %s", cls.name, cls.__name__)
    _REGISTRY[cls.name] = cls
    return cls


def load_plugins() -> None:
    """Pull in any third-party sources advertising the entry-point group."""
    try:
        eps = entry_points(group="rolescan.sources")
    except Exception:  # pragma: no cover - importlib differences across runtimes
        return
    for ep in eps:
        try:
            obj = ep.load()
        except Exception as e:  # pragma: no cover - a broken plugin is not fatal
            log.warning("could not load source plugin %s: %s", ep.name, e)
            continue
        if isinstance(obj, type) and issubclass(obj, Source):
            register(obj)


def available() -> dict[str, type[Source]]:
    return dict(_REGISTRY)


def get_source(
    entry: SourceEntry, fetcher: Fetcher, cache: PostingCache | None = None
) -> Source:
    try:
        cls = _REGISTRY[entry.kind]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "none"
        msg = f"unknown source kind {entry.kind!r}. Registered: {known}"
        raise KeyError(msg) from None
    return cls(entry, fetcher, cache)


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


class _Text(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],  # noqa: ARG002  HTMLParser's signature
    ) -> None:
        if tag in {"br", "p", "li", "div", "tr"}:
            self.parts.append("\n")


def strip_html(raw: object) -> str:
    """HTML to readable text.

    Job descriptions arrive as HTML from every ATS. Feeding raw markup to the
    LLM wastes a third of the tokens on angle brackets, and keyword matching
    against markup produces false hits on things like class names.
    """
    if raw is None:
        return ""
    text = raw if isinstance(raw, str) else str(raw)
    parser = _Text()
    try:
        parser.feed(text)
        parser.close()
        out = "".join(parser.parts)
    except Exception:
        out = _TAG.sub(" ", text)
    return _WS.sub(" ", unescape(out)).strip()


def first_str(payload: dict[str, Any], *keys: str) -> str:
    """Return the first key present and non-empty, as a string."""
    for k in keys:
        v = payload.get(k)
        if v:
            return str(v)
    return ""
