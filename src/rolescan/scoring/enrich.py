"""The enrichment hook: extra work only passing postings need.

Facts mode's job is to decide whether a posting is worth the reader's time -
that decision is `rolescan.scoring.rules.decide`, and it is the same for
every caller. What a caller does ONCE a posting has cleared that bar can
differ enormously, and most of it is a second, more expensive model call: a
private caller might produce a supporting document, draft an outreach note,
or run a domain-specific check that would be wasted on the roles the reader
was never going to pursue.

An `Enricher` is that second call. It runs after `decide`, only for postings
that already cleared `min_report_score` and only for a verdict the reader
will act on (apply or consider) - never for `skip` or `blocked`, and never
for a posting that would not have reached the digest anyway. It takes the
plain `FitVerdict` and returns it unchanged, or a subclass carrying whatever
extra fields the caller's private context can supply.

Registration mirrors `rolescan.scoring.judges`: subclass, decorate with
`@register_enricher`, or ship one from another package under the
`rolescan.enrichers` entry-point group. Core ships none - an enricher is,
by construction, private context this library has no business knowing about,
which is why the seam exists at all rather than a fixed list of extra fields
on `FitVerdict` itself.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from importlib.metadata import entry_points
from typing import ClassVar

from rolescan.config import LLMConfig
from rolescan.models import FitVerdict, Job

__all__ = [
    "Enricher",
    "available_enrichers",
    "get_enricher",
    "register_enricher",
    "unusable_enricher_reason",
]

log = logging.getLogger(__name__)

_REGISTRY: dict[str, type[Enricher]] = {}


class Enricher(ABC):
    """Runs extra work for one posting, after it has already passed the gate."""

    name: ClassVar[str] = ""
    #: The shape `enrich` returns. A plain `FitVerdict` by default; a subclass
    #: that adds fields overrides this so the verdict cache is read back as
    #: that subclass rather than refusing the extra keys and re-scoring.
    verdict_model: ClassVar[type[FitVerdict]] = FitVerdict

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    async def enrich(self, job: Job, verdict: FitVerdict) -> FitVerdict:
        """Return `verdict` unchanged, or a `verdict_model` built from it.

        Raise on failure. The caller (`FitScorer`) logs it and keeps the
        plain verdict rather than dropping the posting - enrichment is
        additive, never load-bearing for whether a posting is reported."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.cfg.model}>"


def register_enricher(cls: type[Enricher]) -> type[Enricher]:
    if not cls.name:
        msg = f"{cls.__name__} must set a class-level `name`"
        raise ValueError(msg)
    _REGISTRY[cls.name] = cls
    return cls


def load_plugins() -> None:
    """Pull in any third-party enrichers advertising the entry-point group."""
    try:
        eps = entry_points(group="rolescan.enrichers")
    except Exception:  # pragma: no cover - importlib differences across runtimes
        return
    for ep in eps:
        try:
            obj = ep.load()
        except Exception as e:  # pragma: no cover - a broken plugin is not fatal
            log.warning("could not load enricher plugin %s: %s", ep.name, e)
            continue
        if isinstance(obj, type) and issubclass(obj, Enricher):
            register_enricher(obj)


def available_enrichers() -> dict[str, type[Enricher]]:
    return dict(_REGISTRY)


def get_enricher(cfg: LLMConfig) -> Enricher | None:
    """The configured enricher, or None when `cfg.enricher` is empty.

    Raises `ValueError` for a name that is not registered, listing what is -
    the same "fail fast, name the known set" contract as `judges.get_judge`,
    since a misspelled enricher name is exactly the kind of thing that should
    surface before a scan starts rather than as a silent no-op.
    """
    if not cfg.enricher:
        return None
    try:
        cls = _REGISTRY[cfg.enricher]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "none"
        msg = f"unknown enricher {cfg.enricher!r}. Registered: {known}"
        raise ValueError(msg) from None
    return cls(cfg)


def unusable_enricher_reason(cfg: LLMConfig) -> str:
    """Why the configured enricher cannot be used, or "" if it can (or none
    is configured).

    A static, no-network check - unlike `judges.unusable_backend_reason`, an
    enricher has no preflight of its own, since it does no work until a
    posting has already cleared the gate. Checked before anything is
    fetched all the same, and reported the same way a bad backend is (see
    `rolescan.pipeline._preflight`), so a misspelled `llm.enricher` is a line
    in the digest up front rather than a warning buried once per posting deep
    into a scan.
    """
    if not cfg.enricher or cfg.enricher in _REGISTRY:
        return ""
    known = ", ".join(sorted(_REGISTRY)) or "none"
    return (
        f"llm.enricher {cfg.enricher!r} is not a registered enricher "
        f"(registered: {known})"
    )


load_plugins()
