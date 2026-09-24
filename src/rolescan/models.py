"""Core domain models.

Everything that crosses a module boundary is a pydantic model, so malformed
upstream payloads fail at the edge with a useful message instead of surfacing
as an AttributeError six frames deep.
"""

from __future__ import annotations

import hashlib
import re
from datetime import date, datetime
from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "CVVariant",
    "Confidence",
    "FitVerdict",
    "Job",
    "ScoredJob",
    "Verdict",
]

_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", text).strip()


class Verdict(StrEnum):
    """What the pipeline decided to do with a posting."""

    APPLY = "apply"
    CONSIDER = "consider"
    SKIP = "skip"
    BLOCKED = "blocked"
    """Structurally ineligible: nationality gate, clearance, visa."""


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CVVariant(StrEnum):
    """The tailored CV variants that exist in the user's project."""

    AI_LLM = "CV_AI-LLM-Engineering"
    QUANT = "CV_Quant-Trading"
    DATA_SCIENCE = "CV_DataScience-Gulf"
    ENERGY = "CV_EnergySystems-Modelling"
    RESEARCH = "CV_Research-DeepLearning"
    CONSULTING = "CV_Consulting-Analytics"


class Job(BaseModel):
    """A posting as fetched from a source, before any scoring."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    source: str
    company: str
    title: str
    location: str = ""
    url: str
    description: str = ""
    posted: date | None = None
    remote: bool = False
    raw_id: str = ""

    @field_validator("title", "company", "location", "description", mode="before")
    @classmethod
    def _clean(cls, v: object) -> str:
        return _norm(str(v)) if v is not None else ""

    @field_validator("posted", mode="before")
    @classmethod
    def _parse_date(cls, v: object) -> date | None:
        """Accept ISO strings, epoch millis, datetimes, or nothing."""
        if v is None or v == "":
            return None
        if isinstance(v, date) and not isinstance(v, datetime):
            return v
        if isinstance(v, datetime):
            return v.date()
        if isinstance(v, int | float):
            # Lever hands back epoch milliseconds.
            seconds = float(v) / 1000 if v > 1e11 else float(v)
            return datetime.fromtimestamp(seconds).date()
        text = str(v)[:10]
        try:
            return date.fromisoformat(text)
        except ValueError:
            return None

    @model_validator(mode="after")
    def _require_identity(self) -> Self:
        if not self.title or not self.url:
            msg = "job needs at least a title and a url"
            raise ValueError(msg)
        return self

    @property
    def uid(self) -> str:
        """Stable identity across runs and across sources.

        Deliberately excludes the URL: the same role reposted with a new
        requisition id should not read as new. Company plus normalised title
        plus location is the tightest key that survives a repost.
        """
        key = f"{self.company}|{self.title}|{self.location}".casefold()
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    @property
    def content_hash(self) -> str:
        """Identity of the *text*, used to cache LLM verdicts.

        If a description is edited the cached verdict is correctly invalidated,
        while an unchanged repost reuses the verdict and costs nothing.
        """
        key = f"{self.title}|{self.description}".casefold()
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    @property
    def blob(self) -> str:
        return f"{self.title}\n{self.description}".casefold()


class FitVerdict(BaseModel):
    """The LLM's judgement on one posting.

    This is the schema handed to the Claude API as a structured output, so the
    field descriptions are load-bearing: they are the only instructions the
    model gets about what each field means.
    """

    model_config = ConfigDict(extra="forbid")

    fit_score: Annotated[int, Field(ge=0, le=100)] = Field(
        description=(
            "0-100 fit between this candidate and this role. 80+ means apply "
            "today. Below 40 means it is a poor use of their time."
        )
    )
    verdict: Verdict = Field(
        description=(
            "apply, consider, skip, or blocked. Use 'blocked' only for hard "
            "structural bars such as a nationality requirement the candidate "
            "cannot meet, a security clearance, or a visa they do not hold."
        )
    )
    confidence: Confidence = Field(
        description="How sure you are, given how much detail the posting gave."
    )
    reason: str = Field(
        max_length=400,
        description="One or two sentences. Concrete and specific to this role.",
    )
    cv_variant: CVVariant = Field(
        description="Which of the candidate's CV variants to submit."
    )
    tailoring: list[str] = Field(
        default_factory=list,
        max_length=5,
        description=(
            "Specific edits to that CV for this role. Each item names what to "
            "change and why. Empty if the CV already fits as-is."
        ),
    )
    blockers: list[str] = Field(
        default_factory=list,
        max_length=5,
        description=(
            "Hard eligibility bars found in the posting text, quoted briefly. "
            "Empty if none."
        ),
    )
    keywords_missing: list[str] = Field(
        default_factory=list,
        max_length=8,
        description=(
            "Skills or tools the posting asks for that the candidate's CV does "
            "not evidence. Drives what to learn next."
        ),
    )


class ScoredJob(BaseModel):
    """A posting plus everything the pipeline worked out about it."""

    model_config = ConfigDict(extra="forbid")

    job: Job
    keyword_score: int = 0
    keyword_hits: list[str] = Field(default_factory=list)
    keyword_penalties: list[str] = Field(default_factory=list)
    blocker_hits: list[str] = Field(default_factory=list)
    """Configured `blockers` terms found in the posting text - a subset of
    keyword_penalties that excludes the location-mismatch flag, which is a
    preference, not a hard bar. A hit here is an instruction the user wrote
    into their config, not a hint: it makes this posting `blocked` regardless
    of what the LLM decides, because a model's opinion does not get to
    outvote it."""
    fit: FitVerdict | None = None
    llm_cached: bool = False

    @property
    def score(self) -> int:
        """LLM score when present, else the keyword score clamped to 0-100."""
        if self.fit is not None:
            return self.fit.fit_score
        return max(0, min(100, self.keyword_score))

    @property
    def verdict(self) -> Verdict:
        """The user's configured blockers are a floor the LLM cannot lift:
        a real blocker hit is always `blocked`, even when an LLM verdict
        scored this posting highly. Short of that floor, the LLM's verdict
        wins when there is one; otherwise this falls back to the keyword
        score alone."""
        if self.blocker_hits:
            return Verdict.BLOCKED
        if self.fit is not None:
            return self.fit.verdict
        return Verdict.CONSIDER if self.keyword_score > 0 else Verdict.SKIP

    @property
    def is_blocked(self) -> bool:
        return self.verdict is Verdict.BLOCKED

    def sort_key(self) -> tuple[int, int]:
        """Blocked roles sink regardless of score."""
        return (0 if self.is_blocked else 1, self.score)
