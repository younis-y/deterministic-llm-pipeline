"""Core domain models.

Everything that crosses a module boundary is a pydantic model, so malformed
upstream payloads fail at the edge with a useful message instead of surfacing
as an AttributeError six frames deep.
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic.json_schema import SkipJsonSchema

__all__ = [
    "BarKind",
    "Confidence",
    "FitVerdict",
    "Job",
    "JobField",
    "Level",
    "ScoredJob",
    "Verdict",
    "normalise_term",
]

_WS = re.compile(r"\s+")

_REASON_CHARS = 220
"""Hard ceiling on `FitVerdict.reason`.

A twelve-role digest is read on a phone before work. At 400 characters the
model filled the space and the digest became a wall of prose nobody scanned,
so the verdict badge and the score - the two things that decide whether a
role gets read at all - were buried under paragraphs. One sentence is the
brief; this is the wall that makes the brief true."""


def _norm(text: str) -> str:
    return _WS.sub(" ", text).strip()


def normalise_term(term: str) -> str:
    """Normalise a configured term the way posting text is normalised.

    Matching is a search over `Job.blob`, which is casefolded and has its
    whitespace collapsed by `_norm`. A term that has not been through the
    same normalisation can never match: `UAE  National` with two spaces, or
    a trailing space picked up from a YAML quote, is a bar that silently
    stops working. The config loader and the matcher both come through here
    so there is exactly one answer to what a term looks like.
    """
    return _norm(term).casefold()


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


class Level(StrEnum):
    """Career stage a role targets."""

    graduate_entry = "graduate_entry"
    junior = "junior"
    mid = "mid"
    senior = "senior"
    lead_principal = "lead_principal"
    not_stated = "not_stated"


# `quant`, `product` and `consulting` (2.5.2) were `other` before: with no
# label of their own, `allowed_fields` could not let one through without also
# letting sales and operations through. `finance` (2.5.3) likewise: M&A,
# investment and equity research, trading and markets, energy and commodity
# market analysts. The docstring below reaches the hosted model as schema text
# (`FieldFact.value`), so this history is a comment.
class JobField(StrEnum):
    """Job category or specialty."""

    data_engineering = "data_engineering"
    ai_llm = "ai_llm"
    data_science = "data_science"
    analytics_bi = "analytics_bi"
    quant = "quant"
    product = "product"
    software = "software"
    consulting = "consulting"
    finance = "finance"
    other = "other"


class BarKind(StrEnum):
    """Type of eligibility barrier in a posting."""

    nationality = "nationality"
    clearance = "clearance"
    work_auth = "work_auth"
    other = "other"


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
            # Lever hands back epoch milliseconds. Converted against UTC
            # explicitly, not against whatever the machine's clock is set to:
            # a naive fromtimestamp puts a posting published at 23:30 UTC on
            # the previous day anywhere west of Greenwich, and `posted` is
            # what the recency window filters on - so the same posting was a
            # day older in New York than in London, from the same payload.
            seconds = float(v) / 1000 if v > 1e11 else float(v)
            return datetime.fromtimestamp(seconds, tz=UTC).date()
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

    This schema constrains sampling on both backends, but the `description`
    text below reaches only one of them. The Anthropic backend passes the
    model as a structured output, descriptions included, so they read as
    instructions. Ollama takes `model_json_schema()` in `format` and uses it
    as a GRAMMAR: it enforces the shape and the enum values and ignores the
    prose entirely.

    That misreading has already cost a release. Guidance that lived only in a
    field description was invisible to the local model, which then skipped 22
    of 25 benchmark postings while returning perfectly valid JSON - a defect
    that looks like a bad model rather than a missing instruction.

    So: anything the model MUST know belongs in `SYSTEM` in
    `rolescan.scoring.llm`, which both backends read. Write descriptions for
    the reader and for the hosted backend, and never as the only place a rule
    appears.
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
        max_length=_REASON_CHARS,
        description=(
            "ONE sentence: the single fact that decides this match. Concrete "
            "and specific to this role, never a summary of the posting."
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
    rule: SkipJsonSchema[str | None] = None
    """Which of `rolescan.scoring.rules.decide`'s rules fired, by its name in
    `RULE_ORDER`, or None when the fit score decided.

    Set by code, never by a model, so it is kept out of the JSON schema:
    `FitVerdict` is also judge mode's output schema on both backends, and a
    `rule` property there would invite the model to name a rule `decide`
    never ran. Optional with a default because cached verdicts written before
    it existed lack the key, and `Store.get_verdict` deletes and re-scores a
    row that fails to parse.

    It exists for the digest. On 2026-10-06, 44 of 109 scored postings were
    hidden by rules with no trace, so a wrong skip was invisible; the digest
    now lists them grouped by this name rather than re-deriving the rule from
    the wording of `reason`."""

    @field_validator("blockers", "keywords_missing", mode="before")
    @classmethod
    def _cap_list(cls, v: object, info: ValidationInfo) -> object:
        """Keep the first N items instead of rejecting the whole verdict.

        Same reasoning as `_one_sentence`: hosted structured outputs do not
        enforce `maxItems` either, and on 2026-09-30 Haiku returned nine
        missing keywords for one posting, which lost a verdict already paid
        for. The order the model gives is its own priority order.
        """
        limit = {"blockers": 5, "keywords_missing": 8}[info.field_name or ""]
        return v[:limit] if isinstance(v, list) else v

    @field_validator("reason", mode="before")
    @classmethod
    def _one_sentence(cls, v: object) -> object:
        """Trim an over-long reason instead of rejecting the whole verdict.

        Two callers need this and neither can be fixed by asking the model
        nicely. Ollama takes the schema as a GRAMMAR and does not enforce
        `maxLength`, so a chatty local model would otherwise fail validation
        and lose a verdict that was already paid for. And the verdict cache
        holds rows written when the limit was 400: `Store.get_verdict` treats
        a ValidationError as a schema change, deletes the row and re-scores,
        so tightening the limit alone would silently re-run every cached
        posting through the LLM once - the one cost the cache exists to
        avoid. Trimming keeps those rows readable.

        The cut lands on a word boundary and is marked with an ellipsis, so a
        trimmed reason reads as trimmed rather than as a model that stopped
        mid-thought.
        """
        if not isinstance(v, str):
            return v
        text = _norm(v)
        if len(text) <= _REASON_CHARS:
            return text
        cut = text[: _REASON_CHARS - 1].rstrip()
        if " " in cut:
            cut = cut[: cut.rindex(" ")].rstrip()
        return cut + "\u2026"


class ScoredJob(BaseModel):
    """A posting plus everything the pipeline worked out about it."""

    model_config = ConfigDict(extra="forbid")

    job: Job
    keyword_score: int = 0
    keyword_hits: list[str] = Field(default_factory=list)
    keyword_penalties: list[str] = Field(default_factory=list)
    blocker_hits: list[str] = Field(default_factory=list)
    """`ProfileConfig.hard_blockers` terms found in the posting text.

    Hardness is its own list, not a weight: `blockers` says how much a term
    costs, `hard_blockers` says which terms are structural. The two change on
    different timescales - weights get retuned whenever the prefilter is
    calibrated, while a passport or a clearance is a fact about the candidate
    that changes once a decade - so one number could not honestly encode both.

    A hit here is an instruction the user wrote into their config, not a hint:
    it makes this posting `blocked` regardless of what the LLM decides,
    because a model's opinion does not get to outvote a genuine structural
    bar. A weighted term that is NOT in `hard_blockers` still lands in
    keyword_penalties and still costs its weight in keyword_score, but never
    here. Neither does the location-mismatch flag.

    Because this list can delete a role from the digest outright, its terms
    are matched on word boundaries rather than as bare substrings: see
    `rolescan.scoring.keyword`."""
    fit: FitVerdict | None = None
    llm_cached: bool = False
    deferred: str = ""
    """Why the intended judge never saw this posting, or "" (2.5.7).

    `"llm_ceiling"`: `max_calls_per_run` was spent before its turn.
    `"digest_cap"`: it cleared `min_report_score` but fell past
    `output.max_roles`. `"thin"`: it has no description to judge (held back for
    `output.thin_unread_after` runs, then listed once as unread and recorded).

    Set by the stage that skipped it, read by the recorder: a deferred posting
    is never written to `seen`, so it comes round again next run. Until 2.5.7
    all three were recorded with everything else and could never surface
    again (26 Sep: 382 of 982 candidates; 5-6 Oct: 15-17 reportable roles a
    day past the cap)."""
    hidden_as: str = ""
    """Which "Hidden by your rules" group a prefilter reject is listed under
    (2.5.7): `"blockers"` or `"gate"`, or "".

    Set by `_rule_hidden` on the copy of a reject it lists, so the digest
    need not recompute weights to say why. "" means the group derives from
    `fit` and `blocker_hits` as before."""

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
