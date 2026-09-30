"""Structured facts an LLM extracts from a posting, plus the quote guard.

A model asked "what does this role require" will happily invent a plausible
answer when the posting is silent - a fabricated "5 years minimum" reads just
as confidently as a real one, and nothing downstream can tell the difference
from the value alone. So every fact here carries the verbatim text it was
read from, and `verify_facts` is the one place that checks the quote is
actually in the posting before the value is allowed to survive. A fact that
fails that check is downgraded to "not stated" rather than trusted, because
an unverifiable fact must never be allowed to decide a verdict.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from rolescan.models import _REASON_CHARS, BarKind, Job, JobField, Level, _norm

__all__ = [
    "FieldFact",
    "HardBar",
    "LevelFact",
    "PostingFacts",
    "StudentFact",
    "YearsFact",
    "verify_facts",
]


def _normalise(text: str) -> str:
    return " ".join(text.split()).casefold()


class LevelFact(BaseModel):
    """The career level the posting targets, with the text that says so."""

    model_config = ConfigDict(extra="forbid")

    value: Level = Field(
        default=Level.not_stated,
        description=(
            "Career level the posting targets. not_stated if the posting "
            "does not say."
        ),
    )
    quote: str = Field(
        default="",
        description=(
            "Text copied verbatim from the posting that states this level. "
            "Empty if not stated."
        ),
    )


class YearsFact(BaseModel):
    """Minimum years of experience the posting requires, if any."""

    model_config = ConfigDict(extra="forbid")

    value: int | None = Field(
        default=None,
        ge=0,
        description=(
            "The minimum years of experience the advert requires. None if "
            "not stated."
        ),
    )
    quote: str = Field(
        default="",
        description=(
            "Text copied verbatim from the posting that states the years "
            "required."
        ),
    )


class StudentFact(BaseModel):
    """Whether the posting restricts the role to current students."""

    model_config = ConfigDict(extra="forbid")

    value: bool | None = Field(
        default=None,
        description=(
            "True only if the posting says the role is open only to current "
            "students. None if not stated."
        ),
    )
    quote: str = Field(
        default="",
        description=(
            "Text copied verbatim from the posting that states this "
            "restriction."
        ),
    )


class FieldFact(BaseModel):
    """The job category or specialty the posting belongs to."""

    model_config = ConfigDict(extra="forbid")

    value: JobField | None = Field(
        default=None,
        description="The job category this posting belongs to. None if unclear.",
    )
    quote: str = Field(
        default="",
        description=(
            "Text copied verbatim from the posting that supports this "
            "category."
        ),
    )


class HardBar(BaseModel):
    """A structural eligibility bar found in the posting, e.g. nationality."""

    model_config = ConfigDict(extra="forbid")

    kind: BarKind = Field(description="The kind of structural bar this is.")
    quote: str = Field(
        description="Text copied verbatim from the posting that states this bar."
    )


class PostingFacts(BaseModel):
    """Everything an LLM extracts from one posting, before any rule is applied.

    This schema constrains sampling on both backends: the Anthropic backend
    passes it as a structured output, descriptions included, so they read as
    instructions there. Ollama takes `model_json_schema()` as a GRAMMAR and
    enforces only the shape, ignoring the prose - see `FitVerdict` in
    `rolescan.models` for the history behind that split. Anything the model
    MUST know still belongs in the shared system prompt, not only here.
    """

    model_config = ConfigDict(extra="forbid")

    level: LevelFact = Field(
        default_factory=LevelFact,
        description="The career level this posting targets.",
    )
    years_required: YearsFact = Field(
        default_factory=YearsFact,
        description="The minimum years of experience this posting requires.",
    )
    student_only: StudentFact = Field(
        default_factory=StudentFact,
        description="Whether only current students may apply.",
    )
    hard_bars: list[HardBar] = Field(
        default_factory=list,
        max_length=5,
        description=(
            "Structural eligibility bars stated in the posting, such as "
            "nationality, clearance, or work authorisation requirements. "
            "Empty if none."
        ),
    )
    field: FieldFact = Field(
        default_factory=FieldFact,
        description="The job category this posting belongs to.",
    )
    fit_score: int = Field(
        ge=0,
        le=100,
        description=(
            "0-100 fit between this candidate and this role. 80+ means apply "
            "today. Below 40 means it is a poor use of their time."
        ),
    )
    reason: str = Field(
        max_length=_REASON_CHARS,
        description="ONE sentence: the single fact that decides this match.",
    )
    keywords_missing: list[str] = Field(
        default_factory=list,
        max_length=8,
        description=(
            "Skills or tools the posting asks for that are not evidenced. "
            "Empty if none."
        ),
    )

    @field_validator("hard_bars", "keywords_missing", mode="before")
    @classmethod
    def _cap_list(cls, v: object, info: ValidationInfo) -> object:
        """Keep the first N items instead of rejecting the whole set of facts.

        Mirrors `FitVerdict._cap_list`: hosted structured outputs do not
        enforce `maxItems`, so a chatty model would otherwise lose a result
        that was already paid for. The order the model gives is its own
        priority order.
        """
        limit = {"hard_bars": 5, "keywords_missing": 8}[info.field_name or ""]
        return v[:limit] if isinstance(v, list) else v

    @field_validator("reason", mode="before")
    @classmethod
    def _one_sentence(cls, v: object) -> object:
        """Trim an over-long reason instead of rejecting the whole set of facts.

        Mirrors `FitVerdict._one_sentence`: Ollama takes the schema as a
        GRAMMAR and does not enforce `maxLength`, so a chatty local model
        would otherwise fail validation and lose a result already paid for.
        """
        if not isinstance(v, str):
            return v
        text = _norm(v)
        if len(text) <= _REASON_CHARS:
            return text
        cut = text[: _REASON_CHARS - 1].rstrip()
        if " " in cut:
            cut = cut[: cut.rindex(" ")].rstrip()
        return cut + "…"


def verify_facts(facts: PostingFacts, job: Job) -> PostingFacts:
    """Downgrade any fact whose quote is not actually in the posting.

    Never mutates `facts`. A fact keeps its value only when its quote is
    non-empty and its normalised text is a substring of the normalised
    posting; otherwise the value is cleared (`Level` falls back to
    `not_stated`, everything else to `None`) and the quote is cleared to
    `""`. Hard bars that fail the check are dropped outright rather than
    kept with a blanked value, since a bar has no meaningful "unstated"
    form - it either is or is not a structural block.
    """
    haystack = _normalise(f"{job.title}\n{job.description}")

    def _verified(quote: str) -> bool:
        return bool(quote) and _normalise(quote) in haystack

    level = facts.level
    if not _verified(level.quote):
        level = LevelFact(value=Level.not_stated, quote="")

    years = facts.years_required
    if not _verified(years.quote):
        years = YearsFact(value=None, quote="")

    student = facts.student_only
    if not _verified(student.quote):
        student = StudentFact(value=None, quote="")

    field = facts.field
    if not _verified(field.quote):
        field = FieldFact(value=None, quote="")

    hard_bars = [bar for bar in facts.hard_bars if _verified(bar.quote)]

    return facts.model_copy(
        update={
            "level": level,
            "years_required": years,
            "student_only": student,
            "field": field,
            "hard_bars": hard_bars,
        }
    )
