"""Structured facts an LLM extracts from a posting, plus the quote guard.

A model asked "what does this role require" will happily invent a plausible
answer when the posting is silent - a fabricated "5 years minimum" reads just
as confidently as a real one, and nothing downstream can tell the difference
from the value alone. So every fact here carries the verbatim text it was
read from, and `verify_facts` is the one place that checks the quote is
actually in the posting before the value is allowed to survive. A fact that
fails that check is downgraded to "not stated" rather than trusted, because
an unverifiable fact must never be allowed to decide a verdict.

Two rules keep that check honest rather than merely convenient:

- Title and description are checked as two separate haystacks, never joined
  with a separator first. A joined string lets a quote splice the title's
  tail onto the description's head (e.g. "Engineer Must relocate" from title
  "Senior Engineer" plus description "Must relocate immediately.") and
  wrongly verify, because the separator collapses under whitespace
  normalisation and the two fields read as one contiguous string.
- Typographic look-alikes fold to their plain-ASCII form before comparison
  (curly quotes to straight, en/em dash and minus sign to hyphen, non-breaking
  space to space) on both the quote and the haystack, so a real quote is not
  rejected only because the posting or the model used a "smart" character
  where the other used a plain one. This folds equivalent characters only -
  it cannot make invented text verify.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
)
from pydantic.json_schema import SkipJsonSchema

from rolescan.models import _REASON_CHARS, BarKind, Job, JobField, Level, _norm

_TYPO_FOLD = str.maketrans(
    {
        chr(0x2018): "'",  # left single quotation mark
        chr(0x2019): "'",  # right single quotation mark
        chr(0x201C): '"',  # left double quotation mark
        chr(0x201D): '"',  # right double quotation mark
        chr(0x2013): "-",  # en dash
        chr(0x2014): "-",  # em dash
        chr(0x2212): "-",  # minus sign
        chr(0x00A0): " ",  # non-breaking space
    }
)

__all__ = [
    "QUOTE_CHARS",
    "FieldFact",
    "GraduationYearFact",
    "HardBar",
    "LevelFact",
    "LevelSource",
    "PostingFacts",
    "StudentFact",
    "YearsFact",
    "field_from_text",
    "graduates_eligible",
    "level_from_text",
    "resolve_field",
    "resolve_level",
    "resolve_student",
    "resolve_years",
    "verify_facts",
    "years_required_stated",
]

#: The longest quote any fact may carry. A local model asked for "the text
#: that states it" will sometimes copy a whole paragraph, and five of those in
#: one answer ran its JSON past the output budget mid-string ("EOF while
#: parsing a string"), losing the posting. The schema carries this as
#: `maxLength`, which Ollama's grammar enforces at sampling time; `_trim_quote`
#: covers backends that do not enforce it.
QUOTE_CHARS = 200


def _trim_quote(v: object) -> object:
    """Keep the first `QUOTE_CHARS` characters of an over-long quote.

    A prefix of a verbatim quote is itself verbatim, so the trimmed quote still
    verifies against the posting; rejecting it would lose a whole set of facts
    over one chatty field.
    """
    if isinstance(v, str) and len(v) > QUOTE_CHARS:
        return v[:QUOTE_CHARS]
    return v


_Quote = Annotated[str, BeforeValidator(_trim_quote)]


def _normalise(text: str) -> str:
    return " ".join(text.translate(_TYPO_FOLD).split()).casefold()


#: Where a resolved level came from (2.5.0). `title`: the job title names it
#: by a level keyword (`_LEVEL_WORDS`). `text`: the model stated or quoted it,
#: wherever the quote sits - even a quote copied from the title is the model's
#: reading, not the title's (2.5.1). `none`: no level.
#: `rules.level_from_title_only` lets only `title` fire the level rule.
LevelSource = Literal["title", "text", "none"]


class LevelFact(BaseModel):
    """The career level the posting targets, with the text that says so."""

    model_config = ConfigDict(extra="forbid")

    value: Level = Field(
        default=Level.not_stated,
        description=(
            "Career level the posting targets. not_stated if the posting does not say."
        ),
    )
    quote: _Quote = Field(
        default="",
        max_length=QUOTE_CHARS,
        description=(
            "Text copied verbatim from the posting that states this level. "
            "Empty if not stated."
        ),
    )
    # Provenance, not something the model states. Left out of the JSON schema
    # (`SkipJsonSchema`), so neither backend ever asks for it, and set by
    # `resolve_level` after the quote guard. Still serialised, so cached facts
    # carry it into every later re-decision. Class docstrings and field
    # descriptions here reach the hosted model as schema text, which is why
    # this note is a comment.
    source: SkipJsonSchema[LevelSource] = Field(
        default="none",
        description=(
            "Where the level was read from: the job title, the rest of the "
            "posting, or nowhere. Set by `resolve_level`, never by the model."
        ),
    )


class YearsFact(BaseModel):
    """Minimum years of experience the posting requires, if any."""

    model_config = ConfigDict(extra="forbid")

    value: int | None = Field(
        default=None,
        ge=0,
        description=(
            "The minimum years of experience the advert requires. None if not stated."
        ),
    )
    quote: _Quote = Field(
        default="",
        max_length=QUOTE_CHARS,
        description=(
            "Text copied verbatim from the posting that states the years required."
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
    quote: _Quote = Field(
        default="",
        max_length=QUOTE_CHARS,
        description=(
            "Text copied verbatim from the posting that states this restriction."
        ),
    )


# 2.5.0. `student_only` alone cannot tell an internship for 2027 graduates
# from one for 2028 graduates; this can. The docstring below reaches the hosted
# model as schema text, so it says only what the model needs.
class GraduationYearFact(BaseModel):
    """The earliest graduation year the posting requires of applicants.

    "graduating in 2028" is 2028; "graduating 2027 or 2028" is 2027; "class
    of 2027" is 2027.
    """

    model_config = ConfigDict(extra="forbid")

    value: int | None = Field(
        default=None,
        description=(
            "The earliest graduation year the advert requires of applicants. "
            "None if not stated."
        ),
    )
    quote: _Quote = Field(
        default="",
        max_length=QUOTE_CHARS,
        description=(
            "Text copied verbatim from the posting that states the graduation year."
        ),
    )


class FieldFact(BaseModel):
    """The job category or specialty the posting belongs to."""

    model_config = ConfigDict(extra="forbid")

    value: JobField | None = Field(
        default=None,
        description="The job category this posting belongs to. None if unclear.",
    )
    quote: _Quote = Field(
        default="",
        max_length=QUOTE_CHARS,
        description=(
            "Text copied verbatim from the posting that supports this category."
        ),
    )


class HardBar(BaseModel):
    """A structural eligibility bar found in the posting, e.g. nationality."""

    model_config = ConfigDict(extra="forbid")

    kind: BarKind = Field(description="The kind of structural bar this is.")
    quote: _Quote = Field(
        max_length=QUOTE_CHARS,
        description="Text copied verbatim from the posting that states this bar.",
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
    graduation_year: GraduationYearFact = Field(
        default_factory=GraduationYearFact,
        description="The earliest graduation year applicants must have.",
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
            "0-100 match between the candidate's skills and domain and this "
            "role's work. Level and eligibility are judged separately; do "
            "not include them."
        ),
    )
    reason: str = Field(
        max_length=_REASON_CHARS,
        description="One sentence on the skills/domain match.",
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


def _year_in_quote(year: int | None, quote: str) -> bool:
    """Whether `year` is stated in `quote` as a whole number (2.5.1).

    `None` (no year) passes: there is no value for the quote to contradict.
    Digit boundaries stop 27 from matching inside "2027".
    """
    if year is None:
        return True
    return re.search(rf"(?<!\d){year}(?!\d)", quote) is not None


# Words a structural bar's own quote must contain (2.5.2). The quote guard
# proves a quote is in the posting, not that it says what the bar claims: one
# evaluation posting was blocked as a nationality-or-clearance bar on a
# verbatim quote about "business policies and procedure", in an advert that
# never mentions either. A block hides the posting for good, so these two
# kinds - the only ones that block - must name what they are. "Right to work"
# and "visa" are work authorisation, never nationality, and a country named as
# a place ("Saudi Arabia") is not a nationality requirement. "UAEN" (UAE
# National) and "Family Book" (the Emirati citizenship record) are how Gulf
# adverts often write the bar itself. SC and DV are matched in capitals only,
# so the "Sc" of "MSc" is not a clearance.
_BAR_WORDS: dict[BarKind, re.Pattern[str]] = {
    BarKind.nationality: re.compile(
        r"\b(?:nationals?|nationality|nationalities|citizens?|citizenship"
        r"|passports?|emiratis?|uaen|family\s+book"
        r"|(?:saudi|emirati|omani|qatari|kuwaiti|bahraini|national)i?[sz]ation)\b",
        re.IGNORECASE,
    ),
    BarKind.clearance: re.compile(
        r"\b(?:clearances?|cleared|vetting|vetted|bpss|nppv\d?|uksv"
        r"|security\s+checks?|top\s+secret|ts/sci|polygraph|(?-i:SC|DV))\b",
        re.IGNORECASE,
    ),
}


def _bar_supported(bar: HardBar) -> bool:
    """Whether a bar's quote names its own kind (2.5.2).

    Only the kinds in `_BAR_WORDS` are checked; any other kind passes, since
    its quote has no fixed vocabulary to check against.
    """
    pattern = _BAR_WORDS.get(bar.kind)
    if pattern is None:
        return True
    return pattern.search(bar.quote.translate(_TYPO_FOLD)) is not None


def verify_facts(facts: PostingFacts, job: Job) -> PostingFacts:
    """Downgrade any fact whose quote is not actually in the posting.

    Never mutates `facts`. A fact keeps its value only when its quote is
    non-empty and its normalised text is a substring of the normalised
    title OR of the normalised description - checked separately, never
    joined into one string, because joining lets a quote splice the title's
    tail onto the description's head and wrongly verify. A quote that fails
    both checks clears the value (`Level` falls back to `not_stated`,
    everything else to `None`) and clears the quote to `""`. Hard bars that
    fail the check are dropped outright rather than kept with a blanked
    value, since a bar has no meaningful "unstated" form - it either is or
    is not a structural block.

    Normalisation also folds typographic look-alikes (curly quotes, en/em
    dash, non-breaking space) to their plain form on both sides before
    comparing, so a real quote does not fail only because the posting and
    the model disagree on which quote character to use; it cannot make an
    invented quote verify.

    A graduation year must also appear in its own quote, as a whole number
    (2.5.1): a verbatim "Current students only" proves the quote exists but
    says nothing about which year, so a year the quote does not state is the
    model's invention and is downgraded like an unverified quote.

    Likewise a nationality or clearance bar must name its own kind in its
    quote (2.5.2, `_BAR_WORDS`): a verbatim sentence about something else
    proves the quote exists but not the bar, so the bar is dropped like an
    unverified one.
    """
    title = _normalise(job.title)
    description = _normalise(job.description)

    def _verified(quote: str) -> bool:
        if not quote:
            return False
        q = _normalise(quote)
        return q in title or q in description

    level = facts.level
    if not _verified(level.quote):
        level = LevelFact(value=Level.not_stated, quote="")

    years = facts.years_required
    if not _verified(years.quote):
        years = YearsFact(value=None, quote="")

    student = facts.student_only
    if not _verified(student.quote):
        student = StudentFact(value=None, quote="")

    graduation_year = facts.graduation_year
    if not _verified(graduation_year.quote) or not _year_in_quote(
        graduation_year.value, graduation_year.quote
    ):
        graduation_year = GraduationYearFact(value=None, quote="")

    field = facts.field
    if not _verified(field.quote):
        field = FieldFact(value=None, quote="")

    hard_bars = [
        bar for bar in facts.hard_bars if _verified(bar.quote) and _bar_supported(bar)
    ]

    return facts.model_copy(
        update={
            "level": level,
            "years_required": years,
            "student_only": student,
            "graduation_year": graduation_year,
            "field": field,
            "hard_bars": hard_bars,
        }
    )


# Graduate eligibility (2.5.2). The model set `student_only` on adverts that
# say, in their own words, that recent graduates may apply too ("university
# students and recent graduates", "Recent graduates or final year students",
# "Currently undertaking or recently completed Bachelor's"), so a graduate was
# skipped as if the role excluded them. These phrases name graduates as
# eligible; "graduate" alone does not ("graduate students" are postgraduate
# students, "graduate scheme" is a programme), and neither does a student-only
# line ("current students only", "returning to study", "penultimate-year
# students"). "Students and graduates" without "recent" or "final-year" is left
# out on purpose: it is employer boilerplate on placement adverts ("each year
# we recruit hundreds of graduates and students", "Career Area: Students and
# Graduates") far more often than an eligibility line.
_GRADUATES_ELIGIBLE = re.compile(
    r"\b(?:(?:recent|fresh)(?:ly)?[- ]graduat(?:es?|ed)"
    r"|graduated\s+(?:with|in|within|less\s+than|no\s+more\s+than)"
    r"|or\s+(?:have\s+|has\s+)?graduated"
    r"|recently\s+completed\s+(?:(?:a|an|your|their|the)\s+)?(?:[\w'-]+\s+){0,3}?"
    r"(?:degrees?|studies|bachelor|master|ph\.?\s?d|doctorate|msc|bsc|mba|university)"
    r"|students?\s+(?:or|/)\s+(?:recent\s+)?graduates"
    r"|final[- ]year\s+students?\s+(?:and|or|/|&)\s+(?:recent\s+)?graduates"
    r"|graduates\s+(?:or|/)\s+(?:(?:current|final[- ]year|penultimate[- ]year"
    r"|university)\s+)?students?"
    r"|graduates\s+(?:are\s+)?(?:also\s+)?(?:welcome|eligible"
    r"|(?:may|can)\s+(?:also\s+)?apply)"
    r"|graduates\s+only)",
    re.IGNORECASE,
)

#: A negation near a graduate phrase turns it into an exclusion ("Recent
#: graduates are not eligible", "not open to recent graduates"), which must
#: not clear `student_only`. Checked within the phrase's own sentence, up to
#: `_NEGATION_REACH` characters either side of it.
_NEGATION = re.compile(
    r"\b(?:not|no|never|cannot|ineligible|excluded|unable)\b|n't\b", re.IGNORECASE
)
#: The word just before a graduate phrase can exclude it too: "If you have
#: already graduated with a bachelor's degree ... you are not eligible" (seen
#: on placement adverts, with the "not" too far away to see), or peers rather
#: than applicants ("share ideas with other recent graduates").
_EXCLUDING_WORD_BEFORE = re.compile(r"\b(?:already|other|fellow)\s+$", re.IGNORECASE)
_NEGATION_REACH = 60
_SENTENCE_END = re.compile(r"[.!?;](?=\s|$)")


#: Eligibility wording that restricts a posting to current students (2.5.3).
#: Only phrases that name the applicant's own enrolment: "penultimate year",
#: "currently pursuing/enrolled/studying", a first-year master's or PhD, or
#: returning to study. "Students" alone, "final year" (often paired with
#: graduates) and "graduating in 2028" (the graduation-year rule's job) are
#: deliberately absent.
_STUDENTS_ONLY = re.compile(
    r"\b(?:penultimate[- ]year"
    r"|currently\s+(?:enrolled|pursuing|studying)"
    r"|(?:1st|first)[- ]year\s+(?:of\s+(?:a|an|your)\s+)?"
    r"(?:master'?s|msc|mba|ph\.?\s?d|postgraduate)"
    r"|returning\s+to\s+(?:university|(?:full[- ]time\s+)?stud(?:y|ies)"
    r"|(?:full[- ]time\s+)?education)"
    r"|current\s+(?:university\s+)?students\s+only)",
    re.IGNORECASE,
)


def _unnegated_sentence(pattern: re.Pattern[str], text: str) -> str | None:
    """The first un-negated match of `pattern`, quoted verbatim, or None.

    Returns the match's sentence when it fits in `QUOTE_CHARS`, else the
    phrase itself - verbatim either way, so it passes the quote guard. A
    phrase with a negation close by in its own sentence does not count.
    """
    folded = text.translate(_TYPO_FOLD)  # one character for one: offsets hold
    ends = [m.end() for m in _SENTENCE_END.finditer(folded)]
    for match in pattern.finditer(folded):
        start = max((e for e in ends if e <= match.start()), default=0)
        end = min((e for e in ends if e >= match.end()), default=len(folded))
        before = folded[max(start, match.start() - _NEGATION_REACH) : match.start()]
        after = folded[match.end() : min(end, match.end() + _NEGATION_REACH)]
        if (
            _NEGATION.search(before)
            or _NEGATION.search(after)
            or _EXCLUDING_WORD_BEFORE.search(before)
        ):
            continue
        sentence = text[start:end].strip()
        if len(sentence) <= QUOTE_CHARS:
            return sentence
        return text[match.start() : match.end()]
    return None


def graduates_eligible(text: str) -> str | None:
    """The text's own words saying graduates may apply, or None (2.5.2)."""
    return _unnegated_sentence(_GRADUATES_ELIGIBLE, text)


def students_only(text: str) -> str | None:
    """The text's own words restricting it to current students, or None (2.5.3)."""
    return _unnegated_sentence(_STUDENTS_ONLY, text)


# Years of experience an advert requires (2.5.5). The local model left this
# fact null on 12 of 13 digest adverts that state it in so many words, so
# code reads it after `verify_facts`, as `resolve_student` does for enrolment
# wording. Shapes read: a count before the word ("3+ years of experience",
# "2-4 years experience", "5 Years + experience", "4+ years in a data role",
# "3+ years as a Data Engineer", "2 years working in a similar role"), a count
# after it ("experience (3+ years)", "experience of at least 3 years") and a
# labelled count ("Seniority: 5+ years", "Years of experience: 3-5"). The
# value is the low end of a range, rounded down; "N+" and "over N" are N.
_YEARS_COUNT = (
    r"(?<![\d.])(?P<n>\d{1,2})(?:\.\d+)?\s*\+?"
    r"(?:\s*(?:-|\u2013|\u2014|to)\s*\d{1,2}(?:\.\d+)?\s*\+?)?"
)
_YEARS_WORD = r"(?:years?|yrs?)(?:\s*\+)?(?:['\u2019]s?)?"
_YEARS_LEAD = (
    r"(?:(?:a\s+)?minimum(?:\s+of)?|min\.?|at\s+least|more\s+than|over|"
    r"no\s+less\s+than|you\s+have)?\s*"
)
# Up to three describing words may sit between "years of" and "experience"
# ("hands-on data engineering experience"); a pronoun, verb, joining word or
# programme noun there means the sentence is about something else ("2 years
# and gives you hands-on experience", "2 year graduate programme, gaining
# experience", "2 years maximum experience").
_YEARS_GAP_WORD = (
    r"(?!(?:you|your|will|and|with|gives?|giving|offer(?:s|ing)?|our|we|the|an?|"
    r"to|in|on|at|for|that|which|is|are|gain|gaining|get|providing|provides?|"
    r"covers?|covering|including|max\.?|maximum|programmes?|programs?|schemes?|"
    r"apprenticeships?|rotations?|contracts?|placements?|internships?|graduate|"
    r"training)\b)[\w/&,'\u2019\-]+\s+"
)
_YEARS_TAIL = (
    rf"\s+(?:of\s+)?(?:{_YEARS_GAP_WORD}){{0,3}}?"
    r"(?:experience\b|exp\b|working\s+(?:in|with)\b|as\s+an?\b|"
    r"in\s+(?:a|an|the)\s+(?:[\w\-]+\s+){0,2}?(?:role|position)\b)"
)
_YEARS_STATED = (
    re.compile(
        rf"\b{_YEARS_LEAD}{_YEARS_COUNT}\s*{_YEARS_WORD}{_YEARS_TAIL}", re.IGNORECASE
    ),
    re.compile(
        rf"\bexperience\s*\(\s*{_YEARS_COUNT}\s*{_YEARS_WORD}\s*\)", re.IGNORECASE
    ),
    re.compile(
        rf"\bexperience\s+of\s+(?:at\s+least\s+|a\s+minimum\s+of\s+|over\s+|"
        rf"more\s+than\s+)?{_YEARS_COUNT}\s*{_YEARS_WORD}",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:seniority|experience(?:\s+level)?)\s*:\s*{_YEARS_COUNT}\s*{_YEARS_WORD}",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\byears?\s+of\s+experience\s*:\s*{_YEARS_COUNT}(?:\s*{_YEARS_WORD})?",
        re.IGNORECASE,
    ),
)
#: A clause is the text since the last of these; a colon is not one, so a
#: heading ("Preferred qualifications:") speaks for the line it introduces.
_CLAUSE_START = re.compile(r"[.;!?\n()]")
#: How far back a count's clause is read. Real adverts need a sentence; the
#: cap keeps an adversarial advert with thousands of rejected counts linear.
_YEARS_LOOKBACK = 240
#: The clause ends in a limit or a programme length, not a minimum ("up to 2
#: years", "no more than 3 years", "within the last 2 years", "your first 2
#: years", "spend 2 years as an Analyst", "gain 2 years of experience").
_YEARS_CAP = re.compile(
    r"(?:\bup\s+to|\bless\s+than|\bfewer\s+than|\bunder|\bno\s+more\s+than|"
    r"\bnot\s+more\s+than|\bmaximum(?:\s+of)?|\bmax\.?|"
    r"\bwithin(?:\s+the)?(?:\s+last|\s+past)?|\bno|\bnot|"
    r"\bfirst|\bnext|\blast|\bpast|\bprevious|\bfollowing|\bafter|"
    r"\bspen[dt]|\bserved?|\bgain(?:ing)?|\bgives?\s+you|\boffers?(?:\s+you)?|"
    r"\bproviding|\bprovides?|\bcover(?:s|ing)?)\s*$",
    re.IGNORECASE,
)
#: "Max." ends a clause by its own full stop, so it is read across one.
_YEARS_CAP_TIGHT = re.compile(r"\bmax\.?\s*$", re.IGNORECASE)
#: A preference, not a requirement, before the count in the same clause or
#: in a heading ("ideally with 6+ years", "Preferred qualifications:") ...
_YEARS_SOFT = re.compile(
    r"\b(?:ideally|preferabl[ey]|preferred|desired|nice\s+to\s+have|"
    r"good\s+to\s+have|bonus(?:\s+points)?|desirable|a\s+plus|advantage(?:ous)?)\b",
    re.IGNORECASE,
)
#: ... or shortly after it ("3+ years of experience with Spark is preferred").
#: "Preferably in banking" after the count qualifies the field, not the years.
_YEARS_SOFT_AFTER = re.compile(
    r"^[^.;!?\n]{0,60}?\b(?:preferred|preferable|a\s+plus|an?\s+advantage|"
    r"advantageous|desirable|desired|nice\s+to\s+have|bonus|beneficial|helpful|"
    r"optional|welcome|valued|not\s+(?:required|essential|a\s+requirement)|"
    r"preferably(?!\s+(?:in|within|with|from|across|at|for|on)\b))\b",
    re.IGNORECASE,
)
#: Someone other than the candidate, named just before the count ("mentored
#: by an engineer with 8+ years", "reporting to the Head of Data, who has 12
#: years", "our team members average 7 years") ...
_YEARS_OTHERS = re.compile(
    r"\b(?:led\s+by|mentored\s+by|managed\s+by|headed\s+by|report(?:s|ing)?\s+to|"
    r"who\s+ha(?:s|ve)|team\s+(?:of|members)|average|"
    r"our\s+(?:team|founders?|engineers|members|leadership|experts|consultants|"
    r"people|staff))\b",
    re.IGNORECASE,
)
#: ... unless a requirement verb follows them ("reports to the CFO and
#: requires 5+ years", "our team is looking for someone with 3+ years") ...
_YEARS_REQ_VERB = re.compile(
    r"\b(?:requires?|required|need(?:s|ed)?|looking\s+for|seeking|seeks?|wants?|"
    r"must\s+have|should\s+have|and\s+have|will\s+have|have\s+at\s+least|brings?|"
    r"with\s+at\s+least)\b",
    re.IGNORECASE,
)
#: ... or "who has" follows the candidate ("someone who has 5 years").
_YEARS_CANDIDATE = re.compile(
    r"\b(?:someone|candidates?|applicants?|person|people|individuals?|you|those)\b"
    r"[^.\n]{0,20}$",
    re.IGNORECASE,
)
#: The company as the clause's subject ("Acme has 10+ years of experience"):
#: a name in capitals, then has/brings. "Must have" and "The ideal candidate
#: has" are not one.
_YEARS_COMPANY_SUBJECT = re.compile(
    r"^\s*(?!(?i:candidates?|applicants?|you|we|our|they|the|must|should|ideal|"
    r"all|successful|strong)\b)(?:[A-Z][\w&.'-]*\s+){1,4}"
    r"(?:has|have|brings?|boasts|offers?)\s*$"
)
#: The sentence goes on about the company ("28 years of experience, we have",
#: "12 years of experience in the region, Acme is the market leader").
_YEARS_COMPANY_AFTER = re.compile(
    r"^[^.;\n]{0,40}?,?\s+(?i:we|our)\b"
    r"|^[^.;\n]{0,40}?,\s+(?:[A-Z][\w&.'-]*\s+){1,3}"
    r"(?:is|has|was|offers?|provides?|delivers?|serves?)\b"
)
#: The years are one of two routes in, the other a degree ("2+ years of ML
#: experience, or a Master's degree"); only an "or" right after the years.
_YEARS_WAIVED = re.compile(
    r"^\s*[,;]?\s*or\b[^.\n]{0,60}?\b(?:master|msc|phd|doctorate|degree|"
    r"equivalent)\b",
    re.IGNORECASE,
)
#: A career path, not a requirement ("3 years as an Associate leads to VP").
_YEARS_CAREER_PATH = re.compile(
    r"^[^.\n]{0,60}?\b(?:before\s+(?:being\s+)?promot|progress(?:ion|ing)?\s+to|"
    r"you\s+will\s+(?:progress|move|be\s+promoted)|leads?\s+to|promotion\s+to)\b",
    re.IGNORECASE,
)
#: No advert aimed at this candidate asks for more; every such figure seen in
#: 3,036 cached adverts was the company's own ("more than 25 years of
#: experience with multiple offices").
_YEARS_MAX = 14


def _years_third_party(clause: str) -> bool:
    """Whether the words just before the count name someone else."""
    near = clause[-70:]
    for t in _YEARS_OTHERS.finditer(near):
        if t.group(0).lower().startswith("who") and _YEARS_CANDIDATE.search(
            near[: t.start()]
        ):
            continue
        if _YEARS_REQ_VERB.search(near[t.end() :]):
            continue
        return True
    return False


def _years_soft_heading(text: str, start: int) -> bool:
    """Whether the line above the count is a preference heading."""
    nl = text.rfind("\n", max(0, start - _YEARS_LOOKBACK), start)
    if nl < 0:
        return False
    prev = text[text.rfind("\n", 0, nl) + 1 : nl].strip()
    if not prev or len(prev) > 60 or "." in prev:
        return False
    if not (prev.endswith(":") or len(prev.split()) <= 3):
        return False
    return _YEARS_SOFT.search(prev) is not None


def years_required_stated(text: str) -> tuple[int, str] | None:
    """The minimum years an advert requires and its own words, or None (2.5.5).

    The first match that is a requirement wins. A match is not one when the
    count is zero or over `_YEARS_MAX`; when its clause ends in a limit or a
    programme length, opens with a preference word or sits under a preference
    heading, or names someone other than the candidate; when the sentence
    goes on to prefer rather than require it, to describe the company, or to
    describe a career path; or when a degree is offered instead of the years.
    """
    matches = sorted(
        (m for pattern in _YEARS_STATED for m in pattern.finditer(text)),
        key=lambda m: m.start(),
    )
    for m in matches:
        n = int(m.group("n"))
        if n == 0 or n > _YEARS_MAX:
            continue
        phrase = m.group(0).strip()
        start = m.start() + len(m.group(0)) - len(m.group(0).lstrip())
        before = text[max(0, start - _YEARS_LOOKBACK) : start]
        boundary = max((b.end() for b in _CLAUSE_START.finditer(before)), default=0)
        clause = before[boundary:]
        after = text[m.end() : m.end() + 200]
        candidate_is_subject = phrase.lower().startswith("you have")
        if (
            _YEARS_CAP.search(clause)
            or _YEARS_CAP_TIGHT.search(before[-16:])
            or _YEARS_SOFT.search(clause)
            or _years_soft_heading(text, start)
            or (
                not candidate_is_subject
                and (
                    _years_third_party(clause) or _YEARS_COMPANY_SUBJECT.search(clause)
                )
            )
            or _YEARS_SOFT_AFTER.search(after)
            or _YEARS_COMPANY_AFTER.search(after)
            or _YEARS_WAIVED.search(after)
            or _YEARS_CAREER_PATH.search(after)
        ):
            continue
        return n, phrase[:QUOTE_CHARS]
    return None


def resolve_years(facts: PostingFacts, job: Job) -> PostingFacts:
    """Fill a null `years_required` from the advert's own words (2.5.5).

    The 2026-10-05/06 digests rated Dubai and London roles APPLY whose
    adverts said "3+ years of experience", "5+ years" and "3-7 years": the
    model's reason text repeated the requirement, but its `years_required`
    came back null in 12 of 13 such adverts, so the owner's
    `max_years_required` rule never fired. When the model did state a value
    it is kept - it has passed the quote guard - and only a null is filled,
    from the title first, quoting the words `years_required_stated` matched,
    which are verbatim and so would pass the guard themselves.

    Pure and idempotent; never mutates `facts`. Expects `facts` to have been
    through `verify_facts` already.
    """
    if facts.years_required.value is not None:
        return facts
    found = years_required_stated(job.title) or years_required_stated(job.description)
    if found is None:
        return facts
    value, quote = found
    return facts.model_copy(
        update={"years_required": YearsFact(value=value, quote=quote)}
    )


def resolve_student(facts: PostingFacts, job: Job) -> PostingFacts:
    """Correct `student_only` against the advert's own words, after `verify_facts`.

    The model was observed setting `student_only` on internships that also
    accept recent graduates - three in one evaluation run, each with an
    explicit graduate clause. The quote guard cannot catch it: the model's
    quote ("in the final year of a Bachelor's") is verbatim, just not the
    whole story. When `student_only` is True and the title or description
    states that graduates are eligible (`graduates_eligible`), the fact
    becomes False, quoting the advert's own words; the title is read first.

    2.5.3 adds the other direction. The local model left `student_only` unset
    on adverts that restrict eligibility in so many words ("A penultimate
    year undergraduate or 1st year master's student", "Currently pursuing a
    Bachelor's"), so a graduate's profile scored them APPLY. When the advert
    has no graduate clause and does name current enrolment
    (`students_only`), the fact becomes True, quoting those words. A graduate
    clause anywhere still wins, so an advert open to "final year students or
    recent graduates" is never made student-only. The graduation-year rule is
    separate - a "graduating in 2028" advert is still judged on its year.

    Pure and idempotent; never mutates `facts`. Expects `facts` to have been
    through `verify_facts` already.
    """
    graduates = graduates_eligible(job.title) or graduates_eligible(job.description)
    if graduates is not None:
        if facts.student_only.value is not True:
            return facts
        student = StudentFact(value=False, quote=graduates[:QUOTE_CHARS])
        return facts.model_copy(update={"student_only": student})
    if facts.student_only.value is True:
        return facts
    enrolled = students_only(job.title) or students_only(job.description)
    if enrolled is None:
        return facts
    student = StudentFact(value=True, quote=enrolled[:QUOTE_CHARS])
    return facts.model_copy(update={"student_only": student})


# Level words. Whole words only, so "Internal", "Headcount", "Staffing" and
# "Leadership" match nothing. Precedence (2.4.2):
#   1. Any graduate/entry word, then any junior word, beats any senior or lead
#      word in the same text: an intern posting is an intern posting whatever
#      else its title says ("Senior Analyst Internship"), and "Junior Product
#      Manager" or "Jr Staff Engineer" is a junior role.
#   2. Between senior and lead_principal, lead_principal wins ("Senior
#      Director of Data").
#   3. (2.5.1) mid comes last: "mid-level" and "intermediate" are mid, but any
#      other level word in the same text beats them ("Senior / Mid-Level").
#      "mid-senior" is senior, spelled out in the senior row; "mid" alone
#      ("Mid Market", "Mid-Office") is not a level word.
# "manager" is not a level word: "Product Manager", "Account Manager" and
# "Assistant Manager" say nothing about seniority on their own. "staff" counts
# only before engineer/scientist/developer ("Staff Accountant" is not staff
# level), and "lead" only immediately before a role word ("Lead Data Engineer"
# counts, "Lead Generation Executive" does not).
# 2.5.3: "staff" also counts with up to three words between it and the role
# word ("Staff Data Analyst", "Staff Machine Learning Engineer"), analyst and
# architect join that role list, and "lead" counts after tech/team or at the
# end of the title or a title segment ("Tech Lead - Payments", "Video Lead",
# "Data Lead, EMEA"). "Staff Accountant", "Staff Nurse" and "Lead Generation
# Executive" still match nothing.
_LEAD_ROLE_WORDS = r"(?:engineer|developer|scientist|analyst|data|architect|ml|ai)"
_STAFF_ROLE_WORDS = r"(?:engineer|developer|scientist|analyst|architect)"
_LEVEL_WORDS: tuple[tuple[Level, re.Pattern[str]], ...] = (
    (
        Level.graduate_entry,
        re.compile(
            r"\b(?:graduate|grad|intern|internship|placement|entry[- ]level|"
            r"trainee|apprentice)\b",
            re.IGNORECASE,
        ),
    ),
    (Level.junior, re.compile(r"\b(?:junior|jr|assistant)\b", re.IGNORECASE)),
    (
        Level.lead_principal,
        re.compile(
            r"\b(?:principal|director|head\s+of"
            rf"|staff\s+(?:[\w&/-]+\s+){{0,3}}?{_STAFF_ROLE_WORDS}"
            r"|(?:tech|technical|team)\s+lead"
            rf"|lead\s+{_LEAD_ROLE_WORDS}"
            r"|lead(?=\s*(?:$|[-\u2013\u2014,(|/:])))\b",
            re.IGNORECASE,
        ),
    ),
    (
        Level.senior,
        re.compile(r"\b(?:mid[- ]senior|senior|sr)\b", re.IGNORECASE),
    ),
    (
        Level.mid,
        re.compile(r"\b(?:mid[- ]level|intermediate)\b", re.IGNORECASE),
    ),
)


def level_from_text(text: str) -> Level | None:
    """The level a title or quote names by keyword, or None if it names none.

    The first matching row of `_LEVEL_WORDS` wins, which encodes the
    precedence described above it.
    """
    folded = text.translate(_TYPO_FOLD)
    for level, pattern in _LEVEL_WORDS:
        if pattern.search(folded):
            return level
    return None


def resolve_level(facts: PostingFacts, job: Job) -> PostingFacts:
    """Settle `level` deterministically, after `verify_facts`.

    The local model was observed getting level wrong in both directions: it
    quoted "Senior Data & BI Engineer" (the title) and still answered
    `not_stated`, and it skipped a two-years role on the one-word quote
    "Senior" lifted from a sentence about stakeholders. The quote guard only
    proves a quote exists, not that it is about this role's level. The title
    is better evidence than either, and the code already has it. In order:

    1. The title names a level (`_LEVEL_WORDS`): that level, quoting the title.
    2. The model stated a level and its verified quote is at least two words:
       keep it. One word ("Senior") is too easily lifted from a sentence about
       someone else.
    3. The model said `not_stated` but its verified quote (again two words or
       more, for the same reason) names a level: derive the level from it.
    4. Otherwise `not_stated`.

    It also records where the level came from (2.5.0), for
    `rules.level_from_title_only`: `title` for step 1 only, `text` for steps
    2-3, `none` for step 4. Steps 2-3 are `text` even when the model's quote is
    copied from the title (2.5.1): the title "Data Engineer" names no level, so
    a model that answers `mid` quoting it has inferred the level, and that
    inference must not fire the title-only rule. Any `source` already on the
    fact is overwritten, so a model can never claim `title` for itself.

    Pure and idempotent; never mutates `facts`. Expects `facts` to have been
    through `verify_facts` already, so every quote it reads is verbatim.
    """
    title_level = level_from_text(job.title)
    if title_level is not None:
        quote = job.title[:QUOTE_CHARS]
        level = LevelFact(value=title_level, quote=quote, source="title")
    else:
        model = facts.level
        enough_words = len(model.quote.split()) >= 2
        derived = level_from_text(model.quote) if enough_words else None
        if model.value != Level.not_stated and enough_words:
            level = LevelFact(value=model.value, quote=model.quote, source="text")
        elif model.value == Level.not_stated and derived is not None:
            level = LevelFact(value=derived, quote=model.quote, source="text")
        else:
            level = LevelFact(value=Level.not_stated, quote="", source="none")
    if level == facts.level:
        return facts
    return facts.model_copy(update={"level": level})


# Field words, as whole phrases. Precedence (2.4.3): the first matching row
# wins, so a title that names both reads as the earlier field - AI/ML beats
# data engineering ("ML Engineer, Data Platform"), data words beat software
# words ("Software Engineer, Data Platform"), data science beats analytics
# ("Data Scientist / Data Analyst") and analytics beats software ("BI
# Developer / Software Engineer"). There is deliberately no `other` row: a
# title that names no field here says nothing, and guessing `other` from it
# would let `allowed_fields` skip a posting on the strength of a missing word.
#
# 2.4.4: the ai_llm/data_engineering/data_science phrases allow the natural
# inflections of the role words ("-ing", plural "-s") so "Data Engineering",
# "AI Engineering" and "Data Platforms" match like their bare forms did
# already - a whole-word match on "engineer" alone missed the gerund, which
# let a software word in the same title win by default, against the design's
# own intent that data/AI words beat software words. "ai engineer" also
# allows one word between "ai" and the role word ("AI Software Engineer",
# "AI Backend Engineer") for the same reason: "software"/"backend" sitting
# between them is not a different field, it is still an AI role. Word
# boundaries and the row order are otherwise unchanged.
#
# 2.5.2 adds three rows, so the order is now ai_llm, data_engineering,
# data_science, analytics_bi, quant, product, software, consulting. Data and AI
# words still win ("Data Science Consultant" is data_science, "Product Manager,
# Data Platform" is data_engineering); quant and product beat software words
# ("Quantitative Developer, Backend" is quant); and software beats a generic
# consultant ("DevOps Consultant" is software). The data_science row keeps its
# "quant research" phrases, so a Quant Researcher is data_science as before.
#
# 2.5.3 adds finance last: M&A, investment and equity research, trading,
# financial and credit analysts, energy and commodity market analysts. Every
# earlier row still wins ("Quantitative Analyst" is quant, "Data Analyst,
# Trading" is analytics_bi, "Software Engineer, Trading Systems" is software),
# and "Marketing Analyst" or "Research Analyst" alone name no field.
_FIELD_WORDS: tuple[tuple[JobField, re.Pattern[str]], ...] = (
    (
        JobField.ai_llm,
        re.compile(
            r"\b(?:machine\s+learning|ml\s+engineer(?:ing|s)?|mlops"
            r"|ai\s+(?:\w+\s+)?engineer(?:ing|s)?|ai/ml"
            r"|llm|nlp|deep\s+learning|computer\s+vision|generative\s+ai"
            r"|applied\s+scientist)\b",
            re.IGNORECASE,
        ),
    ),
    (
        JobField.data_engineering,
        re.compile(
            r"\b(?:data\s+engineer(?:ing|s)?|analytics\s+engineer(?:ing|s)?"
            r"|etl|data\s+platforms?"
            r"|big\s+data|data\s+architect|dataops|data\s+pipelines?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        JobField.data_science,
        re.compile(
            r"\b(?:data\s+scientists?|data\s+science"
            r"|quantitative\s+research(?:er)?|quant\s+research(?:er)?"
            r"|statistician|research\s+scientist)\b",
            re.IGNORECASE,
        ),
    ),
    (
        JobField.analytics_bi,
        re.compile(
            r"\b(?:data\s+analyst|bi\s+analyst|bi\s+developer"
            r"|business\s+intelligence|reporting\s+analyst|insights\s+analyst"
            r"|power\s+bi|tableau\s+developer)\b",
            re.IGNORECASE,
        ),
    ),
    (
        JobField.quant,
        re.compile(
            r"\b(?:quant(?:itative)?\s+(?:analysts?|traders?|developers?"
            r"|strategists?|trading)"
            r"|algorithmic\s+trading|systematic\s+trading)\b",
            re.IGNORECASE,
        ),
    ),
    (
        JobField.product,
        re.compile(
            r"\bproduct\s+(?:managers?|owners?|analysts?|management|associates?"
            r"|interns?|internships?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        JobField.software,
        re.compile(
            r"\b(?:software\s+engineer|software\s+developer|backend|back-end"
            r"|frontend|front-end|full\s+stack|fullstack|full-stack|devops"
            r"|site\s+reliability|sre|mobile\s+developer|ios\s+developer"
            r"|android\s+developer)\b",
            re.IGNORECASE,
        ),
    ),
    (
        JobField.consulting,
        re.compile(
            r"\b(?:consultants?|consulting|advisory\s+(?:analysts?|associates?))\b",
            re.IGNORECASE,
        ),
    ),
    (
        JobField.finance,
        re.compile(
            r"\b(?:m\s*&\s*a|mergers\s+and\s+acquisitions|investment\s+banking"
            r"|investment\s+(?:analysts?|associates?|bankers?)|equity\s+research"
            r"|private\s+equity|venture\s+capital|asset\s+management"
            r"|portfolio\s+(?:analysts?|management)"
            r"|(?:financial|finance|credit|valuation|treasury)\s+analysts?"
            r"|market\s+analysts?"
            r"|(?:energy|power|gas|oil|commodit(?:y|ies)|carbon)\s+(?:market\s+)?analysts?"
            r"|traders?|trading)\b",
            re.IGNORECASE,
        ),
    ),
)


def field_from_text(text: str) -> JobField | None:
    """The field a title or quote names by phrase, or None if it names none.

    The first matching row of `_FIELD_WORDS` wins, which encodes the
    precedence described above it. Never returns `JobField.other`.
    """
    folded = text.translate(_TYPO_FOLD)
    for field, pattern in _FIELD_WORDS:
        if pattern.search(folded):
            return field
    return None


def resolve_field(facts: PostingFacts, job: Job) -> PostingFacts:
    """Settle `field` deterministically, after `verify_facts`.

    The local model left `field` null on 82 of 91 cached postings, so
    `allowed_fields` almost never fired - the same failure `resolve_level`
    fixed for level, and the same remedy: the title is better evidence than
    the model, and the code already has it. In order:

    1. The title names a field (`_FIELD_WORDS`): that field, quoting the title.
    2. The model stated a field and its verified quote is at least two words:
       keep it. One word ("Python") is too thin to put a posting in a field.
    3. The model stated no field but its verified quote (again two words or
       more) names one: derive the field from it.
    4. Otherwise None.

    Pure and idempotent; never mutates `facts`. Expects `facts` to have been
    through `verify_facts` already, so every quote it reads is verbatim.
    """
    title_field = field_from_text(job.title)
    if title_field is not None:
        field = FieldFact(value=title_field, quote=job.title[:QUOTE_CHARS])
    else:
        model = facts.field
        enough_words = len(model.quote.split()) >= 2
        derived = field_from_text(model.quote) if enough_words else None
        if model.value is not None and enough_words:
            field = model
        elif model.value is None and derived is not None:
            field = FieldFact(value=derived, quote=model.quote)
        else:
            field = FieldFact(value=None, quote="")
    if field == facts.field:
        return facts
    return facts.model_copy(update={"field": field})
