"""Deterministic verdicts from verified facts.

A model asked to apply several rules at once - skip if senior, unless the
years match, unless the field is out of scope - does not apply them
reliably: it blends them into one fuzzy judgement instead of checking each
one in order, and a rule that should be absolute (a nationality bar) ends up
negotiable against one that is a preference (a fit score). So the rules
themselves live here, in code, evaluated in a fixed order against facts the
caller has already verified against the posting text - not as wording in a
prompt, which a model can read, partially apply, or ignore under load. An
exception to a rule is expressed by changing this function or its config,
never by hoping the next prompt phrases it more persuasively.

This module is pure: no I/O, no logging, and no knowledge of where facts came
from or where the verdict is going.
"""

from __future__ import annotations

from rolescan.config import RulesConfig
from rolescan.models import BarKind, Confidence, FitVerdict, Level, Verdict
from rolescan.scoring.facts import PostingFacts

APPLY_AT = 65
CONSIDER_AT = 40
BLOCKED_CAP = 20

__all__ = ["APPLY_AT", "BLOCKED_CAP", "CONSIDER_AT", "decide"]

_REASON_CHARS = 220

#: Bars whose category itself constrains the model: a nationality gate, a
#: clearance, a work authorisation. Only these block. `BarKind.other` is open
#: ended, and the quote guard proves a quote exists, not that it is an
#: eligibility bar - the local model has filed an experience requirement under
#: it ("have not worked in a financial services environment"). A block is a
#: one-way door (hidden, recorded as seen, never prepared), so an `other` bar
#: is a rule skip instead: capped below the digest like any other rule skip,
#: but still a skip with its quote, not a deletion.
_STRUCTURAL_BARS = frozenset(
    {BarKind.nationality, BarKind.clearance, BarKind.work_auth}
)


def _quote_reason(prefix: str, quote: str) -> str:
    """Build `"<prefix>: <lead-in> \"<quote>\""`, truncating the quote to fit.

    `prefix` and the lead-in wording are supplied by the caller so each rule
    can name itself (`Blocked: advert says "..."`, `Skip: advert asks for
    "..."`) while every reason stays within `_REASON_CHARS` and keeps the
    quote verbatim rather than truncating from the wrong end.
    """
    room = _REASON_CHARS - len(prefix) - len('""')
    q = quote if len(quote) <= room else quote[: max(room - 1, 0)].rstrip() + "…"
    return f'{prefix}"{q}"'


def decide(
    facts: PostingFacts, rules: RulesConfig | None, min_report_score: int
) -> FitVerdict:
    """Turn verified facts into a verdict, applying `rules` in a fixed order.

    Rule order (first match wins):
      1. A nationality, clearance or work_auth hard bar -> blocked.
      1b. Any other (`BarKind.other`) hard bar -> skip, whatever `rules` says.
      2. `student_only` is True and `rules.student_only == "skip"` -> skip.
      3. `level` is stated and not in `rules.allowed_levels` -> skip.
      4. `years_required` is stated and exceeds `rules.max_years_required`
         (when that cap is set) -> skip.
      5. `field` is stated and `rules.allowed_fields` is set and excludes it
         -> skip.
      6. Otherwise, the model's `fit_score` decides: apply at `APPLY_AT` or
         above, consider at `CONSIDER_AT` or above, else skip.

    `rules is None` means only rules 1, 1b and 6 apply - a not-stated fact
    never fires a rule, and no `RulesConfig` means no rule from 2-5 exists to
    fire at all.

    A rule that fires (1-5) caps the score so a filtered-out posting can
    never look better than one that reached the digest on fit alone: rule 1
    caps at `BLOCKED_CAP`, rules 1b-5 cap just under `min_report_score`.
    `confidence` is `high` when a rule fired, since a rule is a fact check
    rather than a judgement call, and `medium` otherwise. `keywords_missing`
    always passes through from the model unchanged.
    """
    structural = [b for b in facts.hard_bars if b.kind in _STRUCTURAL_BARS]
    if structural:
        return FitVerdict(
            fit_score=min(facts.fit_score, BLOCKED_CAP),
            verdict=Verdict.BLOCKED,
            confidence=Confidence.HIGH,
            reason=_quote_reason("Blocked: advert says ", structural[0].quote),
            blockers=[b.quote for b in structural],
            keywords_missing=facts.keywords_missing,
        )

    skip_reason: str | None = None
    if facts.hard_bars:
        skip_reason = _quote_reason("Skip: advert requires ", facts.hard_bars[0].quote)

    if rules is not None and skip_reason is None:
        if facts.student_only.value is True and rules.student_only == "skip":
            skip_reason = _quote_reason(
                "Skip: advert says ", facts.student_only.quote
            )
        elif (
            facts.level.value != Level.not_stated
            and facts.level.value not in rules.allowed_levels
        ):
            skip_reason = _quote_reason("Skip: advert is for ", facts.level.quote)
        elif (
            facts.years_required.value is not None
            and rules.max_years_required is not None
            and facts.years_required.value > rules.max_years_required
        ):
            skip_reason = _quote_reason(
                "Skip: advert asks for ", facts.years_required.quote
            )
        elif (
            facts.field.value is not None
            and rules.allowed_fields is not None
            and facts.field.value not in rules.allowed_fields
        ):
            skip_reason = _quote_reason("Skip: advert is for ", facts.field.quote)

    if skip_reason is not None:
        capped = min(facts.fit_score, max(min_report_score - 1, 0))
        return FitVerdict(
            fit_score=capped,
            verdict=Verdict.SKIP,
            confidence=Confidence.HIGH,
            reason=skip_reason,
            blockers=[],
            keywords_missing=facts.keywords_missing,
        )

    if facts.fit_score >= APPLY_AT:
        verdict = Verdict.APPLY
    elif facts.fit_score >= CONSIDER_AT:
        verdict = Verdict.CONSIDER
    else:
        verdict = Verdict.SKIP

    return FitVerdict(
        fit_score=facts.fit_score,
        verdict=verdict,
        confidence=Confidence.MEDIUM,
        reason=facts.reason,
        blockers=[],
        keywords_missing=facts.keywords_missing,
    )
