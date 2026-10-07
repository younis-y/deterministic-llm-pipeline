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

__all__ = ["APPLY_AT", "BLOCKED_CAP", "CONSIDER_AT", "RULE_ORDER", "decide"]

_REASON_CHARS = 220

#: The name `decide` writes to `FitVerdict.rule` for each rule, in the order it
#: applies them (see its docstring). `hard_bar` covers rules 1 and 1b: a
#: nationality or clearance block and an `other` bar's skip. The digest groups
#: rule-hidden postings in this order, so a rule added to `decide` belongs here
#: too, at the place it is checked.
RULE_ORDER = (
    "hard_bar",
    "graduation_year",
    "student_only",
    "level",
    "years",
    "field",
)

#: Bars whose category itself constrains the model: a nationality gate or a
#: clearance. Only these block. `BarKind.other` is open
#: ended, and the quote guard proves a quote exists, not that it is an
#: eligibility bar - the local model has filed an experience requirement under
#: it ("have not worked in a financial services environment"). A block is a
#: one-way door (hidden, recorded as seen, never prepared), so an `other` bar
#: is a rule skip instead: capped below the digest like any other rule skip,
#: but still a skip with its quote, not a deletion.
_STRUCTURAL_BARS = frozenset({BarKind.nationality, BarKind.clearance})

#: Bars `decide` ignores entirely (2.4.2). A work_auth bar depends on the
#: model's judgement of which countries the candidate can already work in,
#: read from a free-text summary, and that judgement was not reliable enough
#: to hide a role on. Work authorisation is checked by configured keywords
#: (`profile.hard_blockers`) instead. The bar is still extracted, so an
#: evaluation can keep measuring extraction, but it neither blocks nor skips.
_IGNORED_BARS = frozenset({BarKind.work_auth})


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


def _rule_skip(facts: PostingFacts, rules: RulesConfig) -> tuple[str, str] | None:
    """The name and reason of the first of rules 2-6 that fires, or None.

    Split out of `decide` so the rule order reads top to bottom in one place.
    The name is the rule's entry in `RULE_ORDER`.
    """
    year = facts.graduation_year.value
    limit = rules.max_graduation_year
    # The year rule exists only while a limit is set. While it does, a stated
    # year decides on its own and `student_only` steps aside for that posting;
    # with no limit, a stated year changes nothing and `student_only` decides
    # exactly as it did before the year was extracted.
    judged_by_year = year is not None and limit is not None
    if year is not None and limit is not None and year > limit:
        reason = _quote_reason("Skip: advert says ", facts.graduation_year.quote)
        return "graduation_year", reason
    level = facts.level
    if (
        not judged_by_year
        and facts.student_only.value is True
        and rules.student_only == "skip"
    ):
        return "student_only", _quote_reason(
            "Skip: advert says ", facts.student_only.quote
        )
    if (
        level.value != Level.not_stated
        and level.value not in rules.allowed_levels
        and (not rules.level_from_title_only or level.source == "title")
    ):
        return "level", _quote_reason("Skip: advert is for ", level.quote)
    if (
        facts.years_required.value is not None
        and rules.max_years_required is not None
        and facts.years_required.value > rules.max_years_required
    ):
        return "years", _quote_reason(
            "Skip: advert asks for ", facts.years_required.quote
        )
    if (
        facts.field.value is not None
        and rules.allowed_fields is not None
        and facts.field.value not in rules.allowed_fields
    ):
        return "field", _quote_reason("Skip: advert is for ", facts.field.quote)
    return None


def decide(
    facts: PostingFacts, rules: RulesConfig | None, min_report_score: int
) -> FitVerdict:
    """Turn verified facts into a verdict, applying `rules` in a fixed order.

    Rule order (first match wins):
      0. `work_auth` hard bars are ignored (see `_IGNORED_BARS`).
      1. A nationality or clearance hard bar -> blocked.
      1b. Any other (`BarKind.other`) hard bar -> skip, whatever `rules` says.
      2. `graduation_year` is stated, `rules.max_graduation_year` is set, and
         the year exceeds it -> skip (2.5.0).
      3. `student_only` is True and `rules.student_only == "skip"` -> skip -
         but only when rule 2 did not judge the posting: with a limit set, a
         stated year within it means `student_only` does not fire.
      4. `level` is stated and not in `rules.allowed_levels` -> skip. With
         `rules.level_from_title_only`, only a level whose `source` is
         `title` (see `resolve_level`) can fire this (2.5.0).
      5. `years_required` is stated and exceeds `rules.max_years_required`
         (when that cap is set) -> skip.
      6. `field` is stated and `rules.allowed_fields` is set and excludes it
         -> skip.
      7. Otherwise, the model's `fit_score` decides: apply at `APPLY_AT` or
         above, consider at `CONSIDER_AT` or above, else skip.

    `rules is None` means only rules 1, 1b and 7 apply - a not-stated fact
    never fires a rule, and no `RulesConfig` means no rule from 2-6 exists to
    fire at all.

    A rule that fires (1-6) caps the score so a filtered-out posting can
    never look better than one that reached the digest on fit alone: rule 1
    caps at `BLOCKED_CAP`, rules 1b-6 cap just under `min_report_score`.
    `confidence` is `high` when a rule fired, since a rule is a fact check
    rather than a judgement call, and `medium` otherwise. `keywords_missing`
    always passes through from the model unchanged.

    `rule` names the rule that fired, by its `RULE_ORDER` entry (`hard_bar`
    for 1 and 1b), and is None when rule 7 decided. The digest lists
    rule-hidden postings from it: on 2026-10-06, 44 of 109 scored postings
    were hidden by these rules with no trace, so a wrong skip was invisible.
    """
    bars = [b for b in facts.hard_bars if b.kind not in _IGNORED_BARS]
    structural = [b for b in bars if b.kind in _STRUCTURAL_BARS]
    if structural:
        return FitVerdict(
            fit_score=min(facts.fit_score, BLOCKED_CAP),
            verdict=Verdict.BLOCKED,
            confidence=Confidence.HIGH,
            reason=_quote_reason("Blocked: advert says ", structural[0].quote),
            blockers=[b.quote for b in structural],
            keywords_missing=facts.keywords_missing,
            rule="hard_bar",
        )

    skip: tuple[str, str] | None = None
    if bars:
        skip = "hard_bar", _quote_reason("Skip: advert requires ", bars[0].quote)

    if rules is not None and skip is None:
        skip = _rule_skip(facts, rules)

    if skip is not None:
        rule, skip_reason = skip
        capped = min(facts.fit_score, max(min_report_score - 1, 0))
        return FitVerdict(
            fit_score=capped,
            verdict=Verdict.SKIP,
            confidence=Confidence.HIGH,
            reason=skip_reason,
            blockers=[],
            keywords_missing=facts.keywords_missing,
            rule=rule,
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
