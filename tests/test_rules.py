from __future__ import annotations

import pytest

from rolescan.config import RulesConfig
from rolescan.models import BarKind, FitVerdict, JobField, Level
from rolescan.scoring.facts import (
    FieldFact,
    GraduationYearFact,
    HardBar,
    LevelFact,
    PostingFacts,
    StudentFact,
    YearsFact,
)
from rolescan.scoring.rules import RULE_ORDER, decide

RULES = RulesConfig(
    max_years_required=2,
    allowed_levels=[Level.graduate_entry, Level.junior, Level.not_stated],
    student_only="skip",
    allowed_fields=[JobField.data_engineering, JobField.ai_llm, JobField.data_science],
)


def _f(fit: int = 80, **kw: object) -> PostingFacts:
    base: dict[str, object] = {
        "level": LevelFact(value=Level.not_stated),
        "years_required": YearsFact(value=None),
        "student_only": StudentFact(value=None),
        "hard_bars": [],
        "field": FieldFact(value=None),
        "fit_score": fit,
        "reason": "Model sentence.",
        "keywords_missing": ["Spark"],
    }
    base.update(kw)
    return PostingFacts(**base)


@pytest.mark.parametrize(
    ("facts", "verdict", "reason_part"),
    [
        (
            _f(
                hard_bars=[
                    HardBar(kind=BarKind.clearance, quote="SC clearance required")
                ]
            ),
            "blocked",
            "SC clearance required",
        ),
        (
            _f(student_only=StudentFact(value=True, quote="Undergraduates only")),
            "skip",
            "Undergraduates only",
        ),
        (
            _f(level=LevelFact(value=Level.senior, quote="Senior Data Engineer")),
            "skip",
            "Senior Data Engineer",
        ),
        (_f(years_required=YearsFact(value=5, quote="5+ years")), "skip", "5+ years"),
        (
            _f(field=FieldFact(value=JobField.other, quote="Front desk")),
            "skip",
            "Front desk",
        ),
        (_f(fit=80), "apply", "Model sentence."),
        (_f(fit=50), "consider", "Model sentence."),
        (_f(fit=30), "skip", "Model sentence."),
    ],
)
def test_each_rule(facts: PostingFacts, verdict: str, reason_part: str) -> None:
    v = decide(facts, RULES, min_report_score=50)
    assert v.verdict.value == verdict and reason_part in v.reason


def test_rule_order_hard_bar_beats_seniority() -> None:
    v = decide(
        _f(
            hard_bars=[HardBar(kind=BarKind.nationality, quote="UAE nationals only")],
            level=LevelFact(value=Level.senior, quote="Senior"),
        ),
        RULES,
        50,
    )
    assert v.verdict.value == "blocked"


def test_all_facts_not_stated_is_decided_by_fit_alone() -> None:
    assert decide(_f(fit=70), RULES, 50).verdict.value == "apply"


def test_years_at_the_limit_pass_and_above_skip() -> None:
    assert (
        decide(
            _f(years_required=YearsFact(value=2, quote="2 years")), RULES, 50
        ).verdict.value
        == "apply"
    )
    assert (
        decide(
            _f(years_required=YearsFact(value=3, quote="3 years")), RULES, 50
        ).verdict.value
        == "skip"
    )


def test_student_only_allowed_when_configured() -> None:
    rules = RULES.model_copy(update={"student_only": "allow"})
    assert (
        decide(
            _f(student_only=StudentFact(value=True, quote="Students only")), rules, 50
        ).verdict.value
        == "apply"
    )


def test_no_rules_block_means_no_rule_fires() -> None:
    assert (
        decide(
            _f(level=LevelFact(value=Level.senior, quote="Senior")), None, 50
        ).verdict.value
        == "apply"
    )


def test_scores_are_capped_so_skips_and_blocks_never_reach_the_digest() -> None:
    skip = decide(
        _f(fit=90, years_required=YearsFact(value=5, quote="5+ years")), RULES, 50
    )
    block = decide(
        _f(fit=90, hard_bars=[HardBar(kind=BarKind.clearance, quote="DV")]), RULES, 50
    )
    assert skip.fit_score == 49 and block.fit_score == 20
    assert block.blockers == ["DV"]


def test_verdict_keeps_the_model_gaps() -> None:
    assert decide(_f(), RULES, 50).keywords_missing == ["Spark"]


# --- `other` hard bars skip, never block (fix C) ---------------------------

FS_QUOTE = (
    "The client does not meet candidates who have not worked in a financial "
    "services environment."
)


def test_a_verified_other_bar_is_a_rule_skip_not_a_block() -> None:
    v = decide(
        _f(fit=90, hard_bars=[HardBar(kind=BarKind.other, quote=FS_QUOTE)]), RULES, 50
    )
    assert v.verdict.value == "skip"
    assert v.reason.startswith('Skip: advert requires "')
    assert "financial" in v.reason
    assert v.fit_score == 49
    assert v.confidence.value == "high"
    assert v.blockers == []


def test_an_other_bar_skips_even_with_no_rules_block() -> None:
    v = decide(
        _f(fit=90, hard_bars=[HardBar(kind=BarKind.other, quote="UK driving licence")]),
        None,
        50,
    )
    assert v.verdict.value == "skip" and v.fit_score == 49


@pytest.mark.parametrize("kind", [BarKind.nationality, BarKind.clearance])
def test_structural_bars_still_block(kind: BarKind) -> None:
    v = decide(_f(hard_bars=[HardBar(kind=kind, quote="the bar")]), RULES, 50)
    assert v.verdict.value == "blocked" and v.fit_score == 20


def test_a_structural_bar_listed_after_an_other_bar_still_blocks() -> None:
    v = decide(
        _f(
            hard_bars=[
                HardBar(kind=BarKind.other, quote="UK driving licence"),
                HardBar(kind=BarKind.clearance, quote="SC clearance required"),
            ]
        ),
        RULES,
        50,
    )
    assert v.verdict.value == "blocked"
    assert "SC clearance required" in v.reason
    assert v.blockers == ["SC clearance required"]


# --- work_auth bars no longer decide anything (2.4.2) ----------------------
# Work authorisation is checked by configured keywords (hard_blockers), not by
# the model's judgement of the candidate. The bar is still extracted so the
# eval can measure extraction, but decide() ignores it.


def test_a_verified_work_auth_bar_alone_is_decided_by_fit_score() -> None:
    bar = HardBar(kind=BarKind.work_auth, quote="No visa sponsorship available")
    v = decide(_f(fit=80, hard_bars=[bar]), RULES, 50)
    assert v.verdict.value == "apply"
    assert v.fit_score == 80
    assert v.reason == "Model sentence."
    assert v.blockers == []
    low = decide(_f(fit=45, hard_bars=[bar]), RULES, 50)
    assert low.verdict.value == "consider" and low.fit_score == 45


def test_a_work_auth_bar_alone_with_no_rules_is_decided_by_fit_score() -> None:
    bar = HardBar(kind=BarKind.work_auth, quote="Must have the right to work in the US")
    v = decide(_f(fit=70, hard_bars=[bar]), None, 50)
    assert v.verdict.value == "apply" and v.fit_score == 70


def test_nationality_still_blocks_next_to_a_work_auth_bar() -> None:
    v = decide(
        _f(
            hard_bars=[
                HardBar(kind=BarKind.work_auth, quote="No visa sponsorship"),
                HardBar(kind=BarKind.nationality, quote="UK nationals only"),
            ]
        ),
        RULES,
        50,
    )
    assert v.verdict.value == "blocked"
    assert v.blockers == ["UK nationals only"]
    assert "UK nationals only" in v.reason


def test_an_other_bar_still_skips_next_to_a_work_auth_bar() -> None:
    v = decide(
        _f(
            fit=90,
            hard_bars=[
                HardBar(kind=BarKind.work_auth, quote="No visa sponsorship"),
                HardBar(kind=BarKind.other, quote="UK driving licence"),
            ],
        ),
        RULES,
        50,
    )
    assert v.verdict.value == "skip"
    assert "UK driving licence" in v.reason


# --- the verdict names the rule that fired ----------------------------------
# On 2026-10-06, 44 of 109 scored postings were hidden by these rules with no
# trace in the digest. The digest now lists them by rule, which needs the
# verdict to say which rule fired rather than leaving it to be re-derived from
# the wording of `reason`.


@pytest.mark.parametrize(
    ("facts", "rules", "rule"),
    [
        (
            _f(
                hard_bars=[
                    HardBar(kind=BarKind.nationality, quote="UAE nationals only")
                ]
            ),
            RULES,
            "hard_bar",
        ),
        (
            _f(hard_bars=[HardBar(kind=BarKind.other, quote="UK driving licence")]),
            RULES,
            "hard_bar",
        ),
        (
            _f(hard_bars=[HardBar(kind=BarKind.other, quote="UK driving licence")]),
            None,
            "hard_bar",
        ),
        (
            _f(
                graduation_year=GraduationYearFact(
                    value=2028, quote="graduating in 2028"
                )
            ),
            RULES.model_copy(update={"max_graduation_year": 2026}),
            "graduation_year",
        ),
        (
            _f(student_only=StudentFact(value=True, quote="Undergraduates only")),
            RULES,
            "student_only",
        ),
        (
            _f(level=LevelFact(value=Level.senior, quote="Senior Data Engineer")),
            RULES,
            "level",
        ),
        (_f(years_required=YearsFact(value=5, quote="5+ years")), RULES, "years"),
        (_f(field=FieldFact(value=JobField.other, quote="Front desk")), RULES, "field"),
    ],
)
def test_a_rule_verdict_names_the_rule_that_fired(
    facts: PostingFacts, rules: RulesConfig | None, rule: str
) -> None:
    v = decide(facts, rules, 50)
    assert v.rule == rule
    assert rule in RULE_ORDER


@pytest.mark.parametrize("fit", [80, 50, 30])
def test_a_score_decided_verdict_names_no_rule(fit: int) -> None:
    assert decide(_f(fit=fit), RULES, 50).rule is None


def test_a_work_auth_bar_decided_by_score_names_no_rule() -> None:
    bar = HardBar(kind=BarKind.work_auth, quote="No visa sponsorship available")
    assert decide(_f(fit=80, hard_bars=[bar]), RULES, 50).rule is None


def test_rule_order_lists_the_rules_in_the_order_decide_applies_them() -> None:
    assert RULE_ORDER == (
        "hard_bar",
        "graduation_year",
        "student_only",
        "level",
        "years",
        "field",
    )


def test_a_cached_verdict_written_before_the_rule_field_still_parses() -> None:
    old = (
        '{"fit_score": 49, "verdict": "skip", "confidence": "high", '
        '"reason": "Skip: advert asks for \\"5+ years\\"", "blockers": [], '
        '"keywords_missing": []}'
    )
    assert FitVerdict.model_validate_json(old).rule is None


def test_the_rule_field_is_not_offered_to_a_judge_mode_model() -> None:
    """`FitVerdict` is also judge mode's output schema, on both backends. A
    `rule` property there would invite the model to name a rule that `decide`
    never ran, and the digest would then list a posting as rule-hidden on the
    model's say-so."""
    assert "rule" not in FitVerdict.model_json_schema()["properties"]
