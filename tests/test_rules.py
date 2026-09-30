from __future__ import annotations

import pytest

from rolescan.config import RulesConfig
from rolescan.models import BarKind, JobField, Level
from rolescan.scoring.facts import (
    FieldFact,
    HardBar,
    LevelFact,
    PostingFacts,
    StudentFact,
    YearsFact,
)
from rolescan.scoring.rules import decide

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
            _f(hard_bars=[HardBar(kind=BarKind.clearance, quote="SC clearance required")]),
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
        (_f(field=FieldFact(value=JobField.other, quote="Front desk")), "skip", "Front desk"),
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
        decide(_f(years_required=YearsFact(value=2, quote="2 years")), RULES, 50).verdict.value
        == "apply"
    )
    assert (
        decide(_f(years_required=YearsFact(value=3, quote="3 years")), RULES, 50).verdict.value
        == "skip"
    )


def test_student_only_allowed_when_configured() -> None:
    rules = RULES.model_copy(update={"student_only": "allow"})
    assert (
        decide(_f(student_only=StudentFact(value=True, quote="Students only")), rules, 50).verdict.value
        == "apply"
    )


def test_no_rules_block_means_no_rule_fires() -> None:
    assert decide(_f(level=LevelFact(value=Level.senior, quote="Senior")), None, 50).verdict.value == "apply"


def test_scores_are_capped_so_skips_and_blocks_never_reach_the_digest() -> None:
    skip = decide(_f(fit=90, years_required=YearsFact(value=5, quote="5+ years")), RULES, 50)
    block = decide(_f(fit=90, hard_bars=[HardBar(kind=BarKind.clearance, quote="DV")]), RULES, 50)
    assert skip.fit_score == 49 and block.fit_score == 20
    assert block.blockers == ["DV"]


def test_verdict_keeps_the_model_gaps() -> None:
    assert decide(_f(), RULES, 50).keywords_missing == ["Spark"]
