"""2.5.0: the graduation-year fact and the `max_graduation_year` rule.

`student_only` is one switch, so it cannot tell an internship for 2027
graduates from one for 2028 graduates. The advert's stated graduation year
can, so when it is stated it decides instead.
"""

from __future__ import annotations

import pytest

from rolescan.config import RulesConfig
from rolescan.models import Job, Level
from rolescan.scoring.facts import (
    QUOTE_CHARS,
    GraduationYearFact,
    LevelFact,
    PostingFacts,
    StudentFact,
    verify_facts,
)
from rolescan.scoring.rules import decide

JOB = Job(
    source="t",
    company="Acme",
    title="Data Science Intern",
    url="https://x/1",
    description=(
        "Open to students graduating in 2028. Current students only. "
        "Applicants must be graduating 2027 or 2028."
    ),
)


def _f(fit: int = 80, **kw: object) -> PostingFacts:
    base: dict[str, object] = {"fit_score": fit, "reason": "Model sentence."}
    base.update(kw)
    return PostingFacts(**base)


# --- the fact --------------------------------------------------------------


def test_graduation_year_defaults_to_not_stated() -> None:
    f = _f()
    assert f.graduation_year.value is None
    assert f.graduation_year.quote == ""


def test_graduation_year_parses_from_model_json() -> None:
    f = PostingFacts.model_validate({
        "graduation_year": {"value": 2028, "quote": "graduating in 2028"},
        "fit_score": 10,
        "reason": "r",
    })
    assert f.graduation_year == GraduationYearFact(value=2028, quote="graduating in 2028")


def test_graduation_year_quote_is_capped_like_every_other_quote() -> None:
    schema = PostingFacts.model_json_schema()
    quote = schema["$defs"]["GraduationYearFact"]["properties"]["quote"]
    assert quote["maxLength"] == QUOTE_CHARS
    long = "graduating in 2028 " * 30
    assert len(GraduationYearFact(value=2028, quote=long).quote) == QUOTE_CHARS


def test_graduation_year_sits_in_the_schema_the_model_fills() -> None:
    assert "graduation_year" in PostingFacts.model_json_schema()["properties"]


def test_a_verbatim_graduation_year_quote_survives_the_guard() -> None:
    f = verify_facts(
        _f(graduation_year=GraduationYearFact(value=2028, quote="graduating in 2028")),
        JOB,
    )
    assert f.graduation_year.value == 2028
    assert f.graduation_year.quote == "graduating in 2028"


def test_an_invented_graduation_year_quote_is_downgraded_to_not_stated() -> None:
    f = verify_facts(
        _f(graduation_year=GraduationYearFact(value=2029, quote="class of 2029")),
        JOB,
    )
    assert f.graduation_year == GraduationYearFact()


def test_a_graduation_year_with_no_quote_is_downgraded() -> None:
    f = verify_facts(_f(graduation_year=GraduationYearFact(value=2028, quote="")), JOB)
    assert f.graduation_year.value is None


# --- the rule --------------------------------------------------------------

OWNER = RulesConfig(student_only="skip", max_graduation_year=2027)


def test_max_graduation_year_defaults_to_no_limit() -> None:
    assert RulesConfig().max_graduation_year is None


def test_a_stated_year_above_the_limit_skips_with_its_quote() -> None:
    v = decide(
        _f(fit=90, graduation_year=GraduationYearFact(value=2028, quote="graduating in 2028")),
        OWNER,
        55,
    )
    assert v.verdict.value == "skip"
    assert v.reason == 'Skip: advert says "graduating in 2028"'
    assert v.fit_score == 54
    assert v.confidence.value == "high"


def test_a_year_within_the_limit_passes_even_when_student_only_would_skip() -> None:
    v = decide(
        _f(
            fit=80,
            graduation_year=GraduationYearFact(value=2027, quote="class of 2027"),
            student_only=StudentFact(value=True, quote="Current students only"),
        ),
        OWNER,
        55,
    )
    assert v.verdict.value == "apply"
    assert v.reason == "Model sentence."


def test_no_stated_year_leaves_student_only_in_charge() -> None:
    v = decide(
        _f(fit=80, student_only=StudentFact(value=True, quote="Current students only")),
        OWNER,
        55,
    )
    assert v.verdict.value == "skip"
    assert v.reason == 'Skip: advert says "Current students only"'


def test_the_year_rule_runs_before_student_only() -> None:
    v = decide(
        _f(
            graduation_year=GraduationYearFact(value=2028, quote="graduating in 2028"),
            student_only=StudentFact(value=True, quote="Current students only"),
        ),
        OWNER,
        55,
    )
    assert "graduating in 2028" in v.reason


def test_the_year_rule_runs_before_level() -> None:
    v = decide(
        _f(
            graduation_year=GraduationYearFact(value=2028, quote="graduating in 2028"),
            level=LevelFact(value=Level.senior, quote="Senior Data Scientist"),
        ),
        OWNER.model_copy(update={"allowed_levels": [Level.graduate_entry]}),
        55,
    )
    assert "graduating in 2028" in v.reason


def test_hard_bars_still_beat_the_year_rule() -> None:
    from rolescan.models import BarKind
    from rolescan.scoring.facts import HardBar

    v = decide(
        _f(
            graduation_year=GraduationYearFact(value=2028, quote="graduating in 2028"),
            hard_bars=[HardBar(kind=BarKind.clearance, quote="SC clearance")],
        ),
        OWNER,
        55,
    )
    assert v.verdict.value == "blocked"


def test_a_year_at_the_limit_passes() -> None:
    v = decide(
        _f(fit=80, graduation_year=GraduationYearFact(value=2027, quote="graduating in 2027")),
        OWNER,
        55,
    )
    assert v.verdict.value == "apply"


def test_no_rules_block_means_the_year_never_decides() -> None:
    v = decide(
        _f(fit=80, graduation_year=GraduationYearFact(value=2030, quote="class of 2030")),
        None,
        55,
    )
    assert v.verdict.value == "apply" and v.fit_score == 80


@pytest.mark.parametrize("student_only", ["skip", "allow"])
def test_without_a_limit_a_stated_year_changes_nothing(student_only: str) -> None:
    """No `max_graduation_year` means the year rule does not exist for this
    config, so it cannot take over from `student_only` either: a config that
    does not set the new key decides exactly as 2.4.4 did."""
    rules = RulesConfig(student_only=student_only)
    facts = _f(
        fit=80,
        graduation_year=GraduationYearFact(value=2030, quote="class of 2030"),
        student_only=StudentFact(value=True, quote="Current students only"),
    )
    without_year = facts.model_copy(update={"graduation_year": GraduationYearFact()})
    assert decide(facts, rules, 55) == decide(without_year, rules, 55)


def test_the_facts_prompt_defines_graduation_year_once_clearly() -> None:
    from rolescan.scoring.llm import SYSTEM_FACTS

    text = " ".join(SYSTEM_FACTS.split())
    assert "graduation_year - the EARLIEST graduation year" in text
    assert '"graduating 2027 or 2028" -> 2027' in text
    # Encoded as null when not stated, like the other optional facts.
    assert "graduation_year" in text[text.index('Encode "not stated"'):text.index("level - ")]
