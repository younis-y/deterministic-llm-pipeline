from __future__ import annotations

from rolescan.models import BarKind, Job, JobField, Level
from rolescan.scoring.facts import (
    FieldFact,
    HardBar,
    LevelFact,
    PostingFacts,
    StudentFact,
    YearsFact,
    verify_facts,
)

JOB = Job(source="t", company="Acme", title="Data Engineer", url="https://x/1",
          description="We need 3+ years of Python.   Open to UAE nationals only. Graduates welcome.")


def _facts(**kw: object) -> PostingFacts:
    base: dict[str, object] = {
        "level": LevelFact(value=Level.not_stated),
        "years_required": YearsFact(value=None),
        "student_only": StudentFact(value=None),
        "hard_bars": [],
        "field": FieldFact(value=JobField.data_engineering, quote="Data Engineer"),
        "fit_score": 70,
        "reason": "Python pipelines.",
        "keywords_missing": [],
    }
    base.update(kw)
    return PostingFacts(**base)


def test_a_verbatim_quote_survives_with_whitespace_and_case_normalised() -> None:
    f = verify_facts(_facts(years_required=YearsFact(value=3, quote="need 3+ YEARS of python")), JOB)
    assert f.years_required.value == 3


def test_an_invented_quote_downgrades_the_fact_to_not_stated() -> None:
    f = verify_facts(_facts(years_required=YearsFact(value=5, quote="5 years minimum")), JOB)
    assert f.years_required.value is None and f.years_required.quote == ""


def test_a_level_with_an_invented_quote_becomes_not_stated() -> None:
    f = verify_facts(_facts(level=LevelFact(value=Level.senior, quote="Senior role")), JOB)
    assert f.level.value == Level.not_stated


def test_hard_bars_without_a_verifiable_quote_are_dropped() -> None:
    f = verify_facts(_facts(hard_bars=[
        HardBar(kind=BarKind.nationality, quote="Open to UAE nationals only"),
        HardBar(kind=BarKind.clearance, quote="SC clearance required"),
    ]), JOB)
    assert [b.kind for b in f.hard_bars] == [BarKind.nationality]


def test_a_fact_with_a_value_but_no_quote_is_not_stated() -> None:
    f = verify_facts(_facts(student_only=StudentFact(value=True, quote="")), JOB)
    assert f.student_only.value is None


def test_overlong_lists_and_reason_are_trimmed_not_rejected() -> None:
    f = _facts(keywords_missing=[f"k{i}" for i in range(12)], reason="x" * 400)
    assert len(f.keywords_missing) == 8 and len(f.reason) <= 220
