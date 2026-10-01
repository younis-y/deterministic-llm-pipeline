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


SPLICE_JOB = Job(source="t", company="Acme", title="Senior Engineer", url="https://x/2",
                  description="Must relocate immediately.")


def test_a_quote_spanning_title_and_description_does_not_verify() -> None:
    f = verify_facts(
        _facts(level=LevelFact(value=Level.senior, quote="Engineer Must relocate")),
        SPLICE_JOB,
    )
    assert f.level.value == Level.not_stated


def test_a_quote_wholly_inside_the_title_still_verifies() -> None:
    f = verify_facts(
        _facts(level=LevelFact(value=Level.senior, quote="Senior Engineer")),
        SPLICE_JOB,
    )
    assert f.level.value == Level.senior


_EN_DASH = chr(0x2013)
_RIGHT_SINGLE_QUOTE = chr(0x2019)
TYPO_JOB = Job(
    source="t", company="Acme", title="Data Engineer", url="https://x/4",
    description=f"3{_EN_DASH}5 years{_RIGHT_SINGLE_QUOTE} experience required.",
)


def test_typographic_lookalikes_fold_before_verification() -> None:
    f = verify_facts(
        _facts(years_required=YearsFact(value=3, quote="3-5 years' experience")),
        TYPO_JOB,
    )
    assert f.years_required.value == 3


# --- quote length cap (fix B) ---------------------------------------------

LONG_DESC = ("We need 3+ years of Python. " * 20).strip()
LONG_JOB = Job(source="t", company="Acme", title="Data Engineer", url="https://x/3",
               description=LONG_DESC)


def test_every_quote_field_carries_a_200_char_max_length_in_the_schema() -> None:
    """Ollama takes the schema as a grammar, so maxLength is enforced at
    sampling time and a paragraph-long quote can no longer run the model's
    JSON past its output budget."""
    schema = PostingFacts.model_json_schema()
    defs = schema["$defs"]
    for name in (
        "LevelFact", "YearsFact", "StudentFact", "GraduationYearFact", "FieldFact", "HardBar",
    ):
        assert defs[name]["properties"]["quote"]["maxLength"] == 200, name


def test_an_overlong_quote_is_trimmed_to_its_first_200_chars_not_rejected() -> None:
    f = _facts(
        years_required=YearsFact(value=3, quote=LONG_DESC),
        hard_bars=[HardBar(kind=BarKind.other, quote=LONG_DESC)],
    )
    assert f.years_required.quote == LONG_DESC[:200]
    assert f.hard_bars[0].quote == LONG_DESC[:200]


def test_a_trimmed_quote_is_a_verbatim_prefix_and_still_verifies() -> None:
    f = verify_facts(
        _facts(years_required=YearsFact(value=3, quote=LONG_DESC)), LONG_JOB
    )
    assert f.years_required.value == 3
    assert len(f.years_required.quote) == 200


def test_an_overlong_quote_in_model_json_is_trimmed_on_validate() -> None:
    raw = {
        "level": {"value": "senior", "quote": "x" * 500},
        "fit_score": 10,
        "reason": "r",
    }
    f = PostingFacts.model_validate(raw)
    assert f.level.quote == "x" * 200
