"""2.5.7: the owner's rulings on the field rule (2026-10-07).

Three roles the owner wanted were hidden as "other": Caterpillar's 2027
Internship Program ("digital transformation projects"), Schroders' Business
Transformation Placement Year, Dawsongroup's Finance Project Analyst. One
software role was wanted at a named employer only. An internship or
graduate programme whose field cannot be named is a programme, not a wrong
field, so `other` no longer hides one."""

from __future__ import annotations

from rolescan.config import RulesConfig
from rolescan.models import JobField, Level, Verdict
from rolescan.scoring.facts import FieldFact, LevelFact, PostingFacts, field_from_text
from rolescan.scoring.rules import decide


def test_transformation_titles_are_consulting() -> None:
    assert (
        field_from_text("2027 Business Transformation Placement Year Programme")
        is JobField.consulting
    )
    assert field_from_text("Digital Transformation Analyst") is JobField.consulting


def test_a_finance_project_analyst_is_finance() -> None:
    assert (
        field_from_text("Finance Project Analyst (Fixed Term Contract)")
        is JobField.finance
    )


def test_the_widened_phrases_do_not_claim_neighbouring_titles() -> None:
    # One word between "finance" and "analyst" is allowed; "Project Analyst" and
    # "Marketing Analyst" alone still name no field, and "digital" alone is not
    # "digital transformation".
    assert field_from_text("Project Analyst") is None
    assert field_from_text("Marketing Analyst") is None
    assert field_from_text("Digital Analyst") is None
    assert field_from_text("Transformation Analyst") is None


def _facts(field: JobField | None, level: Level) -> PostingFacts:
    return PostingFacts(
        fit_score=80,
        reason="fits",
        field=FieldFact(value=field, quote="quoted words"),
        level=LevelFact(value=level, quote="quoted words", source="title"),
    )


RULES = RulesConfig(
    allowed_fields=[JobField.data_science], field_exempt_companies=["Example Bank"]
)


def test_other_does_not_hide_a_graduate_programme() -> None:
    verdict = decide(_facts(JobField.other, Level.graduate_entry), RULES, 40)
    assert verdict.rule is None and verdict.verdict is Verdict.APPLY


def test_other_still_hides_an_experienced_role() -> None:
    verdict = decide(_facts(JobField.other, Level.mid), RULES, 40)
    assert verdict.rule == "field"


def test_a_named_wrong_field_still_hides_a_graduate_programme() -> None:
    verdict = decide(_facts(JobField.software, Level.graduate_entry), RULES, 40)
    assert verdict.rule == "field"


def test_an_exempt_company_passes_any_field() -> None:
    verdict = decide(
        _facts(JobField.software, Level.graduate_entry),
        RULES,
        40,
        company="Example Bank Ltd",
    )
    assert verdict.rule is None
    verdict = decide(
        _facts(JobField.software, Level.graduate_entry), RULES, 40, company="Other Co"
    )
    assert verdict.rule == "field"


def test_an_exempt_company_matches_whole_words_only() -> None:
    verdict = decide(
        _facts(JobField.software, Level.graduate_entry),
        RULES,
        40,
        company="Example Banking Group",
    )
    assert verdict.rule == "field"
