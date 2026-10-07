"""2.5.7: rulings on the field rule (2026-10-07).

Three good roles were hidden as "other": an internship programme whose
projects are "digital transformation", a business-transformation placement
year, and a finance project analyst. An internship or graduate programme
whose field cannot be named is a programme, not a wrong field, so `other` no
longer hides one. A field exemption is by company name, for a profile that
wants one software role at a named employer only."""

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


def test_finance_data_analyst_stays_analytics_bi() -> None:
    # The finance row's widening must not outrank the earlier data-analyst row.
    assert field_from_text("Finance Data Analyst") is JobField.analytics_bi


def _hidden(company: str, exempt: list[str]) -> bool:
    rules = RulesConfig(
        allowed_fields=[JobField.data_science], field_exempt_companies=exempt
    )
    verdict = decide(
        _facts(JobField.software, Level.graduate_entry), rules, 40, company=company
    )
    return verdict.rule == "field"


ARABIC = "شركة المثال"
OTHER_ARABIC = "شركة أخرى"


def test_an_empty_or_blank_entry_exempts_nothing() -> None:
    for exempt in ([""], [" "], ["-"]):
        assert _hidden("", exempt), exempt
        assert _hidden("Acme Ltd", exempt), exempt
        assert _hidden(ARABIC, exempt), exempt


def test_an_empty_company_is_never_exempt() -> None:
    assert _hidden("", ["Example Bank"])
    assert _hidden("", [ARABIC])
    assert _hidden("", ["Example Bank", ""])


def test_a_non_latin_entry_exempts_only_the_same_company() -> None:
    assert not _hidden(ARABIC, [ARABIC])
    assert not _hidden(f"{ARABIC} ذ.م.م", [ARABIC])
    assert _hidden(OTHER_ARABIC, [ARABIC])
    assert _hidden("Acme Ltd", [ARABIC])
    assert _hidden("12345", [ARABIC])


def test_accents_case_and_width_are_folded() -> None:
    assert not _hidden("SOCIÉTÉ GÉNÉRALE SA", ["Société Générale"])
    fullwidth = "Example Bank".translate({c: c + 0xFEE0 for c in range(0x21, 0x7F)})
    assert fullwidth != "Example Bank"
    assert not _hidden(fullwidth, ["Example Bank"])
    assert not _hidden("Example-Bank, Ltd.", ["Example Bank"])
    assert _hidden("Societe Generale SA", ["Société Générale"])
