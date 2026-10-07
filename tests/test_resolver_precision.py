"""2.5.7 resolver fixes, each from a real advert the owner's scan hid.

Student: Tikehau Capital's private equity internship ("Currently pursuing or
holding a degree") and LCCC's Energy Analyst Intern ("2:1 degree (or
expected for those in their penultimate year)") were skipped as students-
only; both offer graduates a route. Years: "0-2 years" came back as 2 from
the model and hid Quantcast's ML Engineer under `max_years_required: 1`;
"3 years as an Associate before being promoted to VP" read as a
requirement. Graduation year: "2027 Summer Intern" was accepted as a
graduation year, which switched the student rule off."""

from __future__ import annotations

import pytest

from rolescan.models import Job
from rolescan.scoring.facts import (
    GraduationYearFact,
    PostingFacts,
    YearsFact,
    resolve_student,
    resolve_years,
    students_only,
    verify_facts,
    years_required_stated,
)


def _job(description: str, title: str = "Analyst Intern") -> Job:
    return Job(
        source="t",
        company="Acme",
        title=title,
        url="https://acme.example/1",
        description=description,
    )


def _facts(**kw: object) -> PostingFacts:
    base: dict[str, object] = {"fit_score": 50, "reason": "fits"}
    base.update(kw)
    return PostingFacts.model_validate(base)


@pytest.mark.parametrize(
    "text",
    [
        "Currently pursuing or holding a degree in finance, economics or a related field.",
        "A 2:1 degree (or expected for those in their penultimate year) in a numerate subject.",
        "Currently studying for, or have recently completed, a Master's degree.",
    ],
)
def test_an_alternative_route_is_not_students_only(text: str) -> None:
    assert students_only(text) is None
    facts = resolve_student(_facts(), _job(text))
    assert facts.student_only.value is not True


def test_plain_enrolment_wording_is_still_students_only() -> None:
    assert students_only("You must be in your penultimate year of study.") is not None


def test_zero_to_two_years_is_not_a_requirement() -> None:
    """Review focus 4."""
    job = _job("0-2 years of experience in machine learning.")
    facts = _facts(years_required=YearsFact(value=2, quote="0-2 years of experience"))
    out = resolve_years(verify_facts(facts, job), job)
    assert out.years_required.value is None


def test_a_range_is_read_at_its_low_end() -> None:
    job = _job("3-5 years of experience in data engineering.")
    facts = _facts(years_required=YearsFact(value=5, quote="3-5 years of experience"))
    assert resolve_years(verify_facts(facts, job), job).years_required.value == 3


def test_a_stated_low_end_or_single_figure_is_left_alone() -> None:
    """Only the high end of a range is corrected; nothing else is touched."""
    low = _job("3-5 years of experience in data engineering.")
    kept = _facts(years_required=YearsFact(value=3, quote="3-5 years of experience"))
    assert resolve_years(verify_facts(kept, low), low) == verify_facts(kept, low)
    single = _job("5+ years of experience in data engineering.")
    one = _facts(years_required=YearsFact(value=5, quote="5+ years of experience"))
    assert resolve_years(verify_facts(one, single), single).years_required.value == 5


@pytest.mark.parametrize(
    "text",
    [
        "3 years as an Associate before being promoted to VP.",
        "3 years as an Associate before promotion to VP.",
    ],
)
def test_a_promotion_clause_is_a_career_path_not_a_requirement(text: str) -> None:
    assert years_required_stated(text) is None


def test_a_promotion_in_the_next_sentence_does_not_hide_a_requirement() -> None:
    text = "3 years of experience in sales. Strong performers are promoted to VP."
    found = years_required_stated(text)
    assert found is not None
    assert found[0] == 3


def test_a_programme_year_is_not_a_graduation_year() -> None:
    job = _job("Join our 2027 Summer Intern programme.", title="2027 Summer Intern")
    facts = _facts(
        graduation_year=GraduationYearFact(value=2027, quote="2027 Summer Intern")
    )
    assert verify_facts(facts, job).graduation_year.value is None


@pytest.mark.parametrize(
    "quote",
    [
        "Graduation date January 2028 - September 2028",
        "You must be graduating in 2028",
        "expected graduation date of December 2027",
        "with a completion time frame of 2028",
        "Class of 2027 candidates",
    ],
)
def test_a_graduation_year_needs_a_graduation_word(quote: str) -> None:
    job = _job(quote)
    year = 2028 if "2028" in quote else 2027
    facts = _facts(graduation_year=GraduationYearFact(value=year, quote=quote))
    assert verify_facts(facts, job).graduation_year.value == year
